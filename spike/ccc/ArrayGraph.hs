{-# LANGUAGE GADTs #-}

module ArrayGraph (Ref (..), Captures (..), Step (..), Program (..), Workgroup (..), runProgram, emitWorkgroup, standaloneSource, describe, refId, captureIds) where

import CCC
import Control.Monad (foldM, unless)
import Data.List (foldl')

-- | Refer to one scalar or vector result in a topologically ordered graph.
data Ref value where
    ScalarRef :: Int -> Ref Float
    VectorRef :: Int -> Ref [Float]

-- | Retain the typed product of scalar values captured by a map.
data Captures values where
    NoCaptures :: Captures ()
    Capture :: Ref Float -> Captures remaining -> Captures (Float, remaining)

-- | Describe shape-preserving maps and ordered scalar reductions.
data Step where
    Map :: Ref [Float] -> Cat (Float, values) Float -> Captures values -> Ref [Float] -> Step
    Sum :: Ref Float -> Ref [Float] -> Step
    Maximum :: Ref Float -> Ref [Float] -> Step

-- | Store a finite vector program whose input has reference zero.
data Program = Program [Step] (Ref [Float])

-- | Store generated workgroup storage, ordered statements, and an output accessor.
data Workgroup = Workgroup
    { declarations :: [String]
    , statements :: [String]
    , outputAt :: String -> Either String String
    , scratchVectors :: Int
    }

-- | Return the unique identifier of a graph result.
refId :: Ref value -> Int
refId (ScalarRef identifier) = identifier
refId (VectorRef identifier) = identifier

-- | List scalar dependencies in their product order.
captureIds :: Captures values -> [Int]
captureIds NoCaptures = []
captureIds (Capture reference remaining) = refId reference : captureIds remaining

-- | Interpret scalar captures using one typed lookup function.
captureValues :: [(Int, Float)] -> Captures values -> Either String values
captureValues _ NoCaptures = Right ()
captureValues values (Capture reference remaining) = (,) <$> lookupValue (refId reference) values <*> captureValues values remaining

-- | Reject a reference that does not precede its use.
lookupValue :: Int -> [(Int, value)] -> Either String value
lookupValue identifier values = maybe (Left ("Missing array graph value: " ++ show identifier)) Right (lookup identifier values)

-- | Execute the graph with finite inputs and sequential Float reductions.
runProgram :: Program -> [Float] -> Either String [Float]
runProgram (Program steps output) inputs = do
    unless (not (null inputs) && all finite inputs) (Left "Array input must be nonempty and finite")
    (_, vectors) <- foldM execute ([], [(0, inputs)]) steps
    lookupValue (refId output) vectors
  where
    finite value = not (isNaN value || isInfinite value)
    execute (scalars, vectors) step = case step of
        Map target graph captures source -> do
            values <- lookupValue (refId source) vectors
            environment <- captureValues scalars captures
            let result = map (\value -> runCat graph (value, environment)) values
            unless (all finite result) (Left "Nonfinite array map result")
            pure (scalars, (refId target, result) : vectors)
        Sum target source -> do
            values <- lookupValue (refId source) vectors
            let result = foldl' (+) 0 values
            unless (finite result) (Left "Nonfinite array sum")
            pure ((refId target, result) : scalars, vectors)
        Maximum target source -> do
            values <- lookupValue (refId source) vectors
            case values of
                [] -> Left "Array maximum requires a nonempty vector"
                first : remaining -> pure ((refId target, foldl' max first remaining) : scalars, vectors)

-- | Name a scalar workgroup result without using source identifiers.
scalarName :: Ref Float -> String
scalarName reference = "arrayScalar" ++ show (refId reference)

-- | Interpret captured scalar references as a categorical product of WGSL values.
captureCode :: Captures values -> Code values
captureCode NoCaptures = UnitCode
captureCode (Capture reference remaining) = PairCode (FloatCode (scalarName reference)) (captureCode remaining)

-- | Emit one source-derived scalar map expression.
mapCode :: Cat (Float, values) Float -> Captures values -> String -> Either String String
mapCode graph captures input = do
    code <- emitWGSL graph (PairCode (FloatCode input) (captureCode captures))
    let FloatCode result = code
    pure result

-- | List all references read by an array operation.
dependencies :: Step -> [Int]
dependencies (Map _ _ captures source) = refId source : captureIds captures
dependencies (Sum _ source) = [refId source]
dependencies (Maximum _ source) = [refId source]

-- | Describe source-derived operations and their dependencies.
describe :: Program -> String
describe (Program steps output) = unlines ("input vector0" : map describeStep steps ++ ["output vector" ++ show (refId output)])
  where
    describeStep (Map target graph captures source) = "vector" ++ show (refId target) ++ " = map " ++ showCat graph ++ " vector" ++ show (refId source) ++ " captures " ++ show (captureIds captures)
    describeStep (Sum target source) = scalarName target ++ " = sumOrdered vector" ++ show (refId source)
    describeStep (Maximum target source) = scalarName target ++ " = maximumOrdered vector" ++ show (refId source)

-- | Wrap the same workgroup plan in a standalone kernel for browser checks.
standaloneSource :: Program -> Either String String
standaloneSource program = do
    plan <- emitWorkgroup program 2048 "arrayInput" "dimensions.x"
    result <- outputAt plan "index"
    pure $
        wgslPrelude
            ++ unlines
                ( [ "@group(0) @binding(0) var<storage, read> inputs: array<f32>;"
                  , "@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;"
                  , "@group(0) @binding(2) var<uniform> dimensions: vec4<u32>;"
                  , "var<workgroup> arrayInput: array<f32, 2048>;"
                  ]
                    ++ declarations plan
                    ++ [ "@compute @workgroup_size(128)"
                       , "fn softmax(@builtin(local_invocation_index) lane: u32) {"
                       , "  if (dimensions.x == 0u || dimensions.x > 2048u) { return; }"
                       , "  for (var index = lane; index < dimensions.x; index += 128u) { arrayInput[index] = inputs[index]; }"
                       , "  workgroupBarrier();"
                       , "  if (lane == 0u) {"
                       ]
                    ++ map ("    " ++) (statements plan)
                    ++ [ "  }"
                       , "  workgroupBarrier();"
                       , "  for (var index = lane; index < dimensions.x; index += 128u) { outputs[index] = " ++ result ++ "; }"
                       , "}"
                       ]
                )

-- | Generate a schedule that consumes mutable workgroup input storage.
emitWorkgroup :: Program -> Int -> String -> String -> Either String Workgroup
emitWorkgroup (Program steps output) capacity input count = do
    unless (capacity > 0) (Left "Array workgroup capacity must be positive")
    let (scheduled, result) = case reverse steps of
            Map target graph captures source : remaining
                | refId target == refId output ->
                    (reverse remaining, Left (\buffers index -> lookupValue (refId source) buffers >>= \buffer -> mapCode graph captures (buffer ++ "[" ++ index ++ "]"), refId source : captureIds captures))
            _ -> (steps, Right output)
        finalUses = either snd (\reference -> [refId reference]) result
    (buffers, storage, code) <- schedule finalUses [(0, input)] [] scheduled
    let accessor index = case result of
            Left (generate, _) -> generate buffers index
            Right reference -> do
                buffer <- lookupValue (refId reference) buffers
                pure (buffer ++ "[" ++ index ++ "]")
    pure (Workgroup storage code accessor (length [() | declaration <- storage, "var<workgroup> arrayVector" `prefixOf` declaration]))
  where
    prefixOf prefix value = take (length prefix) value == prefix
    scalarStorage reference = "var<workgroup> " ++ scalarName reference ++ ": f32;"
    loop body = ["for (var arrayIndex = 0u; arrayIndex < " ++ count ++ "; arrayIndex++) {"] ++ map ("  " ++) body ++ ["}"]
    schedule _ buffers storage [] = Right (buffers, storage, [])
    schedule finalUses buffers storage (step : remaining) = case step of
        Map target graph captures source -> do
            sourceBuffer <- lookupValue (refId source) buffers
            let future = finalUses ++ concatMap dependencies remaining
                reused = refId source `notElem` future
                targetBuffer = if reused then sourceBuffer else "arrayVector" ++ show (refId target)
                nextStorage = if reused then storage else storage ++ ["var<workgroup> " ++ targetBuffer ++ ": array<f32, " ++ show capacity ++ ">;"]
                nextBuffers = (refId target, targetBuffer) : if reused then filter ((/= refId source) . fst) buffers else buffers
            element <- mapCode graph captures (sourceBuffer ++ "[arrayIndex]")
            let assignment = targetBuffer ++ "[arrayIndex] = " ++ element ++ ";"
                (declarationsAdded, prefix, body, rest) = case remaining of
                    Sum total vector : later
                        | refId vector == refId target ->
                            ([scalarStorage total], [scalarName total ++ " = 0.0f;"], [assignment, scalarName total ++ " = (" ++ scalarName total ++ " + " ++ targetBuffer ++ "[arrayIndex]);"], later)
                    _ -> ([], [], [assignment], remaining)
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses nextBuffers (nextStorage ++ declarationsAdded) rest
            pure (finalBuffers, finalStorage, prefix ++ loop body ++ laterCode)
        Sum target source -> do
            buffer <- lookupValue (refId source) buffers
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage target]) remaining
            pure (finalBuffers, finalStorage, [scalarName target ++ " = 0.0f;"] ++ loop [scalarName target ++ " = (" ++ scalarName target ++ " + " ++ buffer ++ "[arrayIndex]);"] ++ laterCode)
        Maximum target source -> do
            buffer <- lookupValue (refId source) buffers
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage target]) remaining
            pure (finalBuffers, finalStorage, [scalarName target ++ " = " ++ buffer ++ "[0u];"] ++ ["for (var arrayIndex = 1u; arrayIndex < " ++ count ++ "; arrayIndex++) {", "  " ++ scalarName target ++ " = max(" ++ scalarName target ++ ", " ++ buffer ++ "[arrayIndex]);", "}"] ++ laterCode)
