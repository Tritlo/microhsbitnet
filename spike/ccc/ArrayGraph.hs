{-# LANGUAGE GADTs #-}

module ArrayGraph (Ref (..), Captures (..), Step (..), Program (..), Workgroup (..), runProgram, runVectors, emitWorkgroup, standaloneSource, describe, refId, captureIds) where

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
    Map2 :: Ref [Float] -> Cat (Float, (Float, values)) Float -> Captures values -> Ref [Float] -> Ref [Float] -> Step
    Sum :: Ref Float -> Ref [Float] -> Step
    Maximum :: Ref Float -> Ref [Float] -> Step
    Length :: Ref Float -> Ref [Float] -> Step

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
runProgram program inputs = runVectors program [inputs]

-- | Execute a one- or two-input vector program.
runVectors :: Program -> [[Float]] -> Either String [Float]
runVectors (Program steps output) initialVectors = do
    unless (not (null initialVectors) && all (\values -> not (null values) && all finite values) initialVectors) (Left "Array input must be nonempty and finite")
    (_, vectors) <- foldM execute ([], zip [0 ..] initialVectors) steps
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
        Map2 target graph captures left right -> do
            leftValues <- lookupValue (refId left) vectors
            rightValues <- lookupValue (refId right) vectors
            unless (length leftValues == length rightValues) (Left "Array zip arguments differ in length")
            environment <- captureValues scalars captures
            let result = zipWith (\leftValue rightValue -> runCat graph (leftValue, (rightValue, environment))) leftValues rightValues
            unless (all finite result) (Left "Nonfinite array zip result")
            pure (scalars, (refId target, result) : vectors)
        Length target source -> do
            values <- lookupValue (refId source) vectors
            pure ((refId target, fromIntegral (length values)) : scalars, vectors)
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
captureCode :: String -> Captures values -> Code values
captureCode _ NoCaptures = UnitCode
captureCode prefix (Capture reference remaining) = PairCode (FloatCode (prefixedName prefix reference)) (captureCode prefix remaining)

-- | Keep scalar names distinct when several schedules share one shader.
prefixedName :: String -> Ref Float -> String
prefixedName prefix reference = prefix ++ scalarName reference

-- | Emit one source-derived scalar map expression.
mapCode :: String -> Cat (Float, values) Float -> Captures values -> String -> Either String String
mapCode prefix graph captures input = do
    code <- emitWGSL graph (PairCode (FloatCode input) (captureCode prefix captures))
    let FloatCode result = code
    pure result

-- | Emit one source-derived expression over a pair of elements.
map2Code :: String -> Cat (Float, (Float, values)) Float -> Captures values -> String -> String -> Either String String
map2Code prefix graph captures left right = do
    code <- emitWGSL graph (PairCode (FloatCode left) (PairCode (FloatCode right) (captureCode prefix captures)))
    let FloatCode result = code
    pure result

-- | List all references read by an array operation.
dependencies :: Step -> [Int]
dependencies (Map _ _ captures source) = refId source : captureIds captures
dependencies (Map2 _ _ captures left right) = refId left : refId right : captureIds captures
dependencies (Sum _ source) = [refId source]
dependencies (Maximum _ source) = [refId source]
dependencies (Length _ source) = [refId source]

-- | Describe source-derived operations and their dependencies.
describe :: Program -> String
describe (Program steps output) = unlines ("input vector0" : map describeStep steps ++ ["output vector" ++ show (refId output)])
  where
    describeStep (Map target graph captures source) = "vector" ++ show (refId target) ++ " = map " ++ showCat graph ++ " vector" ++ show (refId source) ++ " captures " ++ show (captureIds captures)
    describeStep (Map2 target graph captures left right) = "vector" ++ show (refId target) ++ " = zip " ++ showCat graph ++ " vector" ++ show (refId left) ++ " vector" ++ show (refId right) ++ " captures " ++ show (captureIds captures)
    describeStep (Sum target source) = scalarName target ++ " = sumOrdered vector" ++ show (refId source)
    describeStep (Maximum target source) = scalarName target ++ " = maximumOrdered vector" ++ show (refId source)
    describeStep (Length target source) = scalarName target ++ " = length vector" ++ show (refId source)

-- | Wrap the same workgroup plan in a standalone kernel for browser checks.
standaloneSource :: Program -> Either String String
standaloneSource program = do
    plan <- emitWorkgroup program 2048 [(0, "arrayInput")] "dimensions.x" "soft"
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

-- | Generate a schedule that consumes named input storage, including a second vector when present.
emitWorkgroup :: Program -> Int -> [(Int, String)] -> String -> String -> Either String Workgroup
emitWorkgroup (Program steps output) capacity inputs count prefix = do
    unless (capacity > 0 && not (null inputs)) (Left "Array workgroup capacity must be positive")
    let (scheduled, result) = case reverse steps of
            Map target graph captures source : remaining
                | refId target == refId output ->
                    (reverse remaining, Left (\buffers index -> lookupValue (refId source) buffers >>= \buffer -> mapCode prefix graph captures (buffer ++ "[" ++ index ++ "]"), refId source : captureIds captures))
            Map2 target graph captures left right : remaining
                | refId target == refId output ->
                    ( reverse remaining
                    , Left
                        ( \buffers index -> do
                            leftBuffer <- lookupValue (refId left) buffers
                            rightBuffer <- lookupValue (refId right) buffers
                            map2Code prefix graph captures (leftBuffer ++ "[" ++ index ++ "]") (rightBuffer ++ "[" ++ index ++ "]")
                        , refId left : refId right : captureIds captures
                        )
                    )
            _ -> (steps, Right output)
        finalUses = either snd (\reference -> [refId reference]) result
    (buffers, storage, code) <- schedule finalUses inputs [] scheduled
    let accessor index = case result of
            Left (generate, _) -> generate buffers index
            Right reference -> do
                buffer <- lookupValue (refId reference) buffers
                pure (buffer ++ "[" ++ index ++ "]")
    pure (Workgroup storage code accessor (length [() | declaration <- storage, "var<workgroup> arrayVector" `prefixOf` declaration]))
  where
    prefixOf prefix value = take (length prefix) value == prefix
    scalarStorage reference = "var<workgroup> " ++ prefixedName prefix reference ++ ": f32;"
    scalarRef reference = prefixedName prefix reference
    loop body = ["for (var arrayIndex = 0u; arrayIndex < " ++ count ++ "; arrayIndex++) {"] ++ map ("  " ++) body ++ ["}"]
    schedule _ buffers storage [] = Right (buffers, storage, [])
    schedule finalUses buffers storage (step : remaining) = case step of
        Map target graph captures source -> do
            sourceBuffer <- lookupValue (refId source) buffers
            let future = finalUses ++ concatMap dependencies remaining
            element <- mapCode prefix graph captures (sourceBuffer ++ "[arrayIndex]")
            case remaining of
                Sum total vector : later
                    | refId vector == refId target && refId target `notElem` (finalUses ++ concatMap dependencies later) -> do
                        (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage total]) later
                        pure (finalBuffers, finalStorage, [scalarRef total ++ " = 0.0f;"] ++ loop [scalarRef total ++ " = (" ++ scalarRef total ++ " + " ++ element ++ ");"] ++ laterCode)
                _ -> do
                    let reused = refId source `notElem` future
                        targetBuffer = if reused then sourceBuffer else "arrayVector" ++ show (refId target)
                        nextStorage = if reused then storage else storage ++ ["var<workgroup> " ++ targetBuffer ++ ": array<f32, " ++ show capacity ++ ">;"]
                        nextBuffers = (refId target, targetBuffer) : if reused then filter ((/= refId source) . fst) buffers else buffers
                        assignment = targetBuffer ++ "[arrayIndex] = " ++ element ++ ";"
                        (declarationsAdded, prefix, body, rest) = case remaining of
                            Sum total vector : later
                                | refId vector == refId target ->
                                    ([scalarStorage total], [scalarRef total ++ " = 0.0f;"], [assignment, scalarRef total ++ " = (" ++ scalarRef total ++ " + " ++ targetBuffer ++ "[arrayIndex]);"], later)
                            _ -> ([], [], [assignment], remaining)
                    (finalBuffers, finalStorage, laterCode) <- schedule finalUses nextBuffers (nextStorage ++ declarationsAdded) rest
                    pure (finalBuffers, finalStorage, prefix ++ loop body ++ laterCode)
        Map2 target graph captures left right -> do
            leftBuffer <- lookupValue (refId left) buffers
            rightBuffer <- lookupValue (refId right) buffers
            element <- map2Code prefix graph captures (leftBuffer ++ "[arrayIndex]") (rightBuffer ++ "[arrayIndex]")
            let targetBuffer = "arrayVector" ++ show (refId target)
                nextBuffers = (refId target, targetBuffer) : buffers
                assignment = targetBuffer ++ "[arrayIndex] = " ++ element ++ ";"
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses nextBuffers (storage ++ ["var<workgroup> " ++ targetBuffer ++ ": array<f32, " ++ show capacity ++ ">;"]) remaining
            pure (finalBuffers, finalStorage, loop [assignment] ++ laterCode)
        Length target _ -> do
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage target]) remaining
            pure (finalBuffers, finalStorage, [scalarRef target ++ " = f32(" ++ count ++ ");"] ++ laterCode)
        Sum target source -> do
            buffer <- lookupValue (refId source) buffers
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage target]) remaining
            pure (finalBuffers, finalStorage, [scalarRef target ++ " = 0.0f;"] ++ loop [scalarRef target ++ " = (" ++ scalarRef target ++ " + " ++ buffer ++ "[arrayIndex]);"] ++ laterCode)
        Maximum target source -> do
            buffer <- lookupValue (refId source) buffers
            (finalBuffers, finalStorage, laterCode) <- schedule finalUses buffers (storage ++ [scalarStorage target]) remaining
            pure (finalBuffers, finalStorage, [scalarRef target ++ " = " ++ buffer ++ "[0u];"] ++ ["for (var arrayIndex = 1u; arrayIndex < " ++ count ++ "; arrayIndex++) {", "  " ++ scalarRef target ++ " = max(" ++ scalarRef target ++ ", " ++ buffer ++ "[arrayIndex]);", "}"] ++ laterCode)
