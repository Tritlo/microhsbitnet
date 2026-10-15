module WebFixtures (runFixtures, fixturePath) where

import ArrayFixtures (runArrayFixtures)
import Control.Exception (finally)
import Control.Monad (forM, forM_, replicateM, unless)
import Data.List (find, foldl', sort)
import LayerProgram (layerStep)
import qualified Programs
import Text.Read (readMaybe)
import WebBrowser
import WebJSON
import WebTensor (Engine, Plan, Step (..), Weight (..), disposeEngine, execute, newEngine, preparePlan, uploadWeight)

-- | Store one raw or BitLinear fixture description.
data Fixture = Fixture
    { fixtureName :: String
    , fixtureColumns :: Int
    , fixtureRows :: Int
    , fixtureScale :: Double
    }

-- | Return a fixture URL after checking both relative paths.
fixturePath :: String -> String -> String
fixturePath directory file
    | validPath directory && validPath file = "../" ++ directory ++ "/" ++ file
    | otherwise = error "Invalid fixture path"

-- | Accept nonempty ASCII path components without traversal or URL syntax.
validPath :: String -> Bool
validPath path =
    not (null part)
        && part /= "."
        && part /= ".."
        && all validCharacter part
        && case remaining of
            [] -> True
            _ : rest -> validPath rest
  where
    (part, remaining) = break (== '/') path
    -- \| Accept the characters used in fixture names.
    validCharacter :: Char -> Bool
    validCharacter character =
        (character >= 'A' && character <= 'Z')
            || (character >= 'a' && character <= 'z')
            || (character >= '0' && character <= '9')
            || character `elem` "_.-"

-- | Require the scalar, four raw matrices, eight BitLinear cases, three layer steps, and seven array cases.
runFixtures :: Browser [JSON]
runFixtures = do
    rawManifest <- fetchText (fixturePath "cases" "cases.txt")
    rawCases <- liftBrowser (parseCases False rawManifest)
    checkShader <- loadShader "../checks.wgsl"
    (checkLayout, checkPipelines) <- makePipelines checkShader ["read-only-storage", "storage", "uniform", "read-only-storage"] ["packed"]
    rawShader <- loadShader "../generated.wgsl"
    (scalarLayout, scalarPipelines) <- makePipelines rawShader ["read-only-storage", "storage", "uniform"] ["scalar"]
    (matrixLayout, matrixPipelines) <- makePipelines rawShader ["read-only-storage", "read-only-storage", "storage", "uniform"] ["matvec"]
    scalarSample <- runRaw (checkPacked checkLayout (head checkPipelines)) scalarLayout (head scalarPipelines) (Fixture "scalar" 9 9 1)
    rawSamples <- mapM (runRaw (checkPacked checkLayout (head checkPipelines)) matrixLayout (head matrixPipelines)) rawCases
    release False (rawShader : scalarLayout : matrixLayout : scalarPipelines ++ matrixPipelines)
    bitManifest <- fetchText (fixturePath "cases" "bitlinear-cases.txt")
    bitCases <- liftBrowser (parseCases True bitManifest)
    bitShader <- loadShader "../bitlinear.wgsl"
    (bitLayout, bitPipelines) <-
        makePipelines
            bitShader
            ["read-only-storage", "read-only-storage", "storage", "storage", "uniform", "uniform", "storage"]
            ["quantize", "bitlinear"]
    bitSamples <- mapM (runBitLinear (checkPacked checkLayout (head checkPipelines)) bitLayout (head bitPipelines) (bitPipelines !! 1)) bitCases
    release False (bitShader : bitLayout : bitPipelines)
    release False (checkShader : checkLayout : checkPipelines)
    layerSamples <- runLayers
    arraySamples <- runArrayFixtures
    pure (scalarSample : rawSamples ++ bitSamples ++ layerSamples ++ arraySamples)

-- | Parse decimal dimensions without machine-integer overflow.
dimension :: String -> IO Int
dimension text = case readMaybe text :: Maybe Integer of
    Just value
        | not (null text)
        , all (\character -> character >= '0' && character <= '9') text
        , value > 0
        , value <= 2147483647 ->
            pure (fromIntegral value)
    _ -> fail ("Invalid fixture dimension: " ++ text)

-- | Parse and check the complete matrix manifest before allocation.
parseCases :: Bool -> String -> IO [Fixture]
parseCases bitLinear manifest = do
    fixtures <- forM (lines manifest) $ \line -> case words line of
        [name, columns, rows] | not bitLinear -> parseCase name columns rows "1"
        [name, columns, rows, scale] | bitLinear -> parseCase name columns rows scale
        _ -> fail "Invalid fixture manifest line"
    let names = if bitLinear then ["tiny", "tail", "q-proj", "down-proj", "zero", "nearzero", "rounding", "nondyadic"] else ["tiny", "tail", "q-proj", "down-proj"]
    unless (sort (map fixtureName fixtures) == sort names) $
        fail "Fixture manifest has missing, duplicate, or unknown cases"
    mapM (\name -> maybe (fail ("Missing fixture: " ++ name)) pure (find ((== name) . fixtureName) fixtures)) names

-- | Check one matrix shape, scale, and fixture name.
parseCase :: String -> String -> String -> String -> IO Fixture
parseCase name columnsText rowsText scaleText = do
    columns <- dimension columnsText
    rows <- dimension rowsText
    scale <- maybe (fail ("Invalid fixture scale: " ++ name)) pure (readMaybe scaleText)
    unless (validPath name && '/' `notElem` name && finite scale && finiteFloat scale) $
        fail ("Invalid fixture name or scale: " ++ name)
    checkShape name columns rows
    let expectedShape = case name of
            "tiny" -> (2, 4)
            "q-proj" -> (2560, 2560)
            "down-proj" -> (6912, 2560)
            _ -> (131, 12)
    unless ((columns, rows) == expectedShape) $
        fail ("Unexpected fixture shape: " ++ name)
    pure (Fixture name columns rows scale)

-- | Bound matrix storage and every signed int32 reduction before multiplication.
checkShape :: String -> Int -> Int -> IO ()
checkShape name columns rows =
    unless
        ( columns > 0
            && rows > 0
            && rows <= 1048576
            && rows `mod` 4 == 0
            && columns <= 2147483647 `div` 128
            && columns <= (128 * 1024 * 1024) `div` rows
        )
        (fail ("Invalid matrix shape or overflow domain: " ++ name))

-- | Reject NaN and infinite doubles.
finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

-- | Require that a double remains finite in a GPU float uniform.
finiteFloat :: Double -> Bool
finiteFloat value = not (isNaN converted || isInfinite converted)
  where
    converted = realToFrac value :: Float

-- | Require the exact byte count for a fixture or readback.
checkSize :: String -> Int -> Int -> IO ()
checkSize name host expected = do
    actual <- hostSize host
    unless (actual == expected) $
        fail ("Invalid fixture size: " ++ name ++ "; bytes " ++ show actual ++ ", expected " ++ show expected)

-- | Copy a small activation or reference vector and require finite floats.
fixtureFloats :: String -> Int -> Int -> IO [Double]
fixtureFloats name host count = do
    checkSize name host (count * 4)
    values <- readFloats host 0 count
    unless (length values == count && all finite values) $
        fail ("Nonfinite or incomplete float fixture: " ++ name)
    pure values

-- | Check resident packed weights, including padding, and read only a status word.
checkPacked :: Int -> Int -> String -> Int -> Int -> Browser ()
checkPacked layout pipeline name weights bytes = do
    identifiers <- liftBrowser (replicateM 3 newHandle)
    let [status, dimensions, readback] = identifiers
        wordCount = (bytes + 3) `div` 4
        groups = min 65535 ((wordCount + 127) `div` 128)
    commands
        [ bufferCommand status 4 132 0
        , command "array" [int dimensions, JString "u32", JArray (map int [wordCount, 0, 0, 0])]
        , bufferCommand readback 4 9 0
        ]
    host <- capture readback $ \encoder ->
        [ dispatchCommand encoder pipeline layout [weights, status, dimensions, weights] groups
        , command "copy" [int encoder, int status, int 0, int readback, int 0, int 4]
        ]
    invalid <- liftBrowser ((checkSize name host 4 >> hostNumber host 0 "u32") `finally` dropHost host)
    release True identifiers
    liftBrowser (unless (invalid == 0) (fail ("Invalid packed ternary field: " ++ name)))

-- | Compare all values exactly and report the first mismatch.
checkExact :: (Eq value, Show value) => String -> [value] -> [value] -> IO ()
checkExact name actual expected = do
    unless (length actual == length expected) (fail ("Invalid result count: " ++ name))
    case find (\(_, value, reference) -> value /= reference) (zip3 [0 :: Int ..] actual expected) of
        Nothing -> pure ()
        Just (index, value, reference) ->
            fail (name ++ "[" ++ show index ++ "]: " ++ show value ++ " != reference " ++ show reference)

-- | Check finite outputs against the existing relative bound and return the largest error.
checkFloats :: String -> [Double] -> [Double] -> IO Double
checkFloats name actual expected = do
    unless (length actual == length expected && all finite actual && all finite expected) $
        fail ("Nonfinite output or invalid result count: " ++ name)
    let errors = zipWith (\value reference -> abs (value - reference)) actual expected
    case find (\(_, difference, reference) -> not (finite difference) || difference > 0.00002 * max 1 (abs reference)) (zip3 [0 :: Int ..] errors expected) of
        Nothing -> pure (maximum (0 : errors))
        Just (index, difference, _) ->
            fail (name ++ "[" ++ show index ++ "]: absolute error " ++ show difference ++ "; bound 2e-5 * max(1, abs(reference))")

-- | Interpret the synthetic packed matrices with the source contribution and reduction.
packedExpected :: Int -> Int -> [Int]
packedExpected columns rows =
    [ fromIntegral $
        foldl'
            Programs.combine
            0
            [ Programs.contribution
                (fromIntegral ((column + row `mod` chunk + row `div` chunk) `mod` 3))
                (fromIntegral ((column * 73 + 19) `mod` 256 - 128))
            | column <- [0 .. columns - 1]
            ]
    | row <- [0 .. rows - 1]
    ]
  where
    chunk = rows `div` 4

-- | Create a shader from host text and release its host bytes.
loadShader :: String -> Browser Int
loadShader path = do
    host <- fetchBytes path
    shader <- liftBrowser newHandle
    commands [command "shader" [int shader, int host]]
    liftBrowser (dropHost host)
    pure shader

-- | Create pipelines that share an explicit buffer layout.
makePipelines :: Int -> [String] -> [String] -> Browser (Int, [Int])
makePipelines shader bindings entries = do
    layout <- liftBrowser newHandle
    pipelines <- liftBrowser (replicateM (length entries) newHandle)
    commands $
        command "layout" [int layout, JArray (map JString bindings)]
            : zipWith (\pipeline entry -> command "pipeline" [int pipeline, int shader, int layout, JString entry]) pipelines entries
    pure (layout, pipelines)

-- | Specify an aligned GPU buffer and its optional host upload.
bufferCommand :: Int -> Int -> Int -> Int -> JSON
bufferCommand identifier size usage host = command "buffer" [int identifier, int (((size + 3) `div` 4) * 4), int usage, int host]

-- | Specify one dispatch with its complete resource list.
dispatchCommand :: Int -> Int -> Int -> [Int] -> Int -> JSON
dispatchCommand encoder pipeline layout resources groups =
    command "dispatch" [int encoder, int pipeline, int layout, JArray (map int resources), int groups]

-- | Submit fixture commands and retain their GPU readback in a new host handle.
capture :: Int -> (Int -> [JSON]) -> Browser Int
capture readback encode = do
    encoder <- liftBrowser newHandle
    host <- liftBrowser newHandle
    commands $
        [command "encoder" [int encoder]]
            ++ encode encoder
            ++ [command "submit" [int encoder], command "readback" [int readback, int host]]
    pure host

-- | Run two warmups and nine measured dispatches with validation outside timing.
repeatChecks :: Browser Int -> (Int -> IO Double) -> Browser (Double, Double)
repeatChecks dispatch validate = do
    results <- forM [0 :: Int .. 10] $ \_ -> do
        started <- liftBrowser clock
        host <- dispatch
        finished <- liftBrowser clock
        difference <- liftBrowser (validate host `finally` dropHost host)
        pure (finished - started, difference)
    pure (sort (map fst (drop 2 results)) !! 4, maximum (map snd results))

-- | Release bridge resource handles and optionally destroy GPU buffers.
release :: Bool -> [Int] -> Browser ()
release destroy identifiers = commands [command "release" [JArray (map int identifiers), JBool destroy]]

-- | Validate one raw scalar or packed integer matrix fixture.
runRaw :: (String -> Int -> Int -> Browser ()) -> Int -> Int -> Fixture -> Browser JSON
runRaw validatePacked layout pipeline fixture = do
    let name = fixtureName fixture
        scalar = name == "scalar"
        columns = fixtureColumns fixture
        rows = fixtureRows fixture
        inputFile = if scalar then "scalar-input.bin" else name ++ ".weights.bin"
        expectedFile = if scalar then "scalar-expected.bin" else name ++ ".expected.bin"
    input <- fetchBytes (fixturePath "cases" inputFile)
    expected <- fetchBytes (fixturePath "cases" expectedFile)
    activations <- if scalar then pure 0 else fetchBytes (fixturePath "cases" (name ++ ".acts.bin"))
    liftBrowser $ do
        checkSize name expected (rows * 4)
        if scalar
            then do
                _ <- fixtureFloats name input rows
                reference <- fixtureFloats name expected rows
                checkExact (name ++ " source parity") reference (map (realToFrac . Programs.scalar) [-32, -3.5, -1, -0.5, 0, 0.5, 1, 3.5, 32])
            else do
                checkSize name input (columns * rows `div` 4)
                checkSize name activations columns
    floatReference <- if scalar then liftBrowser (readFloats expected 0 rows) else pure []
    intReference <- if scalar then pure [] else liftBrowser (readInts expected 0 rows)
    liftBrowser $ case name of
        "tiny" -> checkExact (name ++ " source parity") intReference (packedExpected columns rows)
        "tail" -> checkExact (name ++ " source parity") intReference (packedExpected columns rows)
        _ -> pure ()
    liftBrowser (dropHost expected)
    started <- liftBrowser clock
    identifiers <- liftBrowser (replicateM 5 newHandle)
    let [weights, acts, output, readback, uniform] = identifiers
        resources = if scalar then [weights, output, uniform] else [weights, acts, output, uniform]
        dimensions = if scalar then [rows, 0, 0, 0] else [columns, rows, 0, 0]
        allocated = if scalar then [weights, output, readback, uniform] else identifiers
    commands $
        [ bufferCommand weights (if scalar then rows * 4 else columns * rows `div` 4) 136 input
        , bufferCommand output (rows * 4) 132 0
        , bufferCommand readback (rows * 4) 9 0
        , command "array" [int uniform, JString "u32", JArray (map int dimensions)]
        ]
            ++ [bufferCommand acts columns 136 activations | not scalar]
            ++ [command "wait" []]
    uploaded <- liftBrowser clock
    liftBrowser $ do
        dropHost input
        unless scalar (dropHost activations)
    unless scalar (validatePacked name weights (columns * rows `div` 4))
    (median, _) <-
        repeatChecks
            ( capture readback $ \encoder ->
                [ dispatchCommand encoder pipeline layout resources (if scalar then (rows + 127) `div` 128 else rows)
                , command "copy" [int encoder, int output, int 0, int readback, int 0, int (rows * 4)]
                ]
            )
            ( \host -> do
                checkSize name host (rows * 4)
                if scalar
                    then readFloats host 0 rows >>= \actual -> checkExact name actual floatReference
                    else readInts host 0 rows >>= \actual -> checkExact name actual intReference
                pure 0
            )
    release True allocated
    browserSample name rows (uploaded - started) "medianDispatchReadbackMs" median Nothing

-- | Validate exact activation quantization and bounded BitLinear float outputs.
runBitLinear :: (String -> Int -> Int -> Browser ()) -> Int -> Int -> Int -> Fixture -> Browser JSON
runBitLinear validatePacked layout quantize bitlinear fixture = do
    let name = fixtureName fixture
        label = "bl-" ++ name
        columns = fixtureColumns fixture
        rows = fixtureRows fixture
    packed <- fetchBytes (fixturePath "cases" (name ++ ".weights.bin"))
    input <- fetchBytes (fixturePath "cases" (name ++ ".float.bin"))
    expected <- fetchBytes (fixturePath "cases" (name ++ ".bitlinear.bin"))
    expectedQuantized <- fetchBytes (fixturePath "cases" (name ++ ".quantized.bin"))
    reference <- liftBrowser $ do
        checkSize label packed (columns * rows `div` 4)
        _ <- fixtureFloats label input columns
        fixtureFloats label expected rows
    quantizedReference <- liftBrowser $ do
        checkSize label expectedQuantized (columns * 4)
        values <- readInts expectedQuantized 0 columns
        unless (length values == columns && all (\value -> value >= -128 && value <= 127) values) $
            fail ("Invalid quantized reference: " ++ label)
        dropHost expected
        dropHost expectedQuantized
        pure values
    started <- liftBrowser clock
    identifiers <- liftBrowser (replicateM 8 newHandle)
    let [weights, inputs, quantized, output, dimensions, scales, peak, readback] = identifiers
        resources = take 7 identifiers
    commands
        [ bufferCommand weights (columns * rows `div` 4) 136 packed
        , bufferCommand inputs (columns * 4) 136 input
        , bufferCommand quantized (columns * 4) 132 0
        , bufferCommand output (rows * 4) 132 0
        , command "array" [int dimensions, JString "u32", JArray (map int [columns, rows, 0, 0])]
        , command "array" [int scales, JString "f32", JArray [JNumber (fixtureScale fixture), JNumber 0, JNumber 0, JNumber 0]]
        , bufferCommand peak 4 132 0
        , bufferCommand readback ((rows + columns) * 4) 9 0
        , command "wait" []
        ]
    uploaded <- liftBrowser clock
    liftBrowser (dropHost packed >> dropHost input)
    validatePacked label weights (columns * rows `div` 4)
    (median, maximumError) <-
        repeatChecks
            ( capture readback $ \encoder ->
                [ dispatchCommand encoder quantize layout resources 1
                , dispatchCommand encoder bitlinear layout resources rows
                , command "copy" [int encoder, int output, int 0, int readback, int 0, int (rows * 4)]
                , command "copy" [int encoder, int quantized, int 0, int readback, int (rows * 4), int (columns * 4)]
                ]
            )
            ( \host -> do
                checkSize label host ((rows + columns) * 4)
                actualQuantized <- readInts host (rows * 4) columns
                checkExact (label ++ " quantization") actualQuantized quantizedReference
                actual <- readFloats host 0 rows
                checkFloats label actual reference
            )
    release True identifiers
    browserSample label rows (uploaded - started) "medianDispatchReadbackMs" median (Just maximumError)

-- | Define the seven linear weights and four float norms for layer zero.
layerShapes :: [(String, String, Int, Int)]
layerShapes =
    [ ("self_attn.q_proj.weight", "linear", 2560, 2560)
    , ("self_attn.k_proj.weight", "linear", 2560, 640)
    , ("self_attn.v_proj.weight", "linear", 2560, 640)
    , ("self_attn.o_proj.weight", "linear", 2560, 2560)
    , ("mlp.gate_proj.weight", "linear", 2560, 6912)
    , ("mlp.up_proj.weight", "linear", 2560, 6912)
    , ("mlp.down_proj.weight", "linear", 6912, 2560)
    , ("input_layernorm.weight", "norm", 2560, 1)
    , ("self_attn.attn_sub_norm.weight", "norm", 2560, 1)
    , ("post_attention_layernorm.weight", "norm", 2560, 1)
    , ("mlp.ffn_sub_norm.weight", "norm", 6912, 1)
    ]

-- | Load and check one layer weight while its bytes remain in host memory.
loadLayerWeight :: JSON -> Browser (Weight, Int)
loadLayerWeight description = do
    let name = asString (field "name" description)
        kind = asString (field "kind" description)
        columns = asInt (field "columns" description)
        rows = asInt (field "rows" description)
        scale = asDouble (field "scale" description)
        file = asString (field "file" description)
    liftBrowser $
        unless
            ( (name, kind, columns, rows) `elem` [("model.layers.0." ++ suffix, expectedKind, expectedColumns, expectedRows) | (suffix, expectedKind, expectedColumns, expectedRows) <- layerShapes]
                && finite scale
                && finiteFloat scale
            )
            (fail ("Invalid layer weight shape or scale: " ++ name))
    host <- fetchBytes (fixturePath "layer-cases" file)
    weight <-
        liftBrowser $
            if kind == "linear"
                then do
                    checkShape name columns rows
                    checkSize name host (columns * rows `div` 4)
                    pure (LinearWeight name columns rows scale)
                else do
                    checkSize name host (columns * 4)
                    pure (FloatNorm name columns)
    pure (weight, host)

-- | Load weights once and reuse one plan for three consecutive layer steps.
runLayers :: Browser [JSON]
runLayers = do
    text <- fetchText (fixturePath "layer-cases" "manifest.json")
    manifest <- liftBrowser (either fail pure (parseJSON text))
    let weights = asArray (field "weights" manifest)
        steps = asArray (field "steps" manifest)
        names = map (asString . field "name") weights
    liftBrowser $ do
        unless (sort names == sort ["model.layers.0." ++ suffix | (suffix, _, _, _) <- layerShapes] && length steps == 3) $
            fail "Layer manifest requires eleven unique weights and three steps"
        forM_ weights $ \description ->
            fixturePath "layer-cases" (asString (field "file" description)) `seq` pure ()
        forM_ (zip [0 :: Int ..] steps) $ \(position, step) -> do
            unless (asInt (field "position" step) == position) $
                fail "Layer fixture positions must start at zero and be consecutive"
            fixturePath "layer-cases" (asString (field "input" step)) `seq` pure ()
            fixturePath "layer-cases" (asString (field "expected" step)) `seq` pure ()
    loaded <- mapM loadLayerWeight weights
    engine <- newEngine 1
    started <- liftBrowser clock
    forM_ loaded $ \(weight, host) -> do
        uploadWeight engine weight host
        liftBrowser (dropHost host)
    commands [command "wait" []]
    uploaded <- liftBrowser clock
    plan <- preparePlan engine False True (layerStep engine 0 0)
    samples <- mapM (runLayerStep engine plan (uploaded - started)) steps
    disposeEngine engine
    pure samples

-- | Compare one layer result and release its activation, reference, and readback hosts.
runLayerStep :: Engine -> Plan -> Double -> JSON -> Browser JSON
runLayerStep engine plan residentUploadMs description = do
    let position = asInt (field "position" description)
        name = "layer0-pos" ++ show position
    input <- fetchBytes (fixturePath "layer-cases" (asString (field "input" description)))
    expected <- fetchBytes (fixturePath "layer-cases" (asString (field "expected" description)))
    reference <- liftBrowser $ do
        _ <- fixtureFloats (name ++ " input") input 2560
        values <- fixtureFloats (name ++ " expected") expected 2560
        dropHost expected
        pure values
    result <- execute engine plan position Nothing (Just input)
    liftBrowser (dropHost input)
    maximumError <-
        liftBrowser $
            ( do
                actual <- readFloats (stepData result) 0 2560
                checkFloats name actual reference
            )
                `finally` dropHost (stepData result)
    browserSample name 2560 (if position == 0 then residentUploadMs else 0) "dispatchReadbackMs" (stepMs result) (Just maximumError)

-- | Return and log the sample metrics consumed by the application.
browserSample :: String -> Int -> Double -> String -> Double -> Maybe Double -> Browser JSON
browserSample name count uploadMs timingKey duration maximumError = do
    let sample =
            JObject $
                [("name", JString name), ("count", int count), ("uploadMs", JNumber uploadMs), (timingKey, JNumber duration)]
                    ++ maybe [] (\value -> [("maxAbsoluteError", JNumber value)]) maximumError
    liftBrowser (browserLog (name ++ ": " ++ show count ++ " fixture values PASS"))
    pure sample
