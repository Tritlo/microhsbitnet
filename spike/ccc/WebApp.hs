module WebApp (application) where

import Control.Monad (foldM, forM, forM_, unless, when)
import Data.Char (toLower)
import Data.Float (castWord32ToFloat, floatToDouble)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef)
import Data.List (foldl', intercalate, isInfixOf, nub, sort)
import Foreign.Ptr (Ptr)
import Foreign.Storable (peekElemOff)
import NetworkProgram (networkStep)
import WebBrowser (
    Browser,
    browserLog,
    clock,
    command,
    commands,
    dropHost,
    fetchBytes,
    fetchText,
    hostSize,
    liftBrowser,
    readFloats,
    report,
    request,
    withFloats,
 )
import WebFixtures (fixturePath, runFixtures)
import WebJSON (
    JSON (..),
    asArray,
    asBool,
    asDouble,
    asInt,
    asString,
    field,
    int,
    optional,
    parseJSON,
 )
import WebModel (
    Model,
    TensorInfo (..),
    modelTensors,
    openModel,
    readTensor,
    tensor,
    tensorScale,
 )
import WebTensor (
    Engine,
    Plan,
    Step (..),
    Weight (..),
    disposeEngine,
    execute,
    newEngine,
    preparePlan,
    resetEngine,
    uploadEmbedding,
    uploadWeight,
 )

-- | Retain the four head and checkpoint variants for all network cases.
type NetworkPlans = [(Bool, Bool, Plan)]

-- | Keep native reference buffers outside the WASM heap until comparison.
data Reference = Reference
    { referenceToken :: Int
    , referenceHidden :: Int
    , referenceNormed :: Int
    , referenceArgmax :: Maybe Int
    , referenceLogit :: Maybe Double
    , referenceLayers :: [Int]
    }

-- | Keep one independent prompt and its native C references.
data NetworkCase = NetworkCase
    { caseName :: String
    , casePromptText :: String
    , casePrompt :: [Int]
    , caseExpected :: [Int]
    , caseReferences :: [Reference]
    }

-- | Record one vector comparison without making it an acceptance gate.
data Comparison = Comparison
    { comparisonMaximum :: Double
    , comparisonRms :: Double
    , comparisonRelative :: Double
    , comparisonMismatch :: Maybe JSON
    }

-- | Accumulate full-network diagnostic errors and comparison records.
data Errors = Errors
    { errorHidden :: Double
    , errorNormed :: Double
    , errorRelative :: Double
    , errorLogit :: Double
    , errorVectors :: Int
    , errorWithinBound :: Int
    , errorStrict :: Int
    , errorRecords :: [JSON]
    }

-- | Retain the state of one free-running GPU generation sequence.
data CaseState = CaseState
    { statePosition :: Int
    , stateSelected :: Maybe Int
    , stateInputsMatch :: Bool
    , stateGenerated :: [Int]
    , stateErrors :: Errors
    , statePrefillMs :: Double
    , stateExecutionMs :: Double
    }

-- | Keep one case result and its timing boundaries.
data CaseResult = CaseResult
    { resultCase :: NetworkCase
    , resultTokens :: [Int]
    , resultPositions :: Int
    , resultErrors :: Errors
    , resultPrefillMs :: Double
    , resultDriverMs :: Double
    , resultExecutionMs :: Double
    }

-- | Define the supported decoder projection shapes as columns and rows.
linearShapes :: [(String, Int, Int)]
linearShapes =
    [ ("self_attn.q_proj.weight", 2560, 2560)
    , ("self_attn.k_proj.weight", 2560, 640)
    , ("self_attn.v_proj.weight", 2560, 640)
    , ("self_attn.o_proj.weight", 2560, 2560)
    , ("mlp.gate_proj.weight", 2560, 6912)
    , ("mlp.up_proj.weight", 2560, 6912)
    , ("mlp.down_proj.weight", 6912, 2560)
    ]

-- | Define the supported decoder normalization lengths.
normShapes :: [(String, Int)]
normShapes =
    [ ("input_layernorm.weight", 2560)
    , ("self_attn.attn_sub_norm.weight", 2560)
    , ("post_attention_layernorm.weight", 2560)
    , ("mlp.ffn_sub_norm.weight", 6912)
    ]

-- | Initialize WebGPU, check fixtures, and run the resident-model suite.
application :: Browser ()
application = do
    initialized <- request (command "init" [])
    started <- liftBrowser clock
    messages <- liftBrowser (newIORef [])
    let adapter = field "adapter" initialized
    validateAdapter initialized
    logMessage messages ("GPU: " ++ adapterDescription adapter)
    fixtureStarted <- liftBrowser clock
    samples <- runFixtures
    fixtureFinished <- liftBrowser clock
    cases <- loadSuite
    model <- openModel "/model.safetensors"
    validateModel model
    logMessage messages "Range-loaded model header: 542 tensors; bulk weights remain outside the WASM heap"
    weights <- prepareWeights model
    engine <- newEngine 30
    beforeUpload <- statistics
    uploadStarted <- liftBrowser clock
    uploadModel messages model engine weights
    commands [command "wait" []]
    uploadFinished <- liftBrowser clock
    afterUpload <- statistics
    let copiedWeights = statistic "bulkCopiedToWasmBytes" afterUpload - statistic "bulkCopiedToWasmBytes" beforeUpload
        copiedDuringUpload = statistic "copiedToWasmBytes" afterUpload - statistic "copiedToWasmBytes" beforeUpload
        uploadedBytes = statistic "uploadedBytes" afterUpload - statistic "uploadedBytes" beforeUpload
        residentBytes = statistic "liveBufferBytes" afterUpload - statistic "liveBufferBytes" beforeUpload
        rawBytes = sum [tensorEnd info - tensorBegin info | description <- weights, let { info = tensor model (weightName description) }] + 128256 * 2560 * 2
        uploadMs = uploadFinished - uploadStarted
    require (copiedWeights == 0) "Bulk model weights entered the WASM heap during upload"
    require (uploadedBytes >= fromIntegral rawBytes && residentBytes >= fromIntegral rawBytes) "Model upload or GPU residency accounting is incomplete"
    logMessage messages ("Full model range fetch and GPU upload: " ++ show uploadMs ++ " ms; bulk weight bytes copied into WASM: " ++ show copiedWeights)
    networkStarted <- liftBrowser clock
    plans <- forM [(False, False), (False, True), (True, False), (True, True)] $ \(withHead, checkpoints) -> do
        plan <- preparePlan engine checkpoints False (\_input -> networkStep engine 0 0 (if withHead then 1 else 0))
        pure (withHead, checkpoints, plan)
    plansFinished <- liftBrowser clock
    afterPlans <- statistics
    let planSetupMs = plansFinished - networkStarted
        plansCompiled = statistic "planCompilations" afterPlans - statistic "planCompilations" afterUpload
    require (plansCompiled == 4) "Expected four network plans to be compiled before the suite"
    require (statistic "submissions" afterPlans == statistic "submissions" afterUpload) "Network plan preparation submitted GPU work"
    logMessage messages ("Prepared four reusable network plans: " ++ show planSetupMs ++ " ms")
    results <- mapM (runCase messages engine plans) cases
    networkFinished <- liftBrowser clock
    afterRun <- statistics
    let positions = sum (map resultPositions results)
        submissions = statistic "submissions" afterRun - statistic "submissions" afterUpload
        runBufferAllocations = statistic "bufferAllocations" afterRun - statistic "bufferAllocations" afterPlans
        runBindGroupCreations = statistic "bindGroupCreations" afterRun - statistic "bindGroupCreations" afterPlans
        runPlanCompilations = statistic "planCompilations" afterRun - statistic "planCompilations" afterPlans
        runPlanRuns = statistic "planRuns" afterRun - statistic "planRuns" afterPlans
        errors = foldl' combineErrors emptyErrors (map resultErrors results)
        residency =
            JObject
                [ ("weightsUploadedOnce", JBool True)
                , ("weightBuffers", int (length weights))
                , ("embeddingChunks", int 8)
                , ("rawWeightBytes", int rawBytes)
                , ("uploadedBytes", number uploadedBytes)
                , ("residentModelBytes", number residentBytes)
                , ("modelBulkBytesCopiedToWasm", number copiedWeights)
                , ("copiedToWasmBytesDuringUpload", number copiedDuringUpload)
                , ("networkSubmissions", number submissions)
                , ("plansPrepared", int (length plans))
                , ("planCompilationsDuringSetup", number plansCompiled)
                , ("planSetupResidentBytes", number (statistic "liveBufferBytes" afterPlans - statistic "liveBufferBytes" afterUpload))
                , ("runBufferAllocations", number runBufferAllocations)
                , ("runBindGroupCreations", number runBindGroupCreations)
                , ("runPlanCompilations", number runPlanCompilations)
                , ("runPlanRuns", number runPlanRuns)
                , ("beforeUpload", beforeUpload)
                , ("afterUpload", afterUpload)
                , ("afterPlans", afterPlans)
                , ("afterRun", afterRun)
                ]
    require (submissions == fromIntegral positions) "Expected one GPU submission per consumed network token"
    require (runBufferAllocations == 0) "Network execution allocated new GPU buffers after plan preparation"
    require (runBindGroupCreations == 0) "Network execution created new bind groups after plan preparation"
    require (runPlanCompilations == 0) "Network execution compiled new plans after plan preparation"
    require (runPlanRuns == fromIntegral positions) "Expected one reusable plan run per consumed network token"
    disposeEngine engine
    logMessage messages ("Full-model diagnostics: " ++ show (errorVectors errors) ++ " comparable normalized vectors; maximum relative L2 " ++ show (100 * errorRelative errors) ++ "%; " ++ show (errorWithinBound errors) ++ " within 1%")
    logMessage messages "MicroHs control and validation -> generated WGSL -> physical WebGPU execution PASS; full-model accuracy is diagnostic"
    finished <- liftBrowser clock
    recordedMessages <- liftBrowser (reverse <$> readIORef messages)
    liftBrowser $
        report $
            JObject
                [ ("passed", JBool True)
                , ("executionPassed", JBool True)
                , ("fixturesPassed", JBool True)
                , ("accuracyRequired", JBool False)
                , ("adapter", adapter)
                , ("fixtureMs", number (fixtureFinished - fixtureStarted))
                , ("driverTotalMs", number (finished - started))
                , ("samples", JArray samples)
                , ("network", networkJSON uploadMs planSetupMs (networkFinished - networkStarted) results errors)
                , ("diagnostics", JArray (reverse (errorRecords errors)))
                , ("residency", residency)
                , ("messages", JArray (map JString recordedMessages))
                ]

-- | Reject failed application invariants through the browser continuation.
require :: Bool -> String -> Browser ()
require condition message = unless condition (liftBrowser (fail message))

-- | Test whether a numerical value is finite.
finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

-- | Bound finite binary32 inputs before conversion to Double.
largestFloat :: Float
largestFloat = castWord32ToFloat 0x7f7fffff

-- | Keep undefined relative errors representable in JSON.
number :: Double -> JSON
number value = if finite value then JNumber value else JNull

-- | Record application messages in Haskell and send them to the browser log.
logMessage :: IORef [String] -> String -> Browser ()
logMessage messages message = liftBrowser $ do
    modifyIORef' messages (message :)
    browserLog message

-- | Read a generic counter from a validated statistics response.
statistic :: String -> JSON -> Double
statistic name = asDouble . field name

-- | Request resource counters without exposing application rules to JavaScript.
statistics :: Browser JSON
statistics = do
    result <- request (command "stats" [])
    require
        (all (\name -> let value = statistic name result in finite value && value >= 0) ["uploadedBytes", "submissions", "copiedToWasmBytes", "bulkCopiedToWasmBytes", "liveBufferBytes", "bufferAllocations", "bindGroupCreations", "planCompilations", "planRuns"])
        "Invalid browser resource statistics"
    pure result

-- | Join the browser adapter fields for display and software-device rejection.
adapterDescription :: JSON -> String
adapterDescription adapter = intercalate " " [asString (field name adapter) | name <- ["vendor", "architecture", "device", "description"]]

-- | Require a physical adapter with enough storage for the embedding chunks.
validateAdapter :: JSON -> Browser ()
validateAdapter initialized = do
    let adapter = field "adapter" initialized
        description = map toLower (adapterDescription adapter)
    require
        (not (asBool (field "isFallbackAdapter" adapter)) && not (any (`isInfixOf` description) ["swiftshader", "llvmpipe", "software"]))
        "A physical GPU is required; software fallback is not accepted"
    require
        (all (\name -> let value = asDouble (field name initialized) in finite value && value >= 16384 * 2560 * 2) ["maxBufferSize", "maxStorageBufferBindingSize"])
        "GPU buffer limits cannot hold an embedding chunk"

-- | Parse a text resource as JSON in MicroHs.
fetchJSON :: String -> Browser JSON
fetchJSON url = do
    source <- fetchText url
    case parseJSON source of
        Left message -> liftBrowser (fail (url ++ ": " ++ message))
        Right value -> pure value

-- | Check a vocabulary token independently of reference generation.
validToken :: Int -> Bool
validToken token = token >= 0 && token < 128256

-- | Restrict case names to the suite's ASCII filename form.
validCaseName :: String -> Bool
validCaseName [] = False
validCaseName (first : remaining) =
    first >= 'a' && first <= 'z' && all (\character -> (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') || character == '-') remaining

-- | Discover the six native C cases without fixing their generation lengths.
loadSuite :: Browser [NetworkCase]
loadSuite = do
    entries <- asArray <$> fetchJSON (fixturePath "network-cases" "suite.json")
    let names = map (asString . field "name") entries
    require (length entries == 6 && length (nub names) == 6 && all validCaseName names) "Expected six distinct network cases"
    mapM loadCase entries

-- | Read one manifest and retain its small reference buffers on the host.
loadCase :: JSON -> Browser NetworkCase
loadCase entry = do
    let name = asString (field "name" entry)
        promptText = asString (field "promptText" entry)
        manifestFile = asString (field "manifest" entry)
        directory = reverse (dropWhile (/= '/') (reverse manifestFile))
    manifest <- fetchJSON (fixturePath "network-cases" manifestFile)
    let prompt = map asInt (asArray (field "prompt" manifest))
        generated = map asInt (asArray (field "generated" manifest))
        steps = asArray (field "steps" manifest)
    require
        (not (null prompt) && length prompt + 9 <= 2048 && all validToken prompt && not (null generated) && length generated <= 10 && all validToken generated && length steps <= 2048 && length steps >= length prompt + length generated - 1)
        ("Invalid network reference manifest " ++ name)
    references <- forM (zip [0 ..] steps) $ \(position, step) -> do
        let token = asInt (field "token" step)
            argmax = asInt <$> optional "argmax" step
            logit = asDouble <$> optional "logit" step
            layers = maybe [] (map asString . asArray) (optional "layers" step)
        require
            (asInt (field "position" step) == position && validToken token && maybe True validToken argmax && maybe True finite logit && (null layers || length layers == 30))
            ("Invalid network reference step in " ++ name)
        hidden <- fetchReference (directory ++ asString (field "hidden" step))
        normed <- fetchReference (directory ++ asString (field "normed" step))
        layerData <- mapM (fetchReference . (directory ++)) layers
        pure (Reference token hidden normed argmax logit layerData)
    require (map referenceToken (take (length prompt) references) == prompt) ("Reference prompt differs in " ++ name)
    forM_ (zip generated (drop (length prompt - 1) references)) $ \(token, reference) ->
        require (referenceArgmax reference == Just token && maybe False finite (referenceLogit reference)) ("Inconsistent C token reference in " ++ name)
    pure (NetworkCase name promptText prompt generated references)

-- | Fetch one diagnostic vector without copying it into WASM yet.
fetchReference :: String -> Browser Int
fetchReference file = do
    host <- fetchBytes (fixturePath "network-cases" file)
    size <- liftBrowser (hostSize host)
    require (size == 2560 * 4) ("Invalid network reference size: " ++ file)
    pure host

-- | Check the complete tensor-name schema before loading model data.
validateModel :: Model -> Browser ()
validateModel model = do
    let prefixes = ["model.layers." ++ show layer ++ "." | layer <- [0 .. 29 :: Int]]
        names = [name | (name, _, _) <- linearShapes] ++ map fst normShapes
        expected =
            [prefix ++ name | prefix <- prefixes, name <- names]
                ++ [prefix ++ name ++ "_scale" | prefix <- prefixes, (name, _, _) <- linearShapes]
                ++ ["model.embed_tokens.weight", "model.norm.weight"]
    require (length (modelTensors model) == 542 && sort (map tensorName (modelTensors model)) == sort expected) "Unexpected BitNet model tensor schema"
    validateTensor model "model.embed_tokens.weight" "BF16" [128256, 2560]
    validateTensor model "model.norm.weight" "BF16" [2560]

-- | Check the dtype and exact shape of one supported tensor.
validateTensor :: Model -> String -> String -> [Int] -> Browser ()
validateTensor model name dtype shape = do
    let info = tensor model name
    require (tensorDtype info == dtype && tensorShape info == shape) ("Unexpected model tensor shape or dtype: " ++ name)

-- | Read only scalar scales before the bulk-upload accounting boundary.
prepareWeights :: Model -> Browser [Weight]
prepareWeights model = do
    layers <- forM [0 .. 29 :: Int] $ \layer -> do
        let prefix = "model.layers." ++ show layer ++ "."
        projections <- forM linearShapes $ \(suffix, columns, rows) -> do
            let name = prefix ++ suffix
                scaleName = name ++ "_scale"
                scaleInfo = tensor model scaleName
            validateTensor model name "U8" [rows `div` 4, columns]
            require (tensorDtype scaleInfo == "BF16" && product (tensorShape scaleInfo) == 1) ("Invalid model scale: " ++ scaleName)
            scale <- tensorScale model scaleName
            require (finite scale && scale > 0) ("Nonpositive or nonfinite model scale: " ++ scaleName)
            pure (LinearWeight name columns rows scale)
        norms <- forM normShapes $ \(suffix, count) -> do
            let name = prefix ++ suffix
            validateTensor model name "BF16" [count]
            pure (BFloatNorm name count)
        pure (projections ++ norms)
    pure (concat layers ++ [BFloatNorm "model.norm.weight" 2560])

-- | Read a weight name without exposing tensor metadata to JavaScript.
weightName :: Weight -> String
weightName (LinearWeight name _ _ _) = name
weightName (FloatNorm name _) = name
weightName (BFloatNorm name _) = name

-- | Upload every weight once and release each raw host buffer after upload.
uploadModel :: IORef [String] -> Model -> Engine -> [Weight] -> Browser ()
uploadModel messages model engine weights = do
    forM_ (zip [1 ..] weights) $ \(count, description) -> do
        let name = weightName description
            info = tensor model name
        host <- readTensor model name 0 Nothing
        size <- liftBrowser (hostSize host)
        require (size == tensorEnd info - tensorBegin info) ("Truncated model weight: " ++ name)
        uploadWeight engine description host
        liftBrowser (dropHost host)
        when (count <= 330 && count `mod` 55 == 0) $
            logMessage messages ("Loaded " ++ show count ++ "/330 decoder weights directly into WebGPU")
    forM_ [0, 16384 .. 128255] $ \firstRow -> do
        let rows = min 16384 (128256 - firstRow)
            count = rows * 2560 * 2
        host <- readTensor model "model.embed_tokens.weight" (firstRow * 2560 * 2) (Just count)
        size <- liftBrowser (hostSize host)
        require (size == count) "Truncated model embedding chunk"
        uploadEmbedding engine firstRow host
        liftBrowser (dropHost host)

-- | Initialize a diagnostic accumulator without reference acceptance gates.
emptyErrors :: Errors
emptyErrors = Errors 0 0 0 0 0 0 0 []

-- | Combine per-case diagnostics without hiding noncomparable positions.
combineErrors :: Errors -> Errors -> Errors
combineErrors previous current =
    Errors
        (max (errorHidden previous) (errorHidden current))
        (max (errorNormed previous) (errorNormed current))
        (max (errorRelative previous) (errorRelative current))
        (max (errorLogit previous) (errorLogit current))
        (errorVectors previous + errorVectors current)
        (errorWithinBound previous + errorWithinBound current)
        (errorStrict previous + errorStrict current)
        (errorRecords current ++ errorRecords previous)

-- | Copy only a bounded diagnostic vector and reject nonfinite values.
readVector :: Int -> Int -> Int -> String -> Browser [Double]
readVector host offset count name = do
    size <- liftBrowser (hostSize host)
    require (offset >= 0 && offset `mod` 4 == 0 && count > 0 && offset + count * 4 <= size) ("Invalid diagnostic vector range: " ++ name)
    values <- liftBrowser (readFloats host offset count)
    require (length values == count && all finite values) ("Nonfinite or truncated diagnostic vector: " ++ name)
    pure values

-- | Compare two bounded network buffers without constructing value lists.
compareBuffers :: Int -> Int -> Int -> Browser Comparison
compareBuffers actualHost actualOffset referenceHost = do
    actualSize <- liftBrowser (hostSize actualHost)
    referenceSize <- liftBrowser (hostSize referenceHost)
    require
        (actualOffset >= 0 && actualOffset `mod` 4 == 0 && actualOffset <= actualSize && 2560 * 4 <= actualSize - actualOffset && referenceSize >= 2560 * 4)
        "Invalid network diagnostic vector range"
    liftBrowser $
        withFloats actualHost actualOffset 2560 $ \actual ->
            withFloats referenceHost 0 2560 $ \reference ->
                compareFloatPointers actual reference 0 0 0 0 Nothing

-- | Accumulate finite Float pairs in ascending order with strict Double sums.
compareFloatPointers :: Ptr Float -> Ptr Float -> Int -> Double -> Double -> Double -> Maybe JSON -> IO Comparison
compareFloatPointers actual reference index maximumError squaredError squaredReference firstMismatch
    | index == 2560 = do
        unless (finite maximumError && finite squaredError && finite squaredReference) $
            fail "Nonfinite network diagnostic error"
        let relative = if squaredReference == 0 then (if squaredError == 0 then 0 else 1 / 0) else sqrt (squaredError / squaredReference)
        pure (Comparison maximumError (sqrt (squaredError / 2560)) relative firstMismatch)
    | otherwise = do
        actualFloat <- peekElemOff actual index
        referenceFloat <- peekElemOff reference index
        let value = floatToDouble actualFloat
            expected = floatToDouble referenceFloat
        unless (abs actualFloat <= largestFloat && abs referenceFloat <= largestFloat) $
            fail "Nonfinite network diagnostic vector"
        let difference = abs (value - expected)
            nextMaximum = max maximumError difference
            nextError = squaredError + difference * difference
            nextReference = squaredReference + expected * expected
            nextMismatch = case firstMismatch of
                Just found -> Just found
                Nothing ->
                    if difference > 0.00002 * max 1 (abs expected)
                        then Just (JObject [("index", int index), ("actual", number value), ("reference", number expected)])
                        else Nothing
        nextMaximum `seq`
            nextError `seq`
                nextReference `seq`
                    nextMismatch `seq`
                        compareFloatPointers actual reference (index + 1) nextMaximum nextError nextReference nextMismatch

-- | Add a vector record and count stricter diagnostic differences.
recordComparison :: String -> Comparison -> Errors -> Errors
recordComparison name comparison errors =
    errors
        { errorStrict = errorStrict errors + maybe 0 (const 1) (comparisonMismatch comparison)
        , errorRecords =
            JObject
                ( [ ("name", JString name)
                  , ("maximum", number (comparisonMaximum comparison))
                  , ("rms", number (comparisonRms comparison))
                  , ("relativeL2", number (comparisonRelative comparison))
                  ]
                    ++ maybe [] (\mismatch -> [("firstMismatch", mismatch)]) (comparisonMismatch comparison)
                )
                : errorRecords errors
        }

-- | Compare full-network vectors only while the consumed histories match.
checkStep :: String -> Int -> Step -> Maybe Reference -> Bool -> Errors -> Browser Errors
checkStep name position step reference inputsMatch errors = do
    require (stepHasNorm step) "Missing final normalized vector"
    let offsets = stepLayerOffsets step
    require (null offsets || length offsets == 30) "Invalid network layer checkpoint count"
    case (inputsMatch, reference) of
        (True, Just expected) -> do
            require (length offsets == length (referenceLayers expected)) "Layer checkpoint count differs from C"
            layerErrors <-
                foldM
                    ( \current (layer, offset, host) -> do
                        comparison <- compareBuffers (stepData step) offset host
                        pure (recordComparison (name ++ "/position " ++ show position ++ " layer " ++ show layer) comparison current)
                    )
                    errors
                    (zip3 [0 ..] offsets (referenceLayers expected))
            hiddenComparison <- compareBuffers (stepData step) 0 (referenceHidden expected)
            normedComparison <- compareBuffers (stepData step) 10240 (referenceNormed expected)
            let recorded = recordComparison (name ++ "/normed " ++ show position) normedComparison (recordComparison (name ++ "/hidden " ++ show position) hiddenComparison layerErrors)
            pure $
                recorded
                    { errorHidden = max (errorHidden recorded) (comparisonMaximum hiddenComparison)
                    , errorNormed = max (errorNormed recorded) (comparisonMaximum normedComparison)
                    , errorRelative = max (errorRelative recorded) (comparisonRelative normedComparison)
                    , errorVectors = errorVectors recorded + 1
                    , errorWithinBound = errorWithinBound recorded + if comparisonRelative normedComparison <= 0.01 then 1 else 0
                    }
        _ -> do
            _ <- readVector (stepData step) 0 2560 "GPU hidden output"
            _ <- readVector (stepData step) 10240 2560 "GPU final normalization"
            forM_ offsets $ \offset -> do
                _ <- readVector (stepData step) offset 2560 "GPU layer checkpoint"
                pure ()
            pure errors

-- | Check optional diagnostic logits only when the engine exposes them.
checkLogits :: Step -> Int -> Double -> Browser ()
checkLogits step token logit = case stepLogitsOffset step of
    Nothing -> pure ()
    Just offset -> do
        forM_ [0, 4096 .. 128255] $ \first -> do
            _ <- readVector (stepData step) (offset + first * 4) (min 4096 (128256 - first)) "GPU logits"
            pure ()
        selected <- readVector (stepData step) (offset + token * 4) 1 "GPU selected logit"
        require (selected == [logit]) "GPU selected logit differs from the logits buffer"

-- | Reset cache positions and run a case with an independent ten-token limit.
runCase :: IORef [String] -> Engine -> NetworkPlans -> NetworkCase -> Browser CaseResult
runCase messages engine plans testCase = do
    liftBrowser (resetEngine engine)
    started <- liftBrowser clock
    logMessage messages ("Network case " ++ caseName testCase ++ ": " ++ show (length (casePrompt testCase)) ++ " prompt tokens; up to 10 generated tokens; caches reset")
    result <- runPositions messages engine plans testCase started (CaseState 0 Nothing True [] emptyErrors 0 0)
    forM_ (caseReferences testCase) $ \reference ->
        liftBrowser (mapM_ dropHost (referenceHidden reference : referenceNormed reference : referenceLayers reference))
    logMessage messages (caseName testCase ++ ": " ++ show (errorVectors (resultErrors result)) ++ "/" ++ show (resultPositions result) ++ " normalized vectors comparable; " ++ show (matchingTokens result) ++ "/" ++ show (length (resultTokens result)) ++ " GPU tokens match C; " ++ show (resultDriverMs result) ++ " ms")
    pure result

-- | Consume prompt or GPU-selected tokens without consulting reference length.
runPositions :: IORef [String] -> Engine -> NetworkPlans -> NetworkCase -> Double -> CaseState -> Browser CaseResult
runPositions messages engine plans testCase started state = do
    let position = statePosition state
        prompt = casePrompt testCase
        withHead = position >= length prompt - 1
        reference = case drop position (caseReferences testCase) of
            current : _ -> Just current
            [] -> Nothing
    token <-
        if position < length prompt
            then pure (prompt !! position)
            else case stateSelected state of
                Just selected -> pure selected
                Nothing -> liftBrowser (fail "Missing GPU-selected network input")
    require (position < 2048 && validToken token) "Invalid network position or input token"
    let inputsMatch = stateInputsMatch state && maybe False ((== token) . referenceToken) reference
    plan <- case [prepared | (planHead, planCheckpoints, prepared) <- plans, planHead == withHead && planCheckpoints == (position < 3)] of
        [prepared] -> pure prepared
        _ -> liftBrowser (fail "Missing or duplicate network plan variant")
    step <- execute engine plan position (Just token) Nothing
    checked <- checkStep (caseName testCase) position step reference inputsMatch (stateErrors state)
    (selected, errors) <-
        if withHead
            then case (stepToken step, stepLogit step) of
                (Just chosen, Just logit) -> do
                    require (validToken chosen && finite logit) "Invalid GPU head output"
                    checkLogits step chosen logit
                    let expected = case drop (length (stateGenerated state)) (caseExpected testCase) of
                            current : _ -> Just current
                            [] -> Nothing
                        updated = case (inputsMatch, reference, expected) of
                            (True, Just current, Just expectedToken)
                                | chosen == expectedToken -> case referenceLogit current of
                                    Just expectedLogit ->
                                        let difference = abs (logit - expectedLogit)
                                         in checked
                                                { errorLogit = max (errorLogit checked) difference
                                                , errorStrict = errorStrict checked + if difference > 0.00002 * max 1 (abs expectedLogit) then 1 else 0
                                                }
                                    Nothing -> checked
                            _ -> checked
                    logMessage messages (caseName testCase ++ " position " ++ show position ++ ": GPU token " ++ show chosen ++ "; C reference " ++ show expected ++ "; comparable history " ++ show inputsMatch ++ "; step " ++ show (stepMs step) ++ " ms")
                    pure (Just chosen, updated)
                _ -> liftBrowser (fail "Missing GPU head output")
            else do
                require (stepToken step == Nothing && stepLogit step == Nothing) "Unexpected head output during prompt prefill"
                when (position `mod` 8 == 0) $
                    logMessage messages (caseName testCase ++ " prefill " ++ show (position + 1) ++ "/" ++ show (length prompt) ++ ": hidden computed; step " ++ show (stepMs step) ++ " ms")
                pure (Nothing, checked)
    liftBrowser (dropHost (stepData step))
    now <- liftBrowser clock
    let generated = case selected of
            Just chosen -> chosen : stateGenerated state
            Nothing -> stateGenerated state
        prefillMs = if position == length prompt - 1 then now - started else statePrefillMs state
        executionMs = stateExecutionMs state + stepMs step
        next = CaseState (position + 1) selected inputsMatch generated errors prefillMs executionMs
        stopped = length generated >= 10 || maybe False (`elem` [128009, 128001]) selected
    if stopped
        then pure (CaseResult testCase (reverse generated) (position + 1) errors prefillMs (now - started) executionMs)
        else runPositions messages engine plans testCase started next

-- | Count positional token matches without making generation depend on them.
matchingTokens :: CaseResult -> Int
matchingTokens result = length (filter id (zipWith (==) (resultTokens result) (caseExpected (resultCase result))))

-- | Report the first differing token or unequal output length.
firstDiscrepancy :: Int -> Int -> [Int] -> [Int] -> Maybe JSON
firstDiscrepancy _ _ [] [] = Nothing
firstDiscrepancy promptLength index (actual : actualRest) (expected : expectedRest)
    | actual == expected = firstDiscrepancy promptLength (index + 1) actualRest expectedRest
firstDiscrepancy promptLength index actual expected =
    Just $
        JObject
            [ ("index", int index)
            , ("position", int (promptLength - 1 + index))
            , ("actual", case actual of current : _ -> int current; [] -> JNull)
            , ("reference", case expected of current : _ -> int current; [] -> JNull)
            ]

-- | Preserve per-case metric names and expose the first token discrepancy.
caseJSON :: CaseResult -> JSON
caseJSON result =
    let testCase = resultCase result
        errors = resultErrors result
     in JObject
            [ ("name", JString (caseName testCase))
            , ("promptText", JString (casePromptText testCase))
            , ("promptTokens", int (length (casePrompt testCase)))
            , ("generatedTokens", JArray (map int (resultTokens result)))
            , ("referenceTokens", JArray (map int (caseExpected testCase)))
            , ("matchingTokens", int (matchingTokens result))
            , ("firstDiscrepancy", maybe JNull id (firstDiscrepancy (length (casePrompt testCase)) 0 (resultTokens result) (caseExpected testCase)))
            , ("positionsRun", int (resultPositions result))
            , ("normalizedVectorsChecked", int (errorVectors errors))
            , ("maxNormedRelativeL2", number (errorRelative errors))
            , ("driverMs", number (resultDriverMs result))
            , ("executionMs", number (resultExecutionMs result))
            ]

-- | Preserve full-network metric names while keeping accuracy diagnostic.
networkJSON :: Double -> Double -> Double -> [CaseResult] -> Errors -> JSON
networkJSON uploadMs planSetupMs driverMs results errors =
    let positions = sum (map resultPositions results)
        prefillMs = sum (map resultPrefillMs results)
        caseMs = sum (map resultDriverMs results)
        executionMs = sum (map resultExecutionMs results)
     in JObject
            [ ("count", int (sum (map (length . resultTokens) results)))
            , ("matchingTokens", int (sum (map matchingTokens results)))
            , ("tokenReferenceMatched", JBool (all (\result -> resultTokens result == caseExpected (resultCase result)) results))
            , ("tokenIdsValid", JBool True)
            , ("accuracyRequired", JBool False)
            , ("tokenReferenceRequired", JBool False)
            , ("normalizedVectorBoundRequired", JBool False)
            , ("positionsRun", int positions)
            , ("uploadMs", number uploadMs)
            , ("planSetupMs", number planSetupMs)
            , ("driverMs", number driverMs)
            , ("executionMs", number executionMs)
            , ("validationAndControlMs", number (caseMs - executionMs))
            , ("prefillMs", number prefillMs)
            , ("generationMs", number (caseMs - prefillMs))
            , ("softmaxRoundtrips", int 0)
            , ("maxHiddenAbsoluteError", number (errorHidden errors))
            , ("maxNormedAbsoluteError", number (errorNormed errors))
            , ("maxNormedRelativeL2", number (errorRelative errors))
            , ("normalizedVectorsChecked", int (errorVectors errors))
            , ("normalizedVectorsWithinOnePercent", int (errorWithinBound errors))
            , ("numericalBoundPassed", JBool (errorWithinBound errors == positions))
            , ("strictDiagnosticFailures", int (errorStrict errors))
            , ("maxLogitAbsoluteError", number (errorLogit errors))
            , ("cases", JArray (map caseJSON results))
            ]
