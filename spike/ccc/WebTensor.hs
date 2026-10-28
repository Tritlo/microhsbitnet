module WebTensor (
    Engine,
    Tensor (..),
    Weight (..),
    Step (..),
    Plan,
    newEngine,
    uploadWeight,
    uploadEmbedding,
    resetEngine,
    preparePlan,
    execute,
    disposeEngine,
    weight,
    operation,
    configureRotary,
) where

import Control.Exception (onException)
import Control.Monad (forM_, unless, when)
import Data.Char (isDigit)
import Data.IORef
import Data.List (stripPrefix)
import Data.Maybe (isNothing)
import WebBrowser
import WebJSON

-- | Identify an application tensor without exposing its GPU buffer to WASM.
newtype Tensor = Tensor Int deriving (Show, Read, Eq)

-- | Describe the stored format and logical dimensions of a decoder weight.
data Weight
    = LinearWeight String Int Int Double
    | FloatNorm String Int
    | BFloatNorm String Int
    deriving (Show, Read)

-- | Retain one small host readback until the application completes its checks.
data Step = Step
    { stepData :: Int
    , stepHasNorm :: Bool
    , stepToken :: Maybe Int
    , stepLogit :: Maybe Double
    , stepLayerOffsets :: [Int]
    , stepLogitsOffset :: Maybe Int
    , stepMs :: Double
    }
    deriving (Show, Read)

-- | Store application tensor roles separately from generic browser handles.
data Value
    = Vector Int Int
    | Norm Int Int
    | Linear Int Int Int Double
    | Cache Int Int Char (IORef Int)
    | Embedding [(Int, Int, Int)] Int
    | Chosen Int Int Int Int

-- | Retain command encoding state only while a plan is prepared.
data Frame = Frame
    { framePosition :: Int
    , frameCheckpoints :: Bool
    , frameSetup :: [JSON]
    , frameCommands :: [JSON]
    , frameSlot :: Int
    , frameUniforms :: [Int]
    , frameValues :: [(Int, Value)]
    , frameCaches :: [IORef Int]
    , frameRotary :: Bool
    , frameEmbedded :: Bool
    , frameHead :: Bool
    , frameFinal :: Maybe (Int, Int)
    , frameAdds :: Int
    , frameLayers :: [Int]
    , frameAttention :: Int
    , frameOperations :: Int
    }

-- | Retain a checked program and its reusable GPU resources.
data Plan = Plan
    { planIdentifier :: Int
    , planOwner :: Int
    , planInput :: Maybe Int
    , planNormalized :: Bool
    , planSelectionOffset :: Maybe Int
    , planLayerOffsets :: [Int]
    , planStatusOffset :: Int
    , planReadback :: Int
    , planBytes :: Int
    , planCaches :: [IORef Int]
    , planEmbedding :: [(Int, Int, Int)]
    }

-- | Keep persistent weights, caches, pipelines, and application invariants.
data State = State
    { stateIdentity :: Int
    , stateLayers :: Int
    , statePipelines :: [(String, Int)]
    , stateLayouts :: [(String, Int)]
    , stateValues :: [(Int, Value)]
    , stateWeights :: [(String, Int)]
    , stateResources :: [Int]
    , stateNext :: Int
    , stateClosed :: Bool
    , stateFrame :: Maybe Frame
    , stateRunning :: Bool
    , statePool :: [(Int, Int, Int, Int)]
    , stateUniforms :: [(String, [JSON], Int)]
    , stateDummy :: Int
    , stateDummyChosen :: Int
    , stateRotary :: Int
    , stateFactors :: Int
    , stateMaxBuffer :: Int
    , stateMaxStorage :: Int
    }

-- | Own tensor state in MicroHs rather than in the browser bridge.
newtype Engine = Engine (IORef State)

-- | Reject an application invariant before submitting GPU work.
ensure :: Bool -> String -> IO ()
ensure condition message = unless condition (ioError (userError message))

-- | Test a host scalar without converting it to a narrower format.
finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

-- | Construct an integer list for the generic browser protocol.
integers :: [Int] -> JSON
integers = JArray . map int

-- | Create one aligned GPU allocation command with a caller-owned data handle.
bufferCommand :: Int -> Int -> Int -> IO (Int, JSON)
bufferCommand size usage source = do
    identifier <- newHandle
    pure (identifier, command "buffer" [int identifier, int (((size + 3) `div` 4) * 4), int usage, int source])

-- | Create one typed uniform buffer through the generic browser binding.
uniformCommands :: String -> [JSON] -> IO (Int, [Int], [JSON])
uniformCommands kind values = do
    buffer <- newHandle
    pure (buffer, [buffer], [command "array" [int buffer, JString kind, JArray values]])

-- | Initialize source-generated pipelines and persistent rotary buffers.
newEngine :: Int -> Browser Engine
newEngine layers = do
    liftBrowser (ensure (layers >= 1 && layers <= 30) "Invalid decoder layer count")
    limits <- request (command "init" [])
    sources <- mapM fetchBytes ["../bitlinear.wgsl", "../layer.wgsl", "../head.wgsl", "../checks.wgsl"]
    rotary <- fetchBytes "../rotary.bin"
    size <- liftBrowser (hostSize rotary)
    liftBrowser (ensure (size == 2048 * 128 * 4) "Invalid rotary table size")
    shaders <- liftBrowser (mapM (const newHandle) sources)
    layouts <- liftBrowser (mapM (const newHandle) [0 .. 3 :: Int])
    let [bitLayout, layerLayout, headLayout, checkLayout] = layouts
        bindings =
            [ ["read-only-storage", "read-only-storage", "storage", "storage", "uniform", "uniform", "storage"]
            , ["read-only-storage", "read-only-storage", "storage", "uniform", "uniform", "read-only-storage", "read-only-storage"]
            , ["read-only-storage", "read-only-storage", "storage", "uniform", "storage"]
            , ["read-only-storage", "storage", "uniform", "read-only-storage"]
            ]
        entries =
            [ ("quantize", 0)
            , ("bitlinear", 0)
            , ("rmsnorm", 1)
            , ("add", 1)
            , ("gated", 1)
            , ("rope", 1)
            , ("attention", 1)
            , ("embedding", 2)
            , ("head", 2)
            , ("argmax", 2)
            , ("packed", 3)
            , ("bf16", 3)
            , ("finiteFloat", 3)
            , ("selection", 3)
            ]
    pipelines <- liftBrowser (mapM (const newHandle) entries)
    (dummy, dummyCommand) <- liftBrowser (bufferCommand 4 128 0)
    (dummyChosen, chosenCommand) <- liftBrowser (bufferCommand 8 128 0)
    (rotaryBuffer, rotaryCommand) <- liftBrowser (bufferCommand size 12 rotary)
    (factors, factorsCommand) <- liftBrowser (bufferCommand 512 140 0)
    commands $
        zipWith (\identifier source -> command "shader" [int identifier, int source]) shaders sources
            ++ zipWith (\identifier types -> command "layout" [int identifier, JArray (map JString types)]) layouts bindings
            ++ zipWith (\identifier (name, group) -> command "pipeline" [int identifier, int (shaders !! group), int (layouts !! group), JString name]) pipelines entries
            ++ [dummyCommand, chosenCommand, rotaryCommand, factorsCommand]
    liftBrowser (mapM_ dropHost (rotary : sources))
    identity <- liftBrowser newHandle
    let state =
            State
                { stateIdentity = identity
                , stateLayers = layers
                , statePipelines = zip (map fst entries) pipelines
                , stateLayouts = zip ["bitlinear", "layer", "head", "checks"] layouts
                , stateValues = []
                , stateWeights = []
                , stateResources = dummy : dummyChosen : rotaryBuffer : factors : shaders ++ layouts ++ pipelines
                , stateNext = 0
                , stateClosed = False
                , stateFrame = Nothing
                , stateRunning = False
                , statePool = []
                , stateUniforms = []
                , stateDummy = dummy
                , stateDummyChosen = dummyChosen
                , stateRotary = rotaryBuffer
                , stateFactors = factors
                , stateMaxBuffer = asInt (field "maxBufferSize" limits)
                , stateMaxStorage = asInt (field "maxStorageBufferBindingSize" limits)
                }
    Engine <$> liftBrowser (newIORef state)

-- | Look up one compiler-created object by its application role.
named :: String -> [(String, Int)] -> Int
named name values = case lookup name values of
    Just identifier -> identifier
    Nothing -> error ("Missing GPU pipeline or layout: " ++ name)

-- | Read a live engine before changing application-owned state.
liveState :: Engine -> IO State
liveState (Engine reference) = do
    state <- readIORef reference
    ensure (not (stateClosed state)) "Tensor engine is closed"
    pure state

-- | Require an active token before encoding an operation.
activeFrame :: Engine -> IO (State, Frame)
activeFrame engine = do
    state <- liveState engine
    case stateFrame state of
        Nothing -> ioError (userError "Tensor operation requires an active token")
        Just frame -> pure (state, frame)

-- | Update only the current token's command state.
alterFrame :: Engine -> (Frame -> Frame) -> IO ()
alterFrame engine@(Engine reference) change = do
    (state, frame) <- activeFrame engine
    writeIORef reference state{stateFrame = Just (change frame)}

-- | Append generic browser commands in encoding order.
emit :: Engine -> [JSON] -> IO ()
emit engine additions = alterFrame engine (\frame -> frame{frameCommands = reverse additions ++ frameCommands frame})

-- | Register initialization commands separately from reusable GPU operations.
setup :: Engine -> [JSON] -> IO ()
setup engine additions = alterFrame engine (\frame -> frame{frameSetup = reverse additions ++ frameSetup frame})

-- | Refer to one numeric argument supplied by MicroHs at execution time.
argument :: Int -> JSON
argument index = JArray [JString "arg", int index]

-- | Allocate one persistent slot or reuse a compatible slot from another plan.
allocate :: Engine -> Int -> Int -> Int -> IO Int
allocate engine@(Engine reference) size usage source = do
    (state, frame) <- activeFrame engine
    let aligned = ((size + 3) `div` 4) * 4
        slot = frameSlot frame
    ensure (source == 0 && size > 0 && aligned <= stateMaxBuffer state && (usage < 128 || aligned <= stateMaxStorage state)) "Tensor allocation exceeds the device limit"
    alterFrame engine (\current -> current{frameSlot = slot + 1})
    case [identifier | (storedSlot, storedSize, storedUsage, identifier) <- statePool state, storedSlot == slot && storedSize == aligned && storedUsage == usage] of
        identifier : _ -> pure identifier
        [] -> do
            (identifier, allocation) <- bufferCommand size usage 0
            setup engine [allocation]
            modifyIORef reference (\current -> current{statePool = (slot, aligned, usage, identifier) : statePool current, stateResources = identifier : stateResources current})
            pure identifier

-- | Retain an activation shape while a plan is prepared.
resultVector :: Engine -> Int -> Int -> IO Int
resultVector engine count source = do
    ensure (count > 0 && count <= 6912) "Invalid activation shape"
    buffer <- allocate engine (count * 4) 140 source
    alterFrame engine (\frame -> frame{frameValues = (buffer, Vector buffer count) : frameValues frame})
    pure buffer

-- | Resolve a tensor without exposing tensor metadata to JavaScript.
value :: Engine -> Int -> IO Value
value engine identifier = do
    state <- liveState engine
    let current = maybe [] frameValues (stateFrame state)
    case lookup identifier (current ++ stateValues state) of
        Nothing -> ioError (userError ("Unknown tensor handle " ++ show identifier))
        Just tensor -> pure tensor

-- | Require an activation rather than a weight or cache handle.
vector :: Engine -> Int -> IO (Int, Int)
vector engine identifier = do
    tensor <- value engine identifier
    case tensor of
        Vector buffer count -> pure (buffer, count)
        _ -> ioError (userError "Expected an activation tensor")

-- | Intern static uniforms and update each dynamic uniform once per plan run.
uniform :: Engine -> String -> [JSON] -> IO Int
uniform engine@(Engine reference) kind values = do
    (state, _) <- activeFrame engine
    let dynamic = any (\item -> case item of JArray [JString "arg", _] -> True; _ -> False) values
    buffer <- case [identifier | (storedKind, storedValues, identifier) <- stateUniforms state, storedKind == kind && storedValues == values] of
        identifier : _ -> pure identifier
        [] -> do
            (identifier, additions) <-
                if dynamic
                    then do
                        (identifier, allocation) <- bufferCommand (length values * 4) 72 0
                        pure (identifier, [allocation])
                    else do
                        (identifier, _, additions) <- uniformCommands kind values
                        pure (identifier, additions)
            setup engine additions
            modifyIORef reference (\current -> current{stateUniforms = (kind, values, identifier) : stateUniforms current, stateResources = identifier : stateResources current})
            pure identifier
    (_, frame) <- activeFrame engine
    when (dynamic && buffer `notElem` frameUniforms frame) $ do
        emit engine [command "write" [int buffer, JString kind, JArray values]]
        alterFrame engine (\current -> current{frameUniforms = buffer : frameUniforms current})
    pure buffer

-- | Encode a dispatch with all possible bindings selected by MicroHs.
dispatchChoices :: Engine -> String -> String -> [[Int]] -> JSON -> Int -> IO ()
dispatchChoices engine family name choices selection groups = do
    (state, _) <- activeFrame engine
    ensure (groups > 0 && groups <= 65535 && not (null choices)) "Invalid dispatch group count or bindings"
    emit engine [command "dispatch" [int (named name (statePipelines state)), int (named family (stateLayouts state)), JArray (map integers choices), selection, int groups]]

-- | Encode a dispatch whose binding set does not change between positions.
dispatch :: Engine -> String -> String -> [Int] -> Int -> IO ()
dispatch engine family name buffers groups = dispatchChoices engine family name [buffers] (int 0) groups

-- | Encode the shared embedding, head, or argmax resource layout.
dispatchHead :: Engine -> String -> [Int] -> [Int] -> Int -> IO ()
dispatchHead engine name [weights, inputs, output, chosen] dimensions groups = do
    parameters <- uniform engine "u32" (map int dimensions)
    dispatch engine "head" name [weights, inputs, output, parameters, chosen] groups
dispatchHead _ _ _ _ _ = ioError (userError "Invalid head resource count")

-- | Generate a representation check and its temporary uniform resources.
checkCommands :: State -> Int -> String -> Int -> Int -> Int -> Int -> IO ([JSON], [Int])
checkCommands state encoder entry input count status chosen = do
    (dimensions, resources, additions) <- uniformCommands "u32" (map int [count, 0, 0, 0])
    let groups = if entry == "selection" then 1 else min 65535 ((count + 127) `div` 128)
    pure (additions ++ [command "dispatch" [int encoder, int (named entry (statePipelines state)), int (named "checks" (stateLayouts state)), integers [input, status, dimensions, chosen], int groups]], resources)

-- | Require the exact stored name and shape for a decoder weight.
weightShape :: Int -> String -> Maybe (String, Int, Int)
weightShape layers name
    | name == "model.norm.weight" = Just ("norm", 2560, 1)
    | Just suffix <- stripPrefix "model.layers." name
    , let (digits, remaining) = span isDigit suffix
    , not (null digits)
    , length digits <= 2
    , show (read digits :: Int) == digits
    , read digits < layers
    , '.' : component <- remaining =
        lookup component shapes
    | otherwise = Nothing
  where
    shapes =
        [ ("self_attn.q_proj.weight", ("linear", 2560, 2560))
        , ("self_attn.k_proj.weight", ("linear", 2560, 640))
        , ("self_attn.v_proj.weight", ("linear", 2560, 640))
        , ("self_attn.o_proj.weight", ("linear", 2560, 2560))
        , ("mlp.gate_proj.weight", ("linear", 2560, 6912))
        , ("mlp.up_proj.weight", ("linear", 2560, 6912))
        , ("mlp.down_proj.weight", ("linear", 6912, 2560))
        , ("input_layernorm.weight", ("norm", 2560, 1))
        , ("self_attn.attn_sub_norm.weight", ("norm", 2560, 1))
        , ("post_attention_layernorm.weight", ("norm", 2560, 1))
        , ("mlp.ffn_sub_norm.weight", ("norm", 6912, 1))
        ]

-- | Add an uploaded resource only after its GPU representation check succeeds.
retain :: Engine -> String -> Int -> Value -> IO ()
retain engine@(Engine reference) name identifier tensor = do
    state <- liveState engine
    writeIORef reference state{stateWeights = (name, identifier) : stateWeights state, stateValues = (identifier, tensor) : stateValues state, stateResources = identifier : stateResources state}

-- | Upload and validate a weight while its bytes stay outside WASM memory.
uploadWeight :: Engine -> Weight -> Int -> Browser ()
uploadWeight engine description source = do
    state <- liftBrowser (liveState engine)
    let (name, kind, columns, rows, scale, format) = case description of
            LinearWeight label input output factor -> (label, "linear", input, output, factor, "packed")
            FloatNorm label count -> (label, "norm", count, 1, 1, "finiteFloat")
            BFloatNorm label count -> (label, "norm", count, 1, 1, "bf16")
    bytes <- liftBrowser (hostSize source)
    let expected = if kind == "linear" then columns * (rows `div` 4) else columns * (if format == "bf16" then 2 else 4)
    liftBrowser $ ensure (isNothing (stateFrame state) && stateNext state == 0 && lookup name (stateWeights state) == Nothing && weightShape (stateLayers state) name == Just (kind, columns, rows) && bytes == expected && finite scale && finite (realToFrac (realToFrac scale :: Float))) ("Invalid weight upload: " ++ name)
    (stored, allocation) <- liftBrowser (bufferCommand bytes 140 source)
    (status, statusAllocation) <- liftBrowser (bufferCommand 4 132 0)
    (readback, readbackAllocation) <- liftBrowser (bufferCommand 4 9 0)
    encoder <- liftBrowser newHandle
    result <- liftBrowser newHandle
    (checks, checkResources) <- liftBrowser (checkCommands state encoder format stored (bytes `div` 4) status (stateDummyChosen state))
    (target, conversion, conversionResources) <-
        if format == "bf16"
            then do
                (output, outputAllocation) <- liftBrowser (bufferCommand (columns * 4) 140 0)
                (dimensions, resources, additions) <- liftBrowser (uniformCommands "u32" (map int [columns, 0, 0, 0]))
                let decode = command "dispatch" [int encoder, int (named "embedding" (statePipelines state)), int (named "head" (stateLayouts state)), integers [stored, stateDummy state, output, dimensions, stateDummyChosen state], int ((columns + 127) `div` 128)]
                pure (output, outputAllocation : additions ++ [decode], stored : resources)
            else pure (stored, [], [])
    commands $ [allocation, statusAllocation, readbackAllocation, command "encoder" [int encoder]] ++ checks ++ conversion ++ [command "copy" [int encoder, int status, int 0, int readback, int 0, int 4], command "submit" [int encoder], command "readback" [int readback, int result], command "release" [integers (status : readback : checkResources ++ conversionResources), JBool True]]
    invalid <- liftBrowser (hostNumber result 0 "u32")
    liftBrowser (dropHost result)
    liftBrowser (ensure (invalid == 0) ("Invalid stored values in " ++ name))
    liftBrowser (retain engine name target (if kind == "linear" then Linear target columns rows scale else Norm target columns))

-- | Upload one consecutive BF16 embedding chunk and validate it on the GPU.
uploadEmbedding :: Engine -> Int -> Int -> Browser ()
uploadEmbedding engine sourceRow source = do
    state <- liftBrowser (liveState engine)
    bytes <- liftBrowser (hostSize source)
    let existing = lookup "model.embed_tokens.weight" (stateWeights state)
    previous <- case existing of
        Nothing -> pure ([], 0)
        Just identifier -> do
            stored <- liftBrowser (value engine identifier)
            case stored of
                Embedding chunks rows -> pure (chunks, rows)
                _ -> liftBrowser (ioError (userError "Embedding handle has the wrong role"))
    let (chunks, priorRows) = previous
        rows = bytes `div` (2560 * 2)
    liftBrowser (ensure (isNothing (stateFrame state) && stateNext state == 0 && sourceRow == priorRows && rows > 0 && rows <= 16384 && bytes == rows * 2560 * 2 && sourceRow + rows <= 128256) "Invalid embedding chunk")
    (buffer, allocation) <- liftBrowser (bufferCommand bytes 136 source)
    (status, statusAllocation) <- liftBrowser (bufferCommand 4 132 0)
    (readback, readbackAllocation) <- liftBrowser (bufferCommand 4 9 0)
    encoder <- liftBrowser newHandle
    result <- liftBrowser newHandle
    (checks, resources) <- liftBrowser (checkCommands state encoder "bf16" buffer (bytes `div` 4) status (stateDummyChosen state))
    commands $ [allocation, statusAllocation, readbackAllocation, command "encoder" [int encoder]] ++ checks ++ [command "copy" [int encoder, int status, int 0, int readback, int 0, int 4], command "submit" [int encoder], command "readback" [int readback, int result], command "release" [integers (status : readback : resources), JBool True]]
    invalid <- liftBrowser (hostNumber result 0 "u32")
    liftBrowser (dropHost result)
    liftBrowser (ensure (invalid == 0) "Nonfinite embedding weight")
    identifier <- maybe (liftBrowser newHandle) pure existing
    liftBrowser $ case engine of
        Engine reference -> do
            current <- readIORef reference
            let updated = Embedding (chunks ++ [(buffer, sourceRow, rows)]) (sourceRow + rows)
            writeIORef reference current{stateWeights = ("model.embed_tokens.weight", identifier) : filter ((/= "model.embed_tokens.weight") . fst) (stateWeights current), stateValues = (identifier, updated) : filter ((/= identifier) . fst) (stateValues current), stateResources = buffer : stateResources current}

-- | Restart cache positions without copying or reallocating resident weights.
resetEngine :: Engine -> IO ()
resetEngine engine@(Engine reference) = do
    state <- liveState engine
    ensure (isNothing (stateFrame state) && not (stateRunning state)) "Cannot reset an active tensor engine"
    forM_ (stateValues state) $ \(_, tensor) -> case tensor of
        Cache _ _ _ position -> writeIORef position (-1)
        _ -> pure ()
    writeIORef reference state{stateNext = 0}

-- | Release persistent browser resources after the application completes.
disposeEngine :: Engine -> Browser ()
disposeEngine engine@(Engine reference) = do
    state <- liftBrowser (liveState engine)
    liftBrowser (ensure (isNothing (stateFrame state) && not (stateRunning state)) "Cannot dispose an active tensor engine")
    commands [command "release" [integers (stateResources state), JBool True]]
    liftBrowser (writeIORef reference state{stateClosed = True, stateResources = [], stateValues = [], stateWeights = []})

-- | Resolve a named weight or lazily allocate one persistent KV cache.
weight :: Engine -> String -> IO Tensor
weight engine@(Engine reference) name = do
    (state, _) <- activeFrame engine
    case lookup name (stateWeights state) of
        Just identifier -> pure (Tensor identifier)
        Nothing -> case stripPrefix "cache." name of
            Just suffix
                | let (digits, remaining) = span isDigit suffix
                , not (null digits)
                , length digits <= 2
                , show (read digits :: Int) == digits
                , let layer = read digits
                , layer < stateLayers state
                , ['.', role] <- remaining
                , role == 'k' || role == 'v' -> do
                    (buffer, allocation) <- bufferCommand (2048 * 640 * 4) 140 0
                    position <- newIORef (-1)
                    setup engine [allocation]
                    current <- readIORef reference
                    writeIORef reference current{stateWeights = (name, buffer) : stateWeights current, stateValues = (buffer, Cache buffer layer role position) : stateValues current, stateResources = buffer : stateResources current}
                    pure (Tensor buffer)
            _ -> ioError (userError ("Unknown tensor weight: " ++ name))

-- | Select a GPU-resident rotary row once before a token's operations.
configureRotary :: Engine -> Int -> IO ()
configureRotary engine position = do
    (state, frame) <- activeFrame engine
    ensure (position >= 0 && position < 2048 && position == framePosition frame && position == stateNext state && not (frameRotary frame) && frameOperations frame == 0) "Invalid rotary position or operation order"
    emit engine [command "copy" [int (stateRotary state), argument 3, int (stateFactors state), int 0, int 512]]
    alterFrame engine (\current -> current{frameRotary = True})

-- | Encode one model operation after checking its shapes and tensor roles.
operation :: Engine -> String -> Tensor -> Tensor -> Tensor -> Int -> Double -> IO Tensor
operation engine name (Tensor first) (Tensor second) (Tensor third) parameter scalar = do
    (state, frame) <- activeFrame engine
    ensure (frameOperations frame < 2048 && finite scalar && (name == "attention" || (third == 0 && scalar == 0))) "Invalid tensor operation parameters"
    alterFrame engine (\current -> current{frameOperations = frameOperations current + 1})
    case name of
        "embedding" -> do
            source <- value engine first
            case source of
                Embedding chunks rows | rows == 128256 && second == 0 && parameter >= 0 && parameter < rows && not (frameEmbedded frame) -> do
                    output <- resultVector engine 2560 0
                    dimensions <- uniform engine "u32" [int 2560, int 0, argument 1, int 0]
                    dispatchChoices engine "head" "embedding" [[buffer, stateDummy state, output, dimensions, stateDummyChosen state] | (buffer, _, _) <- chunks] (argument 0) 20
                    alterFrame engine (\current -> current{frameEmbedded = True})
                    pure (Tensor output)
                _ -> ioError (userError "Invalid embedding operation")
        "head" -> do
            (input, count) <- vector engine first
            source <- value engine second
            case (source, frameFinal frame) of
                (Embedding chunks rows, Just (hidden, normalized)) | count == 2560 && rows == 128256 && length chunks == 8 && parameter == 0 && normalized == input && not (frameHead frame) -> do
                    logits <- allocate engine (128256 * 4) 132 0
                    chosen <- allocate engine 8 132 0
                    forM_ chunks $ \(buffer, start, chunkRows) -> dispatchHead engine "head" [buffer, input, logits, chosen] [2560, chunkRows, 0, start] ((chunkRows + 127) `div` 128)
                    dispatchHead engine "argmax" [stateDummy state, stateDummy state, logits, chosen] [2560, 128256, 0, 0] 1
                    alterFrame engine (\current -> current{frameHead = True, frameValues = (chosen, Chosen chosen hidden normalized logits) : frameValues current})
                    pure (Tensor chosen)
                _ -> ioError (userError "Invalid final head input")
        "linear" -> do
            (input, count) <- vector engine first
            matrix <- value engine second
            case matrix of
                Linear buffer columns rows scale | columns == count && parameter == 0 && toInteger columns * 128 <= 2147483647 -> do
                    output <- resultVector engine rows 0
                    quantized <- allocate engine (count * 4) 128 0
                    dimensions <- uniform engine "u32" (map int [columns, rows, 0, 0])
                    parameters <- uniform engine "f32" [JNumber scale, int 0, int 0, int 0]
                    peak <- allocate engine 4 128 0
                    let resources = [buffer, input, quantized, output, dimensions, parameters, peak]
                    dispatch engine "bitlinear" "quantize" resources 1
                    dispatch engine "bitlinear" "bitlinear" resources rows
                    pure (Tensor output)
                _ -> ioError (userError "Invalid BitLinear shape")
        "cache" -> do
            (input, count) <- vector engine first
            destination <- value engine second
            case destination of
                Cache buffer _ _ position -> do
                    previous <- readIORef position
                    ensure (count == 640 && parameter == framePosition frame && previous == parameter - 1 && position `notElem` frameCaches frame) "Invalid or repeated cache update"
                    emit engine [command "copy" [int input, int 0, int buffer, argument 2, int (640 * 4)]]
                    alterFrame engine (\current -> current{frameCaches = position : frameCaches current})
                    pure (Tensor second)
                _ -> ioError (userError "Expected a KV cache")
        _ -> do
            (input, count) <- vector engine first
            (other, keys, values, dimensions, groups) <- case name of
                "rmsnorm" -> do
                    norm <- value engine second
                    case norm of
                        Norm buffer width | width == count && parameter == 0 -> pure (buffer, stateDummy state, stateDummy state, [count, 0, 0, 0], 1)
                        _ -> ioError (userError "Invalid RMS normalization weight")
                "add" -> elementwise state input count
                "gated" -> elementwise state input count
                "rope" -> do
                    ensure (second == 0 && (count == 2560 || count == 640) && parameter == framePosition frame && frameRotary frame) "Invalid rotary operation"
                    pure (stateFactors state, stateDummy state, stateDummy state, [count, 128, parameter, 0], (count + 255) `div` 256)
                "attention" -> do
                    key <- value engine second
                    val <- value engine third
                    case (key, val) of
                        (Cache keyBuffer keyLayer 'k' keyPosition, Cache valueBuffer valueLayer 'v' valuePosition) -> do
                            ensure (count == 2560 && keyLayer == valueLayer && parameter == framePosition frame && keyPosition `elem` frameCaches frame && valuePosition `elem` frameCaches frame && (realToFrac scalar :: Float) == realToFrac (1 / sqrt (128 :: Double)) && frameAttention frame < stateLayers state) "Invalid attention caches or scale"
                            alterFrame engine (\current -> current{frameAttention = frameAttention current + 1})
                            pure (stateDummy state, keyBuffer, valueBuffer, [0, 20, 5, 128], 20)
                        _ -> ioError (userError "Invalid attention cache roles")
                _ -> ioError (userError ("Unsupported tensor operation: " ++ name))
            output <- resultVector engine count 0
            let dimensionValues = case name of
                    "attention" -> [argument 4, int 20, int 5, int 128]
                    "rope" -> [int count, int 128, argument 5, int 0]
                    _ -> map int dimensions
            dimensionsBuffer <- uniform engine "u32" dimensionValues
            parameterBuffer <- uniform engine "f32" [JNumber scalar, int 0, int 0, int 0]
            dispatch engine "layer" name [input, other, output, dimensionsBuffer, parameterBuffer, keys, values] groups
            when (name == "rmsnorm" && lookup "model.norm.weight" (stateWeights state) == Just second) $ alterFrame engine (\current -> current{frameFinal = Just (input, output)})
            when (name == "add" && frameCheckpoints frame) $
                alterFrame engine (\current -> let adds = frameAdds current + 1 in current{frameAdds = adds, frameLayers = if even adds then frameLayers current ++ [output] else frameLayers current})
            pure (Tensor output)
  where
    elementwise state _ count = do
        (other, otherCount) <- vector engine second
        ensure (count == otherCount && parameter == 0) "Invalid elementwise operand"
        pure (other, stateDummy state, stateDummy state, [count, 0, 0, 0], (count + 127) `div` 128)

-- | Check a program once and register its reusable buffers and GPU commands.
preparePlan :: Engine -> Bool -> Bool -> (Int -> IO Int) -> Browser Plan
preparePlan engine@(Engine reference) checkpoints hasInput encode = do
    state <- liftBrowser (liveState engine)
    liftBrowser (ensure (isNothing (stateFrame state) && not (stateRunning state) && stateNext state == 0 && (not checkpoints || stateLayers state == 30)) "Plans must be prepared before token execution")
    let frame =
            Frame
                { framePosition = 0
                , frameCheckpoints = checkpoints
                , frameSetup = []
                , frameCommands = []
                , frameSlot = 0
                , frameUniforms = []
                , frameValues = []
                , frameCaches = []
                , frameRotary = False
                , frameEmbedded = False
                , frameHead = False
                , frameFinal = Nothing
                , frameAdds = 0
                , frameLayers = []
                , frameAttention = 0
                , frameOperations = 0
                }
    liftBrowser (writeIORef reference state{stateFrame = Just frame})
    (input, output) <-
        liftBrowser $
            ( do
                input <- if hasInput then resultVector engine 2560 0 else pure 0
                output <- encode input
                pure (input, output)
            )
                `onException` modifyIORef reference (\current -> current{stateClosed = True, stateFrame = Nothing})
    (current, encoded) <- liftBrowser (activeFrame engine)
    tensor <- liftBrowser (value engine output)
    (hidden, normalized, chosen) <- case tensor of
        Vector buffer count -> do
            liftBrowser (ensure (count == 2560) "Invalid decoder output shape")
            pure (buffer, snd <$> frameFinal encoded, Nothing)
        Chosen selection hidden norm logits -> pure (hidden, Just norm, Just (selection, logits))
        _ -> liftBrowser (ioError (userError "Decoder returned a non-activation tensor"))
    liftBrowser $ do
        ensure (hasInput /= frameEmbedded encoded && frameHead encoded == (chosen /= Nothing)) "Program input or head output differs from its plan"
        ensure (length (frameCaches encoded) == stateLayers current * 2 && frameAttention encoded == stateLayers current) "Decoder omitted a cache or attention operation"
        when (stateLayers current == 30) $ case (frameFinal encoded, normalized) of
            (Just (original, result), Just expected) -> ensure (original == hidden && result == expected) "Network returned the wrong normalized output"
            _ -> ioError (userError "Network omitted final normalization")
        when checkpoints (ensure (frameAdds encoded == 60 && length (frameLayers encoded) == 30) "Network omitted layer checkpoints")
    let hiddenBytes = 2560 * 4
        chosenOffset = hiddenBytes * 2
        layerStart = if chosen == Nothing then hiddenBytes + maybe 0 (const hiddenBytes) normalized else chosenOffset + 8
        offsets = [layerStart + index * hiddenBytes | index <- [0 .. length (frameLayers encoded) - 1]]
        statusOffset = layerStart + length offsets * hiddenBytes
        readbackBytes = statusOffset + 4
    readback <- liftBrowser (allocate engine readbackBytes 9 0)
    status <- liftBrowser (allocate engine 4 140 0)
    liftBrowser $ do
        emit engine [command "clear" [int status, int 0, int 4], command "copy" [int hidden, int 0, int readback, int 0, int hiddenBytes]]
        case normalized of
            Nothing -> pure ()
            Just buffer -> emit engine [command "copy" [int buffer, int 0, int readback, int hiddenBytes, int hiddenBytes]]
        case chosen of
            Nothing -> pure ()
            Just (selection, logits) -> do
                dimensions <- uniform engine "u32" (map int [128256, 0, 0, 0])
                dispatch engine "checks" "finiteFloat" [logits, status, dimensions, selection] 1002
                dispatch engine "checks" "selection" [logits, status, dimensions, selection] 1
                emit engine [command "copy" [int selection, int 0, int readback, int chosenOffset, int 8]]
        forM_ (zip (frameLayers encoded) offsets) $ \(buffer, offset) -> emit engine [command "copy" [int buffer, int 0, int readback, int offset, int hiddenBytes]]
        emit engine [command "copy" [int status, int 0, int readback, int statusOffset, int 4]]
    (_, completed) <- liftBrowser (activeFrame engine)
    chunks <-
        if frameEmbedded completed
            then case lookup "model.embed_tokens.weight" (stateWeights current) of
                Just identifier -> do
                    source <- liftBrowser (value engine identifier)
                    case source of
                        Embedding chunks _ -> pure chunks
                        _ -> liftBrowser (ioError (userError "Plan embedding has the wrong role"))
                Nothing -> liftBrowser (ioError (userError "Plan embedding is missing"))
            else pure []
    identifier <- liftBrowser newHandle
    commands $ reverse (frameSetup completed) ++ [command "plan" [int identifier, JArray (reverse (frameCommands completed))]]
    liftBrowser (modifyIORef reference (\latest -> latest{stateFrame = Nothing, stateResources = identifier : stateResources latest}))
    pure $
        Plan
            { planIdentifier = identifier
            , planOwner = stateIdentity state
            , planInput = if hasInput then Just input else Nothing
            , planNormalized = normalized /= Nothing
            , planSelectionOffset = if chosen == Nothing then Nothing else Just chosenOffset
            , planLayerOffsets = offsets
            , planStatusOffset = statusOffset
            , planReadback = readback
            , planBytes = readbackBytes
            , planCaches = frameCaches completed
            , planEmbedding = chunks
            }

-- | Update explicit token arguments and replay a checked plan without allocations.
execute :: Engine -> Plan -> Int -> Maybe Int -> Maybe Int -> Browser Step
execute engine@(Engine reference) plan position tokenData inputData = do
    state <- liftBrowser (liveState engine)
    liftBrowser (ensure (planOwner plan == stateIdentity state && isNothing (stateFrame state) && not (stateRunning state) && position >= 0 && position < 2048 && position == stateNext state) "Token positions must be consecutive below 2048 in the plan's engine")
    started <- liftBrowser clock
    liftBrowser $ forM_ (planCaches plan) $ \cache -> do
        previous <- readIORef cache
        ensure (previous == position - 1) "Plan cache positions differ from the input position"
    (chunk, row) <- case (planEmbedding plan, tokenData, planInput plan, inputData) of
        (chunks, Just token, Nothing, Nothing) | not (null chunks) && token >= 0 && token < 128256 ->
            case [(index, token - first) | (index, (_, first, count)) <- zip [0 ..] chunks, token >= first && token < first + count] of
                [selected] -> pure selected
                _ -> liftBrowser (ioError (userError "Token has no unique embedding chunk"))
        ([], Nothing, Just _, Just source) -> do
            bytes <- liftBrowser (hostSize source)
            liftBrowser (ensure (bytes == 2560 * 4) "Invalid layer input size")
            pure (0, 0)
        _ -> liftBrowser (ioError (userError "Plan input does not match its program"))
    host <- liftBrowser newHandle
    liftBrowser (modifyIORef reference (\latest -> latest{stateRunning = True}))
    let arguments = integers [chunk, row, position * 640 * 4, position * 512, position + 1, position]
        inputCommands = case (planInput plan, inputData) of
            (Just buffer, Just source) -> [command "write-host" [int buffer, int source]]
            _ -> []
    commands $ inputCommands ++ [command "run" [int (planIdentifier plan), arguments], command "readback" [int (planReadback plan), int host]]
    size <- liftBrowser (hostSize host)
    liftBrowser (ensure (size == planBytes plan) "Invalid plan readback size")
    invalid <- liftBrowser (hostNumber host (planStatusOffset plan) "u32")
    liftBrowser (ensure (invalid == 0) "GPU logits contain nonfinite values or an invalid selection")
    (token, logit) <- case planSelectionOffset plan of
        Nothing -> pure (Nothing, Nothing)
        Just chosenOffset -> do
            selected <- liftBrowser (hostNumber host chosenOffset "u32")
            score <- liftBrowser (hostNumber host (chosenOffset + 4) "f32")
            liftBrowser (ensure (selected >= 0 && selected < 128256 && finite score) "Invalid GPU token or logit")
            pure (Just (round selected), Just score)
    ended <- liftBrowser clock
    liftBrowser $ do
        mapM_ (\cache -> writeIORef cache position) (planCaches plan)
        latest <- readIORef reference
        writeIORef reference latest{stateRunning = False, stateNext = position + 1}
    pure (Step host (planNormalized plan) token logit (planLayerOffsets plan) Nothing (ended - started))
