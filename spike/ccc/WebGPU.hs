module WebGPU (
    newHandle,
    initialize,
    prepareCommand,
    completeCommand,
    retainHost,
    hostSize,
    dropHost,
    copiedBytes,
    statistics,
    cleanup,
) where

import Control.Exception (bracket, onException)
import Control.Monad (forM_, unless, when)
import Data.Bits ((.&.))
import Data.Float (doubleToFloat)
import Data.IORef
import Data.Int (Int64)
import Data.List (groupBy, sortOn)
import Data.Word (Word32)
import Foreign.Marshal.Alloc (free)
import Foreign.Marshal.Array (mallocArray, pokeArray)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (pokeElemOff)
import Mhs.MutArr
import System.IO.Unsafe (unsafePerformIO)
import qualified WebFFI
import WebJSON

-- | Describe browser objects without retaining their JavaScript representation.
data Kind
    = Host Int
    | Buffer Int Int
    | Shader
    | Layout [String] Int
    | Pipeline Int
    | Encoder
    | Binding
    | Prepared Replay

-- | Track one resource identity and its explicit release operation.
data Resource = Resource Int Kind (IORef Bool) (IO ())

-- | Retain checked commands and their reusable host and GPU allocations.
data Replay = Replay [([Int] -> IO ())] [IO ()] [Resource] Int (IORef Int) Int

-- | Represent a constant or a checked argument slot in a prepared command.
data Parameter = Constant Int | Argument Int

-- | Keep allocation and transfer counters in MicroHs.
data Counters = Counters
    { uploadedBytes :: Int64
    , submissions :: Int
    , copiedToWasmBytes :: Int64
    , bulkCopiedToWasmBytes :: Int64
    , liveBufferBytes :: Int64
    , bufferAllocations :: Int
    , bindGroupCreations :: Int
    , planCompilations :: Int
    , planRuns :: Int
    }

-- | Share one identifier supply for requests and browser objects.
identifiers :: IORef Int
identifiers = unsafePerformIO (newIORef 0)
{-# NOINLINE identifiers #-}

-- | Index resource metadata in a growable MicroHs array.
resources :: IORef (MutIOArr (Maybe Resource))
resources = unsafePerformIO (newMutIOArr 1024 Nothing >>= newIORef)
{-# NOINLINE resources #-}

-- | Retain the limits returned by the browser device.
limits :: IORef JSON
limits = unsafePerformIO (newIORef JNull)
{-# NOINLINE limits #-}

-- | Count transfers and allocations independently of the browser bridge.
counters :: IORef Counters
counters = unsafePerformIO (newIORef (Counters 0 0 0 0 0 0 0 0 0))
{-# NOINLINE counters #-}

-- | Invalidate cached dependency checks only when a GPU resource is released.
releaseVersion :: IORef Int
releaseVersion = unsafePerformIO (newIORef 0)
{-# NOINLINE releaseVersion #-}

-- | Reject an invalid command before issuing browser API calls.
ensure :: Bool -> String -> IO ()
ensure condition message = unless condition (ioError (userError message))

-- | Allocate a positive identifier within the WASM Int range.
newHandle :: IO Int
newHandle = do
    current <- readIORef identifiers
    ensure (current < 2147483647) "Browser identifier limit reached"
    let identifier = current + 1
    writeIORef identifiers identifier
    pure identifier

-- | Store the capabilities of the initialized device.
initialize :: JSON -> IO ()
initialize = writeIORef limits

-- | Read one device limit after initialization.
limit :: String -> IO Int
limit name = asInt . field name <$> readIORef limits

-- | Grow the resource table without changing existing resource identities.
tableFor :: Int -> IO (MutIOArr (Maybe Resource))
tableFor identifier = do
    ensure (identifier > 0) "Invalid resource identifier"
    table <- readIORef resources
    size <- sizeMutIOArr table
    if identifier < size
        then pure table
        else do
            let capacity = until (> identifier) (* 2) size
            larger <- newMutIOArr capacity Nothing
            forM_ [0 .. size - 1] $ \index -> unsafeReadMutIOArr table index >>= unsafeWriteMutIOArr larger index
            writeIORef resources larger
            pure larger

-- | Require a live resource of the expected kind.
resource :: Int -> IO Resource
resource identifier = do
    table <- readIORef resources
    size <- sizeMutIOArr table
    ensure (identifier > 0 && identifier < size) ("Invalid resource handle " ++ show identifier)
    found <- unsafeReadMutIOArr table identifier
    case found of
        Just retained -> pure retained
        Nothing -> ioError (userError ("Invalid resource handle " ++ show identifier))

-- | Register one resource and reject duplicate handles.
retain :: Int -> Kind -> IO () -> IO ()
retain identifier kind release = do
    available identifier
    table <- tableFor identifier
    alive <- newIORef True
    unsafeWriteMutIOArr table identifier (Just (Resource identifier kind alive release))

-- | Reject duplicate or unallocated handles before creating browser objects.
available :: Int -> IO ()
available identifier = do
    maximum <- readIORef identifiers
    ensure (identifier > 0 && identifier <= maximum) "Invalid resource identifier"
    table <- tableFor identifier
    found <- unsafeReadMutIOArr table identifier
    ensure (case found of Nothing -> True; _ -> False) "Duplicate resource handle"

-- | Return the byte size and usage flags of a live GPU buffer.
buffer :: Int -> IO (Int, Int)
buffer identifier = do
    Resource _ kind _ _ <- resource identifier
    case kind of
        Buffer size usage -> pure (size, usage)
        _ -> ioError (userError "Expected a GPU buffer")

-- | Check a byte range without overflowing a WASM Int.
bounds :: Int -> Int -> Int -> Int -> IO ()
bounds size offset count alignment =
    ensure (offset >= 0 && count >= 0 && offset `mod` alignment == 0 && count `mod` alignment == 0 && offset <= size && count <= size - offset) "Byte range exceeds its buffer or alignment"

-- | Issue one synchronous browser operation without a JSON conversion.
call :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
call = WebFFI.gpuCall

-- | Copy a short descriptor into WASM memory for one browser API call.
descriptor :: Int -> Int -> Int -> [Int] -> IO ()
descriptor opcode identifier other values =
    bracket (mallocArray (length values)) free $ \pointer -> do
        pokeArray pointer (map toEnum values :: [Word32])
        WebFFI.gpuArray opcode identifier other pointer (length values)

-- | Retain an opaque host allocation with its byte length in MicroHs.
retainHost :: Int -> Int -> IO ()
retainHost identifier size = do
    ensure (size >= 0) "Invalid host buffer length"
    retain identifier (Host size) (call 9 identifier 0 0 0 0 0)

-- | Read host allocation metadata without crossing the browser boundary.
hostSize :: Int -> IO Int
hostSize identifier = do
    Resource _ kind _ _ <- resource identifier
    case kind of
        Host size -> pure size
        _ -> ioError (userError "Expected a host byte buffer")

-- | Release an opaque host allocation. Repeated cleanup has no effect.
dropHost :: Int -> IO ()
dropHost identifier = do
    table <- readIORef resources
    size <- sizeMutIOArr table
    when (identifier > 0 && identifier < size) $ do
        found <- unsafeReadMutIOArr table identifier
        case found of
            Nothing -> pure ()
            Just (Resource _ (Host _) _ _) -> releaseOne True identifier
            _ -> ioError (userError "Cannot drop a GPU resource as host bytes")

-- | Count scalar, protocol, and bulk byte copies into WASM.
copiedBytes :: Int -> Bool -> IO ()
copiedBytes count bulk = modifyIORef' counters $ \state ->
    state
        { copiedToWasmBytes = copiedToWasmBytes state + toEnum count
        , bulkCopiedToWasmBytes = bulkCopiedToWasmBytes state + if bulk then toEnum count else 0
        }

-- | Release one resource and invalidate every plan that depends on it.
releaseOne :: Bool -> Int -> IO ()
releaseOne destroy identifier = do
    table <- readIORef resources
    size <- sizeMutIOArr table
    when (identifier > 0 && identifier < size) $ do
        found <- unsafeReadMutIOArr table identifier
        case found of
            Nothing -> pure ()
            Just (Resource _ kind alive release) -> do
                unsafeWriteMutIOArr table identifier Nothing
                writeIORef alive False
                case kind of
                    Host _ -> pure ()
                    _ -> modifyIORef' releaseVersion (+ 1)
                case kind of
                    Buffer bytes _ -> do
                        modifyIORef' counters (\state -> state{liveBufferBytes = liveBufferBytes state - toEnum bytes})
                        if destroy then release else call 9 identifier 0 0 0 0 0
                    _ -> release

-- | Release all retained GPU objects and plan-owned WASM arrays.
cleanup :: IO ()
cleanup = do
    table <- readIORef resources
    size <- sizeMutIOArr table
    mapM_ (releaseOne True) [1 .. size - 1]

-- | Report counters without an asynchronous browser request.
statistics :: IO JSON
statistics = do
    state <- readIORef counters
    pure $
        JObject
            [ ("uploadedBytes", JNumber (fromIntegral (uploadedBytes state)))
            , ("submissions", int (submissions state))
            , ("copiedToWasmBytes", JNumber (fromIntegral (copiedToWasmBytes state)))
            , ("bulkCopiedToWasmBytes", JNumber (fromIntegral (bulkCopiedToWasmBytes state)))
            , ("liveBufferBytes", JNumber (fromIntegral (liveBufferBytes state)))
            , ("bufferAllocations", int (bufferAllocations state))
            , ("bindGroupCreations", int (bindGroupCreations state))
            , ("planCompilations", int (planCompilations state))
            , ("planRuns", int (planRuns state))
            ]

-- | Create an aligned buffer after checking the device limits.
allocate :: Int -> Int -> Int -> IO ()
allocate identifier size usage = do
    maximum <- limit "maxBufferSize"
    storage <- limit "maxStorageBufferBindingSize"
    ensure (size > 0 && size `mod` 4 == 0 && size <= maximum && usage > 0 && usage <= 1023 && (usage .&. 128 == 0 || size <= storage)) "GPU buffer size exceeds its alignment or device limit"
    retain identifier (Buffer size usage) (call 10 identifier 0 0 0 0 0)
    call 0 identifier size usage 0 0 0
    modifyIORef' counters (\state -> state{liveBufferBytes = liveBufferBytes state + toEnum size, bufferAllocations = bufferAllocations state + 1})

-- | Upload host bytes directly to a checked GPU buffer.
writeHost :: Int -> Int -> IO ()
writeHost identifier host = do
    (size, usage) <- buffer identifier
    bytes <- hostSize host
    let padded = ((bytes + 3) `div` 4) * 4
    ensure (usage .&. 8 /= 0 && padded <= size) "Host upload exceeds its GPU buffer or usage"
    call 1 identifier host padded 0 0 0
    modifyIORef' counters (\state -> state{uploadedBytes = uploadedBytes state + toEnum padded})

-- | Convert a binding representation into the browser descriptor code.
bindingType :: String -> Int
bindingType name = case name of
    "uniform" -> 0
    "storage" -> 1
    "read-only-storage" -> 2
    _ -> error "Invalid buffer binding type"

-- | Validate binding usage and retain one reusable bind group.
bindGroup :: Int -> [Int] -> IO Int
bindGroup layout identifiers_ = do
    Resource _ kind _ _ <- resource layout
    types <- case kind of
        Layout types _ -> pure types
        _ -> ioError (userError "Expected a GPU bind group layout")
    ensure (length identifiers_ == length types) "Dispatch binding count differs"
    maximum <- limit "maxUniformBufferBindingSize"
    forM_ (zip types identifiers_) $ \(kind, identifier) -> do
        (size, usage) <- buffer identifier
        let required = if kind == "uniform" then 64 else 128
        ensure (usage .&. required /= 0 && (kind /= "uniform" || size <= maximum)) "Dispatch buffer usage or uniform size differs"
    identifier <- newHandle
    descriptor 1 identifier layout identifiers_
    retain identifier Binding (call 9 identifier 0 0 0 0 0)
    modifyIORef' counters (\state -> state{bindGroupCreations = bindGroupCreations state + 1})
    pure identifier

-- | Check that a pipeline uses the supplied bind group layout.
checkPipeline :: Int -> Int -> IO ()
checkPipeline identifier layout = do
    Resource _ kind _ _ <- resource identifier
    ensure (case kind of Pipeline expected -> expected == layout; _ -> False) "Pipeline layout handle differs"

-- | Require a live command encoder before recording browser operations.
checkEncoder :: Int -> IO ()
checkEncoder identifier = do
    Resource _ kind _ _ <- resource identifier
    ensure (case kind of Encoder -> True; _ -> False) "Expected a GPU command encoder"

-- | Check a workgroup count against the device limit.
checkGroups :: Int -> IO ()
checkGroups groups = do
    maximum <- limit "maxComputeWorkgroupsPerDimension"
    ensure (groups > 0 && groups <= maximum) "Invalid workgroup count"

-- | Check copy usage and both buffer ranges before encoding.
checkCopy :: Int -> Int -> Int -> Int -> Int -> IO ()
checkCopy source first target second count = do
    (sourceSize, sourceUsage) <- buffer source
    (targetSize, targetUsage) <- buffer target
    ensure (source /= target && sourceUsage .&. 4 /= 0 && targetUsage .&. 8 /= 0) "Invalid copy buffers or usage"
    bounds sourceSize first count 4
    bounds targetSize second count 4

-- | Parse one constant or argument slot without retaining protocol syntax.
parameter :: JSON -> Parameter
parameter (JArray [JString "arg", value])
    | asInt value >= 0 && asInt value < 2147483647 = Argument (asInt value)
parameter value = Constant (asInt value)

-- | Resolve a previously checked argument slot.
resolve :: [Int] -> Parameter -> Int
resolve _ (Constant value) = value
resolve arguments (Argument index) = arguments !! index

-- | Return the number of arguments required by one parameter.
argumentCount :: Parameter -> Int
argumentCount (Constant _) = 0
argumentCount (Argument index) = index + 1

-- | Compile a pass boundary once instead of testing pass state during replay.
passBoundary :: Bool -> Int -> Int -> IORef Bool -> IO (IO ())
passBoundary opening encoder pass active = do
    started <- readIORef active
    writeIORef active opening
    pure $
        if opening == started
            then pure ()
            else
                if opening
                    then call 3 pass encoder 0 0 0 0
                    else call 4 pass 0 0 0 0 0 >> call 9 pass 0 0 0 0 0

-- | Store a typed uniform array without a JavaScript number conversion.
uniformData :: String -> [JSON] -> IO (Ptr ())
uniformData kind values = do
    ensure (not (null values)) "Cannot allocate an empty GPU array"
    pointer <- mallocArray (length values) :: IO (Ptr Word32)
    ( case kind of
            "u32" -> pokeArray pointer (map unsigned values)
            "f32" -> pokeArray (castPtr pointer) (map floating values)
            _ -> ioError (userError "Unknown array representation")
        )
        `onException` free pointer
    pure (castPtr pointer)
  where
    unsigned (JInt number) | number >= 0 = toEnum number
    unsigned value = let number = asDouble value in if number >= 0 && number <= 4294967295 && number == fromInteger (truncate number) then fromInteger (truncate number) else error "Invalid u32 value"
    floating value = let number = doubleToFloat (asDouble value) in if isNaN number || isInfinite number then error "Invalid f32 value" else number

-- | Compile a plan once and retain its static bindings and uniform arrays.
compilePlan :: Int -> [JSON] -> IO ()
compilePlan identifier templates = do
    available identifier
    encoder <- newHandle
    pass <- newHandle
    active <- newIORef False
    dependencies <- newIORef []
    owned <- newIORef []
    allocations <- newIORef []
    writes <- newIORef []
    let depend key = do
            retained <- resource key
            modifyIORef' dependencies (retained :)
        own key = modifyIORef' owned (key :)
        compile template = case asArray template of
            [JString "dispatch", pipelineValue, layoutValue, choicesValue, selectionValue, groupsValue] -> do
                let pipeline = asInt pipelineValue
                    layout = asInt layoutValue
                    choices = map (map asInt . asArray) (asArray choicesValue)
                    selection = parameter selectionValue
                    groups = parameter groupsValue
                checkPipeline pipeline layout
                mapM_ depend (pipeline : layout : concat choices)
                ensure (not (null choices)) "Cannot compile empty binding choices"
                bindings <- mapM (bindGroup layout) choices
                mapM_ own bindings
                mapM_ depend bindings
                opening <- passBoundary True encoder pass active
                prepared <- newIORef (0, 1)
                let required = max (argumentCount selection) (argumentCount groups)
                    prepare arguments = do
                        let selected = resolve arguments selection
                            count = resolve arguments groups
                        ensure (selected >= 0 && selected < length bindings) "Invalid binding selection"
                        checkGroups count
                        writeIORef prepared (bindings !! selected, count)
                    encodeWith (binding, count) = do
                        opening
                        call 5 pass pipeline binding count 0 0
                    encode =
                        if required == 0
                            then encodeWith (bindings !! resolve [] selection, resolve [] groups)
                            else readIORef prepared >>= encodeWith
                pure (required, prepare, encode)
            [JString "copy", sourceValue, firstValue, targetValue, secondValue, countValue] -> do
                let source = asInt sourceValue
                    target = asInt targetValue
                    parameters = map parameter [firstValue, secondValue, countValue]
                (sourceSize, sourceUsage) <- buffer source
                (targetSize, targetUsage) <- buffer target
                ensure (source /= target && sourceUsage .&. 4 /= 0 && targetUsage .&. 8 /= 0) "Invalid plan copy buffers or usage"
                mapM_ depend [source, target]
                closing <- passBoundary False encoder pass active
                prepared <- newIORef (0, 0, 0)
                let prepare arguments = do
                        let [first, second, count] = map (resolve arguments) parameters
                        bounds sourceSize first count 4
                        bounds targetSize second count 4
                        writeIORef prepared (first, second, count)
                    encode = do
                        (first, second, count) <- readIORef prepared
                        closing
                        call 6 encoder source first target second count
                pure (maximum (map argumentCount parameters), prepare, encode)
            [JString "clear", bufferValue, firstValue, countValue] -> do
                let identifier = asInt bufferValue
                    parameters = map parameter [firstValue, countValue]
                depend identifier
                closing <- passBoundary False encoder pass active
                (size, usage) <- buffer identifier
                ensure (usage .&. 8 /= 0) "Plan clear requires COPY_DST"
                prepared <- newIORef (0, 0)
                let prepare arguments = do
                        let [first, count] = map (resolve arguments) parameters
                        bounds size first count 4
                        writeIORef prepared (first, count)
                    encode = do
                        (first, count) <- readIORef prepared
                        closing
                        call 7 encoder identifier first count 0 0
                pure (maximum (map argumentCount parameters), prepare, encode)
            [JString "write", bufferValue, kindValue, valuesValue] -> do
                let identifier = asInt bufferValue
                    kind = asString kindValue
                    values = asArray valuesValue
                    dynamic = [(index, parameter value) | (index, value@(JArray _)) <- zip [0 ..] values]
                    count = maximum (0 : map (argumentCount . snd) dynamic)
                    initial value = case value of JArray _ -> int 0; _ -> value
                    bytes = length values * 4
                depend identifier
                (size, usage) <- buffer identifier
                written <- readIORef writes
                ensure (usage .&. 8 /= 0 && identifier `notElem` written && bytes <= size) "Plan writes require COPY_DST and one bounded write per buffer"
                modifyIORef' writes (identifier :)
                pointer <- uniformData kind (map initial values)
                modifyIORef' allocations (pointer :)
                let prepare arguments = forM_ dynamic $ \(index, value) -> do
                        let number = resolve arguments value
                        ensure (number >= 0 || kind == "f32") "Invalid u32 value"
                        if kind == "u32"
                            then pokeElemOff (castPtr pointer) index (toEnum number :: Word32)
                            else pokeElemOff (castPtr pointer) index (fromIntegral number :: Float)
                    encode = do
                        WebFFI.gpuWrite identifier pointer bytes
                        modifyIORef' counters (\state -> state{uploadedBytes = uploadedBytes state + toEnum bytes})
                pure (count, prepare, encode)
            _ -> ioError (userError "Unknown plan operation")
        release = do
            readIORef owned >>= mapM_ (releaseOne True)
            readIORef allocations >>= mapM_ free
            call 9 encoder 0 0 0 0 0
            call 9 pass 0 0 0 0 0
    ( do
            compiled <- mapM compile templates
            forM_ compiled $ \(count, prepare, _) -> when (count == 0) (prepare [])
            closing <- passBoundary False encoder pass active
            version <- readIORef releaseVersion >>= newIORef
            retained <- readIORef dependencies
            let key (Resource identifier _ _ _) = identifier
                unique = map head (groupBy (\first second -> key first == key second) (sortOn key retained))
                replay = Replay [prepare | (count, prepare, _) <- compiled, count /= 0] ([encode | (_, _, encode) <- compiled] ++ [closing]) unique encoder version (maximum (0 : [count | (count, _, _) <- compiled]))
            retain identifier (Prepared replay) release
        )
        `onException` release
    modifyIORef' counters (\state -> state{planCompilations = planCompilations state + 1})

-- | Validate arguments and lifetimes before replaying synchronous GPU calls.
runPlan :: Int -> [Int] -> IO ()
runPlan identifier arguments = do
    Resource _ kind _ _ <- resource identifier
    replay <- case kind of
        Prepared prepared -> pure prepared
        _ -> ioError (userError "Expected a prepared GPU plan")
    let Replay prepares actions dependencies encoder checkedVersion required = replay
    ensure (length arguments >= required) "Missing plan arguments"
    current <- readIORef releaseVersion
    previous <- readIORef checkedVersion
    when (current /= previous) $ do
        forM_ dependencies $ \(Resource key _ alive _) -> do
            retained <- readIORef alive
            ensure retained ("Released plan dependency " ++ show key)
        writeIORef checkedVersion current
    mapM_ ($ arguments) prepares
    call 2 encoder 0 0 0 0 0
    sequence_ actions
    call 8 encoder 0 0 0 0 0
    call 9 encoder 0 0 0 0 0
    modifyIORef' counters (\state -> state{submissions = submissions state + 1, planRuns = planRuns state + 1})

-- | Execute a synchronous command or return the required asynchronous API call.
prepareCommand :: JSON -> IO (Maybe JSON)
prepareCommand value = case asArray value of
    [JString "plan", identifier, templates] -> compilePlan (asInt identifier) (asArray templates) >> pure Nothing
    [JString "run", identifier, arguments] -> runPlan (asInt identifier) (map asInt (asArray arguments)) >> pure Nothing
    [JString "write-host", identifier, host] -> writeHost (asInt identifier) (asInt host) >> pure Nothing
    [JString "buffer", identifier, size, usage, host] -> do
        allocate (asInt identifier) (asInt size) (asInt usage)
        when (asInt host /= 0) (writeHost (asInt identifier) (asInt host))
        pure Nothing
    [JString "array", identifier, kind, values] -> do
        let items = asArray values
            bytes = length items * 4
        bracket (uniformData (asString kind) items) free $ \pointer -> do
            allocate (asInt identifier) bytes 204
            WebFFI.gpuWrite (asInt identifier) pointer bytes
        modifyIORef' counters (\state -> state{uploadedBytes = uploadedBytes state + toEnum bytes})
        pure Nothing
    [JString "layout", identifier, bindings] -> do
        available (asInt identifier)
        let types = map asString (asArray bindings)
        pipeline <- newHandle
        descriptor 0 (asInt identifier) pipeline (map bindingType types)
        retain (asInt identifier) (Layout types pipeline) (call 9 (asInt identifier) 0 0 0 0 0 >> call 9 pipeline 0 0 0 0 0)
        pure Nothing
    [JString "shader", identifier, host] -> available (asInt identifier) >> hostSize (asInt host) >> pure (Just value)
    [JString "pipeline", identifier, shader, layout, entry] -> do
        available (asInt identifier)
        Resource _ shaderKind _ _ <- resource (asInt shader)
        ensure (case shaderKind of Shader -> True; _ -> False) "Expected a shader module"
        Resource _ kind _ _ <- resource (asInt layout)
        case kind of
            Layout _ pipeline -> pure (Just (JArray [JString "pipeline", identifier, shader, int pipeline, entry]))
            _ -> ioError (userError "Expected a bind group layout")
    [JString "encoder", identifier] -> do
        retain (asInt identifier) Encoder (call 9 (asInt identifier) 0 0 0 0 0)
        call 2 (asInt identifier) 0 0 0 0 0
        pure Nothing
    [JString "dispatch", encoder, pipeline, layout, bindings, groups] -> do
        checkEncoder (asInt encoder)
        checkPipeline (asInt pipeline) (asInt layout)
        checkGroups (asInt groups)
        binding <- bindGroup (asInt layout) (map asInt (asArray bindings))
        pass <- newHandle
        call 3 pass (asInt encoder) 0 0 0 0
        call 5 pass (asInt pipeline) binding (asInt groups) 0 0
        call 4 pass 0 0 0 0 0
        call 9 pass 0 0 0 0 0
        releaseOne True binding
        pure Nothing
    [JString "copy", encoder, source, first, target, second, count] -> do
        checkEncoder (asInt encoder)
        checkCopy (asInt source) (asInt first) (asInt target) (asInt second) (asInt count)
        call 6 (asInt encoder) (asInt source) (asInt first) (asInt target) (asInt second) (asInt count)
        pure Nothing
    [JString "submit", encoder] -> do
        checkEncoder (asInt encoder)
        call 8 (asInt encoder) 0 0 0 0 0
        releaseOne True (asInt encoder)
        modifyIORef' counters (\state -> state{submissions = submissions state + 1})
        pure Nothing
    [JString "readback", identifier, host] -> do
        available (asInt host)
        (_, usage) <- buffer (asInt identifier)
        ensure (usage .&. 1 /= 0) "Readback buffer is not readable"
        pure (Just value)
    [JString "wait"] -> pure (Just value)
    [JString "release", identifiers_, destroy] -> mapM_ (releaseOne (asBool destroy) . asInt) (asArray identifiers_) >> pure Nothing
    _ -> ioError (userError "Unknown GPU command")

-- | Register the result of an asynchronous browser object operation.
completeCommand :: JSON -> JSON -> IO ()
completeCommand value response = case asArray value of
    [JString "shader", identifier, _] -> retain (asInt identifier) Shader (call 9 (asInt identifier) 0 0 0 0 0)
    [JString "pipeline", identifier, _, layout, _] -> retain (asInt identifier) (Pipeline (asInt layout)) (call 9 (asInt identifier) 0 0 0 0 0)
    [JString "readback", source, host] -> do
        (size, _) <- buffer (asInt source)
        ensure (asInt (field "length" response) == size) "Invalid GPU readback length"
        retainHost (asInt host) size
    [JString "wait"] -> pure ()
    _ -> ioError (userError "Unexpected GPU completion")
