module WebBrowser (
    Browser,
    runBrowser,
    liftBrowser,
    request,
    command,
    commands,
    newHandle,
    fetchBytes,
    fetchRange,
    fetchText,
    hostSize,
    hostNumber,
    withFloats,
    readFloats,
    readInts,
    dropHost,
    browserLog,
    clock,
    report,
    browserComplete,
)
where

import Control.Exception (SomeException, bracket, catch, displayException)
import Control.Monad (unless, when)
import Data.Float (floatToDouble)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Int (Int32)
import Data.List (stripPrefix)
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.Marshal.Alloc (free)
import Foreign.Marshal.Array (allocaArray, mallocArray, peekArray)
import Foreign.Ptr (Ptr, castPtr)
import System.IO.Unsafe (unsafePerformIO)
import qualified WebFFI
import qualified WebGPU
import WebJSON

-- | Suspend browser work until the host resumes its continuation.
newtype Browser value = Browser ((value -> IO ()) -> IO ())

instance Functor Browser where
    fmap function (Browser action) = Browser (\resume -> action (resume . function))

instance Applicative Browser where
    pure value = Browser (\resume -> resume value)
    Browser function <*> Browser argument = Browser (\resume -> function (\apply -> argument (resume . apply)))

instance Monad Browser where
    Browser action >>= next = Browser (\resume -> action (\value -> case next value of Browser following -> following resume))

-- | Retain at most one request ticket and its continuation.
pending :: IORef (Maybe (Int, JSON -> IO ()))
pending = unsafePerformIO (newIORef Nothing)
{-# NOINLINE pending #-}

-- | Allocate a positive identifier that also fits the WASM Int representation.
newHandle :: IO Int
newHandle = WebGPU.newHandle

-- | Publish an error and discard the pending application continuation.
browserFailure :: SomeException -> IO ()
browserFailure exception = do
    writeIORef pending Nothing
    report (JObject [("passed", JBool False), ("error", JString (displayException exception))])

-- | Start one browser application without entering an asynchronous WASM call.
runBrowser :: Browser () -> IO ()
runBrowser (Browser action) = catch (action (const (pure ()))) browserFailure

-- | Execute synchronous application work before the next browser request.
liftBrowser :: IO value -> Browser value
liftBrowser action = Browser (\resume -> action >>= resume)

-- | Encode a generic protocol command with an operation name as its first item.
command :: String -> [JSON] -> JSON
command name arguments = JArray (JString name : arguments)

-- | Handle local resource queries before requesting asynchronous browser APIs.
request :: JSON -> Browser JSON
request value = case value of
    JArray [JString "stats"] -> liftBrowser WebGPU.statistics
    JArray [JString "init"] -> do
        response <- browserRequest value
        liftBrowser (WebGPU.initialize response)
        pure response
    _ -> browserRequest value

-- | Retain the continuation before starting an asynchronous browser API call.
browserRequest :: JSON -> Browser JSON
browserRequest value = Browser $ \resume -> do
    waiting <- readIORef pending
    case waiting of
        Just _ -> ioError (userError "A browser request is already pending")
        Nothing -> pure ()
    ticket <- newHandle
    writeIORef pending (Just (ticket, resume))
    withCString (renderJSON value) (WebFFI.browserRequest ticket)

-- | Interpret commands in MicroHs and suspend only for asynchronous browser APIs.
commands :: [JSON] -> Browser ()
commands values = do
    liftBrowser (WebFFI.gpuCall 11 0 0 0 0 0 0)
    mapM_ execute values
    validation <- browserRequest (command "scope" [])
    unless (validation == JNull) (liftBrowser (ioError (userError ("WebGPU: " ++ asString validation))))
  where
    execute value = do
        asynchronous <- liftBrowser (WebGPU.prepareCommand value)
        case asynchronous of
            Nothing -> pure ()
            Just operation -> do
                response <- browserRequest operation
                liftBrowser (WebGPU.completeCommand value response)

-- | Resume the matching request. Status one means success; zero means failure.
browserComplete :: Int -> Int -> CString -> Int -> IO ()
browserComplete ticket status pointer byteCount = catch complete browserFailure
  where
    complete = do
        waiting <- readIORef pending
        resume <- case waiting of
            Just (expected, continuation) | ticket == expected -> pure continuation
            _ -> ioError (userError "Unexpected or stale browser completion ticket")
        writeIORef pending Nothing
        WebGPU.copiedBytes byteCount False
        text <- peekCString pointer
        response <- either (ioError . userError) pure (parseJSON text)
        case status of
            1 -> resume response
            0 -> ioError (userError (asString response))
            _ -> ioError (userError "Invalid browser completion status")

foreign export javascript browserComplete :: Int -> Int -> CString -> Int -> IO ()

-- | Fetch a complete file into an opaque JavaScript host buffer.
fetchBytes :: String -> Browser Int
fetchBytes url = do
    identifier <- liftBrowser newHandle
    response <- request (command "fetch" [int identifier, JString url, int (-1), int (-1)])
    unless (asInt (field "status" response) == 200) (liftBrowser (ioError (userError "Whole-file fetch requires HTTP 200")))
    liftBrowser (WebGPU.retainHost identifier (asInt (field "length" response)))
    pure identifier

-- | Fetch an exact byte range and return its host handle and total file length.
fetchRange :: String -> Int -> Int -> Browser (Int, Int)
fetchRange url start count = do
    unless (start >= 0 && count > 0 && count <= maxBound - start) (liftBrowser (ioError (userError "Invalid fetch range")))
    identifier <- liftBrowser newHandle
    response <- request (command "fetch" [int identifier, JString url, int start, int count])
    let expected = "bytes " ++ show start ++ "-" ++ show (start + count - 1) ++ "/"
    total <- case stripPrefix expected (asString (field "contentRange" response)) of
        Just suffix -> either (liftBrowser . ioError . userError) (pure . asInt) (parseJSON suffix)
        Nothing -> liftBrowser (ioError (userError "HTTP range response differs"))
    unless (asInt (field "status" response) == 206 && asInt (field "length" response) == count && total >= start + count) (liftBrowser (ioError (userError "Browser range length or HTTP status differs")))
    liftBrowser (WebGPU.retainHost identifier count)
    pure (identifier, total)

-- | Decode a fetched UTF-8 file and release its opaque byte buffer.
fetchText :: String -> Browser String
fetchText url = do
    identifier <- fetchBytes url
    response <- request (command "text" [int identifier])
    liftBrowser (dropHost identifier)
    pure (asString response)

-- | Read the byte length of a retained host buffer.
hostSize :: Int -> IO Int
hostSize = WebGPU.hostSize

-- | Read one finite scalar at a byte offset using a generic representation name.
hostNumber :: Int -> Int -> String -> IO Double
hostNumber identifier offset kind = do
    size <- hostSize identifier
    width <- case lookup kind [("u8", 1), ("u16", 2), ("u32", 4), ("i32", 4), ("f32", 4)] of
        Just value -> pure value
        Nothing -> ioError (userError "Unknown host number representation")
    unless (offset >= 0 && offset <= size && width <= size - offset) (ioError (userError "Invalid host scalar range"))
    value <- withCString kind (WebFFI.hostNumber identifier offset)
    when (isNaN value || isInfinite value) (ioError (userError "Nonfinite host scalar"))
    WebGPU.copiedBytes width False
    pure value

-- | Check byte bounds before allocating a temporary diagnostic array.
checkArray :: Int -> Int -> Int -> IO ()
checkArray identifier offset count = do
    size <- hostSize identifier
    unless (offset >= 0 && offset `mod` 4 == 0 && count >= 0 && offset <= size && count <= (size - offset) `div` 4) $
        ioError (userError "Invalid host diagnostic array range")

-- | Copy Float bytes once. The pointer is valid only during the callback.
withFloats :: Int -> Int -> Int -> (Ptr Float -> IO value) -> IO value
withFloats identifier offset count action = do
    checkArray identifier offset count
    bracket (mallocArray count) free $ \pointer -> do
        WebFFI.copyHostBytes identifier offset (count * 4) (castPtr pointer)
        WebGPU.copiedBytes (count * 4) True
        action pointer

-- | Copy a Float diagnostic array once, then read it with Storable.
readFloats :: Int -> Int -> Int -> IO [Double]
readFloats identifier offset count = withFloats identifier offset count $ \pointer -> do
    values <- peekArray count pointer
    let result = map floatToDouble values
    unless (all (\value -> not (isNaN value || isInfinite value)) result) $
        ioError (userError "Nonfinite host diagnostic float")
    pure result

-- | Copy an Int32 diagnostic array once, then read it with Storable.
readInts :: Int -> Int -> Int -> IO [Int]
readInts identifier offset count = do
    checkArray identifier offset count
    allocaArray count $ \pointer -> do
        WebFFI.copyHostBytes identifier offset (count * 4) (castPtr pointer)
        WebGPU.copiedBytes (count * 4) True
        values <- peekArray count (pointer :: Ptr Int32)
        pure (map fromIntegral values)

-- | Release a host buffer. Repeated cleanup calls have no effect.
dropHost :: Int -> IO ()
dropHost = WebGPU.dropHost

-- | Append one message to the browser's retained log.
browserLog :: String -> IO ()
browserLog message = withCString message WebFFI.browserLog

-- | Read a monotonic browser timestamp in milliseconds.
clock :: IO Double
clock = WebFFI.clock

-- | Publish the application result without deriving application metrics in JavaScript.
report :: JSON -> IO ()
report value = WebGPU.cleanup >> withCString (renderJSON value) WebFFI.report
