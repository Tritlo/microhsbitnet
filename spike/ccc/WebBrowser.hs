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
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.Marshal.Alloc (free)
import Foreign.Marshal.Array (allocaArray, mallocArray, peekArray)
import Foreign.Ptr (Ptr, castPtr)
import System.IO.Unsafe (unsafePerformIO)
import qualified WebFFI
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

-- | Share one identifier supply for requests, host buffers, and GPU resources.
identifiers :: IORef Int
identifiers = unsafePerformIO (newIORef 0)
{-# NOINLINE identifiers #-}

-- | Allocate a positive identifier that also fits the WASM Int representation.
newHandle :: IO Int
newHandle = do
    current <- readIORef identifiers
    when (current >= 2147483647) (ioError (userError "Browser identifier limit reached"))
    let identifier = current + 1
    writeIORef identifiers identifier
    pure identifier

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

-- | Submit one request and retain its continuation before invoking JavaScript.
request :: JSON -> Browser JSON
request value = Browser $ \resume -> do
    waiting <- readIORef pending
    case waiting of
        Just _ -> ioError (userError "A browser request is already pending")
        Nothing -> pure ()
    ticket <- newHandle
    writeIORef pending (Just (ticket, resume))
    withCString (renderJSON value) (WebFFI.browserRequest ticket)

-- | Execute a batch of generic host commands in their specified order.
commands :: [JSON] -> Browser ()
commands values = do
    response <- request (command "batch" [JArray values])
    unless (response == JNull) (liftBrowser (ioError (userError "Invalid browser batch response")))

-- | Resume the matching request. Status one means success; zero means failure.
browserComplete :: Int -> Int -> CString -> IO ()
browserComplete ticket status pointer = catch complete browserFailure
  where
    complete = do
        waiting <- readIORef pending
        resume <- case waiting of
            Just (expected, continuation) | ticket == expected -> pure continuation
            _ -> ioError (userError "Unexpected or stale browser completion ticket")
        writeIORef pending Nothing
        text <- peekCString pointer
        response <- either (ioError . userError) pure (parseJSON text)
        case status of
            1 -> resume response
            0 -> ioError (userError (asString response))
            _ -> ioError (userError "Invalid browser completion status")

foreign export javascript browserComplete :: Int -> Int -> CString -> IO ()

-- | Fetch a complete file into an opaque JavaScript host buffer.
fetchBytes :: String -> Browser Int
fetchBytes url = do
    identifier <- liftBrowser newHandle
    _ <- request (command "fetch" [int identifier, JString url, int (-1), int (-1)])
    pure identifier

-- | Fetch an exact byte range and return its host handle and total file length.
fetchRange :: String -> Int -> Int -> Browser (Int, Int)
fetchRange url start count = do
    identifier <- liftBrowser newHandle
    response <- request (command "fetch" [int identifier, JString url, int start, int count])
    unless (asInt (field "length" response) == count) (liftBrowser (ioError (userError "Browser range length differs")))
    pure (identifier, asInt (field "totalBytes" response))

-- | Decode a fetched UTF-8 file and release its opaque byte buffer.
fetchText :: String -> Browser String
fetchText url = do
    identifier <- fetchBytes url
    response <- request (command "text" [int identifier])
    liftBrowser (dropHost identifier)
    pure (asString response)

-- | Read the byte length of a retained host buffer.
hostSize :: Int -> IO Int
hostSize = WebFFI.hostSize

-- | Read one finite scalar at a byte offset using a generic representation name.
hostNumber :: Int -> Int -> String -> IO Double
hostNumber identifier offset kind = withCString kind (WebFFI.hostNumber identifier offset)

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
        values <- peekArray count (pointer :: Ptr Int32)
        pure (map fromIntegral values)

-- | Release a host buffer. Repeated cleanup calls have no effect.
dropHost :: Int -> IO ()
dropHost = WebFFI.dropHost

-- | Append one message to the browser's retained log.
browserLog :: String -> IO ()
browserLog message = withCString message WebFFI.browserLog

-- | Read a monotonic browser timestamp in milliseconds.
clock :: IO Double
clock = WebFFI.clock

-- | Publish the application result without deriving application metrics in JavaScript.
report :: JSON -> IO ()
report value = withCString (renderJSON value) WebFFI.report
