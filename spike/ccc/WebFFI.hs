module WebFFI where

import Data.Word (Word32)
import Foreign.C.String (CString)
import Foreign.Ptr (Ptr)

-- | Invoke one synchronous WebGPU API operation.
foreign import javascript "globalThis.microhsBrowser.call($0, $1, $2, $3, $4, $5, $6)"
    gpuCall :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()

-- | Create layouts or a bind group from a borrowed Word32 array.
foreign import javascript "globalThis.microhsBrowser.array($0, $1, $2, HEAPU8, $3, $4)"
    gpuArray :: Int -> Int -> Int -> Ptr Word32 -> Int -> IO ()

-- | Write borrowed WASM bytes to a GPU buffer.
foreign import javascript "globalThis.microhsBrowser.write($0, HEAPU8, $1, $2)"
    gpuWrite :: Int -> Ptr () -> Int -> IO ()

-- | Submit a generic request. The host completes it after this call returns.
foreign import javascript "globalThis.microhsBrowser.request($0, UTF8ToString($1))"
    browserRequest :: Int -> CString -> IO ()

-- | Read one scalar from an opaque host buffer.
foreign import javascript "return globalThis.microhsBrowser.number($0, $1, UTF8ToString($2))"
    hostNumber :: Int -> Int -> CString -> IO Double

-- | Copy one host byte range into an allocated WASM array.
foreign import javascript "globalThis.microhsBrowser.copy($0, $1, $2, HEAPU8, $3)"
    copyHostBytes :: Int -> Int -> Int -> Ptr () -> IO ()

-- | Append one application message to the browser log.
foreign import javascript "globalThis.microhsBrowser.log(UTF8ToString($0))"
    browserLog :: CString -> IO ()

-- | Read the browser's monotonic clock in milliseconds.
foreign import javascript "return globalThis.microhsBrowser.clock()"
    clock :: IO Double

-- | Publish the application result as JSON.
foreign import javascript "globalThis.microhsBrowser.report(UTF8ToString($0))"
    report :: CString -> IO ()
