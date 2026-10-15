module WebFFI where

import Foreign.C.String (CString)
import Foreign.Ptr (Ptr)

-- | Submit a generic request. The host completes it after this call returns.
foreign import javascript "globalThis.microhsBrowser.request($0, UTF8ToString($1))"
    browserRequest :: Int -> CString -> IO ()

-- | Read the byte length of an opaque host buffer.
foreign import javascript "return globalThis.microhsBrowser.size($0)"
    hostSize :: Int -> IO Int

-- | Read one bounded scalar from an opaque host buffer.
foreign import javascript "return globalThis.microhsBrowser.number($0, $1, UTF8ToString($2))"
    hostNumber :: Int -> Int -> CString -> IO Double

-- | Copy one host byte range into an allocated WASM array.
foreign import javascript "globalThis.microhsBrowser.copy($0, $1, $2, HEAPU8, $3)"
    copyHostBytes :: Int -> Int -> Int -> Ptr () -> IO ()

-- | Release an opaque host buffer. Repeated calls have no effect.
foreign import javascript "globalThis.microhsBrowser.drop($0)"
    dropHost :: Int -> IO ()

-- | Append one application message to the browser log.
foreign import javascript "globalThis.microhsBrowser.log(UTF8ToString($0))"
    browserLog :: CString -> IO ()

-- | Read the browser's monotonic clock in milliseconds.
foreign import javascript "return globalThis.microhsBrowser.clock()"
    clock :: IO Double

-- | Publish the application result as JSON.
foreign import javascript "globalThis.microhsBrowser.report(UTF8ToString($0))"
    report :: CString -> IO ()
