module WebFFI where

import Foreign.C.String (CString)

-- | Start host work. The host must defer completion until this call returns.
foreign import javascript "return globalThis.microhsGPUStart(UTF8ToString($0))"
    gpuStart :: CString -> IO Int

-- | Read one retained GPU result after asynchronous host work completes.
foreign import javascript "return globalThis.microhsGPUResultAt($0)"
    gpuResultAt :: Int -> IO Double

-- | Read one source-generated reference value.
foreign import javascript "return globalThis.microhsGPUReferenceAt($0)"
    gpuReferenceAt :: Int -> IO Double

-- | Find a model weight or a persistent cache by name.
foreign import javascript "return globalThis.microhsLayer.weight(UTF8ToString($0))"
    findWeight :: CString -> IO Int

-- | Encode an array operation without host numerical computation.
foreign import javascript "return globalThis.microhsLayer.op(UTF8ToString($0), $1, $2, $3, $4, $5)"
    encode :: CString -> Int -> Int -> Int -> Int -> Double -> IO Int

-- | Select one position from the source-derived constant table on the GPU.
foreign import javascript "globalThis.microhsLayer.rotary($0)"
    selectRotary :: Int -> IO ()
