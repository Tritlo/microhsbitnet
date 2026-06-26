-- | Weighted dots via the C kernel in @cbits/bndot.c@, called with plain
-- @foreign import ccall@ — no MicroHs runtime patch, no custom primitive. The
-- loops run in C (vectorized + OpenMP-parallel) instead of as per-element
-- combinator graph reduction, and they are /batched/: one FFI call does a whole
-- BitLinear matmul or a whole LM-head argmax, not one dot per call.
--
-- The list-op 'BitLinear.dotTrits' stays the readable spec for the ternary case.
module Dot
  ( matmulTernary
  , argmaxBf16
  , argmaxBf16BS
  , lmLogit
  , f32Dot
  , floatDot
  , bitLinearPrim
  , quantizeActBS
  , packF32
  , unpackF32
  , bnRmsNorm
  , bnBitLinearF
  , bnRope
  , bnAttend
  , bnSqReluMul
  , bnAdd
  , decodeBf16
  ) where

import qualified Data.ByteString as BS
import Data.ByteString.Unsafe (unsafeUseAsCStringLen)
import Data.Float (doubleToFloat)
import Foreign.C.String (CString)
import Foreign.Marshal.Alloc (mallocBytes, free, allocaBytes, alloca)
import Foreign.Marshal.Utils (copyBytes)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, peekElemOff, pokeElemOff)
import System.IO.Unsafe (unsafePerformIO)
import BitLinear (quantizeAct)

foreign import ccall "bndot.h bnmatmul_ternary"
  c_matmul_ternary :: CString -> CString -> Int -> Int -> Ptr Double -> IO ()
foreign import ccall "bndot.h bnargmax_bf16"
  c_argmax_bf16 :: CString -> CString -> Int -> Int -> Ptr Double -> IO Int
foreign import ccall "bndot.h bndot_f32"
  c_f32 :: CString -> CString -> Int -> IO Double
foreign import ccall "bndot.h bnrmsnorm"
  c_rmsnorm :: CString -> CString -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "bndot.h bnbitlinear"
  c_bitlinear :: CString -> Float -> CString -> Int -> Int -> Ptr Float -> IO ()
foreign import ccall "bndot.h bnrope"
  c_rope :: Ptr Float -> Int -> Int -> Int -> Float -> IO ()
foreign import ccall "bndot.h bnattend"
  c_attend :: CString -> CString -> CString -> Int -> Int -> Int -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "bndot.h bnsqrelu_mul"
  c_sqrelu_mul :: CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "bndot.h bnadd"
  c_add :: CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "bndot.h bf16_to_f32"
  c_bf16_to_f32 :: CString -> Int -> Ptr Float -> IO ()

-- | A whole W1.58 BitLinear matmul: @outDim@ raw integer dots (as Double) of the
-- packed ternary weight against the int8 activations @xqBS@. One FFI call; the
-- result buffer is read back before it is freed.
matmulTernary :: BS.ByteString -> BS.ByteString -> Int -> Int -> [Double]
matmulTernary packed xqBS inDim outDim = unsafePerformIO $
  unsafeUseAsCStringLen packed $ \ (wp, _) ->
  unsafeUseAsCStringLen xqBS   $ \ (ap, _) ->
  allocaBytes (outDim * 8)     $ \ out -> do
    c_matmul_ternary wp ap inDim outDim out
    mapM (peekElemOff out) [0 .. outDim - 1]

-- | The tied LM head's argmax: scans the whole @vocab@ of the bf16 @embedB@ rows
-- dotted with the f32 @hidden@ vector, returning @(argmax token, logit)@.
argmaxBf16 :: BS.ByteString -> [Float] -> Int -> Int -> (Int, Float)
argmaxBf16 embedB hidden = argmaxBf16BS embedB (packF32 hidden)

-- | As 'argmaxBf16' but the hidden vector is already an f32 byte buffer (e.g. a
-- 'bnRmsNorm' result), avoiding a pack\/unpack round-trip at the final norm.
argmaxBf16BS :: BS.ByteString -> BS.ByteString -> Int -> Int -> (Int, Float)
argmaxBf16BS embedB hbs hiddenDim vocab = unsafePerformIO $
  unsafeUseAsCStringLen embedB $ \ (ep, _) ->
  unsafeUseAsCStringLen hbs    $ \ (hp, _) ->
  alloca                       $ \ best -> do
    bi <- c_argmax_bf16 ep hp hiddenDim vocab best
    bv <- peek best
    return (bi, doubleToFloat bv)

-- | A single LM-head logit for token @v@ (a diagnostic): the bf16 embedding row
-- @v@ · the f32 @hidden@. Reuses 'argmaxBf16' over the one-row slice at @v@.
lmLogit :: BS.ByteString -> [Float] -> Int -> Int -> Float
lmLogit embedB hidden hiddenDim v =
  snd (argmaxBf16 (BS.drop (v * hiddenDim * 2) embedB) hidden hiddenDim 1)

-- | One float32 · float32 dot of two equal-length float32 byte vectors.
f32Dot :: BS.ByteString -> BS.ByteString -> Int -> Double
f32Dot w a n = unsafePerformIO $
  unsafeUseAsCStringLen w $ \ (wp, _) ->
  unsafeUseAsCStringLen a $ \ (ap, _) -> c_f32 wp ap n

-- | Float32 dot for attention q·k scores (and any full-precision dot).
floatDot :: BS.ByteString -> BS.ByteString -> Int -> Float
floatDot w a len = doubleToFloat (f32Dot w a len)

-- | BitLinear forward: the whole ternary matmul in one C call, then rescale each
-- output by @wscale \/ ascale@ (kept in Haskell so the result is unchanged).
bitLinearPrim :: BS.ByteString -> Int -> Int -> Float -> BS.ByteString -> Float -> [Float]
bitLinearPrim packed inDim outDim wscale xqBS ascale =
  [ doubleToFloat d * wscale / ascale | d <- matmulTernary packed xqBS inDim outDim ]

-- | Quantize activations to int8 and pack them as a ByteString (two's complement,
-- read by the kernel as @int8_t@), with the scale. For the ternary activations.
quantizeActBS :: [Float] -> (BS.ByteString, Float)
quantizeActBS x =
  let (xq, sc) = quantizeAct x
  in (BS.pack (map fromIntegral xq), sc)

-- | Serialize a Float vector to native little-endian float32 bytes. For the bf16
-- and float32 dots' activations.
packF32 :: [Float] -> BS.ByteString
packF32 xs = unsafePerformIO $ do
  let n = length xs
  p <- mallocBytes (n * 4)
  let go _ []       = return ()
      go i (y : ys) = pokeElemOff (p :: Ptr Float) i y >> go (i + 1) ys
  go 0 xs
  bs <- BS.packCStringLen (castPtr p, n * 4)
  free p
  return bs

-- | Read a float32 byte buffer back to a Float list.
unpackF32 :: BS.ByteString -> [Float]
unpackF32 bs = unsafePerformIO $
  unsafeUseAsCStringLen bs $ \ (p, len) ->
    mapM (peekElemOff (castPtr p :: Ptr Float)) [0 .. len `div` 4 - 1]

-- Run a C kernel that writes @n@ float32 results into a fresh buffer, returning
-- the buffer as a ByteString. The buffer is on the stack (no malloc churn) and
-- 'packCStringLen' copies it out before it is freed.
withOut :: Int -> (Ptr Float -> IO ()) -> BS.ByteString
withOut n act = unsafePerformIO $ allocaBytes (n * 4) $ \ p -> do
  act p
  BS.packCStringLen (castPtr p, n * 4)

-- The per-token vector math of the generation path, on float32 byte buffers.

-- | RMSNorm: @w * x / sqrt(mean(x^2) + eps)@ over @n@ floats.
bnRmsNorm :: BS.ByteString -> BS.ByteString -> Int -> Float -> BS.ByteString
bnRmsNorm w x n eps = withOut n $ \ out ->
  unsafeUseAsCStringLen w $ \ (wp, _) ->
  unsafeUseAsCStringLen x $ \ (xp, _) -> c_rmsnorm wp xp n eps out

-- | Fused BitLinear on f32 activations: absmax-quantize to int8, ternary matmul,
-- rescale by @wscale@. @packed@ is the row-major packed ternary weight.
bnBitLinearF :: BS.ByteString -> Float -> BS.ByteString -> Int -> Int -> BS.ByteString
bnBitLinearF packed wscale x inDim outDim = withOut outDim $ \ out ->
  unsafeUseAsCStringLen packed $ \ (pp, _) ->
  unsafeUseAsCStringLen x      $ \ (xp, _) -> c_bitlinear pp wscale xp inDim outDim out

-- | Rotary position embedding at @pos@ on each of @nHeads@ heads of @headDim@.
bnRope :: BS.ByteString -> Int -> Int -> Int -> Float -> BS.ByteString
bnRope vec pos nHeads headDim theta = unsafePerformIO $
  let n = BS.length vec `div` 4 in
  allocaBytes (n * 4) $ \ p -> do
    unsafeUseAsCStringLen vec $ \ (vp, len) -> copyBytes (castPtr p) vp len
    c_rope p pos nHeads headDim theta
    BS.packCStringLen (castPtr p, n * 4)

-- | GQA causal attention of query @q@ against @nPos@ cached keys @cK@ / values
-- @cV@. Returns the @nHeads*headDim@ attention output.
bnAttend :: BS.ByteString -> BS.ByteString -> BS.ByteString
         -> Int -> Int -> Int -> Int -> Float -> BS.ByteString
bnAttend q cK cV nPos nHeads nKV headDim scale = withOut (nHeads * headDim) $ \ out ->
  unsafeUseAsCStringLen q  $ \ (qp, _) ->
  unsafeUseAsCStringLen cK $ \ (kp, _) ->
  unsafeUseAsCStringLen cV $ \ (vp, _) -> c_attend qp kp vp nPos nHeads nKV headDim scale out

-- | Squared-ReLU gated MLP elementwise: @max(0,g)^2 * u@ over @n@ floats.
bnSqReluMul :: BS.ByteString -> BS.ByteString -> Int -> BS.ByteString
bnSqReluMul g u n = withOut n $ \ out ->
  unsafeUseAsCStringLen g $ \ (gp, _) ->
  unsafeUseAsCStringLen u $ \ (up, _) -> c_sqrelu_mul gp up n out

-- | Residual add of two @n@-float buffers.
bnAdd :: BS.ByteString -> BS.ByteString -> Int -> BS.ByteString
bnAdd a b n = withOut n $ \ out ->
  unsafeUseAsCStringLen a $ \ (ap, _) ->
  unsafeUseAsCStringLen b $ \ (bp, _) -> c_add ap bp n out

-- | Decode @n@ bf16 values to a float32 byte buffer (one-time weight setup, e.g.
-- the RMSNorm weights — far faster than @bf16At@ per element in Haskell).
decodeBf16 :: BS.ByteString -> Int -> BS.ByteString
decodeBf16 src n = withOut n $ \ out ->
  unsafeUseAsCStringLen src $ \ (sp, _) -> c_bf16_to_f32 sp n out
