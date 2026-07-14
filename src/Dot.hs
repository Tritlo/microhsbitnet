-- | The C kernels in @cbits/b1dot.c@, called with plain @foreign import ccall@
-- — no MicroHs runtime patch, no custom primitive. The loops run in C
-- (OpenMP-parallel) instead of as per-element combinator graph reduction, and
-- they are batched: one FFI call does a whole B1Linear matmul.
--
-- Weights are @W = (bytes, offset)@ handles into the whole resident GGUF, not
-- pre-sliced ByteStrings: MicroHs's runtime narrows substring offsets to a C
-- int (@addForPtr(.., int s)@), so slicing past 2 GiB wraps and crashes —
-- resolving the offset with 64-bit pointer arithmetic at the FFI boundary
-- avoids the whole class. Activations are small fresh ByteStrings and unaffected.
module Dot
  ( W
  , b1Linear
  , b1EmbedRow
  , bnRmsNorm
  , bnRmsNormHeads
  , bnRope
  , bnAttend
  , bnAdd
  , bnSplitQGate
  , bnSigmoidMul
  , bnSiluMul
  , gdnConvSilu
  , gdnL2Heads
  , gdnStep
  , b1Argmax
  , b1TopK
  , packF32
  , unpackF32
  ) where

import qualified Data.ByteString as BS
import Data.ByteString.Unsafe (unsafeUseAsCStringLen)
import Foreign.C.String (CString)
import Foreign.Marshal.Alloc (mallocBytes, free, allocaBytes)
import Foreign.Marshal.Utils (copyBytes)
import Foreign.Ptr (Ptr, castPtr, plusPtr)
import Foreign.Storable (peekElemOff, pokeElemOff)
import System.IO.Unsafe (unsafePerformIO)

-- | A weight handle: the whole resident file plus a byte offset (which may
-- exceed 2 GiB — see the module comment).
type W = (BS.ByteString, Int)

withW :: W -> (CString -> IO a) -> IO a
withW (bs, off) act = unsafeUseAsCStringLen bs $ \ (p, _) -> act (p `plusPtr` off)

foreign import ccall "b1dot.h b1matmul"
  c_b1matmul :: CString -> CString -> Int -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h b1embed"
  c_b1embed :: CString -> Int -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnrmsnorm"
  c_rmsnorm :: CString -> CString -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnrope"
  c_rope :: Ptr Float -> Int -> Int -> Int -> Int -> Float -> IO ()
foreign import ccall "b1dot.h bnattend"
  c_attend :: CString -> CString -> CString -> Int -> Int -> Int -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnadd"
  c_add :: CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnrmsnorm_heads"
  c_rmsnorm_heads :: CString -> CString -> Int -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnsplit_qgate"
  c_split_qgate :: CString -> Int -> Int -> Ptr Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnsigmoid_mul"
  c_sigmoid_mul :: CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h bnsilu_mul"
  c_silu_mul :: CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h gdn_conv_silu"
  c_gdn_conv :: CString -> CString -> CString -> Int -> Ptr Float -> IO ()
foreign import ccall "b1dot.h gdn_l2norm_heads"
  c_gdn_l2 :: CString -> Int -> Int -> Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h gdn_step"
  c_gdn_step :: CString -> CString -> CString -> CString -> CString -> CString
             -> CString -> CString -> Int -> Int -> Int -> Ptr Float -> Ptr Float -> IO ()
foreign import ccall "b1dot.h b1argmax"
  c_b1argmax :: CString -> CString -> Int -> Int -> Ptr Float -> IO Int
foreign import ccall "b1dot.h b1topk"
  c_b1topk :: CString -> CString -> Int -> Int -> Int -> Ptr Int -> Ptr Float -> IO ()

-- Run a C kernel that writes @n@ float32 results into a fresh buffer, returning
-- the buffer as a ByteString ('packCStringLen' copies it out before it is freed).
withOut :: Int -> (Ptr Float -> IO ()) -> BS.ByteString
withOut n act = unsafePerformIO $ allocaBytes (n * 4) $ \ p -> do
  act p
  BS.packCStringLen (castPtr p, n * 4)

-- | A full B1Linear on f32 activations: quantize q8_0-style in C, Q1_0 matmul,
-- group-scale rescale — one FFI call, f32 bytes in and out.
b1Linear :: W -> BS.ByteString -> Int -> Int -> BS.ByteString
b1Linear w x inDim outDim = withOut outDim $ \ out ->
  withW w $ \ wp ->
  unsafeUseAsCStringLen x $ \ (xp, _) -> c_b1matmul wp xp inDim outDim out

-- | Dequantize one Q1_0 row of @rowLen@ weights (the embedding lookup).
b1EmbedRow :: W -> Int -> Int -> BS.ByteString
b1EmbedRow w row rowLen = withOut rowLen $ \ out ->
  withW w $ \ wp -> c_b1embed wp row rowLen out

-- | RMSNorm: @w * x / sqrt(mean(x^2) + eps)@ over @n@ floats.
bnRmsNorm :: W -> BS.ByteString -> Int -> Float -> BS.ByteString
bnRmsNorm w x n eps = withOut n $ \ out ->
  withW w $ \ wp ->
  unsafeUseAsCStringLen x $ \ (xp, _) -> c_rmsnorm wp xp n eps out

-- | Neox rotary embedding at @pos@ on each of @nHeads@ heads of @headDim@;
-- only the first @nRot@ dims of each head rotate (partial rotary).
bnRope :: BS.ByteString -> Int -> Int -> Int -> Int -> Float -> BS.ByteString
bnRope vec pos nHeads headDim nRot theta = unsafePerformIO $
  let n = BS.length vec `div` 4 in
  allocaBytes (n * 4) $ \ p -> do
    unsafeUseAsCStringLen vec $ \ (vp, len) -> copyBytes (castPtr p) vp len
    c_rope p pos nHeads headDim nRot theta
    BS.packCStringLen (castPtr p, n * 4)

-- | GQA causal attention of query @q@ against @nPos@ cached keys @cK@ / values
-- @cV@. Returns the @nHeads*headDim@ attention output.
bnAttend :: BS.ByteString -> BS.ByteString -> BS.ByteString
         -> Int -> Int -> Int -> Int -> Float -> BS.ByteString
bnAttend q cK cV nPos nHeads nKV headDim scale = withOut (nHeads * headDim) $ \ out ->
  unsafeUseAsCStringLen q  $ \ (qp, _) ->
  unsafeUseAsCStringLen cK $ \ (kp, _) ->
  unsafeUseAsCStringLen cV $ \ (vp, _) -> c_attend qp kp vp nPos nHeads nKV headDim scale out

-- | Residual add of two @n@-float buffers.
bnAdd :: BS.ByteString -> BS.ByteString -> Int -> BS.ByteString
bnAdd a b n = withOut n $ \ out ->
  unsafeUseAsCStringLen a $ \ (ap, _) ->
  unsafeUseAsCStringLen b $ \ (bp, _) -> c_add ap bp n out

-- | Per-head RMSNorm (the q/k norms): nHeads spans of headDim, shared weight.
bnRmsNormHeads :: W -> BS.ByteString -> Int -> Int -> Float -> BS.ByteString
bnRmsNormHeads w x nHeads headDim eps = withOut (nHeads * headDim) $ \ out ->
  withW w $ \ wp ->
  unsafeUseAsCStringLen x $ \ (xp, _) -> c_rmsnorm_heads wp xp nHeads headDim eps out

-- | Deinterleave the fused Q projection into (query, gate) buffers.
bnSplitQGate :: BS.ByteString -> Int -> Int -> (BS.ByteString, BS.ByteString)
bnSplitQGate qg nHeads headDim = unsafePerformIO $
  allocaBytes (n * 4) $ \ qp ->
  allocaBytes (n * 4) $ \ gp -> do
    unsafeUseAsCStringLen qg $ \ (p, _) -> c_split_qgate p nHeads headDim qp gp
    q <- BS.packCStringLen (castPtr qp, n * 4)
    g <- BS.packCStringLen (castPtr gp, n * 4)
    return (q, g)
  where n = nHeads * headDim

-- | Output gate: @a * sigmoid(g)@ over @n@ floats.
bnSigmoidMul :: BS.ByteString -> BS.ByteString -> Int -> BS.ByteString
bnSigmoidMul g a n = withOut n $ \ out ->
  unsafeUseAsCStringLen g $ \ (gp, _) ->
  unsafeUseAsCStringLen a $ \ (ap, _) -> c_sigmoid_mul gp ap n out

-- | SwiGLU elementwise: @silu(g) * u@ over @n@ floats.
bnSiluMul :: BS.ByteString -> BS.ByteString -> Int -> BS.ByteString
bnSiluMul g u n = withOut n $ \ out ->
  unsafeUseAsCStringLen g $ \ (gp, _) ->
  unsafeUseAsCStringLen u $ \ (up, _) -> c_silu_mul gp up n out

-- | Causal 4-tap depthwise conv + SiLU for one token: ring = previous 3
-- channel vectors (oldest first), x = current, kernel per-channel contiguous.
gdnConvSilu :: BS.ByteString -> BS.ByteString -> W -> Int -> BS.ByteString
gdnConvSilu ring x kernel channels = withOut channels $ \ out ->
  unsafeUseAsCStringLen ring   $ \ (rp, _) ->
  unsafeUseAsCStringLen x      $ \ (xp, _) ->
  withW kernel $ \ kp -> c_gdn_conv rp xp kp channels out

-- | Per-head L2 norm: @x / max(sqrt(sum x^2), eps)@ per head span.
gdnL2Heads :: BS.ByteString -> Int -> Int -> Float -> BS.ByteString
gdnL2Heads x nHeads headDim eps = withOut (nHeads * headDim) $ \ out ->
  unsafeUseAsCStringLen x $ \ (xp, _) -> c_gdn_l2 xp nHeads headDim eps out

-- | One gated-delta-rule token step over all heads. Returns the attention
-- output (nV*d floats) and the updated state (nV*d*d floats, fresh buffer).
gdnStep :: BS.ByteString -> BS.ByteString -> BS.ByteString -> BS.ByteString
        -> BS.ByteString -> W -> W -> BS.ByteString
        -> Int -> Int -> Int -> (BS.ByteString, BS.ByteString)
gdnStep st q k v alphaP dt aArr betaP nV nK d = unsafePerformIO $
  allocaBytes (nV * d * 4) $ \ outP ->
  allocaBytes (nV * d * d * 4) $ \ stP -> do
    unsafeUseAsCStringLen st     $ \ (sp, _) ->
      unsafeUseAsCStringLen q      $ \ (qp, _) ->
      unsafeUseAsCStringLen k      $ \ (kp, _) ->
      unsafeUseAsCStringLen v      $ \ (vp, _) ->
      unsafeUseAsCStringLen alphaP $ \ (ap, _) ->
      withW dt   $ \ dp ->
      withW aArr $ \ aap ->
      unsafeUseAsCStringLen betaP  $ \ (bp, _) ->
        c_gdn_step sp qp kp vp ap dp aap bp nV nK d outP stP
    o <- BS.packCStringLen (castPtr outP, nV * d * 4)
    s <- BS.packCStringLen (castPtr stP, nV * d * d * 4)
    return (o, s)

-- | The Q1_0 LM head's argmax over the vocab: @(token, logit)@.
b1Argmax :: W -> BS.ByteString -> Int -> Int -> (Int, Float)
b1Argmax w x inDim vocab = unsafePerformIO $
  withW w $ \ wp ->
  unsafeUseAsCStringLen x $ \ (xp, _) ->
  allocaBytes 4 $ \ best -> do
    i <- c_b1argmax wp xp inDim vocab best
    v <- peekElemOff best 0
    return (i, v)

-- | The top-@k@ @(token, logit)@ pairs (descending) of the Q1_0 LM head.
b1TopK :: W -> BS.ByteString -> Int -> Int -> Int -> [(Int, Float)]
b1TopK w x inDim vocab k = unsafePerformIO $
  withW w $ \ wp ->
  unsafeUseAsCStringLen x $ \ (xp, _) ->
  allocaBytes (k * 8) $ \ idxP ->
  allocaBytes (k * 4) $ \ valP -> do
    c_b1topk wp xp inDim vocab k idxP valP
    is <- mapM (peekElemOff idxP) [0 .. k - 1]
    vs <- mapM (peekElemOff valP) [0 .. k - 1]
    return (zip is vs)

-- | Serialize a Float vector to native little-endian float32 bytes.
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
