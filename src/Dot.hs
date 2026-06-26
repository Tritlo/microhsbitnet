-- | Weighted-dot via the @bndot@ runtime primitive (vendored MicroHs, T_BNDOT in
-- vendor/microhs/src/runtime/eval.c): the inner dot loop runs in C instead of as
-- per-element combinator graph reduction. One primitive covers both BitNet dots:
-- the ternary BitLinears (kind 0) and the bf16 LM head (kind 1).
--
-- The list-op 'BitLinear.dotTrits' stays the readable spec for the kind-0 case.
module Dot
  ( bnDot
  , quantizeActBS
  , packF32
  , bitLinearPrim
  , lmLogit
  , floatDot
  ) where

import qualified Data.ByteString as BS
import Foreign.Marshal.Alloc (mallocBytes, free)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (pokeElemOff)
import System.IO.Unsafe (unsafePerformIO)
import BitLinear (quantizeAct)

-- | @bnDot kind weights acts shift len@ — the dot in C, returned as a Float.
--   kind 0 = ternary × int8: @weights@ pack 4 trits/byte, @shift@ (0,2,4,6)
--     selects the field; @acts@ are int8. Result = sum_c (trit_c) * a_c.
--   kind 1 = bf16 × float32: @weights@ are dense bf16, @shift@ ignored; @acts@ are
--     float32. Result = sum_c bf16(weights[2c..]) * a_c.
bnDot :: Int -> BS.ByteString -> BS.ByteString -> Int -> Int -> Float
bnDot = _primitive "bndot"

-- | Quantize activations to int8 and pack them as a ByteString (two's complement,
-- read by the primitive as @int8_t@), with the scale. For kind-0 activations.
quantizeActBS :: [Float] -> (BS.ByteString, Float)
quantizeActBS x =
  let (xq, sc) = quantizeAct x
  in (BS.pack (map fromIntegral xq), sc)

-- | Serialize a Float vector to native little-endian float32 bytes. For kind-1
-- activations (the LM-head hidden vector).
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

-- | BitLinear forward via kind-0 bndot: one C dot per output row, then rescale.
bitLinearPrim :: BS.ByteString -> Int -> Int -> Float -> BS.ByteString -> Float -> [Float]
bitLinearPrim packed inDim outDim wscale xqBS ascale =
  let chunk = outDim `div` 4
      row j = BS.take inDim (BS.drop (j * inDim) packed)
  in [ bnDot 0 (row (r `mod` chunk)) xqBS (2 * (r `div` chunk)) inDim * wscale / ascale
     | r <- [0 .. outDim - 1] ]

-- | Dense float32 dot of two equal-length float32 byte vectors (kind 2). Used
-- for attention q·k scores and any full-precision dot.
floatDot :: BS.ByteString -> BS.ByteString -> Int -> Float
floatDot w a len = bnDot 2 w a 0 len

-- | LM-head logit for token @v@ via kind-1 bndot: the float32 hidden vector
-- @hF32@ dotted with the bf16 embedding row v. @embedB@ is the full embed_tokens
-- bytes, @hidden@ the model width.
lmLogit :: BS.ByteString -> BS.ByteString -> Int -> Int -> Float
lmLogit embedB hF32 hidden v =
  bnDot 1 (BS.take (hidden * 2) (BS.drop (v * hidden * 2) embedB)) hF32 0 hidden
