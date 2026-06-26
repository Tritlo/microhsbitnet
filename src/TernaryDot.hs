-- | BitLinear forward via the @ternarydot@ runtime primitive (added to the
-- vendored MicroHs runtime, see vendor/microhs/src/runtime/eval.c). The inner
-- dot loop runs in C instead of as per-element combinator graph reduction.
--
-- The list-op 'BitLinear.dotTrits' stays the readable spec the primitive must
-- match; this is just the fast workhorse.
module TernaryDot
  ( ternaryDot
  , quantizeActBS
  , bitLinearPrim
  ) where

import qualified Data.ByteString as BS
import Data.Word (Word8)
import BitLinear (quantizeAct)

-- | Runtime primitive. @ternaryDot rowBytes xq shift@ computes
-- @sum_c (field(rowBytes[c], shift) - 1) * xq[c]@, where @rowBytes@ is one packed
-- BitLinear row (4 trits/byte), @xq@ the int8 activations, and @shift@ (0,2,4,6)
-- selects the 2-bit field for this output. Done in C; same value as
-- @dotTrits (rowTrits ...) xq@.
ternaryDot :: BS.ByteString -> BS.ByteString -> Int -> Int
ternaryDot = _primitive "ternarydot"

-- | Quantize activations to int8 and pack them as a ByteString (two's complement,
-- which the primitive reads as @int8_t@), returning the bytes and the scale.
quantizeActBS :: [Float] -> (BS.ByteString, Float)
quantizeActBS x =
  let (xq, sc) = quantizeAct x
  in (BS.pack (map fromIntegral xq), sc)

-- | A full BitLinear forward using the primitive: one C dot per output row.
bitLinearPrim :: BS.ByteString -> Int -> Int -> Float -> BS.ByteString -> Float -> [Float]
bitLinearPrim packed inDim outDim wscale xqBS ascale =
  let chunk = outDim `div` 4
      row j = BS.take inDim (BS.drop (j * inDim) packed)
  in [ fromIntegral (ternaryDot (row (r `mod` chunk)) xqBS (2 * (r `div` chunk)))
         * wscale / ascale
     | r <- [0 .. outDim - 1] ]
