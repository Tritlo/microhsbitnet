-- | The list-op BitLinear: a ternary-weight linear layer evaluated as list
-- operations rather than multiplications.
--
-- A 'Trit' names an operation on an activation — 'Pos' = identity, 'Neg' =
-- negate, 'Zero' = drop — so a dot product is a two-bucket fold: route each
-- int8 activation into a positive or negative accumulator (or drop it), with the
-- value being @pos - neg@. No negation or multiplication appears in the matmul;
-- the only multiplies are the two final scale factors per output.
module BitLinear
  ( Trit(..)
  , toTrit
  , rowTrits
  , dotTrits
  , dotRow
  , quantizeAct
  ) where

import qualified Data.ByteString as BS
import Data.Bits (shiftR, (.&.))
import Safetensors (tritAt)

-- | A ternary weight, reified as the operation it performs on an activation.
-- Kept as data (not fused into arithmetic) so a combinator/iota rendering of the
-- computation stays derivable later.
data Trit = Neg | Zero | Pos
  deriving (Eq, Show)

toTrit :: Int -> Trit
toTrit n | n < 0     = Neg
         | n > 0     = Pos
         | otherwise = Zero

-- | Unpack logical row @r@ (length @inDim@) of a packed BitLinear weight to trits.
rowTrits :: BS.ByteString -> Int -> Int -> Int -> [Trit]
rowTrits packed inDim outDim r =
  [ toTrit (tritAt packed inDim outDim r c) | c <- [0 .. inDim - 1] ]

-- | The list-op dot product: a strict two-bucket fold over (trits, int8 acts).
-- @Pos@ adds to the positive bucket, @Neg@ to the negative, @Zero@ drops; the
-- result is @pos - neg@ (one subtraction, no per-element negate or multiply).
dotTrits :: [Trit] -> [Int] -> Int
dotTrits = go 0 0
  where
    go p n (Pos  : ts) (x : xs) = let p' = p + x in p' `seq` go p' n ts xs
    go p n (Neg  : ts) (x : xs) = let n' = n + x in n' `seq` go p n' ts xs
    go p n (Zero : ts) (_ : xs) = go p n ts xs
    go p n _           _        = p - n

-- | Workhorse dot for logical row @r@: fold directly over the row's contiguous
-- packed bytes (@packed[j*inDim ..]@, j = r mod chunk), extracting 2-bit field
-- i = r div chunk, and walking the int8 activations in lockstep. Same result as
-- @dotTrits (rowTrits ...) xq@ but with no intermediate @[Trit]@ allocation.
-- @chunk@ is outDim\/4. Field value t encodes trit t-1: 0=neg, 1=drop, 2=pos.
-- This is the byte-level list-op the combinator showcase checks itself against.
dotRow :: BS.ByteString -> Int -> Int -> Int -> [Int] -> Int
dotRow packed inDim chunk r xq =
  let sh   = 2 * (r `div` chunk)
      base = (r `mod` chunk) * inDim
      go p n c (x : xs) =
        let t = fromIntegral ((BS.index packed (base + c) `shiftR` sh) .&. 3) :: Int
        in if t == 0 then let n' = n + x in n' `seq` go p n' (c + 1) xs
           else if t == 2 then let p' = p + x in p' `seq` go p' n (c + 1) xs
           else go p n (c + 1) xs
      go p n _ [] = p - n
  in go 0 0 0 xq

-- | Per-token int8 activation quantization (absmax): @scale = 127 / max|x|@
-- (clamped at 1e-5), @x_q = round(x * scale)@ clamped to [-128, 127]. Returns the
-- int8 activations and the scale.
quantizeAct :: [Float] -> ([Int], Float)
quantizeAct xs =
  let amax  = maximum (map abs xs)
      amax' = if amax < 1.0e-5 then 1.0e-5 else amax
      sc    = 127.0 / amax'
      q v   = clampInt (-128) 127 (round (v * sc))
  in (map q xs, sc)

clampInt :: Int -> Int -> Int -> Int
clampInt lo hi v | v < lo    = lo
                 | v > hi    = hi
                 | otherwise = v
