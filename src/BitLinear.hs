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
  , quantizeAct
  , bitLinear
  , showcaseRow
  ) where

import qualified Data.ByteString as BS
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

-- | A full BitLinear forward: int8 activations @xq@ (with scale @ascale@) through
-- a packed weight of logical shape [outDim, inDim] with per-tensor @wscale@.
-- Each output is @dotTrits(row, xq) * wscale / ascale@. Rows are unpacked and
-- folded one at a time (strict, transient) so memory stays flat.
bitLinear :: BS.ByteString -> Int -> Int -> Float -> [Int] -> Float -> [Float]
bitLinear packed inDim outDim wscale xq ascale =
  [ fromIntegral (dotTrits (rowTrits packed inDim outDim r) xq) * wscale / ascale
  | r <- [0 .. outDim - 1] ]

-- | Showcase view of one row: the contributing operations paired with the
-- activations they act on (drops removed). Evaluating @sum of Pos - Neg@ over
-- this list reproduces 'dotTrits'. For rendering small slices.
showcaseRow :: [Trit] -> [Int] -> [(Trit, Int)]
showcaseRow ts xs = [ tx | tx@(t, _) <- zip ts xs, t /= Zero ]
