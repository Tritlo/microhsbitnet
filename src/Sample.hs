-- | The sampling policy, in Haskell, over the C top-K candidates (C computes the
-- folds; Haskell decides). Temperature + top-p (nucleus over the returned top-K —
-- exact only when K covers the cutoff, so it is "top-p over top-K") + a small
-- xorshift RNG. @temp <= 0@ is greedy (the top candidate).
module Sample
  ( RNG
  , mkRNG
  , sampleTok
  ) where

import Data.Bits (xor, shiftL, shiftR, (.&.))

newtype RNG = RNG Word

-- | Seed the RNG (0 maps to a fixed nonzero constant so xorshift never sticks at 0).
mkRNG :: Int -> RNG
mkRNG s = RNG (let w = fromIntegral s in if w == 0 then 0x9e3779b97f4a7c15 else w)

-- xorshift64
step :: Word -> Word
step x0 =
  let x1 = x0 `xor` (x0 `shiftL` 13)
      x2 = x1 `xor` (x1 `shiftR` 7)
  in x2 `xor` (x2 `shiftL` 17)

-- a uniform Float in [0,1) from the low 24 bits
unit :: RNG -> (Float, RNG)
unit (RNG x) =
  let x' = step x
  in (fromIntegral (x' .&. 0xFFFFFF) / 16777216.0, RNG x')

-- | Sample a token id from descending @(token, logit)@ candidates with
-- temperature and top-p. @temp <= 0@ (or no candidates) is greedy.
sampleTok :: Float -> Float -> [(Int, Float)] -> RNG -> (Int, RNG)
sampleTok temp topP cands rng
  | temp <= 0.0 || null cands = (fst (head cands), rng)
  | otherwise =
      let ls        = [ l / temp | (_, l) <- cands ]
          m         = maximum ls
          es        = [ exp (x - m) | x <- ls ]
          tot       = sum es
          ps        = [ e / tot | e <- es ]               -- probs, descending
          nuc       = nucleus topP (zip (map fst cands) ps)
          ntot      = sum (map snd nuc)
          (r, rng') = unit rng
      in (walk (r * ntot) nuc, rng')

-- The nucleus: the shortest descending prefix whose cumulative prob reaches p
-- (the crossing item included; always at least one).
nucleus :: Float -> [(Int, Float)] -> [(Int, Float)]
nucleus p = go 0.0
  where
    go _   []                = []
    go acc (x@(_, q) : rest) =
      let acc' = acc + q
      in x : (if acc' >= p then [] else go acc' rest)

-- Walk the (token, prob) list subtracting until the target falls in a bucket.
walk :: Float -> [(Int, Float)] -> Int
walk _ [(t, _)]            = t
walk r ((t, q) : rest)
  | r < q     = t
  | otherwise = walk (r - q) rest
walk _ []                  = 0
