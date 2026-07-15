-- | The pure-combinator spec of the 1-bit dot — selection, not multiplication —
-- and its benchmark. A weight bit is a Church boolean: set = include the
-- (quantized) activation in the selected sum, clear = exclude; a row dot is
-- per 128-block  d0 · Σₖ scₖ · (2·Σ_selected − Σ_all)  with the Σ_all
-- correction shared across every row of the matmul. No multiplications and no
-- per-weight negation in the fold — one conditional add per weight.
--
-- ./bin/b1spec times it on real blk.0.attn_qkv rows, checks a row against the
-- C kernel, and extrapolates to a matmul and a full token.
module B1Spec(main) where

import Data.Bits (shiftR, (.&.), testBit)
import qualified Data.ByteString as BS
import System.IO.TimeMilli (getTimeMilli)
import Gguf
import Loader (loadModel)
import Dot (b1Linear, packF32, unpackF32)

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

-- The benchmark activations (exact in f32).
xs :: [Float]
xs = [ fromIntegral ((i * 37) `mod` 101 - 50) / 50 | i <- [0 .. 5120 - 1 :: Int] ]

-- Quantize per 32 (absmax int8, like the kernel; scale kept in f32 here —
-- the spec demonstrates the fold, the kernel owns bit-exactness).
quantize :: [Float] -> [(Float, [Int])]
quantize [] = []
quantize v =
  let (g, rest) = splitAt 32 v
      amax = maximum (map abs g)
      d    = amax / 127
      q    = if d == 0 then map (const 0) g else map (\ x -> round (x / d)) g
  in (d, q) : quantize rest

-- | THE spec: a 1-bit weight row is a *filter*. Zip the bits with the
-- activations, keep the selected, sum. Each bit literally reduces as a Church
-- boolean — keep the cons or drop it.
sel :: [Bool] -> [Int] -> Int
sel ws = sum . map snd . filter fst . zip ws

-- The 32 bits of a group as booleans, LSB-first per byte.
groupBits :: BS.ByteString -> Int -> [Bool]
groupBits bs o = [ testBit (BS.index bs (o + i)) j | i <- [0 .. 3], j <- [0 .. 7] ]

-- | The workhorse spec: the same selection as one conditional add per weight,
-- no intermediate lists (the dotRow analogue; bit-identical to 'sel').
selSum :: Int -> [Int] -> Int
selSum _ []       = 0
selSum b (q : qs) = (if b .&. 1 == 1 then q else 0) + selSum (b `shiftR` 1) qs

-- | The filter-spec row dot: per 128-block, d0 · Σₖ scₖ·(2·sel − tot) with
-- 'sel' the literal zip/filter/sum.
dotFilter :: BS.ByteString -> Int -> [(Float, [Int], Int)] -> Float
dotFilter bs rowOff groups = go rowOff groups
  where
    go _ [] = 0
    go o gs =
      let (blk, rest) = splitAt 4 gs
          d0 = f16At bs o
          inner = sum [ sc * fromIntegral (2 * sel (groupBits bs (o + 2 + 4 * k)) qs - tot)
                      | (k, (sc, qs, tot)) <- zip [0 ..] blk ]
      in d0 * inner + go (o + 18) rest

-- | The workhorse row dot on the same math, folding bytes directly.
dotBits :: BS.ByteString -> Int -> [(Float, [Int], Int)] -> Float
dotBits bs rowOff groups = go rowOff groups
  where
    go _ [] = 0
    go o gs =
      let (blk, rest) = splitAt 4 gs
          d0 = f16At bs o
          inner = sum [ sc * fromIntegral (2 * sel' - tot)
                      | (k, (sc, qs, tot)) <- zip [0 ..] blk
                      , let sel' = selBytes (o + 2 + 4 * k) qs ]
      in d0 * inner + go (o + 18) rest
    selBytes o qs =
      let (a, r1) = splitAt 8 qs
          (b, r2) = splitAt 8 r1
          (c, r3) = splitAt 8 r2
          (d, _)  = splitAt 8 r3
      in selSum (byteAt o) a + selSum (byteAt (o + 1)) b
       + selSum (byteAt (o + 2)) c + selSum (byteAt (o + 3)) d
    byteAt o = fromIntegral (BS.index bs o)

main :: IO ()
main = do
  g <- loadModel modelPath
  let Just t = lookupTensor "blk.0.attn_qkv.weight" g
      base   = gtOff t
      rowB   = (5120 `div` 128) * 18
      qgs    = [ (d, q, sum q) | (d, q) <- quantize xs ]
      rowF r = dotFilter (gBytes g) (base + r * rowB) qgs
      row r  = dotBits (gBytes g) (base + r * rowB) qgs

  -- correctness: the two specs agree exactly; vs the C kernel with tolerance
  -- (the specs skip the fp16 round-trip of activation scales)
  let cOut = take 4 (unpackF32 (b1Linear (gBytes g, base) (packF32 xs) 5120 10240))
  putStrLn ("filter spec rows 0..3: " ++ show (map rowF [0 .. 3]))
  putStrLn ("fold spec   rows 0..3: " ++ show (map row [0 .. 3]))
  putStrLn ("C kernel    rows 0..3: " ++ show cOut)

  t0 <- getTimeMilli
  let sF = sum (map rowF [0 .. 31])
  sF `seq` return ()
  t1 <- getTimeMilli
  let s32 = sum (map row [32 .. 63])
  s32 `seq` return ()
  t2 <- getTimeMilli
  let bench nm ms =
        let msRow   = fromIntegral ms / 32 :: Double
            sToken  = msRow / 1000 / 5120 * 25.6e9
        in putStrLn (nm ++ ": " ++ show ms ++ " ms/32 rows  (" ++ show msRow
                     ++ " ms/row;  matmul ~" ++ show (msRow * 10240 / 1000)
                     ++ " s;  token ~" ++ show (sToken / 3600) ++ " h)")
  bench "filter spec (sum.map snd.filter fst.zip)" (t1 - t0)
  bench "fold spec   (one conditional add/weight)" (t2 - t1)
