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

-- | One selected-sum over a group: a fold that *includes or excludes* each
-- activation by its weight bit — the whole thesis in four lines.
selSum :: Int -> [Int] -> Int
selSum _ []       = 0
selSum b (q : qs) = (if b .&. 1 == 1 then q else 0) + selSum (b `shiftR` 1) qs

-- | The pure 1-bit row dot: blocks of (fp16 scale d0, 16 sign bytes) against
-- quantized groups with per-group scales and precomputed group totals.
dotBits :: BS.ByteString -> Int -> [(Float, [Int], Int)] -> Float
dotBits bs rowOff groups = go rowOff groups
  where
    go _ [] = 0
    go o gs =
      let (blk, rest) = splitAt 4 gs
          d0 = f16At bs o
          inner = sum [ sc * fromIntegral (2 * sel - tot)
                      | (k, (sc, qs, tot)) <- zip [0 ..] blk
                      , let sel = selBytes (o + 2 + 4 * k) qs ]
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
      row r  = dotBits (gBytes g) (base + r * rowB) qgs

  -- correctness: rows 0..3 vs the C kernel (tolerance — the spec skips the
  -- fp16 round-trip of activation scales)
  let cOut = take 4 (unpackF32 (b1Linear (gBytes g, base) (packF32 xs) 5120 10240))
  putStrLn ("spec rows 0..3: " ++ show (map row [0 .. 3]))
  putStrLn ("C    rows 0..3: " ++ show cOut)

  t0 <- getTimeMilli
  let s32 = sum (map row [0 .. 31])
  s32 `seq` return ()
  t1 <- getTimeMilli
  let msRow    = fromIntegral (t1 - t0) / 32 :: Double
      sMatmul  = msRow * 10240 / 1000
      sToken   = msRow / 1000 / 5120 * 25.6e9
  putStrLn ("32 rows (5120 wide): " ++ show (t1 - t0) ++ " ms  (" ++ show msRow ++ " ms/row)")
  putStrLn ("extrapolated: qkv matmul (10240 rows) ~ " ++ show sMatmul ++ " s")
  putStrLn ("extrapolated: one full token (~25.6e9 weights) ~ " ++ show sToken ++ " s"
            ++ " (~" ++ show (sToken / 3600) ++ " h)")
