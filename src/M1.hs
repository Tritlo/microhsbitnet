-- | M1 milestone driver: one real B1Linear (blk.0.attn_qkv, 5120 -> 10240) on a
-- deterministic activation vector, plus an embedding-row dequant. Compared
-- numerically with test/oracle_m1.py.
module M1(main) where

import Gguf
import Loader (loadModel, slice)
import Dot (b1Linear, b1EmbedRow, packF32, unpackF32)

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

-- The deterministic test activations: exact in f32 (small integer ratios).
xs :: [Float]
xs = [ fromIntegral ((i * 37) `mod` 101 - 50) / 50 | i <- [0 .. 5120 - 1 :: Int] ]

main :: IO ()
main = do
  g <- loadModel modelPath
  let Just qkv = lookupTensor "blk.0.attn_qkv.weight" g
      Just emb = lookupTensor "token_embd.weight" g
      out = unpackF32 (b1Linear (slice g qkv) (packF32 xs) 5120 10240)
  putStrLn ("qkv out[0..7]: " ++ show (take 8 out))
  putStrLn ("qkv out[2048..2051]: " ++ show (take 4 (drop 2048 out)))
  putStrLn ("qkv out[10236..10239]: " ++ show (drop 10236 out))
  putStrLn ("qkv sum: " ++ show (sum out))
  let e42 = unpackF32 (b1EmbedRow (slice g emb) 42 5120)
  putStrLn ("embd42[0..7]: " ++ show (take 8 e42))
  putStrLn ("embd42 sum: " ++ show (sum e42))
