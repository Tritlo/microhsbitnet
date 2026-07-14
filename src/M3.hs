-- | M3 milestone driver: layer 3 (the first full-attention block) run
-- token-by-token over three synthetic positions, chaining each output into the
-- next input. Compared numerically with test/oracle_m3.py.
module M3(main) where

import Gguf (lookupTensor)
import Loader (loadModel)
import Dot (packF32, unpackF32)
import Model

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

x0 :: [Float]
x0 = [ fromIntegral ((i * 53) `mod` 199 - 99) / 99 | i <- [0 .. 5120 - 1 :: Int] ]

main :: IO ()
main = do
  g <- loadModel modelPath
  let cfg = cfgFromGguf g
      l3  = loadAttnL g 3
      step (x, kv, pos) = let (x', kv') = attnStep cfg l3 pos kv x in (x', kv', pos + 1)
      s0 = (packF32 x0, Nothing, 0)
      states = take 4 (iterate step s0)
  _ <- return (lookupTensor "output_norm.weight" g)  -- force load sanity
  mapM_ report (zip [0 :: Int ..] (drop 1 states))
  where
    report (i, (x, _, _)) = do
      let xs = unpackF32 x
      putStrLn ("pos" ++ show i ++ " out[0..5]: " ++ show (take 6 xs))
      putStrLn ("pos" ++ show i ++ " out[5114..5119]: " ++ show (drop 5114 xs))
      putStrLn ("pos" ++ show i ++ " sum: " ++ show (sum xs))
