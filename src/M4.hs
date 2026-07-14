-- | M4 milestone driver: layer 0 (the first gated-delta-net block) run
-- token-by-token over three chained synthetic positions. Compared numerically
-- with test/oracle_m4.py.
module M4(main) where

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
      l0  = loadGdnL g 0
      step (x, st) = gdnLayerStep cfg l0 st x
      s0 = (packF32 x0, gdnInit cfg)
      states = take 4 (iterate step s0)
  mapM_ report (zip [0 :: Int ..] (drop 1 states))
  where
    report (i, (x, _)) = do
      let xs = unpackF32 x
      putStrLn ("pos" ++ show i ++ " out[0..5]: " ++ show (take 6 xs))
      putStrLn ("pos" ++ show i ++ " out[5114..5119]: " ++ show (drop 5114 xs))
      putStrLn ("pos" ++ show i ++ " sum: " ++ show (sum xs))
