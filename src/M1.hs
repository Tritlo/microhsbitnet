-- | M1 milestone driver: one BitLinear (layer-0 q_proj) evaluated as list ops.
-- Builds a deterministic input, int8-quantizes it, runs the two-bucket dot, and
-- prints the integer dots (bit-exact core) and rescaled outputs. Compared with
-- test/oracle_m1.py.
module M1(main) where

import qualified Data.ByteString as BS
import Safetensors (Tensor(..), lookupTensor, bf16At)
import Loader (loadModel, slice)
import BitLinear (rowTrits, dotTrits, quantizeAct)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

-- Deterministic input vector of length d. Integer-derived so it is reproduced
-- exactly in the oracle: x_c = ((c*73 + 13) mod 251 - 125) / 64.
mkInput :: Int -> [Float]
mkInput d = [ fromIntegral (((c * 73 + 13) `mod` 251) - 125) / 64.0 | c <- [0 .. d - 1] ]

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  case (lookupTensor "model.layers.0.self_attn.q_proj.weight" ts,
        lookupTensor "model.layers.0.self_attn.q_proj.weight_scale" ts) of
    (Just wt, Just st) -> do
      let packed = slice dat wt
          [packedRows, inDim] = tShape wt
          outDim = packedRows * 4
          wscale = bf16At (slice dat st) 0
          x = mkInput inDim
          (xq, ascale) = quantizeAct x
          idots = [ dotTrits (rowTrits packed inDim outDim r) xq | r <- [0 .. outDim - 1] ]
          ys    = map (\d -> fromIntegral d * wscale / ascale) idots
      putStrLn ("inDim=" ++ show inDim ++ " outDim=" ++ show outDim)
      putStrLn ("wscale=" ++ show wscale)
      putStrLn ("ascale=" ++ show ascale)
      putStrLn ("xq[0..15]=" ++ show (take 16 xq))
      putStrLn ("xq sum=" ++ show (sum xq) ++ " absSum=" ++ show (sum (map abs xq)))
      putStrLn ("idots[0..7]=" ++ show (take 8 idots))
      putStrLn ("idots sum=" ++ show (sum idots))
      putStrLn ("y[0..7]=" ++ show (take 8 ys))
    _ -> putStrLn "q_proj weight/scale MISSING"
