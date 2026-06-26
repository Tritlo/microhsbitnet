-- | M1 milestone driver: one BitLinear (layer-0 q_proj) evaluated as list ops.
-- Builds a deterministic input, int8-quantizes it, runs the two-bucket dot, and
-- prints the integer dots (bit-exact core) and rescaled outputs. Compared with
-- test/oracle_m1.py.
module M1(main) where

import qualified Data.ByteString as BS
import Safetensors (Tensor(..), lookupTensor, bf16At)
import Loader (loadModel, slice)
import TernaryDot (bitLinearPrim, quantizeActBS)

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
          (xqBS, ascale) = quantizeActBS x
          ys    = bitLinearPrim packed inDim outDim wscale xqBS ascale
      putStrLn ("inDim=" ++ show inDim ++ " outDim=" ++ show outDim)
      putStrLn ("wscale=" ++ show wscale)
      putStrLn ("ascale=" ++ show ascale)
      putStrLn ("y[0..7]=" ++ show (take 8 ys))
      putStrLn ("y sum=" ++ show (sum ys) ++ "  (expect ~ -1005.7)")
    _ -> putStrLn "q_proj weight/scale MISSING"
