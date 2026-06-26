-- | M0 milestone driver: verify bf16 decode + ternary unpack against the loader,
-- using the resident data section. Compared bit-for-bit with test/oracle_m0.py.
module M0(main) where

import qualified Data.ByteString as BS
import Safetensors
import Loader (loadModel, slice)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  putStrLn ("tensors: " ++ show (length ts))
  putStrLn ("resident bytes: " ++ show (BS.length dat))

  -- (a) bf16: embed_tokens row 0, first 5 values
  case lookupTensor "model.embed_tokens.weight" ts of
    Just t  -> putStrLn ("embed[0,0..4]: "
                          ++ show [ bf16At (slice dat t) (2 * c) | c <- [0 .. 4] ])
    Nothing -> putStrLn "embed: MISSING"

  -- (b) bf16: a weight scale (scalar)
  case lookupTensor "model.layers.0.self_attn.q_proj.weight_scale" ts of
    Just t  -> putStrLn ("q_proj.weight_scale: " ++ show (bf16At (slice dat t) 0))
    Nothing -> putStrLn "scale: MISSING"

  -- (c) U8 unpack: q_proj weight; first 16 trits of rows 0 and 700 (700 exercises
  --     the out-dim stride), and that row 0 is fully ternary.
  case lookupTensor "model.layers.0.self_attn.q_proj.weight" ts of
    Just t  -> do
      let b = slice dat t
          [packedRows, inDim] = tShape t
          outDim = packedRows * 4
          row0 = unpackRow b inDim outDim 0
      putStrLn ("q_proj trits row0[0..15]: " ++ show (take 16 row0))
      putStrLn ("q_proj row0 all ternary: " ++ show (all (\v -> v >= -1 && v <= 1) row0)
                 ++ "  (len " ++ show (length row0) ++ ")")
      putStrLn ("q_proj trits row700[0..15]: " ++ show (take 16 (unpackRow b inDim outDim 700)))
    Nothing -> putStrLn "q_proj: MISSING"
