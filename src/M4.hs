-- | M4 — greedy autoregressive generation (the "hello world"). Feeds a chat
-- prompt (token ids; tokenization is done offline by test/toktool.py — no BPE in
-- MicroHs) and generates tokens one at a time, printing each. No KV cache, so each
-- step reprocesses the whole sequence — fine for a short demo.
module M4(main) where

import System.IO (hFlush, stdout)
import Loader (loadModel, slice)
import Forward (look, mkNorm, buildLayers, finalHidden, argmaxLogits)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

-- "User: Hello!<|eot_id|>Assistant: " (from test/toktool.py encode).
prompt :: [Int]
prompt = [1502, 25, 22691, 0, 128009, 72803, 25, 220]

eosIds :: [Int]
eosIds = [128001, 128009]

nGen :: Int
nGen = 12

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  let embedB = slice dat (look ts "model.embed_tokens.weight")
      normW  = mkNorm ts dat "model.norm.weight"
      layers = buildLayers ts dat 30
  putStrLn ("prompt ids: " ++ unwords (map show prompt))
  let loop ids 0 = return ids
      loop ids k = do
        let (t, lg) = argmaxLogits embedB (finalHidden layers normW embedB ids)
        putStrLn ("  -> " ++ show t ++ "  (logit " ++ show lg ++ ")")
        hFlush stdout
        if t `elem` eosIds
          then return (ids ++ [t])
          else loop (ids ++ [t]) (k - 1)
  final <- loop prompt nGen
  putStrLn ("generated ids: " ++ unwords (map show (drop (length prompt) final)))
  putStrLn ("all ids: " ++ unwords (map show final))
