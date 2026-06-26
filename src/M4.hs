-- | M4 — greedy autoregressive generation with a KV cache (the "hello world").
-- Feeds a chat prompt (token ids; tokenization is offline via test/toktool.py —
-- no BPE in MicroHs) and generates tokens one at a time, printing each.
--
-- The cache makes it fast: the prompt is processed once (prefill), then each new
-- token only runs ONE position through the layers, reusing the cached keys/values
-- for attention — instead of reprocessing the whole sequence every step.
module M4(main) where

import System.IO (hFlush, stdout)
import Data.List (foldl')
import Loader (loadModel, slice)
import Model (rmsNorm, embedRow)
import Forward (look, mkNorm, buildLayers, argmaxLogits, emptyCaches, stepLayers)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

-- "User: Hello!<|eot_id|>Assistant: " (from test/toktool.py encode).
prompt :: [Int]
prompt = [1502, 25, 22691, 0, 128009, 72803, 25, 220]

eosIds :: [Int]
eosIds = [128001, 128009]

nGen :: Int
nGen = 16

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  let embedB = slice dat (look ts "model.embed_tokens.weight")
      normW  = mkNorm ts dat "model.norm.weight"
      layers = buildLayers ts dat 30
  putStrLn ("prompt ids: " ++ unwords (map show prompt))

  -- Prefill: thread each prompt token through the layers, building the caches;
  -- keep the last position's hidden (which predicts the first generated token).
  let prefill (caches, _) (pos, tok) =
        let (h, caches') = stepLayers layers pos caches (embedRow embedB tok)
        in (caches', h)
      (caches0, lastH0) = foldl' prefill (emptyCaches (length layers), []) (zip [0 ..] prompt)
      pos0 = length prompt

  -- Generation: predict from the running hidden, then step that token (one
  -- position) to get the next hidden.
  let loop caches pos lastH k = do
        let (tok, lg) = argmaxLogits embedB (rmsNorm normW lastH)
        putStrLn ("  -> " ++ show tok ++ "  (logit " ++ show lg ++ ")")
        hFlush stdout
        if tok `elem` eosIds || k <= 0
          then return [tok]
          else do
            let (h, caches') = stepLayers layers pos caches (embedRow embedB tok)
            rest <- loop caches' (pos + 1) h (k - 1)
            return (tok : rest)
  gen <- loop caches0 pos0 lastH0 nGen
  putStrLn ("generated ids: " ++ unwords (map show gen))
  putStrLn ("all ids: " ++ unwords (map show (prompt ++ gen)))
