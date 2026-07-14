-- | Single-turn completion / chat driver for 1-bit Bonsai 27B.
--
-- Usage: @chat PROMPT [NGEN]@ (default 64 tokens).
-- Env: @BN_RAW=1@ completes the raw prompt with no chat template (the greedy
-- parity mode against the fork's llama-completion); otherwise the prompt is
-- wrapped in ChatML with the model's default thinking prefix. Sampling via
-- @BN_TEMP@ (>0 enables; default greedy), @BN_TOPP@, @BN_TOPK@, @BN_SEED@.
-- @BNDOT_THREADS=N@ sets the kernel thread count.
module Chat(main) where

import System.IO (hFlush, stdout)
import System.Environment (getArgs, lookupEnv)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Gguf (kvInt)
import Loader (loadModel, slice)
import Model (cfgFromGguf, Cfg(..))
import Forward
import Dot (packF32, bnRmsNorm, b1Argmax, b1TopK, b1EmbedRow)
import Tokenizer
import Sample (mkRNG, sampleTok)

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

envWith :: (String -> a) -> a -> String -> IO a
envWith parse d name = do
  m <- lookupEnv name
  return (maybe d parse m)

main :: IO ()
main = do
  args <- getArgs
  let (prompt, nGen) = case args of
        (p : n : _) -> (p, read n)
        [p]         -> (p, 64)
        []          -> ("The capital of Iceland is", 64)
  raw  <- envWith (== "1") False "BN_RAW"
  temp <- envWith read 0.0  "BN_TEMP"
  topP <- envWith read 0.95 "BN_TOPP"
  topK <- envWith read 20   "BN_TOPK"
  seed <- envWith read 1234 "BN_SEED"

  putStr "loading model... " >> hFlush stdout
  g   <- loadModel modelPath
  tok <- loadTokenizer g
  let cfg     = cfgFromGguf g
      layers  = buildLayers g cfg
      embedB  = slice g (look g "token_embd.weight")
      outWB   = slice g (look g "output.weight")
      normWB  = slice g (look g "output_norm.weight")
      eosId   = kvInt "tokenizer.ggml.eos_token_id" 248046 g
      hid     = cHidden cfg
      spec s  = case tokenId s of
                  Just i  -> [i]
                  Nothing -> encodeText tok s
      ids | raw       = encodeText tok prompt
          | otherwise = spec "<|im_start|>" ++ encodeText tok ("user\n" ++ prompt)
                     ++ spec "<|im_end|>" ++ encodeText tok "\n"
                     ++ spec "<|im_start|>" ++ encodeText tok "assistant\n"
                     ++ spec "<think>" ++ encodeText tok "\n"
      embedBS t = b1EmbedRow embedB t hid
  putStrLn ("prompt ids: " ++ show ids)

  -- Prefill, threading states and position; keep the last hidden buffer.
  let stepT (sts, _) (p, t) = let (h, sts') = stepModel cfg layers p sts (embedBS t) in (sts', h)
      (sts1, h1) = foldl' stepT (initStates cfg layers, packF32 [0]) (zip [0 ..] ids)

  let pickTok rng normed
        | temp <= 0.0 = (fst (b1Argmax outWB normed hid (cVocab cfg)), rng)
        | otherwise   = sampleTok temp topP (b1TopK outWB normed hid (cVocab cfg) topK) rng

  -- Generate, streaming each token's bytes; stop on EOS.
  let gen _   _ _  _  0 = putStrLn "" >> return ()
      gen sts p h rng k = do
        let normed    = bnRmsNorm normWB h hid (cEps cfg)
            (t, rng') = pickTok rng normed
        if t == eosId
          then putStrLn "\n[eos]"
          else do
            putBytes (decodeBytes tok [t])
            let (h', sts') = stepModel cfg layers p sts (embedBS t)
            gen sts' (p + 1) h' rng' (k - 1)

  gen sts1 (length ids) h1 (mkRNG seed) nGen
