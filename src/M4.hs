-- | M4 — the full self-contained demo: a prompt STRING in, generated TEXT out,
-- entirely in MicroHaskell (tokenizer, model, KV-cache generation — no Python).
module M4(main) where

import System.IO (hFlush, stdout)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Model (embedRow, cHidden, cEps)
import Forward (look, mkNorm, buildLayersC, argmaxLogits, emptyCachesC, stepLayersC)
import Dot (packF32, unpackF32, bnRmsNorm)
import Tokenizer (loadTokenizer, encodeChat, decode)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

tokPath :: String
tokPath = "models/bitnet-b1.58-2B-4T/tokenizer.json"

promptText :: String
promptText = "Hello!"

eosIds :: [Int]
eosIds = [128001, 128009]

nGen :: Int
nGen = 24

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  tok <- loadTokenizer tokPath
  let embedB = slice dat (look ts "model.embed_tokens.weight")
      normWB = packF32 (mkNorm ts dat "model.norm.weight")
      layers = buildLayersC ts dat 30
      prompt = encodeChat tok promptText
  putStrLn ("prompt: " ++ promptText)
  putStrLn ("prompt ids: " ++ unwords (map show prompt))

  -- Prefill the prompt into the KV caches; keep the last position's hidden (a
  -- float32 byte buffer). The fold's initial hidden is a placeholder, overwritten
  -- by the first token.
  let embedBS t = packF32 (embedRow embedB t)
      prefill (caches, _) (pos, t) =
        let (h, caches') = stepLayersC layers pos caches (embedBS t)
        in (caches', h)
      (caches0, lastH0) = foldl' prefill (emptyCachesC (length layers), packF32 [0.0]) (zip [0 ..] prompt)
      pos0 = length prompt

  -- Greedy generation with the cache: final-norm + argmax in C, then step the token.
  let loop caches pos lastH k = do
        let (t, _) = argmaxLogits embedB (unpackF32 (bnRmsNorm normWB lastH cHidden cEps))
        putStr (show t ++ " "); hFlush stdout
        if t `elem` eosIds || k <= 1   -- stop after exactly nGen generated tokens
          then return [t]
          else do
            let (h, caches') = stepLayersC layers pos caches (embedBS t)
            rest <- loop caches' (pos + 1) h (k - 1)
            return (t : rest)
  putStr "generated ids: "
  gen <- loop caches0 pos0 lastH0 nGen
  putStrLn ""
  putStrLn "generated text:"
  BS.putStr (BS.pack (decode tok gen))
  putStrLn ""
