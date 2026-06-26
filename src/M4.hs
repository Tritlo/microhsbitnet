-- | M4 — the full self-contained demo: a prompt STRING in, generated TEXT out,
-- entirely in MicroHaskell (tokenizer, model, KV-cache generation — no Python).
module M4(main) where

import System.IO (hFlush, stdout)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Model (rmsNorm, embedRow)
import Forward (look, mkNorm, buildLayers, argmaxLogits, emptyCaches, stepLayers)
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
      normW  = mkNorm ts dat "model.norm.weight"
      layers = buildLayers ts dat 30
      prompt = encodeChat tok promptText
  putStrLn ("prompt: " ++ promptText)
  putStrLn ("prompt ids: " ++ unwords (map show prompt))

  -- Prefill the prompt into the KV caches; keep the last position's hidden.
  let prefill (caches, _) (pos, t) =
        let (h, caches') = stepLayers layers pos caches (embedRow embedB t)
        in (caches', h)
      (caches0, lastH0) = foldl' prefill (emptyCaches (length layers), []) (zip [0 ..] prompt)
      pos0 = length prompt

  -- Greedy generation with the cache: predict, then step the chosen token.
  let loop caches pos lastH k = do
        let (t, _) = argmaxLogits embedB (rmsNorm normW lastH)
        putStr (show t ++ " "); hFlush stdout
        if t `elem` eosIds || k <= 0
          then return [t]
          else do
            let (h, caches') = stepLayers layers pos caches (embedRow embedB t)
            rest <- loop caches' (pos + 1) h (k - 1)
            return (t : rest)
  putStr "generated ids: "
  gen <- loop caches0 pos0 lastH0 nGen
  putStrLn ""
  putStrLn "generated text:"
  BS.putStr (BS.pack (decode tok gen))
  putStrLn ""
