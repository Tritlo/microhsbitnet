-- | M4 — the full self-contained demo: a prompt STRING in, generated TEXT out,
-- entirely in MicroHaskell (tokenizer, model, KV-cache generation — no Python).
--
-- Usage: @m4 [PROMPT [NGEN]]@ (defaults: "Hello!", 24). Prints phase timings.
module M4(main) where

import System.IO (hFlush, stdout)
import System.Environment (getArgs)
import System.IO.TimeMilli (getTimeMilli)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Model (embedRow, cHidden, cEps)
import Forward (look, mkNormBS, buildLayersC, forceLayersC, emptyCachesC, stepLayersC, cVocab)
import Dot (packF32, bnRmsNorm, argmaxBf16BS)
import Tokenizer (loadTokenizer, encodeChat, decode)

modelPath, tokPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"
tokPath   = "models/bitnet-b1.58-2B-4T/tokenizer.json"

eosIds :: [Int]
eosIds = [128001, 128009]

-- Time an IO action, printing "<label>: <ms> ms".
timed :: String -> IO a -> IO a
timed label act = do
  t0 <- getTimeMilli
  x  <- act
  t1 <- getTimeMilli
  putStrLn (label ++ ": " ++ show (t1 - t0) ++ " ms")
  return x

main :: IO ()
main = do
  args <- getArgs
  let promptText = case args of (p : _)     -> p;      _ -> "Hello!"
      nGen       = case args of (_ : n : _) -> read n; _ -> 24
  (ts, dat) <- timed "model load"     (loadModel modelPath)
  tok       <- timed "tokenizer load" (loadTokenizer tokPath)
  let embedB = slice dat (look ts "model.embed_tokens.weight")
      normWB = mkNormBS ts dat "model.norm.weight"
      layers = buildLayersC ts dat 30
      prompt = encodeChat tok promptText
  -- One-time weight setup (decode the norm weights bf16->f32 into buffers); timed
  -- on its own so it is not blamed on prefill.
  _ <- timed "layer build" (return $! forceLayersC layers)
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
  tp0 <- getTimeMilli
  -- force the (lazy) prefill: the last hidden depends on every layer of every
  -- prompt position via the KV cache, so this runs the whole prefill.
  _   <- BS.length lastH0 `seq` return ()
  tp1 <- getTimeMilli
  putStrLn ("prefill (" ++ show pos0 ++ " tokens): " ++ show (tp1 - tp0) ++ " ms")

  -- Greedy generation with the cache: final-norm + argmax in C (no round-trip),
  -- then step the chosen token.
  let loop caches pos lastH k = do
        let normed = bnRmsNorm normWB lastH cHidden cEps
            (t, _) = argmaxBf16BS embedB normed cHidden cVocab
        putStr (show t ++ " "); hFlush stdout
        if t `elem` eosIds || k <= 1   -- stop after exactly nGen generated tokens
          then return [t]
          else do
            let (h, caches') = stepLayersC layers pos caches (embedBS t)
            rest <- loop caches' (pos + 1) h (k - 1)
            return (t : rest)
  putStr "generated ids: "
  tg0 <- getTimeMilli
  gen <- loop caches0 pos0 lastH0 nGen
  tg1 <- getTimeMilli
  putStrLn ""
  let nTok = length gen
  putStrLn ("generation (" ++ show nTok ++ " tokens): " ++ show (tg1 - tg0) ++ " ms, "
             ++ show ((tg1 - tg0) `div` max 1 nTok) ++ " ms/token")

  -- Strip the stop token from the displayed text (kept only for the stop check).
  putStrLn "generated text:"
  BS.putStr (BS.pack (decode tok (filter (`notElem` eosIds) gen)))
  putStrLn ""
