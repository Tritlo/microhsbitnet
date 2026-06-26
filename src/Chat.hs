-- | Interactive multi-turn chat REPL. The KV cache persists across turns: each
-- turn prefills only the new "User: ...<|eot_id|>Assistant: " segment, continuing
-- from the cache (the previous reply, including its stop token, is already in it).
--
-- Usage: @chat [SYSTEM_PROMPT]@. Type a line per turn; Ctrl-D or "/quit" to exit.
-- Runs in ~2 GB on the default heap; BNDOT_THREADS=N for the thread count.
module Chat(main) where

import System.IO (hFlush, stdout, hSetBuffering, BufferMode(LineBuffering))
import System.Environment (getArgs)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Model (embedRow, cHidden, cEps)
import Forward (look, mkNormBS, buildLayersC, forceLayersC, emptyCachesC, stepLayersC, cVocab)
import Dot (packF32, bnRmsNorm, argmaxBf16BS)
import Tokenizer (loadTokenizer, Role(..), encodeChat, encodeChatMessages, decode)

modelPath, tokPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"
tokPath   = "models/bitnet-b1.58-2B-4T/tokenizer.json"

eosIds :: [Int]
eosIds = [128001, 128009]

maxGen :: Int
maxGen = 256

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  args <- getArgs
  let sysPrompt = case args of (s : _) -> Just s; _ -> Nothing
  putStr "loading model... "; hFlush stdout
  (ts, dat) <- loadModel modelPath
  tok       <- loadTokenizer tokPath
  let embedB  = slice dat (look ts "model.embed_tokens.weight")
      normWB  = mkNormBS ts dat "model.norm.weight"
      layers  = buildLayersC ts dat 30
      nLayers = length layers
      embedBS t = packF32 (embedRow embedB t)
  _ <- return $! forceLayersC layers   -- one-time weight setup
  putStrLn "ready. (Ctrl-D or /quit to exit)"

  -- Prefill a list of token ids, threading the caches/position; returns the last
  -- position's hidden buffer.
  let prefill caches pos toks =
        let step (cs, _) (p, t) = let (h, cs') = stepLayersC layers p cs (embedBS t) in (cs', h)
            (caches', lastH) = foldl' step (caches, packF32 [0.0]) (zip [pos ..] toks)
        in (caches', pos + length toks, lastH)

  -- Greedy reply, streaming each token's text; steps every token (incl. the stop
  -- token) into the cache so the next turn continues cleanly.
  let gen caches pos lastH = go caches pos lastH maxGen
        where
          go cs p _ 0 = return (cs, p)
          go cs p h k = do
            let normed = bnRmsNorm normWB h cHidden cEps
                (t, _) = argmaxBf16BS embedB normed cHidden cVocab
                isEos  = t `elem` eosIds
            if isEos then return () else BS.putStr (BS.pack (decode tok [t]))
            hFlush stdout
            let (h', cs') = stepLayersC layers p cs (embedBS t)
            if isEos then return (cs', p + 1) else go cs' (p + 1) h' (k - 1)

  -- One turn per input line; the first turn carries the optional system prompt.
  ls <- fmap lines getContents
  let turn []           _      _   _     = putStrLn ""
      turn (line : rest) caches pos first
        | line == "/quit" = putStrLn "bye."
        | otherwise = do
            let toks
                  | first, Just s <- sysPrompt = encodeChatMessages tok [(System, s), (User, line)]
                  | otherwise                  = encodeChat tok line
                (caches', pos', lastH) = prefill caches pos toks
            putStr "assistant: "; hFlush stdout
            (caches'', pos'') <- gen caches' pos' lastH
            putStr "\n\nuser: "; hFlush stdout
            turn rest caches'' pos'' False
  putStr "user: "; hFlush stdout
  turn ls (emptyCachesC nLayers) 0 True
