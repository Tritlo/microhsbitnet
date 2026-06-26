-- | Interactive multi-turn chat REPL. The KV cache persists across turns: each
-- turn prefills only the new "User: ...<|eot_id|>Assistant: " segment, continuing
-- from the cache (the previous reply, including its stop token, is already in it).
--
-- Usage: @chat [SYSTEM_PROMPT]@. Type a line per turn; Ctrl-D or "/quit" to exit.
-- Sampling via env (default greedy): BN_TEMP (>0 enables), BN_TOPP, BN_TOPK, BN_SEED.
-- Runs in ~2 GB on the default heap; BNDOT_THREADS=N for the thread count.
module Chat(main) where

import System.IO (hFlush, stdout, hSetBuffering, BufferMode(LineBuffering))
import System.Environment (getArgs, lookupEnv)
import Data.List (foldl')
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Model (embedRow, cHidden, cEps)
import Forward (look, mkNormBS, buildLayersC, forceLayersC, emptyCachesC, stepLayersC, cVocab)
import Dot (packF32, bnRmsNorm, argmaxBf16BS, topkBf16)
import Tokenizer (loadTokenizer, Role(..), encodeChat, encodeChatMessages, decode)
import Sample (RNG, mkRNG, sampleTok)

modelPath, tokPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"
tokPath   = "models/bitnet-b1.58-2B-4T/tokenizer.json"

eosIds :: [Int]
eosIds = [128001, 128009]

maxGen :: Int
maxGen = 256

envWith :: (String -> a) -> a -> String -> IO a
envWith parse d name = do
  m <- lookupEnv name
  return (maybe d parse m)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  args <- getArgs
  let sysPrompt = case args of (s : _) -> Just s; _ -> Nothing
  temp <- envWith read 0.0  "BN_TEMP"
  topP <- envWith read 0.95 "BN_TOPP"
  topK <- envWith read 40   "BN_TOPK"
  seed <- envWith read 1234 "BN_SEED"

  putStr "loading model... "; hFlush stdout
  (ts, dat) <- loadModel modelPath
  tok       <- loadTokenizer tokPath
  let embedB  = slice dat (look ts "model.embed_tokens.weight")
      normWB  = mkNormBS ts dat "model.norm.weight"
      layers  = buildLayersC ts dat 30
      nLayers = length layers
      embedBS t = packF32 (embedRow embedB t)
  _ <- return $! forceLayersC layers   -- one-time weight setup
  putStrLn ("ready (" ++ (if temp <= 0.0 then "greedy" else "sampling temp=" ++ show temp
              ++ " topP=" ++ show topP ++ " topK=" ++ show topK) ++ "). Ctrl-D or /quit to exit.")

  -- Pick the next token: greedy argmax, or sample over the C top-K candidates.
  let pickTok rng normed
        | temp <= 0.0 = (fst (argmaxBf16BS embedB normed cHidden cVocab), rng)
        | otherwise   = sampleTok temp topP (topkBf16 embedB normed cHidden cVocab topK) rng

  -- Prefill token ids, threading caches/position; returns the last hidden buffer.
  let prefill caches pos toks =
        let stepT (cs, _) (p, t) = let (h, cs') = stepLayersC layers p cs (embedBS t) in (cs', h)
            (caches', lastH) = foldl' stepT (caches, packF32 [0.0]) (zip [pos ..] toks)
        in (caches', pos + length toks, lastH)

  -- Greedy/sampled reply, streaming each token's text; steps every token (incl.
  -- the stop token) into the cache so the next turn continues cleanly.
  let gen caches pos lastH rng = go caches pos lastH rng maxGen
        where
          go cs p _ rg 0 = return (cs, p, rg)
          go cs p h rg k = do
            let normed   = bnRmsNorm normWB h cHidden cEps
                (t, rg') = pickTok rg normed
                isEos    = t `elem` eosIds
            if isEos then return () else BS.putStr (BS.pack (decode tok [t]))
            hFlush stdout
            let (h', cs') = stepLayersC layers p cs (embedBS t)
            if isEos then return (cs', p + 1, rg') else go cs' (p + 1) h' rg' (k - 1)

  -- One turn per input line; the first turn carries the optional system prompt.
  ls <- fmap lines getContents
  let turn []            _      _   _    _     = putStrLn ""
      turn (line : rest) caches pos rng first
        | line == "/quit" = putStrLn "bye."
        | otherwise = do
            let toks
                  | first, Just s <- sysPrompt = encodeChatMessages tok [(System, s), (User, line)]
                  | otherwise                  = encodeChat tok line
                (caches', pos', lastH) = prefill caches pos toks
            putStr "assistant: "; hFlush stdout
            (caches'', pos'', rng') <- gen caches' pos' lastH rng
            putStr "\n\nuser: "; hFlush stdout
            turn rest caches'' pos'' rng' False
  putStr "user: "; hFlush stdout
  turn ls (emptyCachesC nLayers) 0 (mkRNG seed) True
