-- | M3 milestone driver: the full forward pass — embed, all 30 decoder layers,
-- final RMSNorm, and the tied LM head — producing next-token logits. Prints the
-- final hidden state (validates the stack against HF last_hidden_state) and the
-- argmax prediction (validates the head). Compared with test/oracle_m3.py.
module M3(main) where

import Loader (loadModel, slice)
import Model (cHidden)
import Dot (lmLogit)
import Forward (look, mkNorm, buildLayers, finalHidden, argmaxLogits)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

ids :: [Int]
ids = [128000]

l2 :: [Float] -> Float
l2 v = sqrt (sum (map (\x -> x * x) v))

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  let embedB = slice dat (look ts "model.embed_tokens.weight")
      normW  = mkNorm ts dat "model.norm.weight"
      layers = buildLayers ts dat 30
      hLast  = finalHidden layers normW embedB ids
  putStrLn ("seq=" ++ show (length ids) ++ " ids=" ++ show ids)
  putStrLn ("final hidden (last pos) [0..4]: " ++ show (take 5 hLast))
  putStrLn ("final hidden L2: " ++ show (l2 hLast))
  putStrLn ("logits at [0,100,1000,50000,128000]: "
             ++ show [ lmLogit embedB hLast cHidden v | v <- [0, 100, 1000, 50000, 128000] ])
  let (bi, bv) = argmaxLogits embedB hLast
  putStrLn ("argmax token: " ++ show bi ++ "  logit: " ++ show bv)
