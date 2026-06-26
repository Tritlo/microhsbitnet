-- | M3 milestone driver: the full forward pass — embed, all 30 decoder layers,
-- final RMSNorm, and the tied LM head — producing next-token logits. Prints the
-- final hidden state (validates the stack against HF last_hidden_state) and the
-- argmax prediction (validates the head). Compared with test/oracle_m3.py.
module M3(main) where

import qualified Data.ByteString as BS
import Data.List (foldl')
import Safetensors (Tensor(..), lookupTensor, bf16At)
import Loader (loadModel, slice)
import Model
import Dot (lmLogit, packF32)

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

ids :: [Int]
ids = [128000]

nLayers :: Int
nLayers = 30

vocab :: Int
vocab = 128256

look :: [Tensor] -> String -> Tensor
look ts name = case lookupTensor name ts of
  Just t  -> t
  Nothing -> error ("missing tensor: " ++ name)

mkLin :: [Tensor] -> BS.ByteString -> String -> Lin
mkLin ts dat name =
  let wt = look ts (name ++ ".weight")
      st = look ts (name ++ ".weight_scale")
      [packedRows, inD] = tShape wt
  in Lin (slice dat wt) inD (packedRows * 4) (bf16At (slice dat st) 0)

mkNorm :: [Tensor] -> BS.ByteString -> String -> [Float]
mkNorm ts dat name =
  let t = look ts name
      d = head (tShape t)
      b = slice dat t
  in [ bf16At b (2 * c) | c <- [0 .. d - 1] ]

buildLayer :: [Tensor] -> BS.ByteString -> Int -> Layer
buildLayer ts dat i =
  let p = "model.layers." ++ show i ++ "."
  in Layer
       (mkNorm ts dat (p ++ "input_layernorm.weight"))
       (mkLin ts dat (p ++ "self_attn.q_proj"))
       (mkLin ts dat (p ++ "self_attn.k_proj"))
       (mkLin ts dat (p ++ "self_attn.v_proj"))
       (mkNorm ts dat (p ++ "self_attn.attn_sub_norm.weight"))
       (mkLin ts dat (p ++ "self_attn.o_proj"))
       (mkNorm ts dat (p ++ "post_attention_layernorm.weight"))
       (mkLin ts dat (p ++ "mlp.gate_proj"))
       (mkLin ts dat (p ++ "mlp.up_proj"))
       (mkNorm ts dat (p ++ "mlp.ffn_sub_norm.weight"))
       (mkLin ts dat (p ++ "mlp.down_proj"))

-- Deeply force a sequence of activation vectors (every element to a number), so
-- the 30-layer fold doesn't build a giant lazy thunk that overflows the stack.
-- Tail-recursive so the forcing itself uses bounded stack.
forceSeq :: [[Float]] -> [[Float]]
forceSeq xss = outer xss `seq` xss
  where
    outer []         = ()
    outer (xs : rest) = inner xs `seq` outer rest
    inner []         = ()
    inner (x : xs)   = x `seq` inner xs

-- Run all layers, fully evaluating each layer's output before the next.
runLayers :: [Layer] -> [[Float]] -> [[Float]]
runLayers ls xs0 = foldl' (\xs l -> forceSeq (decoderLayer l xs)) xs0 ls

l2 :: [Float] -> Float
l2 v = sqrt (sum (map (\x -> x * x) v))

-- argmax over the logits, computed streaming so the whole list need not persist.
argmax :: [Float] -> (Int, Float)
argmax (x : xs) = go 0 0 x xs
  where
    go bi _ bv [] = (bi, bv)
    go bi i bv (y : ys) | y > bv    = go (i + 1) (i + 1) y ys
                        | otherwise = go bi (i + 1) bv ys
argmax [] = error "argmax: empty"

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  let embedB  = slice dat (look ts "model.embed_tokens.weight")
      xs0     = map (embedRow embedB) ids
      layers  = [ buildLayer ts dat i | i <- [0 .. nLayers - 1] ]
      finalH  = runLayers layers xs0
      normW   = mkNorm ts dat "model.norm.weight"
      hLast   = rmsNorm normW (last finalH)
  putStrLn ("seq=" ++ show (length ids) ++ " ids=" ++ show ids)
  putStrLn ("final hidden (last pos) [0..4]: " ++ show (take 5 hLast))
  putStrLn ("final hidden L2: " ++ show (l2 hLast))
  -- probe logits at a few token ids for cross-check, then argmax (the prediction).
  -- LM head via the kind-1 bndot primitive over the bf16 (tied) embedding rows.
  let hF32 = packF32 hLast
      logitAt v = lmLogit embedB hF32 cHidden v
  putStrLn ("logits at [0,100,1000,50000,128000]: "
             ++ show [ logitAt v | v <- [0, 100, 1000, 50000, 128000] ])
  let logits = [ logitAt v | v <- [0 .. vocab - 1] ]
      (bi, bv) = argmax logits
  putStrLn ("argmax token: " ++ show bi ++ "  logit: " ++ show bv)
