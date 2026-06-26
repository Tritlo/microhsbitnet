-- | Shared forward pass: build the layer stack, run a token-id sequence through
-- it, and take the LM-head argmax. Used by M3 (single forward) and M4 (generation).
-- No KV cache — each call reprocesses the whole sequence (fine for short demos).
module Forward
  ( look
  , mkNorm
  , buildLayers
  , finalHidden
  , argmaxLogits
  , cVocab
  ) where

import qualified Data.ByteString as BS
import Data.List (foldl')
import Safetensors (Tensor(..), lookupTensor, bf16At)
import Loader (slice)
import Model
import Dot (lmLogit, packF32)

cVocab :: Int
cVocab = 128256

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

-- | Build all @n@ decoder layers.
buildLayers :: [Tensor] -> BS.ByteString -> Int -> [Layer]
buildLayers ts dat n = [ buildLayer ts dat i | i <- [0 .. n - 1] ]

-- Deeply force a sequence of vectors so the layer fold can't build a giant thunk.
forceSeq :: [[Float]] -> [[Float]]
forceSeq xss = outer xss `seq` xss
  where
    outer []          = ()
    outer (xs : rest) = inner xs `seq` outer rest
    inner []          = ()
    inner (x : xs)    = x `seq` inner xs

-- | Run token ids through embed + all layers + final norm, returning the final
-- (normed) hidden state of the last position.
finalHidden :: [Layer] -> [Float] -> BS.ByteString -> [Int] -> [Float]
finalHidden layers normW embedB ids =
  let xs0 = map (embedRow embedB) ids
      fh  = foldl' (\xs l -> forceSeq (decoderLayer l xs)) xs0 layers
  in rmsNorm normW (last fh)

-- | The argmax token over the tied LM head for a final hidden vector, with its
-- logit. Streams over the vocab so the logits list need not persist.
argmaxLogits :: BS.ByteString -> [Float] -> (Int, Float)
argmaxLogits embedB hLast =
  let hF32 = packF32 hLast
      go bi bv v
        | v >= cVocab = (bi, bv)
        | otherwise   = let l = lmLogit embedB hF32 cHidden v
                        in if l > bv then go v l (v + 1) else go bi bv (v + 1)
  in go 0 (-1.0 / 0.0) 0
