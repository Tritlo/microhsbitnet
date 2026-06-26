-- | M2 milestone driver: run one decoder block (layer 0) on a fixed token
-- sequence and print first-3 values + L2 norm of every intermediate, at the
-- first and last position, for comparison with test/oracle_m2.py.
module M2(main) where

import qualified Data.ByteString as BS
import Safetensors (Tensor(..), lookupTensor, bf16At)
import Loader (loadModel, slice)
import Model

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

ids :: [Int]
ids = [128000, 791, 6864]

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

l2 :: [Float] -> Float
l2 v = sqrt (sum (map (\x -> x * x) v))

summ :: [Float] -> String
summ v = show (take 3 v) ++ " L2=" ++ show (l2 v)

main :: IO ()
main = do
  (ts, dat) <- loadModel modelPath
  let p          = "model.layers.0."
      embedB     = slice dat (look ts "model.embed_tokens.weight")
      inNorm     = mkNorm ts dat (p ++ "input_layernorm.weight")
      attnSubN   = mkNorm ts dat (p ++ "self_attn.attn_sub_norm.weight")
      postNorm   = mkNorm ts dat (p ++ "post_attention_layernorm.weight")
      ffnSubN    = mkNorm ts dat (p ++ "mlp.ffn_sub_norm.weight")
      lQ' = mkLin ts dat (p ++ "self_attn.q_proj")
      lK' = mkLin ts dat (p ++ "self_attn.k_proj")
      lV' = mkLin ts dat (p ++ "self_attn.v_proj")
      lO' = mkLin ts dat (p ++ "self_attn.o_proj")
      lG' = mkLin ts dat (p ++ "mlp.gate_proj")
      lU' = mkLin ts dat (p ++ "mlp.up_proj")
      lD' = mkLin ts dat (p ++ "mlp.down_proj")

      xs     = map (embedRow embedB) ids
      hn     = map (rmsNorm inNorm) xs
      q      = map (applyLin lQ') hn
      k      = map (applyLin lK') hn
      v      = map (applyLin lV') hn
      attn   = attention q k v
      attnN  = map (rmsNorm attnSubN) attn
      o      = map (applyLin lO') attnN
      xs1    = zipWith (zipWith (+)) xs o
      hn2    = map (rmsNorm postNorm) xs1
      g      = map (applyLin lG') hn2
      u      = map (applyLin lU') hn2
      gated  = zipWith (zipWith (\gg uu -> relu2 gg * uu)) g u
      gatedN = map (rmsNorm ffnSubN) gated
      d      = map (applyLin lD') gatedN
      out    = zipWith (zipWith (+)) xs1 d

      stages = [ ("embed", xs), ("hn", hn), ("q", q), ("k", k), ("v", v)
               , ("attn", attn), ("attnN", attnN), ("o", o), ("xs1", xs1)
               , ("hn2", hn2), ("gate", g), ("up", u), ("gated", gated)
               , ("gatedN", gatedN), ("down", d), ("out", out) ]
      positions = [0, length ids - 1]
  putStrLn ("seq=" ++ show (length ids) ++ " ids=" ++ show ids)
  mapM_ (\(lbl, st) -> mapM_ (\pos ->
            putStrLn (lbl ++ " p" ++ show pos ++ ": " ++ summ (st !! pos))) positions)
        stages
