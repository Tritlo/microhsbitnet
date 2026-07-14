-- | M0 milestone driver: load the Bonsai GGUF, print header facts and spot
-- values (F32 and dequantized Q1_0). Compared line-for-line with
-- test/oracle_m0.py.
module M0(main) where

import qualified Data.ByteString as BS
import Gguf
import Loader (loadModel)

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

showKV :: Gguf -> String -> String
showKV g k = case lookupKV k g of
  Just (GInt v)     -> show v
  Just (GFloat v)   -> show v
  Just (GStr v)     -> v
  Just (GArr t n o) -> "arr(t=" ++ show t ++ ",n=" ++ show n ++ ")"
      where _ = o
  Nothing           -> "MISSING"

tensorLine :: Gguf -> String -> String
tensorLine g nm = case lookupTensor nm g of
  Just t  -> nm ++ " " ++ show (gtShape t) ++ " " ++ typeName (gtType t)
                ++ " off=" ++ show (gtOff t - gDataStart g)
  Nothing -> nm ++ " MISSING"

main :: IO ()
main = do
  g <- loadModel modelPath
  putStrLn ("bytes: " ++ show (BS.length (gBytes g)))
  putStrLn ("kvs: " ++ show (length (gKVs g)) ++ "  tensors: " ++ show (length (gTensors g)))
  putStrLn ("data_start: " ++ show (gDataStart g))

  mapM_ (\k -> putStrLn (k ++ " = " ++ showKV g k))
    [ "general.architecture", "qwen35.block_count", "qwen35.embedding_length"
    , "qwen35.feed_forward_length", "qwen35.attention.head_count"
    , "qwen35.attention.head_count_kv", "qwen35.attention.key_length"
    , "qwen35.full_attention_interval", "qwen35.rope.dimension_count"
    , "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon"
    , "qwen35.ssm.conv_kernel", "qwen35.ssm.group_count", "qwen35.ssm.inner_size"
    , "qwen35.ssm.state_size", "qwen35.ssm.time_step_rank"
    , "tokenizer.ggml.model", "tokenizer.ggml.pre"
    , "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"
    ]
  case lookupKV "qwen35.rope.dimension_sections" g of
    Just v  -> putStrLn ("rope_sections = " ++ show (intArray g v))
    Nothing -> putStrLn "rope_sections = MISSING"
  case lookupKV "tokenizer.ggml.tokens" g of
    Just (GArr _ n _) -> putStrLn ("n_tokens = " ++ show n)
    _                 -> putStrLn "n_tokens = MISSING"
  case lookupKV "tokenizer.ggml.merges" g of
    Just (GArr _ n _) -> putStrLn ("n_merges = " ++ show n)
    _                 -> putStrLn "n_merges = MISSING"

  mapM_ (putStrLn . tensorLine g)
    [ "token_embd.weight", "output.weight", "output_norm.weight"
    , "blk.0.attn_qkv.weight", "blk.0.ssm_conv1d.weight", "blk.3.attn_q.weight" ]

  -- F32 spot: output_norm first 5
  case lookupTensor "output_norm.weight" g of
    Just t  -> putStrLn ("output_norm[0..4]: "
                 ++ show [ f32At (gBytes g) (gtOff t + 4 * i) | i <- [0 .. 4] ])
    Nothing -> putStrLn "output_norm: MISSING"

  -- Q1_0 spots: embed rows 0 and 700 (row length 5120), attn_q row 0.
  case lookupTensor "token_embd.weight" g of
    Just t  -> do
      putStrLn ("embd[0,0..7]: "   ++ show (q1Row (gBytes g) (gtOff t) 5120 0 0 8))
      putStrLn ("embd[700,0..7]: " ++ show (q1Row (gBytes g) (gtOff t) 5120 700 0 8))
    Nothing -> putStrLn "embd: MISSING"
  case lookupTensor "blk.3.attn_q.weight" g of
    Just t  -> putStrLn ("attn_q[0,0..7]: " ++ show (q1Row (gBytes g) (gtOff t) 5120 0 0 8))
    Nothing -> putStrLn "attn_q: MISSING"
