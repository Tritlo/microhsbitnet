module LayerProgram where

import Foreign.C.String (withCString)
import Rotary (configureRotary)
import WebFFI (encode, findWeight)

-- | An opaque GPU tensor handle. Tensor data stays outside the WASM heap.
newtype Tensor = Tensor Int

foreign export javascript layerStep :: Int -> Int -> Int -> IO Int

-- | Resolve one named tensor.
weight :: String -> IO Tensor
weight name = Tensor <$> withCString name findWeight

-- | Encode an operation with up to three tensor arguments.
operation :: String -> Tensor -> Tensor -> Tensor -> Int -> Double -> IO Tensor
operation name (Tensor first) (Tensor second) (Tensor third) parameter scale =
    Tensor <$> withCString name (\pointer -> encode pointer first second third parameter scale)

-- | Apply a named layer normalization weight.
normalize :: String -> Tensor -> IO Tensor
normalize name input = do
    norm <- weight name
    operation "rmsnorm" input norm (Tensor 0) 0 0

-- | Apply a packed ternary projection with its stored weight scale.
project :: String -> Tensor -> IO Tensor
project name input = do
    matrix <- weight (name ++ ".weight")
    operation "linear" input matrix (Tensor 0) 0 0

-- | Run one decoder layer. The same program serves prefill and token decode.
decoderLayer :: Int -> Int -> Tensor -> IO Tensor
decoderLayer layer position input = do
    let prefix = "model.layers." ++ show layer ++ "."
    normalized <- normalize (prefix ++ "input_layernorm.weight") input
    queryProjection <- project (prefix ++ "self_attn.q_proj") normalized
    keyProjection <- project (prefix ++ "self_attn.k_proj") normalized
    value <- project (prefix ++ "self_attn.v_proj") normalized
    query <- operation "rope" queryProjection (Tensor 0) (Tensor 0) position 0
    key <- operation "rope" keyProjection (Tensor 0) (Tensor 0) position 0
    keyCache <- weight ("cache." ++ show layer ++ ".k")
    valueCache <- weight ("cache." ++ show layer ++ ".v")
    keys <- operation "cache" key keyCache (Tensor 0) position 0
    values <- operation "cache" value valueCache (Tensor 0) position 0
    attended <- operation "attention" query keys values position (1 / sqrt 128)
    attentionNorm <- normalize (prefix ++ "self_attn.attn_sub_norm.weight") attended
    projected <- project (prefix ++ "self_attn.o_proj") attentionNorm
    residual <- operation "add" input projected (Tensor 0) 0 0
    postNorm <- normalize (prefix ++ "post_attention_layernorm.weight") residual
    gate <- project (prefix ++ "mlp.gate_proj") postNorm
    up <- project (prefix ++ "mlp.up_proj") postNorm
    gated <- operation "gated" gate up (Tensor 0) 0 0
    ffnNorm <- normalize (prefix ++ "mlp.ffn_sub_norm.weight") gated
    down <- project (prefix ++ "mlp.down_proj") ffnNorm
    operation "add" residual down (Tensor 0) 0 0

-- | Expose the decoder program to the asynchronous WebGPU host.
layerStep :: Int -> Int -> Int -> IO Int
layerStep layer position input = do
    configureRotary position
    Tensor output <- decoderLayer layer position (Tensor input)
    pure output
