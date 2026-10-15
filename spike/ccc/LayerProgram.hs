module LayerProgram (Tensor (..), weight, operation, normalize, project, decoderLayer, layerStep) where

import Rotary (configureRotary)
import WebTensor (Engine, Tensor (..))
import qualified WebTensor

-- | Resolve one named tensor.
weight :: Engine -> String -> IO Tensor
weight = WebTensor.weight

-- | Encode an operation with up to three tensor arguments.
operation :: Engine -> String -> Tensor -> Tensor -> Tensor -> Int -> Double -> IO Tensor
operation = WebTensor.operation

-- | Apply a named layer normalization weight.
normalize :: Engine -> String -> Tensor -> IO Tensor
normalize engine name input = do
    norm <- weight engine name
    operation engine "rmsnorm" input norm (Tensor 0) 0 0

-- | Apply a packed ternary projection with its stored weight scale.
project :: Engine -> String -> Tensor -> IO Tensor
project engine name input = do
    matrix <- weight engine (name ++ ".weight")
    operation engine "linear" input matrix (Tensor 0) 0 0

-- | Run one decoder layer. The same program serves prefill and token decode.
decoderLayer :: Engine -> Int -> Int -> Tensor -> IO Tensor
decoderLayer engine layer position input = do
    let prefix = "model.layers." ++ show layer ++ "."
    normalized <- normalize engine (prefix ++ "input_layernorm.weight") input
    queryProjection <- project engine (prefix ++ "self_attn.q_proj") normalized
    keyProjection <- project engine (prefix ++ "self_attn.k_proj") normalized
    value <- project engine (prefix ++ "self_attn.v_proj") normalized
    query <- operation engine "rope" queryProjection (Tensor 0) (Tensor 0) position 0
    key <- operation engine "rope" keyProjection (Tensor 0) (Tensor 0) position 0
    keyCache <- weight engine ("cache." ++ show layer ++ ".k")
    valueCache <- weight engine ("cache." ++ show layer ++ ".v")
    keys <- operation engine "cache" key keyCache (Tensor 0) position 0
    values <- operation engine "cache" value valueCache (Tensor 0) position 0
    attended <- operation engine "attention" query keys values position (1 / sqrt 128)
    attentionNorm <- normalize engine (prefix ++ "self_attn.attn_sub_norm.weight") attended
    projected <- project engine (prefix ++ "self_attn.o_proj") attentionNorm
    residual <- operation engine "add" input projected (Tensor 0) 0 0
    postNorm <- normalize engine (prefix ++ "post_attention_layernorm.weight") residual
    gate <- project engine (prefix ++ "mlp.gate_proj") postNorm
    up <- project engine (prefix ++ "mlp.up_proj") postNorm
    gated <- operation engine "gated" gate up (Tensor 0) 0 0
    ffnNorm <- normalize engine (prefix ++ "mlp.ffn_sub_norm.weight") gated
    down <- project engine (prefix ++ "mlp.down_proj") ffnNorm
    operation engine "add" residual down (Tensor 0) 0 0

-- | Expose the decoder program to the asynchronous WebGPU host.
layerStep :: Engine -> Int -> Int -> Int -> IO Int
layerStep engine layer position input = do
    configureRotary engine position
    Tensor output <- decoderLayer engine layer position (Tensor input)
    pure output
