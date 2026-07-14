-- | The qwen35 hybrid decoder, token-by-token, on float32 byte buffers with
-- all hot math in C (cbits/b1dot.c). Layer kinds: full attention every 4th
-- layer ((i+1) `mod` 4 == 0), gated delta net otherwise (M4). Weights stay
-- packed Q1_0 slices into the resident GGUF; per-layer state is threaded
-- purely (KV cache for attention layers).
module Model
  ( Cfg(..)
  , cfgFromGguf
  , isAttnLayer
  , AttnL(..)
  , loadAttnL
  , KV
  , attnStep
  , ffnStep
  , GdnL(..)
  , loadGdnL
  , GdnState
  , gdnInit
  , gdnLayerStep
  ) where

import qualified Data.ByteString as BS
import Gguf
import Loader (slice)
import Dot

-- | Model shape, read from the GGUF KVs once.
data Cfg = Cfg
  { cLayers  :: Int
  , cHidden  :: Int
  , cFfn     :: Int
  , cHeads   :: Int
  , cKV      :: Int
  , cHeadDim :: Int
  , cRot     :: Int
  , cEps     :: Float
  , cTheta   :: Float
  , cVocab   :: Int
  , cConv    :: Int   -- ^ GDN conv kernel size (4)
  , cKHeads  :: Int   -- ^ GDN k/q heads (ssm.group_count, 16)
  , cVHeads  :: Int   -- ^ GDN v heads (ssm.time_step_rank, 48)
  , cSDim    :: Int   -- ^ GDN head dim (ssm.state_size, 128)
  , cInner   :: Int   -- ^ GDN value dim (ssm.inner_size, 6144)
  } deriving (Show)

cfgFromGguf :: Gguf -> Cfg
cfgFromGguf g = Cfg
  { cLayers  = kvInt "qwen35.block_count" 64 g
  , cHidden  = kvInt "qwen35.embedding_length" 5120 g
  , cFfn     = kvInt "qwen35.feed_forward_length" 17408 g
  , cHeads   = kvInt "qwen35.attention.head_count" 24 g
  , cKV      = kvInt "qwen35.attention.head_count_kv" 4 g
  , cHeadDim = kvInt "qwen35.attention.key_length" 256 g
  , cRot     = kvInt "qwen35.rope.dimension_count" 64 g
  , cEps     = kvFloat "qwen35.attention.layer_norm_rms_epsilon" 1.0e-6 g
  , cTheta   = kvFloat "qwen35.rope.freq_base" 1.0e7 g
  , cVocab   = case lookupTensor "token_embd.weight" g of
                 Just t  -> last (gtShape t)
                 Nothing -> 248320
  , cConv    = kvInt "qwen35.ssm.conv_kernel" 4 g
  , cKHeads  = kvInt "qwen35.ssm.group_count" 16 g
  , cVHeads  = kvInt "qwen35.ssm.time_step_rank" 48 g
  , cSDim    = kvInt "qwen35.ssm.state_size" 128 g
  , cInner   = kvInt "qwen35.ssm.inner_size" 6144 g
  }
  where
    kvFloat k d gg = case lookupKV k gg of
      Just (GFloat v) -> v
      _               -> d

-- | Full attention every 4th layer; the interval is fixed by the GGUF
-- (qwen35.full_attention_interval = 4).
isAttnLayer :: Int -> Bool
isAttnLayer i = (i + 1) `mod` 4 == 0

-- | One full-attention layer's tensors (Q1_0 packed weights + f32 norms),
-- as (file, offset) weight handles.
data AttnL = AttnL
  { aNorm, aQ, aQN, aK, aKN, aV, aO :: W
  , aPostNorm, aGate, aUp, aDown    :: W
  }

tens :: Gguf -> String -> W
tens g nm = case lookupTensor nm g of
  Just t  -> slice g t
  Nothing -> error ("missing tensor: " ++ nm)

loadAttnL :: Gguf -> Int -> AttnL
loadAttnL g i = AttnL
  { aNorm     = blk "attn_norm.weight"
  , aQ        = blk "attn_q.weight"
  , aQN       = blk "attn_q_norm.weight"
  , aK        = blk "attn_k.weight"
  , aKN       = blk "attn_k_norm.weight"
  , aV        = blk "attn_v.weight"
  , aO        = blk "attn_output.weight"
  , aPostNorm = blk "post_attention_norm.weight"
  , aGate     = blk "ffn_gate.weight"
  , aUp       = blk "ffn_up.weight"
  , aDown     = blk "ffn_down.weight"
  }
  where blk s = tens g ("blk." ++ show i ++ "." ++ s)

-- | An attention layer's cache: RoPE'd keys and values for all positions so
-- far, contiguous f32; 'Nothing' before any token.
type KV = Maybe (BS.ByteString, BS.ByteString)

-- | The shared FFN half of every block: x + down(silu(gate(norm x)) * up(norm x)).
ffnStep :: Cfg -> W -> W -> W -> W -> BS.ByteString -> BS.ByteString
ffnStep cfg postNorm wGate wUp wDown x =
  let hid = cHidden cfg
      ffn = cFfn cfg
      xn  = bnRmsNorm postNorm x hid (cEps cfg)
      gf  = b1Linear wGate xn hid ffn
      uf  = b1Linear wUp   xn hid ffn
      h   = bnSiluMul gf uf ffn
      d   = b1Linear wDown h ffn hid
  in bnAdd x d hid

-- | One gated-delta-net layer's tensors.
data GdnL = GdnL
  { gNorm, gQKV, gZ, gBeta, gAlpha        :: W
  , gConv, gDt, gA, gSNorm, gOut          :: W
  , gPostNorm, gGate, gUp, gDown          :: W
  }

loadGdnL :: Gguf -> Int -> GdnL
loadGdnL g i = GdnL
  { gNorm     = blk "attn_norm.weight"
  , gQKV      = blk "attn_qkv.weight"
  , gZ        = blk "attn_gate.weight"
  , gBeta     = blk "ssm_beta.weight"
  , gAlpha    = blk "ssm_alpha.weight"
  , gConv     = blk "ssm_conv1d.weight"
  , gDt       = blk "ssm_dt.bias"
  , gA        = blk "ssm_a"
  , gSNorm    = blk "ssm_norm.weight"
  , gOut      = blk "ssm_out.weight"
  , gPostNorm = blk "post_attention_norm.weight"
  , gGate     = blk "ffn_gate.weight"
  , gUp       = blk "ffn_up.weight"
  , gDown     = blk "ffn_down.weight"
  }
  where blk s = tens g ("blk." ++ show i ++ "." ++ s)

-- | A GDN layer's recurrent state: the conv ring (last conv-1 qkv vectors,
-- oldest first) and the delta-rule state (nV heads of d*d, transposed layout).
type GdnState = (BS.ByteString, BS.ByteString)

-- | Zero state (zero bytes are IEEE 0.0).
gdnInit :: Cfg -> GdnState
gdnInit cfg =
  let convDim = 2 * cKHeads cfg * cSDim cfg + cInner cfg
      ringB   = (cConv cfg - 1) * convDim * 4
      stateB  = cVHeads cfg * cSDim cfg * cSDim cfg * 4
  in (BS.replicate ringB 0, BS.replicate stateB 0)

-- | One GDN block: projections, causal conv + SiLU, per-head L2-normed q/k,
-- the gated delta rule (in C), gated RMSNorm against silu(z), out-projection,
-- then the FFN half. Returns the new residual stream and state.
gdnLayerStep :: Cfg -> GdnL -> GdnState -> BS.ByteString -> (BS.ByteString, GdnState)
gdnLayerStep cfg l (ring, st) x =
  let hid     = cHidden cfg
      eps     = cEps cfg
      nK      = cKHeads cfg
      nV      = cVHeads cfg
      d       = cSDim cfg
      keyDim  = nK * d
      valDim  = cInner cfg
      convDim = 2 * keyDim + valDim
      xn      = bnRmsNorm (gNorm l) x hid eps
      qkv     = b1Linear (gQKV l) xn hid convDim
      z       = b1Linear (gZ l) xn hid valDim
      betaP   = b1Linear (gBeta l) xn hid nV
      alphaP  = b1Linear (gAlpha l) xn hid nV
      conv    = gdnConvSilu ring qkv (gConv l) convDim
      q0      = BS.take (keyDim * 4) conv
      k0      = BS.take (keyDim * 4) (BS.drop (keyDim * 4) conv)
      v       = BS.drop (2 * keyDim * 4) conv
      qn      = gdnL2Heads q0 nK d eps
      kn      = gdnL2Heads k0 nK d eps
      (o, st') = gdnStep st qn kn v alphaP (gDt l) (gA l) betaP nV nK d
      n1      = bnRmsNormHeads (gSNorm l) o nV d eps
      gated   = bnSiluMul z n1 valDim
      aOut    = b1Linear (gOut l) gated valDim hid
      x1      = bnAdd x aOut hid
      out     = ffnStep cfg (gPostNorm l) (gGate l) (gUp l) (gDown l) x1
      ring'   = BS.append (BS.drop (convDim * 4) ring) qkv
  in (out, (ring', st'))

-- | One full-attention block at position @pos@: fused Q+gate projection,
-- per-head q/k RMSNorm, neox RoPE on the first 'cRot' dims, GQA attention,
-- sigmoid output gate, o-projection, then the FFN half. Returns the new
-- residual stream and the appended KV cache.
attnStep :: Cfg -> AttnL -> Int -> KV -> BS.ByteString -> (BS.ByteString, KV)
attnStep cfg l pos kv x =
  let hid  = cHidden cfg
      nh   = cHeads cfg
      nkv  = cKV cfg
      hd   = cHeadDim cfg
      eps  = cEps cfg
      xn   = bnRmsNorm (aNorm l) x hid eps
      qg   = b1Linear (aQ l) xn hid (nh * hd * 2)
      (q0, gate) = bnSplitQGate qg nh hd
      qn   = bnRmsNormHeads (aQN l) q0 nh hd eps
      k0   = b1Linear (aK l) xn hid (nkv * hd)
      kn   = bnRmsNormHeads (aKN l) k0 nkv hd eps
      v    = b1Linear (aV l) xn hid (nkv * hd)
      qr   = bnRope qn pos nh hd (cRot cfg) (cTheta cfg)
      kr   = bnRope kn pos nkv hd (cRot cfg) (cTheta cfg)
      (cK, cV) = case kv of
        Nothing       -> (kr, v)
        Just (ck, cv) -> (BS.append ck kr, BS.append cv v)
      nPos  = BS.length cK `div` (nkv * hd * 4)
      scale = 1 / sqrt (fromIntegral hd)
      att   = bnAttend qr cK cV nPos nh nkv hd scale
      gated = bnSigmoidMul gate att (nh * hd)
      o     = b1Linear (aO l) gated (nh * hd) hid
      x1    = bnAdd x o hid
      out   = ffnStep cfg (aPostNorm l) (aGate l) (aUp l) (aDown l) x1
  in (out, Just (cK, cV))
