-- | BitNet decoder-block components: RMSNorm, RoPE, GQA attention, squared-ReLU
-- gated MLP, and the full layer. Activations are plain @[Float]@ vectors; a
-- sequence is @[[Float]]@. Each 'Lin' is a BitLinear that int8-quantizes its own
-- input before the list-op matmul.
module Model
  ( Lin(..)
  , applyLin
  , rmsNorm
  , relu2
  , applyRope
  , attention
  , embedRow
  , Layer(..)
  , decoderLayer
  , KVCache
  , decoderStep
  , LayerC(..)
  , KVC
  , applyLinC
  , decoderStepC
  , cHidden, cHeads, cKV, cHeadDim, cTheta, cEps
  ) where

import qualified Data.ByteString as BS
import Safetensors (bf16At)
import Dot (quantizeActBS, bitLinearPrim, packF32, floatDot,
            bnRmsNorm, bnRope, bnAttend, bnAdd, bnSqReluMul, bnBitLinearF)

-- Model dimensions (BitNet b1.58 2B4T).
cHidden, cHeads, cKV, cHeadDim :: Int
cHidden  = 2560
cHeads   = 20
cKV      = 5
cHeadDim = 128

cTheta, cEps :: Float
cTheta = 500000.0
cEps   = 1.0e-5

-- | A BitLinear layer: packed ternary weight, logical [outDim, inDim], scale.
data Lin = Lin
  { linPacked :: BS.ByteString
  , linIn     :: Int
  , linOut    :: Int
  , linScale  :: Float
  }

-- | Apply a BitLinear: quantize the input activations to int8, then the list-op
-- matmul and rescale.
applyLin :: Lin -> [Float] -> [Float]
applyLin (Lin p inD outD sc) x =
  let (xqBS, ascale) = quantizeActBS x
  in bitLinearPrim p inD outD sc xqBS ascale

-- | RMSNorm in float32: @weight * x / sqrt(mean(x^2) + eps)@. Count and
-- sum-of-squares are taken in a single strict pass.
rmsNorm :: [Float] -> [Float] -> [Float]
rmsNorm weight x =
  let go cnt acc []       = (cnt, acc)
      go cnt acc (v : vs) = let a = acc + v * v; c = cnt + 1 in a `seq` c `seq` go c a vs
      (n, ss) = go (0 :: Int) 0.0 x
      s       = 1.0 / sqrt (ss / fromIntegral n + cEps)
  in zipWith (\w v -> w * (v * s)) weight x

-- | Squared ReLU: @max(0, g)^2@.
relu2 :: Float -> Float
relu2 g = let r = if g > 0.0 then g else 0.0 in r * r

-- The RoPE inverse frequencies, inv_freq[k] = theta^(-2k/headDim). Independent of
-- position and head, so computed once (avoids the per-element exponentiation).
invFreq :: [Float]
invFreq = [ cTheta ** (negate (2.0 * fromIntegral k) / fromIntegral cHeadDim)
          | k <- [0 .. cHeadDim `div` 2 - 1] ]

-- | Apply rotary position embedding to one head vector (length 'cHeadDim') at
-- position @pos@. rotate_half = [-x2, x1]; cos/sin are bound once per pair.
applyRope :: Int -> [Float] -> [Float]
applyRope pos hv =
  let half = cHeadDim `div` 2
      h1   = take half hv
      h2   = drop half hv
      rot  = [ let a = fromIntegral pos * f; c = cos a; s = sin a
               in (x1 * c - x2 * s, x2 * c + x1 * s)
             | (f, x1, x2) <- zip3 invFreq h1 h2 ]
  in map fst rot ++ map snd rot

softmax :: [Float] -> [Float]
softmax xs =
  let m  = maximum xs
      es = map (\x -> exp (x - m)) xs
      s  = sum es
  in map (/ s) es

sliceHead :: [Float] -> Int -> [Float]
sliceHead vec h = take cHeadDim (drop (h * cHeadDim) vec)

-- | GQA causal self-attention. q has 'cHeads' heads, k\/v have 'cKV' heads (query
-- head h uses kv head h `div` (cHeads\/cKV)); RoPE is applied to q and k. Returns
-- the per-position attention output (length 'cHidden').
attention :: [[Float]] -> [[Float]] -> [[Float]] -> [[Float]]
attention qSeq kSeq vSeq =
  let seqLen  = length qSeq
      nGroups = cHeads `div` cKV
      scale   = 1.0 / sqrt (fromIntegral cHeadDim)
      -- RoPE'd q/k heads packed to float32 bytes so the q·k scores use the
      -- kind-2 (float32) bndot primitive; v stays a list for the weighted sum.
      qH = [ [ packF32 (applyRope p (sliceHead (qSeq !! p) h))  | h  <- [0 .. cHeads - 1] ] | p <- [0 .. seqLen - 1] ]
      kH = [ [ packF32 (applyRope p (sliceHead (kSeq !! p) kv)) | kv <- [0 .. cKV - 1] ]    | p <- [0 .. seqLen - 1] ]
      vH = [ [ sliceHead (vSeq !! p) kv                         | kv <- [0 .. cKV - 1] ]    | p <- [0 .. seqLen - 1] ]
      attnAt p h =
        let kv     = h `div` nGroups
            qv     = (qH !! p) !! h
            scores = [ floatDot qv ((kH !! p') !! kv) cHeadDim * scale | p' <- [0 .. p] ]
            sm     = softmax scores
            vs     = [ (vH !! p') !! kv | p' <- [0 .. p] ]
        in foldr1 (zipWith (+)) (zipWith (\w vv -> map (w *) vv) sm vs)
      attnOut p = concat [ attnAt p h | h <- [0 .. cHeads - 1] ]
  in [ attnOut p | p <- [0 .. seqLen - 1] ]

-- | Look up the embedding row for a token id from the bf16 embed_tokens bytes.
embedRow :: BS.ByteString -> Int -> [Float]
embedRow embedBytes tokId =
  [ bf16At embedBytes ((tokId * cHidden + c) * 2) | c <- [0 .. cHidden - 1] ]

-- | All of one decoder layer's parameters.
data Layer = Layer
  { lInputNorm   :: [Float]
  , lQ, lK, lV   :: Lin
  , lAttnSubNorm :: [Float]
  , lO           :: Lin
  , lPostNorm    :: [Float]
  , lGate, lUp   :: Lin
  , lFfnSubNorm  :: [Float]
  , lDown        :: Lin
  }

-- | A full decoder layer: pre-norm attention then pre-norm MLP, both with
-- residuals. @x += attn(input_norm x); x += mlp(post_norm x)@.
decoderLayer :: Layer -> [[Float]] -> [[Float]]
decoderLayer ly xs =
  let hn     = map (rmsNorm (lInputNorm ly)) xs
      attn   = attention (map (applyLin (lQ ly)) hn)
                         (map (applyLin (lK ly)) hn)
                         (map (applyLin (lV ly)) hn)
      o      = map (applyLin (lO ly) . rmsNorm (lAttnSubNorm ly)) attn
      xs1    = zipWith (zipWith (+)) xs o
      hn2    = map (rmsNorm (lPostNorm ly)) xs1
      gated  = zipWith (zipWith (\g u -> relu2 g * u))
                       (map (applyLin (lGate ly)) hn2)
                       (map (applyLin (lUp ly)) hn2)
      d      = map (applyLin (lDown ly) . rmsNorm (lFfnSubNorm ly)) gated
  in zipWith (zipWith (+)) xs1 d

-- | A layer's KV cache: the RoPE'd keys and the values for every position seen so
-- far (each vector length cKV * cHeadDim).
type KVCache = ([[Float]], [[Float]])

-- Plain dot product (attention scores are tiny — the int8 BitLinears are the work).
dotF :: [Float] -> [Float] -> Float
dotF a b = sum (zipWith (*) a b)

-- Apply RoPE to every head of a packed multi-head vector at position @pos@.
ropeAll :: Int -> Int -> [Float] -> [Float]
ropeAll nHeads pos vec = concat [ applyRope pos (sliceHead vec h) | h <- [0 .. nHeads - 1] ]

-- | Attention for a single new query against the cached keys/values: per query
-- head, softmax over the dot with each cached key (causal is implicit — the cache
-- only holds positions <= now), then the weighted sum of cached values.
attendStep :: [Float] -> [[Float]] -> [[Float]] -> [Float]
attendStep q cK cV =
  let nGroups = cHeads `div` cKV
      scale   = 1.0 / sqrt (fromIntegral cHeadDim)
      headOut h =
        let kv     = h `div` nGroups
            qh     = sliceHead q h
            scores = [ dotF qh (sliceHead k kv) * scale | k <- cK ]
            sm     = softmax scores
            vs     = [ sliceHead v kv | v <- cV ]
        in foldr1 (zipWith (+)) (zipWith (\w vv -> map (w *) vv) sm vs)
  in concat [ headOut h | h <- [0 .. cHeads - 1] ]

-- | One decoder layer for a single new token at position @pos@, given the layer's
-- KV cache. Returns the new hidden and the cache with this token's key/value
-- appended. Same computation as 'decoderLayer' but for one position with cached
-- attention — the BitLinears still go through the 'bndot' primitive.
decoderStep :: Int -> Layer -> KVCache -> [Float] -> ([Float], KVCache)
decoderStep pos ly (cK, cV) x =
  let hn    = rmsNorm (lInputNorm ly) x
      q     = ropeAll cHeads pos (applyLin (lQ ly) hn)
      kNew  = ropeAll cKV pos (applyLin (lK ly) hn)
      vNew  = applyLin (lV ly) hn
      cK'   = kNew : cK   -- prepend (O(1)); attention sums over all positions, so
      cV'   = vNew : cV   -- order doesn't matter and position is baked in via RoPE
      attn  = attendStep q cK' cV'
      o     = applyLin (lO ly) (rmsNorm (lAttnSubNorm ly) attn)
      x1    = zipWith (+) x o
      hn2   = rmsNorm (lPostNorm ly) x1
      gated = zipWith (\g u -> relu2 g * u) (applyLin (lGate ly) hn2) (applyLin (lUp ly) hn2)
      d     = applyLin (lDown ly) (rmsNorm (lFfnSubNorm ly) gated)
  in (zipWith (+) x1 d, (cK', cV'))

-- ---------- C-kernel generation path (all per-token math in C) ----------

-- | A decoder layer for the C path: the four RMSNorm weights are float32 byte
-- buffers (decoded once at build time); the BitLinears stay packed 'Lin's.
data LayerC = LayerC
  { cInputNorm   :: BS.ByteString
  , cQ, cK, cV   :: Lin
  , cAttnSubNorm :: BS.ByteString
  , cO           :: Lin
  , cPostNorm    :: BS.ByteString
  , cGate, cUp   :: Lin
  , cFfnSubNorm  :: BS.ByteString
  , cDown        :: Lin
  }

-- | One layer's KV cache for the C path: the RoPE'd keys and the values for all
-- positions so far, each a contiguous float32 buffer. 'Nothing' before any token,
-- so the first key/value seeds the cache.
type KVC = Maybe (BS.ByteString, BS.ByteString)

-- | Apply a BitLinear to a float32 activation buffer, in C (fused quantize +
-- ternary matmul + rescale). Returns the float32 output buffer.
applyLinC :: Lin -> BS.ByteString -> BS.ByteString
applyLinC (Lin p inD outD sc) x = bnBitLinearF p sc x inD outD

-- | One decoder layer for a single new token at position @pos@, entirely on
-- float32 byte buffers through the C kernels (same computation as 'decoderStep').
decoderStepC :: LayerC -> Int -> KVC -> BS.ByteString -> (BS.ByteString, KVC)
decoderStepC ly pos mc x =
  let hn    = bnRmsNorm (cInputNorm ly) x cHidden cEps
      q     = bnRope (applyLinC (cQ ly) hn) pos cHeads cHeadDim cTheta
      k     = bnRope (applyLinC (cK ly) hn) pos cKV cHeadDim cTheta
      v     = applyLinC (cV ly) hn
      (cK', cV') = case mc of
        Nothing        -> (k, v)
        Just (ck, cv)  -> (ck `BS.append` k, cv `BS.append` v)
      nPos  = pos + 1
      scale = 1.0 / sqrt (fromIntegral cHeadDim)
      attn  = bnAttend q cK' cV' nPos cHeads cKV cHeadDim scale
      o     = applyLinC (cO ly) (bnRmsNorm (cAttnSubNorm ly) attn cHidden cEps)
      x1    = bnAdd x o cHidden
      hn2   = bnRmsNorm (cPostNorm ly) x1 cHidden cEps
      ffn   = linOut (cGate ly)
      gm    = bnSqReluMul (applyLinC (cGate ly) hn2) (applyLinC (cUp ly) hn2) ffn
      d     = applyLinC (cDown ly) (bnRmsNorm (cFfnSubNorm ly) gm ffn cEps)
      x2    = bnAdd x1 d cHidden
  in (x2, Just (cK', cV'))
