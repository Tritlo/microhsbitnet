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
  , cHidden, cHeads, cKV, cHeadDim, cTheta, cEps
  ) where

import qualified Data.ByteString as BS
import Safetensors (bf16At)
import BitLinear (quantizeAct, bitLinearFast)

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
  let (xq, ascale) = quantizeAct x
  in bitLinearFast p inD outD sc xq ascale

-- | RMSNorm in float32: @weight * x / sqrt(mean(x^2) + eps)@.
rmsNorm :: [Float] -> [Float] -> [Float]
rmsNorm weight x =
  let n  = fromIntegral (length x)
      ss = sum (map (\v -> v * v) x)
      s  = 1.0 / sqrt (ss / n + cEps)
  in zipWith (\w v -> w * (v * s)) weight x

-- | Squared ReLU: @max(0, g)^2@.
relu2 :: Float -> Float
relu2 g = let r = if g > 0.0 then g else 0.0 in r * r

-- | Apply rotary position embedding to one head vector (length 'cHeadDim') at
-- position @pos@. inv_freq[k] = theta^(-2k/headDim); rotate_half = [-x2, x1].
applyRope :: Int -> [Float] -> [Float]
applyRope pos hv =
  let half = cHeadDim `div` 2
      h1   = take half hv
      h2   = drop half hv
      ang k = fromIntegral pos
              * (cTheta ** (negate (2.0 * fromIntegral k) / fromIntegral cHeadDim))
      rot  = [ let a = ang k in (x1 * cos a - x2 * sin a, x2 * cos a + x1 * sin a)
             | (k, x1, x2) <- zip3 [0 ..] h1 h2 ]
  in map fst rot ++ map snd rot

softmax :: [Float] -> [Float]
softmax xs =
  let m  = maximum xs
      es = map (\x -> exp (x - m)) xs
      s  = sum es
  in map (/ s) es

dotF :: [Float] -> [Float] -> Float
dotF a b = sum (zipWith (*) a b)

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
      qH = [ [ applyRope p (sliceHead (qSeq !! p) h)  | h  <- [0 .. cHeads - 1] ] | p <- [0 .. seqLen - 1] ]
      kH = [ [ applyRope p (sliceHead (kSeq !! p) kv) | kv <- [0 .. cKV - 1] ]    | p <- [0 .. seqLen - 1] ]
      vH = [ [ sliceHead (vSeq !! p) kv               | kv <- [0 .. cKV - 1] ]    | p <- [0 .. seqLen - 1] ]
      attnAt p h =
        let kv     = h `div` nGroups
            qv     = (qH !! p) !! h
            scores = [ dotF qv ((kH !! p') !! kv) * scale | p' <- [0 .. p] ]
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
