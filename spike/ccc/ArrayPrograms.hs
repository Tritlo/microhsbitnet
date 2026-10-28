module ArrayPrograms (softmax, rmsNorm, residual, gated) where

import qualified SourceArray as A

-- | Normalize a nonempty finite vector with a stable exponential sum.
softmax :: A.Vector Float -> A.Vector Float
softmax scores =
    let peak = A.maximum scores
        numerators = A.map (\score -> exp (score - peak)) scores
        denominator = A.sum numerators
     in A.map (\numerator -> numerator / denominator) numerators

-- | RMSNorm: scale each value by the inverse root-mean-square, then apply a weight.
rmsNorm :: A.Vector Float -> A.Vector Float -> A.Vector Float
rmsNorm values weights =
    let total = A.sum (A.map (\value -> value * value) values)
        scale = 1 / sqrt (total / A.count values + 0.00001)
     in A.zipWith (\weight value -> weight * (value * scale)) weights values

-- | Add two activations in index order.
residual :: A.Vector Float -> A.Vector Float -> A.Vector Float
residual left right = A.zipWith (\leftValue rightValue -> leftValue + rightValue) left right

-- | Squared ReLU gate, multiplied by the up projection.
gated :: A.Vector Float -> A.Vector Float -> A.Vector Float
gated gate up = A.zipWith (\gateValue upValue -> (max 0 gateValue) * (max 0 gateValue) * upValue) gate up
