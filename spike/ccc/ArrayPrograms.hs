module ArrayPrograms (softmax) where

import qualified SourceArray as A

-- | Normalize a nonempty finite vector with a stable exponential sum.
softmax :: A.Vector Float -> A.Vector Float
softmax scores =
    let peak = A.maximum scores
        numerators = A.map (\score -> exp (score - peak)) scores
        denominator = A.sum numerators
     in A.map (\numerator -> numerator / denominator) numerators
