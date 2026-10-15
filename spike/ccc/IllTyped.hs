module IllTyped where

-- | Exercise upstream type-checker rejection.
scalar :: Float -> Float
scalar input = input + True
