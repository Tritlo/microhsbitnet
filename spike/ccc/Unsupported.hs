module Unsupported where

-- | Exercise rejection of a primitive outside the spike vocabulary.
scalar :: Float -> Float
scalar input = log input

-- | Exercise rejection of source let bindings with captured variables.
shadowed :: Float -> Float
shadowed input = let captured = input in let input = 2 in captured + input
