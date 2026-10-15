module Unsupported where

import qualified SourceArray as A

-- | Exercise rejection of a primitive outside the spike vocabulary.
scalar :: Float -> Float
scalar input = log input

-- | Exercise lexical capture when a later binding shadows its source name.
shadowed :: Float -> Float
shadowed input = let captured = input in let input = 2 in captured + input

-- | Exercise rejection of recursive source functions.
recursive :: Float -> Float
recursive input = recursive input

-- | Reject an inner array computation that depends on an outer map element.
dependentArray :: A.Vector Float -> A.Vector Float
dependentArray scores = A.map (\outer -> A.sum (A.map (\inner -> outer + inner) scores)) scores
