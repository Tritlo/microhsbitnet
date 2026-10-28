module Unsupported where

import Data.Int (Int32)
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

-- | Reject an outer element captured by an inner map before scalar addition.
dependentArithmetic :: A.Vector Float -> A.Vector Float
dependentArithmetic values = A.map (\outer -> outer + A.sum (A.map (\inner -> outer * inner) values)) values

-- | Keep an inner reduction that does not depend on the outer element.
independentNested :: A.Vector Float -> A.Vector Float
independentNested values = A.map (\outer -> outer + A.sum (A.map (\inner -> inner * inner) values)) values

-- | Return a captured reduction without an element arithmetic operation.
capturedMap :: A.Vector Float -> A.Vector Float
capturedMap values = A.map (\_ -> A.sum values) values

-- | Return a captured reduction without using either zip element.
capturedZip :: A.Vector Float -> A.Vector Float -> A.Vector Float
capturedZip left right = A.zipWith (\_ _ -> A.sum left) left right

-- | Reject an integer identity through a Float entry point.
intIdentity :: Int32 -> Int32
intIdentity value = value

-- | Reject integer arguments through a binary Float entry point.
intFirst :: Int32 -> Int32 -> Int32
intFirst left _ = left

-- | Reject a vector with an unsupported element type.
intVectorIdentity :: A.Vector Int32 -> A.Vector Int32
intVectorIdentity values = values

-- | Reject a scalar parameter supplied through a vector entry point.
ignoredScalar :: A.Vector Float -> Float -> A.Vector Float
ignoredScalar values _ = values
