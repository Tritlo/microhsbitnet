module SourceArray (Vector, fromList, toList, map, zipWith, sum, maximum, count) where

import Data.List (foldl')
import Prelude hiding (map, maximum, sum, zipWith)
import qualified Prelude

-- | A finite vector with a CPU interpretation for the array compiler intrinsics.
newtype Vector element = Vector [element] deriving (Show, Read, Eq)

-- | Construct a vector without changing element order.
fromList :: [element] -> Vector element
fromList = Vector

-- | Read the elements in index order.
toList :: Vector element -> [element]
toList (Vector values) = values

-- | Apply a scalar function at each index.
map :: (Float -> Float) -> Vector Float -> Vector Float
map function (Vector values) = Vector (Prelude.map function values)

-- | Combine two equal-length vectors in index order.
zipWith :: (Float -> Float -> Float) -> Vector Float -> Vector Float -> Vector Float
zipWith function (Vector left) (Vector right) = Vector (Prelude.zipWith function left right)

-- | Add floats from left to right, starting at zero.
sum :: Vector Float -> Float
sum (Vector values) = foldl' (+) 0 values

-- | Select the maximum from a nonempty finite vector in index order.
maximum :: Vector Float -> Float
maximum (Vector []) = error "Array maximum requires a nonempty vector"
maximum (Vector (first : remaining)) = foldl' max first remaining

-- | Read the vector length as a Float. This is a compiler intrinsic, not a reduction.
count :: Vector Float -> Float
count (Vector values) = fromIntegral (Prelude.length values)
