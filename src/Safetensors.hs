-- | Minimal safetensors reader for the BitNet b1.58 packed weights.
--
-- Parses the JSON header, decodes bf16 scalars, and unpacks the packed ternary
-- ('Trit') weights. Deliberately tiny: only what the loader needs, no general
-- JSON machinery beyond the header's object/array/string/number forms.
module Safetensors
  ( Tensor(..)
  , parseHeader
  , lookupTensor
  , bf16ToFloat
  , bf16At
  , tritAt
  , unpackRow
  ) where

import Data.Char (isDigit, isSpace)
import Data.Word (Word8, Word16)
import Data.Bits (shiftR, (.&.))
import Mhs.Array (Array, listArray, (!))
import qualified Data.ByteString as BS

-- | One tensor's header entry: name, dtype string, logical shape, and the
-- half-open byte range @[begin, end)@ within the data section.
data Tensor = Tensor
  { tName  :: String
  , tDtype :: String
  , tShape :: [Int]
  , tBegin :: Int
  , tEnd   :: Int
  } deriving (Show)

-- A throwaway JSON value, just enough for the header.
data JSON = JStr String | JNum Double | JArr [JSON] | JObj [(String, JSON)] | JBool Bool | JNull

ws :: String -> String
ws = dropWhile isSpace

-- | Parse a JSON value, returning the rest of the input.
parseJSON :: String -> Maybe (JSON, String)
parseJSON s0 =
  case ws s0 of
    ('"' : r) -> case pStr r of
                   Just (str, r') -> Just (JStr str, r')
                   Nothing        -> Nothing
    ('{' : r) -> pObj (ws r)
    ('[' : r) -> pArr (ws r)
    ('t' : r) -> Just (JBool True, drop 3 r)
    ('f' : r) -> Just (JBool False, drop 4 r)
    ('n' : r) -> Just (JNull, drop 3 r)
    s         -> pNum s

-- After the opening quote; naive escape handling (keep the escaped char).
pStr :: String -> Maybe (String, String)
pStr = go []
  where
    go acc ('\\' : c : r) = go (c : acc) r
    go acc ('"' : r)      = Just (reverse acc, r)
    go acc (c : r)        = go (c : acc) r
    go _   []             = Nothing

pNum :: String -> Maybe (JSON, String)
pNum s =
  let (numStr, r) = span (\c -> isDigit c || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') s
  in if null numStr then Nothing else Just (JNum (read numStr), r)

pArr :: String -> Maybe (JSON, String)
pArr (']' : r) = Just (JArr [], r)
pArr s         = go [] s
  where
    go acc str =
      case parseJSON str of
        Nothing -> Nothing
        Just (v, r1) ->
          case ws r1 of
            (',' : r2) -> go (v : acc) (ws r2)
            (']' : r2) -> Just (JArr (reverse (v : acc)), r2)
            _          -> Nothing

pObj :: String -> Maybe (JSON, String)
pObj ('}' : r) = Just (JObj [], r)
pObj s         = go [] s
  where
    go acc str =
      case ws str of
        ('"' : r0) ->
          case pStr r0 of
            Nothing -> Nothing
            Just (k, r1) ->
              case ws r1 of
                (':' : r2) ->
                  case parseJSON (ws r2) of
                    Nothing -> Nothing
                    Just (v, r3) ->
                      case ws r3 of
                        (',' : r4) -> go ((k, v) : acc) (ws r4)
                        ('}' : r4) -> Just (JObj (reverse ((k, v) : acc)), r4)
                        _          -> Nothing
                _ -> Nothing
        _ -> Nothing

-- | Parse the header JSON into tensor entries (dropping @__metadata__@).
parseHeader :: String -> [Tensor]
parseHeader s =
  case parseJSON s of
    Just (JObj kvs, _) -> concatMap entry kvs
    _                  -> []
  where
    entry (k, v) | k == "__metadata__" = []
                 | otherwise           = maybe [] (: []) (toTensor k v)

toTensor :: String -> JSON -> Maybe Tensor
toTensor name (JObj fs) =
  case (lookup "dtype" fs, lookup "shape" fs, lookup "data_offsets" fs) of
    (Just (JStr dt), Just (JArr sh), Just (JArr off)) ->
      case [ truncate d | JNum d <- off ] of
        [b, e] -> Just (Tensor name dt [ truncate d | JNum d <- sh ] b e)
        _      -> Nothing
    _ -> Nothing
toTensor _ _ = Nothing

-- | Find a tensor by name.
lookupTensor :: String -> [Tensor] -> Maybe Tensor
lookupTensor n = go
  where
    go []       = Nothing
    go (t : ts) = if tName t == n then Just t else go ts

-- Exact powers of two 2^(e-127) for each raw bf16 exponent e (0..255), built once.
-- @^^@ is repeated multiply/divide, so these are exact (unlike @**@).
pow2 :: Array Int Float
pow2 = listArray (0, 255) [ 2 ^^ (e - 127) | e <- [0 .. 255] ]

-- | Decode a bf16 bit pattern to 'Float'. Exact: bf16's 7 mantissa bits fit in
-- Float's 23. (-1)^s * 2^(e-127) * (1 + m/128), with e==0 sub-normal/zero. Uses
-- an O(1) power-of-two lookup (the LM head decodes embeddings hundreds of millions
-- of times).
bf16ToFloat :: Word16 -> Float
bf16ToFloat w =
  let s = (w `shiftR` 15) .&. 1
      e = fromIntegral ((w `shiftR` 7) .&. 0xFF) :: Int
      m = fromIntegral (w .&. 0x7F) :: Int
      sign = if s == 1 then -1 else 1
  in if e == 0
       then sign * (fromIntegral m / 128) * (pow2 ! 1)
       else if e == 255
              then (if m == 0 then sign * (1 / 0) else 0 / 0)
              else sign * (1 + fromIntegral m / 128) * (pow2 ! e)

-- | Decode the little-endian bf16 at byte offset k of a ByteString to 'Float'.
bf16At :: BS.ByteString -> Int -> Float
bf16At bs k =
  let lo = fromIntegral (BS.index bs k)       :: Word16
      hi = fromIntegral (BS.index bs (k + 1)) :: Word16
  in bf16ToFloat (lo + 256 * hi)

-- | The trit at logical position (row r, col c) of a BitLinear weight of logical
-- shape [outDim, inDim], stored packed as [outDim\/4, inDim] U8 row-major.
-- Byte j holds out-rows {j, j+outDim\/4, j+2*outDim\/4, j+3*outDim\/4} in 2-bit
-- fields i=0..3, encoded as trit+1; so trit = ((byte >> 2i) & 3) - 1.
tritAt :: BS.ByteString -> Int -> Int -> Int -> Int -> Int
tritAt packed inDim outDim r c =
  let chunk = outDim `div` 4
      i = r `div` chunk
      j = r `mod` chunk
      b = BS.index packed (j * inDim + c)
  in fromIntegral ((b `shiftR` (2 * i)) .&. 3) - 1

-- | Unpack a full logical row r (length inDim) to trits.
unpackRow :: BS.ByteString -> Int -> Int -> Int -> [Int]
unpackRow packed inDim outDim r = [ tritAt packed inDim outDim r c | c <- [0 .. inDim - 1] ]
