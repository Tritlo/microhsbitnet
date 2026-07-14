-- | Minimal GGUF v3 reader for the 1-bit Bonsai weights.
--
-- Parses the header (typed KVs and the tensor table) from a resident
-- ByteString of the whole file. Large arrays (tokenizer vocab/merges) are NOT
-- materialized: an array value is kept as (element type, count, absolute byte
-- offset) so later consumers (the C tokenizer) can walk the raw bytes
-- directly. Only what the loader needs; tensor data types beyond F32/F16/Q1_0
-- are carried numerically, not interpreted.
module Gguf
  ( GVal(..)
  , GTensor(..)
  , Gguf(..)
  , parseGguf
  , lookupKV
  , lookupTensor
  , kvInt
  , intArray
  , typeName
  , tensorBytes
  , f32At
  , f16ToFloat
  , f16At
  , q1At
  , q1Row
  ) where

import Data.Word (Word16)
import Data.Bits (shiftR, shiftL, (.&.), testBit)
import Mhs.Array (Array, listArray, (!))
import qualified Data.ByteString as BS

-- | A decoded metadata value. Arrays stay raw: element type, count, and the
-- absolute file offset of the first element (strings are u64-length-prefixed).
data GVal
  = GInt Int          -- ^ all integer widths and bool (0/1)
  | GFloat Float      -- ^ f32/f64
  | GStr String
  | GArr Int Int Int  -- ^ elem type, count, absolute byte offset of elements
  deriving (Show)

-- | One tensor-table entry: name, logical shape (ne, innermost first, as in
-- GGUF), ggml type id, and the absolute byte offset of its data in the file.
data GTensor = GTensor
  { gtName  :: String
  , gtShape :: [Int]
  , gtType  :: Int
  , gtOff   :: Int
  } deriving (Show)

-- | A parsed file: KVs, tensor table, the resident file bytes, and where the
-- (aligned) data section starts.
data Gguf = Gguf
  { gKVs       :: [(String, GVal)]
  , gTensors   :: [GTensor]
  , gBytes     :: BS.ByteString
  , gDataStart :: Int
  }

-- Little-endian unsigned integer of n bytes at offset o.
leAt :: BS.ByteString -> Int -> Int -> Int
leAt bs o n = go 0 0
  where
    go i acc | i >= n    = acc
             | otherwise = go (i + 1) (acc + fromIntegral (BS.index bs (o + i)) `shiftL` (8 * i))

u8At, u16At, u32At, u64At :: BS.ByteString -> Int -> Int
u8At  bs o = leAt bs o 1
u16At bs o = leAt bs o 2
u32At bs o = leAt bs o 4
u64At bs o = leAt bs o 8

-- Signed views (two's complement).
sAt :: BS.ByteString -> Int -> Int -> Int
sAt bs o n =
  let v = leAt bs o n
      half = 1 `shiftL` (8 * n - 1)
  in if v >= half then v - 2 * half else v

-- Exact powers of two, shared by the float decoders.
pow2 :: Array Int Float
pow2 = listArray (-160, 160) [ 2 ^^ e | e <- [-160 .. 160] ]

-- | Decode an IEEE f32 bit pattern.
f32FromBits :: Int -> Float
f32FromBits w =
  let s = if testBit w 31 then -1 else 1
      e = (w `shiftR` 23) .&. 0xFF
      m = w .&. 0x7FFFFF
  in if e == 0
       then s * (fromIntegral m / 8388608) * (pow2 ! (-126))
       else if e == 255
              then (if m == 0 then s * (1 / 0) else 0 / 0)
              else s * (1 + fromIntegral m / 8388608) * (pow2 ! (e - 127))

-- | The little-endian f32 at byte offset k.
f32At :: BS.ByteString -> Int -> Float
f32At bs k = f32FromBits (u32At bs k)

-- | Decode an IEEE f16 bit pattern (Q1_0 group scales are f16).
f16ToFloat :: Word16 -> Float
f16ToFloat w' =
  let w = fromIntegral w' :: Int
      s = if testBit w 15 then -1 else 1
      e = (w `shiftR` 10) .&. 0x1F
      m = w .&. 0x3FF
  in if e == 0
       then s * (fromIntegral m / 1024) * (pow2 ! (-14))
       else if e == 31
              then (if m == 0 then s * (1 / 0) else 0 / 0)
              else s * (1 + fromIntegral m / 1024) * (pow2 ! (e - 15))

-- | The little-endian f16 at byte offset k.
f16At :: BS.ByteString -> Int -> Float
f16At bs k = f16ToFloat (fromIntegral (u16At bs k))

-- | Q1_0 spot decode: weight j of a tensor whose data starts at byte offset o.
-- 18-byte blocks of 128: f16 scale then 16 sign bytes, LSB-first; bit=1 is +d.
q1At :: BS.ByteString -> Int -> Int -> Float
q1At bs o j =
  let blk = j `div` 128
      r   = j `mod` 128
      bo  = o + 18 * blk
      d   = f16At bs bo
      b   = BS.index bs (bo + 2 + r `div` 8)
  in if testBit b (r `mod` 8) then d else negate d

-- | Dequantize weights [c0 .. c0+n) of the row of length rowLen at index r.
q1Row :: BS.ByteString -> Int -> Int -> Int -> Int -> Int -> [Float]
q1Row bs o rowLen r c0 n =
  [ q1At bs o (r * rowLen + c) | c <- [c0 .. c0 + n - 1] ]

-- A UTF-8-oblivious byte string decode (tokenizer bytes are handled in C).
strAt :: BS.ByteString -> Int -> Int -> String
strAt bs o n = [ toEnum (fromIntegral (BS.index bs (o + i))) | i <- [0 .. n - 1] ]

-- Parse one value of GGUF metadata type t at offset o; return (value, next offset).
pVal :: BS.ByteString -> Int -> Int -> (GVal, Int)
pVal bs t o = case t of
  0  -> (GInt (u8At bs o),          o + 1)  -- u8
  1  -> (GInt (sAt bs o 1),         o + 1)  -- i8
  2  -> (GInt (u16At bs o),         o + 2)  -- u16
  3  -> (GInt (sAt bs o 2),         o + 2)  -- i16
  4  -> (GInt (u32At bs o),         o + 4)  -- u32
  5  -> (GInt (sAt bs o 4),         o + 4)  -- i32
  6  -> (GFloat (f32At bs o),       o + 4)  -- f32
  7  -> (GInt (u8At bs o),          o + 1)  -- bool
  8  -> let n = u64At bs o in (GStr (strAt bs (o + 8) n), o + 8 + n)
  9  -> let et = u32At bs o
            n  = u64At bs (o + 4)
            elems = o + 12
        in (GArr et n elems, skipN bs et n elems)
  10 -> (GInt (u64At bs o),         o + 8)  -- u64
  11 -> (GInt (sAt bs o 8),         o + 8)  -- i64
  12 -> (GFloat (realToFrac (f64At bs o)), o + 8)
  _  -> error ("gguf: unknown kv type " ++ show t)

f64At :: BS.ByteString -> Int -> Double
f64At bs k =
  let w = u64At bs k
      s = if testBit w 63 then -1 else 1 :: Double
      e = (w `shiftR` 52) .&. 0x7FF
      m = w .&. 0xFFFFFFFFFFFFF
      frac = fromIntegral m / 4503599627370496
  in if e == 0 then s * frac * (2 ^^ (-1022))
     else if e == 2047 then (if m == 0 then s * (1/0) else 0/0)
     else s * (1 + frac) * (2 ^^ (e - 1023))

-- Skip n elements of type et starting at offset o (fixed-size or strings).
skipN :: BS.ByteString -> Int -> Int -> Int -> Int
skipN bs et n o
  | et == 8   = goStr n o
  | otherwise = o + n * fixedSize et
  where
    -- strict in the offset: half a million lazy `+`s would overflow the stack
    goStr k p | k <= 0    = p
              | otherwise = let p' = p + 8 + u64At bs p
                            in p' `seq` goStr (k - 1) p'

fixedSize :: Int -> Int
fixedSize t = case t of
  0 -> 1; 1 -> 1; 2 -> 2; 3 -> 2; 4 -> 4; 5 -> 4; 6 -> 4; 7 -> 1
  10 -> 8; 11 -> 8; 12 -> 8
  _ -> error ("gguf: array of type " ++ show t ++ " not sized")

-- Parse a length-prefixed string; return (string, next offset).
pStr :: BS.ByteString -> Int -> (String, Int)
pStr bs o = let n = u64At bs o in (strAt bs (o + 8) n, o + 8 + n)

-- | Parse a GGUF v3 file from its resident bytes.
parseGguf :: BS.ByteString -> Gguf
parseGguf bs
  | u32At bs 0 /= 0x46554747 = error "gguf: bad magic"
  | version /= 3             = error ("gguf: unsupported version " ++ show version)
  | otherwise                = Gguf kvs tensors bs dataStart
  where
    version  = u32At bs 4
    nTensors = u64At bs 8
    nKV      = u64At bs 16

    (kvs, afterKVs) = goKV nKV 24 []
    goKV k o acc
      | k <= 0    = (reverse acc, o)
      | otherwise =
          let (key, o1) = pStr bs o
              t         = u32At bs o1
              (v, o2)   = pVal bs t (o1 + 4)
          in o2 `seq` goKV (k - 1) o2 ((key, v) : acc)

    (infos, headerEnd) = goT nTensors afterKVs []
    goT k o acc
      | k <= 0    = (reverse acc, o)
      | otherwise =
          let (nm, o1) = pStr bs o
              nd       = u32At bs o1
              dims     = [ u64At bs (o1 + 4 + 8 * i) | i <- [0 .. nd - 1] ]
              o2       = o1 + 4 + 8 * nd
              ty       = u32At bs o2
              off      = u64At bs (o2 + 4)
          in o2 `seq` goT (k - 1) (o2 + 12) (GTensor nm dims ty off : acc)

    alignment = case lookup "general.alignment" kvs of
                  Just (GInt a) -> a
                  _             -> 32
    dataStart = ((headerEnd + alignment - 1) `div` alignment) * alignment
    tensors   = [ t { gtOff = dataStart + gtOff t } | t <- infos ]

-- | Look up a metadata value.
lookupKV :: String -> Gguf -> Maybe GVal
lookupKV k g = lookup k (gKVs g)

-- | Look up a tensor by name.
lookupTensor :: String -> Gguf -> Maybe GTensor
lookupTensor n g = go (gTensors g)
  where
    go []       = Nothing
    go (t : ts) = if gtName t == n then Just t else go ts

-- | An integer KV or a default.
kvInt :: String -> Int -> Gguf -> Int
kvInt k d g = case lookupKV k g of
  Just (GInt v) -> v
  _             -> d

-- | Decode a small integer array value (e.g. rope sections).
intArray :: Gguf -> GVal -> [Int]
intArray g (GArr et n o)
  | et == 5   = [ sAt (gBytes g) (o + 4 * i) 4 | i <- [0 .. n - 1] ]
  | et == 4   = [ u32At (gBytes g) (o + 4 * i) | i <- [0 .. n - 1] ]
  | otherwise = error ("gguf: intArray of type " ++ show et)
intArray _ _ = error "gguf: intArray of non-array"

-- | Human name for the tensor types this project meets.
typeName :: Int -> String
typeName 0  = "F32"
typeName 1  = "F16"
typeName 41 = "Q1_0"
typeName t  = "T" ++ show t

-- | On-disk byte size of a tensor (only the types this model uses).
tensorBytes :: GTensor -> Int
tensorBytes t =
  let n = product (gtShape t)
  in case gtType t of
       0  -> 4 * n
       1  -> 2 * n
       41 -> 18 * (n `div` 128)
       ty -> error ("gguf: size of type " ++ show ty)
