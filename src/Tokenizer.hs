-- | A byte-level BPE tokenizer (Llama 3 style) in Haskell, so the demo needs no
-- Python for tokenization. This module covers DECODING: token ids -> text.
--
-- Each non-special token is a string over the GPT-2 "byte-level" alphabet (256
-- codepoints, computed below — not data). The model's tokenizer.json stores those
-- strings with @\\uXXXX@ escapes, so the file is pure ASCII. Decoding a token =
-- map each codepoint back to its byte; the concatenated bytes are the UTF-8 text.
module Tokenizer
  ( loadDecoder
  , decode
  ) where

import Data.Char (isSpace, isDigit, ord, chr)
import Data.Word (Word8)
import Data.Bits ((.&.))
import qualified Data.ByteString as BS
import System.IO (openBinaryFile, hClose, IOMode(ReadMode), Handle)
import Mhs.Array (Array, accumArray, (!))

-- Decode a UTF-8 byte stream to codepoints. The byte-level alphabet chars (e.g.
-- Ġ = U+0120) are stored as raw UTF-8 in tokenizer.json, so this must run before
-- parsing; ASCII (incl. any \\u escapes) passes through unchanged.
utf8 :: [Word8] -> [Int]
utf8 [] = []
utf8 (b : bs)
  | b < 0x80  = fromIntegral b : utf8 bs
  | b < 0xE0  = case bs of (b1 : r)           -> two b b1 : utf8 r;        _ -> []
  | b < 0xF0  = case bs of (b1 : b2 : r)      -> three b b1 b2 : utf8 r;   _ -> []
  | otherwise = case bs of (b1 : b2 : b3 : r) -> four b b1 b2 b3 : utf8 r; _ -> []
  where
    lo x        = fromIntegral (x .&. 0x3F)
    two x y     = fromIntegral (x .&. 0x1F) * 64 + lo y
    three x y z = (fromIntegral (x .&. 0x0F) * 64 + lo y) * 64 + lo z
    four x y z w = ((fromIntegral (x .&. 0x07) * 64 + lo y) * 64 + lo z) * 64 + lo w

-- The 68 byte values that are NOT printable in the GPT-2 byte-level alphabet, in
-- order; codepoints 256+n decode to these. The rest (33..126,161..172,174..255)
-- decode to themselves.
nonPrintable :: [Int]
nonPrintable = [0 .. 32] ++ [127 .. 160] ++ [173]

-- codepoint -> byte for the byte-level alphabet (valid for 33..323).
levelToByte :: Int -> Int
levelToByte cp | cp < 256  = cp
               | otherwise = nonPrintable !! (cp - 256)

-- Read a whole file as a ByteString (chunked; avoids the broken BS.readFile).
readFileBS :: FilePath -> IO BS.ByteString
readFileBS path = do
  h <- openBinaryFile path ReadMode
  first <- BS.hGet h chunkSz
  bs <- go h first
  hClose h
  return bs
  where
    chunkSz = 16 * 1024 * 1024
    go h acc = do
      c <- BS.hGet h chunkSz
      if BS.length c == 0 then return acc else go h (BS.append acc c)

hex4 :: Char -> Char -> Char -> Char -> Int
hex4 a b c d = ((hd a * 16 + hd b) * 16 + hd c) * 16 + hd d
  where hd x | isDigit x = ord x - ord '0'
             | x >= 'a'  = ord x - ord 'a' + 10
             | otherwise = ord x - ord 'A' + 10

-- Parse a JSON string body (after the opening quote) into codepoints; returns the
-- codepoints and the rest after the closing quote.
parseJStr :: String -> ([Int], String)
parseJStr = go []
  where
    go acc ('\\' : 'u' : a : b : c : d : r) = go (hex4 a b c d : acc) r
    go acc ('\\' : 'n' : r) = go (10 : acc) r
    go acc ('\\' : 'r' : r) = go (13 : acc) r
    go acc ('\\' : 't' : r) = go (9 : acc) r
    go acc ('\\' : x : r)   = go (ord x : acc) r
    go acc ('"' : r)        = (reverse acc, r)
    go acc (x : r)          = go (ord x : acc) r
    go acc []               = (reverse acc, [])

-- Drop input up to and including the first occurrence of @pat@.
afterPat :: String -> String -> String
afterPat pat = go
  where
    n = length pat
    go [] = []
    go s | take n s == pat = drop n s
    go (_ : r) = go r

-- Parse @"string": int@ entries of an object body into (id, codepoints), stopping
-- at the closing '}'.
parseObjEntries :: String -> [(Int, [Int])]
parseObjEntries s =
  case dropWhile isSpace s of
    ('"' : r) -> let (cps, r1) = parseJStr r
                     r2 = dropWhile isSpace (drop 1 (dropWhile (/= ':') r1))
                     (ds, r3) = span isDigit r2
                     r4 = dropWhile (\c -> c == ',' || isSpace c) r3
                 in (read ds, cps) : parseObjEntries r4
    _         -> []

-- Parse the added_tokens array body into (id, content-codepoints). Special tokens
-- render as their literal content (e.g. "<|eot_id|>").
parseAdded :: String -> [(Int, [Int])]
parseAdded s = case afterPat "\"id\":" s of
  "" -> []
  r  -> let (ds, r1) = span isDigit (dropWhile isSpace r)
            r2 = dropWhile isSpace (afterPat "\"content\":" r1)
        in case r2 of
             ('"' : r3) -> let (cps, r4) = parseJStr r3 in (read ds, cps) : parseAdded r4
             _          -> []

-- | Build the decoder table id -> bytes (length 128256) from tokenizer.json.
-- Regular tokens decode their byte-level codepoints to bytes; special tokens carry
-- their literal content bytes.
loadDecoder :: String -> IO (Array Int [Word8])
loadDecoder path = do
  bs <- readFileBS path
  let s = map chr (utf8 (BS.unpack bs))
      vocabBody  = drop 1 (dropWhile (/= '{') (afterPat "\"vocab\":" s))
      addedBody  = takeWhile (/= ']') (afterPat "\"added_tokens\":" s)
      regEntries = [ (i, map (fromIntegral . levelToByte) cps) | (i, cps) <- parseObjEntries vocabBody ]
      addEntries = [ (i, map fromIntegral cps)                 | (i, cps) <- parseAdded addedBody ]
  return (accumArray (\_ new -> new) [] (0, 128255) (regEntries ++ addEntries))

-- | Decode token ids to bytes (the UTF-8 text).
decode :: Array Int [Word8] -> [Int] -> [Word8]
decode tbl = concatMap (tbl !)
