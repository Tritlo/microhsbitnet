-- | A byte-level BPE tokenizer (Llama 3 style) in Haskell — no Python for
-- tokenization. Encode (text -> ids) and decode (ids -> text).
--
-- Each non-special token is a string over the GPT-2 "byte-level" alphabet (256
-- codepoints, computed below — not data). tokenizer.json stores those strings as
-- raw UTF-8, so the file is UTF-8 decoded before parsing. No Data.Map in MicroHs,
-- so vocab/merge lookups are sorted arrays + binary search (merges built lazily —
-- with ignore_merges, whole pre-tokens that are in the vocab skip BPE entirely).
module Tokenizer
  ( Tokenizer
  , loadTokenizer
  , encodeChat
  , decode
  ) where

import Data.Char (isSpace, isDigit, isAlpha, ord, chr)
import Data.List (sortBy)
import Data.Word (Word8)
import Data.Bits ((.&.))
import qualified Data.ByteString as BS
import System.IO (openBinaryFile, hClose, IOMode(ReadMode), Handle)
import Mhs.Array (Array, listArray, accumArray, (!))

-- ---------- byte-level alphabet (computed) ----------

nonPrintable :: [Int]
nonPrintable = [0 .. 32] ++ [127 .. 160] ++ [173]

levelToByte :: Int -> Int
levelToByte cp | cp < 256  = cp
               | otherwise = nonPrintable !! (cp - 256)

byteToLevel :: Int -> Int
byteToLevel b | (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) = b
              | otherwise = 256 + length (takeWhile (/= b) nonPrintable)

-- ---------- UTF-8 ----------

utf8dec :: [Word8] -> [Int]
utf8dec [] = []
utf8dec (b : bs)
  | b < 0x80  = fromIntegral b : utf8dec bs
  | b < 0xE0  = case bs of (b1 : r)           -> two b b1 : utf8dec r;        _ -> []
  | b < 0xF0  = case bs of (b1 : b2 : r)      -> three b b1 b2 : utf8dec r;   _ -> []
  | otherwise = case bs of (b1 : b2 : b3 : r) -> four b b1 b2 b3 : utf8dec r; _ -> []
  where
    lo x        = fromIntegral (x .&. 0x3F)
    two x y     = fromIntegral (x .&. 0x1F) * 64 + lo y
    three x y z = (fromIntegral (x .&. 0x0F) * 64 + lo y) * 64 + lo z
    four x y z w = ((fromIntegral (x .&. 0x07) * 64 + lo y) * 64 + lo z) * 64 + lo w

utf8enc :: Int -> [Int]
utf8enc c
  | c < 0x80    = [c]
  | c < 0x800   = [0xC0 + c `div` 64, 0x80 + c `mod` 64]
  | otherwise   = [0xE0 + c `div` 4096, 0x80 + (c `div` 64) `mod` 64, 0x80 + c `mod` 64]

-- ---------- JSON-ish scanning ----------

hex4 :: Char -> Char -> Char -> Char -> Int
hex4 a b c d = ((hd a * 16 + hd b) * 16 + hd c) * 16 + hd d
  where hd x | isDigit x = ord x - ord '0'
             | x >= 'a'  = ord x - ord 'a' + 10
             | otherwise = ord x - ord 'A' + 10

-- Parse a JSON string body (after opening quote) into codepoints; rest after the
-- closing quote.
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

afterPat :: String -> String -> String
afterPat pat = go
  where
    n = length pat
    go [] = []
    go s | take n s == pat = drop n s
    go (_ : r) = go r

parseObjEntries :: String -> [(Int, [Int])]
parseObjEntries s =
  case dropWhile isSpace s of
    ('"' : r) -> let (cps, r1) = parseJStr r
                     r2 = dropWhile isSpace (drop 1 (dropWhile (/= ':') r1))
                     (ds, r3) = span isDigit r2
                     r4 = dropWhile (\c -> c == ',' || isSpace c) r3
                 in (read ds, cps) : parseObjEntries r4
    _         -> []

parseAdded :: String -> [(Int, [Int])]
parseAdded s = case afterPat "\"id\":" s of
  "" -> []
  r  -> let (ds, r1) = span isDigit (dropWhile isSpace r)
            r2 = dropWhile isSpace (afterPat "\"content\":" r1)
        in case r2 of
             ('"' : r3) -> let (cps, r4) = parseJStr r3 in (read ds, cps) : parseAdded r4
             _          -> []

-- Parse the merges array body into (left, right) byte-level strings (rank = order).
parseMerges :: String -> [(String, String)]
parseMerges s = case dropWhile (\c -> isSpace c || c == ',') s of
  ('"' : r) -> let (cps, r1) = parseJStr r
                   str = map chr cps
                   (a, b) = breakSpace str
               in (a, b) : parseMerges r1
  _         -> []
  where breakSpace t = let (a, rest) = break (== ' ') t in (a, drop 1 rest)

-- ---------- sorted-array lookup (no Data.Map) ----------

mkLookup :: Ord k => [(k, v)] -> (k -> Maybe v)
mkLookup kvs =
  let sorted = sortBy (\(a, _) (b, _) -> compare a b) kvs
      n      = length sorted
      arr    = listArray (0, n - 1) sorted
      search lo hi k
        | lo >= hi  = Nothing
        | otherwise = let m = (lo + hi) `div` 2
                          (km, vm) = arr ! m
                      in case compare k km of
                           EQ -> Just vm
                           LT -> search lo m k
                           GT -> search (m + 1) hi k
  in \k -> search 0 n k

-- ---------- pre-tokenizer (Llama 3 regex, ASCII) ----------

-- Split text into pre-tokens following the Llama 3 regex (ASCII: \p{L}=alpha,
-- \p{N}=digit). Tried in order: contraction | (opt punct)+letters | 1-3 digits |
-- (opt space)+punct | whitespace.
pretokenize :: String -> [String]
pretokenize [] = []
pretokenize s  = let (tok, rest) = one s in tok : pretokenize rest
  where
    other c = not (isAlpha c) && not (isDigit c) && not (isSpace c)
    one t =
      case t of
        ('\'' : c : r) | c `elem` "stmd"            -> (['\'', c], r)
        ('\'' : 'r' : 'e' : r)                       -> ("'re", r)
        ('\'' : 'v' : 'e' : r)                       -> ("'ve", r)
        ('\'' : 'l' : 'l' : r)                       -> ("'ll", r)
        (c : r) | not (isAlpha c) && not (isDigit c) && c /= '\n' && c /= '\r'
                , (ls@(_ : _), r') <- span isAlpha r -> (c : ls, r')
        (c : _) | isAlpha c                          -> span isAlpha t
        (c : _) | isDigit c                          -> let ds = takeWhile isDigit t
                                                            k  = min 3 (length ds)
                                                        in (take k t, drop k t)
        (' ' : c : r) | other c                      -> let (p, r') = span other (c : r)
                                                        in (' ' : p, r')
        (_ : _) | other (head t)                     -> span other t
        (c : r)                                      -> ([c], r)   -- whitespace etc.
        []                                           -> ([], [])

-- ---------- the tokenizer ----------

-- | byte->bytes decode table, vocab lookup (byte-level string -> id), and a lazy
-- merges lookup (pair -> rank).
data Tokenizer = Tokenizer
  (Array Int [Word8])
  (String -> Maybe Int)
  ((String, String) -> Maybe Int)

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

-- | Parse tokenizer.json once into a 'Tokenizer'.
loadTokenizer :: String -> IO Tokenizer
loadTokenizer path = do
  bs <- readFileBS path
  let s          = map chr (utf8dec (BS.unpack bs))
      vocabBody  = drop 1 (dropWhile (/= '{') (afterPat "\"vocab\":" s))
      addedBody  = takeWhile (/= ']') (afterPat "\"added_tokens\":" s)
      mergesBody = takeWhile (/= ']') (afterPat "\"merges\":" s)
      vocab      = parseObjEntries vocabBody                       -- [(id, codepoints)]
      added      = parseAdded addedBody
      decTbl     = accumArray (\_ x -> x) [] (0, 128255)
                     ([ (i, map (fromIntegral . levelToByte) cps) | (i, cps) <- vocab ]
                      ++ [ (i, map fromIntegral cps) | (i, cps) <- added ])
      vlookup    = mkLookup [ (map chr cps, i) | (i, cps) <- vocab ]
      mlookup    = mkLookup (zip (parseMerges mergesBody) [0 ..])  -- lazy: only forced if BPE runs
  return (Tokenizer decTbl vlookup mlookup)

-- byte-level encode one pre-token to its byte-level string.
toByteLevel :: String -> String
toByteLevel = concatMap (\c -> map (chr . byteToLevel) (utf8enc (ord c)))

-- BPE one pre-token (byte-level string) to ids. ignore_merges: a whole pre-token
-- in the vocab is emitted directly; otherwise merge by rank.
bpe :: (String -> Maybe Int) -> ((String, String) -> Maybe Int) -> String -> [Int]
bpe vlookup mlookup t =
  case vlookup t of
    Just i  -> [i]
    Nothing -> map (\sym -> maybe 0 id (vlookup sym)) (loop (map (: []) t))
  where
    loop syms =
      let ranked = [ (r, i) | (i, (a, b)) <- zip [0 ..] (zip syms (drop 1 syms))
                            , Just r <- [mlookup (a, b)] ]
      in if null ranked
           then syms
           else let (_, idx) = minimum ranked
                    a = syms !! idx
                    b = syms !! (idx + 1)
                in loop (take idx syms ++ [a ++ b] ++ drop (idx + 2) syms)

-- Encode raw text (no special tokens) to ids.
encodeText :: Tokenizer -> String -> [Int]
encodeText (Tokenizer _ vlookup mlookup) =
  concatMap (bpe vlookup mlookup . toByteLevel) . pretokenize

-- | Encode a user message with the BitNet chat template:
-- @User: {msg}<|eot_id|>Assistant: @ (the eot special token is id 128009).
encodeChat :: Tokenizer -> String -> [Int]
encodeChat tok msg =
  encodeText tok ("User: " ++ msg) ++ [128009] ++ encodeText tok "Assistant: "

-- | Decode token ids to bytes (the UTF-8 text).
decode :: Tokenizer -> [Int] -> [Word8]
decode (Tokenizer tbl _ _) = concatMap (tbl !)
