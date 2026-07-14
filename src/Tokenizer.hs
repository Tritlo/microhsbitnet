-- | The qwen35 byte-level BPE tokenizer — encode (text -> ids) and decode
-- (ids -> model bytes). The pre-tokenizer regex, byte-level mapping and BPE
-- merge loop are Haskell; the vocab/merge/decode tables are built once in C
-- (`cbits/tokparse.c`) straight from the GGUF metadata arrays, and this module
-- calls the three table operations: vocab lookup, merge-rank lookup, decode.
--
-- The pre-tokenizer follows the qwen35 regex (the case-insensitive-contraction
-- GPT-2 variant, see llama-vocab.cpp):
--   '(s|t|re|ve|m|ll|d)  |  [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  |  \p{N}
--   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*  |  \s*[\r\n]+  |  \s+(?!\S)  |  \s+
-- with Unicode classes from Data.Char. BPE has no ignore_merges shortcut
-- (qwen35 leaves it off, unlike llama3).
module Tokenizer
  ( Tokenizer
  , loadTokenizer
  , encodeText
  , decodeBytes
  , tokenId
  , pretokenize
  ) where

import Data.Char (isSpace, isAlpha, isNumber, isMark, chr)
import Data.Word (Word8)
import qualified Data.ByteString as BS
import Data.ByteString.Internal (packUTF8)
import Data.ByteString.Unsafe (unsafeUseAsCStringLen)
import Foreign.C.String (CString)
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Ptr (Ptr, plusPtr, castPtr)
import Foreign.Storable (peekElemOff)
import System.IO.Unsafe (unsafePerformIO)
import Gguf (Gguf(..), GVal(..), lookupKV)

foreign import ccall "tokparse.h b1tok_load"
  c_tok_load :: CString -> Int -> CString -> CString -> Int -> IO Int
foreign import ccall "tokparse.h b1tok_vlookup"
  c_vlookup :: CString -> Int -> IO Int
foreign import ccall "tokparse.h b1tok_mlookup"
  c_mlookup :: CString -> Int -> CString -> Int -> IO Int
foreign import ccall "tokparse.h b1tok_decode"
  c_decode :: Int -> Ptr Word8 -> Int -> IO Int

-- ---------- byte-level alphabet (byte -> codepoint) ----------

nonPrintable :: [Int]
nonPrintable = [0 .. 32] ++ [127 .. 160] ++ [173]

byteToLevel :: Int -> Int
byteToLevel b | (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) = b
              | otherwise = 256 + length (takeWhile (/= b) nonPrintable)

-- ---------- C-backed table lookups (tables are immutable after load) ----------

vlookup :: String -> Maybe Int
vlookup s = unsafePerformIO $
  unsafeUseAsCStringLen (packUTF8 s) $ \ (p, n) -> do
    i <- c_vlookup p n
    return (if i < 0 then Nothing else Just i)

mlookup :: (String, String) -> Maybe Int
mlookup (a, b) = unsafePerformIO $
  unsafeUseAsCStringLen (packUTF8 a) $ \ (pa, na) ->
  unsafeUseAsCStringLen (packUTF8 b) $ \ (pb, nb) -> do
    r <- c_mlookup pa na pb nb
    return (if r < 0 then Nothing else Just r)

-- ---------- pre-tokenizer (the qwen35 regex) ----------

isL, isN, isM, isLM :: Char -> Bool
isL = isAlpha
isN = isNumber
isM = isMark
isLM c = isAlpha c || isMark c

isNL :: Char -> Bool
isNL c = c == '\r' || c == '\n'

-- | Split text into pre-tokens, alternatives tried in regex order.
pretokenize :: String -> [String]
pretokenize [] = []
pretokenize s  = let (tok, rest) = one s in tok : pretokenize rest
  where
    one t = case contraction t of
      Just r  -> r
      Nothing -> case letters t of
        Just r  -> r
        Nothing -> case number t of
          Just r  -> r
          Nothing -> case punct t of
            Just r  -> r
            Nothing -> case wsNewline t of
              Just r  -> r
              Nothing -> case wsTrail t of
                Just r  -> r
                Nothing -> case span isSpace t of      -- \s+ (or a lone stray char)
                  (w@(_ : _), r) -> (w, r)
                  ([], c : r)    -> ([c], r)
                  ([], [])       -> ([], [])

    -- '(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
    contraction ('\'' : c : r)
      | c `elem` "sStTmMdD" = Just (['\'', c], r)
    contraction ('\'' : c1 : c2 : r)
      | [l1, l2] <- [lower c1, lower c2]
      , [l1, l2] `elem` ["re", "ve", "ll"] = Just (['\'', c1, c2], r)
      where lower ch = if ch >= 'A' && ch <= 'Z' then chr (fromEnum ch + 32) else ch
    contraction _ = Nothing

    -- [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    letters (c : r)
      | not (isNL c) && not (isL c) && not (isN c)
      , (ls@(_ : _), r') <- span isLM r = Just (c : ls, r')
    letters t'
      | (ls@(_ : _), r') <- span isLM t' = Just (ls, r')
    letters _ = Nothing

    -- \p{N}
    number (c : r) | isN c = Just ([c], r)
    number _ = Nothing

    -- " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    isP c = not (isSpace c) && not (isL c) && not (isM c) && not (isN c)
    punct (' ' : r)
      | (ps@(_ : _), r') <- span isP r
      , (nl, r'') <- span isNL r' = Just (' ' : ps ++ nl, r'')
    punct t'
      | (ps@(_ : _), r') <- span isP t'
      , (nl, r'') <- span isNL r' = Just (ps ++ nl, r'')
    punct _ = Nothing

    -- \s*[\r\n]+ : the whitespace run up to (and including) its last newline
    wsNewline t' =
      let (ws, r) = span isSpace t'
          j = lastNL ws
      in if null ws || j < 0
           then Nothing
           else Just (take (j + 1) ws, drop (j + 1) ws ++ r)
      where
        lastNL ws = go (-1) 0 ws
          where go best _ []       = best
                go best i (c : cs) = go (if isNL c then i else best) (i + 1) cs

    -- \s+(?!\S) : whole run at end of text; else run minus its final char
    wsTrail t' =
      case span isSpace t' of
        ([], _)      -> Nothing
        (ws, [])     -> Just (ws, [])
        ([_], _)     -> Nothing                        -- single ws before \S: fails
        (ws, r)      -> Just (init ws, [last ws] ++ r)

-- ---------- the tokenizer ----------

-- | A marker: the vocab/merge/decode tables live in C (one set per process),
-- so 'loadTokenizer' must run before any encode/decode.
data Tokenizer = Tokenizer

-- | Build the C tables from the GGUF's tokenizer arrays.
loadTokenizer :: Gguf -> IO Tokenizer
loadTokenizer g = do
  let arr k = case lookupKV k g of
        Just (GArr _ n o) -> (n, o)
        _                 -> error ("tokenizer: missing " ++ k)
      (ntok, otok) = arr "tokenizer.ggml.tokens"
      (ntt,  ott)  = arr "tokenizer.ggml.token_type"
      (nmrg, omrg) = arr "tokenizer.ggml.merges"
  if ntt /= ntok then error "tokenizer: token_type/tokens length mismatch" else return ()
  _ <- unsafeUseAsCStringLen (gBytes g) $ \ (p, _) ->
         c_tok_load (p `plusPtr` otok) ntok (p `plusPtr` ott) (p `plusPtr` omrg) nmrg
  return Tokenizer

-- byte-level encode one pre-token: its UTF-8 bytes mapped through the alphabet.
toByteLevel :: String -> String
toByteLevel s = map (chr . byteToLevel . fromIntegral) (BS.unpack (packUTF8 s))

-- BPE one pre-token (byte-level string) to ids: merge by rank (no
-- ignore_merges), leftmost-lowest-rank first; then vocab-lookup each symbol.
bpe :: String -> [Int]
bpe t = map (\ sym -> maybe 0 id (vlookup sym)) (loop (map (: []) t))
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

-- | Encode raw text (no special-token parsing) to ids.
encodeText :: Tokenizer -> String -> [Int]
encodeText _ = concatMap (bpe . toByteLevel) . pretokenize

-- | The id of an exact vocab string (e.g. a control token like "<|im_start|>").
tokenId :: String -> Maybe Int
tokenId = vlookup

-- | Decode ids to model bytes (UTF-8 of the text).
decodeBytes :: Tokenizer -> [Int] -> BS.ByteString
decodeBytes _ ids = BS.concat (map decodeOne ids)
  where
    decodeOne i = unsafePerformIO $
      allocaBytes 256 $ \ p -> do
        n <- c_decode i p 256
        BS.packCStringLen (castPtr p, min n 256)
