-- | A byte-level BPE tokenizer (Llama 3 style) — no Python. Encode (text -> ids)
-- and decode (ids -> text). The pre-tokenizer, byte-level mapping and BPE merge
-- loop are Haskell; the heavy lifting of parsing tokenizer.json (16 MB) and the
-- vocab/merge/decode tables is done once in C (`cbits/tokparse.c`), which this
-- module calls for the three table operations: vocab lookup, merge-rank lookup,
-- and id->bytes decode. Lookup keys are the canonical UTF-8 of a byte-level
-- string (`packUTF8`), exactly what the C tables are keyed by.
module Tokenizer
  ( Tokenizer
  , loadTokenizer
  , Role(..)
  , encodeChat
  , encodeChatMessages
  , decode
  ) where

import Data.Char (isSpace, isDigit, isAlpha, chr)
import Data.Word (Word8)
import qualified Data.ByteString as BS
import Data.ByteString.Internal (packUTF8)
import Data.ByteString.Unsafe (unsafeUseAsCStringLen)
import Foreign.C.String (CString)
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Ptr (Ptr)
import Foreign.Storable (peekElemOff)
import System.IO.Unsafe (unsafePerformIO)
import System.IO (openBinaryFile, hClose, IOMode(ReadMode))

foreign import ccall "tokparse.h bntok_load"
  c_tok_load :: CString -> Int -> IO Int
foreign import ccall "tokparse.h bntok_vlookup"
  c_vlookup :: CString -> Int -> IO Int
foreign import ccall "tokparse.h bntok_mlookup"
  c_mlookup :: CString -> Int -> CString -> Int -> IO Int
foreign import ccall "tokparse.h bntok_decode"
  c_decode :: Int -> Ptr Word8 -> Int -> IO Int

-- ---------- byte-level alphabet (computed; byte -> codepoint) ----------

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

-- | A marker: the vocab/merge/decode tables live in C (one set per process), so
-- 'loadTokenizer' must run before any encode/decode.
data Tokenizer = Tokenizer

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

-- | Read tokenizer.json and parse it (in C) into the vocab/merge/decode tables.
loadTokenizer :: String -> IO Tokenizer
loadTokenizer path = do
  bs <- readFileBS path
  _  <- unsafeUseAsCStringLen bs $ \ (p, n) -> c_tok_load p n
  return Tokenizer

-- byte-level encode one pre-token to its byte-level string: its UTF-8 bytes
-- (via 'packUTF8') mapped through the GPT-2 byte->codepoint alphabet.
toByteLevel :: String -> String
toByteLevel s = map (chr . byteToLevel . fromIntegral) (BS.unpack (packUTF8 s))

-- BPE one pre-token (byte-level string) to ids. ignore_merges: a whole pre-token
-- in the vocab is emitted directly; otherwise merge by rank.
bpe :: String -> [Int]
bpe t =
  case vlookup t of
    Just i  -> [i]
    Nothing -> map (\ sym -> maybe 0 id (vlookup sym)) (loop (map (: []) t))
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
encodeText _ = concatMap (bpe . toByteLevel) . pretokenize

-- | A chat role. Rendered capitalized in the template ("User"/"Assistant"/"System").
data Role = System | User | Assistant
  deriving (Eq, Show)

roleStr :: Role -> String
roleStr System    = "System"
roleStr User      = "User"
roleStr Assistant = "Assistant"

trim :: String -> String
trim = f . f where f = reverse . dropWhile isSpace

-- | Encode a conversation with the BitNet chat template: each message becomes
-- @{Role}: {trimmed content}<|eot_id|>@ (eot = id 128009), then @Assistant: @
-- prompts the next turn.
encodeChatMessages :: Tokenizer -> [(Role, String)] -> [Int]
encodeChatMessages tok msgs =
  concatMap one msgs ++ encodeText tok "Assistant: "
  where one (r, c) = encodeText tok (roleStr r ++ ": " ++ trim c) ++ [128009]

-- | Encode a single user message (the common one-turn case).
encodeChat :: Tokenizer -> String -> [Int]
encodeChat tok msg = encodeChatMessages tok [(User, msg)]

-- | Decode token ids to bytes (the UTF-8 text), via the C decode table.
decode :: Tokenizer -> [Int] -> [Word8]
decode _ = concatMap decId
  where
    decId i = unsafePerformIO $ allocaBytes 512 $ \ p -> do
      n <- c_decode i (p :: Ptr Word8) 512
      mapM (peekElemOff p) [0 .. min n 512 - 1]
