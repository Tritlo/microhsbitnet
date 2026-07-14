-- | Tokenizer test driver: encode each case in test/tok-cases.txt (one per
-- line, backslash-escaped \n \r \t \\) and print the ids; verify the decode
-- round-trip. Compared with the fork's llama-tokenize by test/oracle_m2.py.
module TestTok(main) where

import qualified Data.ByteString as BS
import Data.ByteString.Internal (packUTF8)
import Loader (loadModel)
import Tokenizer

modelPath :: String
modelPath = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

unescape :: String -> String
unescape ('\\' : 'n' : r)  = '\n' : unescape r
unescape ('\\' : 'r' : r)  = '\r' : unescape r
unescape ('\\' : 't' : r)  = '\t' : unescape r
unescape ('\\' : '\\' : r) = '\\' : unescape r
unescape (c : r)           = c : unescape r
unescape []                = []

main :: IO ()
main = do
  g <- loadModel modelPath
  tok <- loadTokenizer g
  cases <- fmap lines (readFile "test/tok-cases.txt")
  mapM_ (runCase tok) (filter (not . null) cases)
  where
    runCase tok raw = do
      let t   = unescape raw
          ids = encodeText tok t
          rt  = decodeBytes tok ids == packUTF8 t
      putStrLn (show ids ++ (if rt then "" else "  ROUNDTRIP-FAIL"))
