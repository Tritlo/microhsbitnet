-- | Test the Haskell tokenizer round-trip: encode a chat prompt and decode the
-- demo's generated ids, both against the known Python values.
module TestTok(main) where

import qualified Data.ByteString as BS
import Tokenizer (loadTokenizer, encodeChat, decode)

tokPath :: String
tokPath = "models/bitnet-b1.58-2B-4T/tokenizer.json"

main :: IO ()
main = do
  tok <- loadTokenizer tokPath
  let ids = encodeChat tok "Hello!"
  putStrLn ("encoded \"Hello!\": " ++ unwords (map show ids))
  putStrLn "  expect:          1502 25 22691 0 128009 72803 25 220"
  putStrLn "decoded generation:"
  BS.putStr (BS.pack (decode tok [9906, 0, 2650, 649, 358, 7945, 499, 3432, 30, 128009]))
  putStrLn ""
