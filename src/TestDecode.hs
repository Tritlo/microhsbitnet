-- | Test the Haskell decoder: turn the demo's generated token ids back into text.
module TestDecode(main) where

import qualified Data.ByteString as BS
import Tokenizer (loadDecoder, decode)

tokenizerPath :: String
tokenizerPath = "models/bitnet-b1.58-2B-4T/tokenizer.json"

-- "Hello! How can I assist you today?<|eot_id|>" (the M4 output ids).
demoIds :: [Int]
demoIds = [9906, 0, 2650, 649, 358, 7945, 499, 3432, 30, 128009]

main :: IO ()
main = do
  tbl <- loadDecoder tokenizerPath
  putStrLn "decoded text:"
  BS.putStr (BS.pack (decode tbl demoIds))
  putStrLn ""
