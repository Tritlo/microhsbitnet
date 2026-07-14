-- | Shared model loading: read the whole GGUF file into one resident
-- ByteString and parse its header (KVs + tensor table).
--
-- Built with @gmhs@. Read sequentially with fixed-count @hGet@ because MicroHs
-- has no @hSeek@\/@hFileSize@ for random access; @hGetN@ accumulates bounded
-- chunks so a single read needn't size the whole file. The file (~3.8 GB) stays
-- resident off the combinator heap; tensors are O(1) slices into it.
module Loader
  ( loadModel
  , slice
  ) where

import qualified Data.ByteString as BS
import System.IO (openBinaryFile, hClose, IOMode(ReadMode), Handle)
import Gguf (Gguf(..), GTensor(..), parseGguf, tensorBytes)

chunkMax :: Int
chunkMax = 1536 * 1024 * 1024

hGetN :: Handle -> Int -> IO BS.ByteString
hGetN h n = do
  first <- BS.hGet h (min n chunkMax)
  go (n - BS.length first) first
  where
    go k acc | k <= 0 = return acc
             | otherwise = do
                 chunk <- BS.hGet h (min k chunkMax)
                 let got = BS.length chunk
                 if got == 0 then return acc
                             else go (k - got) (BS.append acc chunk)

-- | Load a GGUF file resident and parse it. maxBytes bounds the read (the
-- whole file fits; pass a generous cap).
loadModel :: String -> IO Gguf
loadModel path = do
  h <- openBinaryFile path ReadMode
  bs <- hGetN h maxBytes
  hClose h
  return (parseGguf bs)
  where
    maxBytes = 8 * 1024 * 1024 * 1024

-- | A tensor's raw bytes as an O(1) view into the resident file.
slice :: Gguf -> GTensor -> BS.ByteString
slice g t = BS.take (tensorBytes t) (BS.drop (gtOff t) (gBytes g))
