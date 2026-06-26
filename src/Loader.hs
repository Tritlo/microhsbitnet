-- | Shared model loading: read a safetensors file's header and full data section
-- into one resident ByteString, and slice tensors out of it.
--
-- Works around MicroHs base bugs: builds with @gmhs@; never touches
-- @BS.empty@\/@concat@\/@readFile@ (all hang); reads with fixed-count @hGet@ and
-- combines chunks by seeding with the first read and appending non-empty chunks.
module Loader
  ( loadModel
  , slice
  ) where

import qualified Data.ByteString as BS
import System.IO (openBinaryFile, hClose, IOMode(ReadMode), Handle)
import Safetensors (Tensor(..), parseHeader)

chunkMax :: Int
chunkMax = 1536 * 1024 * 1024

leInt :: BS.ByteString -> Int -> Int
leInt bs n = go 0 0
  where
    go i acc | i >= n    = acc
             | otherwise = go (i + 1) (acc + fromIntegral (BS.index bs i) * (256 ^ i))

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

-- | Load a safetensors file: returns its tensor table and the full data section
-- as one resident ByteString (off the combinator heap).
loadModel :: String -> IO ([Tensor], BS.ByteString)
loadModel path = do
  h <- openBinaryFile path ReadMode
  lenBytes <- BS.hGet h 8
  hdrBytes <- hGetN h (leInt lenBytes 8)
  let ts = parseHeader (map (toEnum . fromIntegral) (BS.unpack hdrBytes) :: String)
      totalData = foldr1 max (map tEnd ts)
  dat <- hGetN h totalData
  hClose h
  return (ts, dat)

-- | A tensor's raw bytes as an O(1) view into the resident data section.
slice :: BS.ByteString -> Tensor -> BS.ByteString
slice dat t = BS.take (tEnd t - tBegin t) (BS.drop (tBegin t) dat)
