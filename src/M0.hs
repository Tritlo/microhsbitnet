-- | M0 milestone driver: load the full safetensors data section into a resident
-- ByteString and verify bf16 decode + ternary unpack on real tensors.
module M0(main) where

import Data.Word (Word8, Word16)
import qualified Data.ByteString as BS
import System.IO (openBinaryFile, hClose, IOMode(ReadMode), Handle)
import Safetensors

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

leInt :: BS.ByteString -> Int -> Int
leInt bs n = go 0 0
  where
    go i acc | i >= n    = acc
             | otherwise = go (i + 1) (acc + fromIntegral (BS.index bs i) * (256 ^ i))

-- Read exactly n bytes (n > 0). Avoids BS.empty/concat (both broken in MicroHs):
-- seeds with the first read, appends only non-empty chunks. Large chunkMax keeps
-- the big data-section read to one or two syscalls (minimal O(n^2) append cost).
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

-- Raw bytes of a tensor, sliced out of the resident data section (O(1) views).
slice :: BS.ByteString -> Tensor -> BS.ByteString
slice dat t = BS.take (tEnd t - tBegin t) (BS.drop (tBegin t) dat)

main :: IO ()
main = do
  h <- openBinaryFile modelPath ReadMode
  lenBytes <- BS.hGet h 8
  let hlen = leInt lenBytes 8
  hdrBytes <- hGetN h hlen
  let ts = parseHeader (map (toEnum . fromIntegral) (BS.unpack hdrBytes) :: String)
  putStrLn ("tensors: " ++ show (length ts))
  let totalData = foldr1 max (map tEnd ts)
  putStrLn ("data section bytes: " ++ show totalData)
  dat <- hGetN h totalData
  putStrLn ("resident bytes read: " ++ show (BS.length dat))
  hClose h

  -- (a) bf16: embed_tokens row 0, first 5 values
  case lookupTensor "model.embed_tokens.weight" ts of
    Just t  -> do
      let b = slice dat t
      putStrLn ("embed[0,0..4]: " ++ show [ bf16At b (2 * c) | c <- [0 .. 4] ])
    Nothing -> putStrLn "embed: MISSING"

  -- (b) bf16: a weight scale (scalar)
  case lookupTensor "model.layers.0.self_attn.q_proj.weight_scale" ts of
    Just t  -> putStrLn ("q_proj.weight_scale: " ++ show (bf16At (slice dat t) 0))
    Nothing -> putStrLn "scale: MISSING"

  -- (c) U8 unpack: q_proj weight, logical [outDim,inDim]; first 16 trits of row 0,
  --     and check the whole row 0 is ternary (all in {-1,0,1}).
  case lookupTensor "model.layers.0.self_attn.q_proj.weight" ts of
    Just t  -> do
      let b = slice dat t
          [packedRows, inDim] = tShape t
          outDim = packedRows * 4
          row0 = unpackRow b inDim outDim 0
      putStrLn ("q_proj trits row0[0..15]: " ++ show (take 16 row0))
      putStrLn ("q_proj row0 all ternary: " ++ show (all (\v -> v >= -1 && v <= 1) row0)
                 ++ "  (len " ++ show (length row0) ++ ")")
      putStrLn ("q_proj trits row700[0..15]: " ++ show (take 16 (unpackRow b inDim outDim 700)))
    Nothing -> putStrLn "q_proj: MISSING"
