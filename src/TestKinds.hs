-- | Unit test for the bndot C kernel: each of its three dots against a
-- hand-computed value on small synthetic vectors — ternary BitLinears, the bf16
-- LM head, and float32 attention q·k, which is all the kernel implements.
module TestKinds(main) where

import qualified Data.ByteString as BS
import Dot (ternaryDot, bf16Dot, f32Dot, packF32)

-- Raw bytes / int8 two's-complement bytes from Ints.
bytes :: [Int] -> BS.ByteString
bytes = BS.pack . map fromIntegral

check :: String -> Double -> Double -> IO ()
check name got want =
  putStrLn (name ++ ": got " ++ show got ++ " want " ++ show want
             ++ (if abs (got - want) < 1.0e-4 then "   OK" else "   ** FAIL **"))

main :: IO ()
main = do
  -- ternary x int8. field0 of [2,0,1] = trits [1,-1,0]; acts [10,20,30].
  check "ternary x int8  " (ternaryDot (bytes [2, 0, 1]) (bytes [10, 20, 30]) 0 3) (-10)
  -- bf16 x f32. bf16 1.0=0x3F80, 2.0=0x4000 (LE bytes); acts [3,4].
  check "bf16    x f32   " (bf16Dot (bytes [128, 63, 0, 64]) (packF32 [3, 4]) 2) 11
  -- f32 x f32.
  check "f32     x f32   " (f32Dot (packF32 [1.5, 2.0]) (packF32 [4.0, 0.5]) 2) 7
  -- ternary with a nonzero shift (the packed-row field selector): byte 0x65 =
  -- 0b01_10_01_01, so field2 (shift 4) = 0b10 = 2 -> trit +1; act 50 -> +50.
  check "ternary shift=4 " (ternaryDot (bytes [101]) (bytes [50]) 4 1) 50
