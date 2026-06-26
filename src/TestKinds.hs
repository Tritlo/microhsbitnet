-- | Unit test for the bndot runtime primitive: each kind against a hand-computed
-- value on small synthetic vectors. (The model only exercises kinds 0/1/2, so the
-- quantized kinds need their own check.) Build with the vendored runtime and run.
module TestKinds(main) where

import qualified Data.ByteString as BS
import Dot (bnDot, packF32)

-- Raw bytes / int8 two's-complement bytes from Ints.
bytes :: [Int] -> BS.ByteString
bytes = BS.pack . map fromIntegral

check :: String -> Double -> Double -> IO ()
check name got want =
  putStrLn (name ++ ": got " ++ show got ++ " want " ++ show want
             ++ (if abs (got - want) < 1.0e-4 then "   OK" else "   ** FAIL **"))

main :: IO ()
main = do
  -- kind 0: W1.58 ternary x int8. field0 of [2,0,1] = trits [1,-1,0]; acts [10,20,30].
  check "k0 ternary x int8" (bnDot 0 (bytes [2, 0, 1]) (bytes [10, 20, 30]) 0 3) (-10)
  -- kind 1: bf16 x f32. bf16 1.0=0x3F80, 2.0=0x4000 (LE bytes); acts [3,4].
  check "k1 bf16    x f32 " (bnDot 1 (bytes [128, 63, 0, 64]) (packF32 [3, 4]) 0 2) 11
  -- kind 2: f32 x f32.
  check "k2 f32     x f32 " (bnDot 2 (packF32 [1.5, 2.0]) (packF32 [4.0, 0.5]) 0 2) 7
  -- kind 3: int8 x int8.  2*10 + (-3)*20 + 5*(-4) = -60.
  check "k3 int8    x int8" (bnDot 3 (bytes [2, -3, 5]) (bytes [10, 20, -4]) 0 3) (-60)
  -- kind 4: int8 x f32.   2*1.5 + (-3)*2 + 5*(-1) = -8.
  check "k4 int8    x f32 " (bnDot 4 (bytes [2, -3, 5]) (packF32 [1.5, 2.0, -1.0]) 0 3) (-8)
  -- kind 5: int4 x int8. weights [1,-2,3,-8] packed low-nibble-first = [0xE1,0x83];
  --         acts [10,1,2,1]: 1*10 + (-2)*1 + 3*2 + (-8)*1 = 6.
  check "k5 int4    x int8" (bnDot 5 (bytes [225, 131]) (bytes [10, 1, 2, 1]) 0 4) 6
  -- kind 6: int4 x f32.  1*2 + (-2)*1 + 3*0.5 + (-8)*1 = -6.5.
  check "k6 int4    x f32 " (bnDot 6 (bytes [225, 131]) (packF32 [2, 1, 0.5, 1]) 0 4) (-6.5)
  -- kind 7: fp16 x f32. fp16 1.0=0x3C00, 2.0=0x4000, -0.5=0xB800; acts [4,1,2]:
  --         1*4 + 2*1 + (-0.5)*2 = 5.
  check "k7 fp16    x f32 " (bnDot 7 (bytes [0, 60, 0, 64, 0, 184]) (packF32 [4, 1, 2]) 0 3) 5
  -- kind 8: fp8 E4M3 x f32. e4m3 1.0=0x38, 1.5=0x3C, -2.0=0xC0; acts [2,4,1]:
  --         1*2 + 1.5*4 + (-2)*1 = 6.
  check "k8 e4m3    x f32 " (bnDot 8 (bytes [56, 60, 192]) (packF32 [2, 4, 1]) 0 3) 6
