-- | Unit test for the bndot C kernels: the dot kernels (ternary matmul, bf16
-- argmax, f32 dot) on hand-computed cases, and the generation-path vector math
-- (rmsnorm, rope, fused bitlinear, residual add, squared-ReLU gate) cross-checked
-- against their Haskell counterparts in Model. The full paths run against the
-- oracles in M2/M3 and the M4 text.
module TestKinds(main) where

import qualified Data.ByteString as BS
import Dot (matmulTernary, argmaxBf16, f32Dot, packF32, unpackF32,
            bnRmsNorm, bnRope, bnBitLinearF, bnAdd, bnSqReluMul)
import Model (Lin(..), applyLin, rmsNorm, applyRope, cHeadDim, cTheta, cEps)

bytes :: [Int] -> BS.ByteString
bytes = BS.pack . map fromIntegral

check :: String -> Double -> Double -> IO ()
check name got want =
  putStrLn (name ++ ": got " ++ show got ++ " want " ++ show want
             ++ (if abs (got - want) < 1.0e-4 then "   OK" else "   ** FAIL **"))

-- Elementwise closeness of a C-kernel result (unpacked) vs a Haskell reference.
closeL :: String -> [Float] -> [Float] -> IO ()
closeL name got want =
  let d = maximum (0 : zipWith (\a b -> abs (a - b)) got want)
      okLen = length got == length want
  in putStrLn (name ++ ": maxdiff " ++ show d
                ++ (if okLen && d < 1.0e-3 then "   OK" else "   ** FAIL **"))

main :: IO ()
main = do
  -- ternary matmul, outDim=4 x inDim=2. Packed [134,24], acts [10,7] -> [3,7,-10,3].
  let m = matmulTernary (bytes [134, 24]) (bytes [10, 7]) 2 4
  check "matmul row0" (m !! 0) 3
  check "matmul row1" (m !! 1) 7
  check "matmul row2" (m !! 2) (-10)
  check "matmul row3" (m !! 3) 3
  -- f32 x f32 dot: 1.5*4 + 2*0.5 = 7.
  check "f32 dot     " (f32Dot (packF32 [1.5, 2.0]) (packF32 [4.0, 0.5]) 2) 7
  -- bf16 argmax over 3 vocab rows, hidden 2, hidden vec [3,4]. logits [3,10,3.5].
  let (bi, bv) = argmaxBf16 (bytes [128, 63, 0, 0, 0, 64, 128, 63, 0, 63, 0, 63]) [3.0, 4.0] 2 3
  check "argmax idx  " (fromIntegral bi) 1
  check "argmax val  " (realToFrac bv) 10

  -- generation-path kernels vs the Haskell reference in Model:
  let w4 = [0.5, 1.0, 1.5, 2.0]; x4 = [1.0, -2.0, 3.0, -4.0]
  closeL "rmsnorm  " (unpackF32 (bnRmsNorm (packF32 w4) (packF32 x4) 4 cEps)) (rmsNorm w4 x4)
  let v = [fromIntegral i * 0.01 | i <- [1 .. cHeadDim]]
  closeL "rope     " (unpackF32 (bnRope (packF32 v) 3 1 cHeadDim cTheta)) (applyRope 3 v)
  let lin = Lin (bytes [134, 24]) 2 4 1.0
  closeL "bitlinear" (unpackF32 (bnBitLinearF (bytes [134, 24]) 1.0 (packF32 [10.0, 7.0]) 2 4))
                     (applyLin lin [10.0, 7.0])
  closeL "add      " (unpackF32 (bnAdd (packF32 [1, 2, 3]) (packF32 [10, 20, 30]) 3)) [11, 22, 33]
  closeL "sqrelu   " (unpackF32 (bnSqReluMul (packF32 [2, -1, 3]) (packF32 [5, 5, 5]) 3)) [20, 0, 45]
