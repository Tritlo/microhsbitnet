{-# LANGUAGE GADTs #-}

module Main (main) where

import Arrays
import BufferChecks (bufferCheckSource)
import CCC
import Compiler
import Control.Exception (ArithException (Overflow), SomeException, evaluate, try)
import Control.Monad (forM, unless)
import Data.Bits (shiftL, (.|.))
import qualified Data.ByteString as BS
import Data.ByteString.Builder (floatLE, int32LE, toLazyByteString)
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int32, Int8)
import Frontend (checkedModule)
import HeadArrays (headSource)
import LayerArrays (layerSource)
import System.Directory (createDirectoryIfMissing)
import System.Process (readProcess)

-- | Compile the selected source, check its interpreter, and generate GPU inputs.
main :: IO ()
main = do
    checked <- checkedModule "spike/ccc/Programs.hs"
    scalar <- either fail pure (compileScalar checked "scalar")
    contribution <- either fail pure (compileIntPair checked "contribution")
    combine <- either fail pure (compileIntPair checked "combine")
    peak <- either fail pure (compileFloatPair checked "peak")
    quantize <- either fail pure (compileQuantize checked "quantize")
    rescale <- either fail pure (compileRescale checked "rescale")
    plan <- either fail pure (packedMatVec contribution combine)
    shader <- either fail pure (metalSource scalar plan)
    webShader <- either fail pure (wgslSource scalar plan)
    bitLinearShader <- either fail pure (bitLinearSource peak quantize rescale plan)
    layerChecked <- checkedModule "spike/ccc/LayerPrograms.hs"
    layerShader <- either fail pure (layerSource layerChecked)
    headShader <- either fail pure (headSource layerChecked)
    frequency <- either fail pure (compileFloatPair layerChecked "frequency")
    multiply <- either fail pure (compileFloatPair layerChecked "product")
    cosine <- either fail pure (compileScalar layerChecked "cosine")
    sine <- either fail pure (compileScalar layerChecked "sine")
    frequencyCode <- either fail pure (emitC frequency (PairCode (FloatCode "(float)component") (FloatCode "128.0f")))
    angleCode <- either fail pure (emitC multiply (PairCode (FloatCode "(float)position") (FloatCode "frequency")))
    cosineCode <- either fail pure (emitC cosine (FloatCode "angle"))
    sineCode <- either fail pure (emitC sine (FloatCode "angle"))
    let FloatCode nativeFrequency = frequencyCode
        FloatCode nativeAngle = angleCode
        FloatCode nativeCosine = cosineCode
        FloatCode nativeSine = sineCode
    let directory = "bin/ccc-spike/cases/"
        inputs = [-32, -3.5, -1, -0.5, 0, 0.5, 1, 3.5, 32]
        expectedScalar = map (runCat scalar) inputs
        expectedContributions = [runCat contribution (field, activation) | field <- [0 .. 2], activation <- [-128, -1, 0, 1, 127]]
    sourceResults <- lines <$> readProcess "bin/ccc-spike/source" [] ""
    case sourceResults of
        [scalarResults, contributionResults, "True", "True", peakResults, quantizeResults, rescaleResults] -> do
            unless ((read scalarResults :: [Float]) == expectedScalar) $ fail "Stock MicroHs and categorical scalar results differ"
            unless ((read contributionResults :: [Int32]) == expectedContributions) $ fail "Stock MicroHs and categorical contribution results differ"
            unless ((read peakResults :: [Float]) == map (runCat peak) [(-8, 4), (0, 0.0000001)]) $ fail "Stock MicroHs and categorical maxima differ"
            unless ((read quantizeResults :: [Int32]) == [runCat quantize (value, 127) | value <- [-127, -126.5, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 126.5, 127]]) $ fail "Stock MicroHs and categorical quantization differ"
            unless ((read rescaleResults :: [Float]) == map (runCat rescale) [(1024, (0.125, 127)), (-17, (0.5, 0))]) $ fail "Stock MicroHs and categorical rescaling differ"
        _ -> fail "Unexpected stock MicroHs reference output"
    unless (runCat (Compose Apply (Pair (Compose (Curry AddInt) First) Second)) (7, 4) == (11 :: Int32)) $
        fail "Categorical curry/apply check failed"
    addition <- try (evaluate (runCat combine (maxBound, 1))) :: IO (Either ArithException Int32)
    negation <- try (evaluate (runCat contribution (0, minBound))) :: IO (Either ArithException Int32)
    unless (case (addition, negation) of (Left Overflow, Left Overflow) -> True; _ -> False) $
        fail "Categorical Int32 overflow behavior differs from stock MicroHs"
    negativeChecks
    createDirectoryIfMissing True directory
    writeFile "bin/ccc-spike/generated.metal" shader
    writeFile "bin/ccc-spike/generated.wgsl" webShader
    writeFile "bin/ccc-spike/bitlinear.wgsl" bitLinearShader
    writeFile "bin/ccc-spike/layer.wgsl" layerShader
    writeFile "bin/ccc-spike/head.wgsl" headShader
    writeFile "bin/ccc-spike/checks.wgsl" bufferCheckSource
    writeFile "bin/ccc-spike/rotary.c" $
        unlines
            [ "#include <fenv.h>"
            , "#include <math.h>"
            , "#include <stdint.h>"
            , "#include <stdio.h>"
            , "#include <string.h>"
            , "int main(int argc, char **argv) {"
            , "  if (argc != 2 || fesetround(FE_TONEAREST) != 0) return 1;"
            , "  FILE *file = fopen(argv[1], \"wb\");"
            , "  if (!file) return 1;"
            , "  for (int position = 0; position < 2048; position++) {"
            , "    for (int component = 0; component < 64; component++) {"
            , "      float frequency = " ++ nativeFrequency ++ ";"
            , "      float angle = " ++ nativeAngle ++ ";"
            , "      float pair[] = {" ++ nativeCosine ++ ", " ++ nativeSine ++ "};"
            , "      for (int index = 0; index < 2; index++) {"
            , "        if (!isfinite(pair[index])) return 1;"
            , "        uint32_t bits; memcpy(&bits, &pair[index], 4);"
            , "        unsigned char bytes[] = {bits, bits >> 8, bits >> 16, bits >> 24};"
            , "        if (fwrite(bytes, 1, 4, file) != 4) return 1;"
            , "      }"
            , "    }"
            , "  }"
            , "  return fclose(file) != 0;"
            , "}"
            ]
    _ <- readProcess "clang" ["-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off", "bin/ccc-spike/rotary.c", "-lm", "-o", "bin/ccc-spike/rotary-constants"] ""
    _ <- readProcess "bin/ccc-spike/rotary-constants" ["bin/ccc-spike/rotary.bin"] ""
    writeFile "bin/ccc-spike/graphs.txt" $
        unlines
            [ "scalar = " ++ showCat scalar
            , "contribution = " ++ showCat contribution
            , "combine = " ++ showCat combine
            , "array = packedRows (fold addInt32 0 (map contribution))"
            ]
    BL.writeFile (directory ++ "scalar-input.bin") (toLazyByteString (foldMap floatLE inputs))
    BL.writeFile (directory ++ "scalar-expected.bin") (toLazyByteString (foldMap floatLE expectedScalar))
    realCases <- lines <$> readFile (directory ++ "real-cases.txt")
    let synthetic = [("tiny", 2, 4), ("tail", 131, 12)]
    mapM_ (writeSynthetic directory) synthetic
    let descriptions = [unwords [name, show columns, show rows] | (name, columns, rows) <- synthetic] ++ realCases
    summaries <- forM descriptions $ \description -> case words description of
        [name, columnsText, rowsText] -> do
            let columns = read columnsText
                rows = read rowsText
                activations = BS.pack [fromIntegral (fromIntegral ((column * 73 + 19) `mod` 256 - 128) :: Int8) | column <- [0 .. columns - 1]]
            weights <- BS.readFile (directory ++ name ++ ".weights.bin")
            expected <- either fail pure (runArray plan columns rows weights activations)
            BL.writeFile (directory ++ name ++ ".expected.bin") (toLazyByteString (foldMap int32LE expected))
            BS.writeFile (directory ++ name ++ ".acts.bin") activations
            putStrLn (name ++ ": categorical CPU reference, " ++ show columns ++ " x " ++ show rows ++ " PASS")
            pure description
        _ -> fail "Invalid matrix fixture description"
    writeFile (directory ++ "cases.txt") (unlines summaries)
    putStrLn "Stock MicroHs / categorical CPU scalar and contribution parity PASS"
    putStrLn "CCC curry/apply and negative frontend checks PASS"
    putStrLn "Generated Metal, WGSL, and graphs.txt"

-- | Create packed matrices that exercise all fields and a partial thread group.
writeSynthetic :: FilePath -> (String, Int, Int) -> IO ()
writeSynthetic directory (name, columns, rows) = do
    let weights =
            BS.pack
                [ foldr (.|.) 0 [fromIntegral ((column + row + field) `mod` 3) `shiftL` (2 * field) | field <- [0 .. 3]]
                | row <- [0 .. rows `div` 4 - 1]
                , column <- [0 .. columns - 1]
                ]
    BS.writeFile (directory ++ name ++ ".weights.bin") weights

-- | Verify that unsupported source and invalid reductions fail explicitly.
negativeChecks :: IO ()
negativeChecks = do
    case emitWGSL (FloatConstant (1 / 0)) (FloatCode "input") of
        Left _ -> pure ()
        Right _ -> fail "A nonfinite GPU constant was accepted"
    unsupported <- checkedModule "spike/ccc/Unsupported.hs"
    case compileScalar unsupported "scalar" of
        Left _ -> pure ()
        Right _ -> fail "Unsupported primitive was accepted"
    case compileScalar unsupported "shadowed" of
        Left _ -> pure ()
        Right _ -> fail "Unsupported source closure was accepted"
    case packedMatVec (Compose AddInt (Pair First Second)) (Compose AddInt (Pair Second First)) of
        Left _ -> pure ()
        Right _ -> fail "Unproved reduction was accepted"
    unsafePlan <- either fail pure (packedMatVec (IntConstant maxBound) (Compose AddInt (Pair First Second)))
    case runArray unsafePlan 2 4 (BS.pack [85, 85]) (BS.pack [0, 0]) of
        Left _ -> pure ()
        Right _ -> fail "A reduction without an overflow proof was accepted"
    malformed <- try (checkedModule "spike/ccc/IllTyped.hs" >> pure ()) :: IO (Either SomeException ())
    case malformed of
        Left _ -> pure ()
        Right _ -> fail "Ill-typed source was accepted"
