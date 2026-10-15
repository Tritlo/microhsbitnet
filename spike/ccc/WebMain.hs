module WebMain where

import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Int (Int32)
import Data.List (foldl', isPrefixOf)
import Foreign.C.String (withCString)
import LayerProgram ()
import NetworkProgram ()
import qualified Programs
import System.IO.Unsafe (unsafePerformIO)
import WebFFI (gpuReferenceAt, gpuResultAt, gpuStart)

foreign export javascript "gpuMain" main :: IO ()
foreign export javascript gpuComplete :: Int -> Int -> IO Int
foreign export javascript gpuStatus :: IO Int

-- | Keep the expected shape and remaining jobs until the host completes.
data Progress
    = Ready
    | Waiting Int String Int [(String, Int)]
    | Finished
    | Failed

-- | Retain driver state between calls from JavaScript.
progress :: IORef Progress
progress = unsafePerformIO (newIORef Ready)
{-# NOINLINE progress #-}

-- | Run the scalar, synthetic, and real matrix fixtures in order.
jobs :: [(String, Int)]
jobs =
    [ ("scalar", 9)
    , ("tiny", 4)
    , ("tail", 12)
    , ("q-proj", 2560)
    , ("down-proj", 2560)
    , ("bl-tiny", 4)
    , ("bl-tail", 12)
    , ("bl-q-proj", 2560)
    , ("bl-down-proj", 2560)
    , ("bl-zero", 12)
    , ("bl-nearzero", 12)
    , ("bl-rounding", 12)
    , ("bl-nondyadic", 12)
    , ("layer0-pos0", 2560)
    , ("layer0-pos1", 2560)
    , ("layer0-pos2", 2560)
    , ("network", 0)
    ]

-- | Interpret the existing synthetic packed matrices with the source functions.
packedExpected :: Int -> Int -> [Int32]
packedExpected columns rows =
    [ foldl'
        Programs.combine
        0
        [ Programs.contribution
            (fromIntegral ((column + row `mod` chunk + row `div` chunk) `mod` 3))
            (fromIntegral ((column * 73 + 19) `mod` 256 - 128))
        | column <- [0 .. columns - 1]
        ]
    | row <- [0 .. rows - 1]
    ]
  where
    chunk = rows `div` 4

-- | Compare fixture values with source references. Check network token validity.
checkResults :: String -> Int -> IO Bool
checkResults name count = do
    actual <- mapM gpuResultAt [0 .. count - 1]
    if name == "network"
        then pure (all (\token -> token >= 0 && token < 128256 && token == fromIntegral (round token :: Int)) actual)
        else do
            expected <- case name of
                "scalar" -> pure (map (realToFrac . Programs.scalar) [-32, -3.5, -1, -0.5, 0, 0.5, 1, 3.5, 32])
                "tiny" -> pure (map fromIntegral (packedExpected 2 4))
                "tail" -> pure (map fromIntegral (packedExpected 131 12))
                _ -> mapM gpuReferenceAt [0 .. count - 1]
            pure $
                if "bl-" `isPrefixOf` name || "layer0-" `isPrefixOf` name
                    then and (zipWith (\value reference -> abs (value - reference) <= 0.00002 * max 1 (abs reference)) actual expected)
                    else actual == expected

-- | Initialize the driver once after the host calls mhs_init.
main :: IO ()
main = do
    current <- readIORef progress
    case current of
        Ready -> submit jobs
        _ -> pure ()

-- | Submit one job. An accepted result permits the next job to start.
submit :: [(String, Int)] -> IO ()
submit [] = do
    writeIORef progress Finished
    putStrLn "MicroHs WASM checked WebGPU fixtures and network token validity PASS"
submit ((name, count) : remaining) = do
    ticket <- withCString name gpuStart
    if ticket > 0
        then writeIORef progress (Waiting ticket name count remaining)
        else reject ("Host rejected WebGPU job " ++ name)

-- | Check fixture values or network token validity. Return one on acceptance.
gpuComplete :: Int -> Int -> IO Int
gpuComplete ticket count = do
    current <- readIORef progress
    case current of
        Waiting expectedTicket name expectedCount remaining
            | ticket == expectedTicket && (count == expectedCount || (name == "network" && expectedCount == 0 && count > 0 && count <= 512)) -> do
                matches <- checkResults name count
                if matches
                    then do
                        putStrLn ("MicroHs WASM consumed WebGPU " ++ name ++ " result PASS")
                        submit remaining
                        pure 1
                    else do
                        reject ("WebGPU output differs from source reference for " ++ name)
                        pure 0
        _ -> do
            reject "Unexpected WebGPU ticket or result count"
            pure 0

-- | Report zero while pending, one after all checks, or minus one on failure.
gpuStatus :: IO Int
gpuStatus = do
    current <- readIORef progress
    pure $ case current of
        Finished -> 1
        Failed -> -1
        _ -> 0

-- | Stop the sequence after an invalid dispatch or completion.
reject :: String -> IO ()
reject message = do
    writeIORef progress Failed
    putStrLn ("MicroHs WASM WebGPU FAIL: " ++ message)
