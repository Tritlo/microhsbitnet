module ArrayFixtures (runArrayFixtures) where

import Control.Exception (finally)
import Control.Monad (forM, replicateM, unless)
import Data.List (find, sort)
import Text.Read (readMaybe)
import WebBrowser
import WebJSON

-- | Check six whole-array softmax cases and one changed-source case.
runArrayFixtures :: Browser [JSON]
runArrayFixtures = do
    manifest <- fetchText "../array-cases/cases.txt"
    fixtures <- liftBrowser (parseArrayCases manifest)
    samples <- forM [("../softmax.wgsl", take 6 fixtures), ("../softmax-doubled.wgsl", drop 6 fixtures)] $ \(path, cases) -> do
        source <- fetchBytes path
        identifiers <- liftBrowser (replicateM 3 newHandle)
        let [shader, layout, pipeline] = identifiers
        commands
            [ command "shader" [int shader, int source]
            , command "layout" [int layout, JArray (map JString ["read-only-storage", "storage", "uniform"])]
            , command "pipeline" [int pipeline, int shader, int layout, JString "softmax"]
            ]
        liftBrowser (dropHost source)
        results <- mapM (runArrayCase layout pipeline) cases
        commands [command "release" [JArray (map int identifiers), JBool False]]
        pure results
    pure (concat samples)

-- | Require each case once with its exact count before any GPU allocation.
parseArrayCases :: String -> IO [(String, Int)]
parseArrayCases manifest = do
    let expected = [("singleton", 1), ("equal", 8), ("negative", 4), ("wide", 5), ("tail", 131), ("capacity", 2048), ("doubled", 4)]
        names = map fst expected
    fixtures <- forM (lines manifest) $ \line -> case words line of
        [name, text] | name `elem` names -> case readMaybe text :: Maybe Integer of
            Just count
                | all (\character -> character >= '0' && character <= '9') text
                , count > 0
                , count <= 2048 ->
                    pure (name, fromIntegral count)
            _ -> fail ("Invalid array fixture count: " ++ name ++ " " ++ text)
        _ -> fail ("Invalid array fixture manifest line: " ++ line)
    unless (sort (map fst fixtures) == sort names) $
        fail "Array fixture manifest has missing, duplicate, or unknown cases"
    forM expected $ \(name, expectedCount) -> case lookup name fixtures of
        Just count
            | count == expectedCount -> pure (name, count)
            | otherwise -> fail ("Unexpected array fixture count: " ++ name ++ "; count " ++ show count ++ ", expected " ++ show expectedCount)
        Nothing -> fail ("Missing array fixture: " ++ name)

{- | Read exactly the requested number of finite Float32 values.
'readFloats' rejects NaN and infinite values.
-}
arrayFloats :: String -> Int -> Int -> IO [Double]
arrayFloats name host count = do
    size <- hostSize host
    unless (size == count * 4) $
        fail ("Invalid array fixture size: " ++ name ++ "; bytes " ++ show size ++ ", expected " ++ show (count * 4))
    values <- readFloats host 0 count
    unless (length values == count) $
        fail ("Incomplete array fixture: " ++ name)
    pure values

-- | Dispatch one workgroup and compare every output with its CPU reference.
runArrayCase :: Int -> Int -> (String, Int) -> Browser JSON
runArrayCase layout pipeline (name, count) = do
    let label = "softmax-" ++ name
        prefix = "../array-cases/" ++ name
        bytes = count * 4
    input <- fetchBytes (prefix ++ ".input.bin")
    _ <- liftBrowser (arrayFloats (label ++ " input") input count)
    expected <- fetchBytes (prefix ++ ".expected.bin")
    reference <- liftBrowser (arrayFloats (label ++ " reference") expected count `finally` dropHost expected)
    identifiers <- liftBrowser (replicateM 5 newHandle)
    let [inputs, output, dimensions, readback, encoder] = identifiers
    host <- liftBrowser newHandle
    commands
        [ command "buffer" [int inputs, int bytes, int 136, int input]
        , command "buffer" [int output, int bytes, int 132, int 0]
        , command "array" [int dimensions, JString "u32", JArray (map int [count, 0, 0, 0])]
        , command "buffer" [int readback, int bytes, int 9, int 0]
        ]
    liftBrowser (dropHost input)
    commands
        [ command "encoder" [int encoder]
        , command "dispatch" [int encoder, int pipeline, int layout, JArray (map int [inputs, output, dimensions]), int 1]
        , command "copy" [int encoder, int output, int 0, int readback, int 0, int bytes]
        , command "submit" [int encoder]
        , command "readback" [int readback, int host]
        ]
    commands [command "release" [JArray (map int [inputs, output, dimensions, readback]), JBool True]]
    actual <- liftBrowser (arrayFloats (label ++ " output") host count `finally` dropHost host)
    let errors = zipWith (\value target -> abs (value - target)) actual reference
    liftBrowser $ case find (\(_, difference, target) -> isNaN difference || isInfinite difference || difference > 0.000002 * max 1 (abs target)) (zip3 [0 :: Int ..] errors reference) of
        Just (index, difference, _) ->
            fail (label ++ "[" ++ show index ++ "]: absolute error " ++ show difference ++ "; bound 2e-6 * max(1, abs(reference))")
        Nothing -> browserLog (label ++ ": " ++ show count ++ " fixture values PASS")
    pure $
        JObject
            [ ("name", JString label)
            , ("count", int count)
            , ("maxAbsoluteError", JNumber (maximum (0 : errors)))
            , ("passed", JBool True)
            ]
