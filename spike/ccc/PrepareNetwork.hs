module PrepareNetwork where

import Control.Monad (forM, unless)
import qualified Data.ByteString as BS
import Data.Char (ord)
import Data.List (intercalate)
import Dot (argmaxBf16BS, bnRmsNorm, packF32)
import Forward (buildLayersC, cVocab, emptyCachesC, forceLayersC, look, mkNormBS, stepLayersC)
import Loader (loadModel, slice)
import Model (KVC, LayerC, cEps, cHidden, decoderStepC, embedRow)
import Numeric (showHex)
import System.Directory (createDirectoryIfMissing)
import System.IO.TimeMilli (getTimeMilli)
import Tokenizer (encodeChat, loadTokenizer)

-- | Keep the decoder layers, embedding bytes, and final norm weights.
data Network = Network [LayerC] BS.ByteString BS.ByteString

-- | Store the original case and the suite index in one root directory.
rootDirectory :: FilePath
rootDirectory = "bin/ccc-spike/network-cases/"

-- | Keep each prompt and its case directory relative to the suite root.
cases :: [(String, String, FilePath)]
cases =
    [ ("hello", "Hello!", "")
    , ("question", "What is the capital of France?", "question/")
    , ("arithmetic", "What is 7 + 5? Reply with just the number.", "arithmetic/")
    , ("code", "Complete this Haskell definition with one expression:\nadd :: Int -> Int -> Int\nadd left right =", "code/")
    , ("swedish", "Säg hej på svenska.", "swedish/")
    , ("longer", "Summarize this paragraph in one sentence:\nA small coastal town opened a library in an unused railway station. Volunteers repaired the rooms, collected donated books, and added free internet access. Within six months, children visited after school, older residents attended reading groups, and local students used the quiet space to study together.", "longer/")
    ]

-- | Load the model once and record six independent native reference cases.
main :: IO ()
main = do
    (tensors, bytes) <- loadModel "models/bitnet-b1.58-2B-4T/model.safetensors"
    tokenizer <- loadTokenizer "models/bitnet-b1.58-2B-4T/tokenizer.json"
    let layers = buildLayersC tensors bytes 30
        embeddings = slice bytes (look tensors "model.embed_tokens.weight")
        normWeights = mkNormBS tensors bytes "model.norm.weight"
        network = Network layers embeddings normWeights
    forceLayersC layers `seq` BS.length normWeights `seq` pure ()
    entries <- forM cases $ \(name, promptText, relativeDirectory) -> do
        let directory = rootDirectory ++ relativeDirectory
            prompt = encodeChat tokenizer promptText
        writeCase directory network name prompt
        pure $
            "{\"name\":"
                ++ jsonString name
                ++ ",\"promptText\":"
                ++ jsonString promptText
                ++ ",\"manifest\":"
                ++ jsonString (relativeDirectory ++ "manifest.json")
                ++ "}"
    writeFile (rootDirectory ++ "suite.json") ("[" ++ intercalate "," entries ++ "]\n")

-- | Quote JSON text without changing Unicode prompt characters.
jsonString :: String -> String
jsonString value = "\"" ++ concatMap escape value ++ "\""
  where
    escape '"' = "\\\""
    escape '\\' = "\\\\"
    escape character
        | ord character < 32 = "\\u" ++ replicate (4 - length digits) '0' ++ digits
        | otherwise = [character]
      where
        digits = showHex (ord character) ""

-- | Record one prompt with fresh caches and at most ten generated tokens.
writeCase :: FilePath -> Network -> String -> [Int] -> IO ()
writeCase directory network@(Network layers _ _) name prompt = do
    createDirectoryIfMissing True directory
    putStrLn ("Case: " ++ name)
    putStrLn ("Prompt ids: " ++ show prompt)
    prefillStart <- getTimeMilli
    (promptSteps, caches, (firstToken, _)) <- prefill directory network 0 (emptyCachesC (length layers)) prompt
    prefillEnd <- getTimeMilli
    putStrLn ("Prefill including reference IO: " ++ show (prefillEnd - prefillStart) ++ " ms")
    generationStart <- getTimeMilli
    (generationSteps, generated) <- generate directory network (length prompt) caches firstToken 10
    generationEnd <- getTimeMilli
    putStrLn ("Generation including reference IO: " ++ show (generationEnd - generationStart) ++ " ms")
    writeFile (directory ++ "manifest.json") $
        "{\"prompt\":"
            ++ show prompt
            ++ ",\"generated\":"
            ++ show generated
            ++ ",\"steps\":["
            ++ intercalate "," (promptSteps ++ generationSteps)
            ++ "]}\n"
    putStrLn ("Generated ids: " ++ show generated)
    putStrLn $
        "Case "
            ++ name
            ++ ": prompt "
            ++ show (length prompt)
            ++ ", generated "
            ++ show (length generated)
            ++ ", prefill "
            ++ show (prefillEnd - prefillStart)
            ++ " ms, generation "
            ++ show (generationEnd - generationStart)
            ++ " ms"

-- | Consume every prompt token. Read the LM head only for the final token.
prefill :: FilePath -> Network -> Int -> [KVC] -> [Int] -> IO ([String], [KVC], (Int, Float))
prefill _ _ _ _ [] = fail "The chat prompt is empty"
prefill directory network position caches (token : remaining) = do
    (nextCaches, entry, prediction) <- writeStep directory network position token caches (null remaining)
    case remaining of
        [] -> case prediction of
            Just result -> pure ([entry], nextCaches, result)
            Nothing -> fail "Missing prediction after prefill"
        _ -> do
            (entries, finalCaches, result) <- prefill directory network (position + 1) nextCaches remaining
            pure (entry : entries, finalCaches, result)

-- | Consume each selected token, including the final token or EOS token.
generate :: FilePath -> Network -> Int -> [KVC] -> Int -> Int -> IO ([String], [Int])
generate directory network position caches token remaining = do
    (nextCaches, entry, Just (nextToken, _)) <- writeStep directory network position token caches True
    if token `elem` [128009, 128001] || remaining <= 1
        then pure ([entry], [token])
        else do
            (entries, tokens) <- generate directory network (position + 1) nextCaches nextToken (remaining - 1)
            pure (entry : entries, token : tokens)

-- | Run all layers, save Float32 buffers, and optionally record the LM head.
writeStep :: FilePath -> Network -> Int -> Int -> [KVC] -> Bool -> IO ([KVC], String, Maybe (Int, Float))
writeStep directory (Network layers embeddings normWeights) position token caches includeArgmax = do
    let input = packF32 (embedRow embeddings token)
    (hidden, nextCaches, layerFiles) <-
        if position < 3
            then writeLayers directory position 0 layers caches input
            else do
                let (output, updatedCaches) = stepLayersC layers position caches input
                pure (output, updatedCaches, [])
    let normed = bnRmsNorm normWeights hidden cHidden cEps
        prefix = "step-" ++ show position
        hiddenFile = prefix ++ ".hidden.bin"
        normedFile = prefix ++ ".normed.bin"
    unless (BS.length hidden == cHidden * 4 && BS.length normed == cHidden * 4) $
        fail "Unexpected network hidden or final norm length"
    BS.writeFile (directory ++ hiddenFile) hidden
    BS.writeFile (directory ++ normedFile) normed
    prediction <-
        if includeArgmax
            then do
                let result@(nextToken, logit) = argmaxBf16BS embeddings normed cHidden cVocab
                unless (nextToken >= 0 && nextToken < cVocab && not (isNaN logit || isInfinite logit)) $
                    fail "Invalid LM head prediction"
                pure (Just result)
            else pure Nothing
    let predictionFields = case prediction of
            Nothing -> ""
            Just (nextToken, logit) -> ",\"argmax\":" ++ show nextToken ++ ",\"logit\":" ++ show logit
        layerFields = if null layerFiles then "" else ",\"layers\":" ++ show layerFiles
        entry =
            "{\"position\":"
                ++ show position
                ++ ",\"token\":"
                ++ show token
                ++ ",\"hidden\":"
                ++ show hiddenFile
                ++ ",\"normed\":"
                ++ show normedFile
                ++ predictionFields
                ++ layerFields
                ++ "}"
    putStrLn ("Network position " ++ show position ++ ", token " ++ show token ++ " reference written")
    pure (nextCaches, entry, prediction)

-- | Run each layer with its input cache and write its output.
writeLayers :: FilePath -> Int -> Int -> [LayerC] -> [KVC] -> BS.ByteString -> IO (BS.ByteString, [KVC], [String])
writeLayers _ _ _ [] [] input = pure (input, [], [])
writeLayers directory position layerIndex (layer : layers) (cache : caches) input = do
    let (output, nextCache) = decoderStepC layer position cache input
        file = "step" ++ show position ++ ".layer-" ++ show layerIndex ++ ".hidden.bin"
    unless (BS.length output == cHidden * 4) $
        fail "Unexpected layer checkpoint length"
    BS.writeFile (directory ++ file) output
    (hidden, nextCaches, files) <- writeLayers directory position (layerIndex + 1) layers caches output
    pure (hidden, nextCache : nextCaches, file : files)
writeLayers _ _ _ _ _ _ = fail "Layer and cache counts differ"
