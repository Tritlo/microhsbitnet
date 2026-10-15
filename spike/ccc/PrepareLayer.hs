module PrepareLayer where

import Control.Monad (unless)
import qualified Data.ByteString as BS
import Data.List (intercalate)
import Dot (packF32)
import Forward (buildLayersC, look)
import Loader (loadModel, slice)
import Model (KVC, LayerC (..), Lin (..), cHidden, decoderStepC, embedRow)
import System.Directory (createDirectoryIfMissing)

-- | Store files relative to the layer manifest.
directory :: FilePath
directory = "bin/ccc-spike/layer-cases/"

-- | Export layer zero and three embedding inputs with a shared KV cache.
main :: IO ()
main = do
    createDirectoryIfMissing True directory
    (tensors, bytes) <- loadModel "models/bitnet-b1.58-2B-4T/model.safetensors"
    let [layer] = buildLayersC tensors bytes 1
        embeddings = slice bytes (look tensors "model.embed_tokens.weight")
        linears =
            [ ("self_attn.q_proj.weight", cQ layer)
            , ("self_attn.k_proj.weight", cK layer)
            , ("self_attn.v_proj.weight", cV layer)
            , ("self_attn.o_proj.weight", cO layer)
            , ("mlp.gate_proj.weight", cGate layer)
            , ("mlp.up_proj.weight", cUp layer)
            , ("mlp.down_proj.weight", cDown layer)
            ]
        norms =
            [ ("input_layernorm.weight", cInputNorm layer)
            , ("self_attn.attn_sub_norm.weight", cAttnSubNorm layer)
            , ("post_attention_layernorm.weight", cPostNorm layer)
            , ("mlp.ffn_sub_norm.weight", cFfnSubNorm layer)
            ]
    linearEntries <- mapM (\(name, linear) -> writeWeight name "linear" (linIn linear) (linOut linear) (linScale linear) (linPacked linear)) linears
    normEntries <- mapM (\(name, norm) -> writeWeight name "norm" (BS.length norm `div` 4) 1 1 norm) norms
    steps <- writeSteps layer embeddings Nothing (zip [0 ..] [128000, 9906, 0])
    writeFile (directory ++ "manifest.json") $
        "{\"weights\":["
            ++ intercalate "," (linearEntries ++ normEntries)
            ++ "],\"steps\":["
            ++ intercalate "," steps
            ++ "]}\n"
    putStrLn "Exported seven linears, four norms, and three layer-zero steps"

-- | Write packed or Float32 weights and return their JSON metadata.
writeWeight :: String -> String -> Int -> Int -> Float -> BS.ByteString -> IO String
writeWeight suffix kind columns rows scale bytes = do
    let name = "model.layers.0." ++ suffix
        file = name ++ ".bin"
        expectedBytes = if kind == "linear" then columns * rows `div` 4 else columns * 4
    unless (BS.length bytes == expectedBytes && scale > 0 && not (isInfinite scale)) $
        fail ("Invalid layer weight shape or scale: " ++ name)
    BS.writeFile (directory ++ file) bytes
    pure $
        "{\"name\":"
            ++ show name
            ++ ",\"kind\":"
            ++ show kind
            ++ ",\"file\":"
            ++ show file
            ++ ",\"columns\":"
            ++ show columns
            ++ ",\"rows\":"
            ++ show rows
            ++ ",\"scale\":"
            ++ show scale
            ++ "}"

-- | Run one layer on fresh embedding rows and retain each updated KV cache.
writeSteps :: LayerC -> BS.ByteString -> KVC -> [(Int, Int)] -> IO [String]
writeSteps _ _ _ [] = pure []
writeSteps layer embeddings cache ((position, token) : remaining) = do
    let prefix = "step-" ++ show position
        inputFile = prefix ++ ".input.bin"
        expectedFile = prefix ++ ".expected.bin"
        input = packF32 (embedRow embeddings token)
        (expected, nextCache) = decoderStepC layer position cache input
    unless (BS.length input == cHidden * 4 && BS.length expected == cHidden * 4) $
        fail "Unexpected layer input or output length"
    BS.writeFile (directory ++ inputFile) input
    BS.writeFile (directory ++ expectedFile) expected
    putStrLn ("Layer zero position " ++ show position ++ ", token " ++ show token ++ " reference written")
    rest <- writeSteps layer embeddings nextCache remaining
    pure $
        ( "{\"position\":"
            ++ show position
            ++ ",\"input\":"
            ++ show inputFile
            ++ ",\"expected\":"
            ++ show expectedFile
            ++ "}"
        )
            : rest
