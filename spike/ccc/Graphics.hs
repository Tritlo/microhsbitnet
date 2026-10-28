{-# LANGUAGE GADTs #-}

module Main (main) where

import qualified ArrayGraph
import CCC (Cat, Code (..), emitWGSL, showCat)
import Combinators (combinatorOf, kernelFloat1, kernelFloat2, kernelFloat4, translateArray)
import Compiler (compileIntPair)
import Data.List (isPrefixOf)
import FloatWGSL (wgslPrelude)
import Frontend (checkedModule)
import MicroHs.Expr (EDef)
import MicroHs.TypeCheck (TModule)
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.FilePath (takeDirectory)
import WebJSON (JSON (..), renderJSON)

-- | Export the source, compiler terms, graphs, and WGSL for eight functions.
main :: IO ()
main = do
    arguments <- getArgs
    output <- case arguments of
        [path] -> pure path
        _ -> fail "Usage: graphics-data OUTPUT.json"
    programs <- checkedModule "spike/ccc/Programs.hs"
    arrays <- checkedModule "spike/ccc/ArrayPrograms.hs"
    layers <- checkedModule "spike/ccc/LayerPrograms.hs"
    scalar <- either fail pure (kernelFloat1 programs "scalar")
    shader <- either fail pure (scalarSource scalar)
    rotate <- either fail pure (kernelFloat4 layers "rotateFirst")
    frequency <- either fail pure (kernelFloat2 layers "frequency")
    contribution <- either fail pure (compileIntPair programs "contribution")
    contributionCode <- either fail pure (emitWGSL contribution (PairCode (IntCode "field") (IntCode "activation")))
    let IntCode contributionWGSL = contributionCode
    entries <-
        sequence
            [ floatEntry "spike/ccc/Programs.hs" programs "scalar" "Scalar arithmetic" scalar (FloatCode "input")
            , arrayEntry arrays ("softmax", "Softmax", 1, "attn")
            , arrayEntry arrays ("rmsNorm", "RMSNorm", 2, "rms")
            , arrayEntry arrays ("residual", "Residual add", 2, "add")
            , arrayEntry arrays ("gated", "Squared-ReLU gate", 2, "gate")
            , floatEntry
                "spike/ccc/LayerPrograms.hs"
                layers
                "rotateFirst"
                "Rotary embedding component"
                rotate
                (PairCode (FloatCode "first") (PairCode (FloatCode "second") (PairCode (FloatCode "cosine") (FloatCode "sine"))))
            , floatEntry "spike/ccc/LayerPrograms.hs" layers "frequency" "Rotary frequency" frequency (PairCode (FloatCode "index") (FloatCode "headDimension"))
            , sourceEntry "spike/ccc/Programs.hs" programs "contribution" "Trit contribution" (showCat contribution) contributionWGSL
            ]
    createDirectoryIfMissing True (takeDirectory output)
    writeFile output (renderJSON (JObject [("entries", JArray entries), ("scalarShader", JString shader)]) ++ "\n")
    putStrLn ("Exported " ++ show (length entries) ++ " source entries to " ++ output)

-- | Emit the direct-combinator scalar graph as a complete compute shader.
scalarSource :: Cat Float Float -> Either String String
scalarSource graph = do
    code <- emitWGSL graph (FloatCode "input")
    let FloatCode expression = code
    pure $
        wgslPrelude
            ++ unlines
                [ "@group(0) @binding(0) var<storage, read> scalarInputs: array<f32>;"
                , "@group(0) @binding(1) var<storage, read_write> scalarOutputs: array<f32>;"
                , "@group(0) @binding(2) var<uniform> scalarParams: vec4<u32>;"
                , "@compute @workgroup_size(128)"
                , "fn scalar(@builtin(global_invocation_id) globalId: vec3<u32>) {"
                , "  let index = globalId.x;"
                , "  if (index >= scalarParams.x) { return; }"
                , "  let input = scalarInputs[index];"
                , "  scalarOutputs[index] = " ++ expression ++ ";"
                , "}"
                ]

-- | Emit one scalar graph as a WGSL expression.
floatEntry :: FilePath -> TModule [EDef] -> String -> String -> Cat input Float -> Code input -> IO JSON
floatEntry file checked name label graph input = do
    code <- either fail pure (emitWGSL graph input)
    let FloatCode expression = code
    sourceEntry file checked name label (showCat graph) expression

-- | Export one array graph and its generated workgroup statements.
arrayEntry :: TModule [EDef] -> (String, String, Int, String) -> IO JSON
arrayEntry checked (name, label, arity, prefix) = do
    graph <- either fail pure (translateArray arity checked name)
    let inputs = if arity == 1 then [(0, "scores")] else [(0, "inputs"), (1, "other")]
        capacity = if name == "softmax" then 2048 else 6912
    plan <- either fail pure (ArrayGraph.emitWorkgroup graph capacity inputs "dimensions.x" prefix)
    expression <- either fail pure (ArrayGraph.outputAt plan "index")
    let shader = unlines (ArrayGraph.declarations plan ++ ArrayGraph.statements plan ++ ["output[index] = " ++ expression ++ ";"])
    sourceEntry "spike/ccc/ArrayPrograms.hs" checked name label (ArrayGraph.describe graph) shader

-- | Combine actual source text and compiler outputs in one export entry.
sourceEntry :: FilePath -> TModule [EDef] -> String -> String -> String -> String -> IO JSON
sourceEntry file checked name label graph shader = do
    source <- readFile file
    snippet <- either fail pure (sourceSnippet name source)
    term <- either fail pure (combinatorOf checked name)
    pure $
        JObject
            [ ("id", JString name)
            , ("label", JString label)
            , ("source", JString snippet)
            , ("sourceFile", JString file)
            , ("combinators", JString term)
            , ("graph", JString graph)
            , ("wgsl", JString shader)
            , ("path", JString (if name == "contribution" then "expression" else "combinators"))
            ]

-- | Read a signature and its function body up to the next blank line.
sourceSnippet :: String -> String -> Either String String
sourceSnippet name source = case dropWhile (not . isPrefixOf (name ++ " ::")) (lines source) of
    [] -> Left ("Missing source signature: " ++ name)
    remaining -> Right (unlines (takeWhile (not . null) remaining))
