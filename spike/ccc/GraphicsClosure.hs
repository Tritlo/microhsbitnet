module GraphicsClosure (compilerClosure) where

import Control.Monad (foldM)
import qualified Data.ByteString.Char8 as BS
import Data.Graph (SCC (..), stronglyConnComp)
import Data.List (foldl', nub)
import GHC.Float (float2Double)
import MicroHs.Abstract (compileOpt)
import MicroHs.Compile (compileCacheTop)
import MicroHs.CompileCache (emptyCache)
import MicroHs.Desugar (LDef)
import MicroHs.EncodeData (encTuple, encTupleSel)
import MicroHs.Exp (Exp (..), apps, freeVars, substExp)
import MicroHs.ExpPrint (removeUnused)
import MicroHs.Expr (Lit (..), showLit)
import MicroHs.Flags (Flags (..), defaultFlags)
import MicroHs.Ident (mkIdent, qualOf, showIdent)
import WebJSON (JSON (..))

-- | Export the reachable stock compiler definitions for all requested roots.
compilerClosure :: [String] -> IO JSON
compilerClosure roots = do
    let flags = defaultFlags{mhsdir = "vendor/microhs", srcPaths = ["spike/ccc", "vendor/microhs/lib"]}
        modules = nub (map (qualOf . mkIdent) roots)
        root = apps (Lit (LPrim "I")) (map (Var . mkIdent) roots)
        compile (definitions, cache) name = do
            ((_, compiled), _, next) <- compileCacheTop flags name cache
            pure (compiled, next)
    (definitions, _) <- foldM compile ([], emptyCache) modules
    let reachable = removeUnused (definitions, root)
        groups = stronglyConnComp [(definition, name, nub (freeVars body)) | definition@(name, body) <- reachable]
        normalized = concat (zipWith closeGroup [0 ..] groups)
        closed = removeUnused (normalized, root)
        cycles = length [() | CyclicSCC _ <- groups]
    putStrLn ("Reachable compiler closure: " ++ show (length closed) ++ " definitions; " ++ show cycles ++ " recursive groups")
    pure (JArray [JObject [("name", JString (showIdent name)), ("body", expressionJSON body)] | (name, body) <- closed])

-- | Encode recursive groups with the tuple and Y construction used by MicroHs.
closeGroup :: Int -> SCC LDef -> [LDef]
closeGroup _ (AcyclicSCC definition) = [definition]
closeGroup _ (CyclicSCC [(name, body)]) = [(name, compileOpt (App (Lit (LPrim "Y")) (Lam name body)))]
closeGroup index (CyclicSCC definitions) =
    let tuple = mkIdent ("GraphicsClosure.$tuple" ++ show index)
        group = mkIdent ("GraphicsClosure.$cycle" ++ show index)
        count = length definitions
        projections = [(name, encTupleSel position count (Var tuple)) | (position, (name, _)) <- zip [0 ..] definitions]
        replace body = foldl' (\current (name, value) -> substExp name value current) body projections
        body = encTuple (map (replace . snd) definitions)
        fixed = compileOpt (App (Lit (LPrim "Y")) (Lam tuple body))
     in (group, fixed) : [(name, compileOpt (encTupleSel position count (Var group))) | (position, (name, _)) <- zip [0 ..] definitions]

-- | Keep applications, references, and literal kinds distinct in JSON.
expressionJSON :: Exp -> JSON
expressionJSON expression = case expression of
    Var name -> JObject [("var", JString (showIdent name))]
    App function argument -> JObject [("app", JArray [expressionJSON function, expressionJSON argument])]
    Lam name body -> JObject [("lam", JArray [JString (showIdent name), expressionJSON body])]
    Lit literal -> literalJSON literal

-- | Retain machine literals and runtime boundaries without numerical expansion.
literalJSON :: Lit -> JSON
literalJSON literal = case literal of
    LInt value -> tagged "int" (integerJSON (toInteger value))
    LInt64 value -> tagged "int64" (integerJSON (toInteger value))
    LInteger value -> tagged "integer" (JString (show value))
    LFloat value -> tagged "float" (floatingJSON (float2Double value))
    LDouble value -> tagged "double" (floatingJSON value)
    LRat value -> tagged "rat" (JString (show value))
    LChar value -> tagged "char" (JString [value])
    LStr value -> tagged "string" (JString value)
    LBStr value -> tagged "bstr" (JString (BS.unpack value))
    LPrim value -> tagged "prim" (JString value)
    LExn value -> tagged "exn" (JString value)
    LForImp{} -> tagged "forimp" (JString (showLit literal))
    LCType{} -> tagged "ctype" (JString (showLit literal))
    LTick value -> tagged "tick" (JString value)
  where
    tagged name value = JObject [(name, value)]

-- | Preserve integers outside the exact JavaScript Number range as decimal text.
integerJSON :: Integer -> JSON
integerJSON value
    | abs value <= 9007199254740991 = JInt (fromInteger value)
    | otherwise = JString (show value)

-- | Preserve nonfinite floating-point literals as explicit named values.
floatingJSON :: Double -> JSON
floatingJSON value
    | isNaN value || isInfinite value = JString (show value)
    | otherwise = JNumber value
