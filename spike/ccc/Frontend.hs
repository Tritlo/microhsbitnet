module Frontend (checkedModule) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Monad (forM)
import MicroHs.Compile (addPreludeImport, compileMany)
import MicroHs.CompileCache (emptyCache, getCacheTables, lookupCache)
import MicroHs.Expr
import MicroHs.Flags (Flags (..), defaultFlags)
import MicroHs.Parse (pTop, parseDie)
import MicroHs.TypeCheck (TModule, typeCheck)

-- | Run the upstream parser and type checker before desugaring.
checkedModule :: FilePath -> IO (TModule [EDef])
checkedModule path = do
    source <- readFile path
    let parsed@(EModule _ _ declarations) = addPreludeImport (parseDie pTop path source)
        imports = [spec | Import spec <- declarations]
        flags =
            defaultFlags
                { mhsdir = "vendor/microhs"
                , srcPaths = ["spike/ccc", "vendor/microhs/lib"]
                }
        names = [name | ImportSpec _ _ name _ _ <- imports]
    cache <- compileMany flags names emptyCache
    imported <- forM imports $ \spec@(ImportSpec _ _ name _ _) ->
        case lookupCache name cache of
            Nothing -> fail ("Missing checked import: " ++ show name)
            Just compiled -> pure (spec, compiled)
    let (checked, _, _, _) = typeCheck flags (getCacheTables cache) ImpNormal imported parsed
    evaluate (force checked)
