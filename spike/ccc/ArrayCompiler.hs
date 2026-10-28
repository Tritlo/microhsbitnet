{-# LANGUAGE GADTs #-}

module ArrayCompiler (compileArray) where

import ArrayGraph
import CCC
import Compiler (compileFloatExpression, normalizeExpression)
import Control.Monad (unless)
import Control.Monad.State.Strict
import MicroHs.Expr
import MicroHs.Ident (Ident, mkIdent, showIdent, unQualIdent)
import MicroHs.TypeCheck (TModule, tBindingsOf)

-- | Keep typed scalar captures and their categorical projections together.
data Scope where
    Scope :: Captures values -> [(Ident, Cat values Float)] -> Scope

-- | Retain shared array nodes while source expressions are translated.
data Build = Build
    { nextId :: Int
    , builtSteps :: [Step]
    , scalarNodes :: [(String, Ref Float)]
    , vectorNodes :: [(String, Ref [Float])]
    }

-- | Compile a monomorphic vector function from checked MicroHs source.
compileArray :: TModule [EDef] -> String -> Either String Program
compileArray checked name = do
    signature <- case [signature | Sign names signature <- tBindingsOf checked, mkIdent name `elem` names] of
        [signature] -> Right signature
        _ -> Left ("Missing array function signature: " ++ name)
    let (arguments, result) = getArrows signature
        vectorType (EApp (EVar constructor) (EVar element)) = showIdent constructor == "SourceArray.Vector" && showIdent element == "Primitives.Float"
        vectorType _ = False
    unless (length arguments >= 1 && length arguments <= 2 && all vectorType (result : arguments)) (Left "Expected one or two Vector Float arguments")
    definition <- case [identifier | Fcn identifier _ <- tBindingsOf checked, unQualIdent identifier == mkIdent name] of
        [identifier] -> Right identifier
        _ -> Left ("Missing array function: " ++ name)
    let inputs = take (length arguments) (map (mkIdent . ("array$input" ++)) (["0", "1"] :: [String]))
        applied = foldl EApp (EVar definition) (map EVar inputs)
    source <- normalizeExpression checked applied
    (output, state) <- runStateT (vector checked inputs source) (Build (length inputs) [] [] [])
    pure (Program (reverse (builtSteps state)) output)

-- | Assign a unique reference to one graph operation.
fresh :: StateT Build (Either String) Int
fresh = do
    state <- get
    put state{nextId = nextId state + 1}
    pure (nextId state)

-- | Separate applications without inspecting source spelling.
application :: Expr -> (Expr, [Expr])
application expression = collect expression []
  where
    collect (EApp function argument) arguments = collect function (argument : arguments)
    collect function arguments = (function, arguments)

-- | Translate shape-preserving maps and equal-length zips.
vector :: TModule [EDef] -> [Ident] -> Expr -> StateT Build (Either String) (Ref [Float])
vector checked inputs source = case application source of
    (EVar identifier, [])
        | Just index <- lookup identifier (zip inputs [0 ..]) -> pure (VectorRef index)
    (EVar identifier, [function, values]) | showIdent identifier == "SourceArray.map" -> do
        sourceRef <- vector checked inputs values
        element <- mkIdent . ("array$element" ++) . show <$> fresh
        body <- lift (normalizeExpression checked (EApp function (EVar element)))
        (expression, references) <- extract checked inputs body
        case scope references of
            Scope captures bindings -> do
                graph <- lift (compileFloatExpression checked ((element, First) : [(name, Compose projection Second) | (name, projection) <- bindings]) expression)
                let key = "map" ++ show (refId sourceRef, captureIds captures) ++ showCat graph
                cached <- gets (lookup key . vectorNodes)
                case cached of
                    Just reference -> pure reference
                    Nothing -> do
                        target <- VectorRef <$> fresh
                        modify (\state -> state{builtSteps = Map target graph captures sourceRef : builtSteps state, vectorNodes = (key, target) : vectorNodes state})
                        pure target
    (EVar identifier, [function, left, right]) | showIdent identifier == "SourceArray.zipWith" -> do
        leftRef <- vector checked inputs left
        rightRef <- vector checked inputs right
        leftElement <- mkIdent . ("array$left" ++) . show <$> fresh
        rightElement <- mkIdent . ("array$right" ++) . show <$> fresh
        body <- lift (normalizeExpression checked (EApp (EApp function (EVar leftElement)) (EVar rightElement)))
        (expression, references) <- extract checked inputs body
        case scope references of
            Scope captures bindings -> do
                let environment = [(leftElement, First), (rightElement, Compose First Second)] ++ [(name, Compose projection (Compose Second Second)) | (name, projection) <- bindings]
                graph <- lift (compileFloatExpression checked environment expression)
                let key = "zip" ++ show (refId leftRef, refId rightRef, captureIds captures) ++ showCat graph
                cached <- gets (lookup key . vectorNodes)
                case cached of
                    Just reference -> pure reference
                    Nothing -> do
                        target <- VectorRef <$> fresh
                        modify (\state -> state{builtSteps = Map2 target graph captures leftRef rightRef : builtSteps state, vectorNodes = (key, target) : vectorNodes state})
                        pure target
    _ -> lift (Left ("Unsupported array expression: " ++ take 240 (showExpr source)))

-- | Lift scalar reductions out of a map and retain their typed dependencies.
extract :: TModule [EDef] -> [Ident] -> Expr -> StateT Build (Either String) (Expr, [(Ident, Ref Float)])
extract checked inputs source = case application source of
    (EVar identifier, [values]) | showIdent identifier `elem` ["SourceArray.sum", "SourceArray.maximum", "SourceArray.count"] -> do
        sourceRef <- vector checked inputs values
        let key = showIdent identifier ++ show (refId sourceRef)
        cached <- gets (lookup key . scalarNodes)
        reference <- case cached of
            Just reference -> pure reference
            Nothing -> do
                target <- ScalarRef <$> fresh
                let step =
                        case showIdent identifier of
                            "SourceArray.sum" -> Sum target sourceRef
                            "SourceArray.maximum" -> Maximum target sourceRef
                            _ -> Length target sourceRef
                modify (\state -> state{builtSteps = step : builtSteps state, scalarNodes = (key, target) : scalarNodes state})
                pure target
        let name = mkIdent ("array$capture" ++ show (refId reference))
        pure (EVar name, [(name, reference)])
    _ -> case source of
        EApp function argument -> do
            (functionExpr, functionRefs) <- extract checked inputs function
            (argumentExpr, argumentRefs) <- extract checked inputs argument
            pure (EApp functionExpr argumentExpr, merge functionRefs argumentRefs)
        EIf condition yes no -> do
            (conditionExpr, conditionRefs) <- extract checked inputs condition
            (yesExpr, yesRefs) <- extract checked inputs yes
            (noExpr, noRefs) <- extract checked inputs no
            pure (EIf conditionExpr yesExpr noExpr, merge conditionRefs (merge yesRefs noRefs))
        ETuple elements -> do
            results <- mapM (extract checked inputs) elements
            pure (ETuple (map fst results), foldr (merge . snd) [] results)
        EVar _ -> pure (source, [])
        ELit _ _ -> pure (source, [])
        _ -> lift (Left ("Unsupported array scalar expression: " ++ take 160 (showExpr source)))

-- | Keep each captured scalar once, in first-use order.
merge :: [(Ident, Ref Float)] -> [(Ident, Ref Float)] -> [(Ident, Ref Float)]
merge left right = left ++ filter (\(name, _) -> name `notElem` map fst left) right

-- | Construct an existential product without untyped scalar slots.
scope :: [(Ident, Ref Float)] -> Scope
scope [] = Scope NoCaptures []
scope ((name, reference) : remaining) = case scope remaining of
    Scope captures bindings -> Scope (Capture reference captures) ((name, First) : [(identifier, Compose projection Second) | (identifier, projection) <- bindings])
