{-# LANGUAGE GADTs #-}

module Compiler (compileScalar, compileIntPair, compileFloatPair, compileFloatTriple, compileFloatQuadruple, compileFloatPredicate, compileQuantize, compileRescale, compileFloatExpression, normalizeExpression) where

import CCC
import Control.Monad (foldM)
import Control.Monad.State.Strict (StateT, evalStateT, get, lift, put)
import Data.Graph (SCC (..), stronglyConnComp)
import Data.Int (Int32)
import Data.List (isPrefixOf)
import MicroHs.Expr
import MicroHs.Ident (Ident, mkIdent, qualOf, showIdent, unQualIdent)
import MicroHs.TypeCheck (TModule, tBindingsOf, tModuleName)

-- | An existential scalar result with a statically typed categorical graph.
data Scalar input
    = Floating (Cat input Float)
    | Integral (Cat input Int32)
    | Boolean (Cat input Bool)

-- | Compile a checked unary Float function.
compileScalar :: TModule [EDef] -> String -> Either String (Cat Float Float)
compileScalar checked name = do
    (parameters, body) <- function checked name ["Primitives.Float", "Primitives.Float"]
    case parameters of
        [parameter] -> do
            result <- compileExpression checked [(parameter, Floating Id)] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected one scalar argument"

-- | Compile a checked binary Int32 function over a product input.
compileIntPair :: TModule [EDef] -> String -> Either String (Cat (Int32, Int32) Int32)
compileIntPair checked name = do
    (parameters, body) <- function checked name (replicate 3 "Data.Int.Int32.Int32")
    case parameters of
        [left, right] -> do
            result <- compileExpression checked [(left, Integral First), (right, Integral Second)] body
            case result of
                Integral graph -> Right graph
                _ -> Left "Expected an Int32 result"
        _ -> Left "Expected two integer arguments"

-- | Compile a checked binary Float function over a product input.
compileFloatPair :: TModule [EDef] -> String -> Either String (Cat (Float, Float) Float)
compileFloatPair checked name = do
    (parameters, body) <- function checked name (replicate 3 "Primitives.Float")
    case parameters of
        [left, right] -> do
            result <- compileExpression checked [(left, Floating First), (right, Floating Second)] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected two Float arguments"

-- | Compile a checked ternary Float function over a nested product input.
compileFloatTriple :: TModule [EDef] -> String -> Either String (Cat (Float, (Float, Float)) Float)
compileFloatTriple checked name = do
    (parameters, body) <- function checked name (replicate 4 "Primitives.Float")
    case parameters of
        [first, second, third] -> do
            result <- compileExpression checked [(first, Floating First), (second, Floating (Compose First Second)), (third, Floating (Compose Second Second))] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected three Float arguments"

-- | Compile a checked four-argument Float function over nested product inputs.
compileFloatQuadruple :: TModule [EDef] -> String -> Either String (Cat (Float, (Float, (Float, Float))) Float)
compileFloatQuadruple checked name = do
    (parameters, body) <- function checked name (replicate 5 "Primitives.Float")
    case parameters of
        [first, second, third, fourth] -> do
            result <- compileExpression checked [(first, Floating First), (second, Floating (Compose First Second)), (third, Floating (Compose First (Compose Second Second))), (fourth, Floating (Compose Second (Compose Second Second)))] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected four Float arguments"

-- | Compile a checked comparison over two Float values.
compileFloatPredicate :: TModule [EDef] -> String -> Either String (Cat (Float, Float) Bool)
compileFloatPredicate checked name = do
    (parameters, body) <- function checked name ["Primitives.Float", "Primitives.Float", "Data.Bool_Type.Bool"]
    case parameters of
        [left, right] -> do
            result <- compileExpression checked [(left, Floating First), (right, Floating Second)] body
            case result of
                Boolean graph -> Right graph
                _ -> Left "Expected a Bool result"
        _ -> Left "Expected two Float arguments"

-- | Compile a checked Float quantizer with an Int32 result.
compileQuantize :: TModule [EDef] -> String -> Either String (Cat (Float, Float) Int32)
compileQuantize checked name = do
    (parameters, body) <- function checked name ["Primitives.Float", "Primitives.Float", "Data.Int.Int32.Int32"]
    case parameters of
        [input, maximum] -> do
            result <- compileExpression checked [(input, Floating First), (maximum, Floating Second)] body
            case result of
                Integral graph -> Right graph
                _ -> Left "Expected an Int32 result"
        _ -> Left "Expected two Float arguments"

-- | Compile checked rescaling over an accumulator and two Float scales.
compileRescale :: TModule [EDef] -> String -> Either String (Cat (Int32, (Float, Float)) Float)
compileRescale checked name = do
    (parameters, body) <- function checked name ["Data.Int.Int32.Int32", "Primitives.Float", "Primitives.Float", "Primitives.Float"]
    case parameters of
        [accumulator, weightScale, maximum] -> do
            result <- compileExpression checked [(accumulator, Integral First), (weightScale, Floating (Compose First Second)), (maximum, Floating (Compose Second Second))] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected an Int32 accumulator and two Float scales"

-- | Compile a checked Float expression with lexical variables in any input context.
compileFloatExpression :: TModule [EDef] -> [(Ident, Cat input Float)] -> Expr -> Either String (Cat input Float)
compileFloatExpression checked variables source = do
    result <- compileExpression checked [(identifier, Floating graph) | (identifier, graph) <- variables] source
    case result of
        Floating graph -> Right graph
        _ -> Left "Expected a Float result"

-- | Normalize source bindings before scalar primitive validation.
compileExpression :: TModule [EDef] -> [(Ident, Scalar input)] -> Expr -> Either String (Scalar input)
compileExpression checked variables source = do
    normalized <- normalizeWith checked (map fst variables) source
    expression variables normalized

{- | Expand nonrecursive bindings and reduce lambda applications without capture.
Unapplied variable lambdas and external names remain in the result.
Consumers must validate external primitives and their dictionaries.
-}
normalizeExpression :: TModule [EDef] -> Expr -> Either String Expr
normalizeExpression checked = normalizeWith checked []

-- | Normalize with protected external lexical variables and fresh lambda binders.
normalizeWith :: TModule [EDef] -> [Ident] -> Expr -> Either String Expr
normalizeWith checked variables source = evalStateT (normalize globals environment [] source) fresh
  where
    globals = [(identifier, equations) | Fcn identifier equations <- tBindingsOf checked, qualOf identifier == tModuleName checked]
    environment = [(identifier, EVar identifier) | identifier <- variables]
    used = variables ++ allVarsExpr source ++ concat [identifier : allVarsEqns equations | (identifier, equations) <- globals]
    fresh = filter (`notElem` used) [mkIdent ("ccc$" ++ show index) | index <- [0 :: Int ..]]

-- | Carry fresh names through normalization and report unsupported source.
type Normalizer = StateT [Ident] (Either String)

-- | Select a name that cannot capture an existing source variable.
freshIdentifier :: Normalizer Ident
freshIdentifier = do
    available <- get
    case available of
        identifier : remaining -> put remaining >> pure identifier
        [] -> lift (Left "No fresh identifiers available")

-- | Normalize one expression with lexical substitutions and exact helper names.
normalize :: [(Ident, [Eqn])] -> [(Ident, Expr)] -> [Ident] -> Expr -> Normalizer Expr
normalize globals variables active source = case source of
    EVar identifier
        | Just value <- lookup identifier variables -> pure value
        | Just equations <- lookup identifier globals ->
            if identifier `elem` active
                then lift (Left ("Recursive helper: " ++ showIdent identifier))
                else do
                    (parameters, body) <- lift (equation equations)
                    normalize globals [] (identifier : active) (eLam (map EVar parameters) body)
        | otherwise -> pure source
    ELit{} -> pure source
    EApp operator argument -> do
        normalizedOperator <- normalize globals variables active operator
        normalizedArgument <- normalize globals variables active argument
        case normalizedOperator of
            ELam location [Eqn (EVar parameter : parameters) (EAlts [([], body)] [])] ->
                normalize globals [(parameter, normalizedArgument)] active $
                    if null parameters then body else ELam location [Eqn parameters (EAlts [([], body)] [])]
            _ -> pure (EApp normalizedOperator normalizedArgument)
    ELam location equations -> do
        (parameters, body) <- lift (equation equations)
        renamed <- mapM (const freshIdentifier) parameters
        normalized <- normalize globals (zip parameters (map EVar renamed) ++ variables) active body
        pure (if null renamed then normalized else ELam location [Eqn (map EVar renamed) (EAlts [([], normalized)] [])])
    ELet bindings body -> do
        values <- lift (sourceBindings bindings)
        dependencies <- lift (mapM (\(identifier, value) -> do free <- freeVariables value; Right ((identifier, value), identifier, free)) values)
        extended <- foldM binding variables (stronglyConnComp dependencies)
        normalize globals extended active body
    EIf condition yes no -> EIf <$> normalize globals variables active condition <*> normalize globals variables active yes <*> normalize globals variables active no
    ETuple values -> ETuple <$> mapM (normalize globals variables active) values
    _ -> lift (Left ("Unsupported checked expression: " ++ take 160 (showExpr source)))
  where
    binding environment (AcyclicSCC (identifier, value)) = do
        normalized <- normalize globals environment active value
        pure ((identifier, normalized) : environment)
    binding _ (CyclicSCC values) = lift (Left ("Recursive let binding: " ++ unwords (map (showIdent . fst) values)))

-- | Accept variable parameters and unwrap only generated dictionary equations.
equation :: [Eqn] -> Either String ([Ident], Expr)
equation [Eqn patterns (EAlts [([], body)] bindings)] = do
    parameters <- mapM variable patterns
    case (body, bindings) of
        (_, []) -> Right (parameters, body)
        (EVar wrapper, _)
            | any (`isPrefixOf` showIdent wrapper) ["fcnD$", "fcnS$"]
            , [nested] <- [equations | Fcn identifier equations <- bindings, identifier == wrapper]
            , let dictionaries = [binding | binding@(Fcn identifier _) <- bindings, identifier /= wrapper]
            , length dictionaries + 1 == length bindings
            , all dictionaryBinding dictionaries -> do
                (nestedParameters, nestedBody) <- equation nested
                Right (parameters ++ nestedParameters, if null dictionaries then nestedBody else ELet dictionaries nestedBody)
        _ -> Left "Source where bindings are not supported"
  where
    variable (EVar identifier) = Right identifier
    variable _ = Left "Only variable parameters are supported"
    dictionaryBinding (Fcn identifier [Eqn [] (EAlts [([], _)] [])]) = "dict$" `isPrefixOf` showIdent identifier
    dictionaryBinding _ = False
equation _ = Left "Expected one unguarded equation without where bindings"

-- | Read local value and function bindings. Signatures contain no runtime code.
sourceBindings :: [EDef] -> Either String [(Ident, Expr)]
sourceBindings = fmap concat . mapM binding
  where
    binding Sign{} = Right []
    binding (Fcn identifier equations) = do
        (parameters, body) <- equation equations
        Right [(identifier, eLam (map EVar parameters) body)]
    binding _ = Left "Only variable let bindings are supported"

-- | Find lexical dependencies without treating shadowed names as recursive uses.
freeVariables :: Expr -> Either String [Ident]
freeVariables source = case source of
    EVar identifier -> Right [identifier]
    ELit{} -> Right []
    EApp operator argument -> (++) <$> freeVariables operator <*> freeVariables argument
    ELam _ equations -> do
        (parameters, body) <- equation equations
        filter (`notElem` parameters) <$> freeVariables body
    ELet bindings body -> do
        values <- sourceBindings bindings
        free <- concat <$> mapM freeVariables (body : map snd values)
        Right (filter (`notElem` map fst values) free)
    EIf condition yes no -> concat <$> mapM freeVariables [condition, yes, no]
    ETuple values -> concat <$> mapM freeVariables values
    _ -> Left ("Unsupported checked expression: " ++ take 160 (showExpr source))

-- | Select one monomorphic function with variable parameters and one equation.
function :: TModule [EDef] -> String -> [String] -> Either String ([Ident], Expr)
function checked name expected = do
    signature <- case [ty | Sign names ty <- definitions, mkIdent name `elem` names] of
        [ty] -> Right ty
        _ -> Left ("Missing monomorphic signature: " ++ name)
    let (arguments, result) = getArrows signature
        actual = map typeName (arguments ++ [result])
    if actual /= map Just expected
        then Left ("Unsupported signature for " ++ name ++ ": " ++ showEType signature)
        else case [equations | Fcn identifier equations <- definitions, unQualIdent identifier == mkIdent name] of
            [equations] -> equation equations
            _ -> Left ("Expected one unguarded equation: " ++ name)
  where
    definitions = tBindingsOf checked
    typeName (EVar identifier) = Just (showIdent identifier)
    typeName _ = Nothing

-- | Translate normalized scalar expressions with exact primitive validation.
expression :: [(Ident, Scalar input)] -> Expr -> Either String (Scalar input)
expression variables source = case source of
    EVar identifier -> case lookup identifier variables of
        Just value -> Right value
        Nothing -> Left ("Unsupported value: " ++ showIdent identifier)
    ELit _ (LFloat value) -> Right (Floating (FloatConstant value))
    EIf condition yes no -> do
        predicate <- expression variables condition
        trueResult <- expression variables yes
        falseResult <- expression variables no
        case (predicate, trueResult, falseResult) of
            (Boolean test, Floating trueGraph, Floating falseGraph) -> Right (Floating (Compose Select (Pair test (Pair trueGraph falseGraph))))
            (Boolean test, Integral trueGraph, Integral falseGraph) -> Right (Integral (Compose Select (Pair test (Pair trueGraph falseGraph))))
            _ -> Left "Unsupported conditional types"
    EApp{} -> do
        let (operator, arguments) = application source
        case (operator, arguments) of
            (EVar identifier, dictionary : remaining) -> do
                let name = showIdent identifier
                initial <- dictionaryNames dictionary
                (resolved, operands) <- case (name, initial, remaining) of
                    ("Data.RealFrac.round", ["inst$Data.RealFrac.RealFrac@Primitives.Float"], resultDictionary : values) -> do
                        result <- dictionaryNames resultDictionary
                        Right (initial ++ result, values)
                    _ -> Right (initial, remaining)
                case (showIdent identifier, resolved, operands) of
                    ("Data.Num.fromInteger", ["inst$Data.Num.Num@Data.Int.Int32.Int32"], [ELit _ (LInteger value)])
                        | value >= toInteger (minBound :: Int32)
                        , value <= toInteger (maxBound :: Int32) ->
                            Right (Integral (IntConstant (fromInteger value)))
                    ("Data.Num.fromInteger", ["inst$Data.Num.Num@Primitives.Float"], [ELit _ (LInteger value)]) ->
                        Right (Floating (FloatConstant (fromInteger value)))
                    ("Data.Fractional.fromRational", ["inst$Data.Fractional.Fractional@Primitives.Float"], [ELit _ (LRat value)]) ->
                        Right (Floating (FloatConstant (fromRational value)))
                    _ -> do
                        values <- mapM (expression variables) operands
                        primitive (showIdent identifier) resolved values
            _ -> Left "Only supported primitive applications can be compiled"
    _ -> Left ("Unsupported checked expression: " ++ take 160 (showExpr source))

-- | Read normalized concrete dictionaries without changing their identities.
dictionaryNames :: Expr -> Either String [String]
dictionaryNames (EVar identifier) = Right [showIdent identifier]
dictionaryNames (ETuple [left, right]) = (++) <$> dictionaryNames left <*> dictionaryNames right
dictionaryNames _ = Left "Only known concrete dictionaries are supported"

-- | Separate an application into its operator and ordered arguments.
application :: Expr -> (Expr, [Expr])
application source = collect source []
  where
    collect (EApp operator argument) arguments = collect operator (argument : arguments)
    collect operator arguments = (operator, arguments)

-- | Interpret only the exact upstream primitive and dictionary combinations.
primitive :: String -> [String] -> [Scalar input] -> Either String (Scalar input)
primitive name dictionary arguments = case (name, dictionary, arguments) of
    ("Data.Num.+", ["inst$Data.Num.Num@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose AddFloat (Pair left right)))
    ("Data.Num.-", ["inst$Data.Num.Num@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose SubFloat (Pair left right)))
    ("Data.Num.*", ["inst$Data.Num.Num@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose MulFloat (Pair left right)))
    ("Data.Fractional./", ["inst$Data.Fractional.Fractional@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose DivFloat (Pair left right)))
    ("Data.Num.abs", ["inst$Data.Num.Num@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose AbsFloat value))
    ("Data.Num.negate", ["inst$Data.Num.Num@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose NegFloat value))
    ("Data.Ord.min", ["inst$Data.Ord.Ord@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose MinFloat (Pair left right)))
    ("Data.Ord.max", ["inst$Data.Ord.Ord@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose MaxFloat (Pair left right)))
    ("Data.Ord.>", ["inst$Data.Ord.Ord@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Boolean (Compose GreaterFloat (Pair left right)))
    ("Data.Floating.sqrt", ["inst$Data.Floating.Floating@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose SqrtFloat value))
    ("Data.Floating.exp", ["inst$Data.Floating.Floating@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose ExpFloat value))
    ("Data.Floating.sin", ["inst$Data.Floating.Floating@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose SinFloat value))
    ("Data.Floating.cos", ["inst$Data.Floating.Floating@Primitives.Float"], [Floating value]) ->
        Right (Floating (Compose CosFloat value))
    ("Data.Floating.**", ["inst$Data.Floating.Floating@Primitives.Float"], [Floating left, Floating right]) ->
        Right (Floating (Compose PowFloat (Pair left right)))
    ("Data.RealFrac.round", ["inst$Data.RealFrac.RealFrac@Primitives.Float", "inst$Data.Integral.Integral@Data.Int.Int32.Int32"], [Floating value]) ->
        Right (Integral (Compose RoundFloat value))
    ("Data.Integral.fromIntegral", ["inst$Data.Integral.Integral@Data.Int.Int32.Int32", "inst$Data.Num.Num@Primitives.Float"], [Integral value]) ->
        Right (Floating (Compose IntToFloat value))
    ("Data.Num.+", ["inst$Data.Num.Num@Data.Int.Int32.Int32"], [Integral left, Integral right]) ->
        Right (Integral (Compose AddInt (Pair left right)))
    ("Data.Num.negate", ["inst$Data.Num.Num@Data.Int.Int32.Int32"], [Integral value]) ->
        Right (Integral (Compose NegInt value))
    ("Data.Eq.==", ["inst$Data.Eq.Eq@Data.Int.Int32.Int32"], [Integral left, Integral right]) ->
        Right (Boolean (Compose EqualInt (Pair left right)))
    _ -> Left ("Unsupported primitive or dictionary: " ++ name ++ " / " ++ show dictionary)
