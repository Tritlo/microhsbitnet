{-# LANGUAGE GADTs #-}

module Compiler (compileScalar, compileIntPair, compileFloatPair, compileFloatTriple, compileFloatQuadruple, compileFloatPredicate, compileQuantize, compileRescale) where

import CCC
import Data.Int (Int32)
import Data.List (isPrefixOf)
import MicroHs.Expr
import MicroHs.Ident (Ident, mkIdent, showIdent, unQualIdent)
import MicroHs.TypeCheck (TModule, tBindingsOf)

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
            result <- expression [] [(parameter, Floating Id)] [] body
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
            result <- expression [] [(left, Integral First), (right, Integral Second)] [] body
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
            result <- expression [] [(left, Floating First), (right, Floating Second)] [] body
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
            result <- expression [] [(first, Floating First), (second, Floating (Compose First Second)), (third, Floating (Compose Second Second))] [] body
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
            result <- expression [] [(first, Floating First), (second, Floating (Compose First Second)), (third, Floating (Compose First (Compose Second Second))), (fourth, Floating (Compose Second (Compose Second Second)))] [] body
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
            result <- expression [] [(left, Floating First), (right, Floating Second)] [] body
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
            result <- expression [] [(input, Floating First), (maximum, Floating Second)] [] body
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
            result <- expression [] [(accumulator, Integral First), (weightScale, Floating (Compose First Second)), (maximum, Floating (Compose Second Second))] [] body
            case result of
                Floating graph -> Right graph
                _ -> Left "Expected a Float result"
        _ -> Left "Expected an Int32 accumulator and two Float scales"

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
            [[Eqn patterns (EAlts [([], body)] [])]] -> do
                parameters <- mapM variable patterns
                Right (parameters, body)
            _ -> Left ("Expected one unguarded equation: " ++ name)
  where
    definitions = tBindingsOf checked
    typeName (EVar identifier) = Just (showIdent identifier)
    typeName _ = Nothing
    variable (EVar identifier) = Right identifier
    variable _ = Left "Only variable parameters are supported"

-- | Translate the checked scalar fragment to categorical operations.
expression :: [(Ident, Expr)] -> [(Ident, Scalar input)] -> [Ident] -> Expr -> Either String (Scalar input)
expression definitions variables active source = case source of
    EVar identifier -> case lookup identifier variables of
        Just value -> Right value
        Nothing -> case lookup identifier definitions of
            Just body | identifier `notElem` active -> expression definitions variables (identifier : active) body
            _ -> Left ("Unsupported or recursive value: " ++ showIdent identifier)
    ELit _ (LFloat value) -> Right (Floating (FloatConstant value))
    ELet bindings body -> do
        values <- mapM valueBinding bindings
        expression (values ++ definitions) variables active body
    EIf condition yes no -> do
        predicate <- expression definitions variables active condition
        trueResult <- expression definitions variables active yes
        falseResult <- expression definitions variables active no
        case (predicate, trueResult, falseResult) of
            (Boolean test, Floating trueGraph, Floating falseGraph) -> Right (Floating (Compose Select (Pair test (Pair trueGraph falseGraph))))
            (Boolean test, Integral trueGraph, Integral falseGraph) -> Right (Integral (Compose Select (Pair test (Pair trueGraph falseGraph))))
            _ -> Left "Unsupported conditional types"
    EApp{} -> do
        let (operator, arguments) = application source
        case (operator, arguments) of
            (EVar identifier, dictionary : remaining) -> do
                let name = showIdent identifier
                initial <- dictionaryNames definitions [] dictionary
                (resolved, operands) <- case (name, initial, remaining) of
                    ("Data.RealFrac.round", ["inst$Data.RealFrac.RealFrac@Primitives.Float"], resultDictionary : values) -> do
                        result <- dictionaryNames definitions [] resultDictionary
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
                        values <- mapM (expression definitions variables active) operands
                        primitive (showIdent identifier) resolved values
            _ -> Left "Only supported primitive applications can be compiled"
    _ -> Left ("Unsupported checked expression: " ++ take 160 (showExpr source))

-- | Accept only generated dictionary bindings.
valueBinding :: EDef -> Either String (Ident, Expr)
valueBinding (Fcn identifier [Eqn [] (EAlts [([], body)] [])])
    | "dict$" `isPrefixOf` showIdent identifier = Right (identifier, body)
valueBinding _ = Left "Only generated dictionary bindings are supported; source let bindings require closure analysis"

-- | Resolve concrete dictionary aliases and pairs. Reject cycles.
dictionaryNames :: [(Ident, Expr)] -> [Ident] -> Expr -> Either String [String]
dictionaryNames definitions active (EVar identifier)
    | identifier `elem` active = Left "Recursive dictionary"
    | Just body <- lookup identifier definitions = dictionaryNames definitions (identifier : active) body
    | otherwise = Right [showIdent identifier]
dictionaryNames definitions active (ETuple [left, right]) =
    (++) <$> dictionaryNames definitions active left <*> dictionaryNames definitions active right
dictionaryNames _ _ _ = Left "Only known concrete dictionaries are supported"

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
