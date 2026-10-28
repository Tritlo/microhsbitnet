{-# LANGUAGE GADTs #-}

module CCC (Cat (..), Code (..), runCat, emitCat, emitC, emitWGSL, emitWGSLRelaxed, emitBinary64, wgslPrelude, showCat) where

import Control.Exception (ArithException (Overflow), throw)
import Data.Int (Int32)
import Data.Word (Word32)
import FloatWGSL (wgslPrelude)

-- | Typed categorical operations and the supported scalar primitives.
data Cat input output where
    Id :: Cat value value
    Compose :: Cat middle output -> Cat input middle -> Cat input output
    First :: Cat (left, right) left
    Second :: Cat (left, right) right
    Pair :: Cat input left -> Cat input right -> Cat input (left, right)
    Terminal :: Cat input ()
    Curry :: Cat (input, argument) result -> Cat input (argument -> result)
    Apply :: Cat (argument -> result, argument) result
    FloatConstant :: Float -> Cat input Float
    IntConstant :: Int32 -> Cat input Int32
    AddFloat :: Cat (Float, Float) Float
    SubFloat :: Cat (Float, Float) Float
    MulFloat :: Cat (Float, Float) Float
    DivFloat :: Cat (Float, Float) Float
    NegFloat :: Cat Float Float
    AbsFloat :: Cat Float Float
    MinFloat :: Cat (Float, Float) Float
    MaxFloat :: Cat (Float, Float) Float
    SqrtFloat :: Cat Float Float
    ExpFloat :: Cat Float Float
    SinFloat :: Cat Float Float
    CosFloat :: Cat Float Float
    PowFloat :: Cat (Float, Float) Float
    RoundFloat :: Cat Float Int32
    IntToFloat :: Cat Int32 Float
    AddInt :: Cat (Int32, Int32) Int32
    NegInt :: Cat Int32 Int32
    EqualInt :: Cat (Int32, Int32) Bool
    GreaterFloat :: Cat (Float, Float) Bool
    Select :: Cat (Bool, (value, value)) value

-- | Values used to interpret a graph during code generation.
data Code value where
    FloatCode :: String -> Code Float
    IntCode :: String -> Code Int32
    BoolCode :: String -> Code Bool
    PairCode :: Code left -> Code right -> Code (left, right)
    UnitCode :: Code ()
    FunctionCode :: (Code argument -> Either String (Code result)) -> Code (argument -> result)

-- | Interpret a categorical graph as a Haskell function.
runCat :: Cat input output -> input -> output
runCat graph input = case graph of
    Id -> input
    Compose after before -> runCat after (runCat before input)
    First -> fst input
    Second -> snd input
    Pair left right -> (runCat left input, runCat right input)
    Terminal -> ()
    Curry body -> \argument -> runCat body (input, argument)
    Apply -> fst input (snd input)
    FloatConstant value -> value
    IntConstant value -> value
    AddFloat -> fst input + snd input
    SubFloat -> fst input - snd input
    MulFloat -> fst input * snd input
    DivFloat -> fst input / snd input
    NegFloat -> negate input
    AbsFloat -> abs input
    MinFloat -> min (fst input) (snd input)
    MaxFloat -> max (fst input) (snd input)
    SqrtFloat -> sqrt input
    ExpFloat -> exp input
    SinFloat -> sin input
    CosFloat -> cos input
    PowFloat -> fst input ** snd input
    RoundFloat -> checkedInt (round input)
    IntToFloat -> fromIntegral input
    AddInt -> checkedInt (toInteger (fst input) + toInteger (snd input))
    NegInt -> checkedInt (negate (toInteger input))
    EqualInt -> fst input == snd input
    GreaterFloat -> fst input > snd input
    Select -> if fst input then fst (snd input) else snd (snd input)

-- | Preserve the pinned MicroHs Int32 overflow exception.
checkedInt :: Integer -> Int32
checkedInt value
    | value < toInteger (minBound :: Int32) || value > toInteger (maxBound :: Int32) = throw Overflow
    | otherwise = fromInteger value

-- | Interpret a graph as first-order Metal expressions.
emitCat :: Cat input output -> Code input -> Either String (Code output)
emitCat = emitCode Metal

-- | Emit native C scalar expressions for build-time constant evaluation.
emitC :: Cat input output -> Code input -> Either String (Code output)
emitC = emitCode NativeC

{- | Emit WGSL expressions. Select evaluates both operands.
All Int32 operations must avoid overflow, including unselected operands.
Float inputs must be finite. Rounded values must fit Int32.
-}
emitWGSL :: Cat input output -> Code input -> Either String (Code output)
emitWGSL = emitCode WGSL

-- | Use native WGSL floats for the separately validated approximate head.
emitWGSLRelaxed :: Cat input output -> Code input -> Either String (Code output)
emitWGSLRelaxed = emitCode WGSLRelaxed

-- | Interpret Float addition and multiplication as sequential binary64 operations.
emitBinary64 :: Cat input output -> Code input -> Either String (Code output)
emitBinary64 graph input = case graph of
    Id -> Right input
    Compose after before -> emitBinary64 before input >>= emitBinary64 after
    First -> case input of PairCode left _ -> Right left
    Second -> case input of PairCode _ right -> Right right
    Pair left right -> PairCode <$> emitBinary64 left input <*> emitBinary64 right input
    Terminal -> Right UnitCode
    AddFloat -> case input of
        PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode ("add_f64(" ++ left ++ ", " ++ right ++ ")"))
    MulFloat -> case input of
        PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode ("mul_f64(" ++ left ++ ", " ++ right ++ ")"))
    _ -> Left ("Unsupported binary64 operation: " ++ showCat graph)

-- | Select the syntax for target-specific scalar operations.
data Target = Metal | NativeC | WGSL | WGSLRelaxed

-- | Interpret the shared graph with target integer and conditional syntax.
emitCode :: Target -> Cat input output -> Code input -> Either String (Code output)
emitCode target graph input = case graph of
    Id -> Right input
    Compose after before -> emitCode target before input >>= emitCode target after
    First -> case input of PairCode left _ -> Right left
    Second -> case input of PairCode _ right -> Right right
    Pair left right -> PairCode <$> emitCode target left input <*> emitCode target right input
    Terminal -> Right UnitCode
    Curry body -> Right (FunctionCode (\argument -> emitCode target body (PairCode input argument)))
    Apply -> case input of PairCode (FunctionCode function) argument -> function argument
    FloatConstant value
        | isNaN value || isInfinite value -> Left "GPU constants must be finite Float values"
        | otherwise -> Right (FloatCode (show value ++ "f"))
    IntConstant value ->
        let literal = show (fromIntegral value :: Word32) ++ "u"
            cast = case target of Metal -> "as_type<int>"; _ -> "bitcast<i32>"
         in Right (IntCode (case target of NativeC -> "((int32_t)" ++ literal ++ ")"; _ -> cast ++ "(" ++ literal ++ ")"))
    AddFloat -> case input of
        PairCode (FloatCode left) (FloatCode right) ->
            Right (FloatCode (binary "+" left right))
    SubFloat -> case input of PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode (binary "-" left right))
    MulFloat -> case input of
        PairCode (FloatCode left) (FloatCode right) ->
            Right (FloatCode (case target of WGSL -> "multiply32(" ++ left ++ ", " ++ right ++ ")"; _ -> binary "*" left right))
    DivFloat -> case input of
        PairCode (FloatCode left) (FloatCode right) ->
            Right (FloatCode (case target of WGSL -> "divide32(" ++ left ++ ", " ++ right ++ ")"; _ -> binary "/" left right))
    NegFloat -> case input of FloatCode value -> Right (FloatCode ("(-(" ++ value ++ "))"))
    AbsFloat -> case input of FloatCode value -> Right (FloatCode ((case target of NativeC -> "fabsf("; _ -> "abs(") ++ value ++ ")"))
    MinFloat -> case input of PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode ((case target of NativeC -> "fminf("; _ -> "min(") ++ left ++ ", " ++ right ++ ")"))
    MaxFloat -> case input of PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode ((case target of NativeC -> "fmaxf("; _ -> "max(") ++ left ++ ", " ++ right ++ ")"))
    SqrtFloat -> case input of
        FloatCode value -> Right (FloatCode ((case target of WGSL -> "sqrt32("; NativeC -> "sqrtf("; _ -> "sqrt(") ++ value ++ ")"))
    ExpFloat -> case input of FloatCode value -> Right (FloatCode ((case target of NativeC -> "expf("; WGSL -> "exp32("; _ -> "exp(") ++ value ++ ")"))
    SinFloat -> case input of FloatCode value -> Right (FloatCode ((case target of NativeC -> "sinf("; _ -> "sin(") ++ value ++ ")"))
    CosFloat -> case input of FloatCode value -> Right (FloatCode ((case target of NativeC -> "cosf("; _ -> "cos(") ++ value ++ ")"))
    PowFloat -> case input of PairCode (FloatCode left) (FloatCode right) -> Right (FloatCode ((case target of NativeC -> "powf("; _ -> "pow(") ++ left ++ ", " ++ right ++ ")"))
    RoundFloat -> case input of
        FloatCode value ->
            let conversion = case target of Metal -> "int(rint("; NativeC -> "((int32_t)rintf("; _ -> "i32(round("
             in Right (IntCode (conversion ++ value ++ "))"))
    IntToFloat -> case input of
        IntCode value ->
            let conversion = case target of Metal -> "float("; NativeC -> "((float)"; _ -> "f32("
             in Right (FloatCode (conversion ++ value ++ ")"))
    AddInt -> case input of PairCode (IntCode left) (IntCode right) -> Right (IntCode (binary "+" left right))
    NegInt -> case input of IntCode value -> Right (IntCode ("(-(" ++ value ++ "))"))
    EqualInt -> case input of PairCode (IntCode left) (IntCode right) -> Right (BoolCode (binary "==" left right))
    GreaterFloat -> case input of PairCode (FloatCode left) (FloatCode right) -> Right (BoolCode (binary ">" left right))
    Select -> case input of
        PairCode (BoolCode condition) (PairCode (FloatCode yes) (FloatCode no)) -> Right (FloatCode (conditional target condition yes no))
        PairCode (BoolCode condition) (PairCode (IntCode yes) (IntCode no)) -> Right (IntCode (conditional target condition yes no))
        _ -> Left "Only scalar conditional results can be emitted"

-- | Render a binary expression with explicit grouping.
binary :: String -> String -> String -> String
binary operator left right = "(" ++ left ++ " " ++ operator ++ " " ++ right ++ ")"

-- | Render a conditional with the target evaluation behavior.
conditional :: Target -> String -> String -> String -> String
conditional Metal condition yes no = "(" ++ condition ++ " ? " ++ yes ++ " : " ++ no ++ ")"
conditional NativeC condition yes no = "(" ++ condition ++ " ? " ++ yes ++ " : " ++ no ++ ")"
conditional _ condition yes no = "select(" ++ no ++ ", " ++ yes ++ ", " ++ condition ++ ")"

-- | Display the structure of a categorical graph.
showCat :: Cat input output -> String
showCat graph = case graph of
    Id -> "id"
    Compose after before -> "(" ++ showCat after ++ " . " ++ showCat before ++ ")"
    First -> "fst"
    Second -> "snd"
    Pair left right -> "<" ++ showCat left ++ ", " ++ showCat right ++ ">"
    Terminal -> "terminal"
    Curry body -> "curry " ++ showCat body
    Apply -> "apply"
    FloatConstant value -> "const " ++ show value
    IntConstant value -> "const " ++ show value
    AddFloat -> "addFloat"
    SubFloat -> "subFloat"
    MulFloat -> "mulFloat"
    DivFloat -> "divFloat"
    NegFloat -> "negFloat"
    AbsFloat -> "absFloat"
    MinFloat -> "minFloat"
    MaxFloat -> "maxFloat"
    SqrtFloat -> "sqrtFloat"
    ExpFloat -> "expFloat"
    SinFloat -> "sinFloat"
    CosFloat -> "cosFloat"
    PowFloat -> "powFloat"
    RoundFloat -> "roundFloatToInt32"
    IntToFloat -> "int32ToFloat"
    AddInt -> "addInt32"
    NegInt -> "negInt32"
    EqualInt -> "equalInt32"
    GreaterFloat -> "greaterFloat"
    Select -> "select"
