{-# LANGUAGE GADTs #-}

{- | Read the combinator term MicroHs actually emits and rebuild the CCC graph
the WGSL backend already consumes.

B is composition, C is flip, and S' f g h = f . <g, h>. S f g x = f x (g x)
is the same duplication when f is a known primitive, so the kernel stays
first-order. S needs a real exponential, apply . <f, g>, only when the
operator is an unknown runtime function. Numeric kernels here do not need one.
-}
module Combinators (
    translateFloat,
    translateBool,
    translateInt,
    translateArray,
    combinatorOf,
    kernelFloat1,
    kernelFloat2,
    kernelFloat4,
    kernelBool,
    checkCombinators,
) where

import ArrayCompiler (compileArray)
import ArrayGraph (Captures (..), Program (..), Ref (..), Step (..), runProgram, runVectors)
import CCC (Cat (..), runCat, showCat)
import Compiler (compileFloatPair, compileFloatPredicate, compileFloatQuadruple, compileFloatTriple, compileIntPair, compileScalar)
import Control.Monad (foldM, unless)
import Control.Monad.State.Strict (StateT, get, modify, runStateT)
import Control.Monad.Trans.Class (lift)
import Data.Int (Int32)
import Data.List (nub)
import Frontend (checkedModule)
import MicroHs.Abstract (compileOpt)
import MicroHs.Desugar (desugar)
import MicroHs.Exp (Exp (..))
import MicroHs.Expr (EDef (..), EType, Expr (EApp, EVar), Lit (..), eqEType, getArrows, showEType)
import MicroHs.Flags (defaultFlags)
import MicroHs.Ident (mkIdent, showIdent, unQualIdent)
import MicroHs.TypeCheck (TModule, tBindingsOf)
import qualified Programs

data Leaf = LElem | LArg Int | LLeft | LRight | LCap Int | LBound Int
    deriving (Eq, Show)

data Prim
    = PAdd
    | PSub
    | PMul
    | PDiv
    | PNeg
    | PAbs
    | PMin
    | PMax
    | PSqrt
    | PExp
    | PSin
    | PCos
    | PPow
    | PGt
    | PAddI
    | PNegI
    deriving (Eq, Show)

data U
    = ULeaf Leaf
    | UCompose U U
    | UPair U U
    | UConstF Float
    | UConstI Int32
    | UPrim Prim
    deriving (Eq, Show)

data Val
    = Arrow U
    | Fun (Val -> M Val)
    | Vec Int
    | Frozen Int
    | Dict String
    | Bits Int

data Build = Build {nextId :: Int, steps :: [Step]}

type M = StateT Build (Either String)

-- | Curried Float function of the given arity, as a first-order arrow.
translateFloat :: Int -> TModule [EDef] -> String -> Either String U
translateFloat arity checked name = do
    unless (arity > 0) (Left "A scalar function needs at least one argument")
    validateSignature checked name (replicate (arity + 1) floatType)
    evalNamed checked name >>= finishFloat arity

-- | Translate a checked comparison of two Float arguments.
translateBool :: TModule [EDef] -> String -> Either String U
translateBool checked name = do
    validateSignature checked name [floatType, floatType, boolType]
    evalNamed checked name >>= finishFloat 2

-- | Translate a checked binary Int32 function.
translateInt :: TModule [EDef] -> String -> Either String U
translateInt checked name = do
    validateSignature checked name (replicate 3 intType)
    evalNamed checked name >>= finishFloat 2

-- | Function of @inputs@ vector arguments, as the existing array graph.
translateArray :: Int -> TModule [EDef] -> String -> Either String Program
translateArray inputs checked name = do
    unless (inputs > 0) (Left "An array function needs at least one argument")
    validateSignature checked name (replicate (inputs + 1) vectorType)
    value <- evalNamed checked name
    (result, build) <- runStateT (applyAll value (map Vec [0 .. inputs - 1])) (Build inputs [])
    case result of
        Vec identifier -> Right (Program (reverse (steps build)) (VectorRef identifier))
        _ -> Left (name ++ " did not return a vector")

combinatorOf :: TModule [EDef] -> String -> Either String String
combinatorOf checked name = paren <$> optimized checked name

reifyFloat :: (Leaf -> Either String (Cat input Float)) -> U -> Either String (Cat input Float)
reifyFloat = reify

kernelFloat1 :: TModule [EDef] -> String -> Either String (Cat Float Float)
kernelFloat1 checked name = reify proj1 =<< translateFloat 1 checked name

kernelFloat2 :: TModule [EDef] -> String -> Either String (Cat (Float, Float) Float)
kernelFloat2 checked name = reify proj2 =<< translateFloat 2 checked name

kernelFloat4 :: TModule [EDef] -> String -> Either String (Cat (Float, (Float, (Float, Float))) Float)
kernelFloat4 checked name = reify proj4 =<< translateFloat 4 checked name

kernelBool :: TModule [EDef] -> String -> Either String (Cat (Float, Float) Bool)
kernelBool checked name = reifyBool =<< translateBool checked name

reifyBool :: U -> Either String (Cat (Float, Float) Bool)
reifyBool (UCompose (UPrim PGt) arg) = do
    (left, right) <- asPair arg
    Compose GreaterFloat <$> (Pair <$> reify proj2 left <*> reify proj2 right)
reifyBool _ = Left "A combinator predicate did not reduce to a Float comparison"

reifyInt :: U -> Either String (Cat (Int32, Int32) Int32)
reifyInt = reifyI projInt

-- | Match combinator kernels against the expression compiler on the same source.
checkCombinators :: TModule [EDef] -> TModule [EDef] -> TModule [EDef] -> IO ()
checkCombinators programs layer arrays = do
    sameF proj1 programs "scalar" 1 (compileScalar programs "scalar") samples1
    sameF proj1 layer "square" 1 (compileScalar layer "square") samples1
    sameF proj1 layer "cosine" 1 (compileScalar layer "cosine") samples1
    sameF proj1 layer "sine" 1 (compileScalar layer "sine") samples1
    sameF proj2 layer "plus" 2 (compileFloatPair layer "plus") pairs
    sameF proj2 layer "product" 2 (compileFloatPair layer "product") pairs
    sameF proj2 layer "normalizer" 2 (compileFloatPair layer "normalizer") positivePairs
    sameF proj2 layer "frequency" 2 (compileFloatPair layer "frequency") positivePairs
    sameF proj2 layer "reluMul" 2 (compileFloatPair layer "reluMul") pairs
    sameF proj2 programs "peak" 2 (compileFloatPair programs "peak") pairs
    sameF proj3 layer "weighted" 3 (compileFloatTriple layer "weighted") triples
    sameF proj4 layer "rotateFirst" 4 (compileFloatQuadruple layer "rotateFirst") quads
    sameF proj4 layer "rotateSecond" 4 (compileFloatQuadruple layer "rotateSecond") quads
    sameB layer
    sameI programs
    sameA arrays "softmax" 1 (\value -> agree "softmax" value (snd <$> Programs.arrayInputs))
    sameA arrays "residual" 2 (\_ -> agree2 "residual" [1, -2] [3, 4])
    sameA arrays "gated" 2 (\_ -> agree2 "gated" [-1, 2, 0.5] [4, 5, 6])
    sameA arrays "rmsNorm" 2 (\_ -> agree2 "rmsNorm" [1, 2, 3, 4] [1, 1, 0.5, 2])
    term <- either fail pure (combinatorOf programs "scalar")
    unless (term == expectedScalar) (fail ("scalar combinator changed:\n" ++ term))
    translated <- either fail pure (reify proj1 =<< translateFloat 1 programs "scalar")
    existing <- either fail pure (compileScalar programs "scalar")
    unless (showCat translated == showCat existing) $
        fail ("scalar CCC graph changed:\n translated " ++ showCat translated ++ "\n existing    " ++ showCat existing)
    reject programs "quantize" [floatType, floatType, intType]
    reject programs "rescale" [intType, floatType, floatType, floatType]
    reject programs "contribution" [intType, intType, intType]
    checkBoundaries layer programs
    putStrLn "Combinator to CCC translation matches the existing scalar and array kernels"
  where
    sameA module_ name inputs check = do
        translated <- either (\message -> fail (name ++ ": " ++ message)) pure (translateArray inputs module_ name)
        reference <- either fail pure (compileArray module_ name)
        check (translated, reference)
    agree name (translated, reference) groups =
        mapM_ (\group -> agree1 name translated reference group) groups
    agree1 name translated reference group = do
        actual <- either fail pure (runProgram translated group)
        expected <- either fail pure (runProgram reference group)
        unless (actual == expected) (fail (name ++ " combinator graph differs"))
    agree2 name left right = do
        translated <- either fail pure (translateArray 2 arrays name)
        reference <- either fail pure (compileArray arrays name)
        actual <- either fail pure (runVectors translated [left, right])
        expected <- either fail pure (runVectors reference [left, right])
        unless (actual == expected) (fail (name ++ " combinator graph differs: " ++ show actual ++ " vs " ++ show expected))

expectedScalar :: String
expectedScalar =
    "(((S' (Data.Num.* inst$Data.Num.Num@Primitives.Float)) ((C (Data.Num.+ inst$Data.Num.Num@Primitives.Float)) 1.0)) ((C (Data.Num.- inst$Data.Num.Num@Primitives.Float)) 1.0))"

samples1 :: [Float]
samples1 = [-2, -0.5, 0, 0.25, 1, 3]

pairs :: [(Float, Float)]
pairs = [(0, 0), (1, -2), (-3, 4), (0.5, 0.5), (8, 0.125)]

positivePairs :: [(Float, Float)]
positivePairs = [(1, 1), (4, 2), (0.5, 8), (9, 3)]

triples :: [(Float, (Float, Float))]
triples = [(1, (2, 3)), (0, (1, -1)), (-2, (0.5, 4))]

quads :: [(Float, (Float, (Float, Float)))]
quads = [(1, (0, (1, 0))), (0.5, (-1, (0.5, 2))), (2, (3, (-1, 0.25)))]

sameF :: (Leaf -> Either String (Cat input Float)) -> TModule [EDef] -> String -> Int -> Either String (Cat input Float) -> [input] -> IO ()
sameF project checked name arity existing inputs = do
    graph <- either (\message -> fail (name ++ ": " ++ message)) pure (reify project =<< translateFloat arity checked name)
    reference <- either fail pure existing
    unless (all (\input -> runCat graph input == runCat reference input) inputs) $
        fail (name ++ " combinator kernel differs from the expression compiler")

sameB :: TModule [EDef] -> IO ()
sameB layer = do
    graph <- either (\message -> fail ("better: " ++ message)) pure (reifyBool =<< translateBool layer "better")
    reference <- either fail pure (compileFloatPredicate layer "better")
    let inputs = [(0, 1), (1, 1), (2, -1), (-3, -4)] :: [(Float, Float)]
    unless (all (\input -> runCat graph input == runCat reference input) inputs) $
        fail "better combinator kernel differs from the expression compiler"

sameI :: TModule [EDef] -> IO ()
sameI programs = do
    graph <- either (\message -> fail ("combine: " ++ message)) pure (reifyInt =<< translateInt programs "combine")
    reference <- either fail pure (compileIntPair programs "combine")
    let inputs = [(0, 0), (2, -5), (100, 3)] :: [(Int32, Int32)]
    unless (all (\input -> runCat graph input == runCat reference input) inputs) $
        fail "combine combinator kernel differs from the expression compiler"

-- | Check unsupported expressions after their signatures have been validated.
reject :: TModule [EDef] -> String -> [EType] -> IO ()
reject checked name expected = do
    either fail pure (validateSignature checked name expected)
    rejects name (evalNamed checked name >>= finishFloat (length expected - 1))

-- | Check lexical array scope, captured results, and public entry signatures.
checkBoundaries :: TModule [EDef] -> TModule [EDef] -> IO ()
checkBoundaries layer programs = do
    checked <- checkedModule "spike/ccc/Unsupported.hs"
    mapM_ (\name -> rejects name (translateArray 1 checked name)) ["dependentArray", "dependentArithmetic"]
    rejects "Int32 identity as Float" (kernelFloat1 checked "intIdentity")
    rejects "Int32 pair as Float" (kernelFloat2 checked "intFirst")
    rejects "Float result as Bool" (translateBool layer "plus")
    rejects "Float arguments as Int32" (translateInt programs "peak")
    rejects "Int32 vector as Float vector" (translateArray 1 checked "intVectorIdentity")
    rejects "scalar argument as vector" (translateArray 2 checked "ignoredScalar")
    mapM_
        ( \(name, inputs, expected) -> do
            graph <- either fail pure (translateArray (length inputs) checked name)
            actual <- either fail pure (runVectors graph inputs)
            unless (actual == expected) (fail (name ++ " combinator graph differs: " ++ show actual))
        )
        [ ("capturedMap", [[1, 2]], [3, 3])
        , ("capturedZip", [[1, 2], [8, 9]], [3, 3])
        , ("independentNested", [[1, 2]], [6, 7])
        ]

-- | Fail when an unsupported translation succeeds.
rejects :: String -> Either String value -> IO ()
rejects _ (Left _) = pure ()
rejects name (Right _) = fail (name ++ " was accepted")

-- | Match a monomorphic source signature before types are removed.
validateSignature :: TModule [EDef] -> String -> [EType] -> Either String ()
validateSignature checked name expected = do
    signature <- case [ty | Sign names ty <- tBindingsOf checked, mkIdent name `elem` names] of
        [ty] -> Right ty
        _ -> Left ("Missing monomorphic signature: " ++ name)
    let (arguments, result) = getArrows signature
        actual = arguments ++ [result]
    unless (length actual == length expected && and (zipWith eqEType actual expected)) $
        Left ("Unsupported signature for " ++ name ++ ": " ++ showEType signature)

-- | Use the exact source types accepted by the scalar and vector backends.
floatType, intType, boolType, vectorType :: EType
floatType = EVar (mkIdent "Primitives.Float")
intType = EVar (mkIdent "Data.Int.Int32.Int32")
boolType = EVar (mkIdent "Data.Bool_Type.Bool")
vectorType = EApp (EVar (mkIdent "SourceArray.Vector")) floatType

optimized :: TModule [EDef] -> String -> Either String Exp
optimized checked name =
    case [compileOpt term | (ident, term) <- tBindingsOf (desugar defaultFlags checked), showIdent (unQualIdent ident) == name] of
        [term] -> Right term
        [] -> Left ("Missing combinator binding: " ++ name)
        _ -> Left ("Ambiguous combinator binding: " ++ name)

evalNamed :: TModule [EDef] -> String -> Either String Val
evalNamed checked name = do
    term <- optimized checked name
    (value, _) <- runStateT (eval term) (Build 0 [])
    Right value

finishFloat :: Int -> Val -> Either String U
finishFloat arity value = do
    (result, _) <- runStateT (supply arity value) (Build 0 [])
    case result of
        Arrow arrow -> Right arrow
        _ -> Left "A numeric function did not reduce to a first-order arrow"

supply :: Int -> Val -> M Val
supply 1 value = apply value (Arrow (ULeaf LElem))
supply arity value = foldM (\current index -> apply current (Arrow (ULeaf (LArg index)))) value [0 .. arity - 1]

applyAll :: Val -> [Val] -> M Val
applyAll = foldM apply

eval :: Exp -> M Val
eval expression = case spine expression of
    (head_, arguments) -> do
        value <- evalHead head_
        foldM (\current argument -> eval argument >>= apply current) value arguments

evalHead :: Exp -> M Val
evalHead (Lit (LPrim name)) = combinator name
evalHead (Lit (LFloat value)) = pure (Arrow (UConstF value))
evalHead (Lit (LInt value)) = pure (Bits value)
evalHead (Var ident)
    | isDict text = pure (Dict text)
    | Just value <- intrinsic text = pure value
    | Just value <- method text = pure value
    | otherwise = oops ("No CCC reading for " ++ text)
  where
    text = showIdent ident
evalHead (Lam _ _) = oops "A lambda survived MicroHs combinator compilation"
evalHead other = oops ("Unsupported combinator literal: " ++ paren other)

apply :: Val -> Val -> M Val
apply (Fun function) value = function value
apply _ _ = oops "Tried to apply a first-order arrow"

combinator :: String -> M Val
combinator "I" = pure (Fun pure)
combinator "K" = pure (Fun (\kept -> pure (Fun (\_ -> pure kept))))
combinator "A" = pure (Fun (\_ -> pure (Fun pure)))
combinator "B" = pure (Fun (\f -> pure (Fun (\g -> pure (Fun (\x -> apply g x >>= apply f))))))
combinator "C" = pure (Fun (\f -> pure (Fun (\x -> pure (Fun (\y -> apply f y >>= (`apply` x)))))))
combinator "S" = pure (Fun (\f -> pure (Fun (\g -> pure (Fun (\x -> do fx <- apply f x; gx <- apply g x; apply fx gx))))))
combinator "S'" = pure (Fun (\f -> pure (Fun (\g -> pure (Fun (\h -> pure (Fun (\x -> do gx <- apply g x; hx <- apply h x; apply f gx >>= (`apply` hx)))))))))
combinator "B'" = pure (Fun (\f -> pure (Fun (\g -> pure (Fun (\h -> pure (Fun (\x -> do hx <- apply h x; fg <- apply f g; apply fg hx))))))))
combinator "C'" = pure (Fun (\f -> pure (Fun (\g -> pure (Fun (\h -> pure (Fun (\x -> do gx <- apply g x; apply f gx >>= (`apply` h)))))))))
combinator "C'B" = pure (Fun (\x -> pure (Fun (\y -> pure (Fun (\z -> pure (Fun (\w -> do yw <- apply y w; apply x z >>= (`apply` yw)))))))))
combinator "Z" = pure (Fun (\x -> pure (Fun (\y -> pure (Fun (\_ -> apply x y))))))
combinator "U" = pure (Fun (\x -> pure (Fun (\y -> apply y x))))
combinator "P" = pure (Fun (\x -> pure (Fun (\y -> pure (Fun (\z -> apply z x >>= (`apply` y)))))))
combinator "R" = pure (Fun (\x -> pure (Fun (\y -> pure (Fun (\z -> apply y z >>= (`apply` x)))))))
combinator "O" = pure (Fun (\x -> pure (Fun (\y -> pure (Fun (\z -> pure (Fun (\w -> apply w x >>= (`apply` y)))))))))
combinator "K2" = pure (Fun (\x -> pure (Fun (\_ -> pure (Fun (\_ -> pure x))))))
combinator "K3" = pure (Fun (\x -> pure (Fun (\_ -> pure (Fun (\_ -> pure (Fun (\_ -> pure x))))))))
combinator "K4" = pure (Fun (\x -> pure (Fun (\_ -> pure (Fun (\_ -> pure (Fun (\_ -> pure (Fun (\_ -> pure x))))))))))
combinator "Y" = oops "Y is general recursion. These GPU kernels have no recursive combinator."
combinator name = oops ("No CCC arrow for combinator " ++ name)

method :: String -> Maybe Val
method "Data.Num.+" = Just (withDict [(floatNum, binary PAdd), (intNum, binary PAddI)])
method "Data.Num.-" = Just (withDict [(floatNum, binary PSub)])
method "Data.Num.*" = Just (withDict [(floatNum, binary PMul)])
method "Data.Num.negate" = Just (withDict [(floatNum, unary PNeg), (intNum, unary PNegI)])
method "Data.Num.abs" = Just (withDict [(floatNum, unary PAbs)])
method "Data.Fractional./" = Just (withDict [(floatFrac, binary PDiv)])
method "Data.Floating.sqrt" = Just (withDict [(floatFloating, unary PSqrt)])
method "Data.Floating.exp" = Just (withDict [(floatFloating, unary PExp)])
method "Data.Floating.sin" = Just (withDict [(floatFloating, unary PSin)])
method "Data.Floating.cos" = Just (withDict [(floatFloating, unary PCos)])
method "Data.Floating.**" = Just (withDict [(floatFloating, binary PPow)])
method "Data.Ord.max" = Just (withDict [(floatOrd, binary PMax)])
method "Data.Ord.min" = Just (withDict [(floatOrd, binary PMin)])
method "Data.Ord.>" = Just (withDict [(floatOrd, binary PGt)])
method "Data.Num.fromInteger" = Just fromIntegerMethod
method "Data.Integer_Type._intToInteger" =
    Just
        ( Fun $ \value -> case value of
            Bits integer -> pure (Bits integer)
            _ -> oops "_intToInteger expects an integer literal"
        )
method _ = Nothing

fromIntegerMethod :: Val
fromIntegerMethod = Fun $ \dictionary -> pure $ Fun $ \integer -> case (dictionary, integer) of
    (Dict name, Bits value) | name == floatNum -> pure (Arrow (UConstF (fromIntegral value)))
    (Dict name, Bits value) | name == intNum -> pure (Arrow (UConstI (fromIntegral value)))
    _ -> oops "fromInteger is only translated for a literal Float or Int32"

intrinsic :: String -> Maybe Val
intrinsic "SourceArray.map" = Just mapIntrinsic
intrinsic "SourceArray.zipWith" = Just zipIntrinsic
intrinsic "SourceArray.sum" = Just (reduce Sum)
intrinsic "SourceArray.maximum" = Just (reduce Maximum)
intrinsic "SourceArray.count" = Just countIntrinsic
intrinsic _ = Nothing

withDict :: [(String, Val)] -> Val
withDict table = Fun $ \value -> case value of
    Dict name -> case lookup name table of
        Just primitive -> pure primitive
        Nothing -> oops ("Unsupported dictionary " ++ name)
    _ -> oops "A primitive method was not given its dictionary"

binary :: Prim -> Val
binary prim = Fun $ \left -> pure $ Fun $ \right -> do
    leftArrow <- asArrow left
    rightArrow <- asArrow right
    pure (Arrow (UCompose (UPrim prim) (UPair leftArrow rightArrow)))

unary :: Prim -> Val
unary prim = Fun $ \value -> do
    arrow <- asArrow value
    pure (Arrow (UCompose (UPrim prim) arrow))

mapIntrinsic :: Val
mapIntrinsic = Fun $ \function -> pure $ Fun $ \vector -> do
    source <- expectVec vector
    element <- fresh
    produced <- apply function (Arrow (ULeaf (LBound element)))
    body <- asArrow produced
    target <- emitMap source (bindLeaves [(element, LElem)] body)
    pure (Vec target)

zipIntrinsic :: Val
zipIntrinsic = Fun $ \function -> pure $ Fun $ \left -> pure $ Fun $ \right -> do
    leftId <- expectVec left
    rightId <- expectVec right
    leftElement <- fresh
    rightElement <- fresh
    applied <- apply function (Arrow (ULeaf (LBound leftElement)))
    produced <- apply applied (Arrow (ULeaf (LBound rightElement)))
    body <- asArrow produced
    target <- emitZip leftId rightId (bindLeaves [(leftElement, LLeft), (rightElement, LRight)] body)
    pure (Vec target)

-- | Replace only the current intrinsic's bound scalar inputs.
bindLeaves :: [(Int, Leaf)] -> U -> U
bindLeaves bindings (ULeaf (LBound identifier)) = ULeaf (maybe (LBound identifier) id (lookup identifier bindings))
bindLeaves bindings (UCompose left right) = UCompose (bindLeaves bindings left) (bindLeaves bindings right)
bindLeaves bindings (UPair left right) = UPair (bindLeaves bindings left) (bindLeaves bindings right)
bindLeaves _ arrow = arrow

reduce :: (Ref Float -> Ref [Float] -> Step) -> Val
reduce step = Fun $ \vector -> do
    source <- expectVec vector
    identifier <- fresh
    modify (\build -> build{steps = step (ScalarRef identifier) (VectorRef source) : steps build})
    pure (Frozen identifier)

countIntrinsic :: Val
countIntrinsic = reduce Length

emitMap :: Int -> U -> M Int
emitMap source body = do
    target <- fresh
    step <- case nub (collect body) of
        [] -> (\graph -> Map (VectorRef target) graph NoCaptures (VectorRef source)) <$> liftE (reify projElem0 body)
        [one] -> (\graph -> Map (VectorRef target) graph (Capture (ScalarRef one) NoCaptures) (VectorRef source)) <$> liftE (reify (projElem1 one) body)
        [first, second] ->
            (\graph -> Map (VectorRef target) graph (Capture (ScalarRef first) (Capture (ScalarRef second) NoCaptures)) (VectorRef source))
                <$> liftE (reify (projElem2 first second) body)
        _ -> oops "A map closes over more than two scalars"
    modify (\build -> build{steps = step : steps build})
    pure target

emitZip :: Int -> Int -> U -> M Int
emitZip left right body = do
    target <- fresh
    step <- case nub (collect body) of
        [] -> (\graph -> Map2 (VectorRef target) graph NoCaptures (VectorRef left) (VectorRef right)) <$> liftE (reify projZip0 body)
        [one] ->
            (\graph -> Map2 (VectorRef target) graph (Capture (ScalarRef one) NoCaptures) (VectorRef left) (VectorRef right))
                <$> liftE (reify (projZip1 one) body)
        [first, second] ->
            (\graph -> Map2 (VectorRef target) graph (Capture (ScalarRef first) (Capture (ScalarRef second) NoCaptures)) (VectorRef left) (VectorRef right))
                <$> liftE (reify (projZip2 first second) body)
        _ -> oops "A zipWith closes over more than two scalars"
    modify (\build -> build{steps = step : steps build})
    pure target

collect :: U -> [Int]
collect (ULeaf (LCap identifier)) = [identifier]
collect (UCompose left right) = collect left ++ collect right
collect (UPair left right) = collect left ++ collect right
collect _ = []

fresh :: M Int
fresh = do
    build <- get
    modify (\current -> current{nextId = nextId current + 1})
    pure (nextId build)

expectVec :: Val -> M Int
expectVec (Vec identifier) = pure identifier
expectVec _ = oops "An array intrinsic expected a vector"

asArrow :: Val -> M U
asArrow (Arrow arrow) = pure arrow
asArrow (Frozen identifier) = pure (ULeaf (LCap identifier))
asArrow _ = oops "A primitive expected a scalar arrow"

reify :: (Leaf -> Either String (Cat input Float)) -> U -> Either String (Cat input Float)
reify project = go
  where
    go (ULeaf leaf) = project leaf
    go (UConstF value) = Right (FloatConstant value)
    go (UCompose (UPrim prim) argument) = applyPrim go prim argument
    go (UCompose _ _) = Left "A non-primitive composition survived; the kernel would need a runtime function"
    go (UPrim _) = Left "A primitive was used before it was given its arguments"
    go (UConstI _) = Left "An Int32 constant appeared in a Float arrow"

applyPrim :: (U -> Either String (Cat input Float)) -> Prim -> U -> Either String (Cat input Float)
applyPrim go PAdd argument = binaryOp go AddFloat argument
applyPrim go PSub argument = binaryOp go SubFloat argument
applyPrim go PMul argument = binaryOp go MulFloat argument
applyPrim go PDiv argument = binaryOp go DivFloat argument
applyPrim go PMin argument = binaryOp go MinFloat argument
applyPrim go PMax argument = binaryOp go MaxFloat argument
applyPrim go PPow argument = binaryOp go PowFloat argument
applyPrim go PNeg argument = Compose NegFloat <$> go argument
applyPrim go PAbs argument = Compose AbsFloat <$> go argument
applyPrim go PSqrt argument = Compose SqrtFloat <$> go argument
applyPrim go PExp argument = Compose ExpFloat <$> go argument
applyPrim go PSin argument = Compose SinFloat <$> go argument
applyPrim go PCos argument = Compose CosFloat <$> go argument
applyPrim _ prim _ = Left ("Float arrow cannot use " ++ show prim)

binaryOp :: (U -> Either String (Cat input Float)) -> Cat (Float, Float) Float -> U -> Either String (Cat input Float)
binaryOp go operator argument = do
    (left, right) <- asPair argument
    Compose operator <$> (Pair <$> go left <*> go right)

reifyI :: (Leaf -> Either String (Cat input Int32)) -> U -> Either String (Cat input Int32)
reifyI project = go
  where
    go (ULeaf leaf) = project leaf
    go (UConstI value) = Right (IntConstant value)
    go (UCompose (UPrim PAddI) argument) = do
        (left, right) <- asPair argument
        Compose AddInt <$> (Pair <$> go left <*> go right)
    go (UCompose (UPrim PNegI) argument) = Compose NegInt <$> go argument
    go other = Left ("Int32 arrow cannot use " ++ show other)

proj1 :: Leaf -> Either String (Cat Float Float)
proj1 LElem = Right Id
proj1 leaf = Left ("Unary float arrow has no " ++ show leaf)

proj2 :: Leaf -> Either String (Cat (Float, Float) Float)
proj2 (LArg 0) = Right First
proj2 (LArg 1) = Right Second
proj2 leaf = Left ("Binary float arrow has no " ++ show leaf)

proj3 :: Leaf -> Either String (Cat (Float, (Float, Float)) Float)
proj3 (LArg 0) = Right First
proj3 (LArg 1) = Right (Compose First Second)
proj3 (LArg 2) = Right (Compose Second Second)
proj3 leaf = Left ("Ternary float arrow has no " ++ show leaf)

proj4 :: Leaf -> Either String (Cat (Float, (Float, (Float, Float))) Float)
proj4 (LArg 0) = Right First
proj4 (LArg 1) = Right (Compose First Second)
proj4 (LArg 2) = Right (Compose First (Compose Second Second))
proj4 (LArg 3) = Right (Compose Second (Compose Second Second))
proj4 leaf = Left ("Four-argument float arrow has no " ++ show leaf)

projInt :: Leaf -> Either String (Cat (Int32, Int32) Int32)
projInt (LArg 0) = Right First
projInt (LArg 1) = Right Second
projInt leaf = Left ("Binary Int32 arrow has no " ++ show leaf)

projElem0 :: Leaf -> Either String (Cat (Float, ()) Float)
projElem0 LElem = Right First
projElem0 leaf = Left ("Map element has no " ++ show leaf)

projElem1 :: Int -> Leaf -> Either String (Cat (Float, (Float, ())) Float)
projElem1 _ LElem = Right First
projElem1 capture (LCap identifier)
    | identifier == capture = Right (Compose First Second)
projElem1 _ leaf = Left ("Map element has no " ++ show leaf)

projElem2 :: Int -> Int -> Leaf -> Either String (Cat (Float, (Float, (Float, ()))) Float)
projElem2 _ _ LElem = Right First
projElem2 first _ (LCap identifier)
    | identifier == first = Right (Compose First Second)
projElem2 _ second (LCap identifier)
    | identifier == second = Right (Compose (Compose First Second) Second)
projElem2 _ _ leaf = Left ("Map element has no " ++ show leaf)

projZip0 :: Leaf -> Either String (Cat (Float, (Float, ())) Float)
projZip0 LLeft = Right First
projZip0 LRight = Right (Compose First Second)
projZip0 leaf = Left ("Zip element has no " ++ show leaf)

projZip1 :: Int -> Leaf -> Either String (Cat (Float, (Float, (Float, ()))) Float)
projZip1 _ LLeft = Right First
projZip1 _ LRight = Right (Compose First Second)
projZip1 capture (LCap identifier)
    | identifier == capture = Right (Compose First (Compose Second Second))
projZip1 _ leaf = Left ("Zip element has no " ++ show leaf)

projZip2 :: Int -> Int -> Leaf -> Either String (Cat (Float, (Float, (Float, (Float, ())))) Float)
projZip2 _ _ LLeft = Right First
projZip2 _ _ LRight = Right (Compose First Second)
projZip2 first _ (LCap identifier)
    | identifier == first = Right (Compose First (Compose Second Second))
projZip2 _ second (LCap identifier)
    | identifier == second = Right (Compose (Compose First Second) (Compose Second Second))
projZip2 _ _ leaf = Left ("Zip element has no " ++ show leaf)

asPair :: U -> Either String (U, U)
asPair (UPair left right) = Right (left, right)
asPair _ = Left "A binary primitive was not given a pair"

spine :: Exp -> (Exp, [Exp])
spine (App function argument) = let (head_, arguments) = spine function in (head_, arguments ++ [argument])
spine expression = (expression, [])

paren :: Exp -> String
paren (Var ident) = showIdent ident
paren (Lit (LPrim name)) = name
paren (Lit (LFloat value)) = show value
paren (Lit (LInt value)) = show value
paren (App function argument) = "(" ++ paren function ++ " " ++ paren argument ++ ")"
paren other = show other

isDict :: String -> Bool
isDict name = name `elem` [floatNum, floatOrd, floatFrac, floatFloating, intNum]

floatNum, floatOrd, floatFrac, floatFloating, intNum :: String
floatNum = "inst$Data.Num.Num@Primitives.Float"
floatOrd = "inst$Data.Ord.Ord@Primitives.Float"
floatFrac = "inst$Data.Fractional.Fractional@Primitives.Float"
floatFloating = "inst$Data.Floating.Floating@Primitives.Float"
intNum = "inst$Data.Num.Num@Data.Int.Int32.Int32"

oops :: String -> M a
oops = liftE . Left

liftE :: Either String a -> M a
liftE = either (lift . Left) pure
