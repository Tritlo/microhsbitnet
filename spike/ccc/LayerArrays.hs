{-# LANGUAGE GADTs #-}

module LayerArrays (layerSource) where

import qualified ArrayGraph
import CCC
import Compiler
import MicroHs.Expr (EDef)
import MicroHs.TypeCheck (TModule)

-- | Extract a scalar expression from the typed WGSL interpretation.
floatExpression :: Cat input Float -> Code input -> Either String String
floatExpression graph input = do
    code <- emitWGSL graph input
    let FloatCode result = code
    pure result

-- | Generate layer array operations from the checked numerical functions.
layerSource :: TModule [EDef] -> ArrayGraph.Program -> Either String String
layerSource checked softmax = do
    square <- compileScalar checked "square"
    plus <- compileFloatPair checked "plus"
    product <- compileFloatPair checked "product"
    normalizer <- compileFloatPair checked "normalizer"
    weighted <- compileFloatTriple checked "weighted"
    reluMul <- compileFloatPair checked "reluMul"
    rotateFirst <- compileFloatQuadruple checked "rotateFirst"
    rotateSecond <- compileFloatQuadruple checked "rotateSecond"
    softmaxPlan <- ArrayGraph.emitWorkgroup softmax 2048 "scores" "dimensions.x"
    let pair left right = PairCode (FloatCode left) (FloatCode right)
        triple first second third = PairCode (FloatCode first) (pair second third)
        quadruple first second third fourth = PairCode (FloatCode first) (triple second third fourth)
    squared <- floatExpression square (FloatCode "inputs[index]")
    sumSquares <- floatExpression plus (pair "total" "squared")
    inverse <- floatExpression normalizer (pair "total" "f32(dimensions.x)")
    normalized <- floatExpression weighted (triple "other[index]" "inputs[index]" "inverseRms")
    residual <- floatExpression plus (pair "inputs[index]" "other[index]")
    gated <- floatExpression reluMul (pair "inputs[index]" "other[index]")
    rotaryFirst <- floatExpression rotateFirst (quadruple "first" "second" "other[component * 2u]" "other[component * 2u + 1u]")
    rotarySecond <- floatExpression rotateSecond (quadruple "first" "second" "other[component * 2u]" "other[component * 2u + 1u]")
    dotElement <- floatExpression product (pair "inputs[head * dimensions.w + component]" "keys[position * kvDimension + kvHead * dimensions.w + component]")
    dotSum <- floatExpression plus (pair "dot" "element")
    scaledDot <- floatExpression product (pair "dot" "parameters.x")
    probability <- ArrayGraph.outputAt softmaxPlan "position"
    weightedValue <- floatExpression product (pair "probability" "values[position * kvDimension + kvHead * dimensions.w + component]")
    attended <- floatExpression plus (pair "total" "element")
    pure $
        wgslPrelude
            ++ unlines
                [ "@group(0) @binding(0) var<storage, read> inputs: array<f32>;"
                , "@group(0) @binding(1) var<storage, read> other: array<f32>;"
                , "@group(0) @binding(2) var<storage, read_write> output: array<f32>;"
                , "@group(0) @binding(3) var<uniform> dimensions: vec4<u32>;"
                , "@group(0) @binding(4) var<uniform> parameters: vec4<f32>;"
                , "@group(0) @binding(5) var<storage, read> keys: array<f32>;"
                , "@group(0) @binding(6) var<storage, read> values: array<f32>;"
                , "var<workgroup> inverseRms: f32;"
                , "var<workgroup> scores: array<f32, 2048>;"
                ]
            ++ unlines (ArrayGraph.declarations softmaxPlan)
            ++ unlines
                [ "@compute @workgroup_size(128)"
                , "fn rmsnorm(@builtin(local_invocation_index) lane: u32) {"
                , "  if (lane == 0u) {"
                , "    var total = 0.0f;"
                , "    for (var index = 0u; index < dimensions.x; index++) {"
                , "      let squared = " ++ squared ++ ";"
                , "      total = " ++ sumSquares ++ ";"
                , "    }"
                , "    inverseRms = " ++ inverse ++ ";"
                , "  }"
                , "  workgroupBarrier();"
                , "  for (var index = lane; index < dimensions.x; index += 128u) { output[index] = " ++ normalized ++ "; }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn add(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let index = global.x;"
                , "  if (index < dimensions.x) { output[index] = " ++ residual ++ "; }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn gated(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let index = global.x;"
                , "  if (index < dimensions.x) { output[index] = " ++ gated ++ "; }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn rope(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let index = global.x;"
                , "  if (index >= dimensions.x / 2u) { return; }"
                , "  let halfDimension = dimensions.y / 2u;"
                , "  let component = index % halfDimension;"
                , "  let base = (index / halfDimension) * dimensions.y + component;"
                , "  let first = inputs[base];"
                , "  let second = inputs[base + halfDimension];"
                , "  output[base] = " ++ rotaryFirst ++ ";"
                , "  output[base + halfDimension] = " ++ rotarySecond ++ ";"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn attention(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {"
                , "  let head = group.x;"
                , "  let kvHead = head / (dimensions.y / dimensions.z);"
                , "  let kvDimension = dimensions.z * dimensions.w;"
                , "  if (lane == 0u) {"
                , "    for (var position = 0u; position < dimensions.x; position++) {"
                , "      var dot = 0.0f;"
                , "      for (var component = 0u; component < dimensions.w; component++) {"
                , "        let element = " ++ dotElement ++ ";"
                , "        dot = " ++ dotSum ++ ";"
                , "      }"
                , "      scores[position] = " ++ scaledDot ++ ";"
                , "    }"
                ]
            ++ unlines (map ("    " ++) (ArrayGraph.statements softmaxPlan))
            ++ unlines
                [ "  }"
                , "  workgroupBarrier();"
                , "  for (var component = lane; component < dimensions.w; component += 128u) {"
                , "    var total = 0.0f;"
                , "    for (var position = 0u; position < dimensions.x; position++) {"
                , "      let probability = " ++ probability ++ ";"
                , "      let element = " ++ weightedValue ++ ";"
                , "      total = " ++ attended ++ ";"
                , "    }"
                , "    output[head * dimensions.w + component] = total;"
                , "  }"
                , "}"
                ]
