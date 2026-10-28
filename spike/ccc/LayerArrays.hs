{-# LANGUAGE GADTs #-}

module LayerArrays (layerSource) where

import qualified ArrayGraph
import CCC
import Combinators (kernelFloat2, kernelFloat4)
import Control.Monad (unless)
import MicroHs.Expr (EDef)
import MicroHs.TypeCheck (TModule)

-- | Extract a scalar expression from the typed WGSL interpretation.
floatExpression :: Cat input Float -> Code input -> Either String String
floatExpression graph input = do
    code <- emitWGSL graph input
    let FloatCode result = code
    pure result

-- | Generate layer array operations from the checked numerical functions.
layerSource :: TModule [EDef] -> ArrayGraph.Program -> ArrayGraph.Program -> ArrayGraph.Program -> ArrayGraph.Program -> Either String String
layerSource checked softmax rms residual gated = do
    plus <- kernelFloat2 checked "plus"
    product <- kernelFloat2 checked "product"
    rotateFirst <- kernelFloat4 checked "rotateFirst"
    rotateSecond <- kernelFloat4 checked "rotateSecond"
    softmaxPlan <- ArrayGraph.emitWorkgroup softmax 2048 [(0, "scores")] "dimensions.x" "attn"
    rmsPlan <- ArrayGraph.emitWorkgroup rms 6912 [(0, "inputs"), (1, "other")] "dimensions.x" "rms"
    addPlan <- ArrayGraph.emitWorkgroup residual 6912 [(0, "inputs"), (1, "other")] "dimensions.x" "add"
    gatedPlan <- ArrayGraph.emitWorkgroup gated 6912 [(0, "inputs"), (1, "other")] "dimensions.x" "gate"
    unless (ArrayGraph.scratchVectors rmsPlan == 0 && ArrayGraph.scratchVectors addPlan == 0 && ArrayGraph.scratchVectors gatedPlan == 0) $
        Left "RMSNorm, residual, and gated schedules must not allocate a full-length workgroup vector"
    let pair left right = PairCode (FloatCode left) (FloatCode right)
        triple first second third = PairCode (FloatCode first) (pair second third)
        quadruple first second third fourth = PairCode (FloatCode first) (triple second third fourth)
    rmsOutput <- ArrayGraph.outputAt rmsPlan "index"
    addOutput <- ArrayGraph.outputAt addPlan "index"
    gatedOutput <- ArrayGraph.outputAt gatedPlan "index"
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
                , "var<workgroup> scores: array<f32, 2048>;"
                ]
            ++ unlines (ArrayGraph.declarations softmaxPlan ++ ArrayGraph.declarations rmsPlan ++ ArrayGraph.declarations addPlan ++ ArrayGraph.declarations gatedPlan)
            ++ unlines
                [ "@compute @workgroup_size(128)"
                , "fn rmsnorm(@builtin(local_invocation_index) lane: u32) {"
                , "  if (lane == 0u) {"
                ]
            ++ unlines (map ("    " ++) (ArrayGraph.statements rmsPlan))
            ++ unlines
                [ "  }"
                , "  workgroupBarrier();"
                , "  for (var index = lane; index < dimensions.x; index += 128u) { output[index] = " ++ rmsOutput ++ "; }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn add(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let index = global.x;"
                , "  if (index < dimensions.x) { output[index] = " ++ addOutput ++ "; }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn gated(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let index = global.x;"
                , "  if (index < dimensions.x) { output[index] = " ++ gatedOutput ++ "; }"
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
