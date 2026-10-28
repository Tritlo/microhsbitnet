{-# LANGUAGE GADTs #-}

module HeadArrays (headSource) where

import CCC
import Combinators (kernelBool, kernelFloat2)
import FloatWGSL (softFloat)
import MicroHs.Expr (EDef)
import MicroHs.TypeCheck (TModule)

-- | Generate BF16 lookup, ordered binary64 head sums, and stable greedy selection.
headSource :: TModule [EDef] -> Either String String
headSource checked = do
    multiply <- kernelFloat2 checked "product"
    add <- kernelFloat2 checked "plus"
    better <- kernelBool checked "better"
    productCode <- emitBinary64 multiply (PairCode (FloatCode "f32_to_f64(bitcast<u32>(weight))") (FloatCode "f32_to_f64(bitcast<u32>(hidden[column]))"))
    laneCode <- emitBinary64 add (PairCode (FloatCode "partial") (FloatCode "term"))
    laneChoice <- emitWGSLRelaxed better (PairCode (FloatCode "candidate") (FloatCode "best"))
    treeChoice <- emitWGSLRelaxed better (PairCode (FloatCode "scratch[lane + stride]") (FloatCode "scratch[lane]"))
    let FloatCode product = productCode
        FloatCode laneSum = laneCode
        BoolCode chooseLane = laneChoice
        BoolCode chooseTree = treeChoice
    pure $
        softFloat
            ++ unlines
                [ "@group(0) @binding(0) var<storage, read> weights: array<u32>;"
                , "@group(0) @binding(1) var<storage, read> hidden: array<f32>;"
                , "@group(0) @binding(2) var<storage, read_write> logits: array<f32>;"
                , "@group(0) @binding(3) var<uniform> dimensions: vec4<u32>;"
                , "@group(0) @binding(4) var<storage, read_write> chosen: array<u32>;"
                , "var<workgroup> scratch: array<f32, 128>;"
                , "var<workgroup> indices: array<u32, 128>;"
                , "fn bf16(index: u32) -> f32 {"
                , "  let bits = (weights[index / 2u] >> (16u * (index % 2u))) & 65535u;"
                , "  return bitcast<f32>(bits << 16u);"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn embedding(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  if (global.x < dimensions.x) { logits[global.x] = bf16(dimensions.z * dimensions.x + global.x); }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn head(@builtin(global_invocation_id) global: vec3<u32>) {"
                , "  let row = global.x;"
                , "  if (row >= dimensions.y) { return; }"
                , "  var partial = vec2<u32>(0u, 0u);"
                , "  for (var column = 0u; column < dimensions.x; column++) {"
                , "    let weight = bf16(row * dimensions.x + column);"
                , "    let term = " ++ product ++ ";"
                , "    partial = " ++ laneSum ++ ";"
                , "  }"
                , "  logits[dimensions.w + row] = bitcast<f32>(f64_to_f32(partial));"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn argmax(@builtin(local_invocation_index) lane: u32) {"
                , "  var index = lane;"
                , "  var best = logits[lane];"
                , "  for (var row = lane + 128u; row < dimensions.y; row += 128u) {"
                , "    let candidate = logits[row];"
                , "    if (" ++ chooseLane ++ ") { best = candidate; index = row; }"
                , "  }"
                , "  scratch[lane] = best;"
                , "  indices[lane] = index;"
                , "  workgroupBarrier();"
                , "  for (var stride = 64u; stride > 0u; stride >>= 1u) {"
                , "    if (lane < stride) {"
                , "      if (" ++ chooseTree ++ " || (scratch[lane + stride] == scratch[lane] && indices[lane + stride] < indices[lane])) {"
                , "        scratch[lane] = scratch[lane + stride];"
                , "        indices[lane] = indices[lane + stride];"
                , "      }"
                , "    }"
                , "    workgroupBarrier();"
                , "  }"
                , "  if (lane == 0u) { chosen[0] = indices[0]; chosen[1] = bitcast<u32>(scratch[0]); }"
                , "}"
                ]
