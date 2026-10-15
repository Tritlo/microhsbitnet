{-# LANGUAGE GADTs #-}

module HeadArrays (headSource) where

import CCC
import Compiler
import MicroHs.Expr (EDef)
import MicroHs.TypeCheck (TModule)

-- | Generate BF16 lookup, FP32 head reduction, and stable greedy selection.
headSource :: TModule [EDef] -> Either String String
headSource checked = do
    multiply <- compileFloatPair checked "product"
    add <- compileFloatPair checked "plus"
    better <- compileFloatPredicate checked "better"
    productCode <- emitWGSLRelaxed multiply (PairCode (FloatCode "weight") (FloatCode "hidden[column]"))
    laneCode <- emitWGSLRelaxed add (PairCode (FloatCode "partial") (FloatCode "element"))
    treeCode <- emitWGSLRelaxed add (PairCode (FloatCode "scratch[lane]") (FloatCode "scratch[lane + stride]"))
    laneChoice <- emitWGSLRelaxed better (PairCode (FloatCode "candidate") (FloatCode "best"))
    treeChoice <- emitWGSLRelaxed better (PairCode (FloatCode "scratch[lane + stride]") (FloatCode "scratch[lane]"))
    let FloatCode product = productCode
        FloatCode laneSum = laneCode
        FloatCode treeSum = treeCode
        BoolCode chooseLane = laneChoice
        BoolCode chooseTree = treeChoice
    pure $
        unlines
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
            , "fn head(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {"
            , "  let row = group.x;"
            , "  if (row >= dimensions.y) { return; }"
            , "  var partial = 0.0f;"
            , "  for (var column = lane; column < dimensions.x; column += 128u) {"
            , "    let weight = bf16(row * dimensions.x + column);"
            , "    let element = " ++ product ++ ";"
            , "    partial = " ++ laneSum ++ ";"
            , "  }"
            , "  scratch[lane] = partial;"
            , "  workgroupBarrier();"
            , "  for (var stride = 64u; stride > 0u; stride >>= 1u) {"
            , "    if (lane < stride) { scratch[lane] = " ++ treeSum ++ "; }"
            , "    workgroupBarrier();"
            , "  }"
            , "  if (lane == 0u) { logits[dimensions.w + row] = scratch[0]; }"
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
