@group(0) @binding(0) var<storage, read> inputs: array<u32>;
@group(0) @binding(1) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> dimensions: vec4<u32>;
@group(0) @binding(3) var<storage, read> chosen: array<u32>;
@compute @workgroup_size(128)
fn packed(@builtin(global_invocation_id) global: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  for (var index = global.x; index < dimensions.x; index += groups.x * 128u) {
    let bits = inputs[index];
    if ((bits & (bits >> 1u) & 0x55555555u) != 0u) { atomicOr(&status[0], 1u); }
  }
}
@compute @workgroup_size(128)
fn bf16(@builtin(global_invocation_id) global: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  for (var index = global.x; index < dimensions.x; index += groups.x * 128u) {
    let bits = inputs[index];
    if ((bits & 0x7f80u) == 0x7f80u || ((bits >> 16u) & 0x7f80u) == 0x7f80u) { atomicOr(&status[0], 1u); }
  }
}
@compute @workgroup_size(128)
fn finiteFloat(@builtin(global_invocation_id) global: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  for (var index = global.x; index < dimensions.x; index += groups.x * 128u) {
    if ((inputs[index] & 0x7f800000u) == 0x7f800000u) { atomicOr(&status[0], 1u); }
  }
}
@compute @workgroup_size(1)
fn selection() {
  let index = chosen[0];
  if (index >= dimensions.x) { atomicOr(&status[0], 1u); return; }
  if (inputs[index] != chosen[1]) { atomicOr(&status[0], 1u); }
}
