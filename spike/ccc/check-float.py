# /// script
# requires-python = ">=3.11"
# dependencies = ["wgpu==0.32.0"]
# ///
# pyright: strict
"""Check exact binary64 arithmetic and compare exp32 with host libm within 1 ULP."""

import ctypes
import ctypes.util
import pathlib
import struct
from typing import cast

import wgpu  # pyright: ignore[reportMissingTypeStubs]

ROOT = pathlib.Path(__file__).resolve().parent
SHADER = (ROOT / "float_soft.wgsl").read_text() + """
@group(0) @binding(0) var<storage, read> lefts: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> rights: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> sums: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> products: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> floats: array<u32>;
@group(0) @binding(5) var<storage, read_write> exponentials: array<u32>;

@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  sums[index] = add_f64(lefts[index], rights[index]);
  products[index] = mul_f64(lefts[index], rights[index]);
  exponentials[index] = bitcast<u32>(exp32(bitcast<f32>(floats[index])));
}
"""


def main() -> None:
    count = 32768
    state = 0x123456789ABCDEF0
    left = bytearray()
    right = bytearray()
    floats = bytearray()
    expected_sum = bytearray()
    expected_product = bytearray()
    expected_exp = bytearray()
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.expf.argtypes = [ctypes.c_float]
    libm.expf.restype = ctypes.c_float

    def push_u64(buf: bytearray, value: int) -> None:
        buf.extend(struct.pack("<Q", value & ((1 << 64) - 1)))

    cursor = 1
    binary_edges = [(0, 0), (0, 0x8000000000000000), (0x8000000000000000, 0), (0x8000000000000000, 0x8000000000000000)]
    for index in range(count):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        left_bits = state
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        right_bits = state
        # Keep a mix of ordinary finite magnitudes.
        left_bits = (left_bits & 0x800FFFFFFFFFFFFF) | (((512 + ((left_bits >> 52) & 0x3FF)) << 52))
        right_bits = (right_bits & 0x800FFFFFFFFFFFFF) | (((512 + ((right_bits >> 52) & 0x3FF)) << 52))
        if index < len(binary_edges):
            left_bits, right_bits = binary_edges[index]
        push_u64(left, left_bits)
        push_u64(right, right_bits)
        left_value = struct.unpack("<d", struct.pack("<Q", left_bits))[0]
        right_value = struct.unpack("<d", struct.pack("<Q", right_bits))[0]
        push_u64(expected_sum, struct.unpack("<Q", struct.pack("<d", left_value + right_value))[0])
        push_u64(expected_product, struct.unpack("<Q", struct.pack("<d", left_value * right_value))[0])
        cursor = (cursor * 1664525 + 1013904223) & 0xFFFFFFFF
        bits = 0x80000000 | (cursor & 0x41FFFFFF)
        edges = [0, 0x80000000, 0x3F800000, 0xBF800000, 0x42B17217, 0xC2CFF1B4, 0xC2CF3583, 0x00000001, 0x7F7FFFFF]
        if index < len(edges):
            bits = edges[index]
        floats.extend(struct.pack("<I", bits))
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        expected_exp.extend(struct.pack("<f", libm.expf(value)))

    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    print("adapter", adapter.summary)
    # wgpu has untyped descriptor dictionaries in these method signatures.
    device = adapter.request_device_sync()  # pyright: ignore[reportUnknownMemberType]
    module = device.create_shader_module(code=SHADER)  # pyright: ignore[reportUnknownMemberType]

    def storage(data: bytes, usage: int) -> wgpu.GPUBuffer:
        buffer = device.create_buffer_with_data(data=data, usage=usage)
        return buffer

    left_b = storage(bytes(left), wgpu.BufferUsage.STORAGE)
    right_b = storage(bytes(right), wgpu.BufferUsage.STORAGE)
    sum_b = device.create_buffer(size=len(left), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    prod_b = device.create_buffer(size=len(left), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    float_b = storage(bytes(floats), wgpu.BufferUsage.STORAGE)
    exp_b = device.create_buffer(size=len(floats), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    layout = device.create_bind_group_layout(entries=[  # pyright: ignore[reportUnknownMemberType]
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    bind = device.create_bind_group(layout=layout, entries=[  # pyright: ignore[reportUnknownMemberType]
        {"binding": 0, "resource": {"buffer": left_b, "offset": 0, "size": len(left)}},
        {"binding": 1, "resource": {"buffer": right_b, "offset": 0, "size": len(right)}},
        {"binding": 2, "resource": {"buffer": sum_b, "offset": 0, "size": len(left)}},
        {"binding": 3, "resource": {"buffer": prod_b, "offset": 0, "size": len(left)}},
        {"binding": 4, "resource": {"buffer": float_b, "offset": 0, "size": len(floats)}},
        {"binding": 5, "resource": {"buffer": exp_b, "offset": 0, "size": len(floats)}},
    ])
    pipeline = device.create_compute_pipeline(layout=device.create_pipeline_layout(bind_group_layouts=[layout]), compute={"module": module, "entry_point": "check"})  # pyright: ignore[reportUnknownMemberType]
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()  # pyright: ignore[reportUnknownMemberType]
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])

    def read(buffer: wgpu.GPUBuffer, size: int) -> bytes:
        tmp = device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffer, 0, tmp, 0, size)
        device.queue.submit([encoder.finish()])
        tmp.map_sync(wgpu.MapMode.READ)
        data = bytes(cast(memoryview, tmp.read_mapped()))
        tmp.unmap()
        return data

    got_sum = read(sum_b, len(left))
    got_prod = read(prod_b, len(left))
    got_exp = read(exp_b, len(floats))
    sum_bad = prod_bad = exp_bad = exp_different = max_exp_ulp = 0
    for index in range(count):
        if got_sum[index * 8:(index + 1) * 8] != bytes(expected_sum[index * 8:(index + 1) * 8]):
            if sum_bad < 3:
                print("sum", index, got_sum[index * 8:(index + 1) * 8].hex(), expected_sum[index * 8:(index + 1) * 8].hex())
            sum_bad += 1
        if got_prod[index * 8:(index + 1) * 8] != bytes(expected_product[index * 8:(index + 1) * 8]):
            if prod_bad < 3:
                print("mul", index, got_prod[index * 8:(index + 1) * 8].hex(), expected_product[index * 8:(index + 1) * 8].hex())
            prod_bad += 1
        actual_bits = struct.unpack_from("<I", got_exp, index * 4)[0]
        expected_bits = struct.unpack_from("<I", expected_exp, index * 4)[0]
        ulp = abs(actual_bits - expected_bits)
        max_exp_ulp = max(max_exp_ulp, ulp)
        if ulp:
            if exp_different < 3:
                print("exp", index, hex(actual_bits), hex(expected_bits), "ULP", ulp)
            exp_different += 1
        if ulp > 1:
            exp_bad += 1
    print(f"binary64 exact: sum_bad {sum_bad} mul_bad {prod_bad} of {count}")
    print(f"exp32 vs host libm: exact_differences {exp_different} max_ulp {max_exp_ulp} over_1ulp {exp_bad} of {count}")
    if sum_bad or prod_bad or exp_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
