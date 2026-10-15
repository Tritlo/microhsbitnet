export interface BitLinearResult {
  output: Float32Array<ArrayBuffer>;
  reference: Float32Array<ArrayBuffer>;
  count: number;
  uploadMs: number;
  medianDispatchReadbackMs: number;
  maxAbsoluteError: number;
}

export interface BitLinearKernels {
  layout: GPUBindGroupLayout;
  quantize: GPUComputePipeline;
  bitlinear: GPUComputePipeline;
}

export async function createBitLinearKernels(
  device: GPUDevice,
  bytes: (path: string) => Promise<ArrayBuffer>,
): Promise<BitLinearKernels> {
  const code = new TextDecoder().decode(await bytes("../bitlinear.wgsl"));
  const shader = createShader(device, code);
  const errors = (await shader.getCompilationInfo()).messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
  const layout = device.createBindGroupLayout({ entries: Array.from({ length: 7 }, (_, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding === 4 || binding === 5 ? "uniform" as const : binding < 2 ? "read-only-storage" as const : "storage" as const },
  })) });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [quantize, bitlinear] = await Promise.all([
    device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shader, entryPoint: "quantize" } }),
    device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shader, entryPoint: "bitlinear" } }),
  ]);
  return { layout, quantize, bitlinear };
}

export async function createBitLinearRunner(
  device: GPUDevice,
  bytes: (path: string) => Promise<ArrayBuffer>,
): Promise<(name: string) => Promise<BitLinearResult>> {
  const { layout, quantize, bitlinear } = await createBitLinearKernels(device, bytes);
  const manifest = new TextDecoder().decode(await bytes("../cases/bitlinear-cases.txt"));
  const cases = new Map(manifest.trim().split("\n").map((line) => {
    const [name, columns, rows, scale] = line.trim().split(/\s+/);
    if (!name || !columns || !rows || !scale) throw new Error("Invalid BitLinear manifest");
    return [name, { columns: Number(columns), rows: Number(rows), scale: Number(scale) }];
  }));
  return async (name) => {
    const dimensions = cases.get(name);
    if (!dimensions) throw new Error(`Unknown BitLinear case ${name}`);
    const { columns, rows, scale } = dimensions;
    if (!Number.isSafeInteger(columns) || !Number.isSafeInteger(rows) || columns <= 0 || rows <= 0 || rows % 4 || columns > Math.floor(2147483647 / 128) || columns * rows > 128 * 1024 * 1024 || !Number.isFinite(scale)) {
      throw new Error(`Invalid BitLinear shape or scale for ${name}`);
    }
    const [packed, input, expected, expectedQuantized] = await Promise.all([
      bytes(`../cases/${name}.weights.bin`), bytes(`../cases/${name}.float.bin`),
      bytes(`../cases/${name}.bitlinear.bin`), bytes(`../cases/${name}.quantized.bin`),
    ]);
    if (packed.byteLength !== columns * rows / 4 || input.byteLength !== columns * 4 || expected.byteLength !== rows * 4 || expectedQuantized.byteLength !== columns * 4) {
      throw new Error(`Invalid BitLinear fixture size ${name}`);
    }
    if (!new Float32Array(input).every(Number.isFinite) || !new Float32Array(expected).every(Number.isFinite)) throw new Error(`Nonfinite BitLinear fixture ${name}`);
    if (new Uint8Array(packed).some((value) => (value & (value >> 1) & 0x55) !== 0)) throw new Error(`Invalid packed field ${name}`);
    const buffers: GPUBuffer[] = [];
    const uploadStart = performance.now();
    function buffer(size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
      const resource = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
      buffers.push(resource);
      if (data) {
        const padded = new Uint8Array(resource.size);
        padded.set(new Uint8Array(data));
        device.queue.writeBuffer(resource, 0, padded);
      }
      return resource;
    }
    const readable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const writable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const resources = [
      buffer(packed.byteLength, readable, packed),
      buffer(input.byteLength, readable, input),
      buffer(columns * 4, writable), buffer(rows * 4, writable),
      buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([columns, rows, 0, 0]).buffer),
      buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([scale, 0, 0, 0]).buffer),
      buffer(4, writable),
    ];
    const group = device.createBindGroup({ layout, entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
    const readback = buffer((rows + columns) * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    await device.queue.onSubmittedWorkDone();
    const uploadMs = performance.now() - uploadStart;
    const durations: number[] = [];
    const reference = new Float32Array(expected);
    const quantizedReference = new Int32Array(expectedQuantized);
    let output = new Float32Array(0);
    let maxAbsoluteError = 0;
    try {
      for (let iteration = 0; iteration < 11; iteration++) {
        const started = performance.now();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setBindGroup(0, group);
        pass.setPipeline(quantize!);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(bitlinear!);
        pass.dispatchWorkgroups(rows);
        pass.end();
        encoder.copyBufferToBuffer(resources[3]!, 0, readback, 0, rows * 4);
        encoder.copyBufferToBuffer(resources[2]!, 0, readback, rows * 4, columns * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const data = readback.getMappedRange().slice(0);
        readback.unmap();
        if (iteration >= 2) durations.push(performance.now() - started);
        output = new Float32Array(data.slice(0, rows * 4));
        const quantized = new Int32Array(data, rows * 4, columns);
        for (let index = 0; index < columns; index++) {
          if (quantized[index] !== quantizedReference[index]) throw new Error(`${name} quantization[${index}] GPU ${quantized[index]} != C ${quantizedReference[index]}`);
        }
        for (let index = 0; index < rows; index++) {
          const error = Math.abs(output[index]! - reference[index]!);
          maxAbsoluteError = Math.max(maxAbsoluteError, error);
          if (!Number.isFinite(error) || error > 2e-5 * Math.max(1, Math.abs(reference[index]!))) throw new Error(`${name} output[${index}] GPU ${output[index]} != C ${reference[index]}`);
        }
      }
      durations.sort((left, right) => left - right);
      return { output, reference, count: rows, uploadMs, medianDispatchReadbackMs: durations[4]!, maxAbsoluteError };
    } finally {
      buffers.forEach((resource) => resource.destroy());
    }
  };
}
import { createShader } from "./shader.js";
