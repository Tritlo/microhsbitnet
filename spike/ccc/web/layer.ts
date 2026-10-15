import { createBitLinearKernels } from "./bitlinear.js";
import { createShader } from "./shader.js";

export interface LayerResult {
  output: Float32Array<ArrayBuffer>;
  reference: Float32Array<ArrayBuffer>;
  count: number;
  uploadMs: number;
  dispatchReadbackMs: number;
  maxAbsoluteError: number;
}

export interface WeightDescription {
  name: string;
  kind: "linear" | "norm";
  file: string;
  columns: number;
  rows: number;
  scale: number;
}

interface LayerManifest {
  weights: WeightDescription[];
  steps: { position: number; input: string; expected: string }[];
}

interface VectorTensor {
  kind: "vector" | "norm";
  buffer: GPUBuffer;
  count: number;
}

interface LinearTensor {
  kind: "linear";
  buffer: GPUBuffer;
  columns: number;
  rows: number;
  scale: number;
}

interface CacheTensor {
  kind: "cache";
  buffer: GPUBuffer;
  layer: number;
  role: "k" | "v";
  position: number;
}

interface EmbeddingTensor {
  kind: "embedding";
  chunks: { buffer: GPUBuffer; firstRow: number; rows: number }[];
  rows: number;
}

interface ChosenTensor {
  kind: "chosen";
  buffer: GPUBuffer;
  hidden: VectorTensor;
  normed: VectorTensor;
  logits: GPUBuffer;
}

type Tensor = VectorTensor | LinearTensor | CacheTensor | EmbeddingTensor | ChosenTensor;

export interface LayerExecution {
  output: Float32Array<ArrayBuffer>;
  layers?: Float32Array<ArrayBuffer>[];
  normed?: Float32Array<ArrayBuffer>;
  token?: number;
  logit?: number;
  uploadMs: number;
  dispatchReadbackMs: number;
  softmaxRoundtrips: number;
}

export interface LayerEngine {
  uploadWeight(weight: WeightDescription, data: ArrayBuffer): void;
  uploadEmbedding(firstRow: number, data: ArrayBuffer): void;
  reset(): void;
  execute(position: number, encode: (input: number) => number, inputData?: ArrayBuffer): Promise<LayerExecution>;
  dispose(): void;
}

declare global {
  var microhsLayer: {
    rotary(position: number): void;
    weight(name: string): number;
    op(name: string, first: number, second: number, third: number, parameter: number, scalar: number): number;
  };
}

export function fixturePath(file: string, directory = "layer-cases"): string {
  if (typeof file !== "string" || !file.split("/").every((part) => /^[A-Za-z0-9_.-]+$/.test(part) && part !== "." && part !== "..")) {
    throw new Error("Invalid layer fixture path");
  }
  return `../${directory}/${file}`;
}

export function finiteFloats(data: ArrayBuffer, count: number, name: string): Float32Array<ArrayBuffer> {
  if (data.byteLength !== count * 4) throw new Error(`Invalid layer fixture size: ${name}`);
  const values = new Float32Array(data);
  if (!values.every(Number.isFinite)) throw new Error(`Nonfinite layer fixture: ${name}`);
  return values;
}

export async function createLayerEngine(
  device: GPUDevice,
  bytes: (path: string) => Promise<ArrayBuffer>,
  layerCount: number,
): Promise<LayerEngine> {
  if (!Number.isSafeInteger(layerCount) || layerCount < 1 || layerCount > 30) throw new Error("Invalid layer count");
  const shapes: Record<string, readonly ["linear" | "norm", number, number]> = {
    "self_attn.q_proj.weight": ["linear", 2560, 2560],
    "self_attn.k_proj.weight": ["linear", 2560, 640],
    "self_attn.v_proj.weight": ["linear", 2560, 640],
    "self_attn.o_proj.weight": ["linear", 2560, 2560],
    "mlp.gate_proj.weight": ["linear", 2560, 6912],
    "mlp.up_proj.weight": ["linear", 2560, 6912],
    "mlp.down_proj.weight": ["linear", 6912, 2560],
    "input_layernorm.weight": ["norm", 2560, 1],
    "self_attn.attn_sub_norm.weight": ["norm", 2560, 1],
    "post_attention_layernorm.weight": ["norm", 2560, 1],
    "mlp.ffn_sub_norm.weight": ["norm", 6912, 1],
  };
  const [bitLinear, code, headCode, rotaryData] = await Promise.all([
    createBitLinearKernels(device, bytes),
    bytes("../layer.wgsl"),
    bytes("../head.wgsl"),
    bytes("../rotary.bin"),
  ]);
  finiteFloats(rotaryData, 2048 * 128, "rotary coefficient table");
  const shader = createShader(device, new TextDecoder().decode(code));
  const errors = (await shader.getCompilationInfo()).messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
  const layout = device.createBindGroupLayout({ entries: Array.from({ length: 7 }, (_, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding === 3 || binding === 4 ? "uniform" as const : binding === 2 ? "storage" as const : "read-only-storage" as const },
  })) });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelines = new Map(await Promise.all(["rmsnorm", "add", "gated", "rope", "attention"].map(async (entryPoint) =>
    [entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shader, entryPoint } })] as const,
  )));
  const headShader = createShader(device, new TextDecoder().decode(headCode));
  const headErrors = (await headShader.getCompilationInfo()).messages.filter((message) => message.type === "error");
  if (headErrors.length) throw new Error(headErrors.map((message) => message.message).join("\n"));
  const headLayout = device.createBindGroupLayout({ entries: Array.from({ length: 5 }, (_, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding === 3 ? "uniform" as const : binding === 2 || binding === 4 ? "storage" as const : "read-only-storage" as const },
  })) });
  const headPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [headLayout] });
  const headPipelines = new Map(await Promise.all(["embedding", "head", "argmax"].map(async (entryPoint) =>
    [entryPoint, await device.createComputePipelineAsync({ layout: headPipelineLayout, compute: { module: headShader, entryPoint } })] as const,
  )));
  const resident: GPUBuffer[] = [];
  const tensors = new Map<number, Tensor>();
  const weights = new Map<string, number>();
  let nextHandle = 1;
  let nextPosition = 0;
  let busy = false;
  let closed = false;
  let encoder: GPUCommandEncoder | undefined;
  let activePosition = -1;
  let operations = 0;
  let transient: GPUBuffer[] = [];
  let transientHandles: number[] = [];
  let finalHidden: VectorTensor | undefined;
  let finalNormed: VectorTensor | undefined;
  let headEncoded = false;
  let embeddingEncoded = false;
  let rotaryConfigured = false;
  let residualAdds = 0;
  let layerOutputs: VectorTensor[] = [];
  let attentionCount = 0;
  const cacheUpdates = new Map<CacheTensor, number>();

  function allocate(size: number, usage: GPUBufferUsageFlags, owner: GPUBuffer[], data?: ArrayBuffer): GPUBuffer {
    const paddedSize = Math.ceil(size / 4) * 4;
    if (!Number.isSafeInteger(size) || size <= 0 || paddedSize > device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) !== 0 && paddedSize > device.limits.maxStorageBufferBindingSize) || (data && data.byteLength > size)) {
      throw new Error("Layer buffer exceeds its shape or device limit");
    }
    const buffer = device.createBuffer({ size: paddedSize, usage });
    owner.push(buffer);
    if (data) {
      const padded = new Uint8Array(paddedSize);
      padded.set(new Uint8Array(data));
      device.queue.writeBuffer(buffer, 0, padded);
    }
    return buffer;
  }

  function handle(tensor: Tensor): number {
    const identifier = nextHandle++;
    tensors.set(identifier, tensor);
    return identifier;
  }

  function vector(identifier: number): VectorTensor {
    const tensor = tensors.get(identifier);
    if (!tensor || tensor.kind !== "vector") throw new Error(`Invalid layer vector handle ${identifier}`);
    return tensor;
  }

  function resultVector(count: number): number {
    if (!Number.isSafeInteger(count) || count <= 0 || count > 6912) throw new Error("Invalid layer tensor count");
    const buffer = allocate(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, transient);
    const identifier = handle({ kind: "vector", buffer, count });
    transientHandles.push(identifier);
    return identifier;
  }

  function dispose(): void {
    if (busy) throw new Error("Cannot dispose a layer engine during execution");
    closed = true;
    resident.forEach((buffer) => buffer.destroy());
    resident.length = 0;
    tensors.clear();
    weights.clear();
  }

  function reset(): void {
    if (busy || closed) throw new Error("Cannot reset a busy or closed layer engine");
    for (const tensor of tensors.values()) if (tensor.kind === "cache") tensor.position = -1;
    nextPosition = 0;
  }

  function uploadWeight(weight: WeightDescription, data: ArrayBuffer): void {
    if (closed || busy || nextPosition !== 0 || !weight || typeof weight.name !== "string" || weights.has(weight.name)) throw new Error("Invalid or duplicate layer weight upload");
    const match = /^model\.layers\.(0|[1-9][0-9]?)\.(.+)$/.exec(weight.name);
    const shape = weight.name === "model.norm.weight" ? ["norm", 2560, 1] as const : match && Number(match[1]) < layerCount ? shapes[match[2]!] : undefined;
    if (!shape || weight.kind !== shape[0] || weight.columns !== shape[1] || weight.rows !== shape[2] || !Number.isFinite(weight.scale) || !Number.isFinite(Math.fround(weight.scale))) throw new Error(`Invalid layer weight shape or scale: ${weight.name}`);
    if (weight.kind === "linear") {
      if (data.byteLength !== weight.columns * weight.rows / 4 || new Uint8Array(data).some((value) => (value & (value >> 1) & 0x55) !== 0)) throw new Error(`Invalid packed layer weight: ${weight.name}`);
      const buffer = allocate(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, resident, data);
      weights.set(weight.name, handle({ kind: "linear", buffer, columns: weight.columns, rows: weight.rows, scale: weight.scale }));
    } else {
      finiteFloats(data, weight.columns, weight.name);
      const buffer = allocate(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, resident, data);
      weights.set(weight.name, handle({ kind: "norm", buffer, count: weight.columns }));
    }
  }

  function uploadEmbedding(firstRow: number, data: ArrayBuffer): void {
    if (closed || busy || nextPosition !== 0 || !Number.isSafeInteger(firstRow) || firstRow < 0 || firstRow >= 128256) throw new Error("Invalid embedding upload state or row");
    const existing = weights.get("model.embed_tokens.weight");
    const tensor = existing === undefined ? undefined : tensors.get(existing);
    if (tensor && tensor.kind !== "embedding") throw new Error("Invalid embedding weight handle");
    const embedding: EmbeddingTensor = tensor ?? { kind: "embedding", chunks: [], rows: 0 };
    const rows = Math.min(16384, 128256 - firstRow);
    if (firstRow !== embedding.rows || data.byteLength !== rows * 2560 * 2) throw new Error("Embedding chunks must be complete and consecutive");
    const view = new DataView(data);
    for (let offset = 0; offset < data.byteLength; offset += 2) {
      if ((view.getUint16(offset, true) & 0x7f80) === 0x7f80) throw new Error("Nonfinite BF16 embedding weight");
    }
    const buffer = allocate(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, resident, data);
    embedding.chunks.push({ buffer, firstRow, rows });
    embedding.rows += rows;
    if (existing === undefined) weights.set("model.embed_tokens.weight", handle(embedding));
  }

  const dummy = allocate(4, GPUBufferUsage.STORAGE, resident);
  const dummyChosen = allocate(8, GPUBufferUsage.STORAGE, resident);
  const rotaryTable = allocate(rotaryData.byteLength, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, resident, rotaryData);
  const rotaryFactors = allocate(512, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, resident);

  function dispatchHead(name: "embedding" | "head" | "argmax", resources: [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer], dimensions: number[], groups: number): void {
    if (!encoder) throw new Error("Head operations require an active encoder");
    const pipeline = headPipelines.get(name);
    if (!pipeline) throw new Error(`Missing head pipeline ${name}`);
    const uniform = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, transient, new Uint32Array(dimensions).buffer);
    const bound = [resources[0], resources[1], resources[2], uniform, resources[3]];
    const group = device.createBindGroup({ layout: headLayout, entries: bound.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  function dispatchLayer(name: string, resources: GPUBuffer[], groups: number): void {
    const pipeline = pipelines.get(name);
    if (!encoder || !pipeline) throw new Error(`Missing layer encoder or pipeline ${name}`);
    const group = device.createBindGroup({ layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  const host = {
    rotary(position: number): void {
      if (!encoder || closed || rotaryConfigured || operations !== 0 || !Number.isSafeInteger(position) || position < 0 || position >= 2048 || position !== activePosition || position !== nextPosition) throw new Error("Invalid, repeated, or nonconsecutive rotary position");
      encoder.copyBufferToBuffer(rotaryTable, position * 512, rotaryFactors, 0, 512);
      rotaryConfigured = true;
    },
    weight(name: string): number {
      if (!encoder || closed) throw new Error("Layer weights can only be used during layer encoding");
      const existing = weights.get(name);
      if (existing !== undefined) return existing;
      const cache = /^cache\.(0|[1-9][0-9]?)\.([kv])$/.exec(name);
      if (!cache || Number(cache[1]) >= layerCount) throw new Error(`Unknown layer weight ${name}`);
      const buffer = allocate(2048 * 640 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, resident);
      const identifier = handle({ kind: "cache", buffer, layer: Number(cache[1]), role: cache[2] as "k" | "v", position: -1 });
      weights.set(name, identifier);
      return identifier;
    },
    op(name: string, first: number, second: number, third: number, parameter: number, scalar: number): number {
      if (!encoder || closed || ++operations > 2048 || !Number.isSafeInteger(parameter) || !Number.isFinite(scalar)) throw new Error("Invalid layer operation state or parameters");
      if (name !== "attention" && (third !== 0 || scalar !== 0)) throw new Error(`Unexpected arguments for ${name}`);
      if (name === "embedding") {
        const embedding = tensors.get(first);
        if (!embedding || embedding.kind !== "embedding" || embedding.rows !== 128256 || second !== 0 || parameter < 0 || parameter >= 128256 || embeddingEncoded) throw new Error("Invalid embedding lookup");
        const chunk = embedding.chunks.find((candidate) => parameter >= candidate.firstRow && parameter < candidate.firstRow + candidate.rows);
        if (!chunk) throw new Error("Missing embedding chunk");
        const identifier = resultVector(2560);
        dispatchHead("embedding", [chunk.buffer, dummy, vector(identifier).buffer, dummyChosen], [2560, 0, parameter - chunk.firstRow, 0], 20);
        embeddingEncoded = true;
        return identifier;
      }
      const input = vector(first);
      if (name === "head") {
        const embedding = tensors.get(second);
        if (!embedding || embedding.kind !== "embedding" || embedding.rows !== 128256 || embedding.chunks.length !== 8 || parameter !== 0 || input.count !== 2560 || finalNormed !== input || !finalHidden || headEncoded) throw new Error("Invalid head input or embedding weights");
        const logits = allocate(128256 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, transient);
        const chosen = allocate(8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, transient);
        for (const chunk of embedding.chunks) dispatchHead("head", [chunk.buffer, input.buffer, logits, chosen], [2560, chunk.rows, 0, chunk.firstRow], chunk.rows);
        dispatchHead("argmax", [dummy, dummy, logits, chosen], [2560, 128256, 0, 0], 1);
        const identifier = handle({ kind: "chosen", buffer: chosen, hidden: finalHidden, normed: input, logits });
        transientHandles.push(identifier);
        headEncoded = true;
        return identifier;
      }
      if (name === "linear") {
        const weight = tensors.get(second);
        if (!weight || weight.kind !== "linear" || input.count !== weight.columns || parameter !== 0 || weight.columns * 128 > 2147483647) throw new Error("Invalid layer linear shape");
        const identifier = resultVector(weight.rows);
        const output = vector(identifier);
        const resources = [
          weight.buffer, input.buffer,
          allocate(input.count * 4, GPUBufferUsage.STORAGE, transient), output.buffer,
          allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, transient, new Uint32Array([weight.columns, weight.rows, 0, 0]).buffer),
          allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, transient, new Float32Array([weight.scale, 0, 0, 0]).buffer),
          allocate(4, GPUBufferUsage.STORAGE, transient),
        ];
        const group = device.createBindGroup({ layout: bitLinear.layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const pass = encoder.beginComputePass();
        pass.setBindGroup(0, group);
        pass.setPipeline(bitLinear.quantize);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(bitLinear.bitlinear);
        pass.dispatchWorkgroups(weight.rows);
        pass.end();
        return identifier;
      }
      if (name === "cache") {
        const cache = tensors.get(second);
        if (!cache || cache.kind !== "cache" || input.count !== 640 || parameter !== activePosition || cache.position !== parameter - 1 || cacheUpdates.has(cache)) throw new Error("Invalid or nonconsecutive layer cache update");
        encoder.copyBufferToBuffer(input.buffer, 0, cache.buffer, parameter * 640 * 4, 640 * 4);
        cacheUpdates.set(cache, parameter);
        return second;
      }
      let other = dummy;
      let keys = dummy;
      let values = dummy;
      let dimensions = [input.count, 0, 0, 0];
      let groups = Math.ceil(input.count / 128);
      switch (name) {
        case "rmsnorm": {
          const norm = tensors.get(second);
          if (!norm || norm.kind !== "norm" || norm.count !== input.count || parameter !== 0) throw new Error("Invalid layer RMS normalization shape");
          other = norm.buffer;
          groups = 1;
          break;
        }
        case "add":
        case "gated": {
          const operand = vector(second);
          if (operand.count !== input.count || parameter !== 0) throw new Error(`Invalid layer ${name} shape`);
          other = operand.buffer;
          break;
        }
        case "rope":
          if (second !== 0 || (input.count !== 2560 && input.count !== 640) || parameter !== activePosition || !rotaryConfigured) throw new Error("Invalid layer rotary shape, position, or coefficients");
          other = rotaryFactors;
          dimensions = [input.count, 128, parameter, 0];
          groups = Math.ceil(input.count / 256);
          break;
        case "attention": {
          const keyCache = tensors.get(second);
          const valueCache = tensors.get(third);
          if (input.count !== 2560 || !keyCache || keyCache.kind !== "cache" || keyCache.role !== "k" || !valueCache || valueCache.kind !== "cache" || valueCache.role !== "v" || keyCache.layer !== valueCache.layer || parameter !== activePosition || cacheUpdates.get(keyCache) !== parameter || cacheUpdates.get(valueCache) !== parameter || Math.fround(scalar) !== Math.fround(1 / Math.sqrt(128))) throw new Error("Invalid layer attention shape, caches, position, or scale");
          keys = keyCache.buffer;
          values = valueCache.buffer;
          dimensions = [parameter + 1, 20, 5, 128];
          groups = 20;
          if (++attentionCount > layerCount) throw new Error("Too many attention operations");
          break;
        }
        default:
          throw new Error(`Unsupported layer operation ${name}`);
      }
      const identifier = resultVector(input.count);
      if (name === "rmsnorm" && second === weights.get("model.norm.weight")) {
        finalHidden = input;
        finalNormed = vector(identifier);
      }
      const resources = [input.buffer, other, vector(identifier).buffer,
        allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, transient, new Uint32Array(dimensions).buffer),
        allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, transient, new Float32Array([scalar, 0, 0, 0]).buffer),
        keys, values];
      dispatchLayer(name, resources, groups);
      if (name === "add" && layerCount === 30 && activePosition < 3) {
        residualAdds++;
        if (residualAdds % 2 === 0) {
          if (input.count !== 2560 || layerOutputs.length >= 30) throw new Error("Invalid layer checkpoint shape or count");
          layerOutputs.push(vector(identifier));
        }
      }
      return identifier;
    },
  };

  async function execute(position: number, encode: (input: number) => number, inputData?: ArrayBuffer): Promise<LayerExecution> {
    if (closed || busy || !Number.isSafeInteger(position) || position < 0 || position >= 2048 || position !== nextPosition) throw new Error("Layer steps must run once in consecutive position order below 2048");
    busy = true;
    transient = [];
    transientHandles = [];
    cacheUpdates.clear();
    finalHidden = undefined;
    finalNormed = undefined;
    headEncoded = false;
    embeddingEncoded = false;
    rotaryConfigured = false;
    residualAdds = 0;
    layerOutputs = [];
    attentionCount = 0;
    let scopeOpen = false;
    try {
      if (inputData) finiteFloats(inputData, 2560, "layer input");
      device.pushErrorScope("validation");
      scopeOpen = true;
      const inputUploadStart = performance.now();
      let inputHandle = 0;
      if (inputData) {
        inputHandle = resultVector(2560);
        device.queue.writeBuffer(vector(inputHandle).buffer, 0, inputData);
      }
      await device.queue.onSubmittedWorkDone();
      const uploadMs = performance.now() - inputUploadStart;
      const started = performance.now();
      encoder = device.createCommandEncoder();
      activePosition = position;
      operations = 0;
      globalThis.microhsLayer = host;
      const outputHandle = encode(inputHandle);
      const tensor = tensors.get(outputHandle);
      const chosen = tensor?.kind === "chosen" ? tensor : undefined;
      const outputTensor = chosen ? chosen.hidden : vector(outputHandle);
      if (outputTensor.count !== 2560 || cacheUpdates.size !== layerCount * 2 || attentionCount !== layerCount) throw new Error(`Layer program returned an invalid output, attention count, or cache updates for ${layerCount} layers`);
      const normalizedTensor = chosen?.normed ?? (layerCount === 30 ? finalNormed : undefined);
      if (layerCount === 30 && (!normalizedTensor || normalizedTensor.count !== 2560 || finalHidden !== outputTensor)) throw new Error("Network program omitted final normalization");
      const captureLayers = layerCount === 30 && position < 3;
      if (captureLayers && (residualAdds !== 60 || layerOutputs.length !== 30)) throw new Error("Expected thirty final residual layer checkpoints");
      const hiddenBytes = 2560 * 4;
      const chosenOffset = hiddenBytes * 2;
      const logitsOffset = chosenOffset + 8;
      const layersOffset = chosen ? logitsOffset + 128256 * 4 : hiddenBytes + (normalizedTensor ? hiddenBytes : 0);
      const readbackBytes = layersOffset + layerOutputs.length * hiddenBytes;
      const readback = allocate(readbackBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, transient);
      encoder.copyBufferToBuffer(outputTensor.buffer, 0, readback, 0, hiddenBytes);
      if (normalizedTensor) encoder.copyBufferToBuffer(normalizedTensor.buffer, 0, readback, hiddenBytes, hiddenBytes);
      if (chosen) {
        encoder.copyBufferToBuffer(chosen.buffer, 0, readback, chosenOffset, 8);
        encoder.copyBufferToBuffer(chosen.logits, 0, readback, logitsOffset, 128256 * 4);
      }
      for (const [index, layer] of layerOutputs.entries()) encoder.copyBufferToBuffer(layer.buffer, 0, readback, layersOffset + index * hiddenBytes, hiddenBytes);
      const commands = encoder.finish();
      encoder = undefined;
      activePosition = -1;
      device.queue.submit([commands]);
      await readback.mapAsync(GPUMapMode.READ);
      const copied = readback.getMappedRange().slice(0);
      readback.unmap();
      const dispatchReadbackMs = performance.now() - started;
      scopeOpen = false;
      const validation = await device.popErrorScope();
      if (validation) throw new Error(`WebGPU layer validation: ${validation.message}`);
      const output = finiteFloats(copied.slice(0, hiddenBytes), 2560, "GPU hidden output");
      const layers = captureLayers ? layerOutputs.map((_, index) => finiteFloats(copied.slice(layersOffset + index * hiddenBytes, layersOffset + (index + 1) * hiddenBytes), 2560, `GPU layer ${index} checkpoint`)) : undefined;
      const normed = normalizedTensor ? finiteFloats(copied.slice(hiddenBytes, chosenOffset), 2560, "GPU final normalization") : undefined;
      let token: number | undefined;
      let logit: number | undefined;
      if (chosen) {
        const selection = new DataView(copied, chosenOffset, 8);
        token = selection.getUint32(0, true);
        logit = selection.getFloat32(4, true);
        const logits = new Float32Array(copied, logitsOffset, 128256);
        if (token >= 128256 || !Number.isFinite(logit) || !logits.every(Number.isFinite) || logits[token] !== logit) throw new Error("Invalid GPU head selection or nonfinite logits");
      }
      for (const [cache, updatedPosition] of cacheUpdates) cache.position = updatedPosition;
      nextPosition++;
      return { output, layers, normed, token, logit, uploadMs, dispatchReadbackMs, softmaxRoundtrips: 0 };
    } catch (error) {
      closed = true;
      throw error;
    } finally {
      encoder = undefined;
      activePosition = -1;
      try {
        if (scopeOpen) await device.popErrorScope();
      } finally {
        transient.forEach((buffer) => buffer.destroy());
        transientHandles.forEach((identifier) => tensors.delete(identifier));
        cacheUpdates.clear();
        layerOutputs = [];
        busy = false;
      }
    }
  }
  return { uploadWeight, uploadEmbedding, reset, execute, dispose };
}

export async function createLayerRunner(
  device: GPUDevice,
  bytes: (path: string) => Promise<ArrayBuffer>,
  wasm: { _layerStep(layer: number, position: number, input: number): number },
): Promise<(position: number) => Promise<LayerResult>> {
  const manifest = JSON.parse(new TextDecoder().decode(await bytes("../layer-cases/manifest.json"))) as LayerManifest;
  if (!manifest || !Array.isArray(manifest.weights) || manifest.weights.length !== 11 || !Array.isArray(manifest.steps) || manifest.steps.length === 0 || manifest.steps.length > 2048) throw new Error("Invalid layer manifest");
  const names = new Set<string>();
  for (const weight of manifest.weights) {
    if (!weight || typeof weight.name !== "string" || !weight.name.startsWith("model.layers.0.") || names.has(weight.name)) throw new Error("Invalid or duplicate fixture layer weight");
    fixturePath(weight.file);
    names.add(weight.name);
  }
  for (const [position, step] of manifest.steps.entries()) {
    if (!step || step.position !== position) throw new Error("Layer fixture positions must start at zero and be consecutive");
    fixturePath(step.input);
    fixturePath(step.expected);
  }
  const weightData = await Promise.all(manifest.weights.map((weight) => bytes(fixturePath(weight.file))));
  const engine = await createLayerEngine(device, bytes, 1);
  const uploadStart = performance.now();
  try {
    manifest.weights.forEach((weight, index) => engine.uploadWeight(weight, weightData[index]!));
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    engine.dispose();
    throw error;
  }
  const residentUploadMs = performance.now() - uploadStart;
  let nextPosition = 0;
  let busy = false;
  let closed = false;
  return async (position: number): Promise<LayerResult> => {
    if (closed || busy || !Number.isSafeInteger(position) || position !== nextPosition || !manifest.steps[position]) throw new Error("Layer steps must run once in consecutive position order");
    busy = true;
    try {
      const step = manifest.steps[position]!;
      const [input, expected] = await Promise.all([bytes(fixturePath(step.input)), bytes(fixturePath(step.expected))]);
      const reference = finiteFloats(expected, 2560, step.expected);
      const execution = await engine.execute(position, (identifier) => wasm._layerStep(0, position, identifier), input);
      let maxAbsoluteError = 0;
      let mismatch = -1;
      for (let index = 0; index < 2560; index++) {
        const error = Math.abs(execution.output[index]! - reference[index]!);
        maxAbsoluteError = Math.max(maxAbsoluteError, error);
        if (mismatch < 0 && error > 2e-5 * Math.max(1, Math.abs(reference[index]!))) mismatch = index;
      }
      if (mismatch >= 0) throw new Error(`Layer ${position} output[${mismatch}] GPU ${execution.output[mismatch]} != C ${reference[mismatch]}; maximum absolute error ${maxAbsoluteError}; tolerance 2e-5 * max(1, abs(reference))`);
      nextPosition++;
      if (nextPosition === manifest.steps.length) {
        closed = true;
        engine.dispose();
      }
      return { output: execution.output, reference, count: 2560, uploadMs: execution.uploadMs + (position === 0 ? residentUploadMs : 0), dispatchReadbackMs: execution.dispatchReadbackMs, maxAbsoluteError };
    } catch (error) {
      closed = true;
      engine.dispose();
      throw error;
    } finally {
      busy = false;
    }
  };
}
