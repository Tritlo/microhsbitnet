import { createBitLinearRunner } from "./bitlinear.js";
import { createLayerRunner } from "./layer.js";
import { openModel } from "./safetensors.js";
import { createNetworkRunner, networkDiagnostics, type NetworkResult } from "./network.js";
import { createShader } from "./shader.js";

interface MicroHsModule {
  _mhs_init(): void;
  _gpuMain(): void;
  _gpuComplete(ticket: number, count: number): number;
  _gpuStatus(): number;
  _layerStep(layer: number, position: number, input: number): number;
  _networkStep(position: number, token: number, withHead: number): number;
}

interface Sample {
  name: string;
  count: number;
  uploadMs: number;
  medianDispatchReadbackMs?: number;
  dispatchReadbackMs?: number;
  maxAbsoluteError?: number;
}

interface CheckResult {
  passed: boolean;
  adapter: GPUAdapterInfo | Record<string, string>;
  compileMs: number;
  driverTotalMs: number;
  samples: Sample[];
  network?: Omit<NetworkResult, "output">;
  diagnostics?: typeof networkDiagnostics;
  messages: string[];
  error?: string;
}

declare global {
  var microhsGPUStart: (name: string) => number;
  var microhsGPUResultAt: (index: number) => number;
  var microhsGPUReferenceAt: (index: number) => number;
  var microhsCheck: CheckResult | undefined;
}

const messages: string[] = [];
const samples: Sample[] = [];
const resultElement = document.querySelector("#results");

function log(message: string): void {
  messages.push(message);
  if (resultElement) resultElement.textContent = messages.join("\n");
  console.log(message);
}

async function bytes(path: string): Promise<ArrayBuffer> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

async function text(path: string): Promise<string> {
  return new TextDecoder().decode(await bytes(path));
}

function checkedBytes(buffer: ArrayBuffer): Uint8Array<ArrayBuffer> {
  const padded = new Uint8Array(Math.ceil(buffer.byteLength / 4) * 4);
  padded.set(new Uint8Array(buffer));
  return padded;
}

async function start(): Promise<void> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  const info = adapter.info;
  if (info.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(`${info.vendor} ${info.architecture} ${info.device} ${info.description}`)) {
    throw new Error("A physical GPU is required; software fallback is not accepted");
  }
  log(`GPU: ${info.vendor} ${info.architecture} ${info.device} ${info.description}`);
  const model = await openModel("/model.safetensors");
  const embedding = model.tensors.get("model.embed_tokens.weight");
  if (model.tensors.size !== 542 || embedding?.dtype !== "BF16" || embedding.shape.join(",") !== "128256,2560") throw new Error("Unexpected BitNet model metadata");
  log(`Range-loaded model header: ${model.tensors.size} tensors; weights do not enter the WASM heap`);
  const device = await adapter.requestDevice();
  let rejectGPU: (error: unknown) => void;
  const gpuFailure = new Promise<never>((_, reject) => { rejectGPU = reject; });
  device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => rejectGPU(new Error(`WebGPU: ${event.error.message}`)));
  void device.lost.then((lost) => {
    if (lost.reason !== "destroyed") rejectGPU(new Error(`WebGPU device lost: ${lost.message}`));
  });
  const compileStart = performance.now();
  const shader = createShader(device, await text("../generated.wgsl"));
  const diagnostics = await shader.getCompilationInfo();
  const errors = diagnostics.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
  const scalarPipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module: shader, entryPoint: "scalar" },
  });
  const matrixPipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module: shader, entryPoint: "matvec" },
  });
  const compileMs = performance.now() - compileStart;
  log(`WGSL fetch, compilation, and pipelines: ${compileMs.toFixed(3)} ms`);

  const descriptions = new Map<string, [number, number]>(
    (await text("../cases/cases.txt")).trim().split("\n").map((line) => {
      const [name, columns, rows] = line.split(" ");
      if (!name || !columns || !rows) throw new Error("Invalid case description");
      return [name, [Number(columns), Number(rows)]];
    }),
  );
  let bitLinearRunner: ReturnType<typeof createBitLinearRunner> | undefined;
  let layerRunner: ReturnType<typeof createLayerRunner> | undefined;
  let networkResult: Omit<NetworkResult, "output"> | undefined;

  async function runCase(name: string): Promise<Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer>> {
    if (name === "network") {
      const run = await createNetworkRunner(device, bytes, wasm, model, log);
      const { output, ...metrics } = await run();
      activeReference = new Int32Array();
      networkResult = metrics;
      log(`Complete network: ${output.length} GPU-selected tokens; ${metrics.matchingTokens} match C; token reference ${metrics.tokenReferenceMatched ? "matched" : "not matched"}; ${metrics.normalizedVectorsChecked}/${metrics.positionsRun} normalized vectors comparable, maximum relative L2 ${(metrics.maxNormedRelativeL2 * 100).toFixed(4)}%; 1% diagnostic ${metrics.numericalBoundPassed ? "met" : "not met"}; prefill ${metrics.prefillMs.toFixed(1)} ms; remaining generation ${metrics.generationMs.toFixed(1)} ms`);
      return output;
    }
    if (name.startsWith("layer0-pos")) {
      layerRunner ??= createLayerRunner(device, bytes, wasm);
      const run = await layerRunner;
      const { output, reference, ...sample } = await run(Number(name.slice("layer0-pos".length)));
      activeReference = reference;
      samples.push({ name, ...sample });
      log(`${name}: complete decoder layer tolerance PASS; max absolute error ${sample.maxAbsoluteError}; dispatch/readback ${sample.dispatchReadbackMs.toFixed(3)} ms`);
      return output;
    }
    if (name.startsWith("bl-")) {
      bitLinearRunner ??= createBitLinearRunner(device, bytes);
      const run = await bitLinearRunner;
      const { output, reference, ...sample } = await run(name.slice(3));
      activeReference = reference;
      samples.push({ name, ...sample });
      log(`${name}: quantization exact, output tolerance PASS; max absolute error ${sample.maxAbsoluteError}; dispatch/readback median ${sample.medianDispatchReadbackMs.toFixed(3)} ms`);
      return output;
    }
    const scalar = name === "scalar";
    const shape = descriptions.get(name);
    if (!scalar && !shape) throw new Error(`Unknown case ${name}`);
    const [input, expected, activations] = await Promise.all([
      bytes(`../cases/${scalar ? "scalar-input" : `${name}.weights`}.bin`),
      bytes(`../cases/${scalar ? "scalar-expected" : `${name}.expected`}.bin`),
      scalar ? Promise.resolve(new ArrayBuffer(0)) : bytes(`../cases/${name}.acts.bin`),
    ]);
    const count = expected.byteLength / 4;
    if (!Number.isSafeInteger(count) || count <= 0 || count > 1048576) throw new Error(`Invalid output count for ${name}`);
    if (scalar) {
      if (input.byteLength !== expected.byteLength || !new Float32Array(input).every(Number.isFinite) || !new Float32Array(expected).every(Number.isFinite)) throw new Error("Invalid scalar fixture");
    } else {
      const [columns, rows] = shape!;
      if (!Number.isSafeInteger(columns) || !Number.isSafeInteger(rows) || columns <= 0 || rows !== count || rows % 4 || columns > Math.floor(2147483647 / 128) || columns * rows > 128 * 1024 * 1024) throw new Error(`Invalid matrix shape ${name}`);
      if (input.byteLength !== columns * rows / 4 || activations.byteLength !== columns) throw new Error(`Invalid matrix fixture size ${name}`);
      if (new Uint8Array(input).some((value) => (value & (value >> 1) & 0x55) !== 0)) throw new Error(`Invalid packed ternary field ${name}`);
    }
    const uploadStart = performance.now();
    const buffers: GPUBuffer[] = [];
    function storage(data: ArrayBuffer): GPUBuffer {
      const padded = checkedBytes(data);
      const buffer = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, padded);
      buffers.push(buffer);
      return buffer;
    }
    const weights = storage(input);
    const acts = scalar ? undefined : storage(activations);
    const output = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniform, 0, new Uint32Array(scalar ? [count, 0, 0, 0] : [shape![0], shape![1], 0, 0]));
    const pipeline = scalar ? scalarPipeline : matrixPipeline;
    const resources = scalar ? [weights, output, uniform] : [weights, acts!, output, uniform];
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    await device.queue.onSubmittedWorkDone();
    const uploadMs = performance.now() - uploadStart;
    let actual: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> = new Int32Array(0);
    const durations: number[] = [];
    for (let iteration = 0; iteration < 11; iteration++) {
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(scalar ? Math.ceil(count / 128) : count);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, expected.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const copied = readback.getMappedRange().slice(0);
      readback.unmap();
      if (iteration >= 2) durations.push(performance.now() - started);
      actual = scalar ? new Float32Array(copied) : new Int32Array(copied);
      const reference = scalar ? new Float32Array(expected) : new Int32Array(expected);
      for (let index = 0; index < count; index++) {
        if (actual[index] !== reference[index]) {
          throw new Error(`${name}[${index}]: GPU ${actual[index]} != source ${reference[index]}`);
        }
      }
    }
    durations.sort((left, right) => left - right);
    const sample = { name, count, uploadMs, medianDispatchReadbackMs: durations[4]! };
    samples.push(sample);
    log(`${name}: ${count} exact values PASS; upload ${uploadMs.toFixed(3)} ms; dispatch/readback median ${sample.medianDispatchReadbackMs.toFixed(3)} ms`);
    [...buffers, output, readback, uniform].forEach((buffer) => buffer.destroy());
    return actual;
  }

  let wasm: MicroHsModule;
  let activeResult: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> = new Int32Array(0);
  let activeReference: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> = new Int32Array(0);
  let nextTicket = 0;
  let resolveDriver: () => void;
  let rejectDriver: (error: unknown) => void;
  const driver = new Promise<void>((resolve, reject) => { resolveDriver = resolve; rejectDriver = reject; });
  globalThis.microhsGPUResultAt = (index) => {
    const value = activeResult[index];
    if (value === undefined) throw new Error(`Invalid WASM result index ${index}`);
    return value;
  };
  globalThis.microhsGPUReferenceAt = (index) => {
    const value = activeReference[index];
    if (value === undefined) throw new Error(`Invalid WASM reference index ${index}`);
    return value;
  };
  globalThis.microhsGPUStart = (name) => {
    const ticket = ++nextTicket;
    setTimeout(() => {
      void runCase(name).then(async (output) => {
        activeResult = output;
        if (!name.startsWith("bl-") && !name.startsWith("layer0-") && name !== "network") {
          const reference = await bytes(`../cases/${name === "scalar" ? "scalar-expected" : `${name}.expected`}.bin`);
          activeReference = name === "scalar" ? new Float32Array(reference) : new Int32Array(reference);
        }
        if (wasm._gpuComplete(ticket, output.length) !== 1) throw new Error(`WASM rejected ${name}`);
        const status = wasm._gpuStatus();
        if (status === 1) resolveDriver();
        else if (status < 0) throw new Error("WASM driver failed");
      }).catch(rejectDriver);
    }, 0);
    return ticket;
  };
  const wasmStart = performance.now();
  const modulePath = "./microhs.mjs";
  const createMicroHs = (await import(modulePath)).default as (options: object) => Promise<MicroHsModule>;
  wasm = await createMicroHs({ print: log, printErr: log });
  if ("_softmax" in wasm) throw new Error("The GPU-only build must not export a WASM softmax fallback");
  wasm._mhs_init();
  wasm._gpuMain();
  await Promise.race([driver, gpuFailure]);
  const driverTotalMs = performance.now() - wasmStart;
  globalThis.microhsCheck = {
    passed: true,
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    compileMs, driverTotalMs, samples, network: networkResult, diagnostics: networkDiagnostics, messages,
  };
  log("MicroHs → WASM → generated WGSL → physical WebGPU PASS");
  device.destroy();
}

void start().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  log(message);
  globalThis.microhsCheck = { passed: false, adapter: {}, compileMs: 0, driverTotalMs: 0, samples, diagnostics: networkDiagnostics, messages, error: message };
});
