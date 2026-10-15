interface WasmModule {
  HEAPU8: Uint8Array<ArrayBufferLike>;
  _malloc(size: number): number;
  _free(pointer: number): void;
  _mhs_init(): void;
  _gpuMain(): void;
  _browserComplete(ticket: number, status: number, pointer: number): void;
}

interface AdapterDescription {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
}

interface BridgeReport extends Record<string, unknown> {
  passed: boolean;
  error?: string;
}

interface PlanAction {
  prepare(arguments_: readonly number[]): void;
  encode(encoder: GPUCommandEncoder, pass: () => GPUComputePassEncoder, endPass: () => void): void;
}

type Resource =
  | { kind: "host"; data: ArrayBuffer }
  | { kind: "buffer"; value: GPUBuffer; size: number; usage: number }
  | { kind: "shader"; value: GPUShaderModule }
  | { kind: "layout"; value: GPUBindGroupLayout; pipeline: GPUPipelineLayout; types: GPUBufferBindingType[] }
  | { kind: "pipeline"; value: GPUComputePipeline; layout: number }
  | { kind: "encoder"; value: GPUCommandEncoder }
  | { kind: "plan"; actions: PlanAction[]; arguments: number; dependencies: Map<number, Resource> };

interface BrowserBridge {
  request(ticket: number, serialized: string): void;
  size(identifier: number): number;
  number(identifier: number, offset: number, kind: string): number;
  copy(identifier: number, offset: number, length: number, heap: Uint8Array<ArrayBufferLike>, target: number): void;
  drop(identifier: number): void;
  log(message: string): void;
  clock(): number;
  report(serialized: string): void;
}

const browser = globalThis as unknown as { microhsBrowser: BrowserBridge; microhsCheck?: BridgeReport };
const resources = new Map<number, Resource>();
const messages: string[] = [];
const abort = new AbortController();
const resultElement = document.querySelector("#results");
let wasm: WasmModule | undefined;
let device: GPUDevice | undefined;
let adapterDescription: AdapterDescription | undefined;
let pendingTicket = 0;
let lastTicket = 0;
let activeWasm = false;
let stopped = false;
let uploadedBytes = 0;
let submissions = 0;
let copiedToWasmBytes = 0;
let bulkCopiedToWasmBytes = 0;
let liveBufferBytes = 0;
let bufferAllocations = 0;
let bindGroupCreations = 0;
let planCompilations = 0;
let planRuns = 0;

function integer(value: unknown, name: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`Invalid ${name}`);
  return value;
}

function finite(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Invalid finite numeric argument");
  return value;
}

function float(value: unknown): number {
  const result = finite(value);
  if (!Number.isFinite(Math.fround(result))) throw new Error("Invalid f32 value");
  return result;
}

function identifier(value: unknown): number {
  return integer(value, "resource identifier", 1, 2147483647);
}

function string(value: unknown, name: string): string {
  if (typeof value !== "string") throw new Error(`Invalid ${name}`);
  return value;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`Invalid ${name}`);
  return value;
}

function argumentsOf(value: unknown): unknown[] {
  const fields = array(value, "command");
  string(fields[0], "operation");
  return fields;
}

function arity(fields: unknown[], count: number): void {
  if (fields.length !== count) throw new Error(`Invalid argument count for ${String(fields[0])}`);
}

function resource<Kind extends Resource["kind"]>(value: unknown, kind: Kind): Extract<Resource, { kind: Kind }> {
  const key = identifier(value);
  const found = resources.get(key);
  if (!found || found.kind !== kind) throw new Error(`Invalid ${kind} handle ${key}`);
  return found as Extract<Resource, { kind: Kind }>;
}

function retain(value: unknown, retained: Resource): void {
  if (stopped) throw new Error("The browser application already completed");
  const key = identifier(value);
  if (resources.has(key)) throw new Error(`Duplicate resource handle ${key}`);
  resources.set(key, retained);
}

function available(value: unknown): number {
  const key = identifier(value);
  if (resources.has(key)) throw new Error(`Duplicate resource handle ${key}`);
  return key;
}

function bounds(size: number, offset: unknown, length: unknown, alignment = 1): [number, number] {
  const first = integer(offset, "byte offset");
  const count = integer(length, "byte length");
  if (first % alignment || count % alignment || first > size || count > size - first) throw new Error("Byte range exceeds its buffer or alignment");
  return [first, count];
}

function gpu(): GPUDevice {
  if (!device) throw new Error("The browser device is not initialized");
  return device;
}

function log(message: string): void {
  messages.push(message);
  if (resultElement) resultElement.textContent = messages.join("\n");
  console.log(message);
}

function cleanup(): void {
  abort.abort();
  for (const retained of resources.values()) if (retained.kind === "buffer") retained.value.destroy();
  resources.clear();
  liveBufferBytes = 0;
  device?.destroy();
}

function publish(result: BridgeReport): void {
  stopped = true;
  pendingTicket = 0;
  browser.microhsCheck = { ...result, adapter: adapterDescription ?? {}, messages: messages.slice() };
  cleanup();
}

function failure(reason: unknown): void {
  if (stopped && browser.microhsCheck?.passed === false) return;
  const error = reason instanceof Error ? reason.stack ?? reason.message : String(reason);
  log(error);
  publish({ passed: false, error });
}

function enter(action: () => void): void {
  if (activeWasm) throw new Error("Cannot reenter the active WASM evaluator");
  activeWasm = true;
  try {
    action();
  } catch (error) {
    failure(error);
  } finally {
    activeWasm = false;
  }
}

function complete(ticket: number, status: number, value: unknown): void {
  if (stopped) return;
  if (pendingTicket !== ticket) { failure(new Error("Stale browser response ticket")); return; }
  const module = wasm;
  if (!module) { failure(new Error("WASM is not initialized")); return; }
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  const size = integer(encoded.byteLength + 1, "WASM response byte length", 1, 2147483647);
  pendingTicket = 0;
  enter(() => {
    const pointer = module._malloc(size) >>> 0;
    if (!pointer) throw new Error("Cannot allocate the browser response in WASM");
    try {
      bounds(module.HEAPU8.byteLength, pointer, size);
      module.HEAPU8.set(encoded, pointer);
      module.HEAPU8[pointer + encoded.byteLength] = 0;
      copiedToWasmBytes += size;
      module._browserComplete(ticket, status, pointer);
    } finally {
      module._free(pointer);
    }
  });
}

async function initialize(): Promise<unknown> {
  if (!device) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter");
    const info = adapter.info;
    adapterDescription = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
    device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => failure(new Error(`WebGPU: ${event.error.message}`)));
    void device.lost.then((lost) => { if (lost.reason !== "destroyed") failure(new Error(`WebGPU device lost: ${lost.message}`)); });
  }
  return { adapter: adapterDescription, maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize };
}

function upload(buffer: GPUBuffer, size: number, data: ArrayBuffer): void {
  const paddedLength = Math.ceil(data.byteLength / 4) * 4;
  if (paddedLength > size) throw new Error("Host upload exceeds its GPU buffer");
  if (paddedLength === data.byteLength) {
    gpu().queue.writeBuffer(buffer, 0, data);
  } else {
    const padded = new Uint8Array(paddedLength);
    padded.set(new Uint8Array(data));
    gpu().queue.writeBuffer(buffer, 0, padded);
  }
  uploadedBytes += paddedLength;
}

function allocate(key: unknown, size: number, usage: number, data?: ArrayBuffer): void {
  available(key);
  if (size <= 0 || size % 4 || size > gpu().limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) !== 0 && size > gpu().limits.maxStorageBufferBindingSize)) throw new Error("GPU buffer size exceeds its alignment or device limit");
  const value = gpu().createBuffer({ size, usage });
  bufferAllocations++;
  retain(key, { kind: "buffer", value, size, usage });
  liveBufferBytes += size;
  if (data) upload(value, size, data);
}

function bindingGroup(layout: Extract<Resource, { kind: "layout" }>, bindings: unknown[], lookup = (key: unknown) => resource(key, "buffer")): GPUBindGroup {
  if (bindings.length !== layout.types.length) throw new Error("Dispatch binding count differs");
  const entries = bindings.map((key, binding) => {
    const buffer = lookup(key);
    const uniform = layout.types[binding] === "uniform";
    const required = uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;
    if ((buffer.usage & required) === 0) throw new Error("Dispatch buffer usage differs");
    if (uniform && buffer.size > gpu().limits.maxUniformBufferBindingSize) throw new Error("Uniform binding exceeds its device limit");
    return { binding, resource: { buffer: buffer.value } };
  });
  const group = gpu().createBindGroup({ layout: layout.value, entries });
  bindGroupCreations++;
  return group;
}

function compilePlan(key: unknown, templates: unknown): void {
  available(key);
  const dependencies = new Map<number, Resource>();
  const writes = new Set<GPUBuffer>();
  let argumentCount = 0;
  let references = 0;
  function dependency<Kind extends Resource["kind"]>(value: unknown, kind: Kind): Extract<Resource, { kind: Kind }> {
    const found = resource(value, kind);
    dependencies.set(identifier(value), found);
    return found;
  }
  function parameter(value: unknown, check: (value: unknown) => number): (arguments_: readonly number[]) => number {
    if (typeof value === "number") {
      const literal = check(value);
      return () => literal;
    }
    const fields = argumentsOf(value);
    arity(fields, 2);
    if (fields[0] !== "arg") throw new Error("Invalid plan argument reference");
    const index = integer(fields[1], "plan argument index", 0, 2147483646);
    argumentCount = Math.max(argumentCount, index + 1);
    references++;
    return (arguments_) => check(arguments_[index]);
  }
  function aligned(value: unknown, maximum: number): number {
    const result = integer(value, "plan byte range", 0, maximum);
    if (result % 4) throw new Error("Unaligned plan byte range");
    return result;
  }
  function compile(value: unknown): PlanAction {
    const fields = argumentsOf(value);
    switch (fields[0]) {
      case "dispatch": {
        arity(fields, 6);
        const pipeline = dependency(fields[1], "pipeline");
        const layout = dependency(fields[2], "layout");
        if (pipeline.layout !== fields[2]) throw new Error("Pipeline layout handle differs");
        const choices = array(fields[3], "binding choices").map((choice) => bindingGroup(layout, array(choice, "dispatch bindings"), (key) => dependency(key, "buffer")));
        if (!choices.length) throw new Error("Cannot compile empty binding choices");
        const select = parameter(fields[4], (value) => integer(value, "binding selection", 0, choices.length - 1));
        const count = parameter(fields[5], (value) => integer(value, "workgroup count", 1, gpu().limits.maxComputeWorkgroupsPerDimension));
        let selection = 0;
        let groups = 1;
        return {
          prepare(arguments_) { selection = select(arguments_); groups = count(arguments_); },
          encode(_encoder, pass) {
            const active = pass();
            active.setPipeline(pipeline.value);
            active.setBindGroup(0, choices[selection]!);
            active.dispatchWorkgroups(groups);
          },
        };
      }
      case "copy": {
        arity(fields, 6);
        const source = dependency(fields[1], "buffer");
        const target = dependency(fields[3], "buffer");
        if (source === target || (source.usage & GPUBufferUsage.COPY_SRC) === 0 || (target.usage & GPUBufferUsage.COPY_DST) === 0) throw new Error("Invalid plan copy buffers or usage");
        const first = parameter(fields[2], (value) => aligned(value, source.size));
        const second = parameter(fields[4], (value) => aligned(value, target.size));
        const length = parameter(fields[5], (value) => aligned(value, Math.min(source.size, target.size)));
        let sourceOffset = 0;
        let targetOffset = 0;
        let size = 0;
        return {
          prepare(arguments_) {
            [sourceOffset, size] = bounds(source.size, first(arguments_), length(arguments_), 4);
            [targetOffset] = bounds(target.size, second(arguments_), size, 4);
          },
          encode(encoder, _pass, endPass) { endPass(); encoder.copyBufferToBuffer(source.value, sourceOffset, target.value, targetOffset, size); },
        };
      }
      case "write": {
        arity(fields, 4);
        const buffer = dependency(fields[1], "buffer");
        const kind = string(fields[2], "plan write representation");
        if (kind !== "u32" && kind !== "f32") throw new Error("Unknown plan write representation");
        if ((buffer.usage & GPUBufferUsage.COPY_DST) === 0 || writes.has(buffer.value)) throw new Error("Plan writes require COPY_DST and one write per buffer");
        writes.add(buffer.value);
        const values = array(fields[3], "plan write values").map((value) => parameter(value, kind === "u32" ? (value) => integer(value, "u32 value", 0, 4294967295) : float));
        if (!values.length || values.length * 4 > buffer.size) throw new Error("Invalid plan write length");
        const data = kind === "u32" ? new Uint32Array(values.length) : new Float32Array(values.length);
        return {
          prepare(arguments_) { values.forEach((read, index) => { data[index] = read(arguments_); }); },
          encode() { upload(buffer.value, buffer.size, data.buffer); },
        };
      }
      case "clear": {
        arity(fields, 4);
        const buffer = dependency(fields[1], "buffer");
        if ((buffer.usage & GPUBufferUsage.COPY_DST) === 0) throw new Error("Plan clear requires COPY_DST");
        const first = parameter(fields[2], (value) => aligned(value, buffer.size));
        const length = parameter(fields[3], (value) => aligned(value, buffer.size));
        let offset = 0;
        let size = 0;
        return {
          prepare(arguments_) { [offset, size] = bounds(buffer.size, first(arguments_), length(arguments_), 4); },
          encode(encoder, _pass, endPass) { endPass(); encoder.clearBuffer(buffer.value, offset, size); },
        };
      }
      default:
        throw new Error(`Unknown plan operation ${String(fields[0])}`);
    }
  }
  const actions = array(templates, "plan templates").map((template) => {
    const previous = references;
    const action = compile(template);
    if (references === previous) action.prepare([]);
    return action;
  });
  retain(key, { kind: "plan", actions, arguments: argumentCount, dependencies });
  planCompilations++;
}

function runPlan(key: unknown, values: unknown): void {
  const plan = resource(key, "plan");
  const arguments_ = array(values, "plan arguments").map(finite);
  if (arguments_.length < plan.arguments) throw new Error("Missing plan arguments");
  for (const [identifier, retained] of plan.dependencies) if (resources.get(identifier) !== retained) throw new Error(`Released plan dependency ${identifier}`);
  for (const action of plan.actions) action.prepare(arguments_);
  const encoder = gpu().createCommandEncoder();
  let active: GPUComputePassEncoder | undefined;
  const pass = (): GPUComputePassEncoder => active ??= encoder.beginComputePass();
  const endPass = (): void => { active?.end(); active = undefined; };
  try {
    for (const action of plan.actions) action.encode(encoder, pass, endPass);
    endPass();
    gpu().queue.submit([encoder.finish()]);
    submissions++;
    planRuns++;
  } finally {
    endPass();
  }
}

function release(values: unknown, destroy: unknown): void {
  if (typeof destroy !== "boolean") throw new Error("Invalid resource destruction flag");
  for (const value of array(values, "release handles")) {
    const key = integer(value, "release handle", 0, 2147483647);
    const retained = resources.get(key);
    if (!retained) continue;
    if (retained.kind === "buffer") {
      if (destroy) retained.value.destroy();
      liveBufferBytes -= retained.size;
    }
    resources.delete(key);
  }
}

async function batchCommand(value: unknown): Promise<void> {
  if (stopped) throw new Error("The browser application already completed");
  const fields = argumentsOf(value);
  switch (fields[0]) {
    case "plan":
      arity(fields, 3);
      compilePlan(fields[1], fields[2]);
      return;
    case "run":
      arity(fields, 3);
      runPlan(fields[1], fields[2]);
      return;
    case "write-host": {
      arity(fields, 3);
      const buffer = resource(fields[1], "buffer");
      if ((buffer.usage & GPUBufferUsage.COPY_DST) === 0) throw new Error("Host upload requires COPY_DST");
      upload(buffer.value, buffer.size, resource(fields[2], "host").data);
      return;
    }
    case "buffer": {
      arity(fields, 5);
      const host = integer(fields[4], "upload host handle", 0, 2147483647);
      allocate(fields[1], integer(fields[2], "buffer size", 1), integer(fields[3], "buffer usage", 1, 1023), host ? resource(host, "host").data : undefined);
      return;
    }
    case "array": {
      arity(fields, 4);
      const kind = string(fields[2], "array representation");
      const values = array(fields[3], "array values");
      if (!values.length) throw new Error("Cannot allocate an empty GPU array");
      let data: ArrayBuffer;
      if (kind === "u32") {
        data = Uint32Array.from(values.map((item) => integer(item, "u32 value", 0, 4294967295))).buffer;
      } else if (kind === "f32") {
        const numbers = values.map((item) => {
          if (typeof item !== "number" || !Number.isFinite(item) || !Number.isFinite(Math.fround(item))) throw new Error("Invalid f32 array value");
          return item;
        });
        data = Float32Array.from(numbers).buffer;
      } else throw new Error("Unknown array representation");
      allocate(fields[1], data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, data);
      return;
    }
    case "shader": {
      arity(fields, 3);
      available(fields[1]);
      const code = new TextDecoder("utf-8", { fatal: true }).decode(resource(fields[2], "host").data);
      const descriptor: GPUShaderModuleDescriptor & { strictMath?: boolean } = { code };
      if (new URLSearchParams(location.search).get("strict") === "1") descriptor.strictMath = true;
      const module = gpu().createShaderModule(descriptor);
      retain(fields[1], { kind: "shader", value: module });
      const errors = (await module.getCompilationInfo()).messages.filter((message) => message.type === "error");
      if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
      return;
    }
    case "layout": {
      arity(fields, 3);
      available(fields[1]);
      const types = array(fields[2], "binding types").map((item): GPUBufferBindingType => {
        if (item !== "uniform" && item !== "storage" && item !== "read-only-storage") throw new Error("Invalid buffer binding type");
        return item;
      });
      const layout = gpu().createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      retain(fields[1], { kind: "layout", value: layout, pipeline: gpu().createPipelineLayout({ bindGroupLayouts: [layout] }), types });
      return;
    }
    case "pipeline": {
      arity(fields, 5);
      available(fields[1]);
      const layout = resource(fields[3], "layout");
      const pipeline = await gpu().createComputePipelineAsync({ layout: layout.pipeline, compute: { module: resource(fields[2], "shader").value, entryPoint: string(fields[4], "entry point") } });
      retain(fields[1], { kind: "pipeline", value: pipeline, layout: identifier(fields[3]) });
      return;
    }
    case "encoder":
      arity(fields, 2);
      available(fields[1]);
      retain(fields[1], { kind: "encoder", value: gpu().createCommandEncoder() });
      return;
    case "dispatch": {
      arity(fields, 6);
      const encoder = resource(fields[1], "encoder");
      const pipeline = resource(fields[2], "pipeline");
      const layout = resource(fields[3], "layout");
      if (pipeline.layout !== fields[3]) throw new Error("Pipeline layout handle differs");
      const groups = integer(fields[5], "workgroup count", 1, gpu().limits.maxComputeWorkgroupsPerDimension);
      const group = bindingGroup(layout, array(fields[4], "dispatch bindings"));
      const pass = encoder.value.beginComputePass();
      pass.setPipeline(pipeline.value);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
      pass.end();
      return;
    }
    case "copy": {
      arity(fields, 7);
      const source = resource(fields[2], "buffer");
      const target = resource(fields[4], "buffer");
      if ((source.usage & GPUBufferUsage.COPY_SRC) === 0 || (target.usage & GPUBufferUsage.COPY_DST) === 0) throw new Error("Copy buffer usage differs");
      const [sourceOffset, size] = bounds(source.size, fields[3], fields[6], 4);
      const [targetOffset] = bounds(target.size, fields[5], size, 4);
      resource(fields[1], "encoder").value.copyBufferToBuffer(source.value, sourceOffset, target.value, targetOffset, size);
      return;
    }
    case "submit": {
      arity(fields, 2);
      const key = identifier(fields[1]);
      const encoder = resource(key, "encoder");
      gpu().queue.submit([encoder.value.finish()]);
      resources.delete(key);
      submissions++;
      return;
    }
    case "readback": {
      arity(fields, 3);
      available(fields[2]);
      const buffer = resource(fields[1], "buffer");
      if ((buffer.usage & GPUBufferUsage.MAP_READ) === 0) throw new Error("Readback buffer is not readable");
      await buffer.value.mapAsync(GPUMapMode.READ);
      try {
        retain(fields[2], { kind: "host", data: buffer.value.getMappedRange().slice(0) });
      } finally {
        buffer.value.unmap();
      }
      return;
    }
    case "wait":
      arity(fields, 1);
      await gpu().queue.onSubmittedWorkDone();
      return;
    case "release":
      arity(fields, 3);
      release(fields[1], fields[2]);
      return;
    default:
      throw new Error(`Unknown batch operation ${String(fields[0])}`);
  }
}

async function perform(value: unknown): Promise<unknown> {
  const fields = argumentsOf(value);
  switch (fields[0]) {
    case "init":
      arity(fields, 1);
      return initialize();
    case "fetch": {
      arity(fields, 5);
      const key = available(fields[1]);
      const url = string(fields[2], "fetch URL");
      const start = integer(fields[3], "fetch offset", -1);
      const count = integer(fields[4], "fetch length", -1);
      const whole = start === -1 && count === -1;
      if (!whole && (start < 0 || count < 1 || !Number.isSafeInteger(start + count))) throw new Error("Invalid fetch range");
      const response = await fetch(url, { signal: abort.signal, headers: whole ? {} : { Range: `bytes=${start}-${start + count - 1}` } });
      if (!response.ok) throw new Error(`Fetch failed: HTTP ${response.status}`);
      let total = 0;
      if (!whole) {
        const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("Content-Range") ?? "");
        if (response.status !== 206 || !range || Number(range[1]) !== start || Number(range[2]) !== start + count - 1) throw new Error("HTTP range response differs");
        total = integer(Number(range[3]), "file byte length", start + count);
      }
      const data = await response.arrayBuffer();
      if (!whole && data.byteLength !== count) throw new Error("HTTP range body length differs");
      retain(key, { kind: "host", data });
      return { length: data.byteLength, totalBytes: whole ? data.byteLength : total };
    }
    case "text":
      arity(fields, 2);
      return new TextDecoder("utf-8", { fatal: true }).decode(resource(fields[1], "host").data);
    case "stats":
      arity(fields, 1);
      return { uploadedBytes, submissions, copiedToWasmBytes, bulkCopiedToWasmBytes, liveBufferBytes, bufferAllocations, bindGroupCreations, planCompilations, planRuns };
    case "batch": {
      arity(fields, 2);
      const commands = array(fields[1], "batch commands");
      const active = gpu();
      active.pushErrorScope("validation");
      let validation: GPUError | null;
      try {
        for (const command of commands) await batchCommand(command);
      } finally {
        validation = await active.popErrorScope();
      }
      if (validation) throw new Error(`WebGPU: ${validation.message}`);
      return null;
    }
    default:
      throw new Error(`Unknown browser operation ${String(fields[0])}`);
  }
}

browser.microhsBrowser = {
  request(ticket, serialized): void {
    integer(ticket, "request ticket", 1, 2147483647);
    if (stopped || pendingTicket || ticket <= lastTicket) throw new Error("Invalid or concurrent browser request ticket");
    pendingTicket = ticket;
    lastTicket = ticket;
    setTimeout(() => {
      if (stopped) return;
      void Promise.resolve().then(() => perform(JSON.parse(serialized) as unknown)).then(
        (value) => complete(ticket, 1, value),
        (error: unknown) => complete(ticket, 0, error instanceof Error ? error.message : String(error)),
      ).catch(failure);
    }, 0);
  },
  size(key): number {
    return integer(resource(key, "host").data.byteLength, "host byte length", 0, 2147483647);
  },
  number(key, offset, kind): number {
    const data = resource(key, "host").data;
    const widths: Record<string, number> = { u8: 1, u16: 2, u32: 4, i32: 4, f32: 4 };
    const width = widths[kind];
    if (width === undefined) throw new Error("Unknown host number representation");
    const [first] = bounds(data.byteLength, offset, width);
    const view = new DataView(data);
    let value: number;
    switch (kind) {
      case "u8": value = view.getUint8(first); break;
      case "u16": value = view.getUint16(first, true); break;
      case "u32": value = view.getUint32(first, true); break;
      case "i32": value = view.getInt32(first, true); break;
      case "f32": value = view.getFloat32(first, true); break;
      default: throw new Error("Unknown host number representation");
    }
    if (!Number.isFinite(value)) throw new Error("Nonfinite host scalar");
    copiedToWasmBytes += width;
    return value;
  },
  copy(key, offset, length, heap, target): void {
    const data = resource(key, "host").data;
    const [first, count] = bounds(data.byteLength, offset, length);
    if (!Number.isInteger(target) || target < -2147483648 || target > 4294967295) throw new Error("Invalid WASM destination pointer");
    const destination = target >>> 0;
    bounds(heap.byteLength, destination, count);
    heap.set(new Uint8Array(data, first, count), destination);
    copiedToWasmBytes += count;
    bulkCopiedToWasmBytes += count;
  },
  drop(key): void {
    integer(key, "host handle", 0, 2147483647);
    const found = resources.get(key);
    if (found && found.kind !== "host") throw new Error("Cannot drop a GPU resource as host bytes");
    resources.delete(key);
  },
  log,
  clock: () => performance.now(),
  report(serialized): void {
    const result: unknown = JSON.parse(serialized);
    if (!result || typeof result !== "object" || Array.isArray(result) || typeof (result as Record<string, unknown>).passed !== "boolean") throw new Error("Invalid browser application report");
    if (stopped) throw new Error("The browser application already completed");
    publish(result as BridgeReport);
  },
};

async function start(): Promise<void> {
  const modulePath = "./microhs.mjs";
  const createMicroHs = (await import(modulePath)).default as (options: { print(message: string): void; printErr(message: string): void }) => Promise<WasmModule>;
  wasm = await createMicroHs({ print: log, printErr: log });
  for (const name of ["_malloc", "_free", "_mhs_init", "_gpuMain", "_browserComplete"] as const) if (typeof wasm[name] !== "function") throw new Error(`Missing WASM bridge export ${name}`);
  if (!(wasm.HEAPU8 instanceof Uint8Array)) throw new Error("Missing WASM byte heap");
  const module = wasm;
  enter(() => { module._mhs_init(); module._gpuMain(); });
}

void start().catch(failure);

export {};
