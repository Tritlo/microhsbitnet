interface WasmModule {
  HEAPU8: Uint8Array<ArrayBufferLike>;
  _malloc(size: number): number;
  _free(pointer: number): void;
  _mhs_init(): void;
  _gpuMain(): void;
  _browserComplete(ticket: number, status: number, pointer: number, size: number): void;
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

type BrowserObject = ArrayBuffer | GPUBuffer | GPUShaderModule | GPUBindGroupLayout | GPUPipelineLayout | GPUBindGroup | GPUComputePipeline | GPUCommandEncoder | GPUComputePassEncoder;

type BrowserRequest =
  | ["init"]
  | ["fetch", number, string, number, number]
  | ["text", number]
  | ["shader", number, number]
  | ["pipeline", number, number, number, string]
  | ["readback", number, number]
  | ["wait"]
  | ["scope"];

interface BrowserBridge {
  call(op: number, first: number, second: number, third: number, fourth: number, fifth: number, sixth: number): void;
  array(op: number, key: number, other: number, heap: Uint8Array<ArrayBufferLike>, pointer: number, count: number): void;
  write(buffer: number, heap: Uint8Array<ArrayBufferLike>, pointer: number, bytes: number): void;
  request(ticket: number, serialized: string): void;
  number(identifier: number, offset: number, kind: string): number;
  copy(identifier: number, offset: number, length: number, heap: Uint8Array<ArrayBufferLike>, target: number): void;
  log(message: string): void;
  clock(): number;
  report(serialized: string): void;
}

const browser = globalThis as unknown as { microhsBrowser: BrowserBridge; microhsCheck?: BridgeReport };
const objects: (BrowserObject | undefined)[] = [];
const messages: string[] = [];
const abort = new AbortController();
const resultElement = document.querySelector("#results");
let wasm: WasmModule | undefined;
let device: GPUDevice | undefined;
let adapterDescription: AdapterDescription | undefined;
let activeWasm = false;
let stopped = false;

function log(message: string): void {
  messages.push(message);
  if (resultElement) resultElement.textContent = messages.join("\n");
  const progress = document.querySelector("#progress");
  const loaded = /Loaded (\d+)\/(\d+)/.exec(message);
  if (progress && loaded) {
    const status = document.querySelector("#status");
    if (status) status.textContent = "Loading model weights…";
    const done = Number(loaded[1]);
    const total = Number(loaded[2]);
    progress.textContent = `Weights ${done}/${total}`;
    const bar = document.querySelector<HTMLElement>("#bar");
    if (bar) bar.style.width = `${Math.round((100 * done) / total)}%`;
  }
  if (message.startsWith("PROMPT ")) {
    const status = document.querySelector("#status");
    if (status) status.textContent = "Generating…";
    appendTranscript(tokenDecoder.decode());
    appendTranscript("\n\n" + message.slice("PROMPT ".length) + "\n");
  }
  if (message.startsWith("TOKEN ")) {
    const id = Number(message.slice("TOKEN ".length));
    if (Number.isInteger(id)) appendTranscript(decodeToken(id));
  }
  console.log(message);
}

function publish(result: BridgeReport): void {
  const status = document.querySelector("#status");
  if (status && result.passed) status.textContent = "Generation finished.";
  stopped = true;
  browser.microhsCheck = { ...result, adapter: adapterDescription ?? {}, messages: messages.slice() };
  abort.abort();
  objects.length = 0;
  device?.destroy();
}

function failure(reason: unknown): void {
  if (stopped && browser.microhsCheck?.passed === false) return;
  const error = reason instanceof Error ? reason.stack ?? reason.message : String(reason);
  log(error);
  const status = document.querySelector("#status");
  if (status) status.textContent = error;
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
  const module = wasm!;
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  const size = encoded.byteLength + 1;
  enter(() => {
    const pointer = module._malloc(size) >>> 0;
    if (!pointer) throw new Error("Cannot allocate the browser response in WASM");
    try {
      module.HEAPU8.set(encoded, pointer);
      module.HEAPU8[pointer + encoded.byteLength] = 0;
      module._browserComplete(ticket, status, pointer, size);
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
  const { maxBufferSize, maxStorageBufferBindingSize, maxUniformBufferBindingSize, maxComputeWorkgroupsPerDimension } = device.limits;
  const params = new URLSearchParams(location.search);
  const showcase = demoMode;
  const model =
    params.get("model") ??
    (showcase ? "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors" : "/model.safetensors");
  return { adapter: adapterDescription, maxBufferSize, maxStorageBufferBindingSize, maxUniformBufferBindingSize, maxComputeWorkgroupsPerDimension, mode: showcase ? "demo" : "check", model };
}

async function perform(fields: BrowserRequest): Promise<unknown> {
  switch (fields[0]) {
    case "init":
      return initialize();
    case "fetch": {
      const [, key, url, start, count] = fields;
      const response = await fetch(url, { signal: abort.signal, headers: start >= 0 ? { Range: `bytes=${start}-${start + count - 1}` } : {} });
      const data = await response.arrayBuffer();
      objects[key] = data;
      return { length: data.byteLength, status: response.status, contentRange: response.headers.get("Content-Range") };
    }
    case "text":
      return new TextDecoder("utf-8", { fatal: true }).decode(objects[fields[1]] as ArrayBuffer);
    case "shader": {
      const descriptor: GPUShaderModuleDescriptor & { strictMath?: boolean } = { code: new TextDecoder("utf-8", { fatal: true }).decode(objects[fields[2]] as ArrayBuffer) };
      if (new URLSearchParams(location.search).get("strict") === "1") descriptor.strictMath = true;
      const shader = device!.createShaderModule(descriptor);
      objects[fields[1]] = shader;
      const errors = (await shader.getCompilationInfo()).messages.filter((message) => message.type === "error");
      if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
      return null;
    }
    case "pipeline":
      objects[fields[1]] = await device!.createComputePipelineAsync({ layout: objects[fields[3]] as GPUPipelineLayout, compute: { module: objects[fields[2]] as GPUShaderModule, entryPoint: fields[4] } });
      return null;
    case "readback": {
      const buffer = objects[fields[1]] as GPUBuffer;
      await buffer.mapAsync(GPUMapMode.READ);
      try {
        const data = buffer.getMappedRange().slice(0);
        objects[fields[2]] = data;
        return { length: data.byteLength };
      } finally {
        buffer.unmap();
      }
    }
    case "wait":
      await device!.queue.onSubmittedWorkDone();
      return null;
    case "scope":
      return (await device!.popErrorScope())?.message ?? null;
    default:
      throw new Error("Unknown browser API request");
  }
}

browser.microhsBrowser = {
  call(op, first, second, third, fourth, fifth, sixth): void {
    switch (op) {
      case 0:
        objects[first] = device!.createBuffer({ size: second, usage: third });
        return;
      case 1: {
        const data = objects[second] as ArrayBuffer;
        if (data.byteLength === third) {
          device!.queue.writeBuffer(objects[first] as GPUBuffer, 0, data);
        } else {
          const padded = new Uint8Array(third);
          padded.set(new Uint8Array(data));
          device!.queue.writeBuffer(objects[first] as GPUBuffer, 0, padded);
        }
        return;
      }
      case 2:
        objects[first] = device!.createCommandEncoder();
        return;
      case 3:
        objects[first] = (objects[second] as GPUCommandEncoder).beginComputePass();
        return;
      case 4:
        (objects[first] as GPUComputePassEncoder).end();
        return;
      case 5: {
        const pass = objects[first] as GPUComputePassEncoder;
        pass.setPipeline(objects[second] as GPUComputePipeline);
        pass.setBindGroup(0, objects[third] as GPUBindGroup);
        pass.dispatchWorkgroups(fourth);
        return;
      }
      case 6:
        (objects[first] as GPUCommandEncoder).copyBufferToBuffer(objects[second] as GPUBuffer, third, objects[fourth] as GPUBuffer, fifth, sixth);
        return;
      case 7:
        (objects[first] as GPUCommandEncoder).clearBuffer(objects[second] as GPUBuffer, third, fourth);
        return;
      case 8:
        device!.queue.submit([(objects[first] as GPUCommandEncoder).finish()]);
        return;
      case 9:
        objects[first] = undefined;
        return;
      case 10:
        (objects[first] as GPUBuffer).destroy();
        objects[first] = undefined;
        return;
      case 11:
        device!.pushErrorScope("validation");
        return;
      default:
        throw new Error("Unknown synchronous WebGPU API operation");
    }
  },
  array(op, key, other, heap, pointer, count): void {
    const values = new Uint32Array(heap.buffer, heap.byteOffset + (pointer >>> 0), count);
    switch (op) {
      case 0: {
        const types: GPUBufferBindingType[] = ["uniform", "storage", "read-only-storage"];
        const layout = device!.createBindGroupLayout({ entries: Array.from(values, (type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: types[type]! } })) });
        objects[key] = layout;
        objects[other] = device!.createPipelineLayout({ bindGroupLayouts: [layout] });
        return;
      }
      case 1:
        objects[key] = device!.createBindGroup({ layout: objects[other] as GPUBindGroupLayout, entries: Array.from(values, (handle, binding) => ({ binding, resource: { buffer: objects[handle] as GPUBuffer } })) });
        return;
      default:
        throw new Error("Unknown WebGPU array operation");
    }
  },
  write(buffer, heap, pointer, bytes): void {
    device!.queue.writeBuffer(objects[buffer] as GPUBuffer, 0, heap, pointer >>> 0, bytes);
  },
  request(ticket, serialized): void {
    if (stopped) throw new Error("The browser application already completed");
    queueMicrotask(() => {
      if (stopped) return;
      void Promise.resolve().then(() => perform(JSON.parse(serialized) as BrowserRequest)).then(
        (value) => complete(ticket, 1, value),
        (error: unknown) => complete(ticket, 0, error instanceof Error ? error.message : String(error)),
      ).catch(failure);
    });
  },
  number(key, offset, kind): number {
    const view = new DataView(objects[key] as ArrayBuffer);
    switch (kind) {
      case "u8": return view.getUint8(offset);
      case "u16": return view.getUint16(offset, true);
      case "u32": return view.getUint32(offset, true);
      case "i32": return view.getInt32(offset, true);
      case "f32": return view.getFloat32(offset, true);
      default: throw new Error("Unknown host number representation");
    }
  },
  copy(key, offset, length, heap, target): void {
    heap.set(new Uint8Array(objects[key] as ArrayBuffer, offset, length), target >>> 0);
  },
  log,
  clock: () => performance.now(),
  report(serialized): void {
    if (stopped) throw new Error("The browser application already completed");
    publish(JSON.parse(serialized) as BridgeReport);
  },
};

const demoParams = new URLSearchParams(location.search);
const demoMode = demoParams.get("demo") === "1" || location.pathname.endsWith("/demo.html");
const tokenBytes = new Map<number, Uint8Array>();
const tokenDecoder = new TextDecoder("utf-8", { fatal: false });

function appendTranscript(text: string): void {
  const transcript = document.querySelector("#transcript");
  if (!transcript || !text) return;
  transcript.textContent = (transcript.textContent ?? "") + text;
}

function levelToByte(codepoint: number): number {
  if (codepoint < 256) return codepoint;
  const nonPrintable = [...Array(33).keys(), ...Array.from({ length: 34 }, (_, index) => 127 + index), 173];
  return nonPrintable[codepoint - 256] ?? 0;
}

function decodeToken(id: number): string {
  if (id === 128009 || id === 128001) return tokenDecoder.decode();
  const bytes = tokenBytes.get(id);
  if (!bytes) return "";
  return tokenDecoder.decode(bytes, { stream: true });
}

async function loadDecoder(): Promise<void> {
  if (!demoMode) return;
  const response = await fetch("https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/tokenizer.json");
  if (!response.ok) throw new Error(`Tokenizer download failed: HTTP ${response.status}`);
  const tokenizer = (await response.json()) as { model?: { vocab?: Record<string, number> }; added_tokens?: { id: number; content: string }[] };
  const vocab = tokenizer.model?.vocab ?? {};
  for (const [token, id] of Object.entries(vocab)) {
    const bytes = Uint8Array.from([...token].map((character) => levelToByte(character.codePointAt(0) ?? 0)));
    tokenBytes.set(id, bytes);
  }
  for (const added of tokenizer.added_tokens ?? []) {
    tokenBytes.set(added.id, new TextEncoder().encode(added.content));
  }
}

async function start(): Promise<void> {
  if (!navigator.gpu) {
    const message = "This browser has no WebGPU. The model runs in the page only where WebGPU is available (current Chrome or Edge on Linux, Windows, or Mac). There is no CPU fallback.";
    log(message);
    const status = document.querySelector("#status");
    if (status) status.textContent = message;
    publish({ passed: false, error: message });
    return;
  }
  if (demoMode) await loadDecoder();
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
