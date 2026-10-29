const browser = globalThis;
const objects = [];
const messages = [];
const abort = new AbortController();
const resultElement = document.querySelector("#results");
let wasm;
let device;
let adapterDescription;
let activeWasm = false;
let stopped = false;
function log(message) {
    messages.push(message);
    if (resultElement)
        resultElement.textContent = messages.join("\n");
    const progress = document.querySelector("#progress");
    const loaded = /Loaded (\d+)\/(\d+)/.exec(message);
    if (progress && loaded) {
        const status = document.querySelector("#status");
        if (status)
            status.textContent = "Loading model weights…";
        const done = Number(loaded[1]);
        const total = Number(loaded[2]);
        progress.textContent = `Weights ${done}/${total}`;
        const bar = document.querySelector("#bar");
        if (bar)
            bar.style.width = `${Math.round((100 * done) / total)}%`;
    }
    if (message.startsWith("PROMPT ")) {
        const status = document.querySelector("#status");
        if (status)
            status.textContent = "Generating…";
        appendTranscript(tokenDecoder.decode());
        appendTranscript("\n\n" + message.slice("PROMPT ".length) + "\n");
    }
    if (message.startsWith("TOKEN ")) {
        const id = Number(message.slice("TOKEN ".length));
        if (Number.isInteger(id))
            appendTranscript(decodeToken(id));
    }
    console.log(message);
}
function publish(result) {
    const status = document.querySelector("#status");
    if (status && result.passed)
        status.textContent = "Generation finished.";
    stopped = true;
    browser.microhsCheck = { ...result, adapter: adapterDescription ?? {}, messages: messages.slice() };
    abort.abort();
    objects.length = 0;
    device?.destroy();
}
function failure(reason) {
    if (stopped && browser.microhsCheck?.passed === false)
        return;
    const error = reason instanceof Error ? reason.stack ?? reason.message : String(reason);
    log(error);
    const status = document.querySelector("#status");
    if (status)
        status.textContent = reason instanceof Error ? reason.message : String(reason);
    publish({ passed: false, error });
}
function enter(action) {
    if (activeWasm)
        throw new Error("Cannot reenter the active WASM evaluator");
    activeWasm = true;
    try {
        action();
    }
    catch (error) {
        failure(error);
    }
    finally {
        activeWasm = false;
    }
}
function complete(ticket, status, value) {
    if (stopped)
        return;
    const module = wasm;
    const encoded = new TextEncoder().encode(JSON.stringify(value));
    const size = encoded.byteLength + 1;
    enter(() => {
        const pointer = module._malloc(size) >>> 0;
        if (!pointer)
            throw new Error("Cannot allocate the browser response in WASM");
        try {
            module.HEAPU8.set(encoded, pointer);
            module.HEAPU8[pointer + encoded.byteLength] = 0;
            module._browserComplete(ticket, status, pointer, size);
        }
        finally {
            module._free(pointer);
        }
    });
}
async function initialize() {
    if (!device) {
        if (!navigator.gpu)
            throw new Error("WebGPU is unavailable");
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter)
            throw new Error("No WebGPU adapter");
        const info = adapter.info;
        const description = [info.vendor, info.architecture, info.device, info.description].join(" ").toLowerCase();
        if (info.isFallbackAdapter || ["swiftshader", "llvmpipe", "software"].some((name) => description.includes(name)))
            throw new Error("A physical WebGPU adapter is required.");
        adapterDescription = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
        device = await adapter.requestDevice();
        device.addEventListener("uncapturederror", (event) => failure(new Error(`WebGPU: ${event.error.message}`)));
        void device.lost.then((lost) => { if (lost.reason !== "destroyed")
            failure(new Error(`WebGPU device lost: ${lost.message}`)); });
    }
    const { maxBufferSize, maxStorageBufferBindingSize, maxUniformBufferBindingSize, maxComputeWorkgroupsPerDimension } = device.limits;
    const params = new URLSearchParams(location.search);
    const showcase = demoMode;
    const model = params.get("model") ??
        (showcase ? "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/04c3b9ad9361b824064a1f25ea60a8be9599b127/model.safetensors" : "/model.safetensors");
    return { adapter: adapterDescription, maxBufferSize, maxStorageBufferBindingSize, maxUniformBufferBindingSize, maxComputeWorkgroupsPerDimension, mode: showcase ? "demo" : "check", model };
}
async function perform(fields) {
    switch (fields[0]) {
        case "init":
            return initialize();
        case "fetch": {
            const [, key, url, start, count] = fields;
            const response = await fetch(url, { signal: abort.signal, headers: start >= 0 ? { Range: `bytes=${start}-${start + count - 1}` } : {} });
            const contentRange = response.headers.get("Content-Range");
            if (start >= 0) {
                const prefix = `bytes ${start}-${start + count - 1}/`;
                const totalText = contentRange?.startsWith(prefix) ? contentRange.slice(prefix.length) : "";
                const total = Number(totalText);
                if (response.status !== 206 || !/^\d+$/.test(totalText) || !Number.isSafeInteger(total) || total < start + count) {
                    await response.body?.cancel();
                    throw new Error("HTTP range response differs.");
                }
            }
            else if (response.status !== 200) {
                await response.body?.cancel();
                throw new Error(`Download failed: HTTP ${response.status}`);
            }
            const data = await response.arrayBuffer();
            objects[key] = data;
            return { length: data.byteLength, status: response.status, contentRange };
        }
        case "text":
            return new TextDecoder("utf-8", { fatal: true }).decode(objects[fields[1]]);
        case "shader": {
            const descriptor = { code: new TextDecoder("utf-8", { fatal: true }).decode(objects[fields[2]]) };
            if (new URLSearchParams(location.search).get("strict") === "1")
                descriptor.strictMath = true;
            const shader = device.createShaderModule(descriptor);
            objects[fields[1]] = shader;
            const errors = (await shader.getCompilationInfo()).messages.filter((message) => message.type === "error");
            if (errors.length)
                throw new Error(errors.map((message) => message.message).join("\n"));
            return null;
        }
        case "pipeline":
            objects[fields[1]] = await device.createComputePipelineAsync({ layout: objects[fields[3]], compute: { module: objects[fields[2]], entryPoint: fields[4] } });
            return null;
        case "readback": {
            const buffer = objects[fields[1]];
            await buffer.mapAsync(GPUMapMode.READ);
            try {
                const data = buffer.getMappedRange().slice(0);
                objects[fields[2]] = data;
                return { length: data.byteLength };
            }
            finally {
                buffer.unmap();
            }
        }
        case "wait":
            await device.queue.onSubmittedWorkDone();
            return null;
        case "scope":
            return (await device.popErrorScope())?.message ?? null;
        default:
            throw new Error("Unknown browser API request");
    }
}
browser.microhsBrowser = {
    call(op, first, second, third, fourth, fifth, sixth) {
        switch (op) {
            case 0:
                objects[first] = device.createBuffer({ size: second, usage: third });
                return;
            case 1: {
                const data = objects[second];
                if (data.byteLength === third) {
                    device.queue.writeBuffer(objects[first], 0, data);
                }
                else {
                    const padded = new Uint8Array(third);
                    padded.set(new Uint8Array(data));
                    device.queue.writeBuffer(objects[first], 0, padded);
                }
                return;
            }
            case 2:
                objects[first] = device.createCommandEncoder();
                return;
            case 3:
                objects[first] = objects[second].beginComputePass();
                return;
            case 4:
                objects[first].end();
                return;
            case 5: {
                const pass = objects[first];
                pass.setPipeline(objects[second]);
                pass.setBindGroup(0, objects[third]);
                pass.dispatchWorkgroups(fourth);
                return;
            }
            case 6:
                objects[first].copyBufferToBuffer(objects[second], third, objects[fourth], fifth, sixth);
                return;
            case 7:
                objects[first].clearBuffer(objects[second], third, fourth);
                return;
            case 8:
                device.queue.submit([objects[first].finish()]);
                return;
            case 9:
                objects[first] = undefined;
                return;
            case 10:
                objects[first].destroy();
                objects[first] = undefined;
                return;
            case 11:
                device.pushErrorScope("validation");
                return;
            default:
                throw new Error("Unknown synchronous WebGPU API operation");
        }
    },
    array(op, key, other, heap, pointer, count) {
        const values = new Uint32Array(heap.buffer, heap.byteOffset + (pointer >>> 0), count);
        switch (op) {
            case 0: {
                const types = ["uniform", "storage", "read-only-storage"];
                const layout = device.createBindGroupLayout({ entries: Array.from(values, (type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: types[type] } })) });
                objects[key] = layout;
                objects[other] = device.createPipelineLayout({ bindGroupLayouts: [layout] });
                return;
            }
            case 1:
                objects[key] = device.createBindGroup({ layout: objects[other], entries: Array.from(values, (handle, binding) => ({ binding, resource: { buffer: objects[handle] } })) });
                return;
            default:
                throw new Error("Unknown WebGPU array operation");
        }
    },
    write(buffer, heap, pointer, bytes) {
        device.queue.writeBuffer(objects[buffer], 0, heap, pointer >>> 0, bytes);
    },
    request(ticket, serialized) {
        if (stopped)
            throw new Error("The browser application already completed");
        queueMicrotask(() => {
            if (stopped)
                return;
            void Promise.resolve().then(() => perform(JSON.parse(serialized))).then((value) => complete(ticket, 1, value), (error) => complete(ticket, 0, error instanceof Error ? error.message : String(error))).catch(failure);
        });
    },
    number(key, offset, kind) {
        const view = new DataView(objects[key]);
        switch (kind) {
            case "u8": return view.getUint8(offset);
            case "u16": return view.getUint16(offset, true);
            case "u32": return view.getUint32(offset, true);
            case "i32": return view.getInt32(offset, true);
            case "f32": return view.getFloat32(offset, true);
            default: throw new Error("Unknown host number representation");
        }
    },
    copy(key, offset, length, heap, target) {
        heap.set(new Uint8Array(objects[key], offset, length), target >>> 0);
    },
    log,
    clock: () => performance.now(),
    report(serialized) {
        if (stopped)
            throw new Error("The browser application already completed");
        publish(JSON.parse(serialized));
    },
};
const demoParams = new URLSearchParams(location.search);
const demoMode = demoParams.get("demo") === "1" || location.pathname.endsWith("/demo.html");
const tokenBytes = new Map();
const tokenDecoder = new TextDecoder("utf-8", { fatal: false });
function appendTranscript(text) {
    const transcript = document.querySelector("#transcript");
    if (!transcript || !text)
        return;
    transcript.textContent = (transcript.textContent ?? "") + text;
}
function levelToByte(codepoint) {
    if (codepoint < 256)
        return codepoint;
    const nonPrintable = [...Array(33).keys(), ...Array.from({ length: 34 }, (_, index) => 127 + index), 173];
    return nonPrintable[codepoint - 256] ?? 0;
}
function decodeToken(id) {
    if (id === 128009 || id === 128001)
        return tokenDecoder.decode();
    const bytes = tokenBytes.get(id);
    if (!bytes)
        return "";
    return tokenDecoder.decode(bytes, { stream: true });
}
async function loadDecoder() {
    if (!demoMode)
        return;
    const response = await fetch("https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/04c3b9ad9361b824064a1f25ea60a8be9599b127/tokenizer.json", { signal: abort.signal });
    if (!response.ok)
        throw new Error(`Tokenizer download failed: HTTP ${response.status}`);
    const tokenizer = (await response.json());
    const vocab = tokenizer.model?.vocab ?? {};
    for (const [token, id] of Object.entries(vocab)) {
        const bytes = Uint8Array.from([...token].map((character) => levelToByte(character.codePointAt(0) ?? 0)));
        tokenBytes.set(id, bytes);
    }
    for (const added of tokenizer.added_tokens ?? []) {
        tokenBytes.set(added.id, new TextEncoder().encode(added.content));
    }
}
async function start() {
    if (!navigator.gpu) {
        const message = "WebGPU is unavailable in this browser.";
        log(message);
        const status = document.querySelector("#status");
        if (status)
            status.textContent = message;
        publish({ passed: false, error: message });
        return;
    }
    if (demoMode) {
        await initialize();
        await loadDecoder();
    }
    const modulePath = "./microhs.mjs";
    const createMicroHs = (await import(modulePath)).default;
    wasm = await createMicroHs({ print: log, printErr: log });
    for (const name of ["_malloc", "_free", "_mhs_init", "_gpuMain", "_browserComplete"])
        if (typeof wasm[name] !== "function")
            throw new Error(`Missing WASM bridge export ${name}`);
    if (!(wasm.HEAPU8 instanceof Uint8Array))
        throw new Error("Missing WASM byte heap");
    const module = wasm;
    enter(() => { module._mhs_init(); module._gpuMain(); });
}
if (demoMode) {
    const button = document.querySelector("#start-demo");
    if (!button)
        failure(new Error("The demo start button is missing."));
    else
        button.addEventListener("click", () => {
            button.disabled = true;
            const status = document.querySelector("#status");
            if (status)
                status.textContent = "Checking GPU…";
            void start().catch(failure);
        }, { once: true });
}
else
    void start().catch(failure);
export {};
