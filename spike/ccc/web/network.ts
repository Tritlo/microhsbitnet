import { createLayerEngine, finiteFloats, fixturePath, type WeightDescription } from "./layer.js";
import { decodeBf16, type ModelFile } from "./safetensors.js";

interface NetworkStep {
  position: number;
  token: number;
  hidden: string;
  normed: string;
  argmax?: number;
  logit?: number;
  layers?: string[];
}

export const networkDiagnostics: {
  name: string;
  maximum: number;
  rms: number;
  relativeL2: number;
  firstMismatch?: { index: number; actual: number; reference: number };
}[] = [];

interface NetworkManifest {
  prompt: number[];
  generated: number[];
  steps: NetworkStep[];
}

interface NetworkCaseResult {
  name: string;
  promptTokens: number;
  generatedTokens: number[];
  referenceTokens: number[];
  matchingTokens: number;
  positionsRun: number;
  normalizedVectorsChecked: number;
  maxNormedRelativeL2: number;
  driverMs: number;
}

export interface NetworkResult {
  output: Int32Array<ArrayBuffer>;
  count: number;
  matchingTokens: number;
  tokenReferenceMatched: boolean;
  positionsRun: number;
  uploadMs: number;
  driverMs: number;
  prefillMs: number;
  generationMs: number;
  softmaxRoundtrips: number;
  maxHiddenAbsoluteError: number;
  maxNormedAbsoluteError: number;
  maxNormedRelativeL2: number;
  normalizedVectorsChecked: number;
  normalizedVectorsWithinOnePercent: number;
  numericalBoundPassed: boolean;
  strictDiagnosticFailures: number;
  maxLogitAbsoluteError: number;
  cases: NetworkCaseResult[];
}

export async function createNetworkRunner(
  device: GPUDevice,
  bytes: (path: string) => Promise<ArrayBuffer>,
  wasm: { _networkStep(position: number, token: number, withHead: number): number },
  model: ModelFile,
  log: (message: string) => void,
): Promise<() => Promise<NetworkResult>> {
  const referenceDirectory = "network-cases";
  log("Independent C reference: token and vector comparisons are diagnostics; generation uses GPU-selected tokens");
  const suite = JSON.parse(new TextDecoder().decode(await bytes(`../${referenceDirectory}/suite.json`))) as { name: string; promptText: string; manifest: string }[];
  if (!Array.isArray(suite) || suite.length !== 6 || new Set(suite.map((entry) => entry.name)).size !== suite.length) throw new Error("Expected six distinct network cases");
  const validToken = (token: number): boolean => Number.isSafeInteger(token) && token >= 0 && token < 128256;
  const cases = await Promise.all(suite.map(async (entry) => {
    if (!entry || !/^[a-z][a-z0-9-]*$/.test(entry.name) || typeof entry.promptText !== "string") throw new Error("Invalid network suite entry");
    const manifest = JSON.parse(new TextDecoder().decode(await bytes(fixturePath(entry.manifest, referenceDirectory)))) as NetworkManifest;
    const folder = entry.manifest.split("/").slice(0, -1).join("/");
    const casePath = (file: string): string => fixturePath(folder ? `${folder}/${file}` : file, referenceDirectory);
    if (!manifest || !Array.isArray(manifest.prompt) || manifest.prompt.length === 0 || !manifest.prompt.every(validToken) || !Array.isArray(manifest.generated) || manifest.generated.length < 1 || manifest.generated.length > 10 || !manifest.generated.every(validToken) || !Array.isArray(manifest.steps) || manifest.steps.length > 2048 || manifest.steps.length < manifest.prompt.length + manifest.generated.length - 1) throw new Error(`Invalid network reference manifest ${entry.name}`);
    const references = await Promise.all(manifest.steps.map(async (step, position) => {
      if (!step || step.position !== position || !validToken(step.token) || (step.argmax !== undefined && !validToken(step.argmax)) || (step.logit !== undefined && !Number.isFinite(step.logit))) throw new Error("Invalid network reference step");
      const [hidden, normed] = await Promise.all([bytes(casePath(step.hidden)), bytes(casePath(step.normed))]);
      const layerValues = await Promise.all((step.layers ?? []).map(async (file) => finiteFloats(await bytes(casePath(file)), 2560, file)));
      return { ...step, hiddenValues: finiteFloats(hidden, 2560, step.hidden), normedValues: finiteFloats(normed, 2560, step.normed), layerValues };
    }));
    return { ...entry, manifest, references };
  }));
  const engine = await createLayerEngine(device, bytes, 30);
  const uploadStart = performance.now();
  try {
    let uploadedWeights = 0;
    for (const [name, tensor] of model.tensors) {
      if (!name.startsWith("model.layers.") || name.endsWith("_scale")) continue;
      let weight: WeightDescription;
      let data = await model.read(name);
      if (tensor.dtype === "U8" && tensor.shape.length === 2) {
        const scaleName = name + "_scale";
        const scale = model.tensors.get(scaleName);
        if (scale?.dtype !== "BF16" || scale.shape.reduce((elements, dimension) => elements * dimension, 1) !== 1) throw new Error(`Invalid model scale ${scaleName}`);
        const value = decodeBf16(await model.read(scaleName))[0]!;
        if (value <= 0) throw new Error(`Nonpositive model scale ${scaleName}`);
        weight = { name, kind: "linear", file: name, columns: tensor.shape[1]!, rows: tensor.shape[0]! * 4, scale: value };
      } else if (tensor.dtype === "BF16" && tensor.shape.length === 1) {
        data = decodeBf16(data).buffer;
        weight = { name, kind: "norm", file: name, columns: tensor.shape[0]!, rows: 1, scale: 1 };
      } else throw new Error(`Unsupported model weight ${name}`);
      engine.uploadWeight(weight, data);
      uploadedWeights++;
      if (uploadedWeights % 55 === 0) log(`Loaded ${uploadedWeights}/330 decoder weights directly into WebGPU`);
    }
    if (uploadedWeights !== 330) throw new Error("Expected 330 decoder weights");
    const normName = "model.norm.weight";
    const norm = model.tensors.get(normName);
    if (norm?.dtype !== "BF16" || norm.shape.join(",") !== "2560") throw new Error("Invalid final model norm");
    engine.uploadWeight({ name: normName, kind: "norm", file: normName, columns: 2560, rows: 1, scale: 1 }, decodeBf16(await model.read(normName)).buffer);
    const embedding = model.tensors.get("model.embed_tokens.weight");
    if (embedding?.dtype !== "BF16" || embedding.shape.join(",") !== "128256,2560") throw new Error("Invalid model embedding");
    for (let firstRow = 0; firstRow < 128256; firstRow += 16384) {
      const rows = Math.min(16384, 128256 - firstRow);
      engine.uploadEmbedding(firstRow, await model.read("model.embed_tokens.weight", firstRow * 2560 * 2, rows * 2560 * 2));
    }
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    engine.dispose();
    throw error;
  }
  const uploadMs = performance.now() - uploadStart;
  log(`Full model range fetch, format conversion, and GPU upload: ${uploadMs.toFixed(1)} ms`);
  let used = false;
  return async () => {
    if (used) throw new Error("The network runner can execute only once");
    used = true;
    let maxHiddenAbsoluteError = 0;
    let maxNormedAbsoluteError = 0;
    let maxNormedRelativeL2 = 0;
    let normalizedVectorsChecked = 0;
    let normalizedVectorsWithinOnePercent = 0;
    let maxLogitAbsoluteError = 0;
    let prefillMs = 0;
    let generationMs = 0;
    let softmaxRoundtrips = 0;
    let positionsRun = 0;
    let selected: number | undefined;
    const generated: number[] = [];
    const caseResults: NetworkCaseResult[] = [];
    const diagnostics: string[] = [];
    const started = performance.now();
    try {
      for (const testCase of cases) {
        engine.reset();
        const { manifest, references, name } = testCase;
        const caseStarted = performance.now();
        const caseGenerated: number[] = [];
        let caseMaximum = 0;
        let casePrefillMs = 0;
        let casePositions = 0;
        let caseVectorsChecked = 0;
        let referenceInputsMatch = true;
        selected = undefined;
        log(`Network case ${name}: ${manifest.prompt.length} prompt tokens; up to 10 generated tokens; caches reset`);
        const steps = manifest.prompt.length + 10 - 1;
        for (let position = 0; position < steps; position++) {
          const token = position < manifest.prompt.length ? manifest.prompt[position]! : selected;
          const reference = references[position];
          if (token === undefined) throw new Error(`Missing network input at ${position}`);
          referenceInputsMatch = referenceInputsMatch && reference?.token === token;
          const withHead = position >= manifest.prompt.length - 1 ? 1 : 0;
          const result = await engine.execute(position, () => wasm._networkStep(position, token, withHead));
          positionsRun++;
          casePositions++;
          softmaxRoundtrips += result.softmaxRoundtrips;
          if (!result.normed) throw new Error(`Missing final normalized vector at ${position}`);
          let normalizedError = "not comparable after token divergence";
          if (referenceInputsMatch && reference) {
            if (result.layers && reference.layerValues.length) {
              if (result.layers.length !== reference.layerValues.length) throw new Error("Layer checkpoint count differs");
              result.layers.forEach((values, layer) => compare(values, reference.layerValues[layer]!, `${name}/position ${position} layer ${layer}`, diagnostics));
            }
            maxHiddenAbsoluteError = Math.max(maxHiddenAbsoluteError, compare(result.output, reference.hiddenValues, `${name}/hidden ${position}`, diagnostics).maximum);
            const normed = compare(result.normed, reference.normedValues, `${name}/normed ${position}`, diagnostics);
            maxNormedAbsoluteError = Math.max(maxNormedAbsoluteError, normed.maximum);
            maxNormedRelativeL2 = Math.max(maxNormedRelativeL2, normed.relativeL2);
            caseMaximum = Math.max(caseMaximum, normed.relativeL2);
            normalizedVectorsChecked++;
            caseVectorsChecked++;
            if (normed.relativeL2 <= 0.01) normalizedVectorsWithinOnePercent++;
            normalizedError = `${(normed.relativeL2 * 100).toFixed(4)}%`;
          }
          if (withHead) {
            if (!result.normed || result.token === undefined || result.logit === undefined || !Number.isFinite(result.logit)) throw new Error("Missing GPU head output");
            const expectedToken = manifest.generated[caseGenerated.length];
            if (expectedToken !== undefined && (!reference || reference.argmax !== expectedToken || reference.logit === undefined)) throw new Error(`Inconsistent C token reference at ${position}`);
            if (referenceInputsMatch && reference?.logit !== undefined && result.token === expectedToken) {
              const error = Math.abs(result.logit - reference.logit);
              maxLogitAbsoluteError = Math.max(maxLogitAbsoluteError, error);
              if (error > 2e-5 * Math.max(1, Math.abs(reference.logit))) diagnostics.push(`Selected logit error ${error} exceeds bound at ${position}`);
            }
            selected = result.token;
            generated.push(selected);
            caseGenerated.push(selected);
            const tokenComparison = expectedToken === undefined ? "has no C reference" : `${selected === expectedToken ? "matches" : "differs from"} C ${expectedToken}`;
            log(`${name} position ${position}: GPU token ${selected} ${tokenComparison}; normalized relative L2 ${normalizedError}; step ${result.dispatchReadbackMs.toFixed(3)} ms`);
          } else if (position % 8 === 0) log(`${name} prefill ${position + 1}/${manifest.prompt.length}: hidden computed; step ${result.dispatchReadbackMs.toFixed(3)} ms`);
          if (position === manifest.prompt.length - 1) casePrefillMs = performance.now() - caseStarted;
          if (withHead && (selected === 128009 || selected === 128001)) break;
        }
        const caseMs = performance.now() - caseStarted;
        prefillMs += casePrefillMs;
        generationMs += caseMs - casePrefillMs;
        const matchingTokens = caseGenerated.filter((token, index) => token === manifest.generated[index]).length;
        caseResults.push({ name, promptTokens: manifest.prompt.length, generatedTokens: caseGenerated, referenceTokens: manifest.generated, matchingTokens, positionsRun: casePositions, normalizedVectorsChecked: caseVectorsChecked, maxNormedRelativeL2: caseMaximum, driverMs: caseMs });
        log(`${name}: ${caseVectorsChecked}/${casePositions} normalized vectors comparable; max relative L2 ${(caseMaximum * 100).toFixed(4)}%; ${matchingTokens}/${caseGenerated.length} GPU tokens match C; ${caseMs.toFixed(1)} ms`);
      }
      const driverMs = performance.now() - started;
      log(`Full-model checks: ${normalizedVectorsChecked} normalized vectors; maximum relative L2 ${(maxNormedRelativeL2 * 100).toFixed(4)}%; ${diagnostics.length} stricter diagnostic differences`);
      const matchingTokens = caseResults.reduce((total, result) => total + result.matchingTokens, 0);
      const tokenReferenceMatched = cases.every((testCase, index) => {
        const result = caseResults[index]!;
        return result.matchingTokens === testCase.manifest.generated.length && result.generatedTokens.length === testCase.manifest.generated.length;
      });
      const numericalBoundPassed = normalizedVectorsWithinOnePercent === positionsRun;
      log(`1% normalized-vector diagnostic: ${normalizedVectorsWithinOnePercent}/${normalizedVectorsChecked} within bound; ${numericalBoundPassed ? "met" : "not met"}`);
      return { output: new Int32Array(generated), count: generated.length, matchingTokens, tokenReferenceMatched, positionsRun, uploadMs, driverMs, prefillMs, generationMs, softmaxRoundtrips, maxHiddenAbsoluteError, maxNormedAbsoluteError, maxNormedRelativeL2, normalizedVectorsChecked, normalizedVectorsWithinOnePercent, numericalBoundPassed, strictDiagnosticFailures: diagnostics.length, maxLogitAbsoluteError, cases: caseResults };
    } finally {
      engine.dispose();
    }
  };
}

function compare(actual: Float32Array<ArrayBuffer>, reference: Float32Array<ArrayBuffer>, name: string, diagnostics: string[]): typeof networkDiagnostics[number] {
  if (actual.length !== reference.length) throw new Error(`Network ${name} length differs`);
  let maximum = 0;
  let mismatch = -1;
  let squaredError = 0;
  let squaredReference = 0;
  for (let index = 0; index < actual.length; index++) {
    const error = Math.abs(actual[index]! - reference[index]!);
    if (!Number.isFinite(error)) throw new Error(`Nonfinite network ${name}[${index}]`);
    maximum = Math.max(maximum, error);
    squaredError += error * error;
    squaredReference += reference[index]! * reference[index]!;
    if (mismatch < 0 && error > 2e-5 * Math.max(1, Math.abs(reference[index]!))) mismatch = index;
  }
  const relativeL2 = squaredReference === 0 ? (squaredError === 0 ? 0 : Infinity) : Math.sqrt(squaredError / squaredReference);
  const result = {
    name, maximum, rms: Math.sqrt(squaredError / actual.length), relativeL2,
    firstMismatch: mismatch < 0 ? undefined : { index: mismatch, actual: actual[mismatch]!, reference: reference[mismatch]! },
  };
  const summary = `Network ${name}: max absolute error ${maximum}; RMS error ${result.rms}; relative L2 error ${relativeL2}`;
  networkDiagnostics.push(result);
  if (mismatch >= 0) diagnostics.push(`${summary}; first mismatch[${mismatch}] GPU ${actual[mismatch]} != C ${reference[mismatch]}; tolerance 2e-5 * max(1, abs(reference))`);
  return result;
}
