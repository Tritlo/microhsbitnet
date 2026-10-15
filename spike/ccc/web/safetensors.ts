export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface ModelFile {
  tensors: Map<string, TensorInfo>;
  read(name: string, firstByte?: number, byteLength?: number): Promise<ArrayBuffer>;
}

export async function openModel(url: string): Promise<ModelFile> {
  let totalBytes = 0;
  async function range(start: number, count: number): Promise<ArrayBuffer> {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0) throw new Error("Invalid tensor byte range");
    const end = start + count - 1;
    const response = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("Content-Range") ?? "");
    if (response.status !== 206 || !match || Number(match[1]) !== start || Number(match[2]) !== end) throw new Error("The model server must support exact byte ranges");
    const length = Number(match[3]);
    if (!Number.isSafeInteger(length) || length <= end || (totalBytes !== 0 && length !== totalBytes)) throw new Error("Model length changed or is invalid");
    totalBytes = length;
    const data = await response.arrayBuffer();
    if (data.byteLength !== count) throw new Error("Truncated model range");
    return data;
  }
  const prefix = new DataView(await range(0, 8));
  const headerLength = Number(prefix.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength <= 0 || headerLength > 16 * 1024 * 1024 || headerLength + 8 > totalBytes) throw new Error("Invalid safetensors header length");
  const dataStart = 8 + headerLength;
  const header: unknown = JSON.parse(new TextDecoder().decode(await range(8, headerLength)));
  if (typeof header !== "object" || header === null || Array.isArray(header)) throw new Error("Invalid tensor table");
  const tensors = new Map<string, TensorInfo>();
  const widths: Record<string, number> = { BF16: 2, F32: 4, U8: 1, I8: 1 };
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (typeof value !== "object" || value === null) throw new Error(`Invalid tensor ${name}`);
    const { dtype, shape, data_offsets: offsets } = value as Partial<TensorInfo>;
    if (typeof dtype !== "string" || !Array.isArray(shape) || !shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) || !Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(Number.isSafeInteger)) throw new Error(`Invalid tensor metadata ${name}`);
    const [start, end] = offsets;
    if (start! < 0 || end! <= start! || end! > totalBytes - dataStart) throw new Error(`Tensor range outside model ${name}`);
    const width = widths[dtype];
    if (!width || shape.reduce((elements, dimension) => elements * dimension, 1) * width !== end! - start!) throw new Error(`Unsupported tensor type or size ${name}`);
    tensors.set(name, { dtype, shape, data_offsets: [start!, end!] });
  }
  return {
    tensors,
    async read(name, firstByte = 0, byteLength) {
      const tensor = tensors.get(name);
      if (!tensor) throw new Error(`Missing tensor ${name}`);
      const [start, end] = tensor.data_offsets;
      const count = byteLength ?? end - start - firstByte;
      if (!Number.isSafeInteger(firstByte) || firstByte < 0 || count <= 0 || firstByte + count > end - start) throw new Error(`Invalid tensor slice ${name}`);
      return range(dataStart + start + firstByte, count);
    },
  };
}

export function decodeBf16(data: ArrayBuffer): Float32Array<ArrayBuffer> {
  if (data.byteLength % 2) throw new Error("Invalid BF16 byte count");
  const input = new DataView(data);
  const output = new ArrayBuffer(data.byteLength * 2);
  const bits = new DataView(output);
  for (let index = 0; index < data.byteLength / 2; index++) bits.setUint32(index * 4, input.getUint16(index * 2, true) << 16, true);
  const values = new Float32Array(output);
  if (!values.every(Number.isFinite)) throw new Error("Nonfinite BF16 model tensor");
  return values;
}
