export function createShader(device: GPUDevice, code: string): GPUShaderModule {
  const descriptor: GPUShaderModuleDescriptor & { strictMath?: boolean } = { code };
  if (new URLSearchParams(location.search).get("strict") === "1") descriptor.strictMath = true;
  return device.createShaderModule(descriptor);
}
