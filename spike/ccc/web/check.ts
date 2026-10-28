import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";

function chromeExecutable(): string {
  if (process.env.CHROME) return process.env.CHROME;
  if (process.platform === "darwin") return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  if (process.platform === "win32") return `${process.env.PROGRAMFILES ?? "C:\\\\Program Files"}\\Google\\Chrome\\Application\\chrome.exe`;
  for (const candidate of ["google-chrome", "google-chrome-stable", "microsoft-edge", "chromium", "chromium-browser"]) {
    try {
      return execFileSync("which", [candidate], { encoding: "utf8" }).trim();
    } catch {
      /* try the next installed browser */
    }
  }
  return "google-chrome";
}

const root = resolve(fileURLToPath(new URL("../../../../bin/ccc-spike/", import.meta.url)));
const mime: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm" };
const server = createServer((request, response) => {
  let requested: string;
  try {
    requested = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
  } catch {
    response.writeHead(400).end();
    return;
  }
  if (requested === "/model.safetensors") {
    const model = resolve(root, "../../models/bitnet-b1.58-2B-4T/model.safetensors");
    void stat(model).then(({ size }) => {
      const range = /^bytes=(\d+)-(\d+)$/.exec(request.headers.range ?? "");
      const start = Number(range?.[1]);
      const end = Number(range?.[2]);
      if (!range || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end >= size) {
        response.writeHead(416, { "Content-Range": `bytes */${size}` }).end();
        return;
      }
      response.writeHead(206, { "Content-Type": "application/octet-stream", "Content-Length": end - start + 1, "Content-Range": `bytes ${start}-${end}/${size}`, "Accept-Ranges": "bytes" });
      const stream = createReadStream(model, { start, end });
      stream.on("error", (error) => response.destroy(error));
      stream.pipe(response);
    }).catch(() => response.writeHead(404).end());
    return;
  }
  const path = resolve(root, `.${requested}`);
  if (!path.startsWith(`${root}${sep}`)) { response.writeHead(403).end(); return; }
  void readFile(path).then((body) => {
    response.writeHead(200, { "Content-Type": mime[extname(path)] ?? "application/octet-stream" });
    response.end(body);
  }).catch(() => response.writeHead(404).end());
});
await new Promise<void>((ready) => server.listen(0, "127.0.0.1", ready));
const address = server.address();
if (!address || typeof address === "string") throw new Error("Server did not start");
let browser: Browser | undefined;
try {
  const gpuArgs = ["--enable-unsafe-webgpu"];
  if (process.platform === "darwin") gpuArgs.push("--use-angle=metal");
  if (process.env.WEBGPU_STRICT === "1") gpuArgs.push("--enable-webgpu-developer-features");
  browser = await chromium.launch({
    executablePath: chromeExecutable(),
    headless: process.env.HEADED !== "1",
    args: gpuArgs,
  });
  const page = await browser.newPage();
  const pageErrors: Error[] = [];
  page.on("console", (message) => console.log(message.text()));
  page.on("pageerror", (error) => { pageErrors.push(error); console.error(error); });
  const query = new URLSearchParams();
  if (process.env.WEBGPU_STRICT === "1") query.set("strict", "1");
  const pageName = process.env.DEMO === "1" ? "demo.html" : "index.html";
  if (process.env.DEMO === "1") query.set("demo", "1");
  await page.goto(`http://127.0.0.1:${address.port}/web/${pageName}?${query}`);
  await page.waitForFunction(() => (globalThis as unknown as { microhsCheck?: unknown }).microhsCheck !== undefined, undefined, { timeout: 600_000 });
  const result = await page.evaluate(() => (globalThis as unknown as { microhsCheck?: { passed: boolean; error?: string; [key: string]: unknown } }).microhsCheck);
  const resultFile = process.env.DEMO === "1" ? "demo-result.json" : "web-result.json";
  await writeFile(resolve(root, resultFile), JSON.stringify({ browserVersion: browser.version(), strictMathDevelopmentProfile: process.env.WEBGPU_STRICT === "1", ...result }, null, 2) + "\n");
  if (!result?.passed) throw new Error(result?.error ?? "Browser check failed");
  if (pageErrors.length) throw pageErrors[0];
} finally {
  await browser?.close();
  await new Promise<void>((done, reject) => server.close((error) => error ? reject(error) : done()));
}
