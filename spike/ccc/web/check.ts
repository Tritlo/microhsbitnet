import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";

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
  browser = await chromium.launch({
    executablePath: process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
    args: [...(process.platform === "darwin" ? ["--enable-unsafe-webgpu", "--use-angle=metal"] : ["--enable-unsafe-webgpu"]), ...(process.env.WEBGPU_STRICT === "1" ? ["--enable-webgpu-developer-features"] : [])],
  });
  const page = await browser.newPage();
  const pageErrors: Error[] = [];
  page.on("console", (message) => console.log(message.text()));
  page.on("pageerror", (error) => { pageErrors.push(error); console.error(error); });
  const query = new URLSearchParams();
  if (process.env.WEBGPU_STRICT === "1") query.set("strict", "1");
  await page.goto(`http://127.0.0.1:${address.port}/web/index.html?${query}`);
  await page.waitForFunction(() => globalThis.microhsCheck !== undefined, undefined, { timeout: 600_000 });
  const result = await page.evaluate(() => globalThis.microhsCheck);
  await writeFile(resolve(root, "web-result.json"), JSON.stringify({ browserVersion: browser.version(), strictMathDevelopmentProfile: process.env.WEBGPU_STRICT === "1", ...result }, null, 2) + "\n");
  if (!result?.passed) throw new Error(result?.error ?? "Browser check failed");
  if (pageErrors.length) throw pageErrors[0];
} finally {
  await browser?.close();
  await new Promise<void>((done, reject) => server.close((error) => error ? reject(error) : done()));
}
