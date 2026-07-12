import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://memory.example/", { headers: { accept: "text/html", host: "memory.example" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Megatron memory estimator", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Megatron SFT 显存估算器<\/title>/i);
  assert.match(html, /看清每一GB/);
  assert.match(html, /Qwen3_5Moe/);
  assert.match(html, /64<!-- --> × <!-- -->80<!-- --> GB/);
  assert.match(html, /模型权重/);
  assert.match(html, /梯度/);
  assert.match(html, /优化器状态/);
  assert.match(html, /激活值/);
  assert.match(html, /PP Rank 显存细分/);
  assert.match(html, /关键网络维度/);
  assert.match(html, /Peak PP/);
  assert.match(html, /并行策略作用表/);
  assert.match(html, /Expert Tensor Parallel/);
  assert.match(html, /Sequence Parallel/);
  assert.match(html, /激活值分层近似/);
  assert.match(html, /PP Rank 汇总/);
  assert.match(html, /可切分激活/);
  assert.match(html, /不可切分激活/);
  assert.match(html, /RMSNorm/);
  assert.match(html, /Router logits/);
  assert.match(html, /All-to-All buffer/);
  assert.doesNotMatch(html, /配置参数如何影响显存公式/);
  assert.doesNotMatch(html, /<h3>优化建议<\/h3>/);
  assert.match(html, /粗略估算公式/);
  assert.match(html, /不同配置下的快速对比/);
  assert.match(html, /分布式 Adam/);
  assert.match(html, /常规 Adam/);
  assert.match(html, /https:\/\/memory\.example\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("keeps the checked-in model configuration wired to the page", async () => {
  const [page, estimator, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/Estimator.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /from "\.\.\/config\.json"/);
  assert.match(estimator, /shapeFromConfig/);
  assert.match(estimator, /distributedOptimizer/);
  assert.match(estimator, /recompute/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
