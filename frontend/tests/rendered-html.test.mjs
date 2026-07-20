import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${path}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

function visibleText(html) {
  return html
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&rarr;/g, "→")
    .replace(/&middot;/g, "·")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

test("renders the answer with three citations and collapsed search details", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  const text = visibleText(html);
  assert.match(text, /1분기 매출 감소의 주요 원인은 무엇인가요/);
  assert.match(text, /답변에 인용된 근거 3건/);
  assert.match(text, /추가로 검색된 관련 문서 2건/);
  assert.match(html, /href="\/verify\/1"/);
  assert.match(html, /href="\/verify\/2"/);
  assert.match(html, /href="\/verify\/3"/);
  assert.match(html, /<details class="search-process">/);
  assert.doesNotMatch(html, /<details class="search-process" open/);
  assert.doesNotMatch(text, /0\.912|0\.940|similarity/i);
  assert.doesNotMatch(text, /codex-preview|Your site is taking shape/);
});

test("renders a citation-specific original document view", async () => {
  const response = await render("/verify/2");
  assert.equal(response.status, 200);

  const html = await response.text();
  const text = visibleText(html);
  assert.match(text, /원가분석_내부메모_3월\.docx/);
  assert.match(text, /판가 반영 지연 → 이익률 3\.1%p 하락/);
  assert.match(html, /id="evidence-2"/);
  assert.match(text, /주장 2의 근거/);
  assert.match(text, /✓ 원문 일치 확인/);
});

test("renders admin-only retrieval scores and filters", async () => {
  const response = await render("/admin/trace/q_20260720_0412?filter=used");
  assert.equal(response.status, 200);

  const html = await response.text();
  const text = visibleText(html);
  assert.match(text, /query_id: q_20260720_0412/);
  assert.match(text, /BM25 \+ 벡터 하이브리드 → rerank/);
  assert.match(text, /ch_1q_014_02/);
  assert.match(text, /0\.912/);
  assert.match(text, /citation verification: 3건 모두 원문 스팬 일치/);
  assert.doesNotMatch(text, /ch_1q_022_01/);
});

test("removes the disposable starter preview from product files", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(layout, /Starter Project|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
