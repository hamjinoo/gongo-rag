"""한 번의 RAG 실행을 답변과 네 단계 Top-k 중심으로 보여준다."""

from __future__ import annotations

import html
import re
from typing import Iterable

import streamlit as st

from rag_answer import verify_citation
from rag_workflow import RAGEvidence, RAGResponse


CITATION_PATTERN = re.compile(r"\[근거\s+(\d+)\]")
CHUNK_COLORS = (
    "#3d6fd1",
    "#2e9e5b",
    "#b7791f",
    "#7c5db5",
    "#c05670",
    "#3f8791",
    "#68758c",
)


APP_STYLE = """
<style>
:root {
  --trace-ink: #172033;
  --trace-muted: #748095;
  --trace-line: #e4e8ef;
  --trace-blue: #3d6fd1;
  --trace-blue-soft: #eef3fc;
  --trace-green: #2e9e5b;
  --trace-warm: #fbfaf7;
}
[data-testid="stAppViewContainer"] { background: #fff; color: var(--trace-ink); }
[data-testid="stHeader"] { background: rgba(255,255,255,.92); }
.block-container { max-width: 1480px; padding-top: 1.2rem; }
.trace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .45rem 0 .9rem;
  border-bottom: 1px solid var(--trace-line);
  margin-bottom: .9rem;
}
.trace-brand { color: var(--trace-ink); font-size: 1.45rem; font-weight: 800; }
.trace-brand span { color: var(--trace-blue); }
.trace-context { color: var(--trace-muted); font-size: .88rem; }
.trace-run-summary {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin: .7rem 0 1.15rem;
}
.trace-summary-chip {
  color: #4d5a70;
  background: #f7f9fc;
  border: 1px solid var(--trace-line);
  border-radius: 999px;
  padding: .3rem .7rem;
  font-size: .78rem;
  font-weight: 700;
}
.trace-answer-box {
  border: 1px solid #d8e2f3;
  border-radius: .9rem;
  background: #f8faff;
  padding: 1.15rem 1.25rem;
  margin: 0;
  min-height: 100%;
}
.trace-answer-label { color: var(--trace-blue); font-size: .75rem; font-weight: 800; }
.trace-answer { color: #273248; font-size: 1.15rem; line-height: 1.8; margin-top: .55rem; font-weight: 650; }
.trace-result-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr);
  gap: .85rem;
  margin: .65rem 0 1.1rem;
}
.trace-evidence-panel {
  border: 1px solid var(--trace-line);
  border-radius: .9rem;
  background: #fff;
  padding: 1rem 1.1rem;
}
.trace-evidence-label { color: #3e4a60; font-size: .75rem; font-weight: 850; margin-bottom: .5rem; }
.trace-evidence-card { border-left: 3px solid var(--trace-blue); padding: .1rem 0 .1rem .75rem; margin: .55rem 0; }
.trace-evidence-source { color: #344056; font-size: .82rem; font-weight: 800; }
.trace-evidence-quote { color: #667287; font-size: .78rem; line-height: 1.55; margin-top: .3rem; }
.trace-check-row { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .8rem; }
.trace-check {
  color: #227849;
  background: #eaf7ef;
  border: 1px solid #cfead9;
  border-radius: 999px;
  padding: .22rem .55rem;
  font-size: .7rem;
  font-weight: 800;
}
.trace-check.stop { color: #a35b2d; background: #fff4e8; border-color: #f1d7bb; }
.trace-citation {
  display: inline-block;
  color: var(--trace-blue);
  background: #e8f0fd;
  border: 1px solid #cfddf5;
  border-radius: .4rem;
  padding: .03rem .36rem;
  margin: 0 .1rem;
  font-size: .8em;
  font-weight: 750;
  white-space: nowrap;
}
.trace-query-note {
  color: #647086;
  background: var(--trace-warm);
  border-left: 3px solid #a9b6ca;
  padding: .55rem .75rem;
  font-size: .82rem;
  margin-bottom: .9rem;
}
.trace-pipeline {
  display: grid;
  grid-template-columns: repeat(8, minmax(105px, 1fr));
  align-items: stretch;
  gap: .68rem;
  margin: .65rem 0 1.15rem;
  overflow: visible;
  padding-bottom: .2rem;
}
.trace-pipeline-node {
  position: relative;
  display: block;
  height: 100%;
  box-sizing: border-box;
  border: 1px solid var(--trace-line);
  border-radius: .7rem;
  background: #fff;
  padding: .7rem .75rem;
  text-align: center;
  color: #69758a;
  font-size: .7rem;
  line-height: 1.45;
  cursor: pointer;
  list-style: none;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}
.trace-pipeline-node::-webkit-details-marker { display: none; }
.trace-pipeline-node:hover,
.trace-pipeline-stage[open] > .trace-pipeline-node {
  border-color: #b8c9e8;
  box-shadow: 0 8px 22px rgba(43, 66, 105, .10);
  transform: translateY(-1px);
}
.trace-pipeline-node strong { display: block; color: #2d384d; font-size: .85rem; margin-top: .12rem; }
.trace-pipeline-node.selected { border-color: #bfd2f3; background: #f4f8ff; }
.trace-pipeline-node.verified { border-color: #cfe9d8; background: #f2faf5; }
.trace-pipeline-stage { position: relative; min-width: 0; }
.trace-pipeline-stage:not(:last-child) > .trace-pipeline-node::after {
  content: "→";
  position: absolute;
  right: -.57rem;
  top: 50%;
  color: #a5afbf;
  font-weight: 900;
  transform: translate(50%, -50%);
  z-index: 2;
}
.trace-pipeline-popover {
  position: absolute;
  top: calc(100% + .5rem);
  left: 0;
  z-index: 50;
  width: min(360px, 82vw);
  max-height: 430px;
  overflow-y: auto;
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  transform: translateY(-4px);
  transition: opacity .14s ease, transform .14s ease, visibility .14s;
  border: 1px solid #d9e1ed;
  border-radius: .8rem;
  background: #fff;
  box-shadow: 0 16px 38px rgba(28, 43, 70, .16);
  padding: .85rem;
  text-align: left;
}
.trace-pipeline-stage:nth-last-child(-n+3) > .trace-pipeline-popover { left: auto; right: 0; }
.trace-pipeline-stage:hover > .trace-pipeline-popover,
.trace-pipeline-stage[open] > .trace-pipeline-popover {
  visibility: visible;
  opacity: 1;
  pointer-events: auto;
  transform: translateY(0);
}
.trace-popover-title { color: #29364c; font-size: .82rem; font-weight: 850; }
.trace-popover-caption { color: #7a8699; font-size: .68rem; line-height: 1.45; margin: .18rem 0 .62rem; }
.trace-popover-item { border-top: 1px solid #edf0f5; padding: .58rem 0; }
.trace-popover-item:first-of-type { border-top: 0; padding-top: .15rem; }
.trace-popover-top { display: flex; gap: .38rem; align-items: baseline; }
.trace-popover-rank { color: var(--trace-blue); font-size: .72rem; font-weight: 850; white-space: nowrap; }
.trace-popover-source { min-width: 0; color: #3b485e; font-size: .72rem; font-weight: 800; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trace-popover-copy { color: #6a7588; font-size: .69rem; line-height: 1.5; margin-top: .25rem; }
.trace-popover-meta { color: #50617d; font-size: .65rem; font-weight: 750; margin-top: .25rem; }
.trace-popover-empty { color: #7a8699; font-size: .72rem; padding: .5rem 0; }
.trace-popover-check { color: #227849; background: #eef9f2; border-radius: .45rem; padding: .48rem .55rem; margin-top: .35rem; font-size: .7rem; font-weight: 750; }
.trace-popover-check.stop { color: #9a592d; background: #fff4e8; }
.trace-search-pair { display: grid; grid-template-columns: 1fr 1fr; gap: .28rem; margin-top: .2rem; }
.trace-search-pair span { background: #f4f6f9; border-radius: .35rem; padding: .25rem; color: #445168; font-weight: 750; }
.trace-rank-card {
  border: 1px solid #dce4f1;
  border-radius: .85rem;
  background: #fbfcff;
  padding: .9rem 1rem;
  margin: .55rem 0 1rem;
}
.trace-rank-source { color: #344056; font-size: .82rem; font-weight: 800; }
.trace-rank-flow { display: flex; align-items: center; gap: .3rem; overflow-x: auto; margin: .7rem 0 .55rem; }
.trace-rank-step { flex: 1 0 105px; border-radius: .55rem; background: #f0f3f8; padding: .55rem; text-align: center; color: #7a8597; font-size: .68rem; }
.trace-rank-step strong { display: block; color: #2e394e; font-size: .9rem; margin-top: .12rem; }
.trace-rank-step.final { color: #315ea9; background: #eaf1fd; }
.trace-rank-step.cited { color: #227849; background: #eaf7ef; }
.trace-rank-arrow { color: #a4afbf; font-size: .8rem; }
.trace-insight { color: #526078; font-size: .8rem; line-height: 1.55; }
.trace-timing-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: .55rem; margin: .6rem 0 1rem; }
.trace-timing-card { border: 1px solid var(--trace-line); border-radius: .7rem; background: #fff; padding: .68rem .75rem; }
.trace-timing-label { color: #7a8698; font-size: .68rem; }
.trace-timing-value { color: #2d384d; font-size: .92rem; font-weight: 850; margin-top: .18rem; }
.trace-topk-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .72rem;
  align-items: start;
  margin: .7rem 0 1.2rem;
}
.trace-stage-column {
  min-width: 0;
  border: 1px solid var(--trace-line);
  border-radius: .85rem;
  background: #fbfcfe;
  overflow: hidden;
}
.trace-stage-head {
  display: flex;
  justify-content: space-between;
  gap: .5rem;
  align-items: center;
  padding: .8rem .85rem;
  border-bottom: 1px solid var(--trace-line);
  background: #f5f7fb;
}
.trace-stage-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.35rem;
  height: 1.35rem;
  border-radius: 50%;
  color: #fff;
  background: var(--trace-blue);
  font-size: .7rem;
  font-weight: 800;
  margin-right: .35rem;
}
.trace-stage-title { color: #263148; font-size: .88rem; font-weight: 800; }
.trace-stage-meta { color: #8792a4; font-size: .68rem; text-align: right; white-space: nowrap; }
.trace-result-list { padding: .45rem; }
.trace-result-row {
  border: 1px solid #e7eaf0;
  border-left: 4px solid;
  border-radius: .55rem;
  background: #fff;
  padding: .58rem .62rem;
  margin-bottom: .42rem;
  min-height: 7.1rem;
}
.trace-result-top {
  display: flex;
  align-items: center;
  gap: .35rem;
  min-width: 0;
  margin-bottom: .36rem;
}
.trace-result-rank { color: #263148; font-size: .82rem; font-weight: 850; }
.trace-result-source {
  color: #59667b;
  font-size: .71rem;
  font-weight: 700;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trace-result-copy {
  color: #657188;
  font-size: .72rem;
  line-height: 1.45;
  min-height: 3.05rem;
}
.trace-result-foot {
  display: flex;
  flex-wrap: wrap;
  gap: .25rem;
  align-items: center;
  margin-top: .42rem;
}
.trace-score {
  color: #66748a;
  background: #f3f5f8;
  border-radius: .3rem;
  padding: .12rem .3rem;
  font-size: .65rem;
  font-weight: 700;
}
.trace-cited-badge {
  color: #227849;
  background: #eaf7ef;
  border-radius: .3rem;
  padding: .12rem .3rem;
  font-size: .65rem;
  font-weight: 800;
}
.trace-move-up { color: #2e7850; }
.trace-move-down { color: #b06335; }
.trace-section-label { color: #3e4a60; font-size: .88rem; font-weight: 800; margin: .8rem 0 .45rem; }
.trace-document {
  background: var(--trace-warm);
  border: 1px solid var(--trace-line);
  border-radius: .8rem;
  padding: 1.25rem;
  min-height: 9rem;
  line-height: 1.8;
}
.trace-document mark {
  color: inherit;
  background: #dce8fb;
  border-bottom: 2px solid var(--trace-blue);
  padding: .08rem .12rem;
}
.trace-claim {
  border: 1px solid #d5e2f7;
  border-radius: .75rem;
  background: #f8faff;
  padding: .9rem;
  margin-bottom: .7rem;
}
.trace-pass { color: var(--trace-green); font-weight: 750; }
.trace-note { color: var(--trace-muted); font-size: .82rem; line-height: 1.55; }
.trace-callout {
  border: 1px solid #dfe5ee;
  border-radius: .8rem;
  background: var(--trace-warm);
  padding: .9rem 1rem;
  color: #596579;
  font-size: .86rem;
  line-height: 1.6;
  margin: .7rem 0;
}
.trace-callout strong { color: #263147; }
.trace-story-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: .65rem 0 1.3rem; }
.trace-story-card { border: 1px solid var(--trace-line); border-radius: .8rem; padding: 1rem; background: #fff; }
.trace-story-number { color: var(--trace-blue); font-size: .73rem; font-weight: 800; }
.trace-story-title { color: #263147; font-weight: 800; margin: .35rem 0; }
.trace-story-copy { color: #6c7789; font-size: .84rem; line-height: 1.55; }
.trace-bar-chart { margin: .7rem 0 1.1rem; }
.trace-bar-row { display: grid; grid-template-columns: 92px 1fr 54px; gap: .6rem; align-items: center; margin: .5rem 0; }
.trace-bar-label { color: #586478; font-size: .78rem; font-weight: 700; }
.trace-bar-track { height: .68rem; border-radius: 999px; background: #edf0f4; overflow: hidden; }
.trace-bar-fill { height: 100%; border-radius: 999px; background: #9aabc6; }
.trace-bar-fill.selected { background: var(--trace-blue); }
.trace-bar-value { color: #3d4759; font-size: .76rem; font-weight: 750; text-align: right; }
[data-testid="stMetric"] { border: 1px solid var(--trace-line); border-radius: .8rem; padding: .8rem .9rem; background: #fff; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { border-radius: .55rem .55rem 0 0; }
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs button[role="tab"][aria-selected="true"] { color: var(--trace-blue) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--trace-blue) !important; }
@media (max-width: 1180px) {
  .trace-topk-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trace-pipeline { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
  .trace-pipeline-stage:nth-child(4n) > .trace-pipeline-popover { left: auto; right: 0; }
}
@media (max-width: 760px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .trace-header { align-items: flex-start; flex-direction: column; }
  .trace-topk-grid, .trace-story-grid, .trace-result-layout { grid-template-columns: 1fr; }
  .trace-timing-grid { grid-template-columns: repeat(2, 1fr); }
  .trace-pipeline { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .trace-pipeline-stage:nth-child(even) > .trace-pipeline-popover { left: auto; right: 0; }
  .trace-pipeline-stage > .trace-pipeline-node::after { display: none; }
}
</style>
"""


def apply_trace_style() -> None:
    """앱 전체에 DocLens Trace의 색과 여백을 적용한다."""

    st.markdown(APP_STYLE, unsafe_allow_html=True)


def render_trace_header(saved_text_count: int) -> None:
    """제품명과 이 화면에서 확인할 실제 RAG 흐름을 표시한다."""

    st.markdown(
        (
            '<div class="trace-header">'
            '<div class="trace-brand">DocLens <span>Trace</span></div>'
            '<div class="trace-context">'
            f'문서 → BM25 · Embedding · RRF · BGE → Ollama 근거 답변 · 기본 문서 {saved_text_count}개'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def citation_numbers(answer: str) -> tuple[int, ...]:
    """답변에 실제 등장한 근거 번호를 중복 없이 반환한다."""

    return tuple(dict.fromkeys(int(value) for value in CITATION_PATTERN.findall(answer)))


def _page_label(item: dict[str, object]) -> str:
    page_number = int(item.get("page_number", 0) or 0)
    return f"p.{page_number}" if page_number > 0 else str(item.get("page_label", ""))


def _excerpt(text: str, limit: int = 105) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}…"


def _answer_html(answer: str, evidence: Iterable[RAGEvidence]) -> str:
    evidence_by_rank = {item["rank"]: item for item in evidence}
    rendered: list[str] = []
    cursor = 0
    for match in CITATION_PATTERN.finditer(answer):
        rendered.append(html.escape(answer[cursor : match.start()]))
        rank = int(match.group(1))
        item = evidence_by_rank.get(rank)
        label = f"근거 {rank}"
        if item is not None:
            label = f"{rank} · {item['source_filename']} {_page_label(item)}"
        rendered.append(f'<span class="trace-citation">{html.escape(label)}</span>')
        cursor = match.end()
    rendered.append(html.escape(answer[cursor:]))
    return "".join(rendered).replace("\n", "<br>")


def _format_score(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _format_rank(value: object) -> str:
    return "—" if value is None else f"{int(value)}위"


def _cited_evidence(response: RAGResponse) -> list[RAGEvidence]:
    cited = set(citation_numbers(response.answer))
    return [item for item in response.evidence if item["rank"] in cited]


def _supporting_excerpt(claim: str, source_text: str, limit: int = 280) -> str:
    """답변 단어와 가장 많이 겹치는 원문 줄만 골라 직접 근거로 보여준다."""

    clean_claim = CITATION_PATTERN.sub("", claim)
    stopwords = {
        "근거",
        "신청",
        "지원",
        "가능",
        "사람",
        "합니다",
        "있습니다",
        "됩니다",
    }
    terms = {
        token
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", clean_claim)
        if token not in stopwords
    }
    segments = [
        segment.strip()
        for segment in re.split(r"[\r\n]+|(?<=[.!?])\s+", source_text)
        if segment.strip()
    ]
    scored = [
        (index, segment, sum(term in segment for term in terms))
        for index, segment in enumerate(segments)
    ]
    selected = sorted(
        sorted((item for item in scored if item[2] > 0), key=lambda item: item[2], reverse=True)[:3],
        key=lambda item: item[0],
    )
    excerpt = " ".join(item[1] for item in selected)
    return _excerpt(excerpt or source_text, limit)


def _stage_count(attempt: dict[str, object], stage_key: str) -> int:
    stages = attempt.get("stages", {})
    stage = stages.get(stage_key, {}) if isinstance(stages, dict) else {}
    return int(stage.get("candidate_count", len(stage.get("results", []))))


def _stage_payload(attempt: dict[str, object], stage_key: str) -> dict[str, object]:
    stages = attempt.get("stages", {})
    if not isinstance(stages, dict):
        return {}
    stage = stages.get(stage_key, {})
    return stage if isinstance(stage, dict) else {}


def _document_summaries_from_trace(
    attempt: dict[str, object],
) -> list[dict[str, object]]:
    """이전 저장 응답은 검색 후보에서 문서 이름과 확인된 페이지를 복원한다."""

    documents: dict[str, dict[str, object]] = {}
    stages = attempt.get("stages", {})
    if not isinstance(stages, dict):
        return []
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        for result in stage.get("results", []):
            filename = str(result.get("source_filename", "")).strip()
            if not filename:
                continue
            page_number = int(result.get("page_number", 0) or 0)
            summary = documents.setdefault(
                filename,
                {"filename": filename, "file_type": "", "pages": 0},
            )
            summary["pages"] = max(int(summary.get("pages", 0)), page_number)
    return list(documents.values())


def _document_popover_html(
    attempt: dict[str, object],
    document_summaries: list[dict[str, object]] | None,
    document_count: int | None,
) -> str:
    summaries = document_summaries or _document_summaries_from_trace(attempt)
    items: list[str] = []
    for item in summaries[:8]:
        filename = str(item.get("filename", "알 수 없는 문서"))
        file_type = str(item.get("file_type", "")).upper()
        pages = int(item.get("pages", 0) or 0)
        chunks = item.get("chunks")
        details = [value for value in (file_type, f"{pages}쪽" if pages else "") if value]
        if chunks is not None:
            details.append(f"Chunk {int(chunks)}개")
        if item.get("ocr"):
            details.append("OCR 사용")
        items.append(
            '<div class="trace-popover-item">'
            f'<div class="trace-popover-source" title="{html.escape(filename)}">{html.escape(filename)}</div>'
            f'<div class="trace-popover-meta">{html.escape(" · ".join(details) or "입력 문서")}</div>'
            "</div>"
        )
    if not items:
        items.append('<div class="trace-popover-empty">저장된 문서 상세 정보가 없습니다.</div>')
    remaining = max((document_count or len(summaries)) - len(items), 0)
    footer = (
        f'<div class="trace-popover-caption">외 {remaining}개 문서</div>' if remaining else ""
    )
    return (
        '<div class="trace-popover-title">검색에 넣은 문서</div>'
        '<div class="trace-popover-caption">파일 형식·페이지·Chunk 수를 확인합니다.</div>'
        f'{"".join(items)}{footer}'
    )


def _chunk_popover_html(
    attempt: dict[str, object],
    document_summaries: list[dict[str, object]] | None,
    chunk_count: int | None,
) -> str:
    summaries = document_summaries or _document_summaries_from_trace(attempt)
    items: list[str] = []
    for item in summaries:
        chunks = item.get("chunks")
        if chunks is None:
            continue
        filename = str(item.get("filename", "알 수 없는 문서"))
        items.append(
            '<div class="trace-popover-item">'
            f'<div class="trace-popover-source" title="{html.escape(filename)}">{html.escape(filename)}</div>'
            f'<div class="trace-popover-meta">Chunk {int(chunks)}개</div>'
            "</div>"
        )
    if not items:
        items.append(
            '<div class="trace-popover-empty">전체 Chunk 목록은 실행 당시 저장하지 않았습니다. '
            '아래 검색 단계에서 실제 후보 Chunk를 확인할 수 있습니다.</div>'
        )
    total = "—" if chunk_count is None else f"{chunk_count}개"
    return (
        f'<div class="trace-popover-title">만들어진 Chunk · {total}</div>'
        '<div class="trace-popover-caption">문서별로 검색 가능한 글 조각을 몇 개 만들었는지 보여줍니다.</div>'
        f'{"".join(items)}'
    )


def _result_popover_html(
    attempt: dict[str, object],
    stage_key: str,
    title: str,
    description: str,
) -> str:
    stage = _stage_payload(attempt, stage_key)
    results = list(stage.get("results", []))[:3]
    candidate_count = int(stage.get("candidate_count", len(results)))
    elapsed_ms = float(stage.get("elapsed_ms", 0.0))
    items: list[str] = []
    for result in results:
        source = f"{result.get('source_filename', '')} {_page_label(result)}".strip()
        score_label, score = _stage_score(stage_key, result)
        detail = re.sub(r"<[^>]+>", "", _rank_detail(stage_key, result))
        meta = f"{score_label} {_format_score(score)}"
        if detail:
            meta += f" · {detail}"
        items.append(
            '<div class="trace-popover-item">'
            '<div class="trace-popover-top">'
            f'<span class="trace-popover-rank">#{int(result.get("rank", 0))}</span>'
            f'<span class="trace-popover-source" title="{html.escape(source)}">{html.escape(source)}</span>'
            "</div>"
            f'<div class="trace-popover-copy">{html.escape(_excerpt(str(result.get("text", "")), 92))}</div>'
            f'<div class="trace-popover-meta">{html.escape(meta)}</div>'
            "</div>"
        )
    if not items:
        items.append('<div class="trace-popover-empty">이 단계에 저장된 후보가 없습니다.</div>')
    return (
        f'<div class="trace-popover-title">{html.escape(title)} 후보 Chunk Top-{len(results)}</div>'
        f'<div class="trace-popover-caption">{html.escape(description)} · '
        f'전체 {candidate_count}개 · {elapsed_ms:.0f}ms</div>'
        f'{"".join(items)}'
    )


def _evidence_popover_html(response: RAGResponse) -> str:
    items: list[str] = []
    for item in _cited_evidence(response):
        claim = _claim_for_citation(response.answer, item["rank"])
        source = f"{item['source_filename']} · {_page_label(item)}"
        items.append(
            '<div class="trace-popover-item">'
            '<div class="trace-popover-top">'
            f'<span class="trace-popover-rank">근거 {item["rank"]}</span>'
            f'<span class="trace-popover-source" title="{html.escape(source)}">{html.escape(source)}</span>'
            "</div>"
            f'<div class="trace-popover-copy">“{html.escape(_supporting_excerpt(claim, item["text"], 150))}”</div>'
            "</div>"
        )
    if not items:
        items.append('<div class="trace-popover-empty">답변에 사용된 인용 근거가 없습니다.</div>')
    return (
        '<div class="trace-popover-title">최종 답변에 사용한 근거</div>'
        '<div class="trace-popover-caption">BGE 상위 후보 중 답변이 실제로 인용한 Chunk입니다.</div>'
        f'{"".join(items)}'
    )


def _verification_popover_html(response: RAGResponse) -> str:
    cited_items = _cited_evidence(response)
    cited_claims = [
        _claim_for_citation(response.answer, item["rank"]) for item in cited_items
    ]
    numeric_checks = [
        verify_citation(claim, [item["text"]])
        for claim, item in zip(cited_claims, cited_items, strict=True)
    ]
    has_numeric_claim = any(re.search(r"\d", claim) for claim in cited_claims)
    if not cited_items:
        numeric_label = "숫자를 검증할 인용 근거가 없음"
    elif has_numeric_claim:
        numeric_label = "답변의 숫자가 인용 원문과 일치"
    else:
        numeric_label = "별도로 검증할 숫자 주장이 없음"
    checks = [
        (bool(cited_items), "인용 번호가 실제 검색 결과에 존재"),
        (
            bool(cited_items) and all(check["grounded"] for check in numeric_checks),
            numeric_label,
        ),
        (response.status == "answered" and bool(cited_items), "로컬 LLM의 의미 근거 판정 통과"),
    ]
    rows = "".join(
        f'<div class="trace-popover-check{"" if passed else " stop"}">'
        f'{"✓" if passed else "!"} {html.escape(label)}</div>'
        for passed, label in checks
    )
    return (
        '<div class="trace-popover-title">근거 검증 결과</div>'
        '<div class="trace-popover-caption">인용·숫자는 코드로, 의미 충분성은 로컬 LLM으로 구분해 확인합니다.</div>'
        f"{rows}"
    )


def _pipeline_stage_html(label: str, value: str, body: str, css_class: str = "") -> str:
    return (
        '<details class="trace-pipeline-stage" name="trace-pipeline-stage">'
        f'<summary class="trace-pipeline-node {css_class}">{html.escape(label)}'
        f'<strong>{html.escape(value)}</strong></summary>'
        f'<div class="trace-pipeline-popover">{body}</div>'
        "</details>"
    )


def _pipeline_html(
    response: RAGResponse,
    attempt: dict[str, object],
    *,
    document_count: int | None,
    chunk_count: int | None,
    document_summaries: list[dict[str, object]] | None,
) -> str:
    document_value = "—" if document_count is None else f"{document_count}개"
    chunk_value = "—" if chunk_count is None else f"{chunk_count}개"
    cited_count = len(citation_numbers(response.answer))
    verification = "PASS" if response.status == "answered" and cited_count else "STOP"
    verification_class = "verified" if verification == "PASS" else ""
    nodes = [
        _pipeline_stage_html(
            "입력 문서",
            document_value,
            _document_popover_html(attempt, document_summaries, document_count),
        ),
        _pipeline_stage_html(
            "Chunk",
            chunk_value,
            _chunk_popover_html(attempt, document_summaries, chunk_count),
        ),
        _pipeline_stage_html(
            "같은 단어",
            f"BM25 {_stage_count(attempt, 'bm25')}개",
            _result_popover_html(attempt, "bm25", "BM25", "질문과 같은 단어를 찾음"),
        ),
        _pipeline_stage_html(
            "비슷한 뜻",
            f"Embedding {_stage_count(attempt, 'vector')}개",
            _result_popover_html(attempt, "vector", "Embedding", "질문과 의미가 비슷한 글을 찾음"),
        ),
        _pipeline_stage_html(
            "순위 결합",
            f"RRF {_stage_count(attempt, 'rrf')}개",
            _result_popover_html(attempt, "rrf", "RRF", "두 검색 순위를 점수가 아닌 순위로 합침"),
        ),
        _pipeline_stage_html(
            "재정렬",
            f"BGE {_stage_count(attempt, 'reranker')}개",
            _result_popover_html(attempt, "reranker", "BGE", "질문과 후보를 함께 읽고 다시 정렬"),
            "selected",
        ),
        _pipeline_stage_html(
            "답변 근거",
            f"{cited_count}개",
            _evidence_popover_html(response),
            "selected",
        ),
        _pipeline_stage_html(
            "근거 검증",
            verification,
            _verification_popover_html(response),
            verification_class,
        ),
    ]
    return '<div class="trace-pipeline">' + "".join(nodes) + "</div>"


def _rank_insight(item: RAGEvidence) -> str:
    bm25_rank = item.get("bm25_rank")
    vector_rank = item.get("vector_rank")
    rrf_rank = item.get("rrf_rank")
    final_rank = item["rank"]
    if bm25_rank is None and vector_rank is not None:
        return "키워드 검색에서 놓친 근거를 의미 검색이 찾고 BGE가 최종 근거로 선택했습니다."
    if bm25_rank is not None and int(bm25_rank) > final_rank:
        return (
            f"BM25 {int(bm25_rank)}위였던 근거가 의미 검색·순위 결합을 거쳐 "
            f"BGE {final_rank}위로 올라왔습니다."
        )
    if rrf_rank is not None and int(rrf_rank) > final_rank:
        return f"RRF {int(rrf_rank)}위 후보를 BGE가 {final_rank}위로 끌어올렸습니다."
    return "여러 검색기가 공통으로 찾은 상위 근거를 BGE가 유지하고 답변에 인용했습니다."


def _rank_journey_html(response: RAGResponse) -> str:
    cards: list[str] = []
    for item in _cited_evidence(response):
        steps = [
            ("BM25", _format_rank(item.get("bm25_rank")), ""),
            ("Embedding", _format_rank(item.get("vector_rank")), ""),
            ("RRF", _format_rank(item.get("rrf_rank")), ""),
            ("BGE", f"{item['rank']}위", "final"),
            ("최종 답변", "인용", "cited"),
        ]
        journey = '<span class="trace-rank-arrow">→</span>'.join(
            (
                f'<div class="trace-rank-step {css_class}">{html.escape(label)}'
                f'<strong>{html.escape(value)}</strong></div>'
            )
            for label, value, css_class in steps
        )
        source = f"{item['source_filename']} · {_page_label(item)}"
        cards.append(
            '<div class="trace-rank-card">'
            f'<div class="trace-rank-source">{html.escape(source)}</div>'
            f'<div class="trace-rank-flow">{journey}</div>'
            f'<div class="trace-insight">{html.escape(_rank_insight(item))}</div>'
            "</div>"
        )
    return "".join(cards)


def _format_duration(elapsed_ms: float | None) -> str:
    if elapsed_ms is None:
        return "—"
    if elapsed_ms < 1000:
        return f"{elapsed_ms:.0f}ms"
    return f"{elapsed_ms / 1000:.1f}s"


def _timing_html(
    response: RAGResponse,
    attempts: list[dict[str, object]],
    *,
    elapsed_seconds: float | None,
    document_prep_ms: float | None,
) -> str:
    timings = response.timings_ms
    retrieval_ms = timings.get("retrieval")
    if retrieval_ms is None:
        retrieval_ms = sum(float(item.get("total_ms", 0.0)) for item in attempts)
    generation_ms = timings.get("generation")
    validation_ms = timings.get("validation")
    if generation_ms is None and elapsed_seconds is not None:
        generation_ms = max(
            elapsed_seconds * 1000
            - (document_prep_ms or 0.0)
            - retrieval_ms
            - (validation_ms or 0.0),
            0.0,
        )
    values = (
        ("문서 준비", document_prep_ms),
        ("검색·재정렬", retrieval_ms),
        ("답변 판단·생성", generation_ms),
        ("근거 검증", validation_ms),
        ("전체", elapsed_seconds * 1000 if elapsed_seconds is not None else None),
    )
    cards = "".join(
        '<div class="trace-timing-card">'
        f'<div class="trace-timing-label">{html.escape(label)}</div>'
        f'<div class="trace-timing-value">{_format_duration(value)}</div>'
        "</div>"
        for label, value in values
    )
    return f'<div class="trace-timing-grid">{cards}</div>'


def _overview_html(
    response: RAGResponse,
    *,
    corpus_label: str | None,
    llm_label: str | None,
) -> str:
    cited_items = _cited_evidence(response)
    evidence_cards: list[str] = []
    for item in cited_items[:2]:
        claim = _claim_for_citation(response.answer, item["rank"])
        source = f"{item['source_filename']} · {_page_label(item)}"
        quote = _supporting_excerpt(claim, item["text"])
        evidence_cards.append(
            '<div class="trace-evidence-card">'
            f'<div class="trace-evidence-source">{html.escape(source)}</div>'
            f'<div class="trace-evidence-quote">“{html.escape(quote)}”</div>'
            "</div>"
        )
    if not evidence_cards:
        evidence_cards.append('<div class="trace-note">검증을 통과한 인용 근거가 없습니다.</div>')

    answer_without_citations = CITATION_PATTERN.sub("", response.answer)
    has_numeric_claim = bool(re.search(r"\d", answer_without_citations))
    if response.status == "answered" and cited_items:
        checks = ["✓ 인용 존재", "✓ 의미 근거 판정"]
        checks.append("✓ 숫자 원문 일치" if has_numeric_claim else "✓ 숫자 주장 없음")
        check_html = "".join(f'<span class="trace-check">{item}</span>' for item in checks)
        answer_label = "근거가 있는 답변"
    else:
        check_html = '<span class="trace-check stop">근거 검증 중단</span>'
        answer_label = "근거 부족 · 답변 중단"

    summary_values = [value for value in (corpus_label, llm_label) if value]
    summary_html = "".join(
        f'<span class="trace-summary-chip">{html.escape(value)}</span>'
        for value in summary_values
    )
    return (
        '<div class="trace-result-layout">'
        '<section class="trace-answer-box">'
        f'<div class="trace-answer-label">{html.escape(answer_label)}</div>'
        f'<div class="trace-answer">{_answer_html(response.answer, response.evidence)}</div>'
        f'<div class="trace-run-summary">{summary_html}</div>'
        "</section>"
        '<section class="trace-evidence-panel">'
        '<div class="trace-evidence-label">직접 근거와 검증</div>'
        f'{"".join(evidence_cards)}'
        f'<div class="trace-check-row">{check_html}</div>'
        "</section></div>"
    )


def build_rank_flow_rows(response: RAGResponse) -> list[dict[str, object]]:
    """최종 근거가 BGE에서 어떻게 움직였는지 표 형식으로 만든다."""

    rows: list[dict[str, object]] = []
    cited = set(citation_numbers(response.answer))
    for item in response.evidence:
        rrf_rank = item.get("rrf_rank")
        final_rank = item["rank"]
        if rrf_rank is None:
            change = "—"
        else:
            difference = int(rrf_rank) - final_rank
            change = f"▲ {difference}" if difference > 0 else (
                f"▼ {abs(difference)}" if difference < 0 else "유지"
            )
        rows.append(
            {
                "근거": f"{final_rank}",
                "문서": f"{item['source_filename']} {_page_label(item)}",
                "BM25": _format_rank(item.get("bm25_rank")),
                "Embedding": _format_rank(item.get("vector_rank")),
                "RRF": _format_rank(rrf_rank),
                "BGE 최종": f"{final_rank}위",
                "순위 변화": change,
                "답변 사용": "인용" if final_rank in cited else "후보",
            }
        )
    return rows


def _fallback_attempt(response: RAGResponse) -> dict[str, object]:
    """이전 저장 응답에도 화면이 깨지지 않도록 최종 근거로 trace를 만든다."""

    results = [dict(item) for item in response.evidence]
    return {
        "attempt": 1,
        "query": response.final_query,
        "total_ms": 0.0,
        "stages": {
            key: {
                "label": label,
                "candidate_count": len(results),
                "elapsed_ms": 0.0,
                "results": results,
            }
            for key, label in (
                ("bm25", "BM25"),
                ("vector", "Embedding"),
                ("rrf", "RRF"),
                ("reranker", "BGE"),
            )
        },
    }


def _stage_score(stage_key: str, result: dict[str, object]) -> tuple[str, object]:
    if stage_key == "bm25":
        return "BM25", result.get("score", result.get("bm25_score"))
    if stage_key == "vector":
        return "cos", result.get("similarity", result.get("vector_similarity"))
    if stage_key == "rrf":
        return "RRF", result.get("rrf_score")
    return "BGE", result.get("reranker_score", result.get("score"))


def _rank_detail(stage_key: str, result: dict[str, object]) -> str:
    if stage_key == "rrf":
        return f"BM25 {_format_rank(result.get('bm25_rank'))} · Embedding {_format_rank(result.get('vector_rank'))}"
    if stage_key == "reranker" and result.get("rrf_rank") is not None:
        difference = int(result["rrf_rank"]) - int(result["rank"])
        if difference > 0:
            return f'<span class="trace-move-up">RRF {int(result["rrf_rank"])}위 → ▲ {difference}</span>'
        if difference < 0:
            return f'<span class="trace-move-down">RRF {int(result["rrf_rank"])}위 → ▼ {abs(difference)}</span>'
        return f"RRF {int(result['rrf_rank'])}위 → 유지"
    return ""


def _color_map(attempt: dict[str, object]) -> dict[str, str]:
    ordered_ids: list[str] = []
    stages = attempt.get("stages", {})
    if isinstance(stages, dict):
        for stage in stages.values():
            if not isinstance(stage, dict):
                continue
            for result in stage.get("results", []):
                chunk_id = str(result.get("chunk_id", ""))
                if chunk_id and chunk_id not in ordered_ids:
                    ordered_ids.append(chunk_id)
    return {
        chunk_id: CHUNK_COLORS[index % len(CHUNK_COLORS)]
        for index, chunk_id in enumerate(ordered_ids)
    }


def _topk_html(response: RAGResponse, attempt: dict[str, object], top_k: int = 5) -> str:
    cited_ranks = set(citation_numbers(response.answer))
    cited_ids = {
        item["chunk_id"] for item in response.evidence if item["rank"] in cited_ranks
    }
    colors = _color_map(attempt)
    stages = attempt.get("stages", {})
    columns: list[str] = []
    stage_order = (
        ("bm25", "BM25", "같은 단어"),
        ("vector", "Embedding", "비슷한 뜻"),
        ("rrf", "RRF", "두 순위 결합"),
        ("reranker", "BGE", "질문과 함께 재검토"),
    )
    for stage_index, (stage_key, title, subtitle) in enumerate(stage_order, start=1):
        stage = stages.get(stage_key, {}) if isinstance(stages, dict) else {}
        results = list(stage.get("results", []))[:top_k]
        candidate_count = int(stage.get("candidate_count", len(results)))
        elapsed_ms = float(stage.get("elapsed_ms", 0.0))
        cards: list[str] = []
        for result in results:
            chunk_id = str(result.get("chunk_id", ""))
            color = colors.get(chunk_id, CHUNK_COLORS[-1])
            score_label, score = _stage_score(stage_key, result)
            cited_badge = (
                '<span class="trace-cited-badge">답변 인용</span>'
                if chunk_id in cited_ids
                else ""
            )
            detail = _rank_detail(stage_key, result)
            detail_badge = f'<span class="trace-score">{detail}</span>' if detail else ""
            source = f"{result.get('source_filename', '')} {_page_label(result)}".strip()
            cards.append(
                f'<article class="trace-result-row" style="border-left-color:{color}">'
                '<div class="trace-result-top">'
                f'<span class="trace-result-rank">#{int(result.get("rank", 0))}</span>'
                f'<span class="trace-result-source" title="{html.escape(source)}">{html.escape(source)}</span>'
                "</div>"
                f'<div class="trace-result-copy">{html.escape(_excerpt(str(result.get("text", ""))))}</div>'
                '<div class="trace-result-foot">'
                f'<span class="trace-score">{score_label} {_format_score(score)}</span>'
                f"{detail_badge}{cited_badge}"
                "</div></article>"
            )
        columns.append(
            '<section class="trace-stage-column">'
            '<div class="trace-stage-head"><div>'
            f'<span class="trace-stage-number">{stage_index}</span>'
            f'<span class="trace-stage-title">{html.escape(title)}</span>'
            "</div>"
            f'<div class="trace-stage-meta">{html.escape(subtitle)}<br>Top {min(top_k, candidate_count)} / {candidate_count} · {elapsed_ms:.0f}ms</div>'
            "</div>"
            f'<div class="trace-result-list">{"".join(cards)}</div>'
            "</section>"
        )
    return f'<div class="trace-topk-grid">{"".join(columns)}</div>'


def _claim_for_citation(answer: str, rank: int) -> str:
    marker = f"[근거 {rank}]"
    marker_index = answer.find(marker)
    if marker_index < 0:
        return answer
    left_boundaries = [answer.rfind(symbol, 0, marker_index) for symbol in (".", "!", "?", "\n")]
    left = max(left_boundaries) + 1
    right_candidates = [
        position
        for symbol in (".", "!", "?", "\n")
        if (position := answer.find(symbol, marker_index + len(marker))) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(answer)
    claim = CITATION_PATTERN.sub("", answer[left:right]).strip()
    return claim or CITATION_PATTERN.sub("", answer).strip()


def _render_verification_view(response: RAGResponse) -> None:
    cited = citation_numbers(response.answer)
    evidence_by_rank = {item["rank"]: item for item in response.evidence}
    available = [rank for rank in cited if rank in evidence_by_rank]
    if not available:
        st.info("답변에 인용된 근거가 없어 검증할 항목이 없습니다.")
        return

    selected_rank = st.selectbox(
        "검증할 근거",
        available,
        format_func=lambda rank: (
            f"근거 {rank} · {evidence_by_rank[rank]['source_filename']} "
            f"{_page_label(evidence_by_rank[rank])}"
        ),
        key="trace_selected_citation",
    )
    item = evidence_by_rank[selected_rank]
    claim = _claim_for_citation(response.answer, selected_rank)
    numeric_check = verify_citation(claim, [item["text"]])

    source_column, claim_column = st.columns([2, 1], gap="large")
    with source_column:
        st.markdown(f"#### {item['source_filename']} · {_page_label(item)}")
        st.caption("답변 주장을 직접 지지하는 원문 구간")
        supporting_excerpt = _supporting_excerpt(claim, item["text"], limit=520)
        st.markdown(
            '<div class="trace-document"><mark>'
            f'{html.escape(supporting_excerpt).replace(chr(10), "<br>")}'
            "</mark></div>",
            unsafe_allow_html=True,
        )
    with claim_column:
        st.markdown("#### 답변과 근거 연결")
        st.markdown(
            '<div class="trace-claim">'
            f'<strong>주장 {selected_rank}</strong><br><br>{html.escape(claim)}'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="trace-pass">✓ 인용 번호가 실제 검색 결과에 존재</div>',
            unsafe_allow_html=True,
        )
        if numeric_check["grounded"]:
            numeric_message = (
                "✓ 주장의 숫자가 원문에 있음"
                if re.search(r"\d", CITATION_PATTERN.sub("", claim))
                else "✓ 별도로 검증할 숫자 주장 없음"
            )
            st.markdown(
                f'<div class="trace-pass">{numeric_message}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"숫자 확인 필요: {numeric_check['missing']}")
        st.markdown(
            '<div class="trace-pass">✓ 로컬 LLM의 의미적 근거 충분성 판정 통과</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="trace-note">인용·숫자 검증은 코드가 수행합니다. 의미적 근거 판정은 현재 로컬 LLM의 1차 판정이며, 독립적인 Ragas 평가와 사람 검토는 평가 탭에서 구분합니다.</p>',
            unsafe_allow_html=True,
        )

    if st.checkbox("전체 Chunk 원문 보기", key=f"trace_full_chunk_{selected_rank}"):
        st.caption(f"chunk ID · {item['chunk_id']}")
        st.code(item["text"], language=None)


def _raw_rows(response: RAGResponse, attempt: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stages = attempt.get("stages", {})
    if not isinstance(stages, dict):
        return rows
    for stage_key in ("bm25", "vector", "rrf", "reranker"):
        stage = stages.get(stage_key, {})
        for result in stage.get("results", []):
            score_label, score = _stage_score(stage_key, result)
            rows.append(
                {
                    "단계": str(stage.get("label", stage_key)),
                    "순위": result.get("rank"),
                    "Chunk ID": result.get("chunk_id"),
                    "출처": f"{result.get('source_filename', '')} {_page_label(result)}".strip(),
                    "점수 종류": score_label,
                    "원점수": _format_score(score),
                    "BM25 순위": _format_rank(result.get("bm25_rank")),
                    "Embedding 순위": _format_rank(result.get("vector_rank")),
                    "RRF 순위": _format_rank(result.get("rrf_rank")),
                }
            )
    return rows


def render_trace_workspace(
    response: RAGResponse,
    *,
    elapsed_seconds: float | None,
    trace_id: str,
    corpus_label: str | None = None,
    llm_label: str | None = None,
    document_prep_ms: float | None = None,
    document_count: int | None = None,
    chunk_count: int | None = None,
    document_summaries: list[dict[str, object]] | None = None,
) -> None:
    """답변·직접 근거를 먼저, 전체 검색 trace를 그 다음 계층에 보여준다."""

    attempts = list(response.retrieval_trace) or [_fallback_attempt(response)]
    final_attempt = attempts[-1]

    st.markdown("### 답변과 직접 근거")
    st.markdown(
        _overview_html(
            response,
            corpus_label=corpus_label,
            llm_label=llm_label,
        ),
        unsafe_allow_html=True,
    )
    if response.final_query != response.question:
        st.markdown(
            '<div class="trace-query-note">'
            f'<strong>질문 재작성:</strong> {html.escape(response.question)} → {html.escape(response.final_query)}'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 한 번의 실행에서 근거가 좁혀진 과정")
    st.markdown(
        _pipeline_html(
            response,
            final_attempt,
            document_count=document_count,
            chunk_count=chunk_count,
            document_summaries=document_summaries,
        ),
        unsafe_allow_html=True,
    )
    st.caption("각 단계를 가리키면 Top-3 미리보기가 열리고, 클릭하면 고정됩니다. 전체 후보는 아래 실행 추적에서 확인합니다.")

    cited_journey = _rank_journey_html(response)
    if cited_journey:
        st.markdown("### 최종 근거가 선택된 이유")
        st.markdown(cited_journey, unsafe_allow_html=True)

    st.markdown("### 어디에서 시간이 걸렸나")
    st.markdown(
        _timing_html(
            response,
            attempts,
            elapsed_seconds=elapsed_seconds,
            document_prep_ms=document_prep_ms,
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "첫 실행은 Embedding·BGE·Ollama 모델을 메모리에 올리는 시간이 포함될 수 있습니다. "
        "단계별 시간을 분리해 검색 병목과 생성 병목을 구분합니다."
    )

    with st.expander("실행 추적 · 전체 단계별 Top-k"):
        if len(attempts) > 1:
            selected_index = st.selectbox(
                "검색 시도",
                range(len(attempts)),
                index=len(attempts) - 1,
                format_func=lambda index: (
                    f"{index + 1}차 · {attempts[index].get('query', '')}"
                    + (" (질문 재작성)" if index else "")
                ),
                key="trace_attempt",
            )
            selected_attempt = attempts[selected_index]
        else:
            selected_attempt = final_attempt
        st.caption(
            f"{selected_attempt.get('attempt', 1)}차 검색 · "
            f'“{selected_attempt.get("query", response.final_query)}” · '
            "같은 색은 같은 Chunk입니다. 단계별 원점수 범위는 서로 다릅니다."
        )
        st.markdown(_topk_html(response, selected_attempt), unsafe_allow_html=True)

    with st.expander("답변과 원문 자세히 비교"):
        _render_verification_view(response)

    with st.expander("개발자 정보 · Chunk ID와 원점수"):
        st.caption(f"실행 ID · {trace_id} · {' → '.join(response.steps)}")
        st.caption(f"LangGraph 판단 · {response.decision_reason}")
        st.dataframe(
            _raw_rows(response, final_attempt),
            hide_index=True,
            use_container_width=True,
        )


__all__ = [
    "apply_trace_style",
    "build_rank_flow_rows",
    "citation_numbers",
    "render_trace_header",
    "render_trace_workspace",
]
