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
  margin: .55rem 0 1.35rem;
}
.trace-answer-label { color: var(--trace-blue); font-size: .75rem; font-weight: 800; }
.trace-answer { color: #344056; font-size: 1.03rem; line-height: 1.85; margin-top: .45rem; }
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
  min-height: 18rem;
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
@media (max-width: 1180px) { .trace-topk-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 760px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .trace-header { align-items: flex-start; flex-direction: column; }
  .trace-topk-grid, .trace-story-grid { grid-template-columns: 1fr; }
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
            f'문서 → 질문 → BM25 · Embedding · RRF · BGE Top-k → 근거 답변 · 기본 문서 {saved_text_count}개'
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
        st.caption(f"검색된 원문 구간 · chunk ID {item['chunk_id']}")
        st.markdown(
            '<div class="trace-document"><mark>'
            f'{html.escape(item["text"]).replace(chr(10), "<br>")}'
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
        st.markdown('<div class="trace-pass">✓ 실제 검색 결과의 인용 번호</div>', unsafe_allow_html=True)
        if numeric_check["grounded"]:
            st.markdown('<div class="trace-pass">✓ 주장의 숫자가 원문에 있음</div>', unsafe_allow_html=True)
        else:
            st.warning(f"숫자 확인 필요: {numeric_check['missing']}")
        st.markdown(
            '<p class="trace-note">인용 번호와 숫자는 자동 검사하고, 문장 전체의 의미는 Ragas와 사람 검토로 추가 확인합니다.</p>',
            unsafe_allow_html=True,
        )


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
) -> None:
    """답변과 BM25→Embedding→RRF→BGE Top-k를 한 화면에 이어서 보여준다."""

    cited = set(citation_numbers(response.answer))
    attempts = list(response.retrieval_trace) or [_fallback_attempt(response)]
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
        attempt = attempts[selected_index]
    else:
        attempt = attempts[0]

    elapsed_label = f"{elapsed_seconds:.2f}s" if elapsed_seconds is not None else "측정 안 됨"
    summary_values = (
        corpus_label or "기본 공고문",
        f"검색 {len(attempts)}회",
        f"최종 Top-{len(response.evidence)}",
        f"답변 인용 {len(cited)}개",
        f"전체 {elapsed_label}",
    )
    st.markdown(
        '<div class="trace-run-summary">'
        + "".join(f'<span class="trace-summary-chip">{html.escape(value)}</span>' for value in summary_values)
        + "</div>",
        unsafe_allow_html=True,
    )

    answer_copy = response.answer
    answer_label = "근거가 있는 답변" if response.status == "answered" else "근거 부족 · 답변 중단"
    st.markdown(
        '<section class="trace-answer-box">'
        f'<div class="trace-answer-label">{html.escape(answer_label)}</div>'
        f'<div class="trace-answer">{_answer_html(answer_copy, response.evidence)}</div>'
        "</section>",
        unsafe_allow_html=True,
    )

    if response.final_query != response.question:
        st.markdown(
            '<div class="trace-query-note">'
            f'<strong>질문 재작성:</strong> {html.escape(response.question)} → {html.escape(response.final_query)}'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 같은 질문의 단계별 Top-k")
    st.caption(
        f"현재 표시: {attempt.get('attempt', 1)}차 검색 · “{attempt.get('query', response.final_query)}” · "
        "같은 색은 같은 Chunk입니다. 단계마다 점수 범위가 달라 원점수끼리는 직접 비교하지 않습니다."
    )
    st.markdown(_topk_html(response, attempt), unsafe_allow_html=True)

    with st.expander("인용 근거 원문 확인"):
        _render_verification_view(response)

    with st.expander("원점수와 실행 정보"):
        st.caption(f"실행 ID · {trace_id} · {' → '.join(response.steps)}")
        st.caption(f"LangGraph 판단 · {response.decision_reason}")
        st.dataframe(_raw_rows(response, attempt), hide_index=True, use_container_width=True)


__all__ = [
    "apply_trace_style",
    "build_rank_flow_rows",
    "citation_numbers",
    "render_trace_header",
    "render_trace_workspace",
]
