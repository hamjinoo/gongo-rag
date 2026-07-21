"""기존 Streamlit RAG 결과를 DocLens Trace 화면으로 보여준다."""

from __future__ import annotations

import html
import re
from typing import Iterable

import streamlit as st

from rag_answer import verify_citation
from rag_workflow import RAGEvidence, RAGResponse


CITATION_PATTERN = re.compile(r"\[근거\s+(\d+)\]")


APP_STYLE = """
<style>
:root {
  --trace-ink: #172033;
  --trace-muted: #7b879b;
  --trace-line: #e7e3dc;
  --trace-blue: #3d6fd1;
  --trace-blue-soft: #eef3fc;
  --trace-green: #2e9e5b;
  --trace-warm: #fbfaf7;
}
[data-testid="stAppViewContainer"] {
  background: #fff;
  color: var(--trace-ink);
}
[data-testid="stHeader"] { background: rgba(255,255,255,.92); }
.block-container { max-width: 1180px; padding-top: 1.4rem; }
.trace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .5rem 0 1rem;
  border-bottom: 1px solid var(--trace-line);
  margin-bottom: 1rem;
}
.trace-brand { color: var(--trace-ink); font-size: 1.55rem; font-weight: 800; }
.trace-brand span { color: var(--trace-blue); }
.trace-context { color: var(--trace-muted); font-size: .92rem; }
.trace-hero {
  background: #f7f9fd;
  border: 1px solid #dfe7f4;
  border-radius: 1rem;
  padding: 2.2rem 2.3rem;
  margin: .5rem 0 1.4rem;
}
.trace-eyebrow {
  color: var(--trace-blue);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: .7rem;
}
.trace-hero h1 {
  color: var(--trace-ink);
  font-size: clamp(1.8rem, 4vw, 2.7rem);
  line-height: 1.25;
  letter-spacing: -.035em;
  margin: 0 0 .8rem;
}
.trace-hero p {
  color: #596579;
  font-size: 1.03rem;
  line-height: 1.75;
  max-width: 760px;
  margin: 0;
}
.trace-chip-row { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1.1rem; }
.trace-chip {
  color: #41506a;
  background: #fff;
  border: 1px solid #dfe5ee;
  border-radius: 999px;
  padding: .28rem .65rem;
  font-size: .78rem;
  font-weight: 650;
}
.trace-story-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .8rem;
  margin: .65rem 0 1.4rem;
}
.trace-story-card {
  border: 1px solid var(--trace-line);
  border-radius: .85rem;
  padding: 1rem 1.05rem;
  background: #fff;
}
.trace-story-number { color: var(--trace-blue); font-size: .75rem; font-weight: 800; }
.trace-story-title { color: #263147; font-weight: 800; margin: .38rem 0; }
.trace-story-copy { color: #6c7789; font-size: .86rem; line-height: 1.6; }
.trace-architecture {
  display: grid;
  grid-template-columns: repeat(7, minmax(105px, 1fr));
  gap: .5rem;
  margin: .65rem 0 1.25rem;
}
.trace-architecture-stage {
  position: relative;
  border: 1px solid #dfe5ee;
  border-radius: .75rem;
  padding: .8rem .7rem;
  background: var(--trace-warm);
  min-height: 6.5rem;
}
.trace-architecture-stage:not(:last-child)::after {
  content: "→";
  position: absolute;
  right: -.58rem;
  top: 2.45rem;
  color: #a4afc0;
  z-index: 2;
}
.trace-architecture-index { color: var(--trace-blue); font-size: .7rem; font-weight: 800; }
.trace-architecture-title { color: #2d384d; font-size: .86rem; font-weight: 800; margin: .25rem 0; }
.trace-architecture-copy { color: #7a8596; font-size: .72rem; line-height: 1.4; }
.trace-answer {
  color: #384255;
  font-size: 1.12rem;
  line-height: 1.95;
  padding: .25rem 0 .9rem;
}
.trace-citation {
  display: inline-block;
  color: var(--trace-blue) !important;
  background: var(--trace-blue-soft);
  border: 1px solid #d5e2f7;
  border-radius: .45rem;
  padding: .05rem .42rem;
  margin: 0 .12rem;
  font-size: .82em;
  font-weight: 700;
  text-decoration: none !important;
  white-space: nowrap;
}
.trace-section-label {
  color: #4c5769;
  font-size: .9rem;
  font-weight: 750;
  margin: .7rem 0 .45rem;
}
.trace-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
  gap: .8rem;
  margin-bottom: .8rem;
}
.trace-card {
  border: 1px solid #d5e2f7;
  border-radius: .8rem;
  background: #f8faff;
  padding: .95rem 1rem;
  min-height: 9.5rem;
}
.trace-card.secondary { background: #fff; border-color: var(--trace-line); }
.trace-card-title { color: #253044; font-weight: 750; margin-bottom: .4rem; }
.trace-rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.55rem;
  height: 1.55rem;
  color: #fff;
  background: var(--trace-blue);
  border-radius: .42rem;
  font-size: .8rem;
  margin-right: .38rem;
}
.trace-page { color: var(--trace-muted); font-size: .8rem; margin-left: .25rem; }
.trace-excerpt { color: #667185; font-size: .88rem; line-height: 1.55; }
.trace-document {
  background: var(--trace-warm);
  border: 1px solid var(--trace-line);
  border-radius: .85rem;
  padding: 1.5rem;
  min-height: 22rem;
  line-height: 1.85;
}
.trace-document mark {
  color: inherit;
  background: #dce8fb;
  border-bottom: 2px solid var(--trace-blue);
  padding: .08rem .12rem;
}
.trace-claim {
  border: 1px solid #d5e2f7;
  border-radius: .8rem;
  background: #f8faff;
  padding: 1rem;
  margin-bottom: .75rem;
}
.trace-pass { color: var(--trace-green); font-weight: 750; }
.trace-note { color: var(--trace-muted); font-size: .84rem; line-height: 1.55; }
.trace-stage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
  gap: .55rem;
  margin: .7rem 0 1.15rem;
}
.trace-stage {
  border: 1px solid #e2e6ed;
  border-radius: .75rem;
  background: #f8f8f7;
  padding: .75rem;
  min-height: 5.3rem;
}
.trace-stage.active { border-color: #cbdaf4; background: var(--trace-blue-soft); }
.trace-stage.pass { border-color: #c9e8d7; background: #f1faf5; }
.trace-stage.stop { border-color: #efd9b8; background: #fff8ee; }
.trace-stage-label { color: #2f3a4d; font-size: .84rem; font-weight: 800; }
.trace-stage-detail { color: #758094; font-size: .73rem; line-height: 1.45; margin-top: .35rem; }
.trace-decision-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .7rem;
  margin: .75rem 0 1rem;
}
.trace-decision-card {
  border-left: 3px solid var(--trace-blue);
  background: #f8faff;
  padding: .75rem .9rem;
}
.trace-decision-label { color: #8993a3; font-size: .72rem; font-weight: 750; }
.trace-decision-value { color: #354158; font-size: .84rem; line-height: 1.5; margin-top: .25rem; }
.trace-callout {
  border: 1px solid #dfe5ee;
  border-radius: .85rem;
  background: var(--trace-warm);
  padding: 1rem 1.1rem;
  color: #596579;
  font-size: .88rem;
  line-height: 1.65;
  margin: .75rem 0;
}
.trace-callout strong { color: #263147; }
.trace-bar-chart { margin: .75rem 0 1.25rem; }
.trace-bar-row {
  display: grid;
  grid-template-columns: 92px 1fr 54px;
  gap: .65rem;
  align-items: center;
  margin: .55rem 0;
}
.trace-bar-label { color: #586478; font-size: .8rem; font-weight: 700; }
.trace-bar-track { height: .7rem; border-radius: 999px; background: #edf0f4; overflow: hidden; }
.trace-bar-fill { height: 100%; border-radius: 999px; background: #9aabc6; }
.trace-bar-fill.selected { background: var(--trace-blue); }
.trace-bar-value { color: #3d4759; font-size: .78rem; font-weight: 750; text-align: right; }
[data-testid="stMetric"] {
  border: 1px solid var(--trace-line);
  border-radius: .8rem;
  padding: .85rem 1rem;
  background: #fff;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--trace-line); }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { border-radius: .55rem .55rem 0 0; }
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs button[role="tab"][aria-selected="true"] {
  color: var(--trace-blue) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background-color: var(--trace-blue) !important;
}
@media (max-width: 760px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .trace-header { align-items: flex-start; flex-direction: column; }
  .trace-card-grid { grid-template-columns: 1fr; }
  .trace-story-grid, .trace-decision-grid { grid-template-columns: 1fr; }
  .trace-architecture { grid-template-columns: repeat(2, 1fr); }
  .trace-architecture-stage:not(:last-child)::after { display: none; }
}
</style>
"""


def apply_trace_style() -> None:
    """앱 전체에 DocLens Trace의 색과 여백을 적용한다."""

    st.markdown(APP_STYLE, unsafe_allow_html=True)


def render_trace_header(saved_text_count: int) -> None:
    """기존 RAG 앱 위에 제품명과 현재 문서 수를 표시한다."""

    st.markdown(
        (
            '<div class="trace-header">'
            '<div class="trace-brand">DocLens <span>Trace</span></div>'
            '<div class="trace-context">'
            f'한국어 근거 추적 RAG · 저장 문서 {saved_text_count}개'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def citation_numbers(answer: str) -> tuple[int, ...]:
    """답변에 실제 등장한 근거 번호를 중복 없이 반환한다."""

    return tuple(
        dict.fromkeys(int(value) for value in CITATION_PATTERN.findall(answer))
    )


def _page_label(item: RAGEvidence) -> str:
    page_number = item.get("page_number", 0)
    return f"p.{page_number}" if page_number > 0 else item["page_label"]


def _excerpt(text: str, limit: int = 180) -> str:
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
            label = f"{rank} {item['source_filename']} {_page_label(item)}"
        rendered.append(
            f'<a class="trace-citation" href="#evidence-{rank}">'
            f"{html.escape(label)}</a>"
        )
        cursor = match.end()
    rendered.append(html.escape(answer[cursor:]))
    return "".join(rendered).replace("\n", "<br>")


def _evidence_cards(
    evidence: Iterable[RAGEvidence],
    *,
    secondary: bool = False,
) -> str:
    cards: list[str] = []
    secondary_class = " secondary" if secondary else ""
    for item in evidence:
        cards.append(
            f'<article class="trace-card{secondary_class}" '
            f'id="evidence-{item["rank"]}">'
            '<div class="trace-card-title">'
            f'<span class="trace-rank">{item["rank"]}</span>'
            f'{html.escape(item["source_filename"])}'
            f'<span class="trace-page">{html.escape(_page_label(item))}</span>'
            "</div>"
            f'<div class="trace-excerpt">“{html.escape(_excerpt(item["text"]))}”</div>'
            "</article>"
        )
    return f'<div class="trace-card-grid">{"".join(cards)}</div>'


def _claim_for_citation(answer: str, rank: int) -> str:
    marker = f"[근거 {rank}]"
    marker_index = answer.find(marker)
    if marker_index < 0:
        return answer
    left_boundaries = [
        answer.rfind(symbol, 0, marker_index) for symbol in (".", "!", "?", "\n")
    ]
    left = max(left_boundaries) + 1
    right_candidates = [
        position
        for symbol in (".", "!", "?", "\n")
        if (position := answer.find(symbol, marker_index + len(marker))) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(answer)
    claim = CITATION_PATTERN.sub("", answer[left:right]).strip()
    return claim or CITATION_PATTERN.sub("", answer).strip()


def _format_score(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _format_rank(value: object) -> str:
    return "—" if value is None else f"{int(value)}위"


def build_rank_flow_rows(response: RAGResponse) -> list[dict[str, object]]:
    """검색기별 순위가 BGE 최종 순위까지 움직인 모습을 표로 만든다."""

    rows: list[dict[str, object]] = []
    cited = set(citation_numbers(response.answer))
    for item in response.evidence:
        rrf_rank = item.get("rrf_rank")
        final_rank = item["rank"]
        if rrf_rank is None:
            change = "—"
        else:
            difference = int(rrf_rank) - final_rank
            if difference > 0:
                change = f"▲ {difference}"
            elif difference < 0:
                change = f"▼ {abs(difference)}"
            else:
                change = "유지"
        rows.append(
            {
                "근거": f"{final_rank}",
                "문서": f"{item['source_filename']} {_page_label(item)}",
                "BM25": _format_rank(item.get("bm25_rank")),
                "의미 검색": _format_rank(item.get("vector_rank")),
                "RRF": _format_rank(rrf_rank),
                "BGE 최종": f"{final_rank}위",
                "순위 변화": change,
                "답변 사용": "인용" if final_rank in cited else "후보",
            }
        )
    return rows


def _execution_pipeline_html(response: RAGResponse) -> str:
    step_counts = {
        step: response.steps.count(step)
        for step in set(response.steps)
    }
    answered = response.status == "answered"
    stages = [
        ("질문", "active", "사용자 질문을 그대로 시작"),
        (
            "하이브리드 검색",
            "active" if step_counts.get("retrieve") else "",
            f"BM25 + Chroma · {step_counts.get('retrieve', 0)}회",
        ),
        (
            "근거 판단",
            "active" if step_counts.get("assess_evidence") else "",
            f"LangGraph · {step_counts.get('assess_evidence', 0)}회",
        ),
        (
            "질문 재작성",
            "active" if step_counts.get("rewrite_query") else "",
            "검색어를 고쳐 한 번 더 검색"
            if step_counts.get("rewrite_query")
            else "필요할 때만 실행",
        ),
        (
            "답변 생성",
            "active" if step_counts.get("answer") else "",
            "검색된 근거만 사용"
            if step_counts.get("answer")
            else "근거 부족으로 실행 안 함",
        ),
        (
            "인용 검증",
            "pass" if answered else ("stop" if step_counts.get("answer") else ""),
            "인용 번호·숫자 통과"
            if answered
            else "통과하지 못한 답변은 숨김",
        ),
        (
            "최종 결과",
            "pass" if answered else "stop",
            "근거가 있는 답변" if answered else "정보 없음 · 안전한 거절",
        ),
    ]
    cards = []
    for label, status, detail in stages:
        cards.append(
            f'<div class="trace-stage {status}">'
            f'<div class="trace-stage-label">{html.escape(label)}</div>'
            f'<div class="trace-stage-detail">{html.escape(detail)}</div>'
            "</div>"
        )
    return f'<div class="trace-stage-grid">{"".join(cards)}</div>'


def _render_execution_view(response: RAGResponse) -> None:
    st.markdown("### 이번 질문은 이렇게 처리됐습니다")
    st.caption("색이 들어온 단계만 이번 실행에서 실제로 사용됐습니다.")
    st.markdown(_execution_pipeline_html(response), unsafe_allow_html=True)

    result_label = "근거가 있는 답변" if response.status == "answered" else "안전한 거절"
    st.markdown(
        (
            '<div class="trace-decision-grid">'
            '<div class="trace-decision-card">'
            '<div class="trace-decision-label">원래 질문</div>'
            f'<div class="trace-decision-value">{html.escape(response.question)}</div>'
            "</div>"
            '<div class="trace-decision-card">'
            '<div class="trace-decision-label">실제로 검색한 질문</div>'
            f'<div class="trace-decision-value">{html.escape(response.final_query)}</div>'
            "</div>"
            '<div class="trace-decision-card">'
            '<div class="trace-decision-label">LangGraph 결정</div>'
            f'<div class="trace-decision-value">{html.escape(result_label)}</div>'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 검색 순위는 어떻게 바뀌었나요?")
    st.caption(
        "작은 숫자일수록 질문과 더 관련 있다고 판단한 결과입니다. "
        "RRF 후보를 BGE가 다시 읽고 최종 순서를 정합니다."
    )
    st.dataframe(
        build_rank_flow_rows(response),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        (
            '<div class="trace-callout"><strong>왜 네 단계를 사용하나요?</strong><br>'
            "BM25는 정확히 같은 단어에 강하고, Chroma는 표현이 달라도 비슷한 뜻을 "
            "찾습니다. RRF는 서로 다른 두 순위를 안전하게 합치고, 로컬 BGE는 질문과 "
            "본문을 함께 읽어 마지막 순서를 결정합니다.</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_answer_view(response: RAGResponse) -> None:
    st.markdown(f"## {html.escape(response.question)}", unsafe_allow_html=True)
    if response.status == "answered":
        st.markdown(
            f'<div class="trace-answer">{_answer_html(response.answer, response.evidence)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("문서 안에서 충분한 근거를 찾지 못해 답변을 만들지 않았습니다.")
        st.markdown(
            f'<div class="trace-answer">{html.escape(response.answer)}</div>',
            unsafe_allow_html=True,
        )

    cited = set(citation_numbers(response.answer))
    cited_evidence = [item for item in response.evidence if item["rank"] in cited]
    additional = [item for item in response.evidence if item["rank"] not in cited]

    if cited_evidence:
        st.markdown(
            f'<div class="trace-section-label">답변에 인용된 근거 {len(cited_evidence)}건</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_evidence_cards(cited_evidence), unsafe_allow_html=True)
    if additional:
        with st.expander(f"추가로 검색된 관련 근거 {len(additional)}건"):
            st.markdown(
                _evidence_cards(additional, secondary=True),
                unsafe_allow_html=True,
            )

    with st.expander("검색 과정 자세히 보기", expanded=False):
        step_names = {
            "retrieve": "검색",
            "assess_evidence": "근거 판단",
            "rewrite_query": "질문 재작성",
            "answer": "답변 생성",
            "refuse": "안전한 거절",
        }
        st.write(" → ".join(step_names.get(step, step) for step in response.steps))
        st.caption(f"근거 판단 · {response.decision_reason}")
        if response.final_query != response.question:
            st.caption(f"재검색 질문 · {response.final_query}")
        if response.refusal_reason:
            st.caption(f"거절 이유 · {response.refusal_reason}")


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
        st.markdown(f"### {item['source_filename']} · {_page_label(item)}")
        st.caption(f"검색된 원문 구간 · chunk ID {item['chunk_id']}")
        st.markdown(
            '<div class="trace-document"><mark>'
            f'{html.escape(item["text"]).replace(chr(10), "<br>")}'
            "</mark></div>",
            unsafe_allow_html=True,
        )
    with claim_column:
        st.markdown("### 답변과 근거 연결")
        st.markdown(
            '<div class="trace-claim">'
            f'<strong>주장 {selected_rank}</strong><br><br>{html.escape(claim)}'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="trace-pass">통과 · 실제 검색 결과에 있는 근거 번호입니다.</div>',
            unsafe_allow_html=True,
        )
        if numeric_check["grounded"]:
            st.markdown(
                '<div class="trace-pass">통과 · 주장의 숫자가 이 구간에 있습니다.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "이 주장에 여러 근거가 함께 쓰였거나 숫자 확인이 더 필요합니다: "
                f"{numeric_check['missing']}"
            )
        st.markdown(
            '<p class="trace-note">현재 버전은 인용 번호와 숫자를 자동 검사합니다. '
            "문장 전체의 의미가 정확한지는 Ragas와 사람 검토 단계에서 추가로 확인합니다.</p>",
            unsafe_allow_html=True,
        )


def _render_admin_view(
    response: RAGResponse,
    *,
    elapsed_seconds: float | None,
    trace_id: str,
) -> None:
    cited = set(citation_numbers(response.answer))
    st.caption(f"실행 ID · {trace_id}")
    metrics = st.columns(4)
    metrics[0].metric(
        "전체 소요",
        f"{elapsed_seconds:.2f}s" if elapsed_seconds is not None else "측정 안 됨",
    )
    metrics[1].metric("최종 근거", f"{len(response.evidence)}개")
    metrics[2].metric("답변에 인용", f"{len(cited)}개")
    metrics[3].metric("재검색", f"{response.rewrite_count}회")

    rows = []
    for item in response.evidence:
        is_cited = item["rank"] in cited
        rows.append(
            {
                "순위": item["rank"],
                "Chunk ID": item["chunk_id"],
                "출처": f"{item['source_filename']} {_page_label(item)}",
                "BM25": _format_score(item.get("bm25_score")),
                "벡터": _format_score(item.get("vector_similarity")),
                "RRF": _format_score(item.get("rrf_score")),
                "rerank": _format_score(item.get("reranker_score")),
                "상태": "인용" if is_cited else "검색됨",
                "인용 검사": "통과" if is_cited else "—",
            }
        )
    st.markdown("### 검색 결과 상세")
    st.caption("BM25 + Chroma → RRF → 로컬 BGE reranker")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(f"실행 경로 · {' → '.join(response.steps)}")


def render_trace_workspace(
    response: RAGResponse,
    *,
    elapsed_seconds: float | None,
    trace_id: str,
) -> None:
    """한 RAG 실행 결과를 답변·검증·관리 화면으로 나누어 보여준다."""

    answer_tab, execution_tab, verify_tab, admin_tab = st.tabs(
        ["답변", "실행 과정", "근거 검증", "검색 상세"]
    )
    with answer_tab:
        _render_answer_view(response)
    with execution_tab:
        _render_execution_view(response)
    with verify_tab:
        _render_verification_view(response)
    with admin_tab:
        _render_admin_view(
            response,
            elapsed_seconds=elapsed_seconds,
            trace_id=trace_id,
        )


__all__ = [
    "apply_trace_style",
    "build_rank_flow_rows",
    "citation_numbers",
    "render_trace_header",
    "render_trace_workspace",
]
