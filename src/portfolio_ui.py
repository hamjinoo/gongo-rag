"""gongo-rag의 문제, 동작 방식, 실제 평가 결과를 포트폴리오로 보여준다."""

from __future__ import annotations

import html

import streamlit as st


DEV_RESULTS = (
    {"검색 방식": "BM25", "Hit@1": 0.700, "Hit@3": 0.950, "MRR": 0.827, "평균 ms": 0.7},
    {"검색 방식": "Chroma", "Hit@1": 0.600, "Hit@3": 0.700, "MRR": 0.662, "평균 ms": 17.9},
    {"검색 방식": "RRF", "Hit@1": 0.600, "Hit@3": 0.850, "MRR": 0.718, "평균 ms": 18.2},
    {"검색 방식": "BGE 재정렬", "Hit@1": 0.850, "Hit@3": 1.000, "MRR": 0.925, "평균 ms": 4197.2},
)

TEST_RESULTS = (
    {"검색 방식": "BM25", "Hit@1": 0.800, "Hit@3": 0.900, "MRR": 0.833},
    {"검색 방식": "Chroma", "Hit@1": 0.500, "Hit@3": 0.700, "MRR": 0.633},
    {"검색 방식": "RRF", "Hit@1": 0.800, "Hit@3": 0.900, "MRR": 0.833},
    {"검색 방식": "BGE 재정렬", "Hit@1": 0.800, "Hit@3": 0.900, "MRR": 0.850},
)


def _story_cards(cards: tuple[tuple[str, str, str], ...]) -> str:
    rendered = []
    for number, title, copy in cards:
        rendered.append(
            '<article class="trace-story-card">'
            f'<div class="trace-story-number">{html.escape(number)}</div>'
            f'<div class="trace-story-title">{html.escape(title)}</div>'
            f'<div class="trace-story-copy">{html.escape(copy)}</div>'
            "</article>"
        )
    return f'<div class="trace-story-grid">{"".join(rendered)}</div>'


def _architecture_html() -> str:
    stages = (
        ("01", "문서 입력", "PDF·DOCX·이미지 OCR"),
        ("02", "Chunk", "문단을 검색 조각으로"),
        ("03", "두 검색기", "Kiwi BM25 + Chroma"),
        ("04", "RRF", "서로 다른 순위를 결합"),
        ("05", "로컬 BGE", "후보를 다시 읽고 정렬"),
        ("06", "LangGraph", "근거 판단·재검색·거절"),
        ("07", "인용 검증", "없는 번호와 숫자 차단"),
    )
    rendered = []
    for index, title, copy in stages:
        rendered.append(
            '<div class="trace-architecture-stage">'
            f'<div class="trace-architecture-index">{index}</div>'
            f'<div class="trace-architecture-title">{html.escape(title)}</div>'
            f'<div class="trace-architecture-copy">{html.escape(copy)}</div>'
            "</div>"
        )
    return f'<div class="trace-architecture">{"".join(rendered)}</div>'


def _bar_chart_html(
    rows: tuple[dict[str, object], ...],
    metric: str,
) -> str:
    bars = []
    for row in rows:
        value = float(row[metric])
        selected = " selected" if row["검색 방식"] == "BGE 재정렬" else ""
        bars.append(
            '<div class="trace-bar-row">'
            f'<div class="trace-bar-label">{html.escape(str(row["검색 방식"]))}</div>'
            '<div class="trace-bar-track">'
            f'<div class="trace-bar-fill{selected}" style="width:{value * 100:.1f}%"></div>'
            "</div>"
            f'<div class="trace-bar-value">{value:.3f}</div>'
            "</div>"
        )
    return f'<div class="trace-bar-chart">{"".join(bars)}</div>'


def render_portfolio_overview(saved_text_count: int) -> None:
    """면접관이 프로젝트의 문제와 해결 방법을 먼저 이해하게 한다."""

    st.markdown(
        (
            '<section class="trace-hero">'
            '<div class="trace-eyebrow">홍진우 · AI Engineer Portfolio</div>'
            "<h1>근거를 찾고, 보여주고,<br>없으면 답하지 않는 한국어 RAG</h1>"
            "<p>정부 지원사업 공고문은 조건과 숫자가 많아 그럴듯한 오답이 위험합니다. "
            "DocLens Trace는 한국어 문서를 두 가지 방식으로 검색하고, 좋은 근거만 "
            "다시 골라 답변과 원문을 연결합니다.</p>"
            '<div class="trace-chip-row">'
            '<span class="trace-chip">LangChain · LangGraph</span>'
            '<span class="trace-chip">Kiwi BM25 · Chroma</span>'
            '<span class="trace-chip">RRF · Local BGE</span>'
            '<span class="trace-chip">Citation Guard</span>'
            '<span class="trace-chip">Ragas · next</span>'
            "</div></section>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 무엇을 해결했나요?")
    st.markdown(
        _story_cards(
            (
                (
                    "문제 01",
                    "같은 단어만 찾으면 놓칩니다",
                    "한국어 조사와 다른 표현을 함께 다루기 위해 BM25와 의미 검색을 같이 사용했습니다.",
                ),
                (
                    "문제 02",
                    "검색됐다고 모두 좋은 근거는 아닙니다",
                    "RRF 후보를 다국어 BGE가 질문과 함께 다시 읽어 최종 순서를 결정합니다.",
                ),
                (
                    "문제 03",
                    "LLM은 모르면 지어낼 수 있습니다",
                    "LangGraph가 재검색과 거절을 제어하고 인용 번호와 숫자를 한 번 더 검사합니다.",
                ),
            )
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 실제로 어떻게 동작하나요?")
    st.caption("각 단계는 코드와 테스트로 분리되어 있어 같은 질문으로 품질을 비교할 수 있습니다.")
    st.markdown(_architecture_html(), unsafe_allow_html=True)

    metrics = st.columns(4)
    metrics[0].metric("현재 검색 문서", f"{saved_text_count}개")
    metrics[1].metric("Dev Hit@1", "0.85", "+0.15 vs BM25")
    metrics[2].metric("Dev MRR", "0.925", "+0.098 vs BM25")
    metrics[3].metric("Test Hit@1", "0.80", "잠근 10문항 1회")
    st.caption(
        "Dev 20문항으로 설정을 고른 뒤 test 10문항은 한 번만 실행했습니다. "
        "작은 데이터셋이므로 결과를 과장하지 않고 실패 사례도 함께 공개합니다."
    )

    st.markdown("### 이 포트폴리오를 보는 순서")
    st.markdown(
        _story_cards(
            (
                ("01 · RAG 데모", "질문하고 근거 확인", "답변, 실제 실행 경로, 순위 변화와 원문 연결을 봅니다."),
                ("02 · 문서 실험실", "각 단계를 직접 비교", "추출부터 BM25, Chroma, RRF, BGE까지 하나씩 실행합니다."),
                ("03 · 평가 결과", "좋아졌는지 숫자로 확인", "기준선, 최종 설정, 속도 trade-off와 실패 사례를 확인합니다."),
            )
        ),
        unsafe_allow_html=True,
    )


def render_evaluation_portfolio() -> None:
    """실제 실험 결과와 모델 선택 근거를 과장 없이 보여준다."""

    st.markdown("## 결과가 정말 좋아졌나요?")
    st.caption(
        "같은 문서 3개, 같은 Chunk 38개, 같은 질문으로 검색 단계만 바꿔 비교했습니다."
    )

    metrics = st.columns(4)
    metrics[0].metric("Dev 질문", "20개", "설정 선택용")
    metrics[1].metric("Hit@1", "0.70 → 0.85", "+21.4%")
    metrics[2].metric("MRR", "0.827 → 0.925", "+11.9%")
    metrics[3].metric("Test 질문", "10개", "최종 1회")

    hit_column, mrr_column = st.columns(2, gap="large")
    with hit_column:
        st.markdown("#### Dev Hit@1")
        st.caption("첫 번째 결과가 정답인 질문의 비율")
        st.markdown(_bar_chart_html(DEV_RESULTS, "Hit@1"), unsafe_allow_html=True)
    with mrr_column:
        st.markdown("#### Dev MRR")
        st.caption("첫 정답이 위에 있을수록 높은 점수")
        st.markdown(_bar_chart_html(DEV_RESULTS, "MRR"), unsafe_allow_html=True)

    st.markdown("### 같은 시험지로 비교한 결과")
    st.dataframe(DEV_RESULTS, hide_index=True, use_container_width=True)
    st.caption(
        "BGE는 정확도가 가장 높지만 CPU 평균 약 4.2초로 느립니다. "
        "정확도와 지연 시간을 같은 숫자로 합치지 않고 따로 판단했습니다."
    )
    with st.expander("잠근 test 결과 10문항 보기"):
        st.dataframe(TEST_RESULTS, hide_index=True, use_container_width=True)
        st.caption(
            "Dev에서 고른 설정을 바꾸지 않고 한 번 실행했습니다. "
            "BGE 재정렬은 Hit@1 0.80, MRR 0.85였습니다."
        )

    st.markdown("### 왜 작은 모델로 바꾸지 않았나요?")
    model_columns = st.columns(3)
    model_columns[0].metric("BGE Hit@1", "0.85", "품질 우선")
    model_columns[1].metric("MiniLM Hit@1", "0.70", "-0.15")
    model_columns[2].metric("MiniLM 속도", "약 9.8배", "더 빠름")
    st.markdown(
        (
            '<div class="trace-callout"><strong>선택: 로컬 BGE 유지</strong><br>'
            "MiniLM은 훨씬 빨랐지만 숫자·이메일·신청 조건 질문에서 정답 순위가 "
            "떨어졌습니다. 이 프로젝트는 속도보다 공고문의 정확한 조건을 찾는 것이 "
            "중요해 BGE를 기본값으로 잠갔습니다.</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 실패를 숨기지 않았습니다")
    failure_copy, failure_source = st.columns([1, 1], gap="large")
    with failure_copy:
        st.markdown("#### q026 · 자동 평가는 실패, 사람 검토는 답변 가능")
        st.write(
            "골든셋은 `모집기간`이 있는 Chunk만 정답으로 표시했지만, 검색 1위에는 "
            "같은 마감 시각이 적힌 `제출기간` 근거가 있었습니다. 실제 답변은 가능하지만 "
            "단일 정답 span 평가가 놓친 false negative입니다."
        )
        st.markdown(
            '<div class="trace-callout"><strong>결정</strong><br>'
            "점수를 사후에 올리지 않고 test Hit@5 0.90을 그대로 남겼습니다. "
            "자동 metric과 사람 검토를 함께 사용해야 한다는 사례입니다.</div>",
            unsafe_allow_html=True,
        )
    with failure_source:
        st.caption("검색 1위에서 사람이 확인한 별도 유효 근거")
        st.code("제출기간 : 26. 2. 24.(화) 18:00", language="text")
        st.caption("골든셋이 표시했던 원래 정답 구간")
        st.code("모집기간 : 공고일 ~ 26. 2. 24.(화) 18:00까지", language="text")

    st.markdown("### 평가 원칙과 다음 단계")
    st.markdown(
        _story_cards(
            (
                ("DEV", "설정을 고르는 시험지", "후보 수와 모델은 dev 결과만 보고 선택했습니다."),
                ("TEST", "마지막에 한 번 확인", "test 결과를 본 뒤 검색 설정을 다시 맞추지 않았습니다."),
                ("NEXT", "Ragas + 사람 검토", "검색 점수 다음에는 답변 충실성과 인용 품질을 평가합니다."),
            )
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "상세 원본: experiments/retrieval-evaluation-dev.md · "
        "experiments/retrieval-evaluation-test.md · "
        "experiments/reranker-model-comparison-dev.md"
    )


__all__ = [
    "DEV_RESULTS",
    "TEST_RESULTS",
    "render_evaluation_portfolio",
    "render_portfolio_overview",
]
