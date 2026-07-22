"""gongo-rag의 실제 평가 결과와 선택 근거를 포트폴리오로 보여준다."""

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
]
