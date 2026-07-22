"""RRF 후보를 CrossEncoder로 재정렬해 설명하는 Streamlit 패널."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from bm25 import KiwiUnavailableError
from chunker import DocumentChunk
from hybrid_search import (
    DEFAULT_FETCH_K,
    DEFAULT_RANK_CONSTANT,
    HybridSearchError,
)
from hybrid_search_ui import _build_hybrid_retriever, _component_summary
from reranker import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankerError,
    RerankResult,
    SentenceTransformersCrossEncoderScorer,
)
from vector_search import VectorSearchError


PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / ".chroma"


@st.cache_resource(show_spinner=False)
def _build_reranker(
    chunks: tuple[DocumentChunk, ...],
    rank_constant: int,
    rrf_fetch_k: int,
    rerank_candidate_k: int,
    model_name: str,
    batch_size: int,
    max_length: int,
    persist_directory: str,
) -> CrossEncoderReranker:
    hybrid_retriever = _build_hybrid_retriever(
        chunks,
        rank_constant,
        rrf_fetch_k,
        persist_directory,
    )
    scorer = SentenceTransformersCrossEncoderScorer(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device="cpu",
    )
    return CrossEncoderReranker(
        hybrid_retriever,
        scorer,
        candidate_k=rerank_candidate_k,
    )


def render_reranker(
    chunks: list[DocumentChunk],
    *,
    key_prefix: str = "document_reranker",
) -> list[RerankResult]:
    """BM25 + Chroma → RRF 후보 → CrossEncoder 재정렬 UI."""

    st.divider()
    st.subheader("6. CrossEncoder 재정렬")
    st.caption(
        "RRF 상위 후보만 질문과 본문을 한 쌍으로 같이 읽어 관련성이 높은 순서로 "
        "다시 정렬합니다."
    )

    if not chunks:
        st.info("먼저 위에서 문서를 검색용 chunk로 나눠주세요.")
        return []

    st.markdown(
        f"**로컬 다국어 모델** · `{DEFAULT_RERANKER_MODEL}` · CPU 실행 · API 키 불필요"
    )
    st.caption(
        "첫 실행에는 모델을 내려받고 메모리에 올리는 시간이 필요합니다. "
        "이후에는 같은 프로세스에서 모델을 재사용합니다."
    )

    query = st.text_input(
        "재정렬 질문",
        placeholder="예: 어떤 회사가 지원받을 수 있나요?",
        key=f"{key_prefix}_query",
    )
    max_output_k = min(10, len(chunks))
    top_k = st.slider(
        "최종 결과 수",
        min_value=1,
        max_value=max_output_k,
        value=min(5, max_output_k),
        key=f"{key_prefix}_top_k",
    )
    max_candidate_k = min(20, len(chunks))
    if top_k == max_candidate_k:
        rerank_candidate_k = top_k
        st.caption(
            f"CrossEncoder가 읽을 RRF 후보 수 · {rerank_candidate_k}개 "
            "(현재 chunk 수로 고정)"
        )
    else:
        rerank_candidate_k = st.slider(
            "CrossEncoder가 읽을 RRF 후보 수",
            min_value=top_k,
            max_value=max_candidate_k,
            value=max(top_k, min(DEFAULT_RERANK_CANDIDATES, max_candidate_k)),
            help=(
                "후보를 늘리면 정답을 살릴 가능성은 커지지만 질문-본문 쌍을 더 많이 "
                "계산하므로 느려집니다."
            ),
            key=f"{key_prefix}_candidate_k",
        )

    with st.expander("재정렬 고급 설정", expanded=False):
        rank_constant = int(
            st.number_input(
                "RRF rank constant",
                min_value=1,
                max_value=200,
                value=DEFAULT_RANK_CONSTANT,
                key=f"{key_prefix}_rank_constant",
            )
        )
        max_rrf_fetch_k = min(50, len(chunks))
        if rerank_candidate_k == max_rrf_fetch_k:
            rrf_fetch_k = rerank_candidate_k
            st.caption(
                f"BM25와 Chroma에서 각각 가져올 후보 수 · {rrf_fetch_k}개 "
                "(현재 chunk 수로 고정)"
            )
        else:
            rrf_fetch_k = st.slider(
                "BM25와 Chroma에서 각각 가져올 후보 수",
                min_value=rerank_candidate_k,
                max_value=max_rrf_fetch_k,
                value=max(
                    rerank_candidate_k,
                    min(DEFAULT_FETCH_K, len(chunks)),
                ),
                key=f"{key_prefix}_rrf_fetch_k",
            )
        batch_size = int(
            st.number_input(
                "CrossEncoder batch size",
                min_value=1,
                max_value=32,
                value=DEFAULT_RERANK_BATCH_SIZE,
                help="CPU 메모리가 부족하면 값을 낮춥니다.",
                key=f"{key_prefix}_batch_size",
            )
        )
        max_length = int(
            st.number_input(
                "질문 + 본문 최대 token 수",
                min_value=128,
                max_value=2048,
                step=128,
                value=DEFAULT_RERANK_MAX_LENGTH,
                help="넘는 본문은 모델 입력에서 잘립니다. 현재 chunk 기준선은 512입니다.",
                key=f"{key_prefix}_max_length",
            )
        )

    state_key = f"{key_prefix}_result"
    source_signature = _chunk_signature(chunks)

    if st.button(
        "CrossEncoder 재정렬",
        type="primary",
        disabled=not query.strip(),
        key=f"{key_prefix}_search",
    ):
        try:
            with st.spinner(
                "RRF 후보를 모은 뒤 질문과 각 후보 본문을 함께 읽는 중..."
            ):
                reranker = _build_reranker(
                    tuple(chunks),
                    rank_constant,
                    rrf_fetch_k,
                    rerank_candidate_k,
                    DEFAULT_RERANKER_MODEL,
                    batch_size,
                    max_length,
                    str(PERSIST_DIRECTORY),
                )
                results = reranker.search(query, k=top_k)
        except (
            KiwiUnavailableError,
            VectorSearchError,
            HybridSearchError,
            RerankerError,
        ) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"CrossEncoder 재정렬에 실패했습니다: {exc}")
        else:
            st.session_state[state_key] = {
                "source_signature": source_signature,
                "query": query,
                "top_k": top_k,
                "rerank_candidate_k": rerank_candidate_k,
                "rank_constant": rank_constant,
                "rrf_fetch_k": rrf_fetch_k,
                "batch_size": batch_size,
                "max_length": max_length,
                "results": results,
            }

    result_state = st.session_state.get(state_key)
    if not result_state or result_state["source_signature"] != source_signature:
        return []

    current_settings = {
        "query": query,
        "top_k": top_k,
        "rerank_candidate_k": rerank_candidate_k,
        "rank_constant": rank_constant,
        "rrf_fetch_k": rrf_fetch_k,
        "batch_size": batch_size,
        "max_length": max_length,
    }
    if any(
        result_state[name] != value
        for name, value in current_settings.items()
    ):
        st.info("질문이나 재정렬 설정이 바뀌었습니다. 재정렬 버튼을 다시 눌러주세요.")
        return []

    results: list[RerankResult] = result_state["results"]
    st.caption(
        f"RRF 후보 {max(top_k, rerank_candidate_k)}개를 읽어 최종 "
        f"{top_k}개 선택 · 입력 최대 {max_length} tokens"
    )

    if not results:
        st.warning("재정렬할 검색 결과가 없습니다.")
        return []

    query_key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:10]
    for result in results:
        chunk = result.chunk
        with st.expander(
            (
                f"{result.rank}위 · CrossEncoder {result.reranker_score:.6f} · "
                f"RRF {result.rrf_rank}위 → {result.rank}위 · "
                f"{chunk.source_filename} · {chunk.page_label}"
            ),
            expanded=result.rank == 1,
        ):
            score_columns = st.columns(3)
            score_columns[0].metric(
                "CrossEncoder 점수",
                f"{result.reranker_score:.6f}",
            )
            score_columns[1].metric("이전 RRF 순위", f"{result.rrf_rank}위")
            score_columns[2].metric(
                "순위 변화",
                _rank_change_label(result.rank_change),
            )
            st.caption(_component_summary(result.rrf_result))
            st.caption(
                "모델 점수는 정답 확률이 아니라 이 질문의 후보끼리 순서를 비교하는 값입니다."
            )
            st.text_area(
                "재정렬된 Chunk",
                value=chunk.text,
                height=220,
                disabled=True,
                key=f"{key_prefix}_preview_{query_key}_{result.rank}_{chunk.id}",
            )
            st.json(chunk.metadata, expanded=False)

    payload = [result.to_dict() for result in results]
    st.download_button(
        "재정렬 결과 JSON 받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="reranked-search-results.json",
        mime="application/json",
        key=f"{key_prefix}_download_{query_key}",
    )
    st.info(
        "dev에서 후보 7개가 Hit@1 0.85를 유지하면서 CPU 평균 지연을 약 "
        "6.28초에서 4.20초로 줄였습니다. MiniLM은 약 9.8배 빨랐지만 "
        "Hit@1이 0.70으로 떨어져 BGE를 유지했습니다. 다음은 Cohere 결정입니다."
    )
    return results


def _rank_change_label(rank_change: int) -> str:
    if rank_change > 0:
        return f"▲ {rank_change}"
    if rank_change < 0:
        return f"▼ {abs(rank_change)}"
    return "변화 없음"


def _chunk_signature(chunks: list[DocumentChunk]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            chunk.id,
            chunk.source_filename,
            chunk.source_sha256,
            chunk.page_number,
            chunk.extraction_method,
            chunk.start_char,
            chunk.end_char,
            chunk.strategy,
            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:12],
        )
        for chunk in chunks
    )
