"""BM25와 Chroma 순위를 RRF로 결합해 설명하는 Streamlit 패널."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from bm25 import BM25ChunkRetriever, KiwiUnavailableError
from chunker import DocumentChunk
from hybrid_search import (
    DEFAULT_FETCH_K,
    DEFAULT_RANK_CONSTANT,
    HybridRRFRetriever,
    HybridSearchError,
    HybridSearchResult,
)
from vector_search import (
    DEFAULT_EMBEDDING_MODEL,
    ChromaChunkRetriever,
    VectorSearchError,
)


PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / ".chroma"


@st.cache_resource(show_spinner=False)
def _build_hybrid_retriever(
    chunks: tuple[DocumentChunk, ...],
    rank_constant: int,
    fetch_k: int,
    persist_directory: str,
) -> HybridRRFRetriever:
    chunk_list = list(chunks)
    bm25_retriever = BM25ChunkRetriever(
        chunk_list,
        tokenizer_name="kiwi",
    )
    vector_retriever = ChromaChunkRetriever(
        chunk_list,
        model_name=DEFAULT_EMBEDDING_MODEL,
        persist_directory=persist_directory,
    )
    return HybridRRFRetriever(
        bm25_retriever,
        vector_retriever,
        rank_constant=rank_constant,
        fetch_k=fetch_k,
    )


def render_hybrid_search(
    chunks: list[DocumentChunk],
    *,
    key_prefix: str = "document_hybrid",
) -> list[HybridSearchResult]:
    """BM25 + Chroma → RRF 통합 순위 → 기여도·출처 표시 UI."""

    st.divider()
    st.subheader("5. RRF 통합 검색")
    st.caption(
        "Kiwi BM25의 단어 순위와 Chroma의 의미 순위를 점수가 아닌 순위로 합칩니다. "
        "이 패널은 RRF 결과를 그대로 보여주며, 아래 패널에서 reranker와 비교합니다."
    )

    if not chunks:
        st.info("먼저 위에서 문서를 검색용 chunk로 나눠주세요.")
        return []

    st.markdown(
        "**결합 검색기** · Kiwi BM25 1표 + "
        f"`{DEFAULT_EMBEDDING_MODEL}` Chroma 1표"
    )

    query = st.text_input(
        "통합 검색 질문",
        placeholder="예: 어떤 회사가 얼마나 지원받을 수 있나요?",
        key=f"{key_prefix}_query",
    )
    top_k = st.slider(
        "통합 검색 결과 수",
        min_value=1,
        max_value=min(10, len(chunks)),
        value=min(5, len(chunks)),
        key=f"{key_prefix}_top_k",
    )

    with st.expander("RRF 설정", expanded=False):
        rank_constant = int(
            st.number_input(
                "Rank constant",
                min_value=1,
                max_value=200,
                value=DEFAULT_RANK_CONSTANT,
                help=(
                    "공식의 k입니다. 값이 클수록 아래 순위 후보의 차이가 완만해집니다. "
                    "현재 기준선은 60입니다."
                ),
                key=f"{key_prefix}_rank_constant",
            )
        )
        fetch_k = st.slider(
            "각 검색기에서 가져올 후보 수",
            min_value=1,
            max_value=min(50, len(chunks)),
            value=min(DEFAULT_FETCH_K, len(chunks)),
            help="최종 결과를 합치기 전에 BM25와 Chroma에서 각각 가져올 후보 창입니다.",
            key=f"{key_prefix}_fetch_k",
        )

    state_key = f"{key_prefix}_result"
    source_signature = _chunk_signature(chunks)

    if st.button(
        "RRF 통합 검색",
        type="primary",
        disabled=not query.strip(),
        key=f"{key_prefix}_search",
    ):
        try:
            with st.spinner("BM25와 Chroma를 각각 검색한 뒤 순위를 합치는 중..."):
                retriever = _build_hybrid_retriever(
                    tuple(chunks),
                    rank_constant,
                    fetch_k,
                    str(PERSIST_DIRECTORY),
                )
                results = retriever.search(query, k=top_k)
        except (KiwiUnavailableError, VectorSearchError, HybridSearchError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"RRF 통합 검색에 실패했습니다: {exc}")
        else:
            st.session_state[state_key] = {
                "source_signature": source_signature,
                "query": query,
                "top_k": top_k,
                "rank_constant": rank_constant,
                "fetch_k": fetch_k,
                "results": results,
            }

    result_state = st.session_state.get(state_key)
    if not result_state or result_state["source_signature"] != source_signature:
        return []
    if (
        result_state["query"] != query
        or result_state["top_k"] != top_k
        or result_state["rank_constant"] != rank_constant
        or result_state["fetch_k"] != fetch_k
    ):
        st.info("질문이나 RRF 설정이 바뀌었습니다. 통합 검색을 다시 눌러주세요.")
        return []

    results: list[HybridSearchResult] = result_state["results"]
    st.caption(
        f"공식 · 1 / ({rank_constant} + 순위) · "
        f"각 검색기 후보 {max(fetch_k, top_k)}개"
    )

    if not results:
        st.warning("통합할 검색 결과가 없습니다.")
        return []

    query_key = hashlib.sha256(
        result_state["query"].encode("utf-8")
    ).hexdigest()[:10]

    for result in results:
        chunk = result.chunk
        with st.expander(
            (
                f"{result.rank}위 · RRF {result.rrf_score:.6f} · "
                f"{chunk.source_filename} · {chunk.page_label}"
            ),
            expanded=result.rank == 1,
        ):
            st.caption(_component_summary(result))
            contribution_columns = st.columns(2)
            contribution_columns[0].metric(
                "BM25 RRF 기여",
                f"{result.bm25_contribution:.6f}",
            )
            contribution_columns[1].metric(
                "Chroma RRF 기여",
                f"{result.vector_contribution:.6f}",
            )
            st.text_area(
                "RRF로 선택된 Chunk",
                value=chunk.text,
                height=200,
                disabled=True,
                key=f"{key_prefix}_preview_{query_key}_{result.rank}_{chunk.id}",
            )
            st.json(chunk.metadata, expanded=False)

    payload = [result.to_dict() for result in results]
    st.download_button(
        "RRF 검색 결과 JSON 받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="rrf-search-results.json",
        mime="application/json",
        key=f"{key_prefix}_download_{query_key}",
    )
    st.info(
        "아래에서 RRF가 모은 후보를 CrossEncoder가 직접 읽고 다시 정렬합니다."
    )
    return results


def _component_summary(result: HybridSearchResult) -> str:
    components: list[str] = []
    if result.bm25_rank is not None:
        components.append(
            f"BM25 {result.bm25_rank}위"
            + (
                f" · 원점수 {result.bm25_score:.3f}"
                if result.bm25_score is not None
                else ""
            )
        )
    else:
        components.append("BM25 후보에 없음")

    if result.vector_rank is not None:
        components.append(
            f"Chroma {result.vector_rank}위"
            + (
                f" · similarity {result.vector_similarity:.3f}"
                if result.vector_similarity is not None
                else ""
            )
        )
    else:
        components.append("Chroma 후보에 없음")

    components.append(f"검색기 {result.source_count}개에서 발견")
    return " · ".join(components)


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
