"""Chroma 의미 검색 결과와 출처를 확인하는 Streamlit 패널."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from chunker import DocumentChunk
from vector_search import (
    DEFAULT_EMBEDDING_MODEL,
    ChromaChunkRetriever,
    EmbeddingModelUnavailableError,
    VectorSearchDependencyError,
    VectorSearchResult,
)


PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / ".chroma"


@st.cache_resource(show_spinner=False)
def _build_vector_retriever(
    chunks: tuple[DocumentChunk, ...],
    model_name: str,
    persist_directory: str,
) -> ChromaChunkRetriever:
    return ChromaChunkRetriever(
        list(chunks),
        model_name=model_name,
        persist_directory=persist_directory,
    )


def render_vector_search(
    chunks: list[DocumentChunk],
    *,
    key_prefix: str = "document_vector",
) -> list[VectorSearchResult]:
    """chunk → embedding → Chroma 색인 → 의미 검색 → 출처 표시 UI."""

    st.divider()
    st.subheader("4. Chroma 의미 검색")
    st.caption(
        "문장을 숫자 벡터로 바꿔 단어가 달라도 뜻이 가까운 chunk를 찾습니다. "
        "BM25와 결과를 합치는 RRF는 아직 사용하지 않습니다."
    )

    if not chunks:
        st.info("먼저 위에서 문서를 검색용 chunk로 나눠주세요.")
        return []

    st.markdown(f"**Embedding 모델** · `{DEFAULT_EMBEDDING_MODEL}`")
    st.caption(
        "처음 검색할 때 모델 약 500MB를 한 번 내려받을 수 있습니다. "
        "문서 벡터는 로컬 `.chroma` 폴더에 저장되어 같은 문서는 재사용합니다."
    )

    query = st.text_input(
        "의미 검색 질문",
        placeholder="예: 돈을 얼마나 받을 수 있나요?",
        key=f"{key_prefix}_query",
    )
    top_k = st.slider(
        "의미 검색 결과 수",
        min_value=1,
        max_value=min(10, len(chunks)),
        value=min(5, len(chunks)),
        key=f"{key_prefix}_top_k",
    )

    state_key = f"{key_prefix}_result"
    source_signature = _chunk_signature(chunks)

    if st.button(
        "Chroma 의미 검색",
        type="primary",
        disabled=not query.strip(),
        key=f"{key_prefix}_search",
    ):
        try:
            with st.spinner(
                "처음이면 embedding 모델을 받고, 문서 벡터를 만들어 검색합니다..."
            ):
                retriever = _build_vector_retriever(
                    tuple(chunks),
                    DEFAULT_EMBEDDING_MODEL,
                    str(PERSIST_DIRECTORY),
                )
                results = retriever.search(query, k=top_k)
        except (VectorSearchDependencyError, EmbeddingModelUnavailableError) as exc:
            st.error(str(exc))
            st.caption("`pip install -r requirements.txt` 후 다시 실행해주세요.")
        except Exception as exc:
            st.error(f"Chroma 의미 검색에 실패했습니다: {exc}")
        else:
            st.session_state[state_key] = {
                "source_signature": source_signature,
                "query": query,
                "top_k": top_k,
                "results": results,
                "collection_name": retriever.collection_name,
                "index_size": retriever.index_size,
            }

    result_state = st.session_state.get(state_key)
    if not result_state or result_state["source_signature"] != source_signature:
        return []
    if result_state["query"] != query or result_state["top_k"] != top_k:
        st.info("질문이나 결과 수가 바뀌었습니다. Chroma 의미 검색을 다시 눌러주세요.")
        return []

    results: list[VectorSearchResult] = result_state["results"]
    st.caption(
        f"Chroma collection · `{result_state['collection_name']}` · "
        f"저장된 chunk {result_state['index_size']}개"
    )
    st.caption(
        "similarity는 현재 결과의 순서를 비교하는 값입니다. "
        "정답 확률이나 답변 가능 여부를 뜻하지 않습니다."
    )

    if not results:
        st.warning("검색 결과가 없습니다.")
        return []

    query_key = hashlib.sha256(
        result_state["query"].encode("utf-8")
    ).hexdigest()[:10]

    for result in results:
        chunk = result.chunk
        with st.expander(
            (
                f"{result.rank}위 · similarity {result.similarity:.3f} · "
                f"{chunk.source_filename} · {chunk.page_label}"
            ),
            expanded=result.rank == 1,
        ):
            st.caption(
                f"cosine distance · {result.distance:.3f} · "
                f"추출 방식 · {chunk.extraction_method} · "
                f"chunk · {chunk.page_chunk_index + 1}"
            )
            st.text_area(
                "의미 검색된 Chunk",
                value=chunk.text,
                height=200,
                disabled=True,
                key=f"{key_prefix}_preview_{query_key}_{result.rank}_{chunk.id}",
            )
            st.json(chunk.metadata, expanded=False)

    payload = [result.to_dict() for result in results]
    st.download_button(
        "Chroma 검색 결과 JSON 받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="chroma-search-results.json",
        mime="application/json",
        key=f"{key_prefix}_download_{query_key}",
    )
    st.info(
        "다음 단계에서는 같은 chunk ID의 BM25 순위와 Chroma 순위를 RRF로 합칩니다."
    )
    return results


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
