"""DocumentChunk를 한국어 BM25로 검색하고 근거를 확인하는 Streamlit 패널."""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from bm25 import (
    BM25ChunkRetriever,
    KiwiUnavailableError,
    SearchResult,
    TokenizerName,
)
from chunker import DocumentChunk


TOKENIZER_LABELS = {
    "kiwi": "Kiwi 한국어 형태소 (추천)",
    "simple": "기본 단어 분리 (기준선)",
}


@st.cache_resource(show_spinner=False)
def _build_retriever(
    chunks: tuple[DocumentChunk, ...],
    tokenizer_name: TokenizerName,
) -> BM25ChunkRetriever:
    return BM25ChunkRetriever(
        list(chunks),
        tokenizer_name=tokenizer_name,
    )


def render_bm25_search(
    chunks: list[DocumentChunk],
    *,
    key_prefix: str = "document_bm25",
) -> list[SearchResult]:
    """chunk 목록 → BM25 색인 → 질문 검색 → 출처 표시 UI."""

    st.divider()
    st.subheader("3. BM25 키워드 검색")
    st.caption(
        "질문의 단어가 들어 있는 chunk를 찾습니다. "
        "아직 embedding과 ChromaDB는 사용하지 않습니다."
    )

    if not chunks:
        st.info("먼저 위에서 문서를 검색용 chunk로 나눠주세요.")
        return []

    tokenizer_name: TokenizerName = st.selectbox(
        "검색 단어를 나누는 방법",
        options=list(TOKENIZER_LABELS),
        format_func=TOKENIZER_LABELS.get,
        help=(
            "Kiwi는 '자격이/자격은'에서 조사를 떼어 같은 '자격'으로 봅니다. "
            "기본 방식은 문장부호와 공백만 사용합니다."
        ),
        key=f"{key_prefix}_tokenizer",
    )
    query = st.text_input(
        "검색 질문",
        placeholder="예: 신청 자격은 어떻게 되나요?",
        key=f"{key_prefix}_query",
    )
    top_k = st.slider(
        "가져올 결과 수",
        min_value=1,
        max_value=min(10, len(chunks)),
        value=min(5, len(chunks)),
        key=f"{key_prefix}_top_k",
    )

    state_key = f"{key_prefix}_result"
    source_signature = _chunk_signature(chunks)

    if st.button(
        "BM25 검색",
        type="primary",
        disabled=not query.strip(),
        key=f"{key_prefix}_search",
    ):
        try:
            with st.spinner("질문을 분석하고 BM25 점수를 계산하는 중..."):
                retriever = _build_retriever(tuple(chunks), tokenizer_name)
                query_tokens = retriever.analyze_query(query)
                results = retriever.search(query, k=top_k)
        except KiwiUnavailableError as exc:
            st.error(str(exc))
            st.caption("`pip install -r requirements.txt` 후 다시 실행해주세요.")
        except Exception as exc:
            st.error(f"BM25 검색에 실패했습니다: {exc}")
        else:
            st.session_state[state_key] = {
                "source_signature": source_signature,
                "query": query,
                "tokenizer_name": tokenizer_name,
                "top_k": top_k,
                "query_tokens": query_tokens,
                "results": results,
            }

    result_state = st.session_state.get(state_key)
    if not result_state or result_state["source_signature"] != source_signature:
        return []
    if (
        result_state["query"] != query
        or result_state["tokenizer_name"] != tokenizer_name
        or result_state["top_k"] != top_k
    ):
        st.info("질문이나 검색 설정이 바뀌었습니다. BM25 검색 버튼을 다시 눌러주세요.")
        return []

    results: list[SearchResult] = result_state["results"]
    tokens = result_state["query_tokens"]
    st.markdown(
        "**질문에서 사용한 검색 단어** · "
        + (" / ".join(tokens) if tokens else "검색 가능한 단어 없음")
    )
    st.caption(
        f"적용 tokenizer · {TOKENIZER_LABELS[result_state['tokenizer_name']]}"
    )

    if not results:
        st.warning("일치하는 단어가 있는 chunk를 찾지 못했습니다.")
        return []

    query_key = hashlib.sha256(
        result_state["query"].encode("utf-8")
    ).hexdigest()[:10]

    for result in results:
        chunk = result.chunk
        matched = ", ".join(result.matched_terms) or "없음"
        with st.expander(
            (
                f"{result.rank}위 · score {result.score:.3f} · "
                f"{chunk.source_filename} · {chunk.page_label}"
            ),
            expanded=result.rank == 1,
        ):
            st.caption(
                f"일치 단어 · {matched} · "
                f"추출 방식 · {chunk.extraction_method} · "
                f"chunk · {chunk.page_chunk_index + 1}"
            )
            st.text_area(
                "검색된 Chunk",
                value=chunk.text,
                height=200,
                disabled=True,
                key=f"{key_prefix}_preview_{query_key}_{result.rank}_{chunk.id}",
            )
            st.json(chunk.metadata, expanded=False)

    payload = [result.to_dict() for result in results]
    st.download_button(
        "BM25 검색 결과 JSON 받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="bm25-search-results.json",
        mime="application/json",
        key=f"{key_prefix}_download_{query_key}",
    )
    st.info(
        "다음 단계에서 같은 chunk에 embedding을 만들고 ChromaDB 검색을 추가합니다."
    )
    return results


def _chunk_signature(chunks: list[DocumentChunk]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            chunk.id,
            chunk.start_char,
            chunk.end_char,
            chunk.strategy,
            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:12],
        )
        for chunk in chunks
    )
