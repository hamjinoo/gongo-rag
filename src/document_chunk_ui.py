"""추출 문서를 검색용 chunk로 나누고 확인하는 Streamlit 패널."""

from __future__ import annotations

import json
from collections import defaultdict

import streamlit as st

from chunker import (
    ChunkingConfig,
    DocumentChunk,
    chunk_documents,
    summarize_chunks,
)
from document_ingestion import ExtractedDocument


STRATEGY_LABELS = {
    "paragraph": "문단 우선 (추천)",
    "fixed": "고정 길이 (기준선)",
}


def render_document_chunking(
    documents: list[ExtractedDocument],
    *,
    key_prefix: str = "document_chunking",
) -> list[DocumentChunk]:
    """추출 결과 → chunk 생성 → 본문·metadata 미리보기 UI."""

    st.divider()
    st.subheader("2. 글자 나누기")
    st.caption(
        "긴 문서를 질문에 답하기 좋은 작은 조각으로 나눕니다. "
        "아직 ChromaDB에는 저장하지 않습니다."
    )

    if not documents:
        st.info("먼저 위에서 문서를 올리고 텍스트를 추출해주세요.")
        return []

    with st.expander("Chunk 설정", expanded=True):
        strategy = st.selectbox(
            "나누는 방법",
            options=list(STRATEGY_LABELS),
            format_func=STRATEGY_LABELS.get,
            help=(
                "문단 우선은 문단·줄·문장 경계를 먼저 찾습니다. "
                "고정 길이는 글자 수만으로 자르는 비교용 기준선입니다."
            ),
            key=f"{key_prefix}_strategy",
        )
        chunk_size = st.slider(
            "Chunk 최대 크기",
            min_value=300,
            max_value=1200,
            value=700,
            step=100,
            help="하나의 chunk에 들어갈 최대 글자 수입니다.",
            key=f"{key_prefix}_chunk_size",
        )
        overlap = st.slider(
            "앞 chunk와 겹칠 글자",
            min_value=0,
            max_value=250,
            value=120,
            step=10,
            help="경계에서 문장이 잘릴 때 앞뒤 내용을 이어주는 중복 구간입니다.",
            key=f"{key_prefix}_overlap",
        )

    state_key = f"{key_prefix}_result"
    source_signature = tuple(document.source_sha256 for document in documents)

    if st.button(
        "Chunk 만들기",
        type="primary",
        key=f"{key_prefix}_create",
    ):
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks = chunk_documents(documents, config=config, deduplicate=True)
        st.session_state[state_key] = {
            "source_signature": source_signature,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": chunks,
        }

    result_state = st.session_state.get(state_key)
    if not result_state or result_state["source_signature"] != source_signature:
        return []

    chunks: list[DocumentChunk] = result_state["chunks"]
    if not chunks:
        st.warning("만들어진 chunk가 없습니다. 추출된 글자가 있는지 확인해주세요.")
        return []

    duplicate_count = len(documents) - len(
        {document.source_sha256 for document in documents}
    )
    if duplicate_count:
        st.info(f"내용이 같은 중복 문서 {duplicate_count}개는 한 번만 나눴습니다.")

    stats = summarize_chunks(chunks)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunk 수", stats["count"])
    col2.metric("평균 크기", f"{stats['average_chars']:.0f}자")
    col3.metric("최소 크기", f"{stats['min_chars']}자")
    col4.metric("최대 크기", f"{stats['max_chars']}자")

    st.caption(
        "적용 설정 · "
        f"{STRATEGY_LABELS[result_state['strategy']]} · "
        f"최대 {result_state['chunk_size']}자 · "
        f"overlap {result_state['overlap']}자"
    )

    chunks_by_document: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.source_filename].append(chunk)

    for document_index, (filename, document_chunks) in enumerate(
        chunks_by_document.items()
    ):
        with st.expander(
            f"✂️ {filename} · {len(document_chunks)}개",
            expanded=document_index == 0,
        ):
            chunk_lookup = {chunk.id: chunk for chunk in document_chunks}
            selected_id = st.selectbox(
                "확인할 chunk",
                options=[chunk.id for chunk in document_chunks],
                format_func=lambda chunk_id: _format_chunk_option(
                    chunk_lookup[chunk_id]
                ),
                key=(
                    f"{key_prefix}_selected_{document_index}_"
                    f"{document_chunks[0].source_sha256[:8]}"
                ),
            )
            selected = chunk_lookup[selected_id]

            st.text_area(
                "Chunk 내용",
                value=selected.text,
                height=220,
                disabled=True,
                key=f"{key_prefix}_preview_{selected.id}",
            )
            st.markdown("**Metadata**")
            st.json(selected.metadata, expanded=False)

    payload = [chunk.to_dict() for chunk in chunks]
    st.download_button(
        "Chunk JSON 받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="chunks.json",
        mime="application/json",
        key=f"{key_prefix}_download",
    )
    st.info("다음 단계에서 이 chunk들을 BM25와 ChromaDB 검색에 연결합니다.")

    return chunks


def _format_chunk_option(chunk: DocumentChunk) -> str:
    return (
        f"{chunk.page_label} · chunk {chunk.page_chunk_index + 1} "
        f"· {chunk.char_count}자 · {chunk.extraction_method}"
    )
