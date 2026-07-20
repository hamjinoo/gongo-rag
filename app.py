"""문서 업로드와 근거 기반 질문·답변을 한 화면에서 확인하는 Streamlit 데모."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bm25 import BM25  # noqa: E402
from chunker import chunk_fixed  # noqa: E402
from document_chunk_ui import render_document_chunking  # noqa: E402
from document_search_ui import render_bm25_search  # noqa: E402
from document_upload_ui import render_document_upload  # noqa: E402
import rag_answer  # noqa: E402

TEXT_DIR = Path(__file__).resolve().parent / "docs" / "text"


@st.cache_resource
def build_index():
    """저장된 텍스트를 읽어 BM25 기준선 색인을 만든다."""
    chunks = []
    for f in sorted(TEXT_DIR.glob("*.txt")):
        chunks += chunk_fixed(f.read_text(encoding="utf-8"), doc_id=f.stem)
    if not chunks:
        return chunks, None
    return chunks, BM25([c["text"] for c in chunks])


st.set_page_config(page_title="gongo-rag", page_icon="📄", layout="wide")
st.title("📄 gongo-rag")
st.caption("문서를 글자로 바꾸고, 그 글에서 근거를 찾아 답하는 한국어 RAG")

chunks, bm25 = build_index()
st.sidebar.markdown(f"**색인 현황**\n\n- chunk 수: {len(chunks)}\n- 검색: BM25 (top-3)")
st.sidebar.caption(
    "업로드 문서는 첫 번째 탭의 BM25에서 메모리 검색할 수 있지만 "
    "ChromaDB에는 아직 저장되지 않습니다."
)

upload_tab, question_tab = st.tabs(["1. 문서 넣기", "2. 질문하기"])

with upload_tab:
    uploaded_documents = render_document_upload()
    uploaded_chunks = render_document_chunking(uploaded_documents)
    render_bm25_search(uploaded_chunks)

with question_tab:
    st.subheader("저장된 문서에 질문하기")
    st.caption("현재는 `docs/text` 폴더에 저장된 텍스트를 검색합니다.")

    if bm25 is None:
        st.warning("검색할 문서가 없습니다. `docs/text` 폴더에 TXT 파일을 먼저 넣어주세요.")
    else:
        question = st.text_input(
            "질문을 입력하세요",
            placeholder="예: 신청 자격이 어떻게 되나요?",
        )

        if question:
            top = bm25.search(question, k=3)
            retrieved = [chunks[i]["text"] for i, _ in top]

            with st.spinner("답변 생성 중..."):
                try:
                    answer = rag_answer.answer(question, retrieved)
                except Exception as exc:
                    answer = f"(생성 실패: {exc} — API 키를 확인해주세요.)"

            st.subheader("답변")
            st.write(answer)

            st.subheader("근거 chunk")
            for rank, (index, score) in enumerate(top, 1):
                chunk = chunks[index]
                with st.expander(f"[근거 {rank}] {chunk['id']} (score {score:.2f})"):
                    st.text(chunk["text"])
