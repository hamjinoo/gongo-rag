"""문서 업로드와 LangGraph 근거 기반 답변을 한 화면에서 확인하는 Streamlit 데모."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from document_chunk_ui import render_document_chunking  # noqa: E402
from document_search_ui import render_bm25_search  # noqa: E402
from document_upload_ui import render_document_upload  # noqa: E402
from hybrid_search_ui import render_hybrid_search  # noqa: E402
from rag_workflow import RAGWorkflow, RAGWorkflowConfig  # noqa: E402
from rag_trace_ui import (  # noqa: E402
    apply_trace_style,
    render_trace_header,
    render_trace_workspace,
)
from reranker_ui import render_reranker  # noqa: E402
from run_rag_workflow import build_locked_reranker  # noqa: E402
from vector_search_ui import render_vector_search  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
TEXT_DIR = PROJECT_ROOT / "docs" / "text"
load_dotenv()


@st.cache_resource
def build_answer_workflow() -> RAGWorkflow:
    """잠근 BGE 검색기와 LangGraph를 한 번만 준비한다."""

    retriever = build_locked_reranker(
        text_dir=TEXT_DIR,
        persist_directory=PROJECT_ROOT / ".chroma" / "rag-workflow",
    )
    return RAGWorkflow(
        retriever,
        config=RAGWorkflowConfig(top_k=5, max_rewrites=1),
    )


st.set_page_config(page_title="DocLens Trace · gongo-rag", page_icon="📄", layout="wide")
apply_trace_style()

saved_text_count = len(list(TEXT_DIR.glob("*.txt")))
render_trace_header(saved_text_count)
st.sidebar.markdown(
    "**gongo-rag 파이프라인**\n\n"
    f"- 저장 문서: {saved_text_count}개\n"
    "- 검색: BM25 + Chroma → RRF\n"
    "- 재정렬: 로컬 BGE, 후보 7개\n"
    "- 제어: LangGraph, 재검색 최대 1회"
)
st.sidebar.caption(
    "업로드 문서는 첫 번째 탭에서 BM25와 Chroma를 각각 확인하고 "
    "RRF 통합 순위와 CrossEncoder 재정렬까지 비교할 수 있습니다."
)

upload_tab, question_tab = st.tabs(["1. 문서 넣기", "2. 질문하기"])

with upload_tab:
    uploaded_documents = render_document_upload()
    uploaded_chunks = render_document_chunking(uploaded_documents)
    render_bm25_search(uploaded_chunks)
    render_vector_search(uploaded_chunks)
    render_hybrid_search(uploaded_chunks)
    render_reranker(uploaded_chunks)

with question_tab:
    st.subheader("문서에 질문해 보세요")
    st.caption(
        "답변과 함께 실제로 사용한 문서 구간을 보여줍니다. 근거가 부족하면 "
        "질문을 한 번 고쳐 다시 찾고, 그래도 없으면 추측하지 않습니다."
    )

    if saved_text_count == 0:
        st.warning("검색할 문서가 없습니다. `docs/text` 폴더에 TXT 파일을 먼저 넣어주세요.")
    else:
        question = st.text_input(
            "질문을 입력하세요",
            placeholder="예: 신청 자격이 어떻게 되나요?",
            key="rag_question",
        )
        run_clicked = st.button(
            "근거를 찾아 답변하기",
            type="primary",
            disabled=not question.strip(),
        )

        if not os.getenv("OPENAI_API_KEY"):
            st.info(
                "답변·근거 판단에는 OPENAI_API_KEY가 필요합니다. "
                "`.env.example`을 `.env`로 복사하고 키를 채워주세요."
            )

        if run_clicked:
            st.session_state.pop("rag_response", None)
            st.session_state.pop("rag_elapsed_seconds", None)
            st.session_state.pop("rag_trace_id", None)
            response = None
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY를 설정한 뒤 다시 실행해주세요.")
            else:
                with st.spinner(
                    "검색 → 근거 판단 → 필요 시 재검색 → 답변을 실행합니다. "
                    "첫 실행은 모델을 불러오느라 오래 걸릴 수 있습니다."
                ):
                    try:
                        started_at = time.perf_counter()
                        response = build_answer_workflow().invoke(question)
                        st.session_state["rag_elapsed_seconds"] = (
                            time.perf_counter() - started_at
                        )
                        st.session_state["rag_trace_id"] = datetime.now().strftime(
                            "q_%Y%m%d_%H%M%S"
                        )
                        st.session_state["rag_response"] = response
                    except Exception as exc:
                        st.error(f"RAG 실행에 실패했습니다: {exc}")

        response = st.session_state.get("rag_response")
        if response is not None:
            render_trace_workspace(
                response,
                elapsed_seconds=st.session_state.get("rag_elapsed_seconds"),
                trace_id=st.session_state.get("rag_trace_id", "local-run"),
            )
