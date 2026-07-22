"""문서 입력부터 단계별 Top-k와 근거 답변까지 한 번에 보여주는 Streamlit 앱."""

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
from local_llm import OllamaStatus, get_ollama_status  # noqa: E402
from portfolio_ui import render_evaluation_portfolio  # noqa: E402
from rag_demo import prepare_uploaded_corpus  # noqa: E402
from rag_workflow import RAGWorkflow, RAGWorkflowConfig  # noqa: E402
from rag_trace_ui import (  # noqa: E402
    apply_trace_style,
    render_trace_header,
    render_trace_workspace,
)
from reranker_ui import render_reranker  # noqa: E402
from run_rag_workflow import (  # noqa: E402
    build_locked_reranker,
    build_locked_reranker_for_chunks,
)
from vector_search_ui import render_vector_search  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
TEXT_DIR = PROJECT_ROOT / "docs" / "text"
load_dotenv()


@st.cache_resource(show_spinner=False)
def build_answer_workflow() -> RAGWorkflow:
    """평가로 잠근 기본 문서 검색기와 LangGraph를 한 번만 준비한다."""

    retriever = build_locked_reranker(
        text_dir=TEXT_DIR,
        persist_directory=PROJECT_ROOT / ".chroma" / "rag-workflow",
    )
    return RAGWorkflow(retriever, config=RAGWorkflowConfig(top_k=5, max_rewrites=1))


@st.cache_resource(show_spinner=False)
def build_uploaded_workflow(signature: str, _chunks: tuple[object, ...]) -> RAGWorkflow:
    """같은 업로드 문서는 다시 embedding하지 않고 준비한 검색기를 재사용한다."""

    retriever = build_locked_reranker_for_chunks(
        list(_chunks),
        persist_directory=PROJECT_ROOT / ".chroma" / "rag-demo" / signature,
    )
    return RAGWorkflow(retriever, config=RAGWorkflowConfig(top_k=5, max_rewrites=1))


@st.cache_data(ttl=5, show_spinner=False)
def load_local_llm_status() -> OllamaStatus:
    """화면을 느리게 하지 않도록 Ollama 상태를 잠깐 캐시한다."""

    return get_ollama_status(timeout_seconds=0.5)


st.set_page_config(
    page_title="DocLens Trace · gongo-rag",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_trace_style()

saved_text_count = len(list(TEXT_DIR.glob("*.txt")))
render_trace_header(saved_text_count)

run_tab, evaluation_tab, lab_tab = st.tabs(["RAG 실행", "평가", "세부 실험"])

with run_tab:
    saved_response = st.session_state.get("rag_response")
    if saved_response is not None:
        render_trace_workspace(
            saved_response,
            elapsed_seconds=st.session_state.get("rag_elapsed_seconds"),
            trace_id=st.session_state.get("rag_trace_id", "local-run"),
            corpus_label=st.session_state.get("rag_corpus_label"),
            llm_label=st.session_state.get("rag_llm_label"),
            document_prep_ms=st.session_state.get("rag_document_prep_ms"),
            document_count=st.session_state.get("rag_document_count"),
            chunk_count=st.session_state.get("rag_chunk_count"),
            document_summaries=st.session_state.get("rag_document_summaries"),
        )
        control_container = st.expander(
            "새 문서나 질문으로 다시 실행",
            expanded=False,
        )
    else:
        control_container = st.container()

    with control_container:
        st.markdown("## 문서를 넣고 질문하세요")
        st.caption(
            "파일을 올리지 않으면 준비된 공고문을 사용합니다. 실행 한 번으로 텍스트 추출, "
            "Chunk, Chroma 색인, 검색, 재정렬, 답변까지 이어집니다."
        )
        uploaded_files = st.file_uploader(
            "검색할 문서 (선택)",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key="rag_source_files",
            help="PDF·스캔 PDF·DOCX·이미지·텍스트를 여러 개 올릴 수 있습니다.",
        )
        if uploaded_files:
            st.caption(f"이번 실행: 업로드 문서 {len(uploaded_files)}개")
        else:
            st.caption(f"이번 실행: 준비된 공고문 {saved_text_count}개")

        question = st.text_input(
            "질문",
            placeholder="예: 신청 대상과 지원 금액은 어떻게 되나요?",
            key="rag_question",
        )
        is_rag_running = bool(st.session_state.get("rag_is_running", False))
        local_llm_status = load_local_llm_status()
        run_clicked = st.button(
            "전체 RAG 실행",
            type="primary",
            disabled=(
                is_rag_running
                or not local_llm_status.ready
                or not question.strip()
                or (not uploaded_files and saved_text_count == 0)
            ),
            help="실행이 끝날 때까지 다시 누를 수 없습니다." if is_rag_running else None,
        )

        if local_llm_status.ready:
            st.caption(
                f"🟢 {local_llm_status.message} · 문서와 질문을 외부 API로 보내지 않습니다."
            )
        else:
            st.caption(f"🟠 {local_llm_status.message}")
            if saved_response is not None:
                st.write("Ollama를 설치·실행한 뒤 모델을 한 번 내려받습니다.")
                st.code(f"ollama pull {local_llm_status.model}", language="powershell")
            else:
                with st.expander("로컬 LLM 준비 방법", expanded=False):
                    st.write("Ollama를 설치·실행한 뒤 모델을 한 번 내려받습니다.")
                    st.code(
                        f"ollama pull {local_llm_status.model}",
                        language="powershell",
                    )
        if run_error := st.session_state.pop("rag_run_error", None):
            st.error(str(run_error))

        if run_clicked:
            for key in (
                "rag_response",
                "rag_elapsed_seconds",
                "rag_trace_id",
                "rag_corpus_label",
                "rag_llm_label",
                "rag_document_prep_ms",
                "rag_document_count",
                "rag_chunk_count",
                "rag_document_summaries",
                "rag_run_error",
            ):
                st.session_state.pop(key, None)
            st.session_state["rag_is_running"] = True
            st.rerun()

        if is_rag_running:
            load_local_llm_status.clear()
            run_llm_status = get_ollama_status(timeout_seconds=1.0)
            if not run_llm_status.ready:
                st.session_state["rag_is_running"] = False
                st.session_state["rag_run_error"] = run_llm_status.message
                st.rerun()
            else:
                with st.spinner(
                    "문서 준비 → BM25·Embedding → RRF → BGE → LangGraph 답변을 실행합니다. "
                    "처음 실행은 로컬 모델을 불러오느라 시간이 걸릴 수 있습니다."
                ):
                    try:
                        started_at = time.perf_counter()
                        document_prep_started_at = time.perf_counter()
                        if uploaded_files:
                            prepared = prepare_uploaded_corpus(
                                [(item.name, item.getvalue()) for item in uploaded_files]
                            )
                            workflow = build_uploaded_workflow(
                                prepared.signature,
                                prepared.chunks,
                            )
                            ocr_count = sum(
                                document.used_ocr for document in prepared.documents
                            )
                            corpus_label = (
                                f"업로드 {len(prepared.documents)}개 · "
                                f"Chunk {len(prepared.chunks)}개"
                                + (f" · OCR {ocr_count}개" if ocr_count else "")
                            )
                            document_count = len(prepared.documents)
                            chunk_count = len(prepared.chunks)
                            chunk_counts: dict[str, int] = {}
                            for chunk in prepared.chunks:
                                chunk_counts[chunk.source_filename] = (
                                    chunk_counts.get(chunk.source_filename, 0) + 1
                                )
                            document_summaries = [
                                {
                                    "filename": document.filename,
                                    "file_type": document.file_type,
                                    "pages": len(document.pages),
                                    "chunks": chunk_counts.get(document.filename, 0),
                                    "ocr": document.used_ocr,
                                }
                                for document in prepared.documents
                            ]
                        else:
                            workflow = build_answer_workflow()
                            corpus_label = f"기본 공고문 {saved_text_count}개"
                            default_chunks = tuple(
                                getattr(workflow.retriever, "source_chunks", ())
                            )
                            chunk_count = len(default_chunks) or None
                            summaries_by_filename: dict[str, dict[str, object]] = {}
                            for chunk in default_chunks:
                                summary = summaries_by_filename.setdefault(
                                    chunk.source_filename,
                                    {
                                        "filename": chunk.source_filename,
                                        "file_type": Path(chunk.source_filename)
                                        .suffix.lstrip("."),
                                        "pages": 0,
                                        "chunks": 0,
                                        "ocr": False,
                                    },
                                )
                                summary["pages"] = max(
                                    int(summary["pages"]), chunk.page_number
                                )
                                summary["chunks"] = int(summary["chunks"]) + 1
                            document_summaries = list(summaries_by_filename.values())
                            if not document_summaries:
                                document_summaries = [
                                    {
                                        "filename": path.name,
                                        "file_type": path.suffix.lstrip("."),
                                        "pages": 0,
                                    }
                                    for path in sorted(TEXT_DIR.glob("*.txt"))
                                ]
                            document_count = len(document_summaries) or saved_text_count

                        document_prep_ms = (
                            time.perf_counter() - document_prep_started_at
                        ) * 1000
                        response = workflow.invoke(question)
                        st.session_state["rag_elapsed_seconds"] = (
                            time.perf_counter() - started_at
                        )
                        st.session_state["rag_trace_id"] = datetime.now().strftime(
                            "q_%Y%m%d_%H%M%S"
                        )
                        st.session_state["rag_corpus_label"] = corpus_label
                        st.session_state["rag_llm_label"] = (
                            f"로컬 LLM {run_llm_status.model}"
                        )
                        st.session_state["rag_document_prep_ms"] = document_prep_ms
                        st.session_state["rag_document_count"] = document_count
                        st.session_state["rag_chunk_count"] = chunk_count
                        st.session_state["rag_document_summaries"] = document_summaries
                        st.session_state["rag_response"] = response
                        st.session_state["rag_is_running"] = False
                        st.rerun()
                    except Exception as exc:
                        st.session_state["rag_is_running"] = False
                        st.session_state["rag_run_error"] = (
                            f"RAG 실행에 실패했습니다: {exc}"
                        )
                        st.rerun()

with evaluation_tab:
    render_evaluation_portfolio()

with lab_tab:
    st.markdown("## 단계를 하나씩 확인하는 실험실")
    st.caption("학습하거나 설정을 비교할 때만 사용합니다. 메인 RAG 실행과 같은 구성 요소입니다.")
    uploaded_documents = render_document_upload()
    uploaded_chunks = render_document_chunking(uploaded_documents)
    render_bm25_search(uploaded_chunks)
    render_vector_search(uploaded_chunks)
    render_hybrid_search(uploaded_chunks)
    render_reranker(uploaded_chunks)
