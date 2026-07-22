"""Streamlit 문서 업로드 패널의 최소 사용자 흐름 테스트."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from rag_workflow import RAGResponse  # noqa: E402
from rag_trace_ui import build_rank_flow_rows  # noqa: E402


def open_app() -> AppTest:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20)
    app.run()
    return app


def lab_uploader(app: AppTest):
    return next(
        uploader
        for uploader in app.get("file_uploader")
        if uploader.label == "PDF, DOCX, 이미지 또는 텍스트 파일을 올려주세요."
    )


def test_upload_screen_renders():
    app = open_app()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "RAG 실행",
        "평가",
        "세부 실험",
    ]
    assert len(app.get("file_uploader")) == 2
    assert any(uploader.label == "검색할 문서 (선택)" for uploader in app.get("file_uploader"))
    assert any(button.label == "텍스트 추출" for button in app.button)
    assert any(button.label == "전체 RAG 실행" for button in app.button)


def test_saved_rag_response_renders_inside_existing_streamlit_app():
    app = open_app()
    evidence = {
        "rank": 1,
        "chunk_id": "chunk-demo-1",
        "text": "사업화 지원 금액은 최대 1억원입니다.",
        "source_filename": "지원사업 공고문.pdf",
        "page_number": 3,
        "page_label": "페이지 3",
        "score": 0.93,
        "bm25_rank": 2,
        "bm25_score": 8.1,
        "vector_rank": 1,
        "vector_similarity": 0.88,
        "rrf_rank": 1,
        "rrf_score": 0.0325,
        "reranker_score": 0.93,
    }
    response = RAGResponse(
        question="지원 금액은 얼마인가요?",
        final_query="지원 금액은 얼마인가요?",
        answer="사업화 지원 금액은 최대 1억원입니다. [근거 1]",
        status="answered",
        evidence=(evidence,),
        rewrite_count=0,
        steps=("retrieve", "assess_evidence", "answer"),
        decision_reason="지원 금액이 원문에 직접 있습니다.",
        refusal_reason=None,
    )
    app.session_state["rag_response"] = response
    app.session_state["rag_elapsed_seconds"] = 1.23
    app.session_state["rag_trace_id"] = "q_test"
    app.session_state["rag_document_prep_ms"] = 120.0
    app.session_state["rag_document_count"] = 1
    app.session_state["rag_chunk_count"] = 20
    app.session_state["rag_document_summaries"] = [
        {
            "filename": "지원사업 공고문.pdf",
            "file_type": "pdf",
            "pages": 12,
            "chunks": 20,
            "ocr": False,
        }
    ]
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["RAG 실행", "평가", "세부 실험"]
    assert any("지원사업 공고문.pdf" in block.value for block in app.markdown)
    assert any("답변과 직접 근거" in block.value for block in app.markdown)
    assert any("최종 근거가 선택된 이유" in block.value for block in app.markdown)
    assert any("어디에서 시간이 걸렸나" in block.value for block in app.markdown)
    assert any(expander.label == "실행 추적 · 전체 단계별 Top-k" for expander in app.expander)
    assert any("BM25" in block.value and "Embedding" in block.value and "RRF" in block.value and "BGE" in block.value for block in app.markdown)
    assert any("각 단계를 가리키면 Top-3 미리보기" in block.value for block in app.caption)
    assert any("지원사업 공고문.pdf" in block.value and "Chunk 20개" in block.value for block in app.markdown)
    assert len(app.dataframe) >= 3
    assert build_rank_flow_rows(response)[0]["순위 변화"] == "유지"


def test_text_file_can_be_uploaded_and_previewed():
    app = open_app()
    expected = "지원 대상은 창업 3년 이내 기업입니다."

    lab_uploader(app).upload(
        "sample.txt",
        expected.encode("utf-8"),
        "text/plain",
    ).run()
    next(button for button in app.button if button.label == "텍스트 추출").click().run()

    assert not app.exception
    assert any(expected in area.value for area in app.text_area)
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["글자 수"] == str(len(expected))
    assert metric_values["구역 수"] == "1"
    assert metric_values["OCR"] == "미사용"
    assert any(button.label == "추출 텍스트 받기" for button in app.get("download_button"))


def test_extracted_text_can_be_chunked_and_previewed():
    app = open_app()
    text = (
        "신청 자격은 창업 3년 이내 기업입니다.\n\n"
        "지원 금액은 최대 1억원입니다.\n\n"
    ) * 30

    lab_uploader(app).upload(
        "long-sample.txt",
        text.encode("utf-8"),
        "text/plain",
    ).run()
    next(button for button in app.button if button.label == "텍스트 추출").click().run()

    assert {"텍스트 추출", "Chunk 만들기", "전체 RAG 실행"}.issubset(
        {button.label for button in app.button}
    )
    next(button for button in app.button if button.label == "Chunk 만들기").click().run()

    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert "Chunk 수" in metric_labels
    assert "평균 크기" in metric_labels
    assert any(area.label == "Chunk 내용" and area.value for area in app.text_area)
    assert len(app.json) == 1
    assert any(
        button.label == "Chunk JSON 받기"
        for button in app.get("download_button")
    )


def test_chunked_text_can_be_searched_with_bm25():
    app = open_app()
    text = (
        "신청 자격은 창업 3년 이내 기업입니다.\n\n"
        "지원 금액은 최대 1억원입니다.\n\n"
        "접수 기간은 7월 31일까지입니다.\n\n"
    ) * 20

    lab_uploader(app).upload(
        "search-sample.txt",
        text.encode("utf-8"),
        "text/plain",
    ).run()
    next(button for button in app.button if button.label == "텍스트 추출").click().run()
    next(button for button in app.button if button.label == "Chunk 만들기").click().run()

    tokenizer = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "검색 단어를 나누는 방법"
    )
    tokenizer.set_value("simple").run()
    query = next(
        text_input
        for text_input in app.text_input
        if text_input.label == "검색 질문"
    )
    query.set_value("지원 금액").run()
    next(button for button in app.button if button.label == "BM25 검색").click().run()

    assert not app.exception
    assert any(
        area.label == "검색된 Chunk" and "최대 1억원" in area.value
        for area in app.text_area
    )
    assert any(
        "질문에서 사용한 검색 단어" in markdown.value
        for markdown in app.markdown
    )
    assert any(
        button.label == "BM25 검색 결과 JSON 받기"
        for button in app.get("download_button")
    )


def test_chunked_text_can_be_searched_with_chroma_ui():
    import vector_search_ui
    from vector_search import VectorSearchResult

    class FakeVectorRetriever:
        collection_name = "gongo-ui-test"

        def __init__(self, chunks):
            self.chunks = list(chunks)
            self.index_size = len(self.chunks)

        def search(self, query: str, k: int):
            matching_chunk = next(
                chunk for chunk in self.chunks if "최대 1억원" in chunk.text
            )
            return [
                VectorSearchResult(
                    rank=1,
                    similarity=0.91,
                    distance=0.09,
                    chunk=matching_chunk,
                    model_name="test-model",
                )
            ]

    original_builder = vector_search_ui._build_vector_retriever
    vector_search_ui._build_vector_retriever = (
        lambda chunks, model_name, persist_directory: FakeVectorRetriever(chunks)
    )
    try:
        app = open_app()
        text = (
            "신청 자격은 창업 3년 이내 기업입니다.\n\n"
            "사업화 지원 금액은 최대 1억원입니다.\n\n"
            "접수 기간은 7월 31일까지입니다.\n\n"
        ) * 20

        lab_uploader(app).upload(
            "semantic-search-sample.txt",
            text.encode("utf-8"),
            "text/plain",
        ).run()
        next(
            button for button in app.button if button.label == "텍스트 추출"
        ).click().run()
        next(
            button for button in app.button if button.label == "Chunk 만들기"
        ).click().run()

        query = next(
            text_input
            for text_input in app.text_input
            if text_input.label == "의미 검색 질문"
        )
        query.set_value("돈을 얼마나 받을 수 있나요?").run()
        next(
            button for button in app.button if button.label == "Chroma 의미 검색"
        ).click().run()

        assert not app.exception
        assert any(
            area.label == "의미 검색된 Chunk" and "최대 1억원" in area.value
            for area in app.text_area
        )
        assert any(
            "gongo-ui-test" in caption.value
            for caption in app.caption
        )
        assert any(
            button.label == "Chroma 검색 결과 JSON 받기"
            for button in app.get("download_button")
        )
    finally:
        vector_search_ui._build_vector_retriever = original_builder


def test_chunked_text_can_be_searched_with_rrf_ui():
    import hybrid_search_ui
    from hybrid_search import HybridSearchResult

    class FakeHybridRetriever:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def search(self, query: str, k: int):
            matching_chunk = next(
                chunk for chunk in self.chunks if "최대 1억원" in chunk.text
            )
            return [
                HybridSearchResult(
                    rank=1,
                    rrf_score=(1 / 61) + (1 / 62),
                    chunk=matching_chunk,
                    rank_constant=60,
                    bm25_rank=1,
                    bm25_score=3.2,
                    bm25_contribution=1 / 61,
                    vector_rank=2,
                    vector_similarity=0.91,
                    vector_contribution=1 / 62,
                )
            ]

    original_builder = hybrid_search_ui._build_hybrid_retriever
    hybrid_search_ui._build_hybrid_retriever = (
        lambda chunks, rank_constant, fetch_k, persist_directory:
        FakeHybridRetriever(chunks)
    )
    try:
        app = open_app()
        text = (
            "신청 자격은 창업 3년 이내 기업입니다.\n\n"
            "사업화 지원 금액은 최대 1억원입니다.\n\n"
            "접수 기간은 7월 31일까지입니다.\n\n"
        ) * 20

        lab_uploader(app).upload(
            "hybrid-search-sample.txt",
            text.encode("utf-8"),
            "text/plain",
        ).run()
        next(
            button for button in app.button if button.label == "텍스트 추출"
        ).click().run()
        next(
            button for button in app.button if button.label == "Chunk 만들기"
        ).click().run()

        query = next(
            text_input
            for text_input in app.text_input
            if text_input.label == "통합 검색 질문"
        )
        query.set_value("돈을 얼마나 받을 수 있나요?").run()
        next(
            button for button in app.button if button.label == "RRF 통합 검색"
        ).click().run()

        assert not app.exception
        assert any(
            area.label == "RRF로 선택된 Chunk" and "최대 1억원" in area.value
            for area in app.text_area
        )
        assert "BM25 RRF 기여" in [metric.label for metric in app.metric]
        assert "Chroma RRF 기여" in [metric.label for metric in app.metric]
        assert any(
            button.label == "RRF 검색 결과 JSON 받기"
            for button in app.get("download_button")
        )
    finally:
        hybrid_search_ui._build_hybrid_retriever = original_builder


def test_rrf_candidates_can_be_reranked_with_cross_encoder_ui():
    import reranker_ui
    from hybrid_search import HybridSearchResult
    from reranker import RerankResult

    class FakeReranker:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def search(self, query: str, k: int):
            matching_chunk = next(
                chunk for chunk in self.chunks if "최대 1억원" in chunk.text
            )
            rrf_result = HybridSearchResult(
                rank=2,
                rrf_score=(1 / 62) + (1 / 63),
                chunk=matching_chunk,
                rank_constant=60,
                bm25_rank=2,
                bm25_score=2.8,
                bm25_contribution=1 / 62,
                vector_rank=3,
                vector_similarity=0.88,
                vector_contribution=1 / 63,
            )
            return [
                RerankResult(
                    rank=1,
                    reranker_score=0.97,
                    rrf_result=rrf_result,
                    model_name="test-cross-encoder",
                )
            ]

    original_builder = reranker_ui._build_reranker
    reranker_ui._build_reranker = (
        lambda chunks, rank_constant, rrf_fetch_k, rerank_candidate_k,
        model_name, batch_size, max_length, persist_directory:
        FakeReranker(chunks)
    )
    try:
        app = open_app()
        text = (
            "신청 자격은 창업 3년 이내 기업입니다.\n\n"
            "사업화 지원 금액은 최대 1억원입니다.\n\n"
            "접수 기간은 7월 31일까지입니다.\n\n"
        ) * 20

        lab_uploader(app).upload(
            "reranker-sample.txt",
            text.encode("utf-8"),
            "text/plain",
        ).run()
        next(
            button for button in app.button if button.label == "텍스트 추출"
        ).click().run()
        next(
            button for button in app.button if button.label == "Chunk 만들기"
        ).click().run()

        query = next(
            text_input
            for text_input in app.text_input
            if text_input.label == "재정렬 질문"
        )
        query.set_value("돈을 얼마나 받을 수 있나요?").run()
        next(
            button for button in app.button
            if button.label == "CrossEncoder 재정렬"
        ).click().run()

        assert not app.exception
        assert any(
            area.label == "재정렬된 Chunk" and "최대 1억원" in area.value
            for area in app.text_area
        )
        metric_labels = [metric.label for metric in app.metric]
        assert "CrossEncoder 점수" in metric_labels
        assert "이전 RRF 순위" in metric_labels
        assert "순위 변화" in metric_labels
        assert any(
            button.label == "재정렬 결과 JSON 받기"
            for button in app.get("download_button")
        )
    finally:
        reranker_ui._build_reranker = original_builder


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as error:
            print(f"  ❌ {test.__name__}: {error}")
    print(f"\n{passed}/{len(tests)} 통과")
    if passed != len(tests):
        raise SystemExit(1)
