"""Streamlit 문서 업로드 패널의 최소 사용자 흐름 테스트."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")


def open_app() -> AppTest:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20)
    app.run()
    return app


def test_upload_screen_renders():
    app = open_app()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["1. 문서 넣기", "2. 질문하기"]
    assert len(app.get("file_uploader")) == 1
    assert any(button.label == "텍스트 추출" for button in app.button)


def test_text_file_can_be_uploaded_and_previewed():
    app = open_app()
    expected = "지원 대상은 창업 3년 이내 기업입니다."

    app.get("file_uploader")[0].upload(
        "sample.txt",
        expected.encode("utf-8"),
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert any(expected in area.value for area in app.text_area)
    assert [metric.value for metric in app.metric] == [str(len(expected)), "1", "미사용"]
    assert any(button.label == "추출 텍스트 받기" for button in app.get("download_button"))


def test_extracted_text_can_be_chunked_and_previewed():
    app = open_app()
    text = (
        "신청 자격은 창업 3년 이내 기업입니다.\n\n"
        "지원 금액은 최대 1억원입니다.\n\n"
    ) * 30

    app.get("file_uploader")[0].upload(
        "long-sample.txt",
        text.encode("utf-8"),
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert [button.label for button in app.button] == ["텍스트 추출", "Chunk 만들기"]
    app.button[1].click().run()

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

    app.get("file_uploader")[0].upload(
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

        app.get("file_uploader")[0].upload(
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

        app.get("file_uploader")[0].upload(
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
