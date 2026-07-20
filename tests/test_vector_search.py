"""LangChain Document 변환과 Chroma 의미 검색 테스트."""

import gc
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.embeddings import Embeddings  # noqa: E402

from chunker import DocumentChunk  # noqa: E402
from vector_search import (  # noqa: E402
    ChromaChunkRetriever,
    VectorIndexMismatchError,
    build_collection_name,
    chunk_to_langchain_document,
    create_embedding_model,
)


class KeywordEmbeddings(Embeddings):
    """테스트 문장의 뜻 범주를 3차원으로 바꾸는 작은 가짜 모델."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        if any(word in text for word in ("금액", "자금", "돈", "얼마")):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ("기간", "마감", "언제")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def make_chunk(index: int, text: str, page_number: int) -> DocumentChunk:
    return DocumentChunk(
        id=f"sample-p{page_number}-c{index}",
        text=text,
        source_filename="청년창업 공고.pdf",
        source_sha256="a" * 64,
        file_type="pdf",
        page_number=page_number,
        page_label=f"페이지 {page_number}",
        extraction_method="native",
        chunk_index=index,
        page_chunk_index=0,
        start_char=0,
        end_char=len(text),
        strategy="paragraph",
    )


def make_chunks() -> list[DocumentChunk]:
    return [
        make_chunk(0, "사업화 지원 금액은 최대 1억원입니다.", 1),
        make_chunk(1, "온라인 접수 기간은 7월 31일까지입니다.", 2),
        make_chunk(2, "신청 대상은 창업 3년 이내 기업입니다.", 3),
    ]


def close_chroma(*retrievers: ChromaChunkRetriever) -> None:
    """Windows 테스트 종료 시 Chroma의 파일 잠금을 해제한다."""

    systems = {
        id(retriever.vector_store._client._system): retriever.vector_store._client._system
        for retriever in retrievers
    }
    for system in systems.values():
        system.stop()

    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    gc.collect()


def test_chunk_becomes_langchain_document_with_metadata():
    chunk = make_chunks()[0]
    document = chunk_to_langchain_document(chunk)

    assert document.id == chunk.id
    assert document.page_content == chunk.text
    assert document.metadata["chunk_id"] == chunk.id
    assert document.metadata["source_filename"] == "청년창업 공고.pdf"
    assert document.metadata["page_number"] == 1


def test_semantic_search_finds_same_meaning_without_same_word():
    retriever = ChromaChunkRetriever(
        make_chunks(),
        embeddings=KeywordEmbeddings(),
        model_name="test-keyword-model",
    )
    results = retriever.search("돈을 얼마나 받을 수 있나요?", k=2)

    assert results[0].chunk.page_number == 1
    assert results[0].similarity == 1.0
    assert results[0].distance == 0.0
    assert results[0].rank == 1


def test_search_preserves_original_chunk_metadata():
    retriever = ChromaChunkRetriever(
        make_chunks(),
        embeddings=KeywordEmbeddings(),
        model_name="test-keyword-model",
    )
    result = retriever.search("접수 마감은 언제인가요?", k=1)[0]
    payload = result.to_dict()

    assert result.chunk.page_number == 2
    assert result.chunk.extraction_method == "native"
    assert payload["chunk"]["metadata"]["source_filename"] == "청년창업 공고.pdf"
    assert payload["model_name"] == "test-keyword-model"


def test_empty_query_and_invalid_k_return_no_results():
    retriever = ChromaChunkRetriever(
        make_chunks(),
        embeddings=KeywordEmbeddings(),
        model_name="test-keyword-model",
    )

    assert retriever.search("", k=3) == []
    assert retriever.search("돈", k=0) == []


def test_retrieve_texts_connects_to_existing_evaluator():
    retriever = ChromaChunkRetriever(
        make_chunks(),
        embeddings=KeywordEmbeddings(),
        model_name="test-keyword-model",
    )

    assert retriever.retrieve_texts("지원 자금", k=1) == [make_chunks()[0].text]


def test_duplicate_chunk_ids_are_rejected():
    chunk = make_chunks()[0]

    try:
        ChromaChunkRetriever(
            [chunk, chunk],
            embeddings=KeywordEmbeddings(),
            model_name="test-keyword-model",
        )
    except ValueError as exc:
        assert "중복" in str(exc)
    else:
        raise AssertionError("중복 chunk id가 거절되지 않았습니다.")


def test_persistent_collection_reuses_existing_vectors():
    chunks = make_chunks()
    with tempfile.TemporaryDirectory() as temp_directory:
        first_embeddings = KeywordEmbeddings()
        first = ChromaChunkRetriever(
            chunks,
            embeddings=first_embeddings,
            model_name="test-keyword-model",
            persist_directory=temp_directory,
            collection_name="gongo-persistence-test",
        )
        second_embeddings = KeywordEmbeddings()
        second = ChromaChunkRetriever(
            chunks,
            embeddings=second_embeddings,
            model_name="test-keyword-model",
            persist_directory=temp_directory,
            collection_name="gongo-persistence-test",
        )

        assert first.index_size == len(chunks)
        assert second.index_size == len(chunks)
        assert first_embeddings.document_calls == 1
        assert second_embeddings.document_calls == 0
        close_chroma(first, second)


def test_collection_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as temp_directory:
        chunks = make_chunks()
        retriever = ChromaChunkRetriever(
            chunks,
            embeddings=KeywordEmbeddings(),
            model_name="test-keyword-model",
            persist_directory=temp_directory,
            collection_name="gongo-mismatch-test",
        )

        try:
            changed_chunks = [
                replace(
                    chunks[0],
                    text="사업화 지원 금액이 변경됐습니다.",
                    end_char=len("사업화 지원 금액이 변경됐습니다."),
                ),
                *chunks[1:],
            ]
            ChromaChunkRetriever(
                changed_chunks,
                embeddings=KeywordEmbeddings(),
                model_name="test-keyword-model",
                persist_directory=temp_directory,
                collection_name="gongo-mismatch-test",
            )
        except VectorIndexMismatchError as exc:
            assert "다른 chunk" in str(exc)
        else:
            raise AssertionError("서로 다른 색인이 같은 collection을 재사용했습니다.")
        finally:
            close_chroma(retriever)


def test_collection_name_changes_with_model_or_content():
    chunks = make_chunks()

    first = build_collection_name(chunks, model_name="model-a")
    same = build_collection_name(chunks, model_name="model-a")
    other_model = build_collection_name(chunks, model_name="model-b")
    other_content = build_collection_name(chunks[:2], model_name="model-a")

    assert first == same
    assert first != other_model
    assert first != other_content
    assert first.startswith("gongo-")


def test_default_embedding_model_uses_e5_prompts_and_normalization():
    sentinel = object()
    create_embedding_model.cache_clear()

    with patch(
        "langchain_huggingface.HuggingFaceEmbeddings",
        return_value=sentinel,
    ) as embedding_class:
        result = create_embedding_model("intfloat/multilingual-e5-small")

    assert result is sentinel
    kwargs = embedding_class.call_args.kwargs
    assert kwargs["model"] == "intfloat/multilingual-e5-small"
    assert kwargs["encode_kwargs"]["prompt"] == "passage: "
    assert kwargs["query_encode_kwargs"]["prompt"] == "query: "
    assert kwargs["encode_kwargs"]["normalize_embeddings"] is True
    assert kwargs["query_encode_kwargs"]["normalize_embeddings"] is True
    create_embedding_model.cache_clear()


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
