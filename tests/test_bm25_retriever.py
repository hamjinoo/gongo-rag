"""DocumentChunk 기반 한국어 BM25 검색 테스트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from bm25 import (  # noqa: E402
    BM25,
    BM25ChunkRetriever,
    KiwiTokenizer,
    tokenize_simple,
)
from chunker import DocumentChunk  # noqa: E402


def make_chunk(
    index: int,
    text: str,
    *,
    page_number: int,
) -> DocumentChunk:
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
        make_chunk(
            0,
            "신청 자격은 창업 3년 이내 기업입니다. 대표자는 만 39세 이하여야 합니다.",
            page_number=1,
        ),
        make_chunk(
            1,
            "지원 금액은 최대 1억원이며 사업화 자금으로 사용할 수 있습니다.",
            page_number=2,
        ),
        make_chunk(
            2,
            "접수 기간은 7월 31일까지이며 온라인으로 신청합니다.",
            page_number=3,
        ),
    ]


def test_simple_tokenizer_removes_punctuation():
    assert tokenize_simple("신청 자격은? AI-기업 1억원!") == [
        "신청",
        "자격은",
        "ai",
        "기업",
        "1",
        "억원",
    ]


def test_bm25_handles_empty_corpus_and_invalid_k():
    bm25 = BM25([])
    assert bm25.search("질문", k=3) == []
    assert bm25.search("질문", k=0) == []


def test_chunk_retriever_ranks_keyword_match_first():
    retriever = BM25ChunkRetriever(make_chunks(), tokenizer_name="simple")
    results = retriever.search("지원 금액", k=2)

    assert results
    assert results[0].chunk.page_number == 2
    assert results[0].rank == 1
    assert results[0].score > 0
    assert "지원" in results[0].matched_terms


def test_chunk_retriever_preserves_metadata():
    retriever = BM25ChunkRetriever(make_chunks(), tokenizer_name="simple")
    result = retriever.search("접수 기간", k=1)[0]
    payload = result.to_dict()

    assert payload["rank"] == 1
    assert payload["chunk"]["metadata"]["source_filename"] == "청년창업 공고.pdf"
    assert payload["chunk"]["metadata"]["page_number"] == 3
    assert result.chunk.extraction_method == "native"


def test_no_matching_word_returns_no_results():
    retriever = BM25ChunkRetriever(make_chunks(), tokenizer_name="simple")

    assert retriever.search("김치찌개", k=3) == []
    assert retriever.search("   ", k=3) == []
    assert retriever.search("지원", k=0) == []


def test_retrieve_texts_connects_to_existing_evaluator():
    retriever = BM25ChunkRetriever(make_chunks(), tokenizer_name="simple")
    texts = retriever.retrieve_texts("최대 1억원", k=1)

    assert texts == [make_chunks()[1].text]


def test_kiwi_tokenizer_removes_particles_and_uses_lemmas():
    tokenizer = KiwiTokenizer()
    tokens = tokenizer("신청 자격이 어떻게 되나요?")

    assert "신청" in tokens
    assert "자격" in tokens
    assert "어떻다" in tokens
    assert "되다" in tokens
    assert "이" not in tokens


def test_kiwi_matches_korean_words_with_different_particles():
    chunks = make_chunks()
    simple = BM25ChunkRetriever(chunks, tokenizer_name="simple")
    kiwi = BM25ChunkRetriever(chunks, tokenizer_name="kiwi")

    simple_result = simple.search("신청 자격이 어떻게 되나요?", k=1)[0]
    kiwi_result = kiwi.search("신청 자격이 어떻게 되나요?", k=1)[0]

    assert "신청" in simple_result.matched_terms
    assert "자격이" not in simple_result.matched_terms
    assert "신청" in kiwi_result.matched_terms
    assert "자격" in kiwi_result.matched_terms
    assert len(kiwi_result.matched_terms) > len(simple_result.matched_terms)


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
