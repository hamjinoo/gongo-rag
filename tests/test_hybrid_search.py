"""BM25와 Chroma 결과를 결합하는 RRF 테스트."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from bm25 import SearchResult  # noqa: E402
from chunker import DocumentChunk  # noqa: E402
from hybrid_search import (  # noqa: E402
    HybridRRFRetriever,
    HybridSearchMismatchError,
    reciprocal_rank_score,
)
from vector_search import VectorSearchResult  # noqa: E402


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
        make_chunk(3, "문의처는 창업지원팀입니다.", 4),
    ]


def bm25_result(rank: int, chunk: DocumentChunk, score: float) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=score,
        chunk=chunk,
        matched_terms=("지원",),
    )


def vector_result(
    rank: int,
    chunk: DocumentChunk,
    similarity: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        rank=rank,
        similarity=similarity,
        distance=1.0 - similarity,
        chunk=chunk,
        model_name="test-model",
    )


class FakeBM25Retriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.requested_k.append(k)
        return self.results[:k]


class FakeVectorRetriever:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        self.requested_k.append(k)
        return self.results[:k]


def test_reciprocal_rank_score_matches_hand_calculation():
    assert reciprocal_rank_score(1, rank_constant=60) == 1 / 61
    assert reciprocal_rank_score(3, rank_constant=60, weight=2) == 2 / 63


def test_rrf_combines_two_independent_rankings():
    chunks = make_chunks()
    bm25 = FakeBM25Retriever(
        [
            bm25_result(1, chunks[0], 12.0),
            bm25_result(2, chunks[1], 8.0),
            bm25_result(3, chunks[2], 4.0),
        ]
    )
    vector = FakeVectorRetriever(
        [
            vector_result(1, chunks[2], 0.95),
            vector_result(2, chunks[1], 0.90),
            vector_result(3, chunks[3], 0.85),
        ]
    )
    retriever = HybridRRFRetriever(bm25, vector, fetch_k=3)

    results = retriever.search("지원받을 수 있나요?", k=4)

    assert [result.chunk.id for result in results[:2]] == [
        chunks[2].id,
        chunks[1].id,
    ]
    assert results[0].rrf_score == (1 / 63) + (1 / 61)
    assert results[0].bm25_rank == 3
    assert results[0].vector_rank == 1
    assert results[0].source_count == 2


def test_rrf_uses_ranks_not_raw_score_scale():
    chunks = make_chunks()

    def search_with_scores(
        bm25_score: float,
        vector_similarity: float,
    ) -> list[str]:
        retriever = HybridRRFRetriever(
            FakeBM25Retriever(
                [
                    bm25_result(1, chunks[0], bm25_score),
                    bm25_result(2, chunks[1], 0.0001),
                ]
            ),
            FakeVectorRetriever(
                [
                    vector_result(1, chunks[1], vector_similarity),
                    vector_result(2, chunks[0], -100.0),
                ]
            ),
            fetch_k=2,
        )
        return [result.chunk.id for result in retriever.search("질문", k=2)]

    assert search_with_scores(1_000_000, 0.1) == search_with_scores(0.01, 0.999)


def test_result_preserves_metadata_and_component_scores():
    chunk = make_chunks()[0]
    retriever = HybridRRFRetriever(
        FakeBM25Retriever([bm25_result(1, chunk, 3.2)]),
        FakeVectorRetriever([vector_result(2, chunk, 0.88)]),
    )

    result = retriever.search("지원금", k=1)[0]
    payload = result.to_dict()

    assert result.chunk.page_number == 1
    assert payload["chunk"]["metadata"]["source_filename"] == "청년창업 공고.pdf"
    assert payload["bm25"]["raw_score"] == 3.2
    assert payload["vector"]["similarity"] == 0.88


def test_one_retriever_can_supply_a_candidate_alone():
    chunk = make_chunks()[0]
    retriever = HybridRRFRetriever(
        FakeBM25Retriever([]),
        FakeVectorRetriever([vector_result(1, chunk, 0.9)]),
    )

    result = retriever.search("다른 표현", k=1)[0]

    assert result.bm25_rank is None
    assert result.vector_rank == 1
    assert result.source_count == 1


def test_duplicate_result_in_one_retriever_counts_once():
    chunk = make_chunks()[0]
    retriever = HybridRRFRetriever(
        FakeBM25Retriever(
            [
                bm25_result(1, chunk, 5.0),
                bm25_result(2, chunk, 4.0),
            ]
        ),
        FakeVectorRetriever([]),
    )

    result = retriever.search("지원", k=1)[0]

    assert result.bm25_rank == 1
    assert result.rrf_score == 1 / 61


def test_fetch_window_is_at_least_requested_result_count():
    chunks = make_chunks()
    bm25 = FakeBM25Retriever([bm25_result(1, chunks[0], 1.0)])
    vector = FakeVectorRetriever([vector_result(1, chunks[0], 0.9)])
    retriever = HybridRRFRetriever(bm25, vector, fetch_k=2)

    retriever.search("질문", k=4)

    assert bm25.requested_k == [4]
    assert vector.requested_k == [4]


def test_empty_query_and_invalid_k_return_no_results():
    retriever = HybridRRFRetriever(
        FakeBM25Retriever([]),
        FakeVectorRetriever([]),
    )

    assert retriever.search("", k=3) == []
    assert retriever.search("질문", k=0) == []


def test_invalid_settings_are_rejected():
    bm25 = FakeBM25Retriever([])
    vector = FakeVectorRetriever([])

    invalid_options = [
        {"rank_constant": 0},
        {"fetch_k": 0},
        {"bm25_weight": -1},
        {"vector_weight": float("nan")},
        {"bm25_weight": 0, "vector_weight": 0},
    ]
    for options in invalid_options:
        try:
            HybridRRFRetriever(bm25, vector, **options)
        except ValueError:
            continue
        raise AssertionError(f"잘못된 설정이 거절되지 않았습니다: {options}")


def test_same_chunk_id_with_different_source_is_rejected():
    chunk = make_chunks()[0]
    changed = replace(chunk, text="서로 다른 본문", end_char=len("서로 다른 본문"))
    retriever = HybridRRFRetriever(
        FakeBM25Retriever([bm25_result(1, chunk, 1.0)]),
        FakeVectorRetriever([vector_result(1, changed, 0.9)]),
    )

    try:
        retriever.search("질문", k=1)
    except HybridSearchMismatchError as exc:
        assert chunk.id in str(exc)
    else:
        raise AssertionError("서로 다른 원문이 같은 chunk ID로 결합됐습니다.")


def test_retrieve_texts_connects_to_existing_evaluator():
    chunk = make_chunks()[0]
    retriever = HybridRRFRetriever(
        FakeBM25Retriever([bm25_result(1, chunk, 1.0)]),
        FakeVectorRetriever([vector_result(1, chunk, 0.9)]),
    )

    assert retriever.retrieve_texts("지원금", k=1) == [chunk.text]


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
