"""한 번의 검색으로 네 단계 Top-k가 모두 기록되는지 확인한다."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import DocumentChunk  # noqa: E402
from hybrid_search import HybridRRFRetriever  # noqa: E402
from reranker import CrossEncoderReranker  # noqa: E402
from retrieval_trace import trace_reranker  # noqa: E402


def make_chunk(chunk_id: str, text: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        text=text,
        source_filename="공고문.pdf",
        source_sha256="a" * 64,
        file_type="pdf",
        page_number=index + 1,
        page_label=f"페이지 {index + 1}",
        extraction_method="native",
        chunk_index=index,
        page_chunk_index=0,
        start_char=0,
        end_char=len(text),
        strategy="paragraph",
    )


def test_traced_reranker_keeps_all_stage_results_without_duplicate_searches():
    first = make_chunk("chunk-a", "신청 자격은 창업 3년 이내 기업입니다.", 0)
    second = make_chunk("chunk-b", "지원 금액은 최대 1억원입니다.", 1)

    class FakeBM25:
        def __init__(self):
            self.calls = 0

        def search(self, query: str, k: int):
            self.calls += 1
            return [
                SimpleNamespace(rank=1, score=4.2, chunk=first),
                SimpleNamespace(rank=2, score=3.1, chunk=second),
            ][:k]

    class FakeVector:
        def __init__(self):
            self.calls = 0

        def search(self, query: str, k: int):
            self.calls += 1
            return [
                SimpleNamespace(rank=1, similarity=0.92, chunk=second),
                SimpleNamespace(rank=2, similarity=0.81, chunk=first),
            ][:k]

    class FakeScorer:
        model_name = "fake-bge"

        def score_pairs(self, query: str, passages: list[str]):
            return [0.2 if "자격" in passage else 0.9 for passage in passages]

    bm25 = FakeBM25()
    vector = FakeVector()
    hybrid = HybridRRFRetriever(bm25, vector, fetch_k=2)
    reranker = CrossEncoderReranker(hybrid, FakeScorer(), candidate_k=2)
    traced = trace_reranker(reranker)

    results = traced.search("지원 금액", k=2)
    trace = traced.retrieval_trace[0]
    stages = trace["stages"]

    assert [result.chunk.id for result in results] == ["chunk-b", "chunk-a"]
    assert bm25.calls == 1
    assert vector.calls == 1
    assert stages["bm25"]["candidate_count"] == 2
    assert stages["vector"]["candidate_count"] == 2
    assert stages["rrf"]["candidate_count"] == 2
    assert stages["reranker"]["candidate_count"] == 2
    assert stages["vector"]["results"][0]["similarity"] == 0.92
    assert stages["reranker"]["results"][0]["rrf_rank"] == 2

    traced.reset_trace()
    assert traced.retrieval_trace == ()
