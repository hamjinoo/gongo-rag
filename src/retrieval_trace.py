"""한 번의 검색에서 BM25·벡터·RRF·BGE Top-k를 모두 기록한다."""

from __future__ import annotations

import time
from typing import Any


class RetrievalTraceError(RuntimeError):
    """단계별 검색기를 추적 가능한 구조로 연결할 수 없음."""


class _RecordingRetriever:
    """기존 검색기의 결과와 실행 시간을 바꾸지 않고 기록한다."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.results: tuple[Any, ...] = ()
        self.elapsed_ms = 0.0

    def reset(self) -> None:
        self.results = ()
        self.elapsed_ms = 0.0

    def search(self, query: str, k: int = 5) -> list[Any]:
        started_at = time.perf_counter()
        results = self.retriever.search(query, k=k)
        self.elapsed_ms = (time.perf_counter() - started_at) * 1000
        self.results = tuple(results)
        return results


class TracedReranker:
    """최종 검색 결과는 그대로 반환하면서 네 단계의 후보를 남긴다."""

    def __init__(
        self,
        reranker: Any,
        *,
        bm25: _RecordingRetriever,
        vector: _RecordingRetriever,
        hybrid: _RecordingRetriever,
    ) -> None:
        self.reranker = reranker
        self.bm25 = bm25
        self.vector = vector
        self.hybrid = hybrid
        self.attempts: list[dict[str, object]] = []

    def reset_trace(self) -> None:
        self.attempts.clear()
        self.bm25.reset()
        self.vector.reset()
        self.hybrid.reset()

    @property
    def retrieval_trace(self) -> tuple[dict[str, object], ...]:
        return tuple(self.attempts)

    def search(self, query: str, k: int = 5) -> list[Any]:
        self.bm25.reset()
        self.vector.reset()
        self.hybrid.reset()

        started_at = time.perf_counter()
        results = self.reranker.search(query, k=k)
        total_ms = (time.perf_counter() - started_at) * 1000

        rrf_only_ms = max(
            0.0,
            self.hybrid.elapsed_ms - self.bm25.elapsed_ms - self.vector.elapsed_ms,
        )
        reranker_only_ms = max(0.0, total_ms - self.hybrid.elapsed_ms)
        attempt = {
            "attempt": len(self.attempts) + 1,
            "query": query,
            "total_ms": total_ms,
            "stages": {
                "bm25": _stage_payload(
                    "BM25",
                    self.bm25.results,
                    elapsed_ms=self.bm25.elapsed_ms,
                ),
                "vector": _stage_payload(
                    "Embedding",
                    self.vector.results,
                    elapsed_ms=self.vector.elapsed_ms,
                ),
                "rrf": _stage_payload(
                    "RRF",
                    self.hybrid.results,
                    elapsed_ms=rrf_only_ms,
                ),
                "reranker": _stage_payload(
                    "BGE",
                    tuple(results),
                    elapsed_ms=reranker_only_ms,
                ),
            },
        }
        self.attempts.append(attempt)
        return results

    def retrieve_texts(self, query: str, k: int = 5) -> list[str]:
        return [result.chunk.text for result in self.search(query, k=k)]


def trace_reranker(reranker: Any) -> TracedReranker:
    """기존 잠근 reranker 내부에 기록기를 끼워 넣는다."""

    if isinstance(reranker, TracedReranker):
        return reranker

    hybrid = getattr(reranker, "candidate_retriever", None)
    if hybrid is None:
        raise RetrievalTraceError("reranker의 RRF 후보 검색기를 찾을 수 없습니다.")
    bm25_retriever = getattr(hybrid, "bm25_retriever", None)
    vector_retriever = getattr(hybrid, "vector_retriever", None)
    if bm25_retriever is None or vector_retriever is None:
        raise RetrievalTraceError("RRF 내부의 BM25 또는 벡터 검색기를 찾을 수 없습니다.")

    bm25 = _RecordingRetriever(bm25_retriever)
    vector = _RecordingRetriever(vector_retriever)
    hybrid.bm25_retriever = bm25
    hybrid.vector_retriever = vector

    recorded_hybrid = _RecordingRetriever(hybrid)
    reranker.candidate_retriever = recorded_hybrid
    return TracedReranker(
        reranker,
        bm25=bm25,
        vector=vector,
        hybrid=recorded_hybrid,
    )


def _stage_payload(
    label: str,
    results: tuple[Any, ...],
    *,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "label": label,
        "candidate_count": len(results),
        "elapsed_ms": elapsed_ms,
        "results": [_result_payload(result) for result in results],
    }


def _result_payload(result: Any) -> dict[str, object]:
    chunk = getattr(result, "chunk", None)
    if chunk is None:
        raise RetrievalTraceError("검색 결과에 chunk가 없습니다.")

    payload: dict[str, object] = {
        "rank": int(getattr(result, "rank")),
        "chunk_id": str(chunk.id),
        "text": str(chunk.text),
        "source_filename": str(chunk.source_filename),
        "page_number": int(chunk.page_number),
        "page_label": str(chunk.page_label),
    }
    for attribute in (
        "score",
        "similarity",
        "rrf_score",
        "reranker_score",
        "bm25_rank",
        "vector_rank",
        "rrf_rank",
    ):
        value = getattr(result, attribute, None)
        if value is not None:
            payload[attribute] = value

    rrf_result = getattr(result, "rrf_result", None)
    if rrf_result is not None:
        payload["rrf_rank"] = int(rrf_result.rank)
        payload["rrf_score"] = float(rrf_result.rrf_score)
        payload["bm25_rank"] = rrf_result.bm25_rank
        payload["vector_rank"] = rrf_result.vector_rank
    return payload


__all__ = [
    "RetrievalTraceError",
    "TracedReranker",
    "trace_reranker",
]
