"""BM25와 Chroma 순위를 Reciprocal Rank Fusion으로 결합한다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from bm25 import SearchResult
from chunker import DocumentChunk
from vector_search import VectorSearchResult


DEFAULT_RANK_CONSTANT = 60
DEFAULT_FETCH_K = 20


class HybridSearchError(RuntimeError):
    """하이브리드 검색을 안전하게 결합할 수 없음."""


class HybridSearchMismatchError(HybridSearchError):
    """같은 chunk ID가 서로 다른 원문을 가리킴."""


class BM25Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """BM25 순위 결과를 반환한다."""


class VectorRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        """Chroma 의미 검색 순위 결과를 반환한다."""


@dataclass(frozen=True)
class HybridSearchResult:
    """RRF 점수와 각 검색기의 근거 순위를 함께 가진 결과."""

    rank: int
    rrf_score: float
    chunk: DocumentChunk
    rank_constant: int
    bm25_rank: int | None = None
    bm25_score: float | None = None
    bm25_contribution: float = 0.0
    vector_rank: int | None = None
    vector_similarity: float | None = None
    vector_contribution: float = 0.0

    @property
    def source_count(self) -> int:
        return int(self.bm25_rank is not None) + int(self.vector_rank is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "rrf_score": self.rrf_score,
            "rank_constant": self.rank_constant,
            "source_count": self.source_count,
            "bm25": {
                "rank": self.bm25_rank,
                "raw_score": self.bm25_score,
                "rrf_contribution": self.bm25_contribution,
            },
            "vector": {
                "rank": self.vector_rank,
                "similarity": self.vector_similarity,
                "rrf_contribution": self.vector_contribution,
            },
            "chunk": self.chunk.to_dict(),
        }


@dataclass
class _Candidate:
    chunk: DocumentChunk
    bm25_rank: int | None = None
    bm25_score: float | None = None
    bm25_contribution: float = 0.0
    vector_rank: int | None = None
    vector_similarity: float | None = None
    vector_contribution: float = 0.0

    @property
    def rrf_score(self) -> float:
        return self.bm25_contribution + self.vector_contribution

    @property
    def source_count(self) -> int:
        return int(self.bm25_rank is not None) + int(self.vector_rank is not None)

    @property
    def best_rank(self) -> int:
        ranks = [
            rank
            for rank in (self.bm25_rank, self.vector_rank)
            if rank is not None
        ]
        return min(ranks)


def reciprocal_rank_score(
    rank: int,
    *,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
    weight: float = 1.0,
) -> float:
    """한 검색 결과의 순위를 RRF 기여 점수로 바꾼다."""

    if rank < 1:
        raise ValueError("rank는 1 이상이어야 합니다.")
    if rank_constant < 1:
        raise ValueError("rank_constant는 1 이상이어야 합니다.")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight는 유한한 0 이상의 값이어야 합니다.")
    return weight / (rank_constant + rank)


class HybridRRFRetriever:
    """BM25와 Chroma 결과를 chunk ID 기준 RRF로 결합한다."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        *,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        fetch_k: int = DEFAULT_FETCH_K,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        if rank_constant < 1:
            raise ValueError("rank_constant는 1 이상이어야 합니다.")
        if fetch_k < 1:
            raise ValueError("fetch_k는 1 이상이어야 합니다.")
        if (
            not math.isfinite(bm25_weight)
            or not math.isfinite(vector_weight)
            or bm25_weight < 0
            or vector_weight < 0
        ):
            raise ValueError("검색기 weight는 유한한 0 이상의 값이어야 합니다.")
        if bm25_weight == 0 and vector_weight == 0:
            raise ValueError("적어도 한 검색기의 weight는 0보다 커야 합니다.")

        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rank_constant = rank_constant
        self.fetch_k = fetch_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(self, query: str, k: int = 5) -> list[HybridSearchResult]:
        if k <= 0 or not query.strip():
            return []

        candidate_k = max(k, self.fetch_k)
        bm25_results = (
            self.bm25_retriever.search(query, k=candidate_k)
            if self.bm25_weight > 0
            else []
        )
        vector_results = (
            self.vector_retriever.search(query, k=candidate_k)
            if self.vector_weight > 0
            else []
        )

        candidates: dict[str, _Candidate] = {}
        seen_bm25: set[str] = set()
        for result in bm25_results:
            chunk_id = result.chunk.id
            if chunk_id in seen_bm25:
                continue
            seen_bm25.add(chunk_id)
            candidate = self._get_candidate(candidates, result.chunk)
            candidate.bm25_rank = result.rank
            candidate.bm25_score = result.score
            candidate.bm25_contribution = reciprocal_rank_score(
                result.rank,
                rank_constant=self.rank_constant,
                weight=self.bm25_weight,
            )

        seen_vector: set[str] = set()
        for result in vector_results:
            chunk_id = result.chunk.id
            if chunk_id in seen_vector:
                continue
            seen_vector.add(chunk_id)
            candidate = self._get_candidate(candidates, result.chunk)
            candidate.vector_rank = result.rank
            candidate.vector_similarity = result.similarity
            candidate.vector_contribution = reciprocal_rank_score(
                result.rank,
                rank_constant=self.rank_constant,
                weight=self.vector_weight,
            )

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.rrf_score,
                -candidate.source_count,
                candidate.best_rank,
                candidate.chunk.chunk_index,
                candidate.chunk.id,
            ),
        )[:k]

        return [
            HybridSearchResult(
                rank=rank,
                rrf_score=candidate.rrf_score,
                chunk=candidate.chunk,
                rank_constant=self.rank_constant,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
                bm25_contribution=candidate.bm25_contribution,
                vector_rank=candidate.vector_rank,
                vector_similarity=candidate.vector_similarity,
                vector_contribution=candidate.vector_contribution,
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]

    def retrieve_texts(self, query: str, k: int = 5) -> list[str]:
        """기존 Hit@k 평가 함수와 연결할 수 있는 어댑터."""

        return [result.chunk.text for result in self.search(query, k=k)]

    @staticmethod
    def _get_candidate(
        candidates: dict[str, _Candidate],
        chunk: DocumentChunk,
    ) -> _Candidate:
        candidate = candidates.get(chunk.id)
        if candidate is None:
            candidate = _Candidate(chunk=chunk)
            candidates[chunk.id] = candidate
            return candidate

        if candidate.chunk != chunk:
            raise HybridSearchMismatchError(
                f"같은 chunk ID가 서로 다른 원문을 가리킵니다: {chunk.id}"
            )
        return candidate
