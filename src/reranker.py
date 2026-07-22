"""RRF 후보를 로컬 CrossEncoder로 다시 정렬한다."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from chunker import DocumentChunk
from hybrid_search import HybridSearchResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SMALL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LOCAL_RERANKER_PROVIDER = "local"
RERANKER_PROVIDERS = (LOCAL_RERANKER_PROVIDER,)
DEFAULT_RERANK_CANDIDATES = 7
DEFAULT_RERANK_BATCH_SIZE = 4
DEFAULT_RERANK_MAX_LENGTH = 512


class RerankerError(RuntimeError):
    """후보를 안전하게 재정렬할 수 없음."""


class RerankerDependencyError(RerankerError):
    """reranker 실행에 필요한 패키지가 설치되지 않음."""


class RerankerModelUnavailableError(RerankerError):
    """reranker 모델 또는 API client를 준비하지 못함."""


class RerankerScoringError(RerankerError):
    """모델이 올바른 후보 점수를 반환하지 못함."""


class RerankerCandidateError(RerankerError):
    """RRF가 동일한 chunk를 중복 후보로 반환함."""


@dataclass(frozen=True)
class RerankerModelSpec:
    """실험을 재현하고 원격 코드를 제한하기 위한 허용 모델 정보."""

    model_name: str
    revision: str
    code_revision: str | None
    trust_remote_code: bool
    parameter_count: int
    languages: str
    license_name: str


RERANKER_MODEL_SPECS = {
    DEFAULT_RERANKER_MODEL: RerankerModelSpec(
        model_name=DEFAULT_RERANKER_MODEL,
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        code_revision=None,
        trust_remote_code=False,
        parameter_count=567_755_777,
        languages="multilingual",
        license_name="Apache-2.0",
    ),
    SMALL_RERANKER_MODEL: RerankerModelSpec(
        model_name=SMALL_RERANKER_MODEL,
        revision="1427fd652930e4ba29e8149678df786c240d8825",
        code_revision=None,
        trust_remote_code=False,
        parameter_count=117_641_603,
        languages="15 training languages; Korean zero-shot evaluation",
        license_name="Apache-2.0",
    ),
}


def get_reranker_model_spec(model_name: str) -> RerankerModelSpec:
    """검토하고 revision을 고정한 모델만 반환한다."""

    try:
        return RERANKER_MODEL_SPECS[model_name]
    except KeyError as exc:
        supported = ", ".join(RERANKER_MODEL_SPECS)
        raise RerankerModelUnavailableError(
            f"허용되지 않은 reranker 모델입니다: {model_name}. "
            f"지원 모델: {supported}"
        ) from exc


def resolve_reranker_model(
    provider: str,
    model_name: str | None = None,
) -> str:
    """provider별 기본 모델을 정하고 검토한 모델만 허용한다."""

    if provider == LOCAL_RERANKER_PROVIDER:
        resolved = model_name or DEFAULT_RERANKER_MODEL
        get_reranker_model_spec(resolved)
        return resolved
    raise RerankerModelUnavailableError(
        f"지원하지 않는 reranker provider입니다: {provider}. "
        f"지원 provider: {', '.join(RERANKER_PROVIDERS)}"
    )


def _current_process_rss_mb() -> float | None:
    """현재 프로세스 RSS를 반환하며 측정 의존성이 없으면 None을 반환한다."""

    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


class CandidateRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[HybridSearchResult]:
        """RRF 후보를 반환한다."""


class PairScorer(Protocol):
    model_name: str

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """질문과 각 본문의 관련성 점수를 반환한다."""


@dataclass(frozen=True)
class RerankResult:
    """reranker 순위와 이전 RRF 근거를 모두 보존한 결과."""

    rank: int
    reranker_score: float
    rrf_result: HybridSearchResult
    model_name: str

    @property
    def chunk(self) -> DocumentChunk:
        return self.rrf_result.chunk

    @property
    def rrf_rank(self) -> int:
        return self.rrf_result.rank

    @property
    def rank_change(self) -> int:
        """양수면 RRF보다 위로 올라간 칸 수."""

        return self.rrf_rank - self.rank

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "reranker_score": self.reranker_score,
            "model_name": self.model_name,
            "rrf_rank": self.rrf_rank,
            "rank_change": self.rank_change,
            "rrf": self.rrf_result.to_dict(),
        }


@lru_cache(maxsize=2)
def load_cross_encoder(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    max_length: int = DEFAULT_RERANK_MAX_LENGTH,
    device: str = "cpu",
) -> Any:
    """신뢰할 수 있는 Hub 모델을 로컬 CrossEncoder로 한 번만 불러온다."""

    if max_length < 1:
        raise ValueError("max_length는 1 이상이어야 합니다.")
    spec = get_reranker_model_spec(model_name)

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankerDependencyError(
            "sentence-transformers가 없습니다. requirements.txt를 설치해주세요."
        ) from exc

    try:
        code_kwargs = (
            {"code_revision": spec.code_revision}
            if spec.code_revision is not None
            else None
        )
        return CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            model_kwargs=code_kwargs,
            config_kwargs=code_kwargs,
        )
    except Exception as exc:
        raise RerankerModelUnavailableError(
            f"reranker 모델을 불러오지 못했습니다: {model_name} ({exc})"
        ) from exc


class SentenceTransformersCrossEncoderScorer:
    """Sentence Transformers CrossEncoder를 작은 점수 인터페이스로 감싼다."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model: Any | None = None,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        if max_length < 1:
            raise ValueError("max_length는 1 이상이어야 합니다.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model_load_seconds: float | None = None
        self.model_rss_delta_mb: float | None = None
        self.process_rss_after_load_mb: float | None = None
        self.peak_process_rss_mb: float | None = None

        if model is not None:
            self.model = model
        else:
            rss_before = _current_process_rss_mb()
            started = time.perf_counter()
            self.model = load_cross_encoder(
                model_name,
                max_length=max_length,
                device=device,
            )
            self.model_load_seconds = time.perf_counter() - started
            rss_after = _current_process_rss_mb()
            self.process_rss_after_load_mb = rss_after
            self.peak_process_rss_mb = rss_after
            if rss_before is not None and rss_after is not None:
                self.model_rss_delta_mb = max(0.0, rss_after - rss_before)

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if not query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")

        pairs = [(query, passage) for passage in passages]
        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerScoringError(
                f"CrossEncoder 점수 계산에 실패했습니다: {exc}"
            ) from exc
        rss_after_predict = _current_process_rss_mb()
        if rss_after_predict is not None:
            self.peak_process_rss_mb = max(
                self.peak_process_rss_mb or 0.0,
                rss_after_predict,
            )

        try:
            scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RerankerScoringError(
                "CrossEncoder 점수를 숫자로 변환할 수 없습니다."
            ) from exc

        if len(scores) != len(passages):
            raise RerankerScoringError(
                "CrossEncoder가 후보 수와 다른 개수의 점수를 반환했습니다."
            )
        if not all(math.isfinite(float(score)) for score in scores):
            raise RerankerScoringError(
                "CrossEncoder가 NaN 또는 무한대 점수를 반환했습니다."
            )
        return [float(score) for score in scores]


class CrossEncoderReranker:
    """RRF 상위 후보를 PairScorer로 읽어 관련성 순서로 재정렬한다."""

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        scorer: PairScorer,
        *,
        candidate_k: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k는 1 이상이어야 합니다.")

        self.candidate_retriever = candidate_retriever
        self.scorer = scorer
        self.candidate_k = candidate_k

    def search(self, query: str, k: int = 5) -> list[RerankResult]:
        if k <= 0 or not query.strip():
            return []

        requested_k = max(k, self.candidate_k)
        candidates = self.candidate_retriever.search(
            query,
            k=requested_k,
        )[:requested_k]
        if not candidates:
            return []

        chunk_ids = [candidate.chunk.id for candidate in candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RerankerCandidateError(
                "RRF 후보의 chunk ID가 중복되어 안전하게 재정렬할 수 없습니다."
            )

        raw_scores = self.scorer.score_pairs(
            query,
            [candidate.chunk.text for candidate in candidates],
        )
        if len(raw_scores) != len(candidates):
            raise RerankerScoringError(
                "reranker 점수 수와 RRF 후보 수가 일치하지 않습니다."
            )
        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError) as exc:
            raise RerankerScoringError(
                "reranker 점수를 숫자로 변환할 수 없습니다."
            ) from exc
        if not all(math.isfinite(score) for score in scores):
            raise RerankerScoringError(
                "reranker가 NaN 또는 무한대 점수를 반환했습니다."
            )

        scored_candidates = list(zip(candidates, scores, strict=True))
        ordered = sorted(
            scored_candidates,
            key=lambda item: (
                -item[1],
                item[0].rank,
                item[0].chunk.chunk_index,
                item[0].chunk.id,
            ),
        )[:k]

        return [
            RerankResult(
                rank=rank,
                reranker_score=score,
                rrf_result=candidate,
                model_name=self.scorer.model_name,
            )
            for rank, (candidate, score) in enumerate(ordered, start=1)
        ]

    def retrieve_texts(self, query: str, k: int = 5) -> list[str]:
        """기존 Hit@k 평가 함수와 연결할 수 있는 어댑터."""

        return [result.chunk.text for result in self.search(query, k=k)]
