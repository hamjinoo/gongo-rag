"""RRF 후보를 CrossEncoder로 재정렬하는 테스트."""

import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import DocumentChunk  # noqa: E402
from hybrid_search import HybridSearchResult  # noqa: E402
from reranker import (  # noqa: E402
    COHERE_RERANKER_PROVIDER,
    DEFAULT_COHERE_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL,
    LOCAL_RERANKER_PROVIDER,
    RERANKER_MODEL_SPECS,
    SMALL_RERANKER_MODEL,
    CohereRerankScorer,
    CrossEncoderReranker,
    RerankerCandidateError,
    RerankerModelUnavailableError,
    RerankerScoringError,
    SentenceTransformersCrossEncoderScorer,
    get_reranker_model_spec,
    resolve_reranker_model,
)


def make_chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=f"sample-p{index + 1}-c{index}",
        text=text,
        source_filename="청년창업 공고.pdf",
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


def make_candidates() -> list[HybridSearchResult]:
    chunks = [
        make_chunk(0, "접수 기간은 7월 31일까지입니다."),
        make_chunk(1, "신청 대상은 창업 3년 이내 기업입니다."),
        make_chunk(2, "사업화 지원 금액은 최대 1억원입니다."),
        make_chunk(3, "문의처는 창업지원팀입니다."),
    ]
    return [
        HybridSearchResult(
            rank=rank,
            rrf_score=0.04 - (rank * 0.001),
            chunk=chunk,
            rank_constant=60,
            bm25_rank=rank,
            bm25_score=5.0 - rank,
            bm25_contribution=1 / (60 + rank),
            vector_rank=rank,
            vector_similarity=1.0 - (rank * 0.1),
            vector_contribution=1 / (60 + rank),
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]


class FakeCandidateRetriever:
    def __init__(self, candidates: list[HybridSearchResult]) -> None:
        self.candidates = candidates
        self.requested_k: list[int] = []

    def search(self, query: str, k: int = 5) -> list[HybridSearchResult]:
        self.requested_k.append(k)
        return self.candidates[:k]


class FakeScorer:
    model_name = "test-cross-encoder"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, passages))
        return self.scores[: len(passages)]


class FakeCrossEncoderModel:
    def __init__(self, scores) -> None:
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


class FakeCohereClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_reranker_uses_pair_score_instead_of_rrf_order():
    candidates = make_candidates()
    retriever = FakeCandidateRetriever(candidates)
    scorer = FakeScorer([0.1, 0.95, 0.5, 0.2])
    reranker = CrossEncoderReranker(retriever, scorer, candidate_k=4)

    results = reranker.search("누가 신청할 수 있나요?", k=3)

    assert [result.chunk.id for result in results] == [
        candidates[1].chunk.id,
        candidates[2].chunk.id,
        candidates[3].chunk.id,
    ]
    assert [result.rrf_rank for result in results] == [2, 3, 4]
    assert scorer.calls == [
        (
            "누가 신청할 수 있나요?",
            [candidate.chunk.text for candidate in candidates],
        )
    ]


def test_candidate_window_is_at_least_output_k():
    candidates = make_candidates()
    retriever = FakeCandidateRetriever(candidates)
    reranker = CrossEncoderReranker(
        retriever,
        FakeScorer([0.4, 0.3, 0.2, 0.1]),
        candidate_k=2,
    )

    reranker.search("지원", k=4)

    assert retriever.requested_k == [4]


def test_result_preserves_rrf_evidence_and_metadata():
    candidate = make_candidates()[2]
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever([candidate]),
        FakeScorer([0.88]),
        candidate_k=1,
    )

    result = reranker.search("지원금", k=1)[0]
    payload = result.to_dict()

    assert result.chunk == candidate.chunk
    assert result.model_name == "test-cross-encoder"
    assert result.rrf_rank == 3
    assert result.rank_change == 2
    assert payload["reranker_score"] == 0.88
    assert (
        payload["rrf"]["chunk"]["metadata"]["source_filename"]
        == "청년창업 공고.pdf"
    )
    assert payload["rrf"]["bm25"]["rank"] == 3


def test_equal_scores_keep_the_better_rrf_rank():
    candidates = make_candidates()
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever(candidates),
        FakeScorer([0.5, 0.5, 0.5, 0.5]),
        candidate_k=4,
    )

    results = reranker.search("질문", k=4)

    assert [result.rrf_rank for result in results] == [1, 2, 3, 4]


def test_empty_query_invalid_k_and_empty_candidates_return_nothing():
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever([]),
        FakeScorer([]),
    )

    assert reranker.search("", k=5) == []
    assert reranker.search("질문", k=0) == []
    assert reranker.search("질문", k=5) == []


def test_invalid_settings_are_rejected():
    try:
        CrossEncoderReranker(
            FakeCandidateRetriever([]),
            FakeScorer([]),
            candidate_k=0,
        )
    except ValueError as exc:
        assert "candidate_k" in str(exc)
    else:
        raise AssertionError("candidate_k=0을 거부해야 합니다.")

    for kwargs in ({"batch_size": 0}, {"max_length": 0}):
        try:
            SentenceTransformersCrossEncoderScorer(
                model=FakeCrossEncoderModel([]),
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 설정을 거부해야 합니다: {kwargs}")

    for kwargs in (
        {"max_tokens_per_doc": 0},
        {"timeout_seconds": 0},
    ):
        try:
            CohereRerankScorer(client=FakeCohereClient(None), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 Cohere 설정을 거부해야 합니다: {kwargs}")


def test_duplicate_chunk_id_is_rejected():
    candidate = make_candidates()[0]
    duplicate = replace(candidate, rank=2)
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever([candidate, duplicate]),
        FakeScorer([0.9, 0.8]),
        candidate_k=2,
    )

    try:
        reranker.search("질문", k=2)
    except RerankerCandidateError as exc:
        assert "중복" in str(exc)
    else:
        raise AssertionError("중복 chunk ID를 거부해야 합니다.")


def test_wrong_number_of_scores_is_rejected():
    candidates = make_candidates()[:2]
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever(candidates),
        FakeScorer([0.9]),
        candidate_k=2,
    )

    try:
        reranker.search("질문", k=2)
    except RerankerScoringError as exc:
        assert "일치하지 않습니다" in str(exc)
    else:
        raise AssertionError("후보와 다른 개수의 점수를 거부해야 합니다.")


def test_non_finite_score_is_rejected():
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever(make_candidates()[:2]),
        FakeScorer([0.9, math.nan]),
        candidate_k=2,
    )

    try:
        reranker.search("질문", k=2)
    except RerankerScoringError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("NaN 점수를 거부해야 합니다.")


def test_non_numeric_score_is_rejected():
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever(make_candidates()[:1]),
        FakeScorer(["관련 있음"]),
        candidate_k=1,
    )

    try:
        reranker.search("질문", k=1)
    except RerankerScoringError as exc:
        assert "숫자" in str(exc)
    else:
        raise AssertionError("숫자가 아닌 점수를 거부해야 합니다.")


def test_sentence_transformers_adapter_builds_query_passage_pairs():
    model = FakeCrossEncoderModel([0.2, 0.8])
    scorer = SentenceTransformersCrossEncoderScorer(
        model_name="fake-model",
        model=model,
        batch_size=2,
    )

    scores = scorer.score_pairs("지원 대상은?", ["첫 문서", "둘째 문서"])

    assert scores == [0.2, 0.8]
    pairs, kwargs = model.calls[0]
    assert pairs == [
        ("지원 대상은?", "첫 문서"),
        ("지원 대상은?", "둘째 문서"),
    ]
    assert kwargs == {
        "batch_size": 2,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


def test_model_registry_pins_reviewed_models_and_limits_remote_code():
    default = get_reranker_model_spec(DEFAULT_RERANKER_MODEL)
    small = get_reranker_model_spec(SMALL_RERANKER_MODEL)

    assert set(RERANKER_MODEL_SPECS) == {
        DEFAULT_RERANKER_MODEL,
        SMALL_RERANKER_MODEL,
    }
    assert len(default.revision) == 40
    assert len(small.revision) == 40
    assert default.code_revision is None
    assert small.code_revision is None
    assert small.parameter_count < default.parameter_count
    assert default.trust_remote_code is False
    assert small.trust_remote_code is False
    assert default.license_name == "Apache-2.0"
    assert small.license_name == "Apache-2.0"

    try:
        get_reranker_model_spec("unknown/model")
    except RerankerModelUnavailableError as exc:
        assert "허용되지 않은" in str(exc)
    else:
        raise AssertionError("검토하지 않은 원격 모델을 허용하면 안 됩니다.")


def test_provider_model_resolution_uses_reviewed_defaults():
    assert (
        resolve_reranker_model(LOCAL_RERANKER_PROVIDER)
        == DEFAULT_RERANKER_MODEL
    )
    assert (
        resolve_reranker_model(COHERE_RERANKER_PROVIDER)
        == DEFAULT_COHERE_RERANKER_MODEL
    )

    for provider, model in (
        ("unknown", None),
        (COHERE_RERANKER_PROVIDER, "rerank-future"),
        (LOCAL_RERANKER_PROVIDER, "unknown/local"),
    ):
        try:
            resolve_reranker_model(provider, model)
        except RerankerModelUnavailableError:
            pass
        else:
            raise AssertionError(
                f"검토하지 않은 provider/model을 거부해야 합니다: {provider}"
            )


def test_cohere_adapter_restores_original_candidate_order_and_records_usage():
    response = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.91),
            SimpleNamespace(index=0, relevance_score=0.23),
        ],
        meta=SimpleNamespace(
            billed_units=SimpleNamespace(search_units=1),
        ),
    )
    client = FakeCohereClient(response)
    scorer = CohereRerankScorer(
        client=client,
        max_tokens_per_doc=512,
    )

    scores = scorer.score_pairs("지원 대상은?", ["첫 문서", "둘째 문서"])

    assert scores == [0.23, 0.91]
    assert client.calls == [
        {
            "model": DEFAULT_COHERE_RERANKER_MODEL,
            "query": "지원 대상은?",
            "documents": ["첫 문서", "둘째 문서"],
            "top_n": 2,
            "max_tokens_per_doc": 512,
        }
    ]
    assert scorer.api_request_count == 1
    assert scorer.search_units == 1.0


def test_cohere_adapter_requires_key_and_validates_response():
    previous_key = os.environ.pop("COHERE_API_KEY", None)
    try:
        try:
            CohereRerankScorer()
        except RerankerModelUnavailableError as exc:
            assert "COHERE_API_KEY" in str(exc)
        else:
            raise AssertionError("Cohere API 키가 없으면 즉시 거부해야 합니다.")
    finally:
        if previous_key is not None:
            os.environ["COHERE_API_KEY"] = previous_key

    invalid_response = SimpleNamespace(
        results=[
            SimpleNamespace(index=0, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.8),
        ],
        meta=None,
    )
    scorer = CohereRerankScorer(client=FakeCohereClient(invalid_response))
    try:
        scorer.score_pairs("질문", ["첫 문서", "둘째 문서"])
    except RerankerScoringError as exc:
        assert "중복" in str(exc)
    else:
        raise AssertionError("중복 Cohere index를 거부해야 합니다.")


def test_sentence_transformers_adapter_validates_model_output():
    scorer = SentenceTransformersCrossEncoderScorer(
        model=FakeCrossEncoderModel([[0.1, 0.9]]),
    )

    try:
        scorer.score_pairs("질문", ["후보"])
    except RerankerScoringError as exc:
        assert "다른 개수" in str(exc)
    else:
        raise AssertionError("후보당 여러 점수를 반환하면 거부해야 합니다.")


def test_retrieve_texts_connects_to_existing_evaluator():
    candidates = make_candidates()[:2]
    reranker = CrossEncoderReranker(
        FakeCandidateRetriever(candidates),
        FakeScorer([0.1, 0.9]),
        candidate_k=2,
    )

    texts = reranker.retrieve_texts("신청 대상", k=1)

    assert texts == [candidates[1].chunk.text]


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
