"""고정 질문 기반 검색 평가 로직 테스트."""

import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import DocumentChunk, chunk_documents  # noqa: E402
from document_ingestion import extract_document_from_path  # noqa: E402
from evaluate import (  # noqa: E402
    EvaluationError,
    GoldenQuestion,
    GoldenSetValidationError,
    RetrievalCaseResult,
    evaluate_retriever,
    hit_rate_at_k,
    is_hit,
    load_golden_questions,
    no_answer_report,
    render_comparison_markdown,
    validate_golden_set,
)


def make_chunk(
    index: int,
    text: str,
    *,
    source_filename: str = "공고.txt",
) -> DocumentChunk:
    return DocumentChunk(
        id=f"sample-p1-c{index}",
        text=text,
        source_filename=source_filename,
        source_sha256="a" * 64,
        file_type="txt",
        page_number=1,
        page_label="페이지 1",
        extraction_method="plain_text",
        chunk_index=index,
        page_chunk_index=index,
        start_char=0,
        end_char=len(text),
        strategy="paragraph",
    )


@dataclass(frozen=True)
class FakeResult:
    chunk: DocumentChunk


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[DocumentChunk]]) -> None:
        self.results_by_query = results_by_query
        self.requested_k: list[int] = []

    def search(self, query: str, k: int = 5) -> list[FakeResult]:
        self.requested_k.append(k)
        return [
            FakeResult(chunk)
            for chunk in self.results_by_query.get(query, [])[:k]
        ]


def normal_question(
    question_id: str,
    question: str,
    answer_span: str,
    *,
    split: str = "dev",
) -> GoldenQuestion:
    return GoldenQuestion(
        id=question_id,
        type="normal",
        question=question,
        answer_span=answer_span,
        doc_id="공고.txt",
        split=split,
    )


def test_is_hit_basic_and_whitespace_normalization():
    chunks = [
        "신청 자격: 만 39세 이하 예비창업자",
        "자격 요건은 창업 3년\n이내   기업입니다",
    ]

    assert is_hit(chunks, "만 39세 이하") is True
    assert is_hit(chunks, "창업 3년 이내 기업") is True
    assert is_hit(chunks, "사무실 제공") is False


def test_legacy_hit_rate_excludes_no_answer():
    golden = [
        {
            "id": "q1",
            "type": "normal",
            "question": "자격은?",
            "answer_span": "만 39세 이하",
        },
        {
            "id": "q2",
            "type": "normal",
            "question": "금액은?",
            "answer_span": "최대 1억원",
        },
        {
            "id": "q3",
            "type": "normal",
            "question": "기한은?",
            "answer_span": "7월 31일",
        },
        {
            "id": "q4",
            "type": "no_answer",
            "question": "사무실은?",
            "answer_span": None,
        },
    ]

    def fake_retrieve(question: str, k: int) -> list[str]:
        db = {
            "자격은?": ["신청 자격: 만 39세 이하"],
            "금액은?": ["지원 금액은 최대 1억원"],
            "기한은?": ["엉뚱한 내용"],
        }
        return db.get(question, [])[:k]

    result = hit_rate_at_k(golden, fake_retrieve, k=3)

    assert result["n"] == 3
    assert result["hits"] == 2
    assert result["hit_rate"] == 2 / 3
    assert result["misses"] == ["q3"]


def test_no_answer_report_is_separate_from_retrieval_evaluation():
    golden = [
        {"id": "q1", "type": "normal", "question": "자격은?"},
        {"id": "q2", "type": "no_answer", "question": "숙소는?"},
        {"id": "q3", "type": "no_answer", "question": "항공료는?"},
    ]

    result = no_answer_report(
        golden,
        lambda question: "정보 없음" if question == "숙소는?" else "지원합니다.",
    )

    assert result == {
        "n": 2,
        "correct_refusals": 1,
        "hallucinated": ["q3"],
    }


def test_load_golden_questions_validates_and_filters_split():
    rows = [
        {
            "id": "q1",
            "type": "normal",
            "split": "dev",
            "question": "자격은?",
            "answer_span": "창업 기업",
            "doc_id": "공고.txt",
        },
        {
            "id": "q2",
            "type": "no_answer",
            "split": "test",
            "question": "숙소는?",
            "answer_span": None,
            "doc_id": None,
        },
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "golden.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )

        dev = load_golden_questions(path, split="dev")

    assert [question.id for question in dev] == ["q1"]
    assert dev[0].answer_span == "창업 기업"


def test_load_golden_questions_rejects_duplicate_id():
    row = {
        "id": "q1",
        "type": "normal",
        "question": "자격은?",
        "answer_span": "창업 기업",
        "doc_id": "공고.txt",
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "golden.jsonl"
        path.write_text(
            json.dumps(row, ensure_ascii=False)
            + "\n"
            + json.dumps(row, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            load_golden_questions(path)
        except GoldenSetValidationError as exc:
            assert "중복" in str(exc)
        else:
            raise AssertionError("중복 질문 ID를 거부해야 합니다.")


def test_validate_golden_set_maps_answer_to_chunk_ids():
    chunks = [
        make_chunk(0, "신청 대상은 창업 3년 이내 기업입니다."),
        make_chunk(1, "접수 기간은 7월 31일까지입니다."),
    ]
    questions = [
        normal_question("q1", "누가 신청하나요?", "창업 3년 이내 기업"),
        GoldenQuestion(
            id="q2",
            type="no_answer",
            question="숙소를 주나요?",
            answer_span=None,
            doc_id=None,
            split="dev",
        ),
    ]

    validation = validate_golden_set(questions, chunks)

    assert validation.total_questions == 2
    assert validation.normal_questions == 1
    assert validation.no_answer_questions == 1
    assert validation.relevant_chunk_ids["q1"] == frozenset({chunks[0].id})


def test_validate_golden_set_rejects_missing_answer_chunk():
    question = normal_question(
        "q1",
        "지원금은?",
        "최대 1억원",
    )

    try:
        validate_golden_set([question], [make_chunk(0, "접수 기간 안내")])
    except GoldenSetValidationError as exc:
        assert "정답 문장" in str(exc)
    else:
        raise AssertionError("현재 Chunk에 없는 정답을 거부해야 합니다.")


def test_retrieval_case_metrics_handle_multiple_relevant_chunks():
    case = RetrievalCaseResult(
        question_id="q1",
        question="질문",
        split="dev",
        returned_chunk_ids=("x", "a", "b", "y"),
        relevant_chunk_ids=frozenset({"a", "b"}),
        relevant_ranks=(2, 3),
        latency_ms=10.0,
    )

    assert case.first_relevant_rank == 2
    assert case.reciprocal_rank == 0.5
    assert case.hit_at(1) is False
    assert case.hit_at(3) is True
    expected_dcg = (1 / math.log2(3)) + (1 / math.log2(4))
    ideal_dcg = 1 + (1 / math.log2(3))
    assert abs(case.ndcg_at(3) - (expected_dcg / ideal_dcg)) < 1e-12


def test_evaluate_retriever_calculates_rank_metrics_and_latency():
    chunks = [
        make_chunk(0, "정답 하나"),
        make_chunk(1, "정답 둘"),
        make_chunk(2, "무관한 내용"),
    ]
    questions = [
        normal_question("q1", "첫 질문", "정답 하나"),
        normal_question("q2", "둘째 질문", "정답 둘"),
    ]
    retriever = FakeRetriever(
        {
            "첫 질문": [chunks[0], chunks[2]],
            "둘째 질문": [chunks[2], chunks[1]],
        }
    )
    clock_values = iter([1.0, 1.01, 2.0, 2.02])

    summary = evaluate_retriever(
        "가짜 검색기",
        retriever,
        questions,
        chunks,
        ks=(1, 2),
        clock=lambda: next(clock_values),
    )

    assert retriever.requested_k == [2, 2]
    assert summary.question_count == 2
    assert summary.hit_rates == {1: 0.5, 2: 1.0}
    assert summary.mrr == 0.75
    expected_ndcg_2 = (1 + (1 / math.log2(3))) / 2
    assert abs(summary.ndcg_scores[2] - expected_ndcg_2) < 1e-12
    assert abs(summary.mean_latency_ms - 15.0) < 1e-9
    assert abs(summary.p95_latency_ms - 20.0) < 1e-9
    assert summary.misses_at(1) == ["q2"]
    assert summary.misses_at(2) == []


def test_evaluate_retriever_rejects_duplicate_results():
    chunk = make_chunk(0, "정답")
    retriever = FakeRetriever({"질문": [chunk, chunk]})
    question = normal_question("q1", "질문", "정답")

    try:
        evaluate_retriever(
            "중복 검색기",
            retriever,
            [question],
            [chunk],
            ks=(2,),
        )
    except EvaluationError as exc:
        assert "중복" in str(exc)
    else:
        raise AssertionError("중복 Chunk 결과를 거부해야 합니다.")


def test_render_comparison_markdown_contains_quality_and_latency():
    chunks = [make_chunk(0, "정답")]
    question = normal_question("q1", "질문", "정답")
    summary = evaluate_retriever(
        "BM25",
        FakeRetriever({"질문": chunks}),
        [question],
        chunks,
        ks=(1,),
        clock=iter([1.0, 1.01]).__next__,
    )

    markdown = render_comparison_markdown([summary])

    assert "Hit@1" in markdown
    assert "MRR" in markdown
    assert "nDCG@1" in markdown
    assert "평균 ms" in markdown
    assert "BM25" in markdown


def test_real_golden_set_has_balanced_splits_and_matches_current_chunks():
    questions = load_golden_questions(
        PROJECT_ROOT / "data" / "golden_set.jsonl"
    )
    dev_questions = [question for question in questions if question.split == "dev"]
    test_questions = [question for question in questions if question.split == "test"]
    documents = [
        extract_document_from_path(path)
        for path in sorted((PROJECT_ROOT / "docs" / "text").glob("*.txt"))
    ]
    chunks = chunk_documents(documents)

    validation = validate_golden_set(questions, chunks)

    assert len(questions) == 36
    assert sum(question.type == "normal" for question in dev_questions) == 20
    assert sum(question.type == "normal" for question in test_questions) == 10
    assert sum(question.type == "no_answer" for question in dev_questions) == 3
    assert sum(question.type == "no_answer" for question in test_questions) == 3
    assert validation.normal_questions == 30
    assert validation.no_answer_questions == 6
    assert len(chunks) == 38


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
