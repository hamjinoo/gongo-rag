"""LangGraph RAG의 성공·재검색·거절 경로 테스트."""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import DocumentChunk  # noqa: E402
from rag_workflow import (  # noqa: E402
    ANSWERED,
    NO_ANSWER,
    REFUSED,
    EvidenceDecision,
    RAGWorkflow,
    RAGWorkflowConfig,
    assess_evidence_with_llm,
    generate_answer_with_llm,
    parse_evidence_decision,
    rewrite_query_with_llm,
)


@dataclass
class FakeRetriever:
    results_by_query: dict[str, list[SimpleNamespace]]

    def __post_init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[SimpleNamespace]:
        self.queries.append((query, k))
        return self.results_by_query.get(query, [])[:k]


def make_result(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    page_number: int = 2,
) -> SimpleNamespace:
    chunk = DocumentChunk(
        id=chunk_id,
        text=text,
        source_filename="지원사업 공고문.pdf",
        source_sha256="a" * 64,
        file_type="pdf",
        page_number=page_number,
        page_label=f"페이지 {page_number}",
        extraction_method="text",
        chunk_index=rank - 1,
        page_chunk_index=rank - 1,
        start_char=0,
        end_char=len(text),
        strategy="paragraph",
    )
    return SimpleNamespace(
        rank=rank,
        reranker_score=0.9,
        chunk=chunk,
    )


def test_answer_path_returns_cited_answer_and_source_metadata():
    question = "신청 마감은 언제인가요?"
    retriever = FakeRetriever(
        {
            question: [
                make_result(
                    "chunk-1",
                    "신청 마감은 2026년 2월 24일 18시입니다.",
                )
            ]
        }
    )

    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: EvidenceDecision(
            True,
            "마감 날짜가 근거에 직접 있습니다.",
        ),
        rewriter=lambda _question, _evidence: (_ for _ in ()).throw(
            AssertionError("충분한 근거에서는 재작성하면 안 됩니다.")
        ),
        answer_generator=lambda _question, _evidence: (
            "신청 마감은 2026년 2월 24일 18시입니다. [근거 1]"
        ),
    )

    response = workflow.invoke(question)

    assert response.status == ANSWERED
    assert response.rewrite_count == 0
    assert response.steps == ("retrieve", "assess_evidence", "answer")
    assert response.evidence[0]["source_filename"] == "지원사업 공고문.pdf"
    assert response.evidence[0]["page_number"] == 2
    assert retriever.queries == [(question, 5)]


def test_judge_draft_is_validated_without_second_llm_call():
    question = "누가 신청할 수 있나요?"
    retriever = FakeRetriever(
        {question: [make_result("target", "예비창업자가 신청할 수 있습니다.")]}
    )
    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: EvidenceDecision(
            True,
            "신청 대상이 직접 있습니다.",
            "예비창업자가 신청할 수 있습니다. [근거 1]",
        ),
        answer_generator=lambda _question, _evidence: (_ for _ in ()).throw(
            AssertionError("판정과 함께 만든 초안이 있으면 LLM을 다시 호출하면 안 됩니다.")
        ),
    )

    response = workflow.invoke(question)

    assert response.status == ANSWERED
    assert response.answer == "예비창업자가 신청할 수 있습니다. [근거 1]"


def test_answer_preserves_search_scores_for_trace_screen():
    question = "지원 금액은 얼마인가요?"
    result = make_result("money", "지원 금액은 최대 1억원입니다.")
    result.reranker_score = 0.93
    result.rrf_rank = 2
    result.rrf_result = SimpleNamespace(
        rank=2,
        rrf_score=0.031,
        bm25_rank=1,
        bm25_score=8.1,
        vector_rank=3,
        vector_similarity=0.88,
    )
    workflow = RAGWorkflow(
        FakeRetriever({question: [result]}),
        judge=lambda _question, _evidence: EvidenceDecision(True, "금액이 있습니다."),
        rewriter=lambda original, _evidence: original,
        answer_generator=lambda _question, _evidence: (
            "지원 금액은 최대 1억원입니다. [근거 1]"
        ),
    )

    evidence = workflow.invoke(question).evidence[0]

    assert evidence["bm25_rank"] == 1
    assert evidence["bm25_score"] == 8.1
    assert evidence["vector_rank"] == 3
    assert evidence["vector_similarity"] == 0.88
    assert evidence["rrf_rank"] == 2
    assert evidence["rrf_score"] == 0.031
    assert evidence["reranker_score"] == 0.93


def test_rewrite_path_searches_once_more_then_answers_original_question():
    question = "접수는 언제 끝나요?"
    rewritten = "지원사업 신청 접수 마감일 제출기간"
    retriever = FakeRetriever(
        {
            question: [make_result("weak", "사업 개요를 안내합니다.")],
            rewritten: [
                make_result("strong", "제출기간은 2026년 3월 10일까지입니다.")
            ],
        }
    )
    decisions = iter(
        [
            EvidenceDecision(False, "마감 날짜가 없습니다."),
            EvidenceDecision(True, "제출기간이 직접 있습니다."),
        ]
    )

    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: next(decisions),
        rewriter=lambda original, _evidence: (
            rewritten if original == question else original
        ),
        answer_generator=lambda original, _evidence: (
            "접수는 2026년 3월 10일까지입니다. [근거 1]"
            if original == question
            else NO_ANSWER
        ),
    )

    response = workflow.invoke(question)

    assert response.status == ANSWERED
    assert response.final_query == rewritten
    assert response.rewrite_count == 1
    assert response.steps == (
        "retrieve",
        "assess_evidence",
        "rewrite_query",
        "retrieve",
        "assess_evidence",
        "answer",
    )
    assert retriever.queries == [(question, 5), (rewritten, 5)]


def test_refusal_path_stops_after_one_rewrite_when_evidence_is_still_missing():
    question = "공고문에서 숙박비를 지원하나요?"
    rewritten = "숙박비 지원 비용 항목"
    retriever = FakeRetriever({question: [], rewritten: []})

    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: (_ for _ in ()).throw(
            AssertionError("검색 결과가 없으면 LLM 판정기를 호출하지 않습니다.")
        ),
        rewriter=lambda _question, _evidence: rewritten,
        answer_generator=lambda _question, _evidence: (_ for _ in ()).throw(
            AssertionError("근거가 없으면 답변을 생성하면 안 됩니다.")
        ),
        config=RAGWorkflowConfig(max_rewrites=1),
    )

    response = workflow.invoke(question)

    assert response.status == REFUSED
    assert response.answer == NO_ANSWER
    assert response.rewrite_count == 1
    assert response.steps == (
        "retrieve",
        "assess_evidence",
        "rewrite_query",
        "retrieve",
        "assess_evidence",
        "refuse",
    )
    assert len(retriever.queries) == 2


def test_unchanged_rewrite_does_not_repeat_the_same_expensive_search():
    question = "어떤 사람이 신청 가능해?"
    retriever = FakeRetriever(
        {question: [make_result("weak", "사업 개요를 안내합니다.")]}
    )
    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: EvidenceDecision(
            False,
            "신청 대상을 찾지 못했습니다.",
        ),
        rewriter=lambda original, _evidence: original,
    )

    response = workflow.invoke(question)

    assert response.status == REFUSED
    assert response.steps == (
        "retrieve",
        "assess_evidence",
        "rewrite_query",
        "refuse",
    )
    assert retriever.queries == [(question, 5)]


def test_invalid_citation_fails_closed_instead_of_showing_answer():
    question = "지원 금액은 얼마인가요?"
    retriever = FakeRetriever(
        {question: [make_result("money", "지원 금액은 최대 1억원입니다.")]}
    )
    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: EvidenceDecision(True, "금액이 있습니다."),
        rewriter=lambda original, _evidence: original,
        answer_generator=lambda _question, _evidence: (
            "지원 금액은 최대 1억원입니다. [근거 2]"
        ),
    )

    response = workflow.invoke(question)

    assert response.status == REFUSED
    assert response.answer == NO_ANSWER
    assert response.refusal_reason is not None
    assert "존재하지 않는 근거 번호" in response.refusal_reason


def test_number_must_exist_in_the_cited_evidence_not_an_uncited_chunk():
    question = "지원 금액은 얼마인가요?"
    retriever = FakeRetriever(
        {
            question: [
                make_result("cited", "지원 대상은 중소기업입니다.", rank=1),
                make_result("uncited", "지원 금액은 최대 2억원입니다.", rank=2),
            ]
        }
    )
    workflow = RAGWorkflow(
        retriever,
        judge=lambda _question, _evidence: EvidenceDecision(True, "근거가 있습니다."),
        rewriter=lambda original, _evidence: original,
        answer_generator=lambda _question, _evidence: (
            "지원 금액은 최대 2억원입니다. [근거 1]"
        ),
    )

    response = workflow.invoke(question)

    assert response.status == REFUSED
    assert response.refusal_reason is not None
    assert "근거에 없는 숫자" in response.refusal_reason


def test_malformed_evidence_decision_fails_closed():
    decision = parse_evidence_decision("아마 답할 수 있을 것 같습니다.")
    assert decision.sufficient is False
    assert "안전하게 부족함" in decision.reason


def test_structured_evidence_decision_is_parsed_without_reasoning_text():
    decision = parse_evidence_decision(
        '{"sufficient": true, "reason": "신청 자격이 근거에 직접 있습니다."}'
    )

    assert decision == EvidenceDecision(
        True,
        "신청 자격이 근거에 직접 있습니다.",
    )


def test_structured_decision_can_carry_answer_draft():
    decision = parse_evidence_decision(
        '{"sufficient": true, "reason": "자격이 있습니다.", '
        '"draft_answer": "예비창업자가 신청할 수 있습니다. [근거 1]"}'
    )

    assert decision.draft_answer == "예비창업자가 신청할 수 있습니다. [근거 1]"


def test_failed_full_judgment_retries_with_top_three_evidence():
    evidence = [
        {
            "rank": rank,
            "chunk_id": f"chunk-{rank}",
            "text": text,
            "source_filename": "공고문.pdf",
            "page_label": f"페이지 {rank}",
        }
        for rank, text in enumerate(
            [
                "사업 개요",
                "창업기업과 예비창업자가 신청할 수 있습니다.",
                "신청 서류",
                "상담 절차",
                "문의처",
            ],
            start=1,
        )
    ]
    responses = iter(
        [
            "형식이 잘못된 첫 응답",
            '{"sufficient": true, "reason": "신청 대상이 있습니다.", '
            '"draft_answer": "창업기업과 예비창업자가 신청할 수 있습니다. [근거 2]"}',
        ]
    )
    prompts: list[str] = []

    decision = assess_evidence_with_llm(
        "어떤 사람이 신청 가능해?",
        evidence,  # type: ignore[arg-type]
        llm_call=lambda prompt: prompts.append(prompt) or next(responses),
    )

    assert decision.sufficient is True
    assert decision.draft_answer is not None
    assert decision.reason.startswith("상위 3개 근거로 재확인:")
    assert len(prompts) == 2
    assert "문의처" in prompts[0]
    assert "문의처" not in prompts[1]


def test_structured_query_rewrite_uses_only_query_field():
    rewritten = rewrite_query_with_llm(
        "어떤 사람이 신청 가능해?",
        [],
        llm_call=lambda _prompt: '{"query": "스타트업 원스톱 지원센터 신청 자격 지원 대상"}',
    )

    assert rewritten == "스타트업 원스톱 지원센터 신청 자격 지원 대상"


def test_english_reasoning_is_not_used_as_retrieval_query():
    question = "어떤 사람이 신청 가능해?"
    rewritten = rewrite_query_with_llm(
        question,
        [],
        llm_call=lambda _prompt: (
            "Okay, let's tackle this problem. The user wants to know who can apply."
        ),
    )

    assert rewritten == question


def test_structured_answer_uses_only_answer_field():
    generated = generate_answer_with_llm(
        "누가 신청할 수 있나요?",
        [],
        llm_call=lambda _prompt: (
            '{"answer": "예비창업자와 창업기업이 신청할 수 있습니다. [근거 1]"}'
        ),
    )

    assert generated == "예비창업자와 창업기업이 신청할 수 있습니다. [근거 1]"


def test_empty_question_is_rejected_before_search():
    workflow = RAGWorkflow(
        FakeRetriever({}),
        judge=lambda _question, _evidence: EvidenceDecision(False, ""),
        rewriter=lambda question, _evidence: question,
        answer_generator=lambda _question, _evidence: NO_ANSWER,
    )

    try:
        workflow.invoke("  ")
    except ValueError as exc:
        assert "비어 있을 수 없습니다" in str(exc)
    else:
        raise AssertionError("빈 질문을 거부해야 합니다.")
