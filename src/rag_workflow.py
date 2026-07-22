"""LangGraph로 검색·재검색·근거 인용·안전한 거절 흐름을 연결한다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_answer import call_llm, verify_citation


ANSWERED = "answered"
REFUSED = "refused"
NO_ANSWER = "정보 없음"


class RankedRetriever(Protocol):
    """Reranker처럼 chunk가 포함된 순위 결과를 반환하는 검색기."""

    def search(self, query: str, k: int = 5) -> list[Any]:
        """질문에 맞는 검색 결과를 반환한다."""


class RAGEvidence(TypedDict):
    """LangGraph 상태와 최종 응답에 보존할 한 개의 근거."""

    rank: int
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    page_label: str
    score: float | None


class RAGState(TypedDict, total=False):
    """노드 사이에서 공유하는 RAG 실행 상태."""

    question: str
    active_query: str
    evidence: list[RAGEvidence]
    sufficient: bool
    decision_reason: str
    rewrite_count: int
    answer: str
    status: Literal["answered", "refused"]
    refusal_reason: str | None
    steps: list[str]


@dataclass(frozen=True)
class EvidenceDecision:
    """검색 근거만으로 원래 질문에 답할 수 있는지에 대한 판정."""

    sufficient: bool
    reason: str


@dataclass(frozen=True)
class AnswerValidation:
    """생성된 답변의 인용 번호와 숫자 근거 검사 결과."""

    grounded: bool
    reason: str
    citation_numbers: tuple[int, ...]
    unsupported_numbers: tuple[str, ...]


@dataclass(frozen=True)
class RAGWorkflowConfig:
    """그래프가 한 번에 사용할 근거 수와 재검색 제한."""

    top_k: int = 5
    max_rewrites: int = 1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        if self.max_rewrites < 0:
            raise ValueError("max_rewrites는 0 이상이어야 합니다.")


@dataclass(frozen=True)
class RAGResponse:
    """UI·CLI·평가기가 공통으로 읽을 수 있는 최종 결과."""

    question: str
    final_query: str
    answer: str
    status: Literal["answered", "refused"]
    evidence: tuple[RAGEvidence, ...]
    rewrite_count: int
    steps: tuple[str, ...]
    decision_reason: str
    refusal_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "final_query": self.final_query,
            "answer": self.answer,
            "status": self.status,
            "evidence": [dict(item) for item in self.evidence],
            "rewrite_count": self.rewrite_count,
            "steps": list(self.steps),
            "decision_reason": self.decision_reason,
            "refusal_reason": self.refusal_reason,
        }


EvidenceJudge = Callable[[str, list[RAGEvidence]], EvidenceDecision]
QueryRewriter = Callable[[str, list[RAGEvidence]], str]
AnswerGenerator = Callable[[str, list[RAGEvidence]], str]


EVIDENCE_ASSESSMENT_PROMPT = """당신은 정부 지원사업 RAG의 근거 판정기입니다.
아래 근거만으로 원래 질문에 정확히 답할 수 있는지 판단하세요.

규칙:
1. 질문이 요구한 대상, 조건, 숫자, 날짜가 근거에 직접 있어야 합니다.
2. 비슷한 사업이나 다른 대상의 내용만 있으면 부족함입니다.
3. 추측하거나 상식으로 채우지 마세요.
4. 반드시 아래 두 줄 형식으로만 답하세요.

판정: 충분함 또는 부족함
이유: 한 문장

[원래 질문]
{question}

[검색 근거]
{context}
"""


QUERY_REWRITE_PROMPT = """당신은 한국어 정부 공고문 검색어를 고치는 도우미입니다.
원래 질문과 부족했던 검색 근거를 보고, 같은 뜻을 유지하면서 공고문에 나올 법한
핵심 명사·사업명·날짜 표현을 포함한 검색 질문 하나를 만드세요.

규칙:
1. 새로운 사실을 추가하지 마세요.
2. 질문 하나만 한 줄로 답하세요.
3. 설명이나 따옴표를 붙이지 마세요.

[원래 질문]
{question}

[부족했던 검색 근거]
{context}

고친 질문:
"""


ANSWER_PROMPT = """당신은 정부 지원사업 공고문 안내 도우미입니다.
아래 [근거]만 사용해서 [질문]에 답하세요.

규칙:
1. 근거에 없는 내용은 추측하거나 지어내지 마세요.
2. 근거만으로 답할 수 없으면 정확히 "정보 없음"이라고만 답하세요.
3. 답할 수 있다면 사용한 근거 번호를 문장 끝에 [근거 1]처럼 표시하세요.
4. 파일명과 페이지는 근거 표시에 이미 있으므로 새로 만들지 마세요.

[근거]
{context}

[질문]
{question}

답변:
"""


def build_evidence_context(evidence: list[RAGEvidence]) -> str:
    """본문과 파일·페이지를 함께 LLM에 전달한다."""

    if not evidence:
        return "(검색된 근거 없음)"
    blocks = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"[근거 {item['rank']}]",
                    (
                        "출처: "
                        f"{item['source_filename']}, {item['page_label']} "
                        f"(chunk {item['chunk_id']})"
                    ),
                    item["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def parse_evidence_decision(raw_response: str) -> EvidenceDecision:
    """정해진 두 줄 형식만 허용하며 형식 오류는 안전하게 부족함으로 본다."""

    decision_match = re.search(
        r"^\s*판정\s*:\s*(충분함|부족함)\s*$",
        raw_response,
        flags=re.MULTILINE,
    )
    reason_match = re.search(
        r"^\s*이유\s*:\s*(.+?)\s*$",
        raw_response,
        flags=re.MULTILINE,
    )
    if decision_match is None or reason_match is None:
        return EvidenceDecision(
            sufficient=False,
            reason="근거 판정 응답 형식이 올바르지 않아 안전하게 부족함으로 처리했습니다.",
        )
    return EvidenceDecision(
        sufficient=decision_match.group(1) == "충분함",
        reason=reason_match.group(1).strip(),
    )


def assess_evidence_with_llm(
    question: str,
    evidence: list[RAGEvidence],
    *,
    llm_call: Callable[[str], str] = call_llm,
) -> EvidenceDecision:
    """LLM이 근거 충분성을 판정하되 잘못된 형식은 fail-closed 처리한다."""

    if not evidence:
        return EvidenceDecision(False, "검색 결과가 없습니다.")
    prompt = EVIDENCE_ASSESSMENT_PROMPT.format(
        question=question,
        context=build_evidence_context(evidence),
    )
    return parse_evidence_decision(llm_call(prompt))


def rewrite_query_with_llm(
    question: str,
    evidence: list[RAGEvidence],
    *,
    llm_call: Callable[[str], str] = call_llm,
) -> str:
    """원래 의미를 유지한 한국어 재검색 질문 한 줄을 만든다."""

    prompt = QUERY_REWRITE_PROMPT.format(
        question=question,
        context=build_evidence_context(evidence),
    )
    raw_response = llm_call(prompt).strip()
    if not raw_response:
        return question
    rewritten = raw_response.splitlines()[0].strip()
    rewritten = re.sub(r"^(고친\s*질문|질문)\s*:\s*", "", rewritten)
    return rewritten or question


def generate_answer_with_llm(
    question: str,
    evidence: list[RAGEvidence],
    *,
    llm_call: Callable[[str], str] = call_llm,
) -> str:
    """출처 metadata가 포함된 근거로 인용 답변을 생성한다."""

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=build_evidence_context(evidence),
    )
    return llm_call(prompt).strip()


def validate_generated_answer(
    answer_text: str,
    evidence: list[RAGEvidence],
) -> AnswerValidation:
    """없는 근거 번호와 근거에 없는 숫자를 차단한다."""

    if answer_text.strip() == NO_ANSWER:
        return AnswerValidation(True, "정보 없음 응답", (), ())

    raw_citations = re.findall(r"\[근거\s+(\d+)\]", answer_text)
    if not raw_citations:
        return AnswerValidation(
            False,
            "답변에 [근거 N] 인용이 없습니다.",
            (),
            (),
        )

    citation_numbers = tuple(dict.fromkeys(int(value) for value in raw_citations))
    valid_numbers = {item["rank"] for item in evidence}
    invalid = tuple(
        number for number in citation_numbers if number not in valid_numbers
    )
    if invalid:
        return AnswerValidation(
            False,
            f"존재하지 않는 근거 번호를 인용했습니다: {list(invalid)}",
            citation_numbers,
            (),
        )

    cited_texts = [
        item["text"]
        for item in evidence
        if item["rank"] in citation_numbers
    ]
    numeric_check = verify_citation(answer_text, cited_texts)
    unsupported = tuple(str(value) for value in numeric_check["missing"])
    if unsupported:
        return AnswerValidation(
            False,
            f"검색 근거에 없는 숫자가 있습니다: {list(unsupported)}",
            citation_numbers,
            unsupported,
        )

    return AnswerValidation(
        True,
        "인용 번호와 숫자가 검색 근거에 있습니다.",
        citation_numbers,
        (),
    )


class RAGWorkflow:
    """의존성을 주입할 수 있는 LangGraph RAG workflow."""

    def __init__(
        self,
        retriever: RankedRetriever,
        *,
        judge: EvidenceJudge | None = None,
        rewriter: QueryRewriter | None = None,
        answer_generator: AnswerGenerator | None = None,
        config: RAGWorkflowConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.judge = judge or assess_evidence_with_llm
        self.rewriter = rewriter or rewrite_query_with_llm
        self.answer_generator = answer_generator or generate_answer_with_llm
        self.config = config or RAGWorkflowConfig()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess_evidence", self._assess_evidence)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("answer", self._answer)
        builder.add_node("refuse", self._refuse)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess_evidence")
        builder.add_conditional_edges(
            "assess_evidence",
            self._route_after_assessment,
            {
                "answer": "answer",
                "rewrite_query": "rewrite_query",
                "refuse": "refuse",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("answer", END)
        builder.add_edge("refuse", END)
        return builder.compile()

    def invoke(self, question: str) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question은 비어 있을 수 없습니다.")

        state = self.graph.invoke(
            {
                "question": normalized_question,
                "active_query": normalized_question,
                "evidence": [],
                "rewrite_count": 0,
                "steps": [],
            }
        )
        return RAGResponse(
            question=state["question"],
            final_query=state["active_query"],
            answer=state["answer"],
            status=state["status"],
            evidence=tuple(state.get("evidence", [])),
            rewrite_count=state.get("rewrite_count", 0),
            steps=tuple(state.get("steps", [])),
            decision_reason=state.get("decision_reason", ""),
            refusal_reason=state.get("refusal_reason"),
        )

    def _retrieve(self, state: RAGState) -> dict[str, object]:
        query = state.get("active_query") or state["question"]
        results = self.retriever.search(query, k=self.config.top_k)
        evidence = _results_to_evidence(results)
        return {
            "active_query": query,
            "evidence": evidence,
            "steps": [*state.get("steps", []), "retrieve"],
        }

    def _assess_evidence(self, state: RAGState) -> dict[str, object]:
        evidence = state.get("evidence", [])
        decision = (
            self.judge(state["question"], evidence)
            if evidence
            else EvidenceDecision(False, "검색 결과가 없습니다.")
        )
        return {
            "sufficient": decision.sufficient,
            "decision_reason": decision.reason,
            "steps": [*state.get("steps", []), "assess_evidence"],
        }

    def _route_after_assessment(
        self,
        state: RAGState,
    ) -> Literal["answer", "rewrite_query", "refuse"]:
        if state.get("sufficient", False):
            return "answer"
        if state.get("rewrite_count", 0) < self.config.max_rewrites:
            return "rewrite_query"
        return "refuse"

    def _rewrite_query(self, state: RAGState) -> dict[str, object]:
        rewritten = self.rewriter(
            state["question"],
            state.get("evidence", []),
        ).strip()
        return {
            "active_query": rewritten or state["question"],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "steps": [*state.get("steps", []), "rewrite_query"],
        }

    def _answer(self, state: RAGState) -> dict[str, object]:
        evidence = state.get("evidence", [])
        generated = self.answer_generator(state["question"], evidence).strip()
        if generated == NO_ANSWER:
            return {
                "answer": NO_ANSWER,
                "status": REFUSED,
                "refusal_reason": "답변 생성기가 근거 부족으로 정보 없음을 반환했습니다.",
                "steps": [*state.get("steps", []), "answer", "refuse"],
            }

        validation = validate_generated_answer(generated, evidence)
        if not validation.grounded:
            return {
                "answer": NO_ANSWER,
                "status": REFUSED,
                "refusal_reason": f"답변 근거 검증 실패: {validation.reason}",
                "steps": [*state.get("steps", []), "answer", "refuse"],
            }
        return {
            "answer": generated,
            "status": ANSWERED,
            "refusal_reason": None,
            "steps": [*state.get("steps", []), "answer"],
        }

    def _refuse(self, state: RAGState) -> dict[str, object]:
        return {
            "answer": NO_ANSWER,
            "status": REFUSED,
            "refusal_reason": state.get(
                "decision_reason",
                "재검색 후에도 충분한 근거를 찾지 못했습니다.",
            ),
            "steps": [*state.get("steps", []), "refuse"],
        }


def _results_to_evidence(results: list[Any]) -> list[RAGEvidence]:
    evidence: list[RAGEvidence] = []
    seen_ids: set[str] = set()

    for fallback_rank, result in enumerate(results, start=1):
        chunk = getattr(result, "chunk", None)
        if chunk is None:
            raise TypeError("검색 결과에는 chunk 속성이 필요합니다.")
        chunk_id = str(getattr(chunk, "id", "")).strip()
        text = str(getattr(chunk, "text", "")).strip()
        if not chunk_id or not text:
            raise ValueError("검색 근거의 chunk ID와 본문은 비어 있을 수 없습니다.")
        if chunk_id in seen_ids:
            raise ValueError(f"검색 결과의 chunk ID가 중복됐습니다: {chunk_id}")
        seen_ids.add(chunk_id)

        score = _result_score(result)
        evidence.append(
            {
                "rank": int(getattr(result, "rank", fallback_rank)),
                "chunk_id": chunk_id,
                "text": text,
                "source_filename": str(
                    getattr(chunk, "source_filename", "알 수 없는 문서")
                ),
                "page_number": int(getattr(chunk, "page_number", 0)),
                "page_label": str(
                    getattr(chunk, "page_label", "페이지 정보 없음")
                ),
                "score": score,
            }
        )
    return evidence


def _result_score(result: Any) -> float | None:
    for attribute in (
        "reranker_score",
        "rrf_score",
        "score",
        "similarity",
    ):
        value = getattr(result, attribute, None)
        if value is not None:
            return float(value)
    return None
