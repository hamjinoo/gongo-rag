"""검색 근거로 LLM 답변을 만들고 숫자 근거를 검사하는 공통 함수.

검색·재검색·거절 전체 흐름은 ``rag_workflow.py``에서 이 함수를 사용한다.
"""

import json
import re

from local_llm import DEFAULT_OLLAMA_MODEL, call_ollama


LOCAL_LLM_MODEL = DEFAULT_OLLAMA_MODEL

EVIDENCE_DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
        "draft_answer": {"type": "string"},
    },
    "required": ["sufficient", "reason", "draft_answer"],
    "additionalProperties": False,
}

QUERY_REWRITE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# ── 단순 답변 프롬프트 템플릿 ────────────────────────────────
PROMPT_TEMPLATE = """당신은 정부 지원사업 공고문 안내 도우미입니다.
아래 [근거] 조각들만 사용해서 [질문]에 답하세요.

규칙:
1. 근거에 없는 내용은 절대 추측하거나 지어내지 마세요.
2. 근거만으로 답할 수 없으면 정확히 "정보 없음"이라고만 답하세요.
3. 답할 수 있다면, 답변 끝에 사용한 근거 번호를 [근거 1]처럼 표기하세요.

[근거]
{context}

[질문]
{question}

답변:"""


def build_context(chunks: list[str]) -> str:
    """chunk들을 번호 붙여 프롬프트용 문자열로 만든다."""
    return "\n\n".join(f"[근거 {i + 1}]\n{c}" for i, c in enumerate(chunks))


def call_llm(prompt: str) -> str:
    """API 키 없이 로컬 Ollama 모델을 호출한다."""

    return call_ollama(prompt)


def call_evidence_judge_llm(prompt: str) -> str:
    """근거 판정만 담긴 짧은 JSON을 생성한다."""

    return call_ollama(
        prompt,
        response_format=EVIDENCE_DECISION_SCHEMA,
        max_tokens=256,
    )


def call_query_rewrite_llm(prompt: str) -> str:
    """재검색 질문 한 줄만 담긴 JSON을 생성한다."""

    return call_ollama(
        prompt,
        response_format=QUERY_REWRITE_SCHEMA,
        max_tokens=96,
    )


def call_answer_llm(prompt: str) -> str:
    """근거 인용 답변만 담긴 JSON을 생성한다."""

    return call_ollama(
        prompt,
        response_format=ANSWER_SCHEMA,
        max_tokens=384,
    )


def answer(question: str, retrieved_chunks: list[str]) -> str:
    """검색 결과 → 프롬프트 조립 → 생성."""
    prompt = PROMPT_TEMPLATE.format(
        context=build_context(retrieved_chunks), question=question
    )
    raw_response = call_answer_llm(prompt)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return raw_response.strip()
    value = payload.get("answer") if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) else raw_response.strip()


# ──────────────────────────────────────────────────────────────
# 근거 인용 검증 (grounding check)
# ──────────────────────────────────────────────────────────────


def verify_citation(answer_text: str, chunks: list[str]) -> dict:
    """답변의 숫자가 검색 근거에 실제로 존재하는지 검사한다.

    v1은 금액·나이·날짜처럼 위험도가 높은 숫자 환각만 확인한다.
    문장 전체의 의미가 근거에 부합하는지는 이후 평가와 사람 검토가 필요하다.
    """

    if answer_text.strip() == "정보 없음":
        return {"grounded": True, "missing": []}

    answer_without_citations = re.sub(r"\[근거\s+\d+\]", "", answer_text)
    answer_numbers = list(
        dict.fromkeys(re.findall(r"\d+(?:[.,]\d+)?", answer_without_citations))
    )
    evidence = "\n".join(chunks)
    missing = [number for number in answer_numbers if number not in evidence]

    return {"grounded": len(missing) == 0, "missing": missing}


__all__ = [
    "ANSWER_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA",
    "LOCAL_LLM_MODEL",
    "PROMPT_TEMPLATE",
    "QUERY_REWRITE_SCHEMA",
    "answer",
    "build_context",
    "call_answer_llm",
    "call_evidence_judge_llm",
    "call_llm",
    "call_query_rewrite_llm",
    "verify_citation",
]
