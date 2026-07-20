"""고정 질문과 정답 근거로 순위형 검색기의 품질과 지연을 평가한다."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, cast

from chunker import DocumentChunk


QuestionType = Literal["normal", "no_answer"]
DatasetSplit = Literal["dev", "test", "unassigned"]


class EvaluationError(RuntimeError):
    """검색 평가를 신뢰할 수 없는 상태임."""


class GoldenSetValidationError(EvaluationError):
    """질문 또는 정답 근거가 현재 문서·Chunk와 맞지 않음."""


class RankedResult(Protocol):
    @property
    def chunk(self) -> DocumentChunk:
        """검색된 원본 Chunk."""


class RankedRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[RankedResult]:
        """질문의 상위 k개 결과를 순서대로 반환한다."""


@dataclass(frozen=True)
class GoldenQuestion:
    """한 개의 고정 평가 질문과 사람이 확인한 정답 근거."""

    id: str
    type: QuestionType
    question: str
    answer_span: str | None
    doc_id: str | None
    note: str = ""
    split: DatasetSplit = "unassigned"

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> GoldenQuestion:
        question_id = str(row.get("id") or "").strip()
        question_type = str(row.get("type") or "").strip()
        question = str(row.get("question") or "").strip()
        raw_answer = row.get("answer_span")
        raw_doc_id = row.get("doc_id")
        split = str(row.get("split") or "unassigned").strip()

        if not question_id:
            raise GoldenSetValidationError("골든셋 id는 비어 있을 수 없습니다.")
        if question_type not in ("normal", "no_answer"):
            raise GoldenSetValidationError(
                f"{question_id}: type은 normal 또는 no_answer여야 합니다."
            )
        if not question:
            raise GoldenSetValidationError(
                f"{question_id}: question은 비어 있을 수 없습니다."
            )
        if split not in ("dev", "test", "unassigned"):
            raise GoldenSetValidationError(
                f"{question_id}: split은 dev, test, unassigned 중 하나여야 합니다."
            )

        answer_span = (
            str(raw_answer).strip()
            if raw_answer is not None
            else None
        )
        doc_id = (
            str(raw_doc_id).strip()
            if raw_doc_id is not None
            else None
        )
        if question_type == "normal" and (not answer_span or not doc_id):
            raise GoldenSetValidationError(
                f"{question_id}: normal 문항에는 answer_span과 doc_id가 필요합니다."
            )
        if question_type == "no_answer" and answer_span is not None:
            raise GoldenSetValidationError(
                f"{question_id}: no_answer 문항의 answer_span은 null이어야 합니다."
            )

        return cls(
            id=question_id,
            type=cast(QuestionType, question_type),
            question=question,
            answer_span=answer_span,
            doc_id=doc_id,
            note=str(row.get("note") or "").strip(),
            split=cast(DatasetSplit, split),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "answer_span": self.answer_span,
            "doc_id": self.doc_id,
            "note": self.note,
            "split": self.split,
        }


@dataclass(frozen=True)
class GoldenSetValidation:
    total_questions: int
    normal_questions: int
    no_answer_questions: int
    relevant_chunk_ids: dict[str, frozenset[str]]
    multiple_relevant_questions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_questions": self.total_questions,
            "normal_questions": self.normal_questions,
            "no_answer_questions": self.no_answer_questions,
            "multiple_relevant_questions": list(
                self.multiple_relevant_questions
            ),
            "relevant_chunk_counts": {
                question_id: len(chunk_ids)
                for question_id, chunk_ids in self.relevant_chunk_ids.items()
            },
        }


@dataclass(frozen=True)
class RetrievalCaseResult:
    """질문 하나의 검색 순위, 정답 위치와 지연."""

    question_id: str
    question: str
    split: DatasetSplit
    returned_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    relevant_ranks: tuple[int, ...]
    latency_ms: float

    @property
    def first_relevant_rank(self) -> int | None:
        return self.relevant_ranks[0] if self.relevant_ranks else None

    @property
    def reciprocal_rank(self) -> float:
        if self.first_relevant_rank is None:
            return 0.0
        return 1.0 / self.first_relevant_rank

    def hit_at(self, k: int) -> bool:
        return any(rank <= k for rank in self.relevant_ranks)

    def ndcg_at(self, k: int) -> float:
        """binary relevance를 사용하는 한 질문의 nDCG@k."""

        if k < 1:
            raise ValueError("k는 1 이상이어야 합니다.")

        returned_relevance = [
            1 if chunk_id in self.relevant_chunk_ids else 0
            for chunk_id in self.returned_chunk_ids[:k]
        ]
        dcg = sum(
            relevance / math.log2(rank + 1)
            for rank, relevance in enumerate(returned_relevance, start=1)
        )
        ideal_count = min(len(self.relevant_chunk_ids), k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        return dcg / ideal_dcg if ideal_dcg else 0.0

    def to_dict(self, *, ks: tuple[int, ...]) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "split": self.split,
            "returned_chunk_ids": list(self.returned_chunk_ids),
            "relevant_chunk_ids": sorted(self.relevant_chunk_ids),
            "relevant_ranks": list(self.relevant_ranks),
            "first_relevant_rank": self.first_relevant_rank,
            "reciprocal_rank": self.reciprocal_rank,
            "latency_ms": self.latency_ms,
            "hits": {f"hit@{k}": self.hit_at(k) for k in ks},
            "ndcg": {f"ndcg@{k}": self.ndcg_at(k) for k in ks},
        }


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """한 검색기의 전체 고정 질문 평가 결과."""

    system_name: str
    split: str
    ks: tuple[int, ...]
    cases: tuple[RetrievalCaseResult, ...]

    @property
    def question_count(self) -> int:
        return len(self.cases)

    @property
    def hit_rates(self) -> dict[int, float]:
        return {
            k: (
                sum(case.hit_at(k) for case in self.cases)
                / self.question_count
            )
            for k in self.ks
        }

    @property
    def mrr(self) -> float:
        return (
            sum(case.reciprocal_rank for case in self.cases)
            / self.question_count
        )

    @property
    def ndcg_scores(self) -> dict[int, float]:
        return {
            k: (
                sum(case.ndcg_at(k) for case in self.cases)
                / self.question_count
            )
            for k in self.ks
        }

    @property
    def mean_latency_ms(self) -> float:
        return (
            sum(case.latency_ms for case in self.cases)
            / self.question_count
        )

    @property
    def p95_latency_ms(self) -> float:
        ordered = sorted(case.latency_ms for case in self.cases)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    def misses_at(self, k: int) -> list[str]:
        return [
            case.question_id
            for case in self.cases
            if not case.hit_at(k)
        ]

    def to_dict(self) -> dict[str, object]:
        max_k = max(self.ks)
        return {
            "system_name": self.system_name,
            "split": self.split,
            "question_count": self.question_count,
            "ks": list(self.ks),
            "hit_rates": {
                f"hit@{k}": score
                for k, score in self.hit_rates.items()
            },
            "mrr": self.mrr,
            "ndcg": {
                f"ndcg@{k}": score
                for k, score in self.ndcg_scores.items()
            },
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            f"misses@{max_k}": self.misses_at(max_k),
            "cases": [
                case.to_dict(ks=self.ks)
                for case in self.cases
            ],
        }


def normalize(text: str) -> str:
    """PDF 줄바꿈과 연속 공백 차이를 무시한다."""

    return " ".join(text.split())


def load_golden_questions(
    path: str | Path,
    *,
    split: str | None = None,
    question_type: QuestionType | None = None,
) -> list[GoldenQuestion]:
    """JSONL 골든셋을 검증하며 불러오고 선택적으로 필터링한다."""

    source = Path(path)
    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw_row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetValidationError(
                f"{source.name}:{line_number} JSON 형식이 잘못됐습니다: {exc}"
            ) from exc
        if not isinstance(raw_row, dict):
            raise GoldenSetValidationError(
                f"{source.name}:{line_number} 객체 형식이어야 합니다."
            )

        question = GoldenQuestion.from_dict(raw_row)
        if question.id in seen_ids:
            raise GoldenSetValidationError(
                f"골든셋 id가 중복됐습니다: {question.id}"
            )
        seen_ids.add(question.id)
        questions.append(question)

    if split is not None:
        questions = [
            question
            for question in questions
            if question.split == split
        ]
    if question_type is not None:
        questions = [
            question
            for question in questions
            if question.type == question_type
        ]
    return questions


def load_golden(path: str | Path) -> list[dict[str, object]]:
    """기존 학습 코드와 호환되는 dict 목록 loader."""

    return [
        question.to_dict()
        for question in load_golden_questions(path)
    ]


def is_hit(retrieved_texts: list[str], answer_span: str) -> bool:
    """검색된 본문 중 정규화한 정답 문장을 포함한 본문이 있는지 확인한다."""

    answer = normalize(answer_span)
    return any(answer in normalize(text) for text in retrieved_texts)


def hit_rate_at_k(
    golden: list[dict[str, object]],
    retrieve_fn: Callable[[str, int], list[str]],
    k: int = 3,
) -> dict[str, object]:
    """기존 text 기반 검색기의 normal 문항 Hit@k를 계산한다."""

    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")

    normal_rows = [row for row in golden if row.get("type") == "normal"]
    misses: list[str] = []
    hits = 0

    for row in normal_rows:
        question = str(row["question"])
        answer_span = str(row["answer_span"])
        retrieved = retrieve_fn(question, k)
        if is_hit(retrieved, answer_span):
            hits += 1
        else:
            misses.append(str(row["id"]))

    count = len(normal_rows)
    return {
        "k": k,
        "n": count,
        "hits": hits,
        "hit_rate": hits / count if count else 0.0,
        "misses": misses,
    }


def no_answer_report(
    golden: list[dict[str, object]],
    answer_fn: Callable[[str], str],
) -> dict[str, object]:
    """no_answer 문항에서 최종 답변이 정보 없음을 올바르게 말하는지 확인한다."""

    no_answer_rows = [
        row for row in golden if row.get("type") == "no_answer"
    ]
    hallucinated: list[str] = []
    correct_refusals = 0

    for row in no_answer_rows:
        answer = answer_fn(str(row["question"]))
        if "정보 없음" in answer:
            correct_refusals += 1
        else:
            hallucinated.append(str(row["id"]))

    return {
        "n": len(no_answer_rows),
        "correct_refusals": correct_refusals,
        "hallucinated": hallucinated,
    }


def validate_golden_set(
    questions: list[GoldenQuestion],
    chunks: list[DocumentChunk],
) -> GoldenSetValidation:
    """모든 normal 정답이 지정 문서의 현재 Chunk에 실제로 들어 있는지 확인한다."""

    chunks_by_document: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.source_filename, []).append(chunk)

    relevant_by_question: dict[str, frozenset[str]] = {}
    multiple_relevant: list[str] = []
    errors: list[str] = []

    for question in questions:
        if question.type != "normal":
            continue
        if not question.answer_span or not question.doc_id:
            errors.append(
                f"{question.id}: normal 문항의 정답 또는 문서 ID가 비어 있음"
            )
            continue

        document_chunks = chunks_by_document.get(question.doc_id)
        if not document_chunks:
            errors.append(
                f"{question.id}: 문서를 찾을 수 없음 ({question.doc_id})"
            )
            continue

        answer = normalize(question.answer_span)
        relevant_ids = frozenset(
            chunk.id
            for chunk in document_chunks
            if answer in normalize(chunk.text)
        )
        if not relevant_ids:
            errors.append(
                f"{question.id}: 현재 Chunk에서 정답 문장을 찾을 수 없음"
            )
            continue

        relevant_by_question[question.id] = relevant_ids
        if len(relevant_ids) > 1:
            multiple_relevant.append(question.id)

    if errors:
        raise GoldenSetValidationError(
            "골든셋과 현재 Chunk가 맞지 않습니다:\n- "
            + "\n- ".join(errors)
        )

    return GoldenSetValidation(
        total_questions=len(questions),
        normal_questions=sum(
            question.type == "normal"
            for question in questions
        ),
        no_answer_questions=sum(
            question.type == "no_answer"
            for question in questions
        ),
        relevant_chunk_ids=relevant_by_question,
        multiple_relevant_questions=tuple(multiple_relevant),
    )


def evaluate_retriever(
    system_name: str,
    retriever: RankedRetriever,
    questions: list[GoldenQuestion],
    chunks: list[DocumentChunk],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    clock: Callable[[], float] = time.perf_counter,
) -> RetrievalEvaluationSummary:
    """한 검색기를 같은 normal 문항으로 평가한다."""

    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("ks에는 1 이상의 값이 필요합니다.")

    normal_questions = [
        question
        for question in questions
        if question.type == "normal"
    ]
    if not normal_questions:
        raise GoldenSetValidationError("평가할 normal 문항이 없습니다.")

    validation = validate_golden_set(normal_questions, chunks)
    max_k = max(normalized_ks)
    cases: list[RetrievalCaseResult] = []

    for question in normal_questions:
        started = clock()
        results = retriever.search(question.question, k=max_k)
        latency_ms = max(0.0, (clock() - started) * 1000)

        returned_ids = tuple(result.chunk.id for result in results)
        if len(returned_ids) != len(set(returned_ids)):
            raise EvaluationError(
                f"{system_name}가 중복 Chunk ID를 반환했습니다: {question.id}"
            )

        relevant_ids = validation.relevant_chunk_ids[question.id]
        relevant_ranks = tuple(
            rank
            for rank, chunk_id in enumerate(returned_ids, start=1)
            if chunk_id in relevant_ids
        )
        cases.append(
            RetrievalCaseResult(
                question_id=question.id,
                question=question.question,
                split=question.split,
                returned_chunk_ids=returned_ids,
                relevant_chunk_ids=relevant_ids,
                relevant_ranks=relevant_ranks,
                latency_ms=latency_ms,
            )
        )

    split_names = sorted({question.split for question in normal_questions})
    split_label = (
        split_names[0]
        if len(split_names) == 1
        else "mixed"
    )
    return RetrievalEvaluationSummary(
        system_name=system_name,
        split=split_label,
        ks=normalized_ks,
        cases=tuple(cases),
    )


def render_comparison_markdown(
    summaries: list[RetrievalEvaluationSummary],
) -> str:
    """사람이 한눈에 비교할 수 있는 Markdown 표를 만든다."""

    if not summaries:
        return "(평가 결과 없음)"

    ks = summaries[0].ks
    if any(summary.ks != ks for summary in summaries):
        raise EvaluationError("비교할 평가 결과의 k 설정이 서로 다릅니다.")

    headers = [
        "검색 방식",
        "문항",
        *[f"Hit@{k}" for k in ks],
        "MRR",
        *[f"nDCG@{k}" for k in ks],
        "평균 ms",
        "p95 ms",
        f"실패@{max(ks)}",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    for summary in summaries:
        row = [
            summary.system_name,
            str(summary.question_count),
            *[
                f"{summary.hit_rates[k]:.3f}"
                for k in ks
            ],
            f"{summary.mrr:.3f}",
            *[
                f"{summary.ndcg_scores[k]:.3f}"
                for k in ks
            ],
            f"{summary.mean_latency_ms:.1f}",
            f"{summary.p95_latency_ms:.1f}",
            str(len(summary.misses_at(max(ks)))),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_evaluation_report(
    path: str | Path,
    summaries: list[RetrievalEvaluationSummary],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """재현에 필요한 설정과 문항별 결과를 UTF-8 JSON으로 저장한다."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "systems": [summary.to_dict() for summary in summaries],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_report(rows: list[dict[str, object]]) -> None:
    """기존 학습 코드에서 사용하던 단순 Markdown 출력 함수."""

    if not rows:
        print("(결과 없음)")
        return
    headers = list(rows[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    questions = load_golden_questions(
        project_root / "data" / "golden_set.jsonl"
    )
    print(
        f"골든셋 {len(questions)}문항 로드 "
        f"(normal {sum(q.type == 'normal' for q in questions)}, "
        f"no_answer {sum(q.type == 'no_answer' for q in questions)})"
    )
