"""BM25·Chroma·RRF·reranker를 같은 골든셋으로 비교 실행한다."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from bm25 import BM25ChunkRetriever
from chunker import ChunkingConfig, DocumentChunk, chunk_documents
from document_ingestion import ExtractedDocument, extract_document_from_path
from evaluate import (
    EvaluationError,
    GoldenQuestion,
    RankedRetriever,
    RetrievalEvaluationSummary,
    evaluate_retriever,
    load_golden_questions,
    render_comparison_markdown,
    save_evaluation_report,
    validate_golden_set,
)
from hybrid_search import DEFAULT_FETCH_K, DEFAULT_RANK_CONSTANT, HybridRRFRetriever
from reranker import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_SPECS,
    CrossEncoderReranker,
    SentenceTransformersCrossEncoderScorer,
    get_reranker_model_spec,
)
from vector_search import DEFAULT_EMBEDDING_MODEL, ChromaChunkRetriever


SYSTEM_NAMES = ("bm25", "chroma", "rrf", "reranker")


def configure_utf8_console() -> None:
    """Windows의 비 UTF-8 콘솔·파이프에서도 한글 상태 문구를 안전하게 출력한다."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "같은 질문·정답 Chunk로 BM25, Chroma, RRF, reranker의 "
            "순위 품질과 지연을 비교합니다."
        )
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="기본값 dev. 튜닝 중에는 test를 반복 실행하지 않습니다.",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEM_NAMES),
        help="쉼표로 구분: bm25,chroma,rrf,reranker",
    )
    parser.add_argument(
        "--ks",
        default="1,3,5",
        help=(
            "평가할 순위 컷오프. 기본값 1,3,5. Hit@10은 "
            "--rerank-candidates 10 이상과 함께 사용합니다."
        ),
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/golden_set.jsonl"),
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("docs/text"),
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=Path(".chroma"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON 결과 경로. 기본값 experiments/retrieval-evaluation-{split}.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Markdown 결과 경로. 기본값 experiments/retrieval-evaluation-{split}.md",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
    )
    parser.add_argument(
        "--reranker-model",
        choices=tuple(RERANKER_MODEL_SPECS),
        default=DEFAULT_RERANKER_MODEL,
        help="revision을 고정하고 검토한 로컬 reranker 모델",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--rerank-max-length",
        type=int,
        default=DEFAULT_RERANK_MAX_LENGTH,
    )
    return parser.parse_args()


def parse_systems(raw_value: str) -> tuple[str, ...]:
    systems = tuple(
        value.strip().lower()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not systems:
        raise ValueError("적어도 한 검색 방식을 선택해야 합니다.")
    unknown = sorted(set(systems) - set(SYSTEM_NAMES))
    if unknown:
        raise ValueError(f"지원하지 않는 검색 방식: {', '.join(unknown)}")
    if len(systems) != len(set(systems)):
        raise ValueError("검색 방식은 중복될 수 없습니다.")
    return systems


def parse_ks(raw_value: str) -> tuple[int, ...]:
    try:
        ks = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in raw_value.split(",")
                    if value.strip()
                }
            )
        )
    except ValueError as exc:
        raise ValueError("ks는 쉼표로 구분한 정수여야 합니다.") from exc
    if not ks or ks[0] < 1:
        raise ValueError("ks에는 1 이상의 값이 필요합니다.")
    return ks


def validate_run_settings(
    systems: tuple[str, ...],
    ks: tuple[int, ...],
    *,
    rerank_candidates: int,
    rerank_batch_size: int,
    rerank_max_length: int,
) -> None:
    """겉보기 설정과 실제 reranker 후보 수가 달라지는 실험을 막는다."""

    if rerank_candidates < 1:
        raise ValueError("rerank-candidates는 1 이상이어야 합니다.")
    if rerank_batch_size < 1:
        raise ValueError("rerank-batch-size는 1 이상이어야 합니다.")
    if rerank_max_length < 1:
        raise ValueError("rerank-max-length는 1 이상이어야 합니다.")
    if "reranker" in systems and rerank_candidates < max(ks):
        raise ValueError(
            "rerank-candidates는 가장 큰 평가 k 이상이어야 합니다. "
            "후보 5개 실험은 --ks 1,3,5를 사용하세요."
        )


def load_corpus(
    text_directory: Path,
    *,
    chunking_config: ChunkingConfig,
) -> tuple[list[ExtractedDocument], list[DocumentChunk]]:
    paths = sorted(text_directory.glob("*.txt"))
    if not paths:
        raise EvaluationError(
            f"평가할 TXT 문서가 없습니다: {text_directory}"
        )
    documents = [extract_document_from_path(path) for path in paths]
    chunks = chunk_documents(documents, config=chunking_config)
    if not chunks:
        raise EvaluationError("평가할 Chunk가 없습니다.")
    return documents, chunks


def build_retrievers(
    chunks: list[DocumentChunk],
    systems: tuple[str, ...],
    *,
    persist_directory: Path,
    rerank_candidates: int,
    rerank_batch_size: int,
    rerank_max_length: int,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, RankedRetriever]:
    retrievers: dict[str, RankedRetriever] = {}
    needs_hybrid = any(name in systems for name in ("rrf", "reranker"))
    needs_bm25 = "bm25" in systems or needs_hybrid
    needs_chroma = "chroma" in systems or needs_hybrid

    bm25 = (
        BM25ChunkRetriever(chunks, tokenizer_name="kiwi")
        if needs_bm25
        else None
    )
    chroma = (
        ChromaChunkRetriever(
            chunks,
            model_name=DEFAULT_EMBEDDING_MODEL,
            persist_directory=persist_directory,
        )
        if needs_chroma
        else None
    )
    hybrid = (
        HybridRRFRetriever(
            bm25,
            chroma,
            rank_constant=DEFAULT_RANK_CONSTANT,
            fetch_k=DEFAULT_FETCH_K,
        )
        if needs_hybrid and bm25 is not None and chroma is not None
        else None
    )

    if "bm25" in systems and bm25 is not None:
        retrievers["BM25"] = bm25
    if "chroma" in systems and chroma is not None:
        retrievers["Chroma"] = chroma
    if "rrf" in systems and hybrid is not None:
        retrievers["RRF"] = hybrid
    if "reranker" in systems and hybrid is not None:
        scorer = SentenceTransformersCrossEncoderScorer(
            model_name=reranker_model,
            batch_size=rerank_batch_size,
            max_length=rerank_max_length,
            device="cpu",
        )
        retrievers["Reranker"] = CrossEncoderReranker(
            hybrid,
            scorer,
            candidate_k=rerank_candidates,
        )
    return retrievers


def evaluate_systems(
    retrievers: dict[str, RankedRetriever],
    questions: list[GoldenQuestion],
    chunks: list[DocumentChunk],
    *,
    ks: tuple[int, ...],
) -> list[RetrievalEvaluationSummary]:
    summaries: list[RetrievalEvaluationSummary] = []
    for system_name, retriever in retrievers.items():
        print(f"[평가] {system_name} · {len(questions)}개 전체 문항 중 normal만 사용")
        summary = evaluate_retriever(
            system_name,
            retriever,
            questions,
            chunks,
            ks=ks,
        )
        summaries.append(summary)
        print(
            f"  Hit@1 {summary.hit_rates.get(1, 0):.3f} · "
            f"MRR {summary.mrr:.3f} · "
            f"평균 {summary.mean_latency_ms:.1f}ms"
        )
    return summaries


def build_metadata(
    *,
    args: argparse.Namespace,
    systems: tuple[str, ...],
    ks: tuple[int, ...],
    questions: list[GoldenQuestion],
    documents: list[ExtractedDocument],
    chunks: list[DocumentChunk],
    chunking_config: ChunkingConfig,
    retrievers: dict[str, RankedRetriever],
) -> dict[str, object]:
    reranker_spec = get_reranker_model_spec(args.reranker_model)
    reranker_runtime: dict[str, object] = {}
    reranker = retrievers.get("Reranker")
    if isinstance(reranker, CrossEncoderReranker) and isinstance(
        reranker.scorer,
        SentenceTransformersCrossEncoderScorer,
    ):
        scorer = reranker.scorer
        reranker_runtime = {
            "model_load_seconds": scorer.model_load_seconds,
            "model_rss_delta_mb": scorer.model_rss_delta_mb,
            "process_rss_after_load_mb": scorer.process_rss_after_load_mb,
            "peak_process_rss_mb": scorer.peak_process_rss_mb,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "systems": list(systems),
        "ks": list(ks),
        "golden_path": str(args.golden),
        "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        "question_counts": {
            "total": len(questions),
            "normal": sum(q.type == "normal" for q in questions),
            "no_answer": sum(q.type == "no_answer" for q in questions),
        },
        "documents": [
            {
                "filename": document.filename,
                "sha256": document.source_sha256,
            }
            for document in documents
        ],
        "chunking": {
            "strategy": chunking_config.strategy,
            "chunk_size": chunking_config.chunk_size,
            "overlap": chunking_config.overlap,
            "chunk_count": len(chunks),
        },
        "retrieval": {
            "bm25_tokenizer": "kiwi",
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "rrf_rank_constant": DEFAULT_RANK_CONSTANT,
            "rrf_fetch_k": DEFAULT_FETCH_K,
            "reranker_model": reranker_spec.model_name,
            "reranker_model_revision": reranker_spec.revision,
            "reranker_code_revision": reranker_spec.code_revision,
            "reranker_parameter_count": reranker_spec.parameter_count,
            "reranker_languages": reranker_spec.languages,
            "reranker_license": reranker_spec.license_name,
            "reranker_trust_remote_code": reranker_spec.trust_remote_code,
            "reranker_candidates": args.rerank_candidates,
            "reranker_batch_size": args.rerank_batch_size,
            "reranker_max_length": args.rerank_max_length,
        },
        "reranker_runtime": reranker_runtime,
        "note": (
            "검색 평가는 normal 문항만 사용합니다. no_answer는 최종 답변의 "
            "거절 평가에서 별도로 사용합니다."
        ),
    }


def render_full_markdown_report(
    summaries: list[RetrievalEvaluationSummary],
    metadata: dict[str, object],
) -> str:
    max_k = max(summaries[0].ks)
    question_counts = metadata["question_counts"]
    chunking = metadata["chunking"]
    retrieval = metadata["retrieval"]
    reranker_runtime = metadata.get("reranker_runtime", {})
    if not isinstance(question_counts, dict):
        raise EvaluationError("question_counts metadata 형식이 잘못됐습니다.")
    if not isinstance(chunking, dict):
        raise EvaluationError("chunking metadata 형식이 잘못됐습니다.")
    if not isinstance(retrieval, dict):
        raise EvaluationError("retrieval metadata 형식이 잘못됐습니다.")
    if not isinstance(reranker_runtime, dict):
        raise EvaluationError("reranker_runtime metadata 형식이 잘못됐습니다.")

    runtime_lines: list[str] = []
    if reranker_runtime:
        load_seconds = reranker_runtime.get("model_load_seconds")
        rss_delta = reranker_runtime.get("model_rss_delta_mb")
        peak_rss = reranker_runtime.get("peak_process_rss_mb")
        if isinstance(load_seconds, (int, float)):
            runtime_lines.append(f"- 모델 로드: {load_seconds:.2f}초")
        if isinstance(rss_delta, (int, float)):
            runtime_lines.append(f"- 모델 로드 RSS 증가: {rss_delta:.1f}MB")
        if isinstance(peak_rss, (int, float)):
            runtime_lines.append(f"- 평가 중 프로세스 최대 RSS: {peak_rss:.1f}MB")

    best_hit1 = max(summaries, key=lambda summary: summary.hit_rates.get(1, 0))
    fastest = min(summaries, key=lambda summary: summary.mean_latency_ms)
    lines = [
        f"# 검색 평가 결과 · {metadata['split']} split",
        "",
        "> 동일한 문서, Chunk, 질문, 정답 근거로 검색 단계만 비교한 결과입니다.",
        "",
        "## 실행 조건",
        "",
        (
            f"- 질문: 전체 {question_counts['total']}개 "
            f"(검색 평가 normal {question_counts['normal']}개, "
            f"별도 거절 평가 no-answer {question_counts['no_answer']}개)"
        ),
        (
            f"- Chunk: `{chunking['strategy']}` "
            f"{chunking['chunk_size']}/{chunking['overlap']} · "
            f"총 {chunking['chunk_count']}개"
        ),
        f"- BM25 tokenizer: `{retrieval['bm25_tokenizer']}`",
        f"- Embedding: `{retrieval['embedding_model']}`",
        (
            f"- RRF: rank constant {retrieval['rrf_rank_constant']} · "
            f"검색기별 후보 {retrieval['rrf_fetch_k']}개"
        ),
        (
            f"- Reranker: `{retrieval['reranker_model']}` · "
            f"{retrieval['reranker_parameter_count'] / 1_000_000:.0f}M params · "
            f"{retrieval['reranker_license']} · "
            f"후보 {retrieval['reranker_candidates']}개 · "
            f"batch {retrieval['reranker_batch_size']} · "
            f"max length {retrieval['reranker_max_length']}"
        ),
        *runtime_lines,
        f"- 골든셋 SHA-256: `{metadata['golden_sha256']}`",
        "",
        "## 비교 결과",
        "",
        render_comparison_markdown(summaries),
        "",
        "지연 시간은 현재 컴퓨터의 참고값이며 품질 metric과 분리해 해석합니다.",
        "",
        "## 핵심 관찰",
        "",
        (
            f"- Hit@1 최고: **{best_hit1.system_name} "
            f"{best_hit1.hit_rates.get(1, 0):.3f}**"
        ),
        (
            f"- 평균 지연 최저: **{fastest.system_name} "
            f"{fastest.mean_latency_ms:.1f}ms**"
        ),
        (
            "- 품질이 가장 높은 방식과 가장 빠른 방식이 다르므로, "
            "정확도와 응답 시간 요구를 함께 보고 선택합니다."
        ),
        "",
        "## 첫 정답 순위가 달라진 문항",
        "",
        *_render_changed_rank_table(summaries),
        "",
        f"## 시스템별 Hit@{max_k} 실패 문항",
        "",
    ]
    for summary in summaries:
        misses = summary.misses_at(max_k)
        lines.extend(
            [
                f"### {summary.system_name}",
                "",
                ", ".join(f"`{question_id}`" for question_id in misses)
                if misses
                else "없음",
                "",
            ]
        )
    lines.extend(
        [
            "## 읽는 방법",
            "",
            "- Hit@k: 상위 k개 안에 정답 Chunk가 들어간 질문의 비율",
            "- MRR: 첫 정답이 위에 있을수록 커지는 평균 역순위",
            "- nDCG@k: 여러 정답 Chunk가 있을 때도 위쪽 배치를 더 높게 평가",
            "- 평균/p95 ms: 품질과 함께 보는 응답 지연 참고값",
            "",
            "dev 결과로 설정을 고르고, test split은 최종 선택 뒤 한 번 확인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_changed_rank_table(
    summaries: list[RetrievalEvaluationSummary],
) -> list[str]:
    system_names = [summary.system_name for summary in summaries]
    cases_by_system = {
        summary.system_name: {
            case.question_id: case
            for case in summary.cases
        }
        for summary in summaries
    }
    question_ids = [
        case.question_id
        for case in summaries[0].cases
    ]
    changed_rows: list[list[str]] = []

    for question_id in question_ids:
        ranks = [
            cases_by_system[name][question_id].first_relevant_rank
            for name in system_names
        ]
        if len(set(ranks)) <= 1:
            continue
        changed_rows.append(
            [
                question_id,
                *[
                    str(rank) if rank is not None else "-"
                    for rank in ranks
                ],
            ]
        )

    if not changed_rows:
        return ["모든 시스템의 첫 정답 순위가 같았습니다."]

    headers = ["문항", *system_names]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in changed_rows
    )
    return lines


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    systems = parse_systems(args.systems)
    ks = parse_ks(args.ks)
    validate_run_settings(
        systems,
        ks,
        rerank_candidates=args.rerank_candidates,
        rerank_batch_size=args.rerank_batch_size,
        rerank_max_length=args.rerank_max_length,
    )

    split_filter = None if args.split == "all" else args.split
    questions = load_golden_questions(args.golden, split=split_filter)
    if not questions:
        raise EvaluationError(f"{args.split} split에 질문이 없습니다.")

    chunking_config = ChunkingConfig()
    documents, chunks = load_corpus(
        args.text_dir,
        chunking_config=chunking_config,
    )
    validation = validate_golden_set(questions, chunks)
    print(
        f"[골든셋] total={validation.total_questions} "
        f"normal={validation.normal_questions} "
        f"no_answer={validation.no_answer_questions} "
        f"chunks={len(chunks)}"
    )

    retrievers = build_retrievers(
        chunks,
        systems,
        persist_directory=args.persist_directory,
        rerank_candidates=args.rerank_candidates,
        rerank_batch_size=args.rerank_batch_size,
        rerank_max_length=args.rerank_max_length,
        reranker_model=args.reranker_model,
    )
    summaries = evaluate_systems(
        retrievers,
        questions,
        chunks,
        ks=ks,
    )
    metadata = build_metadata(
        args=args,
        systems=systems,
        ks=ks,
        questions=questions,
        documents=documents,
        chunks=chunks,
        chunking_config=chunking_config,
        retrievers=retrievers,
    )

    output = args.output or Path(
        f"experiments/retrieval-evaluation-{args.split}.json"
    )
    markdown_output = args.markdown_output or Path(
        f"experiments/retrieval-evaluation-{args.split}.md"
    )
    save_evaluation_report(output, summaries, metadata=metadata)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_full_markdown_report(summaries, metadata),
        encoding="utf-8",
    )

    print()
    print(render_comparison_markdown(summaries))
    print(f"\nJSON: {output}")
    print(f"Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
