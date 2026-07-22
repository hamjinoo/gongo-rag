"""잠근 검색 파이프라인과 LangGraph RAG workflow를 실행한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from chunker import ChunkingConfig, DocumentChunk
from local_llm import get_ollama_status
from rag_workflow import RAGWorkflow, RAGWorkflowConfig
from retrieval_trace import TracedReranker, trace_reranker
from reranker import DEFAULT_RERANKER_MODEL
from run_retrieval_evaluation import (
    build_retrievers,
    configure_utf8_console,
    load_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BGE 검색 결과를 LangGraph로 판단·재검색·답변합니다.",
    )
    parser.add_argument("question", help="공고문에 물어볼 한국어 질문")
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "text",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=PROJECT_ROOT / ".chroma" / "rag-workflow",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rewrites", type=int, default=1)
    parser.add_argument(
        "--json",
        action="store_true",
        help="사람용 출력 대신 전체 상태를 JSON으로 출력",
    )
    return parser.parse_args()


def build_locked_reranker(
    *,
    text_dir: Path,
    persist_directory: Path,
) -> TracedReranker:
    """dev로 선택하고 test 한 번으로 확인한 검색 설정을 그대로 만든다."""

    _, chunks = load_corpus(
        text_dir,
        chunking_config=ChunkingConfig(),
    )
    return build_locked_reranker_for_chunks(
        chunks,
        persist_directory=persist_directory,
    )


def build_locked_reranker_for_chunks(
    chunks: list[DocumentChunk],
    *,
    persist_directory: Path,
) -> TracedReranker:
    """업로드 문서 Chunk에도 평가에서 잠근 동일한 검색 설정을 적용한다."""

    retrievers = build_retrievers(
        chunks,
        ("reranker",),
        persist_directory=persist_directory,
        rerank_candidates=7,
        rerank_batch_size=2,
        rerank_max_length=512,
        reranker_model=DEFAULT_RERANKER_MODEL,
    )
    traced = trace_reranker(retrievers["Reranker"])
    traced.source_chunks = tuple(chunks)
    return traced


def main() -> None:
    configure_utf8_console()
    load_dotenv()
    args = parse_args()
    local_llm_status = get_ollama_status(timeout_seconds=1.0)
    if not local_llm_status.ready:
        raise SystemExit(local_llm_status.message)

    workflow = RAGWorkflow(
        build_locked_reranker(
            text_dir=args.text_dir,
            persist_directory=args.persist_directory,
        ),
        config=RAGWorkflowConfig(
            top_k=args.top_k,
            max_rewrites=args.max_rewrites,
        ),
    )
    response = workflow.invoke(args.question)

    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"질문: {response.question}")
    if response.final_query != response.question:
        print(f"재작성 질문: {response.final_query}")
    print(f"경로: {' → '.join(response.steps)}")
    print(f"판정 이유: {response.decision_reason}")
    print(f"\n답변: {response.answer}")
    if response.refusal_reason:
        print(f"거절 이유: {response.refusal_reason}")
    if response.evidence:
        print("\n최종 근거:")
        for item in response.evidence:
            print(
                f"- [근거 {item['rank']}] {item['source_filename']} · "
                f"{item['page_label']} · {item['chunk_id']}"
            )


if __name__ == "__main__":
    main()
