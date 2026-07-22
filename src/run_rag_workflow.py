"""잠근 검색 파이프라인과 LangGraph RAG workflow를 실행한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from chunker import ChunkingConfig
from rag_workflow import RAGWorkflow, RAGWorkflowConfig
from reranker import (
    DEFAULT_RERANKER_MODEL,
    LOCAL_RERANKER_PROVIDER,
)
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
):
    """dev로 선택하고 test 한 번으로 확인한 검색 설정을 그대로 만든다."""

    _, chunks = load_corpus(
        text_dir,
        chunking_config=ChunkingConfig(),
    )
    retrievers = build_retrievers(
        chunks,
        ("reranker",),
        persist_directory=persist_directory,
        rerank_candidates=7,
        rerank_batch_size=2,
        rerank_max_length=512,
        reranker_provider=LOCAL_RERANKER_PROVIDER,
        reranker_model=DEFAULT_RERANKER_MODEL,
    )
    return retrievers["Reranker"]


def main() -> None:
    configure_utf8_console()
    load_dotenv()
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY가 필요합니다. .env.example을 .env로 복사하고 "
            "OPENAI_API_KEY만 채워주세요."
        )

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
