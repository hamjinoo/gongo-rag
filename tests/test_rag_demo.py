"""업로드 문서를 한 번의 RAG 실행용 corpus로 준비하는 테스트."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_demo import prepare_uploaded_corpus  # noqa: E402


def test_uploaded_text_becomes_deterministic_search_corpus():
    files = [
        (
            "지원사업.txt",
            "신청 대상은 창업 3년 이내 기업입니다.\n\n지원 금액은 최대 1억원입니다.".encode("utf-8"),
        )
    ]

    first = prepare_uploaded_corpus(files)
    second = prepare_uploaded_corpus(files)

    assert len(first.documents) == 1
    assert first.documents[0].filename == "지원사업.txt"
    assert first.chunks
    assert first.chunks[0].source_filename == "지원사업.txt"
    assert first.signature == second.signature


def test_uploaded_corpus_requires_at_least_one_file():
    with pytest.raises(ValueError, match="업로드 문서"):
        prepare_uploaded_corpus([])
