"""단계별 검색 평가 CLI와 리포트 테스트."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import ChunkingConfig  # noqa: E402
from run_retrieval_evaluation import (  # noqa: E402
    build_retrievers,
    load_corpus,
    parse_ks,
    parse_systems,
)


def test_parse_systems_keeps_requested_order():
    assert parse_systems("rrf,bm25") == ("rrf", "bm25")


def test_parse_systems_rejects_unknown_and_duplicate_values():
    for raw_value in ("bm25,unknown", "bm25,bm25", ""):
        try:
            parse_systems(raw_value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 검색 방식 목록을 거부해야 합니다: {raw_value}")


def test_parse_ks_sorts_and_deduplicates_positive_values():
    assert parse_ks("10,1,3,3") == (1, 3, 10)
    for raw_value in ("", "0,3", "one,3"):
        try:
            parse_ks(raw_value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 k 목록을 거부해야 합니다: {raw_value}")


def test_real_text_corpus_builds_expected_bm25_only_retriever():
    _, chunks = load_corpus(
        PROJECT_ROOT / "docs" / "text",
        chunking_config=ChunkingConfig(),
    )

    retrievers = build_retrievers(
        chunks,
        ("bm25",),
        persist_directory=PROJECT_ROOT / ".chroma",
        rerank_candidates=10,
        rerank_batch_size=2,
        rerank_max_length=512,
    )

    assert len(chunks) == 38
    assert list(retrievers) == ["BM25"]


def test_bm25_cli_writes_auditable_json_and_markdown_reports():
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "result.json"
        markdown = Path(temp_dir) / "result.md"
        completed = subprocess.run(
            [
                str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
                str(PROJECT_ROOT / "src" / "run_retrieval_evaluation.py"),
                "--split",
                "dev",
                "--systems",
                "bm25",
                "--ks",
                "1,3",
                "--golden",
                str(PROJECT_ROOT / "data" / "golden_set.jsonl"),
                "--text-dir",
                str(PROJECT_ROOT / "docs" / "text"),
                "--persist-directory",
                str(PROJECT_ROOT / ".chroma"),
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        report = markdown.read_text(encoding="utf-8")

    assert payload["metadata"]["split"] == "dev"
    assert payload["metadata"]["question_counts"] == {
        "total": 23,
        "normal": 20,
        "no_answer": 3,
    }
    assert [system["system_name"] for system in payload["systems"]] == ["BM25"]
    assert len(payload["systems"][0]["cases"]) == 20
    assert "검색 평가 결과 · dev split" in report
    assert "Hit@1" in report
    assert "test split은 최종 선택 뒤 한 번" in report


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
