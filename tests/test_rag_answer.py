"""test_rag_answer.py — 숫자 근거 일치 검사의 최소 회귀 테스트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from rag_answer import verify_citation  # noqa: E402


def test_information_not_found_passes():
    result = verify_citation("정보 없음", [])
    assert result == {"grounded": True, "missing": []}


def test_supported_number_passes_and_citation_label_is_ignored():
    chunks = ["사업화 자금은 최대 1억원까지 지원합니다."]
    result = verify_citation("지원 금액은 최대 1억원입니다. [근거 1]", chunks)
    assert result == {"grounded": True, "missing": []}


def test_unsupported_number_is_reported():
    chunks = ["사업화 자금은 최대 1억원까지 지원합니다."]
    result = verify_citation("지원 금액은 최대 2억원입니다. [근거 1]", chunks)
    assert result == {"grounded": False, "missing": ["2"]}


def test_duplicate_numbers_are_reported_once():
    chunks = ["지원 기간은 3개월입니다."]
    result = verify_citation("지원 기간은 6개월이며 6개월 연장은 없습니다.", chunks)
    assert result == {"grounded": False, "missing": ["6"]}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as error:
            print(f"  ❌ {test.__name__}: {error}")
    print(f"\n{passed}/{len(tests)} 통과")
