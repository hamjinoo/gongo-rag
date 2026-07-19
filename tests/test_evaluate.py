"""
test_evaluate.py — evaluate.py 자가 채점  [✅ 배관: 수정하지 말 것]

실행 (02-gongo-rag 폴더에서):  python tests\\test_evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# [배관] 윈도우 콘솔/파이프에서 ✅ 이모지 출력이 UnicodeEncodeError로 죽는 것 방지 (채점과 무관)
sys.stdout.reconfigure(encoding="utf-8")

from evaluate import is_hit, hit_rate_at_k  # noqa: E402


def test_is_hit_basic():
    chunks = ["신청 자격: 만 39세 이하 예비창업자", "지원 금액은 최대 1억원입니다"]
    assert is_hit(chunks, "만 39세 이하") is True
    assert is_hit(chunks, "사무실 제공") is False


def test_is_hit_whitespace():
    """PDF 추출 텍스트의 제멋대로 공백/줄바꿈에도 매칭돼야 한다 (normalize!)."""
    chunks = ["자격 요건은 창업 3년\n이내   기업입니다"]
    assert is_hit(chunks, "창업 3년 이내 기업") is True, \
        "줄바꿈/연속 공백 차이로 놓치면 안 됩니다 (힌트: 양쪽 다 normalize)"


def test_hit_rate_two_thirds():
    """3문항 중 2문항 hit → hit_rate 2/3. no_answer 문항은 계산에서 제외."""
    golden = [
        {"id": "q1", "type": "normal", "question": "자격은?", "answer_span": "만 39세 이하"},
        {"id": "q2", "type": "normal", "question": "금액은?", "answer_span": "최대 1억원"},
        {"id": "q3", "type": "normal", "question": "기한은?", "answer_span": "7월 31일"},
        {"id": "q4", "type": "no_answer", "question": "사무실은?", "answer_span": None},
    ]

    def fake_retrieve(question: str, k: int) -> list[str]:
        # q1, q2의 정답만 찾아주고 q3(기한)은 못 찾는 가짜 검색기
        db = {"자격은?": ["신청 자격: 만 39세 이하"],
              "금액은?": ["지원 금액은 최대 1억원"],
              "기한은?": ["엉뚱한 내용의 chunk"]}
        return db.get(question, ["무관한 chunk"])[:k]

    result = hit_rate_at_k(golden, fake_retrieve, k=3)
    assert result["n"] == 3, f"normal 문항만 세야 합니다 (no_answer 제외). n={result['n']}"
    assert result["hits"] == 2, f"hits={result['hits']}, 기대 2"
    assert abs(result["hit_rate"] - 2 / 3) < 1e-9, f"hit_rate={result['hit_rate']}, 기대 0.667"
    assert result["misses"] == ["q3"], f"실패 문항 id 기록 필요. misses={result.get('misses')}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except NotImplementedError as e:
            print(f"  ⏳ {t.__name__}: 아직 구현 전 ({e})")
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 통과")
