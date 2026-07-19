"""
test_bm25.py — bm25.py 자가 채점  [✅ 배관: 수정하지 말 것]

실행 (02-gongo-rag 폴더에서):  python tests\\test_bm25.py
기준값은 04-concepts/BM25-완전정복.md 의 손계산 예제와 동일.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# [배관] 윈도우 콘솔/파이프에서 ✅ 이모지 출력이 UnicodeEncodeError로 죽는 것 방지 (채점과 무관)
sys.stdout.reconfigure(encoding="utf-8")

from bm25 import BM25  # noqa: E402

CORPUS = [
    "청년 창업 지원 사업 공고",   # D1
    "창업 기업 지원 금액 안내",   # D2
    "청년 주택 정책 안내",       # D3
]


def test_idf_rare_beats_common():
    """희귀 단어(금액, df=1)의 IDF > 흔한 단어(청년, df=2)의 IDF."""
    bm25 = BM25(CORPUS)
    assert bm25.idf("금액") > bm25.idf("청년"), \
        "희귀한 단어가 더 높은 IDF를 가져야 합니다 (df 세는 법 확인: 문서당 1회!)"


def test_idf_hand_calculated():
    """손계산 값과 비교 (BM25-완전정복.md ②단계)."""
    bm25 = BM25(CORPUS)
    assert abs(bm25.idf("청년") - 0.470) < 0.01, f"IDF(청년)={bm25.idf('청년'):.3f}, 기대 0.470"
    assert abs(bm25.idf("금액") - 0.981) < 0.01, f"IDF(금액)={bm25.idf('금액'):.3f}, 기대 0.981"


def test_ranking_hand_calculated():
    """질문 '청년 지원 금액' → D2 > D1 > D3 (희귀 단어 '금액'이 승부를 가름)."""
    bm25 = BM25(CORPUS)
    results = bm25.search("청년 지원 금액", k=3)
    order = [idx for idx, _ in results]
    assert order[0] == 1, f"1위가 D2(인덱스 1)여야 합니다. 현재: D{order[0]+1}"
    assert order == [1, 0, 2], f"기대 순위 D2>D1>D3, 현재: {[f'D{i+1}' for i in order]}"
    scores = dict(results)
    assert abs(scores[1] - 1.406) < 0.02, f"score(D2)={scores[1]:.3f}, 기대 ≈1.406"
    assert abs(scores[0] - 0.911) < 0.02, f"score(D1)={scores[0]:.3f}, 기대 ≈0.911"


def test_absent_word_zero():
    """코퍼스에 없는 단어만으로 이뤄진 질문은 모든 점수가 0이어야 한다."""
    bm25 = BM25(CORPUS)
    results = bm25.search("김치찌개", k=3)
    assert all(abs(s) < 1e-9 for _, s in results), \
        "없는 단어의 점수 기여는 0이어야 합니다 (f=0인 토큰은 건너뛰기)"


def test_search_k():
    bm25 = BM25(CORPUS)
    assert len(bm25.search("창업", k=2)) == 2, "k=2면 2개만 반환해야 합니다"


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
