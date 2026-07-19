"""
test_chunker.py — chunker.py 자가 채점  [✅ 배관: 수정하지 말 것]

실행 (02-gongo-rag 폴더에서):  python tests\\test_chunker.py
pytest가 설치돼 있다면:        pytest tests\\test_chunker.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# [배관] 윈도우 콘솔/파이프에서 ✅ 이모지 출력이 UnicodeEncodeError로 죽는 것 방지 (채점과 무관)
sys.stdout.reconfigure(encoding="utf-8")

from chunker import chunk_fixed, chunk_by_paragraph  # noqa: E402

SAMPLE = "가나다라마바사아자차카타파하" * 100  # 1400자, 문단 없음
PARA_SAMPLE = (
    "제1장 신청 자격\n\n"
    "만 39세 이하의 예비창업자 또는 창업 3년 이내 기업이어야 합니다.\n\n"
    "제2장 지원 내용\n\n"
    "사업화 자금을 최대 1억원까지 지원합니다.\n\n"
    "문의처\n\n"
    "창업진흥원 000-0000\n"
)


def test_fixed_basic():
    chunks = chunk_fixed(SAMPLE, doc_id="t", chunk_size=300, overlap=50)
    assert len(chunks) >= 2, "1400자를 300자로 잘랐는데 조각이 2개 미만?"
    assert all(c["text"] for c in chunks), "빈 조각이 있으면 안 됩니다"
    assert chunks[0]["start"] == 0 and chunks[0]["text"] == SAMPLE[:300]
    assert all(len(c["text"]) <= 300 for c in chunks), "chunk_size 초과 조각 발견"
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "id가 중복됩니다"


def test_fixed_full_coverage():
    """모든 글자가 최소 한 조각에는 포함돼야 한다 (정보 유실 금지)."""
    chunks = chunk_fixed(SAMPLE, chunk_size=300, overlap=50)
    covered = set()
    for c in chunks:
        covered.update(range(c["start"], c["start"] + len(c["text"])))
    missing = set(range(len(SAMPLE))) - covered
    assert not missing, f"원문에서 유실된 위치 {len(missing)}곳 (예: {sorted(missing)[:5]})"
    # 조각 내용과 원문 위치가 실제로 일치하는지
    for c in chunks:
        assert SAMPLE[c["start"]:c["start"] + len(c["text"])] == c["text"], \
            f"{c['id']}: start 위치와 text 내용이 불일치"


def test_fixed_overlap():
    """이웃 조각은 overlap만큼 겹쳐야 한다 (마지막 조각 제외)."""
    chunks = chunk_fixed(SAMPLE, chunk_size=300, overlap=50)
    for a, b in zip(chunks[:-2], chunks[1:-1]):  # 마지막 쌍은 길이가 다를 수 있어 제외
        assert b["start"] == a["start"] + 250, \
            f"시작 간격이 chunk_size-overlap(250)이어야 함: {a['id']}→{b['id']}"


def test_paragraph_basic():
    chunks = chunk_by_paragraph(PARA_SAMPLE, doc_id="p", max_chars=200)
    assert chunks, "조각이 하나도 안 나왔습니다"
    assert all(c["text"].strip() for c in chunks), "빈 조각 금지"
    joined = " ".join(c["text"] for c in chunks)
    for key in ["만 39세 이하", "최대 1억원", "창업진흥원"]:
        assert key in joined, f"문단 내용 유실: {key!r}"


def test_paragraph_oversize():
    """max_chars를 크게 넘는 문단은 쪼개져야 한다."""
    big = "매우 긴 문단입니다. " * 100  # 약 1200자, 문단 구분 없음
    chunks = chunk_by_paragraph(big, max_chars=300)
    assert all(len(c["text"]) <= 450 for c in chunks), \
        "한 문단이 max_chars(300)를 한참 넘는데 쪼개지지 않았습니다 (힌트: chunk_fixed 재사용)"


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
