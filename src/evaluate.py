"""
evaluate.py — 골든셋 기반 검색 평가 (hit rate@k)  [✍️ Week 5: 직접 구현]

먼저 읽기: ../../01-basics/07-헷갈림-FAQ.md Q8 (hit rate@k란)
자가 채점:  python tests\\test_evaluate.py
GPT 규칙:  hit rate@k 전체 구현 대행 금지 (WORKFLOW.md).

골든셋 형식 (data/golden_set_example.jsonl — 예시는 가상! 내 공고문 기준으로 30~40개 직접 작성):
    {"id": "q001", "type": "normal",    "question": "...", "answer_span": "정답 근거 문장", "doc_id": "...", "note": "..."}
    {"id": "q004", "type": "no_answer", "question": "...", "answer_span": null, ...}

설계 결정 — 왜 "정답 chunk 번호"가 아니라 "정답 문자열(answer_span)"로 채점하나?
    chunk 번호는 chunking 전략을 바꾸면 전부 달라진다 (500자→800자로 바꾸면 번호가 무의미).
    "정답 문자열을 포함한 chunk가 top-k에 있는가"로 채점하면 어떤 chunking과도 비교 가능.
    → chunk size 3종 비교(실험③)가 가능한 것이 이 설계 덕분. 면접에서 말할 가치가 있는 결정!
"""
import json
from pathlib import Path


def load_golden(path: str | Path) -> list[dict]:
    """[✅ 배관] JSONL 골든셋 로드."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize(s: str) -> str:
    """[✅ 배관] 공백 정규화 — PDF 추출 텍스트는 공백/줄바꿈이 제멋대로라서,
    비교 전에 모든 연속 공백을 한 칸으로 만든다."""
    return " ".join(s.split())


# ──────────────────────────────────────────────────────────────
# ✍️ 여기부터 직접 구현
# ──────────────────────────────────────────────────────────────

def is_hit(retrieved_texts: list[str], answer_span: str) -> bool:
    """검색된 chunk들 중에 정답 근거 문자열을 포함한 것이 하나라도 있는가?

    TODO(직접 구현) 힌트:
      - 양쪽 다 normalize() 한 뒤에 'in' 으로 포함 여부 확인.
      - 왜 normalize가 필수인지: 정답은 "창업 3년 이내"인데 추출 텍스트는
        "창업 3년\\n이내"일 수 있다. (관찰노트에서 봤던 그 문제!)
    """
    raise NotImplementedError("is_hit을 직접 구현하세요")


def hit_rate_at_k(golden: list[dict], retrieve_fn, k: int = 3) -> dict:
    """골든셋 전체에 대한 hit rate@k 계산.

    retrieve_fn: (question: str, k: int) -> list[str]  형태의 함수.
        검색 방식(BM25/임베딩/hybrid)을 함수로 갈아끼우며 같은 잣대로 비교하기 위함.
        예: lambda q, k: [chunks[i]["text"] for i, _ in bm25.search(q, k)]

    TODO(직접 구현) — 단계 힌트:
      1. type이 "normal"인 문항만 대상으로 한다 (no_answer는 아래 별도 평가).
      2. 각 문항: retrieve_fn(question, k) → is_hit(결과, answer_span)
      3. 반환 예: {"k": 3, "n": 30, "hits": 24, "hit_rate": 0.8,
                   "misses": [실패한 문항 id 리스트]}
         ← misses가 곧 8주차 실패 분석의 재료. 꼭 남겨라!
    """
    raise NotImplementedError("hit_rate_at_k를 직접 구현하세요")


def no_answer_report(golden: list[dict], answer_fn) -> dict:
    """(Week 5~8 선택 과제) 답 없는 질문 처리 평가.

    answer_fn: (question: str) -> str  (RAG 최종 답변을 반환)
    아이디어: type=="no_answer" 문항에서 답변에 "정보 없음"이 포함되면 정답.
    지어내면(환각) 실패 → 그 사례가 가장 값진 기록이 된다.

    TODO(선택 구현): {"n": ..., "correct_refusals": ..., "hallucinated": [id들]}
    """
    raise NotImplementedError("no_answer_report는 5~8주차에 구현하세요 (선택)")


# ──────────────────────────────────────────────────────────────
# 완성 배관: 결과를 markdown 표로 (experiments/에 붙여넣기 좋게)
# ──────────────────────────────────────────────────────────────

def print_report(rows: list[dict]) -> None:
    """rows 예: [{"검색방식": "BM25", "k": 3, "hit_rate": 0.73, "hits": "22/30"}, ...]"""
    if not rows:
        print("(결과 없음)")
        return
    headers = list(rows[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")


if __name__ == "__main__":
    example = Path(__file__).resolve().parents[1] / "data" / "golden_set_example.jsonl"
    golden = load_golden(example)
    print(f"골든셋 예시 {len(golden)}문항 로드 (normal {sum(r['type']=='normal' for r in golden)}, "
          f"no_answer {sum(r['type']=='no_answer' for r in golden)})")
    print("→ 이 예시는 가상입니다. 내 공고문 기준으로 30~40문항을 새 파일(data/golden_set.jsonl)로 작성하세요.")
    print("→ 구현 후 채점: python tests\\test_evaluate.py")
