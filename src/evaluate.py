"""
evaluate.py — 골든셋 기반 검색 평가 (hit rate@k)  [✍️ Week 5: 직접 구현]

먼저 읽기: ../../01-basics/07-헷갈림-FAQ.md Q8 (hit rate@k란)
자가 채점:  python tests\\test_evaluate.py
GPT 규칙:  hit rate@k 전체 구현 대행 금지 (WORKFLOW.md).

이 파일을 한 문장으로: "검색기에게 시험을 보게 하고, 점수를 매기는 채점기".
    골든셋      = 내가 미리 만들어 둔 시험지 (질문 + 그 답이 적힌 원문 문장).
    hit rate@k  = 검색기가 가져온 상위 k개 조각 안에 정답 조각이 든 비율.
                  30문제 중 24문제 성공이면 24/30 = 0.8.

골든셋 형식 (data/golden_set_example.jsonl — 예시는 가상! 내 공고문 기준으로 30~40개 직접 작성):
    {"id": "q001", "type": "normal",    "question": "...", "answer_span": "정답 근거 문장", "doc_id": "...", "note": "..."}
    {"id": "q004", "type": "no_answer", "question": "...", "answer_span": null, ...}

설계 결정 — 왜 "정답은 몇 번째 조각"이 아니라 "정답 문장(answer_span)"으로 채점하나?
    조각 번호는 자르는 방법을 바꾸면 전부 달라진다 (500자→800자로 바꾸면 번호가 무의미).
    "정답 문장이 들어있는 조각을 찾아왔는가"로 채점하면 어떤 자르기 방법과도 비교 가능.
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
    """보물찾기 판정: 검색이 가져온 조각들 중 정답 문장이 적힌 게 하나라도 있으면 True.

    TODO(직접 구현) 힌트:
      - "이 문자열 안에 저 문자열이 들어있나?"는 파이썬에서  작은것 in 큰것  으로 확인.
      - 단, 비교 전에 양쪽 다 normalize()로 공백을 청소할 것.
        왜냐하면: 정답은 "창업 3년 이내"인데 PDF에서 뽑은 텍스트는
        "창업 3년\\n이내"처럼 중간에 줄바꿈이 끼어 있을 수 있어서요.
        글자는 똑같은데 공백 때문에 "없다"고 판정하면 억울하잖아요. (관찰노트의 그 문제!)
    """
    answer = normalize(answer_span)

    for text in retrieved_texts:
        nor_text = normalize(text)
        if answer in nor_text:
            return True
    return False

def hit_rate_at_k(golden: list[dict], retrieve_fn, k: int = 3) -> dict:
    """시험지(골든셋) 전체를 풀게 하고 점수 내기: 성공 횟수 ÷ 문제 수.

    retrieve_fn = "검색기"를 함수 모양으로 받은 것.
        (question, k)를 주면 조각(문자열) k개를 돌려주는 함수여야 함.
        왜 함수로 받나: BM25 검색기든 임베딩 검색기든 여기에 갈아끼우면서
        똑같은 시험지로 공평하게 비교하려고.
        예: lambda q, k: [chunks[i]["text"] for i, _ in bm25.search(q, k)]

    TODO(직접 구현) — 단계 힌트:
      1. type이 "normal"인 문항만 채점한다 (no_answer는 다른 시험이라 건너뜀).
      2. 각 문항마다: retrieve_fn(question, k)으로 조각들을 받아와 → is_hit으로 판정.
      3. 반환 형식 (테스트가 이 키 이름들을 그대로 확인합니다):
         {"k": 3, "n": 30, "hits": 24, "hit_rate": 0.8,
          "misses": [틀린 문항의 id 리스트]}
         ← misses(틀린 문제 목록)가 곧 8주차 실패 분석의 재료. 꼭 남겨라!
    """
    n = 0
    hits = 0
    hit_rate = 0
    misses = []
    for data in golden:
        if data["type"] == 'normal':
            n += 1

            retrieve = retrieve_fn(data["question"], k)
            # print(f"retrieve {retrieve} {data}")

            hit = is_hit(retrieve, data["answer_span"])
            if hit == True:
                hits += 1
            else:
                misses.append(data["id"])

    hit_rate = hits / n
    result = {
        "k": k,
        "n": n,
        "hits": hits,
        "hit_rate": hit_rate,
        "misses": misses
    }

    return result



def no_answer_report(golden: list[dict], answer_fn) -> dict:
    """(Week 5~8 선택 과제) "답이 없는 질문"을 제대로 거절하는지 평가.

    answer_fn: (question: str) -> str  (RAG의 최종 답변을 돌려주는 함수)
    아이디어: 공고문에 없는 걸 물었을 때(type=="no_answer") "정보 없음"이라고
    답하면 정답. 모르면서 그럴듯하게 지어내면(환각) 실패 →
    그 지어낸 사례가 가장 값진 기록이 된다.

    TODO(선택 구현): {"n": ..., "correct_refusals": ..., "hallucinated": [id들]}
    """

    n = 0
    correct_refusals = 0
    hallucinated = []
    for data in golden:
        if data["type"] == "no_answer":          # 판단1(필터): 채점 대상? → data를 봄
            n += 1
            answer = answer_fn(data["question"])
            if "정보 없음" in answer:              # 판단2(거절?): RAG 답 내용 → answer를 봄
                correct_refusals += 1
            else:
                hallucinated.append(data["id"])
    return {
        "n": n,
        "correct_refusals": correct_refusals,
        "hallucinated": hallucinated,
    }


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

def fake_retrieve(question: str, k: int) -> list[str]:
    print("검색기에 들어온 질문:", question)
    print("요청한 개수:", k)

    sample_results = [
        "지원 대상은 서울 소재 기업입니다.",
        "신청 자격은 창업 3년 이내 기업입니다.",
        "모집 기간은 7월 31일까지입니다.",
    ]

    return sample_results[:k]


if __name__ == "__main__":
    example = Path(__file__).resolve().parents[1] / "data" / "golden_set.jsonl"
    golden = load_golden(example)
    print(f"골든셋 예시 {len(golden)}문항 로드 (normal {sum(r['type']=='normal' for r in golden)}, "
          f"no_answer {sum(r['type']=='no_answer' for r in golden)})")

    result = hit_rate_at_k(
        golden,
        fake_retrieve,
        k=3,
    )

    print("평가 결과:", result)

    print("→ 구현 후 채점: python tests\\test_evaluate.py")
