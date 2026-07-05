"""
rag_answer.py — 검색 결과로 프롬프트를 조립해 LLM 답변 생성  [Week 4]
    - 프롬프트/LLM 호출: ✅ 배관 제공 (계획서: "API 호출 코드"는 맡겨도 됨)
    - 근거 인용 검증:   ✍️ 직접 구현

먼저 읽기: ../../01-basics/02-LLM-기초.md (프롬프트, temperature, 환각)
준비:      requirements.txt에서 openai 주석 해제 후 설치.
           PowerShell에서 API 키 설정:  $env:OPENAI_API_KEY = "sk-..."
           (영구 저장: setx OPENAI_API_KEY "sk-..."  후 터미널 재시작)
"""
import os
import sys
from pathlib import Path

# 같은 폴더의 모듈을 어디서 실행해도 import 가능하게 (배관)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── LLM 설정 ─────────────────────────────────────────────────
# 모델명은 바뀌니 공식 문서에서 확인. 저렴한 소형 모델이면 충분하다.
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2   # 사실 기반 답변이므로 낮게 (02-LLM-기초.md 3번)

# ── 프롬프트 템플릿 [✅ 배관 — 하지만 직접 고쳐가며 실험할 것!] ──
# 규칙을 번호 목록으로 명시하는 이유: FAQ Q18
PROMPT_TEMPLATE = """당신은 정부 지원사업 공고문 안내 도우미입니다.
아래 [근거] 조각들만 사용해서 [질문]에 답하세요.

규칙:
1. 근거에 없는 내용은 절대 추측하거나 지어내지 마세요.
2. 근거만으로 답할 수 없으면 정확히 "정보 없음"이라고만 답하세요.
3. 답할 수 있다면, 답변 끝에 사용한 근거 번호를 [근거 1]처럼 표기하세요.

[근거]
{context}

[질문]
{question}

답변:"""


def build_context(chunks: list[str]) -> str:
    """[✅ 배관] chunk들을 번호 붙여 프롬프트용 문자열로."""
    return "\n\n".join(f"[근거 {i+1}]\n{c}" for i, c in enumerate(chunks))


def call_llm(prompt: str) -> str:
    """[✅ 배관] LLM 호출. OpenAI 기본, Anthropic 대안은 아래 주석."""
    from openai import OpenAI
    client = OpenAI()  # 환경변수 OPENAI_API_KEY 사용
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# --- Anthropic(Claude)을 쓰고 싶다면 (pip install anthropic, $env:ANTHROPIC_API_KEY) ---
# def call_llm(prompt: str) -> str:
#     import anthropic
#     client = anthropic.Anthropic()
#     resp = client.messages.create(
#         model="claude-haiku-4-5",   # 소형·저렴. 최신 모델명은 공식 문서 확인
#         max_tokens=1000,
#         temperature=TEMPERATURE,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return resp.content[0].text.strip()
# ---------------------------------------------------------------------------------


def answer(question: str, retrieved_chunks: list[str]) -> str:
    """[✅ 배관] 검색 결과 → 프롬프트 조립 → 생성."""
    prompt = PROMPT_TEMPLATE.format(context=build_context(retrieved_chunks),
                                    question=question)
    return call_llm(prompt)


# ──────────────────────────────────────────────────────────────
# ✍️ 직접 구현: 근거 인용 검증 (grounding check)
# ──────────────────────────────────────────────────────────────

def verify_citation(answer_text: str, chunks: list[str]) -> dict:
    """답변이 정말 근거에서 나왔는지 코드로 검사한다.

    왜 필요한가: 모델이 [근거 1]이라고 써놓고 실제로는 지어냈을 수 있다.
    "근거 인용 검증은 어떻게 했나?"는 12주차 관문 질문이다.

    TODO(직접 구현) — 시작은 단순하게, 단계 힌트:
      1. "정보 없음" 답변이면 검증 통과로 처리 (검사할 주장 없음).
      2. v1 (숫자 검사): 답변에서 숫자/금액 토큰(예: 1억, 39세, 7월 31일의 '31')을 뽑아
         각각이 어느 chunk에든 존재하는지 확인. 숫자는 환각이 가장 위험한 부분!
         힌트: import re; re.findall(r"\\d+", text)
      3. v2 (선택): 답변을 문장으로 쪼개, 각 문장의 핵심 단어들이 chunk와 겹치는 비율 계산.
      4. 반환 예: {"grounded": True/False, "missing": ["근거에 없는 숫자들"]}
    """
    raise NotImplementedError("verify_citation을 직접 구현하세요")


# ──────────────────────────────────────────────────────────────
# 완성 배관: 전체 흐름 연결 데모 (chunker + bm25 구현 후 동작)
# 실행: python src\rag_answer.py "신청 자격이 어떻게 되나요?"
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from chunker import chunk_fixed
    from bm25 import BM25

    question = sys.argv[1] if len(sys.argv) > 1 else "신청 자격이 어떻게 되나요?"

    text_dir = Path(__file__).resolve().parents[1] / "docs" / "text"
    txt_files = sorted(text_dir.glob("*.txt"))
    if not txt_files:
        print("docs/text/ 가 비어 있습니다. 먼저 extract_pdf.py를 실행하세요.")
        raise SystemExit(0)

    try:
        chunks = []
        for f in txt_files:
            chunks += chunk_fixed(f.read_text(encoding="utf-8"), doc_id=f.stem)
        bm25 = BM25([c["text"] for c in chunks])
        top = bm25.search(question, k=3)
        retrieved = [chunks[i]["text"] for i, _ in top]

        print(f"질문: {question}\n")
        for rank, (i, s) in enumerate(top, 1):
            print(f"[근거 {rank}] (score={s:.2f}, {chunks[i]['id']}) {chunks[i]['text'][:60]!r}...")

        if not os.environ.get("OPENAI_API_KEY"):
            print("\nOPENAI_API_KEY가 없어 생성은 건너뜁니다. 검색 결과만 확인하세요.")
            raise SystemExit(0)

        ans = answer(question, retrieved)
        print(f"\n답변:\n{ans}")
        try:
            print(f"\n인용 검증: {verify_citation(ans, retrieved)}")
        except NotImplementedError:
            print("\n(verify_citation은 아직 구현 전)")
    except NotImplementedError as e:
        print(f"아직 구현 전: {e} → chunker.py, bm25.py의 TODO를 먼저 완성하세요.")
