"""
embeddings.py — 임베딩(의미) 검색  [Week 4: 모델 호출은 ✅ 제공, 유사도/검색은 ✍️ 직접]

먼저 읽기: ../../04-concepts/임베딩-완전정복.md + ../../01-basics/04-최소한의-수학.md(코사인)
설치:      requirements.txt에서 sentence-transformers 주석 해제 후 pip install -r requirements.txt
자가 채점:  이 파일의 데모 + 검색 결과를 BM25와 눈으로 비교
GPT 규칙:  cosine similarity 전체 구현 대행 금지 (WORKFLOW.md).
"""
from pathlib import Path

import numpy as np

# 임베딩 모델 (한국어 특화, CPU OK). 모델 교체 자체가 좋은 실험 (FAQ Q17)
MODEL_NAME = "jhgan/ko-sroberta-multitask"

_model = None


def load_model():
    """[✅ 배관] 임베딩 모델 로드 (최초 1회 ~400MB 다운로드, 이후 캐시)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # 무거워서 지연 import
        print(f"임베딩 모델 로드 중: {MODEL_NAME} (최초 실행은 다운로드로 몇 분 걸릴 수 있음)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """[✅ 배관] 텍스트 N개 → (N, 차원수) 벡터 배열."""
    model = load_model()
    return np.asarray(model.encode(texts, show_progress_bar=len(texts) > 16))


def save_vectors(vectors: np.ndarray, path: str | Path) -> None:
    """[✅ 배관] 벡터 캐시 저장 — 색인은 문서가 바뀔 때만 다시 하면 됨."""
    np.save(str(path), vectors)


def load_vectors(path: str | Path) -> np.ndarray:
    """[✅ 배관] 벡터 캐시 로드. ⚠️ 캐시를 만들 때와 MODEL_NAME이 같아야 유효!"""
    return np.load(str(path))


# ──────────────────────────────────────────────────────────────
# ✍️ 여기부터 직접 구현
# ──────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터가 얼마나 닮았는지 재는 함수. 완전히 같은 방향이면 1, 상관없으면 0.

    벡터 = 그냥 "숫자가 줄줄이 담긴 목록"입니다. 게임 캐릭터의 능력치표 같은 것.

    ⚠️ 여기서 가장 흔한 실수: 벡터를 "숫자 2개짜리"라고 생각하는 것!
       아래 데모의 [3, 4]는 연습용일 뿐이고, 진짜 임베딩 벡터에는 숫자가
       768개 들어 있습니다. a[0], a[1]처럼 자리 번호를 직접 찍어 쓰면
       앞의 2개만 보고 나머지 766개는 무시합니다 — 에러도 안 나고 조용히 틀려요.
       → 숫자가 몇 개 들어있든 "전부" 훑는 코드여야 합니다.

    공식: cos = (a·b) / (|a| * |b|)
      - a·b (내적)  = 같은 자리 숫자끼리 곱한 뒤 → 전부 더하기
      - |a| (길이)  = 각 숫자를 제곱해서 전부 더한 뒤 → 제곱근(√)

    TODO(직접 구현) — 추천 순서:
      1단계: numpy 없이 순수 반복문으로 (내적 = 곱해서 더하기, 길이 = 제곱합의 제곱근)
      2단계: np.dot / np.linalg.norm 버전으로 바꾸고 두 결과가 같은지 확인
      → 이 두 단계를 거치면 "이해"가 남는다.
    자가 검증: cosine([3,4],[6,8]) == 1.0,  cosine([3,4],[4,-3]) == 0.0
              + 숫자 5개짜리 벡터를 넣어도 돌아가야 진짜 완성!
    """
    nae = 0
    a_sq = 0
    b_sq = 0
    for i in range(len(a)):
        nae += (a[i] * b[i])
        a_sq += (a[i] ** 2)
        b_sq += (b[i] ** 2)

    gob = (a_sq * b_sq) ** 0.5
    cosine = nae / gob

    return cosine


def search(query: str, chunk_texts: list[str], chunk_vectors: np.ndarray,
           k: int = 3) -> list[tuple[int, float]]:
    """질문과 뜻이 가장 비슷한 조각(chunk) 상위 k개 → [(번호, 유사도), ...] 점수 높은 순.

    말로 풀면: "질문도 숫자 목록(벡터)으로 바꾼 다음, 모든 조각의 벡터와
    하나하나 닮은 정도를 재서, 1등부터 k등까지 뽑기".
    이미 완성한 bm25.py의 search와 뼈대가 완전히 같습니다 — 점수 재는 방법만 다를 뿐!

    TODO(직접 구현) — 단계 힌트:
      1. 질문 하나를 임베딩: embed_texts([query])[0]
      2. 모든 chunk 벡터와 cosine_similarity 계산 (반복문이면 충분)
         ⚠️ range( ) 괄호 안에는 "몇 번 돌지"라는 숫자 1개가 들어가야 합니다.
            chunk_vectors는 숫자가 아니라 벡터 묶음이에요. 묶음에서 "개수"를
            꺼내주는 함수, chunker.py에서 이미 써봤죠?
      3. 점수 높은 순 상위 k개.  힌트: sorted 또는 np.argsort(...)[::-1][:k]
    """
    q_vec = embed_texts([query])[0]
    results = []
    for i in range(len(chunk_vectors)):
        s = cosine_similarity(q_vec, chunk_vectors[i])
        results.append((i, s))

    return sorted(results, key=lambda x: x[1], reverse=True)[:k]



# ──────────────────────────────────────────────────────────────
# 완성 배관: 미니 데모 — 실행:  python src\embeddings.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sents = [
        "지원 금액은 최대 1억원입니다",
        "돈은 얼마나 받을 수 있나요",      # 표현은 다르지만 의미가 가까움 → 높게 나와야
        "오늘 점심은 김치찌개를 먹었다",    # 무관 → 낮게 나와야
    ]
    try:
        # cosine부터 손계산 예제로 검증 (모델 다운로드 전에 확인 가능)
        c1 = cosine_similarity(np.array([3.0, 4.0]), np.array([6.0, 8.0]))
        c2 = cosine_similarity(np.array([3.0, 4.0]), np.array([4.0, -3.0]))
        print(f"cosine([3,4],[6,8]) = {c1:.4f}  (기대: 1.0000)")
        print(f"cosine([3,4],[4,-3]) = {c2:.4f}  (기대: 0.0000)\n")

        vecs = embed_texts(sents)
        print(f"임베딩 shape: {vecs.shape}  (문장 3개 × 차원수)\n")
        base = vecs[0]
        for s, v in zip(sents, vecs):
            print(f"  {cosine_similarity(base, v):+.3f}  {s}")
        print("\n0번 문장 기준: 1번(의미 유사)이 2번(무관)보다 확실히 높으면 성공.")
    except NotImplementedError as e:
        print(f"아직 구현 전: {e}")
