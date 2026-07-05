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
    """두 벡터의 코사인 유사도.  cos = (a·b) / (|a| * |b|)

    TODO(직접 구현) — 추천 순서:
      1단계: numpy 없이 순수 반복문으로 (내적 = 곱해서 더하기, 길이 = 제곱합의 제곱근)
      2단계: np.dot / np.linalg.norm 버전으로 바꾸고 두 결과가 같은지 확인
      → 이 두 단계를 거치면 "이해"가 남는다.
    자가 검증: cosine([3,4],[6,8]) == 1.0,  cosine([3,4],[4,-3]) == 0.0
              (수학 문서의 손계산 예제 그대로)
    """
    raise NotImplementedError("cosine_similarity를 직접 구현하세요")


def search(query: str, chunk_texts: list[str], chunk_vectors: np.ndarray,
           k: int = 3) -> list[tuple[int, float]]:
    """질문과 의미가 가장 가까운 chunk 상위 k개 → [(인덱스, 유사도), ...] 내림차순.

    TODO(직접 구현) — 단계 힌트:
      1. 질문 하나를 임베딩: embed_texts([query])[0]
      2. 모든 chunk 벡터와 cosine_similarity 계산 (반복문이면 충분)
      3. 내림차순 상위 k개.  힌트: sorted 또는 np.argsort(...)[::-1][:k]
    """
    raise NotImplementedError("search를 직접 구현하세요")


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
