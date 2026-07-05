# gongo-rag — 한국어 공고문 RAG 시스템

> 정부 지원사업 공고문(PDF)을 읽고, 질문에 **근거를 인용하며** 답하고, 근거가 없으면 **"정보 없음"** 이라고 답하는 RAG 시스템.
> 검색 품질을 **hit rate@k로 측정**하고, 실패를 **검색 실패 / 생성 실패 / 데이터 문제**로 분리해 분석한다.

⚠️ 이 README는 시작용입니다. 9주차에 여러분의 실험 결과로 다시 씁니다 — 그때의 틀은 맨 아래 "최종 README 뼈대" 참고.

## 아키텍처

```mermaid
flowchart LR
    A[PDF 3개<br/>docs/raw] -->|extract_pdf.py| B[텍스트<br/>docs/text]
    B -->|chunker.py ✍️| C[chunks]
    C -->|bm25.py ✍️| D[BM25 색인]
    C -->|embeddings.py ✍️| E[벡터 색인]
    Q[질문] --> D & E
    D & E -->|top-k| F[rag_answer.py<br/>프롬프트 조립+LLM]
    F --> G[답변 + 근거 인용]
    GS[골든셋 30~40문항] -->|evaluate.py ✍️| H[hit@3 / hit@5<br/>실험 기록]
```

✍️ = 직접 구현 (뼈대+힌트 제공) / 나머지 = 완성 배관 제공

## 폴더 구조

```
02-gongo-rag/
├── README.md            ← 지금 이 파일
├── WORKFLOW.md          ← GPT 사용 규칙, 하루 루틴 (매일 아침 볼 것)
├── requirements.txt
├── app.py               ← Streamlit 데모 뼈대 (⚠️ 11주차 전에 열지 말 것)
├── docs/raw/            ← 공고문 PDF를 여기에
├── docs/text/           ← 추출된 텍스트 (자동 생성)
├── data/                ← 골든셋 (golden_set_example.jsonl 참고)
├── src/                 ← 소스 코드 (아래 순서대로 작업)
├── tests/               ← 자가 채점 테스트
├── experiments/         ← 실험 기록 (실험가이드.md)
└── notes/               ← 관찰 노트, 주간 기록
```

## 작업 순서 (파일마다 상단에 "먼저 읽을 문서"가 적혀 있음)

| 순서 | 파일 | 하는 일 | 방식 | 주차 |
|---|---|---|---|---|
| 1 | `src/extract_pdf.py` | PDF → 텍스트 | ✅ 실행만 | W1 |
| 2 | `src/chunker.py` | 텍스트 → 조각 2방식 | ✍️ 직접 | W2 |
| 3 | `src/bm25.py` | 키워드 검색 | ✍️ 직접 | W3 |
| 4 | `src/embeddings.py` | 의미 검색 (cosine) | ✍️ 직접 | W4 |
| 5 | `src/rag_answer.py` | 프롬프트+생성+인용검증 | 배관 제공, 검증 ✍️ | W4 |
| 6 | `src/evaluate.py` | 골든셋 + hit@k | ✍️ 직접 | W5 |
| 7 | `experiments/` | 비교 실험 5종 | 가이드 따라 | W6~8 |
| 8 | `app.py` | 데모 UI | ✅ 뼈대 제공 | W11 |

각 단계 후 자가 채점: `python tests\test_<이름>.py`

## 시작하기

```powershell
cd 02-gongo-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 공고문 PDF 3개를 docs/raw/ 에 넣은 뒤:
python src\extract_pdf.py
```

**공고문 구하는 곳**: [기업마당](https://www.bizinfo.go.kr) · [K-Startup](https://www.k-startup.go.kr) — "공고" 메뉴에서 PDF 첨부파일.
고르는 기준: ① 텍스트 복사가 되는 PDF (스캔 이미지 ❌) ② 5~20페이지 ③ 자격/금액/기한이 명확한 것 (질문 만들기 좋음)

## 이 프로젝트의 의사결정 기록 (10주차에 채우기 — 면접 방어력의 원천)

| 결정 | 선택 | 이유 (실험 근거) |
|---|---|---|
| chunk size | (예: 500자, overlap 100) | (hit@3 비교 결과...) |
| 검색 방식 | (예: hybrid) | |
| 벡터 DB | 미사용 (numpy) | chunk 수백 개 규모에선 전수 비교가 밀리초. 도입 기준: ~10만 벡터 |
| 임베딩 모델 | | |
| "정보 없음" 처리 | 프롬프트 규칙 + 골든셋에 무응답 문항 포함 | |

## 최종 README 뼈대 (9주차에 이 순서로 다시 쓰기)

1. 한 줄 소개 + 데모 GIF/스크린샷
2. 아키텍처 다이어그램
3. **실험 결과 표** (BM25 vs 임베딩 vs hybrid × hit@3/@5)
4. **실패 분석 표** (검색/생성/데이터 분류와 대표 사례)
5. 의사결정과 이유
6. 실행 방법
7. 한계와 다음 단계 (reranking은 언제 필요해지는가 등)

> 채용 담당자는 1→3→4만 봅니다. 실험 표와 실패 분석이 앞에 오도록. ([05-career/포지셔닝-이력서-JD.md](../05-career/포지셔닝-이력서-JD.md) 참고)
