# gongo-rag

한국어 정부 지원사업 공고문을 검색하고, **근거를 인용해 답하며 근거가 부족하면 거절하는 RAG**를 만드는 프로젝트입니다.

이 저장소의 목적은 단순 데모가 아니라 다음 세 가지를 증명하는 것입니다.

1. 검색과 답변 품질을 분리해 측정할 수 있다.
2. 한국어 문서 검색의 실패를 숫자와 사례로 설명할 수 있다.
3. 직접 구현한 기준선에서 현업형 RAG 구조로 발전시킬 수 있다.

## 현재 상태

> **v0 기준선 구현 완료, 현업형 구조로 전환 중**

| 영역 | 현재 구현 | 다음 단계 |
|---|---|---|
| 문서 처리 | PDF·DOCX·이미지 추출, 문단 우선 chunking, LangChain Document 변환 | 색인 수명 관리 |
| 검색 | Kiwi BM25 + E5/Chroma + chunk ID 기반 RRF 통합 검색 | dev 실패 질문 분석과 후보 설정 개선 |
| 재정렬 | `BAAI/bge-reranker-v2-m3` 로컬 CrossEncoder | Cohere와 품질·속도·비용 비교 |
| 답변 | LLM prompt, 숫자 근거 일치 검사, `정보 없음` 처리 | LangGraph 재검색·거절 흐름 |
| 평가 | 골든셋 36문항, dev/test 분리, Hit@k·MRR·nDCG·지연 리포트 | reranker 최적화 후 test 1회 + Ragas |
| 서비스 | Streamlit 문서 업로드·추출·질문 데모 | FastAPI + Docker |

현재 dev normal 20문항의 Hit@1은 BM25 0.70, Chroma 0.60, RRF 0.60,
reranker 0.80입니다. reranker MRR은 0.90으로 가장 높았지만 CPU 평균 지연은
약 6.28초였습니다. 작은 dev 결과이므로 최종 성능으로 일반화하지 않으며, 설정 선택이
끝난 뒤 test normal 10문항을 한 번만 확인합니다.

## 현재 아키텍처

```mermaid
flowchart LR
    A["PDF · DOCX · 이미지"] --> B["텍스트 추출 · 필요한 페이지만 OCR"]
    B --> C["Chunking"]
    C --> D["BM25 검색"]
    C --> E["E5 embedding + Chroma 의미 검색"]
    Q["사용자 질문"] --> D
    Q --> E
    D --> F["RRF 순위 결합"]
    E --> F
    F --> U["RRF 후보"]
    U --> R["다국어 CrossEncoder 재정렬"]
    R --> V["최종 Top-k + 출처"]
    V -. "다음: LangGraph 연결" .-> G["LLM 답변 + 인용"]
    H["dev/test 골든셋"] --> I["Hit@k / MRR / nDCG / 지연 시간"]
    D --> I
    E --> I
    F --> I
    R --> I
```

목표 구조는 `LangChain → BM25/Chroma → RRF → reranker → LangGraph → Ragas`입니다. 각 도구를 추가하기 전에 현재 기준선을 보존하고, 같은 평가셋으로 개선 여부를 확인합니다.

## 실행

Windows PowerShell 기준입니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python tests\test_chunker.py
python tests\test_bm25.py
python tests\test_bm25_retriever.py
python tests\test_vector_search.py
python tests\test_hybrid_search.py
python tests\test_reranker.py
python tests\test_evaluate.py
python tests\test_retrieval_evaluation.py
python tests\test_rag_answer.py
python tests\test_document_ingestion.py
python tests\test_document_chunking.py
python tests\test_document_upload_ui.py
```

검색 결과만 확인할 때는 API 키가 없어도 됩니다.

### 파일을 올려 글자로 바꾸기

```powershell
streamlit run app.py
```

브라우저의 `1. 문서 넣기` 탭에서 일반 PDF, 스캔 PDF, DOCX, 이미지를
올릴 수 있습니다. 일반 PDF와 DOCX는 바로 읽고, 스캔 PDF와 이미지는
Tesseract OCR로 한국어와 영어를 읽습니다. OCR 엔진 설치와 파일별 제한은
[1번 작업: 파일 속 글자 꺼내기](docs/INGESTION.md)에 쉬운 설명과 면접 대비
기술 내용을 함께 정리했습니다.

추출 결과를 미리 보고 TXT로 받을 수 있습니다. 이어서 같은 화면에서 문단
우선 또는 고정 길이 방식으로 chunk를 만들고 metadata를 확인할 수 있습니다.
자세한 학습 내용은
[2번 작업: 긴 글을 검색용 조각으로 나누기](docs/CHUNKING.md)에 정리했습니다.
만든 chunk는 Kiwi 형태소 분석을 사용하는 BM25로 바로 검색할 수 있습니다.
검색 원리, 한국어 조사 처리와 면접 대비 내용은
[3번 작업: 질문과 같은 단어가 있는 조각 찾기](docs/BM25.md)에 정리했습니다.
같은 chunk를 한국어 지원 E5 모델로 embedding하고 로컬 Chroma에 저장해
의미로 검색할 수도 있습니다. 자세한 내용은
[4번 작업: 같은 뜻의 문서 조각 찾기](docs/VECTOR_SEARCH.md)에 정리했습니다.
두 검색 결과는 공통 chunk ID와 순위를 사용해 RRF로 합칩니다. 원점수를
직접 더하지 않는 이유와 실제 성공·실패 사례는
[5번 작업: 두 검색기의 순위를 합치기](docs/RRF.md)에 정리했습니다.
RRF 상위 후보는 한국어를 포함한 다국어 CrossEncoder가 질문과 본문을 함께
읽고 다시 정렬합니다. 실제 한국어 PDF의 개선·실패 사례와 면접 대비 내용은
[6번 작업: 질문과 후보를 함께 읽어 다시 정렬하기](docs/RERANKER.md)에
정리했습니다.
실제 공고문 3개에서 만든 고정 질문으로 네 검색기를 같은 조건에서 비교하는 방법과
dev 결과는 [7번 작업: 같은 시험지로 검색기 성적 비교하기](docs/EVALUATION.md)에
정리했습니다. 사람이 읽는 실제 결과는
[dev 검색 평가 리포트](experiments/retrieval-evaluation-dev.md)에서 바로 볼 수
있습니다.

```powershell
python src\run_retrieval_evaluation.py --split dev
```

```powershell
python src\rag_answer.py "신청 자격이 어떻게 되나요?"
```

답변 생성과 Streamlit 데모를 실행할 때는 환경 변수에 API 키를 설정합니다.

```powershell
$env:OPENAI_API_KEY = "your-api-key"
streamlit run app.py
```

## 평가 원칙

- 골든셋은 실험 도중 정답을 맞추기 위해 수정하지 않습니다.
- 설정을 고르는 동안 dev만 사용하고 test 결과는 반복해서 보지 않습니다.
- chunk 크기, tokenizer, 검색기처럼 **한 번에 하나의 변수만** 바꿉니다.
- Hit@1·3·5·10, MRR, nDCG와 지연 시간을 함께 봅니다.
- 검색 실패, 답변 실패, 원문 데이터 문제를 따로 기록합니다.
- Ragas 점수만 믿지 않고 한국어 질문·근거·답변을 사람이 함께 확인합니다.

## 저장소 구조

```text
gongo-rag/
├── app.py                 # 업로드·추출·질문 Streamlit 데모
├── .chroma/               # 로컬 vector DB (git 제외)
├── data/                  # 골든셋
├── docs/raw/              # 원본 공고문 PDF
├── docs/text/             # 추출 텍스트
├── experiments/           # 비교 실험과 결정 기록
├── notes/                 # 관찰 기록
├── src/                   # 추출·chunking·BM25·Chroma·RRF·reranker·답변·평가
└── tests/                 # 핵심 로직 자가 검증
```

전체 학습 순서와 “무엇을 왜 만드는지”는 [RAG 전체 지도](https://github.com/hamjinoo/ai-engineer-study/blob/main/rag/RAG-%EC%A0%84%EC%B2%B4%EC%A7%80%EB%8F%84.md)에 정리합니다.

## 다음 마일스톤

1. 후보 수·최적화 모델과 Cohere Rerank를 dev에서 비교하기
2. 선택한 설정을 고정하고 test split을 한 번 실행하기
3. LangGraph 재검색·거절 흐름 구현하기
4. Ragas·수동 검토를 포함한 답변 평가표 작성하기
5. FastAPI·Docker와 재현 가능한 실행 환경 제공하기
