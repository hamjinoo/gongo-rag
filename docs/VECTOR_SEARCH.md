# 4번 작업: 같은 뜻의 문서 조각 찾기

## 이 문서 읽는 방법

- 처음 공부한다면 `한 줄로 설명하면`부터 `다음 작업`까지 읽습니다.
- 면접을 준비한다면 `면접 대비 기술 설명`부터 읽습니다.
- 직접 실행할 때는 맨 아래 `개발자용 실행 부록`을 봅니다.

## 한 줄로 설명하면

질문과 문서의 단어가 달라도 **뜻이 비슷한 chunk를 찾는 작업**입니다.

```text
문서에서 글자 꺼내기
→ 긴 글을 chunk로 나누기
→ BM25로 같은 단어 찾기
→ Chroma로 같은 뜻 찾기 ← 지금 여기
→ 두 검색 순위를 합치기
```

BM25가 책에서 같은 낱말에 형광펜을 칠하는 검색이라면, embedding 검색은
문장의 뜻을 보고 비슷한 내용끼리 같은 책장에 놓는 검색입니다.

## 이번 작업에서 얻어야 하는 것

다음 다섯 가지를 자기 말로 설명할 수 있으면 됩니다.

1. embedding은 문장의 뜻을 숫자 목록으로 바꾼 것입니다.
2. 비슷한 뜻의 문장은 숫자 공간에서도 가까이 놓입니다.
3. LangChain Document는 본문과 출처 metadata를 담는 공통 상자입니다.
4. Chroma는 embedding을 저장하고 가까운 벡터를 찾는 vector store입니다.
5. 의미 검색도 틀릴 수 있으므로 BM25와 결과를 비교해야 합니다.

## 왜 BM25만으로는 부족한가요?

다음 질문과 문서는 뜻이 비슷하지만 같은 단어가 거의 없습니다.

```text
질문: 돈을 얼마나 받을 수 있나요?
문서: 사업화 지원 금액은 최대 1억원입니다.
```

BM25는 `돈`, `얼마나`를 문서에서 찾으려 하기 때문에 놓칠 수 있습니다.
의미 검색은 두 문장 모두 “받을 수 있는 자금 규모”에 관한 문장이라고
판단할 가능성이 있습니다.

반대로 정확한 사업명, 날짜, 금액, 공고 번호는 BM25가 더 잘 찾을 수
있습니다. 둘 중 하나만 정답 검색기인 것이 아닙니다.

| 검색기 | 잘하는 일 | 어려운 일 |
|---|---|---|
| BM25 | 같은 단어, 고유명사, 번호, 금액 | 같은 뜻의 다른 표현 |
| Embedding + Chroma | 동의어, 자연스러운 질문, 문맥 | 정확한 문자열, 아주 비슷한 문장 구별 |

## Embedding은 무엇인가요?

게임 캐릭터의 능력치표를 생각하면 쉽습니다.

```text
캐릭터 A = [힘 90, 속도 20, 마법 10]
캐릭터 B = [힘 85, 속도 25, 마법 12]
캐릭터 C = [힘 10, 속도 20, 마법 95]
```

A와 B는 숫자가 비슷하므로 비슷한 캐릭터입니다. 문장 embedding도
이와 비슷합니다. 다만 사람이 정한 세 가지 능력치가 아니라 모델이 만든
수백 개의 숫자를 사용합니다.

```text
"지원 금액은 1억원" → [0.12, -0.31, 0.08, ...]
"돈을 얼마나 받나" → [0.10, -0.29, 0.11, ...]
"접수는 7월 마감"  → [-0.42, 0.07, 0.33, ...]
```

첫 번째와 두 번째 벡터의 방향이 가까우면 같은 뜻에 가깝다고 봅니다.

## LangChain Document는 왜 필요한가요?

지금까지 만든 `DocumentChunk`에는 본문과 파일명, 페이지가 들어 있습니다.
LangChain의 여러 검색 도구가 같은 형식으로 읽을 수 있도록 다음 상자로
바꿉니다.

```text
Document
├── id: chunk의 고유 번호
├── page_content: 검색할 본문
└── metadata
    ├── 원본 파일명
    ├── 페이지 번호
    ├── 추출 방식
    ├── 원문 위치
    └── chunk 전략
```

글만 embedding하고 출처를 버리면 나중에 “이 답이 어느 PDF 몇 페이지에서
왔는가?”를 보여줄 수 없습니다. 그래서 변환할 때 metadata를 함께
보존합니다.

## Chroma는 무엇을 하나요?

Chroma는 다음 두 가지를 담당합니다.

```text
저장할 때
chunk 본문 → embedding 숫자 → Chroma에 ID·metadata와 함께 저장

검색할 때
질문 → embedding 숫자 → 가장 가까운 chunk 벡터 찾기
```

이번 프로젝트는 외부 서버나 API 키 없이 로컬 `.chroma` 폴더에 저장합니다.
같은 chunk와 같은 모델로 다시 실행하면 이미 만든 벡터를 재사용합니다.

## 이번에 선택한 한국어 모델

기본 모델은 `intfloat/multilingual-e5-small`입니다.

- 한국어를 포함한 여러 언어를 지원합니다.
- 로컬 CPU에서 실행할 수 있어 embedding API 비용이 없습니다.
- 약 384개의 숫자로 문장을 표현합니다.
- 첫 실행에서 모델 파일 약 500MB를 내려받습니다.
- 긴 입력은 최대 512 token에서 잘릴 수 있습니다.

E5는 질문과 문서 앞에 서로 다른 이름표를 붙여 학습한 모델입니다.

```text
질문: query: 어떤 회사가 도움을 받을 수 있나요?
문서: passage: 지원대상은 대전시 소재 기업입니다.
```

이 접두어는 화면에 보이도록 원문을 바꾸는 것이 아니라 embedding 모델에
전달할 때만 사용합니다. 공식 모델 지침과 다르게 접두어를 빼면 검색 품질이
낮아질 수 있습니다.

## Cosine similarity는 무엇인가요?

두 벡터가 같은 방향을 보는지 확인하는 값입니다.

```text
방향이 매우 비슷함 → similarity가 1에 가까움
방향이 다름         → 값이 작아짐
```

현재 Chroma는 cosine distance를 사용합니다.

```text
similarity = 1 - distance
```

중요한 점은 similarity가 정답 확률이 아니라는 것입니다. `0.86`은
“86% 확률로 정답”이라는 뜻이 아닙니다. 같은 질문에서 후보의 상대적인
순서를 비교하는 값입니다.

## 화면에서 사용하는 방법

1. `1. 문서 넣기` 탭에서 파일을 올립니다.
2. `텍스트 추출`을 누릅니다.
3. `2. 글자 나누기`에서 `Chunk 만들기`를 누릅니다.
4. 먼저 `3. BM25 키워드 검색`으로 질문합니다.
5. 아래 `4. Chroma 의미 검색`에서 같은 질문을 입력합니다.
6. 처음이면 모델 다운로드와 문서 embedding을 기다립니다.
7. 두 검색기의 1~3위 chunk와 페이지를 비교합니다.
8. 필요하면 Chroma 결과를 JSON으로 받습니다.

처음 실행만 느릴 수 있습니다. 두 번째 실행에서는 Hugging Face 모델 캐시와
`.chroma`에 저장된 문서 벡터를 재사용합니다.

## 실제 공고문으로 확인한 결과

저장소의 실제 5페이지 한국어 공고문을 사용했습니다.

```text
문서: 2026년 글로벌화 사전검증 지원사업 참여기업 모집 공고
분할: 문단 우선 700자 / overlap 120자
생성 chunk: 11개
질문: 어떤 회사가 도움을 받을 수 있나요?

1위 · similarity 0.8610 · 페이지 1 · "지원대상..."
2위 · similarity 0.8488 · 페이지 2 · 참여 제한 조건
3위 · similarity 0.8457 · 페이지 3 · 기업지원 내용
```

질문에는 `지원대상`이라는 정확한 단어가 없지만 지원 가능한 회사가 적힌
1페이지를 첫 번째로 찾았습니다. 모든 결과에 파일명, 페이지, 추출 방법과
chunk ID가 유지되는 것도 확인했습니다.

## 공부하면서 직접 해볼 작은 실험

같은 문서에서 다음 질문을 BM25와 Chroma에 각각 넣습니다.

```text
A. 지원 대상은 누구인가요?
B. 어떤 회사가 도움을 받을 수 있나요?
C. 돈을 얼마나 받을 수 있나요?
D. 접수 마감일은 언제인가요?
```

기록할 내용은 다섯 가지뿐입니다.

```text
질문:
BM25에서 정답 chunk 순위:
Chroma에서 정답 chunk 순위:
두 결과가 다른 이유:
이상하거나 놓친 결과:
```

- A는 정확한 단어가 있어 BM25도 잘 찾을 가능성이 큽니다.
- B와 C는 다른 표현을 사용해 Chroma의 장점을 보기 좋습니다.
- D처럼 정확한 날짜 주변을 찾을 때는 두 검색기의 순서를 비교합니다.

어느 검색기가 “항상 더 좋다”고 결론 내리지는 않습니다. 두 검색기의 실패 유형을
비교한 기록은 현재 구현된 RRF와 이후 reranker를 개선하는 근거가 됩니다.

## 현재 한계

- 의미가 비슷하기만 하고 답은 아닌 chunk가 상위에 올 수 있습니다.
- Chroma는 질문마다 항상 가까운 결과를 가져오므로 정답 없음 판단이 필요합니다.
- similarity의 절대값만 보고 정답 여부를 결정할 수 없습니다.
- 모델의 512 token 제한을 넘는 긴 chunk는 뒤쪽이 잘릴 수 있습니다.
- CPU에서 첫 모델 로딩과 많은 문서의 embedding은 느릴 수 있습니다.
- OCR 오류나 잘못 나눈 chunk는 embedding으로도 완전히 고칠 수 없습니다.
- `.chroma`에 사용하지 않는 collection을 정리하는 관리 기능은 아직 없습니다.
- BM25와 Chroma 결과는 RRF로 합치고 CrossEncoder로 재정렬합니다.
- 재정렬 결과는 LangGraph 최종 답변·재검색·거절 흐름에 연결했습니다.

## 4번 작업의 완료 상태

- [x] `DocumentChunk`를 LangChain `Document`로 변환
- [x] chunk ID와 출처 metadata 보존
- [x] 한국어 지원 E5 embedding 모델 연결
- [x] `query:`와 `passage:` 접두어 분리
- [x] embedding 정규화
- [x] cosine 기준 Chroma 색인
- [x] 로컬 `.chroma` 영구 저장
- [x] 같은 색인의 문서 벡터 재사용
- [x] 다른 chunk가 같은 collection을 쓰는 오류 차단
- [x] similarity·distance·출처 표시 UI
- [x] 검색 결과 JSON 다운로드
- [x] 자동 테스트와 실제 한국어 PDF 확인
- [x] BM25와 Chroma 순위를 RRF로 결합
- [x] reranker 추가

## 이후 작업

현재는 두 검색 결과를 다음처럼 합치고 재정렬합니다.

```text
질문
├── Kiwi BM25 순위
└── E5 + Chroma 순위
          ↓
        RRF
          ↓
   합쳐진 후보 순위
```

BM25 점수와 Chroma similarity는 크기와 의미가 다르므로 숫자를 직접
더하지 않고 각 검색기의 **순위**를 이용했습니다. 자세한 내용은
[5번 작업: 두 검색기의 순위를 합치기](RRF.md)에 정리했습니다.

---

## 면접 대비 기술 설명

### 30초 설명

> 구조화된 `DocumentChunk`를 LangChain `Document`로 변환하고 로컬
> Chroma vector store에 색인하는 의미 검색 계층을 구현했습니다.
> 한국어를 지원하는 `multilingual-e5-small`을 사용하며 모델 지침에 맞게
> 문서에는 `passage:`, 질문에는 `query:` 접두어를 적용하고 벡터를
> 정규화했습니다. cosine similarity와 원본 chunk metadata를 함께
> 반환하며, chunk 내용과 모델명으로 결정적인 collection 이름을 만들어
> 같은 벡터를 재사용합니다. 기존 BM25는 독립 기준선으로 유지해 다음
> 단계에서 chunk ID 기반 RRF 결합이 가능하도록 했습니다.

### 코드 구조

| 파일 | 책임 |
|---|---|
| [`src/vector_search.py`](../src/vector_search.py) | LangChain 변환, E5 모델, Chroma 색인·검색 |
| [`src/vector_search_ui.py`](../src/vector_search_ui.py) | 의미 검색 설정, 결과·출처, JSON 다운로드 |
| [`app.py`](../app.py) | 추출 → chunking → BM25 → Chroma 화면 연결 |
| [`tests/test_vector_search.py`](../tests/test_vector_search.py) | 변환·색인·검색·영구 저장 테스트 |
| [`tests/test_document_upload_ui.py`](../tests/test_document_upload_ui.py) | 업로드부터 Chroma 화면까지 사용자 흐름 |

### LangChain Document 변환

```python
Document(
    id=chunk.id,
    page_content=chunk.text,
    metadata={
        **chunk.metadata,
        "chunk_id": chunk.id,
    },
)
```

`page_content`는 embedding 대상이고 `metadata`는 검색 후 출처 복원에
사용합니다. `chunk_id`를 metadata에도 명시한 이유는 vector store가
반환한 문서를 원본 `DocumentChunk`와 안정적으로 연결하기 위해서입니다.

metadata는 Chroma가 저장할 수 있는 문자열과 숫자의 평평한 구조를
사용합니다. 중첩 dict를 넣지 않습니다.

### LangChain Embeddings 인터페이스

LangChain은 모델마다 다른 호출 방식을 다음 두 메서드로 통일합니다.

```text
embed_documents(list[str]) → 문서 벡터 여러 개
embed_query(str)            → 질문 벡터 한 개
```

현재는 `HuggingFaceEmbeddings`를 사용하지만 나중에 OpenAI, Cohere 또는
사내 embedding endpoint로 교체해도 Chroma 검색 계층의 인터페이스는
유지할 수 있습니다.

### 모델 선택과 E5 접두어

`multilingual-e5-small`을 시작 모델로 선택한 이유는 다음과 같습니다.

- 한국어를 포함한 다국어 검색 지원
- 로컬 실행과 API 키 없는 재현
- large 모델보다 작은 CPU·메모리 부담
- query와 passage가 다른 검색 문제에 맞춘 학습 방식

설정은 다음과 같습니다.

```python
HuggingFaceEmbeddings(
    model="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "prompt": "passage: ",
        "normalize_embeddings": True,
    },
    query_encode_kwargs={
        "prompt": "query: ",
        "normalize_embeddings": True,
    },
)
```

접두어는 단순한 장식이 아니라 모델 학습 방식의 일부입니다. query와
passage 역할을 바꾸거나 접두어를 빼는 것은 검색 성능 실험의 변수가
되므로 코드와 테스트에서 고정했습니다.

### 벡터 정규화와 cosine

각 embedding을 L2 norm 1로 정규화합니다. 정규화된 벡터에서는 내적과
cosine similarity의 순위가 같아지고, 벡터 크기보다 방향을 비교하기
쉬워집니다.

Chroma collection은 `hnsw:space = cosine`으로 만듭니다. Chroma의 cosine
distance가 0이면 같은 방향이므로 화면에는 이해하기 쉬운
`similarity = 1 - distance`도 함께 표시합니다.

### Chroma 영구 저장

UI에서는 다음 경로를 사용합니다.

```text
gongo-rag/.chroma/
```

이 폴더는 `.gitignore`에 포함됩니다. 모델 weight는 Hugging Face의 사용자
캐시에, 문서 embedding과 metadata는 `.chroma`에 저장됩니다. 저장소에
수백 MB의 모델이나 로컬 DB 파일을 커밋하지 않습니다.

### 결정적인 collection 이름

collection 이름은 다음 값의 SHA-256으로 만듭니다.

```text
embedding 모델명
embedding 설정 버전
각 chunk ID
각 chunk 본문
각 chunk metadata
```

```text
gongo-{해시 앞 20자}
```

같은 문서·설정·모델이면 같은 이름이므로 기존 벡터를 재사용합니다. 본문이나
모델이 바뀌면 다른 collection이 생겨 예전 모델의 벡터를 새 모델로 검색하는
오류를 막습니다.

### 기존 색인 검증

collection이 비어 있으면 문서와 ID를 추가합니다. 이미 값이 있으면 저장된
ID 집합과 현재 chunk ID 집합을 비교합니다.

```text
ID가 모두 같음 → 기존 embedding 재사용
ID가 다름      → VectorIndexMismatchError
```

자동으로 기존 collection을 지우지 않는 이유는 잘못된 collection 이름으로
사용자 색인을 덮어쓰는 일을 막기 위해서입니다.

### 검색 결과 복원

Chroma 결과의 `chunk_id`로 원본 `DocumentChunk`를 찾고 다음 객체를
반환합니다.

```text
VectorSearchResult
├── rank
├── similarity
├── distance
├── model_name
└── 원본 DocumentChunk
```

따라서 현재 RRF 구현은 BM25와 Chroma의 서로 다른 객체 형식을 다시
파싱하지 않고 공통 `chunk.id`로 합칩니다.

### 캐시와 UI 상태

- embedding 모델은 `lru_cache`로 프로세스 안에서 재사용합니다.
- Streamlit retriever는 `cache_resource`로 재사용합니다.
- 문서 벡터는 Chroma 영구 collection에서 재사용합니다.
- 질문이나 top-k가 바뀌면 이전 결과를 숨기고 재검색을 요청합니다.
- chunk signature가 바뀌면 이전 문서의 결과를 보여주지 않습니다.

모델, retriever, vector DB는 서로 다른 수명을 가지므로 세 층으로
나눠 캐시했습니다.

### 테스트 전략

실제 모델을 매 테스트마다 다운로드하면 느리고 네트워크에 따라 실패합니다.
따라서 자동 테스트에서는 문장의 의미 범주를 3차원으로 만드는 작은
`KeywordEmbeddings`를 주입합니다.

검증 항목은 다음과 같습니다.

- LangChain Document 본문·ID·metadata 변환
- 같은 단어가 없어도 같은 의미 결과가 1위인지
- 원본 파일명·페이지·추출 방법 보존
- 빈 질문과 잘못된 top-k 처리
- 중복 chunk ID 거절
- 결정적인 collection 이름
- 영구 collection에서 문서 embedding 재사용
- 같은 collection의 다른 chunk 구성 거절
- E5 query/passage 접두어와 정규화 설정
- 기존 Hit@k 평가 함수용 text adapter
- 업로드 → 추출 → chunking → Chroma UI 흐름

기존 기능을 포함한 자동 테스트 **59개가 모두 통과**했습니다. 별도로 실제
E5 모델과 5페이지 한국어 PDF를 사용해 11개 chunk의 로컬 Chroma 색인과
검색 결과를 확인했습니다.

### 기술 선택과 트레이드오프

| 선택 | 이유 | 한계 |
|---|---|---|
| multilingual-e5-small | 한국어 지원, 로컬 재현, 비교적 작은 모델 | 더 큰 최신 모델보다 품질이 낮을 수 있음 |
| LangChain Document | 도구 교체가 쉬운 공통 인터페이스 | 얇은 변환 계층이 추가됨 |
| Chroma 로컬 저장 | API 키 없이 데모와 영구 저장 가능 | 운영 규모·동시성 검증은 별도 필요 |
| cosine + 정규화 | 의미 방향 비교가 쉽고 재현 가능 | 절대 점수로 정답 판단 불가 |
| 결정적 collection 이름 | 같은 embedding 재사용, 모델 혼용 방지 | 오래된 collection 정리 필요 |
| 가짜 embedding 테스트 | 빠르고 오프라인에서 결정적 | 실제 모델 품질은 통합 테스트가 필요 |

### 예상 면접 질문

#### Q1. 왜 한국어 전용 SBERT 대신 multilingual E5를 선택했나요?

현재 목표는 한국어 공고문 검색과 LangChain 표준 인터페이스를 함께
검증하는 것입니다. E5는 다국어 retrieval 용도와 query/passage 역할이
명확하고 로컬 실행이 가능합니다. 다만 최종 선택은 한국어 고정 질문셋으로
`ko-sroberta`, E5, BGE-M3 등의 Hit@k와 지연 시간을 비교해야 합니다.

#### Q2. 왜 `query:`와 `passage:`를 붙이나요?

E5가 비대칭 검색에서 질문과 문서를 다른 역할로 학습했기 때문입니다.
모델 카드도 비영어 입력을 포함해 이 접두어 사용을 요구합니다. 접두어를
빼면 동작은 해도 검색 품질이 낮아질 수 있습니다.

#### Q3. 왜 embedding을 정규화했나요?

문장 벡터의 크기가 아니라 방향을 비교하기 위해서입니다. L2 정규화 후
cosine similarity를 사용하면 모델 출력 크기의 영향을 줄이고 일관된
유사도 비교가 가능합니다.

#### Q4. Chroma score는 높은 것이 좋은가요?

LangChain이 반환하는 relevance similarity는 높을수록 가깝고, Chroma의
원래 cosine distance는 낮을수록 가깝습니다. 코드에서는 둘을 명시적으로
나눠 저장해 이름 때문에 생기는 혼동을 막았습니다.

#### Q5. similarity threshold로 정답 없음을 판단할 수 있나요?

바로 판단하면 안 됩니다. E5 점수는 높은 범위에 모일 수 있고 corpus와
질문 종류에 따라 분포가 달라집니다. answerable/no-answer 평가셋에서
정답과 오답의 점수 분포를 확인한 뒤 threshold를 정해야 합니다.

#### Q6. 왜 Chroma를 선택했나요?

로컬 설치가 쉽고 LangChain 통합이 있으며 metadata와 영구 저장을
지원해 학습·포트폴리오 데모에 적합합니다. 운영 트래픽, 분산 확장,
필터링 요구가 커지면 Qdrant, pgvector, OpenSearch 같은 대안을 같은
평가셋으로 비교할 수 있습니다.

#### Q7. 문서가 바뀌면 어떻게 다시 색인하나요?

chunk 본문과 ID가 collection 이름의 해시에 포함됩니다. 내용이 바뀌면
새 collection 이름이 생성되어 다시 embedding합니다. 같은 이름인데 ID가
다르면 오류를 내서 조용히 잘못된 색인을 쓰지 않습니다.

#### Q8. 대규모 문서에서는 무엇을 바꿔야 하나요?

embedding batch 크기, GPU 또는 TEI 서빙, 비동기 색인 작업, collection
수명 관리와 외부 Chroma 서버를 고려해야 합니다. 앱 요청 안에서 모든
문서를 embedding하지 않고 별도 ingestion job으로 분리해야 합니다.

#### Q9. 실제 모델 테스트와 가짜 모델 테스트를 왜 나눴나요?

단위 테스트는 코드 계약을 빠르고 결정적으로 확인해야 합니다. 실제 모델은
다운로드, 라이브러리 버전, CPU 속도에 영향을 받습니다. 따라서 가짜
embedding으로 로직을 검증하고 실제 한국어 PDF 통합 테스트로 모델 품질과
호환성을 따로 확인했습니다.

#### Q10. RRF는 어떻게 연결했나요?

BM25와 Chroma가 모두 원본 `chunk.id`를 보존합니다. 각 결과의 점수는 직접
더하지 않고 순위를 `1 / (k + rank)`로 바꿔 같은 ID끼리 합산합니다.
한 검색기에만 나온 chunk도 후보로 유지할 수 있습니다.

### 공식 참고 자료

- [LangChain embedding 개념과 인터페이스](https://docs.langchain.com/oss/python/integrations/embeddings)
- [LangChain Chroma 통합](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [multilingual-e5-small 모델 카드](https://huggingface.co/intfloat/multilingual-e5-small)

---

## 개발자용 실행 부록

의존성을 설치하고 앱을 실행합니다.

```powershell
Set-Location "C:\Users\mae\Desktop\260704\publish-worktrees\gongo-rag"

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

처음 Chroma 검색을 누르면 모델을 내려받습니다. API 키는 필요하지 않습니다.

관련 테스트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe tests\test_vector_search.py
.\.venv\Scripts\python.exe tests\test_document_upload_ui.py
```

코드에서 직접 사용합니다.

```python
import sys

sys.path.insert(0, "src")

from vector_search import ChromaChunkRetriever

retriever = ChromaChunkRetriever(
    chunks,
    persist_directory=".chroma",
)

for result in retriever.search("돈을 얼마나 받을 수 있나요?", k=5):
    print(result.rank, result.similarity)
    print(result.chunk.source_filename, result.chunk.page_label)
    print(result.chunk.text)
```
