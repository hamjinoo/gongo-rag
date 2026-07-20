# 5번 작업: 두 검색기의 순위를 합치기

## 이 문서 읽는 방법

- 처음 공부한다면 `한 줄로 설명하면`부터 `다음 작업`까지 읽습니다.
- 면접을 준비한다면 `면접 대비 기술 설명`부터 읽습니다.
- 직접 실행할 때는 맨 아래 `개발자용 실행 부록`을 봅니다.

## 한 줄로 설명하면

BM25와 Chroma가 각각 뽑은 검색 순위를 **RRF라는 투표 방식으로 합치는
작업**입니다.

```text
질문
├── BM25: 같은 단어를 잘 찾는 심사위원
└── Chroma: 같은 뜻을 잘 찾는 심사위원
             ↓
       두 순위표를 RRF로 합치기
             ↓
          통합 Top-k
```

## 이번 작업에서 얻어야 하는 것

다음 다섯 가지를 이해하면 됩니다.

1. BM25 점수와 Chroma similarity는 단위가 달라 바로 더하면 안 됩니다.
2. RRF는 원래 점수 대신 각 검색기의 **순위**를 사용합니다.
3. 양쪽 검색기에서 높은 순위인 chunk가 더 많은 표를 받습니다.
4. 한 검색기에만 나온 chunk도 후보에서 사라지지 않습니다.
5. RRF도 항상 좋아지는 것은 아니므로 실제 질문으로 평가해야 합니다.

## 왜 원래 점수를 그냥 더하지 않나요?

두 검색기의 점수는 의미가 다릅니다.

```text
BM25 점수: 8.42
Chroma similarity: 0.86
```

이 숫자를 그대로 더하면 BM25의 숫자가 크다는 이유만으로 BM25가 대부분을
결정할 수 있습니다. 그렇다고 Chroma 점수에 10을 곱하는 것도 근거 없는
조정입니다.

```text
잘못된 예:
8.42 + 0.86 = 9.28

문제:
두 숫자의 자와 단위가 서로 다름
```

RRF는 점수를 버리고 “몇 등이었는가?”만 봅니다.

## RRF를 심사위원 투표로 이해하기

두 명의 심사위원이 있다고 생각해봅시다.

```text
BM25 심사위원
1위 A
2위 B
3위 C

Chroma 심사위원
1위 C
2위 B
3위 D
```

B와 C는 두 심사위원 모두에게 표를 받았습니다. A와 D는 한쪽에만
있습니다. RRF는 높은 순위의 표에 더 큰 값을 주고 같은 문서의 표를
더합니다.

```text
C: BM25 3위 표 + Chroma 1위 표
B: BM25 2위 표 + Chroma 2위 표
A: BM25 1위 표
D: Chroma 3위 표
```

이 예에서는 C와 B가 통합 순위 위쪽으로 올라갑니다.

## RRF 공식

```text
각 검색기의 기여 = 1 / (k + 그 검색기에서의 순위)

최종 RRF 점수
  = BM25 기여
  + Chroma 기여
```

현재 기본값은 `k=60`입니다.

```text
1위 표 = 1 / (60 + 1) = 0.016393
2위 표 = 1 / (60 + 2) = 0.016129
3위 표 = 1 / (60 + 3) = 0.015873
```

예를 들어 같은 chunk가 BM25 3위, Chroma 1위라면:

```text
1 / 63 + 1 / 61
= 0.015873 + 0.016393
= 0.032266
```

한쪽 검색기에 나오지 않았다면 그 검색기의 기여는 0입니다.

## 공식의 k는 무엇인가요?

여기서 `k`는 최종 결과 개수인 top-k와 다른 값입니다. 코드에서는 혼동을
막기 위해 `rank_constant`라고 부릅니다.

- 값이 작으면 1위와 아래 순위의 차이가 커집니다.
- 값이 크면 아래 순위까지 비교적 비슷한 표를 받습니다.
- 현재는 널리 쓰이는 시작값 `60`을 사용합니다.

이 값도 영원한 정답은 아닙니다. 고정 질문셋에서 비교해야 합니다.

## 후보 수는 왜 따로 필요한가요?

최종 5개를 보여주더라도 각 검색기에서 5개보다 많은 후보를 받아 합칠 수
있습니다.

```text
BM25 후보 20개 ─┐
                 ├→ RRF → 최종 5개
Chroma 후보 20개 ┘
```

이를 후보 창 또는 `fetch_k`라고 부릅니다.

- 너무 작으면 한 검색기의 좋은 후보가 결합 전에 잘립니다.
- 너무 크면 느려지고 낮은 순위의 잡음이 늘 수 있습니다.
- 현재 시작값은 20이며 문서가 20개보다 적으면 전체 chunk를 사용합니다.

## 현재 전체 검색 흐름

```text
PDF·DOCX·이미지
→ 텍스트 추출
→ chunking
→ 같은 DocumentChunk 두 갈래
   ├── Kiwi BM25
   └── E5 embedding + Chroma
→ chunk ID 기준 RRF
→ 통합 후보와 출처 표시
```

RRF는 새 문서를 만들거나 embedding하지 않습니다. 이미 존재하는 두
순위표를 합치는 역할만 합니다.

## 화면에서 사용하는 방법

1. `1. 문서 넣기` 탭에서 문서를 올립니다.
2. `텍스트 추출`을 누릅니다.
3. `Chunk 만들기`를 누릅니다.
4. BM25와 Chroma에서 같은 질문을 각각 검색해봅니다.
5. `5. RRF 통합 검색`으로 내려갑니다.
6. 같은 질문을 입력하고 `RRF 통합 검색`을 누릅니다.
7. 각 결과에서 BM25 순위와 Chroma 순위를 확인합니다.
8. 두 검색기의 RRF 기여가 어떻게 더해졌는지 확인합니다.
9. 필요하면 RRF 결과 JSON을 받습니다.

처음 실행하는 컴퓨터라면 Chroma 검색과 마찬가지로 embedding 모델을
내려받는 시간이 필요할 수 있습니다.

## 검색 결과를 어떻게 읽나요?

```text
1위 · RRF 0.032787 · 공고.pdf · 페이지 1
BM25 1위 · 원점수 7.231
Chroma 1위 · similarity 0.882
BM25 RRF 기여 0.016393
Chroma RRF 기여 0.016393
```

원점수는 왜 각 검색기가 그런 순위를 만들었는지 살펴보기 위해 표시할 뿐,
RRF 계산에는 사용하지 않습니다.

## 실제 공고문 성공 사례

5페이지 한국어 공고문을 11개 chunk로 나눠 검색했습니다.

```text
질문: 지원대상 대전시 소재 기업

지원대상 chunk:
BM25 1위
Chroma 1위
RRF 1위
RRF 점수 0.032787
페이지 1
```

정확한 키워드와 의미가 모두 맞았기 때문에 두 검색기가 같은 chunk에
1위 표를 줬습니다.

## 실제 공고문 실패 사례

더 자연스럽게 질문하자 다른 결과가 나왔습니다.

```text
질문: 어떤 회사가 지원받을 수 있나요?

진짜 보고 싶은 지원대상 chunk:
BM25 7위
Chroma 1위
RRF 4위
```

Chroma는 뜻을 보고 지원대상을 1위로 찾았지만, BM25는 `지원`, `수`처럼
여러 곳에 나오는 단어에 영향을 받았습니다. 동일 가중치 RRF로 합치자
BM25의 좋지 않은 순위가 섞여 정답 chunk가 4위로 내려갔습니다.

이것은 구현 오류가 아니라 **RRF가 항상 개별 검색기보다 좋아지는 것은
아니라는 실패 사례**입니다.

이 사례에서 얻는 교훈은 다음과 같습니다.

1. RRF를 추가했다는 사실만으로 성능 향상을 주장하면 안 됩니다.
2. 한국어 BM25 tokenizer와 불용어를 더 살펴봐야 합니다.
3. BM25와 Chroma 가중치를 평가로 결정할 수 있습니다.
4. 상위 후보 본문을 직접 읽는 reranker가 필요한 근거가 됩니다.

## 공부하면서 직접 해볼 실험

다음 질문을 BM25, Chroma, RRF에 똑같이 넣습니다.

```text
A. 지원대상 대전시 소재 기업
B. 어떤 회사가 지원받을 수 있나요?
C. 돈을 얼마나 받을 수 있나요?
D. 접수 마감일은 언제인가요?
```

기록할 것은 다음과 같습니다.

```text
질문:
정답 chunk:
BM25 순위:
Chroma 순위:
RRF 순위:
RRF가 좋아졌나, 나빠졌나:
그 이유에 대한 추측:
```

`rank_constant`와 후보 수를 바꾸기 전에 기본값 결과를 먼저 기록합니다.
한 번에 하나의 설정만 바꿔야 원인을 설명할 수 있습니다.

## 현재 한계

- RRF는 본문의 내용을 다시 읽지 않고 순위만 봅니다.
- 두 검색기가 함께 틀리면 RRF도 틀립니다.
- 한 검색기의 나쁜 순위가 좋은 검색 결과를 아래로 내릴 수 있습니다.
- 기본 동일 가중치가 이 데이터의 최적값이라는 근거는 아직 없습니다.
- `rank_constant=60`, 후보 20개는 dev 평가에 사용한 시작값이며 아직 최적값은 아닙니다.
- 자연어 질문에서 한국어 BM25의 흔한 단어가 잡음이 될 수 있습니다.
- dev 20문항으로 설정을 고른 뒤 test normal 10문항을 한 번 실행했습니다. 이제 test 결과에 맞춰 검색 설정을 다시 바꾸지 않습니다.
- RRF 후보는 현재 CrossEncoder reranker에 연결했지만 LLM 답변에는 아직 연결하지 않았습니다.

## 5번 작업의 완료 상태

- [x] BM25와 Chroma를 같은 질문으로 각각 실행
- [x] 공통 chunk ID로 결과 결합
- [x] 원래 점수를 직접 더하지 않음
- [x] `1 / (rank_constant + rank)` 공식 구현
- [x] 기본 `rank_constant=60`
- [x] 검색기별 후보 창 분리
- [x] 한 검색기에만 나온 후보 보존
- [x] 같은 검색기의 중복 ID 한 번만 반영
- [x] 같은 ID가 다른 원문이면 오류 처리
- [x] 결정적인 동점 처리
- [x] BM25·Chroma 원점수와 RRF 기여 표시
- [x] RRF 결과 JSON 다운로드
- [x] 자동 테스트와 실제 한국어 PDF 성공·실패 사례 확인
- [x] reranker 추가
- [x] 고정 질문셋에서 BM25·Chroma·RRF·reranker 비교

## 이후 작업

RRF가 모은 후보를 질문과 함께 직접 읽고 순서를 다시 정하는 reranker까지
구현했습니다.

```text
BM25 + Chroma에서 각각 후보 20개
→ RRF 상위 후보 7개
→ CrossEncoder reranker
→ 최종 근거 5개
```

구현 내용과 실제 한국어 PDF 결과는
[6번 작업: RRF 후보를 질문에 더 잘 맞는 순서로 다시 세우기](RERANKER.md)에서
이어집니다. 고정 질문으로 비교한 결과는
[7번 작업: 같은 시험지로 검색기 성적 비교하기](EVALUATION.md)에 있습니다.
후보 수 비교에서는 7개를 선택했습니다. 작은 MiniLM은 약 9.8배 빨랐지만 Hit@1이
0.85에서 0.70으로 떨어져 BGE를 기본값으로 유지했습니다. Cohere API adapter도
같은 평가기에 연결했지만 API 키가 없어 실제 비교는 보류했습니다. 로컬 BGE와 후보
7개로 test를 한 번 실행해 검색 설정을 잠갔으며, 다음은 LangGraph 답변·재검색·거절
흐름입니다.

RRF는 순위표만 보지만 reranker는 `질문 + chunk 본문`을 한꺼번에 읽습니다.
따라서 자연스러운 질문에서 RRF 4위로 내려간 지원대상 chunk를 다시 올릴
가능성이 있습니다. dev 평가에서 RRF Hit@1 0.60은 후보 7개 reranker 적용 후
0.85로 올랐고, 평균 지연은 후보 10개 기준 약 6.28초에서 4.20초로 줄었습니다.

---

## 면접 대비 기술 설명

### 30초 설명

> Kiwi BM25와 E5/Chroma의 이질적인 검색 점수를 직접 정규화하지 않고
> Reciprocal Rank Fusion으로 결합했습니다. 각 결과를 공통 chunk ID로
> 합치고 기본 `rank_constant=60`에서 `1 / (60 + rank)`를 검색기별로
> 더합니다. 한 검색기에만 나온 후보도 유지하고, 원본 BM25 점수와 vector
> similarity는 설명용으로만 보존합니다. 중복 ID, 서로 다른 원문의 ID
> 충돌과 동점 순서를 명시적으로 처리했으며, 실제 한국어 PDF에서 성공
> 사례와 RRF가 Chroma 단독보다 나빠진 실패 사례도 함께 기록했습니다.

### 코드 구조

| 파일 | 책임 |
|---|---|
| [`src/hybrid_search.py`](../src/hybrid_search.py) | RRF 공식, 후보 결합, 통합 결과 모델 |
| [`src/hybrid_search_ui.py`](../src/hybrid_search_ui.py) | RRF 설정, 기여도·출처 표시, JSON |
| [`app.py`](../app.py) | BM25 → Chroma → RRF 비교 화면 연결 |
| [`tests/test_hybrid_search.py`](../tests/test_hybrid_search.py) | 공식·결합·예외·결정성 테스트 |
| [`tests/test_document_upload_ui.py`](../tests/test_document_upload_ui.py) | 업로드부터 RRF 화면까지 사용자 흐름 |

### 왜 score normalization 대신 RRF인가요?

BM25 점수는 term frequency, IDF와 문서 길이에서 나오고 Chroma similarity는
embedding 벡터의 cosine에서 나옵니다. 질문과 corpus가 바뀌면 두 점수의
분포도 달라집니다.

Min-max normalization은 후보 집합의 최댓값과 최솟값에 민감하고, 고정
가중 합은 튜닝 데이터가 필요합니다. RRF는 각 검색기의 상대 순위만
사용하므로 이질적인 검색기를 간단하고 재현 가능하게 결합하는 기준선입니다.

가중 합이 항상 나쁘다는 뜻은 아닙니다. 충분한 평가 데이터가 생기면
normalized score fusion과 weighted RRF를 비교할 수 있습니다.

### 공식과 구현

두 검색기에서 chunk `d`의 최종 점수는 다음과 같습니다.

```text
RRF(d)
  = weight_bm25 / (rank_constant + rank_bm25(d))
  + weight_vector / (rank_constant + rank_vector(d))
```

해당 검색기의 후보에 없으면 그 항은 0입니다. 현재 UI 기준선은 두 weight가
모두 1입니다. 코어 구현은 향후 평가를 위해 서로 다른 weight도 지원합니다.

### Rank constant의 역할

`rank_constant`가 크면 인접 순위의 점수 차이가 작아집니다.

```text
k=60: 1위 0.016393, 10위 0.014286
k=1:  1위 0.500000, 10위 0.090909
```

큰 값은 여러 검색기의 합의에 무게를 두고, 작은 값은 각 검색기의 최상단
결과에 더 큰 영향력을 줍니다. `60`은 널리 쓰이는 기본값이지만 프로젝트의
최적값으로 주장하지 않습니다.

### Candidate window와 top-k

`fetch_k`는 각 검색기에서 결합 전에 가져오는 결과 수이고, `k`는 결합 후
사용자에게 돌려줄 결과 수입니다.

```python
candidate_k = max(result_k, configured_fetch_k)
bm25_results = bm25.search(query, k=candidate_k)
vector_results = vector.search(query, k=candidate_k)
fused_results = fuse(...)[0:result_k]
```

최종 결과 10개를 요청했는데 후보 창이 5개라서 결과가 부족해지지 않도록
실제 후보 수는 최소한 최종 결과 수 이상으로 확장합니다.

### 공통 chunk ID

BM25와 Chroma는 점수 객체가 다르지만 모두 원본 `DocumentChunk`를
보존합니다.

```text
BM25 SearchResult.chunk.id
Chroma VectorSearchResult.chunk.id
```

RRF는 이 ID를 join key로 사용합니다. 본문 문자열만으로 합치면 overlap이나
중복 문장 때문에 서로 다른 위치의 chunk를 잘못 합칠 수 있습니다.

### ID 충돌 검증

같은 ID가 두 검색기에 있지만 `DocumentChunk` 전체 값이 다르면
`HybridSearchMismatchError`를 발생시킵니다.

조용히 한쪽 metadata를 선택하면 잘못된 파일명이나 페이지를 인용할 수 있기
때문입니다.

### 중복과 동점 처리

한 검색기가 같은 ID를 두 번 반환해도 가장 먼저 나온 순위만 반영합니다.
그렇지 않으면 한 검색기가 같은 문서에 여러 표를 주는 문제가 생깁니다.

RRF 점수가 같은 경우 다음 순서로 정렬합니다.

1. 더 많은 검색기에서 발견된 후보
2. 두 검색기 중 가장 좋은 순위
3. 원본 chunk 순서
4. chunk ID

이 기준을 명시해 실행할 때마다 결과 순서가 바뀌지 않도록 했습니다.

### 원점수 보존

`HybridSearchResult`에는 다음 값이 있습니다.

```text
rank
rrf_score
chunk
bm25_rank
bm25_score
bm25_contribution
vector_rank
vector_similarity
vector_contribution
```

BM25 점수와 vector similarity는 RRF 계산에는 쓰지 않지만 디버깅과
설명 가능성을 위해 남깁니다. “왜 이 chunk가 1위인가?”를 각 검색기의
순위와 기여도로 추적할 수 있습니다.

### 시간 복잡도

BM25와 Chroma 검색 비용을 제외한 fusion 자체는 후보 수를 `M`이라 할 때:

```text
후보 수집: O(M)
최종 정렬: O(M log M)
메모리: O(M)
```

실제 비용은 embedding 검색과 BM25 후보 생성이 더 큽니다. RRF 계산 자체는
가볍습니다.

### 오류 처리와 부분 결과

- BM25 결과가 비어도 Chroma 후보는 유지됩니다.
- Chroma에만 나온 후보도 최종 결과가 될 수 있습니다.
- weight가 0인 검색기는 실행하지 않습니다.
- 두 weight가 모두 0이면 설정 오류로 거절합니다.
- 음수, NaN weight와 잘못된 rank·constant·후보 수를 거절합니다.

운영 환경에서 한 검색기 장애 시 부분 결과로 계속 응답할지는 별도의
정책입니다. 현재는 검색기 실행 오류를 UI에 보여줘 조용한 품질 저하를
막습니다.

### 테스트 전략

자동 테스트에서는 실제 모델 대신 미리 정한 BM25·Chroma 순위표를
주입합니다. RRF는 원점수 대신 순위만 봐야 하므로 이 방식으로 공식과
결정성을 정확히 확인할 수 있습니다.

검증 항목:

- 손계산 `1 / (60 + rank)` 일치
- 두 독립 순위표 결합
- raw score 크기가 바뀌어도 같은 순위면 RRF 결과 유지
- 한 검색기에만 나온 후보 유지
- 같은 검색기 중복 ID 한 번만 계산
- metadata와 원점수 보존
- 최종 결과보다 작은 후보 창 자동 확장
- 빈 질문과 잘못된 설정 처리
- 같은 ID의 서로 다른 원문 거절
- 기존 Hit@k 평가용 text adapter
- 업로드 → 추출 → chunking → RRF UI 흐름

기존 기능을 포함한 자동 테스트 **71개가 모두 통과**했습니다. 실제
한국어 PDF에서는 BM25·Chroma 모두 1위인 성공 사례와 Chroma 1위가 RRF
4위로 떨어진 실패 사례를 모두 확인했습니다.

### 기술 선택과 트레이드오프

| 선택 | 이유 | 한계 |
|---|---|---|
| 직접 구현한 RRF | 공식과 기여도를 투명하게 설명 가능 | 검색 엔진 내장 RRF보다 분산 최적화 부족 |
| rank_constant 60 | 널리 쓰이는 재현 가능한 시작값 | 작은 corpus 최적값은 아님 |
| 동일 가중치 | 튜닝 전 공정한 기준선 | 약한 검색기가 결과를 악화시킬 수 있음 |
| 후보 20개 | recall과 처리량의 시작점 | 데이터에 따라 너무 작거나 클 수 있음 |
| chunk ID 결합 | 본문 중복에도 출처를 정확히 유지 | upstream ID 안정성이 필수 |
| 원점수 보존 | 오류 분석과 UI 설명 가능 | 결과 객체가 조금 커짐 |

### 예상 면접 질문

#### Q1. 왜 BM25와 vector score를 직접 더하지 않았나요?

두 점수는 생성 방식과 분포가 다릅니다. 고정 가중 합을 쓰려면 정규화와
튜닝 데이터가 필요합니다. RRF는 각 검색기의 상대 순위만 사용해 간단하고
강건한 하이브리드 기준선을 만들 수 있습니다.

#### Q2. RRF 공식에서 60은 무엇인가요?

상위 순위의 영향력을 완화하는 rank constant입니다. 값이 클수록 아래
순위와 점수 차이가 작아지고 여러 검색기의 합의가 중요해집니다. 널리 쓰이는
기본값일 뿐 최적값은 평가로 결정해야 합니다.

#### Q3. 후보 창은 왜 최종 top-k보다 커야 하나요?

각 검색기 5위 밖에 있던 문서가 다른 검색기에서도 발견되면 결합 후 상위로
올라올 수 있습니다. 결합 전에 너무 일찍 자르면 이런 후보를 잃습니다.

#### Q4. 한 검색기에만 나온 문서는 버리나요?

아닙니다. 해당 검색기의 RRF 기여만 받고 후보에 남습니다. 하이브리드
검색의 목적 중 하나가 서로 다른 실패 유형을 보완하는 것이기 때문입니다.

#### Q5. RRF가 항상 개별 검색기보다 좋은가요?

아닙니다. 실제 자연어 질문에서 Chroma 1위 정답이 BM25 7위 때문에 RRF
4위가 된 사례가 있습니다. 검색기 품질, 가중치, 후보 창과 질문 유형에 따라
악화될 수 있으므로 같은 평가셋으로 비교해야 합니다.

#### Q6. 그러면 weighted RRF를 바로 적용하면 되나요?

코드는 weight를 지원하지만 기본 UI는 1:1입니다. 실패 사례 몇 개만 보고
weight를 정하면 과적합될 수 있습니다. dev 질문셋으로 선택하고 test
질문셋에서 한 번만 확인해야 합니다.

#### Q7. 동점은 어떻게 처리했나요?

검색기 수, 가장 좋은 개별 순위, 원본 chunk 순서, chunk ID 순으로
결정합니다. 명시적인 tie-breaker가 없으면 실행 환경에 따라 순서가 흔들려
평가 재현성이 떨어질 수 있습니다.

#### Q8. 왜 LangChain EnsembleRetriever를 사용하지 않았나요?

이번 단계의 학습 목표는 RRF 공식, ID 결합과 기여도를 직접 검증하는
것입니다. 직접 구현해 raw score가 계산에 들어가지 않는지 손계산
테스트했습니다. 이후 retriever가 많아지면 프레임워크 구현과 결과·지연을
비교할 수 있습니다.

#### Q9. 운영 환경에서는 어디에서 RRF를 수행하나요?

작은 서비스에서는 애플리케이션 계층에서 합칠 수 있습니다. 대규모 환경은
Elasticsearch/OpenSearch 같은 검색 엔진의 내장 RRF를 사용하면 후보
전송과 분산 검색 비용을 줄일 수 있습니다.

#### Q10. RRF 다음에 왜 reranker가 필요한가요?

RRF는 순위만 보고 질문과 chunk 본문의 세부 관계를 읽지 않습니다.
CrossEncoder reranker는 질문과 각 후보 본문을 함께 입력해 더 정확하게
재정렬할 수 있습니다. 계산 비용이 크므로 전체 문서가 아니라 RRF 상위
후보에만 적용합니다.

### 공식 참고 자료

- [RRF 원 논문: Cormack, Clarke, Büttcher, SIGIR 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [Elasticsearch RRF 공식과 기본 설정](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

---

## 개발자용 실행 부록

앱을 실행합니다.

```powershell
Set-Location "C:\Users\mae\Desktop\260704\publish-worktrees\gongo-rag"

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

관련 테스트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe tests\test_hybrid_search.py
.\.venv\Scripts\python.exe tests\test_document_upload_ui.py
```

코드에서 직접 사용합니다.

```python
import sys

sys.path.insert(0, "src")

from bm25 import BM25ChunkRetriever
from hybrid_search import HybridRRFRetriever
from vector_search import ChromaChunkRetriever

bm25 = BM25ChunkRetriever(chunks, tokenizer_name="kiwi")
vector = ChromaChunkRetriever(chunks, persist_directory=".chroma")
hybrid = HybridRRFRetriever(
    bm25,
    vector,
    rank_constant=60,
    fetch_k=20,
)

for result in hybrid.search("어떤 회사가 지원받을 수 있나요?", k=5):
    print(result.rank, result.rrf_score)
    print("BM25", result.bm25_rank, result.bm25_contribution)
    print("Chroma", result.vector_rank, result.vector_contribution)
    print(result.chunk.source_filename, result.chunk.page_label)
```
