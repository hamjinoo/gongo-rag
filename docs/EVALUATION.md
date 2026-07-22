# 7번 작업: 같은 시험지로 검색기 성적 비교하기

이 문서는 두 가지 목적으로 작성했습니다.

1. 처음 보는 사람도 **고정 평가 질문이 왜 필요한지** 이해하기
2. 면접에서 Hit@k, MRR, nDCG, dev/test, latency를 설명하기

---

## 0. 한눈에 보기

### 이번에 만든 것

```text
공고문 3개
  ↓
같은 Chunk 38개
  ↓
같은 dev 질문 20개
  ↓
┌────────┬────────┬────────┬──────────┐
│ BM25   │ Chroma │ RRF    │ Reranker │
└────────┴────────┴────────┴──────────┘
  ↓
Hit@1·3·5·10 + MRR + nDCG + 지연 시간
  ↓
문항별 첫 정답 순위와 실패 목록 저장
```

### 왜 지금 평가하나요?

이제 다음 기능이 모두 있습니다.

- 텍스트 추출
- Chunking
- BM25
- E5 embedding과 Chroma
- RRF
- CrossEncoder reranker

기능이 많아졌다고 좋은 RAG가 된 것은 아닙니다. 같은 질문으로 비교해야 어느 단계가
정말 도움이 됐는지 알 수 있습니다.

### 이번 단계에서 얻어야 하는 것

- 고정 질문셋은 누가 만들고 왜 고정하는가?
- 정답 Chunk를 어떻게 판정하는가?
- Hit@k, MRR, nDCG는 각각 무엇을 보는가?
- 품질과 속도를 왜 같이 봐야 하는가?
- dev와 test를 왜 나누는가?
- Ragas는 왜 검색 평가 다음에 사용하는가?

---

## 1. 12살도 이해할 수 있는 설명

### 네 명에게 같은 시험 보기

BM25, Chroma, RRF, reranker가 학생이라고 생각해 봅시다.

학생마다 다른 문제를 주면 누가 더 잘하는지 알 수 없습니다.

```text
BM25에게 쉬운 문제
Chroma에게 어려운 문제
→ 점수를 비교할 수 없음
```

그래서 다음을 모두 같게 고정합니다.

```text
같은 공고문
같은 Chunk
같은 질문
같은 정답 근거
같은 결과 개수 k
```

이제 검색 방법만 바뀌므로 성적 차이가 어디에서 생겼는지 설명할 수 있습니다.

---

## 2. 고정 질문셋은 누가 만드나요?

최종 책임은 **문서를 이해하는 사람**에게 있습니다.

이 프로젝트에서는 다음 순서가 안전합니다.

1. 사람이 실제 사용자가 물을 질문을 적습니다.
2. 원문에서 답이 적힌 정확한 문장을 표시합니다.
3. 답이 정말 해당 문서에 있는지 확인합니다.
4. AI는 질문 초안, 표현 변형, 형식 검사를 도울 수 있습니다.
5. 최종 정답 여부는 사람이 승인합니다.

AI가 질문과 정답을 전부 자동으로 만들고 그대로 채점하면, AI가 잘못 만든 정답을
기준으로 시스템을 고칠 위험이 있습니다.

### 현재 골든셋

```text
공고문 3개
전체 질문 36개
├── 답이 있는 normal 30개
└── 문서에 답이 없는 no-answer 6개
```

현재 문서와 Chunk를 다시 검사한 결과:

- 지정한 문서 3개가 모두 존재함
- normal 30개 정답 문장이 원문에 모두 존재함
- 기본 Chunk 38개 안에서도 정답 문장을 모두 찾을 수 있음
- overlap 때문에 정답 Chunk가 여러 개인 문항도 별도로 기록함

즉 현재 데이터는 형식만 있는 가짜 예제가 아니라 실제 공고문과 연결된 평가셋입니다.

---

## 3. dev와 test는 무엇인가요?

### dev

개발하면서 반복해서 보는 연습 시험입니다.

```text
normal 20개
no-answer 3개
```

Chunk 크기, 후보 수, 모델 등을 바꿀 때 dev 결과를 비교합니다.

### test

최종 선택이 끝난 뒤 확인하는 시험입니다.

```text
normal 10개
no-answer 3개
```

test 결과를 계속 보면서 설정을 바꾸면 test 문제의 답에 맞춘 시스템이 됩니다.
실제 새 질문에서도 좋은지 알기 어려워집니다.

그래서 이번 작업에서는 **dev만 실행하고 test는 실행하지 않았습니다.**

---

## 4. 정답은 어떻게 표시하나요?

한 문항은 다음 정보를 가집니다.

```json
{
  "id": "q001",
  "type": "normal",
  "split": "dev",
  "question": "대전의 어떤 기업이 신청할 수 있나요?",
  "answer_span": "대전에 본사, 지사, 또는 기업부설연구소가 소재한 기업",
  "doc_id": "공고문.txt",
  "note": "지원자격 질문"
}
```

### 왜 Chunk 번호를 정답으로 쓰지 않나요?

Chunk 크기를 바꾸면 Chunk 번호가 바뀝니다.

```text
700자 Chunk에서 정답: chunk 4
500자 Chunk로 변경: 같은 문장이 chunk 6
```

정답을 `chunk 4`로 저장하면 Chunk 실험을 할 수 없습니다.

그래서 다음 두 가지로 정답을 찾습니다.

- 어느 문서인가? `doc_id`
- 어떤 원문 문장인가? `answer_span`

평가를 시작할 때 현재 Chunk에서 그 문장을 포함한 Chunk ID를 다시 계산합니다.

---

## 5. 답이 없는 질문은 왜 검색 점수에서 빼나요?

Chroma와 같은 검색기는 질문을 받으면 항상 가까운 Chunk를 반환합니다.

하지만 검색 결과가 나온 것과 문서에 답이 있는 것은 다른 문제입니다.

```text
normal 질문
→ 정답 Chunk를 몇 위에 찾았는지 평가

no-answer 질문
→ 최종 답변이 "정보 없음"이라고 거절하는지 평가
```

현재 단계는 **검색 순위 평가**이므로 normal만 사용합니다.
no-answer 6개는 이후 LangGraph와 최종 답변 거절 평가에서 사용합니다.

두 문제를 한 숫자로 섞지 않는 것이 중요합니다.

---

## 6. Hit@k란 무엇인가요?

### Hit@1

첫 번째 결과가 정답인 질문의 비율입니다.

```text
20문제 중 16문제의 1위가 정답
Hit@1 = 16 / 20 = 0.8
```

### Hit@3

상위 3개 안에 정답이 있는 질문의 비율입니다.

사용자에게 근거 3개를 보여주거나 LLM에 3개를 전달한다면 중요한 값입니다.

### Hit@10

reranker에게 전달할 후보 10개 안에 정답이 있는지 봅니다.

```text
Hit@10이 낮음
→ reranker가 정답을 읽을 기회조차 없음
```

---

## 7. MRR은 무엇인가요?

MRR은 **첫 정답이 얼마나 위에 있는지** 봅니다.

```text
정답 1위 → 1 / 1 = 1.0점
정답 2위 → 1 / 2 = 0.5점
정답 5위 → 1 / 5 = 0.2점
정답 없음 → 0점
```

모든 질문의 점수를 평균냅니다.

Hit@10은 1위와 10위를 모두 성공으로 보지만, MRR은 1위에 더 큰 점수를 줍니다.

---

## 8. nDCG는 무엇인가요?

overlap 때문에 같은 정답 문장을 포함한 Chunk가 여러 개 생길 수 있습니다.

nDCG는 정답 Chunk들이 위쪽에 잘 배치됐는지 봅니다.

```text
정답, 정답, 오답
→ 높은 점수

오답, 오답, 정답
→ 더 낮은 점수
```

순위가 내려갈수록 `log2(rank + 1)`로 점수를 할인하고, 완벽한 순서의 점수로 나눠
0과 1 사이로 정규화합니다.

현재 평가는 원문 정답 문장을 포함하면 relevant, 아니면 non-relevant인
binary relevance를 사용합니다.

---

## 9. 왜 지연 시간도 재나요?

가장 정확한 검색기가 너무 느리면 실제 서비스에서 사용하기 어렵습니다.

```text
정확도 1% 증가
응답 시간 100배 증가
```

이 선택이 항상 옳은 것은 아닙니다.

현재는 질문별로 검색 호출 시간을 측정하고 다음을 기록합니다.

- 평균 latency
- p95 latency: 질문 95%가 이 시간 안에 끝나는 경계

현재 측정값은 개발 PC의 참고값입니다. 운영 성능을 보장하지 않습니다.

---

## 10. 실제 dev 평가 결과

실행 조건:

```text
문서 3개
Chunk 38개
dev normal 질문 20개
Kiwi BM25
multilingual-e5-small + Chroma
RRF rank constant 60, 검색기별 후보 20개
bge-reranker-v2-m3, RRF 후보 10개
CPU
```

| 검색 방식 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@10 | 평균 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.700 | 0.950 | 1.000 | 1.000 | 0.827 | 0.854 | 0.6 | 0.8 |
| Chroma | 0.600 | 0.700 | 0.800 | 0.900 | 0.673 | 0.711 | 16.3 | 17.5 |
| RRF | 0.600 | 0.850 | 0.900 | 1.000 | 0.734 | 0.790 | 19.2 | 23.1 |
| Reranker | **0.800** | **1.000** | **1.000** | **1.000** | **0.900** | **0.914** | 6277.0 | 6625.4 |

전체 결과:

- [후보 수 비교 리포트](../experiments/reranker-candidate-comparison-dev.md)
- [최초 후보 10개 JSON 원본](../experiments/reranker-candidates-10-dev.json)

---

## 11. 결과를 어떻게 해석해야 하나요?

### 11-1. Reranker 품질은 가장 좋았습니다

```text
RRF Hit@1 0.60
→ Reranker Hit@1 0.80

RRF MRR 0.734
→ Reranker MRR 0.900
```

RRF에서 정답이 낮았던 다음 문항을 reranker가 1위로 올렸습니다.

| 문항 | RRF | Reranker |
|---|---:|---:|
| q028 | 7위 | 1위 |
| q030 | 5위 | 1위 |
| q031 | 6위 | 1위 |

### 11-2. 후보 7개로 품질을 유지하며 지연을 줄였습니다

```text
최초 후보 10개: Hit@1 0.80, MRR 0.900, 평균 6277.0ms
현재 후보 7개: Hit@1 0.85, MRR 0.925, 평균 4197.2ms
```

후보 7개는 dev에서 품질을 유지·개선하면서 평균 지연을 33.1% 줄였습니다.
후보 5개는 더 빨랐지만 RRF 6·7위에 있던 `q028`, `q031`의 정답을 잃어
채택하지 않았습니다.

전체 비교는 [reranker 후보 수 비교](../experiments/reranker-candidate-comparison-dev.md)에서
확인할 수 있습니다. 4.20초도 실시간 서비스에는 부담이 있으므로 최적화가 필요합니다.

- GPU 또는 ONNX/OpenVINO
- 더 작은 로컬 reranker 또는 양자화
- 필요할 때만 reranker를 실행하는 routing

### 11-3. 작은 모델은 빨랐지만 품질이 떨어졌습니다

후보 7개를 고정한 새 실행에서 568M BGE와 118M MiniLM을 비교했습니다.

| 모델 | Hit@1 | Hit@5 | MRR | 평균 지연 | 최대 프로세스 RSS |
|---|---:|---:|---:|---:|---:|
| BGE | **0.85** | **1.00** | **0.925** | 4513.5ms | 3041.2MB |
| MiniLM | 0.70 | 0.95 | 0.800 | **460.6ms** | **1969.8MB** |

MiniLM은 약 9.8배 빨랐지만 `q028` 정답을 Top 5에서 놓쳤습니다. 따라서 품질
우선 기본값은 BGE로 유지합니다. 자세한 모델 검토와 실패 사례는
[작은 로컬 reranker 비교](../experiments/reranker-model-comparison-dev.md)에
기록했습니다.

### 11-4. 잠근 설정으로 test를 한 번 확인했습니다

dev 결과를 본 뒤 `local BGE + 후보 7개`를 고정하고 test normal 10문항을 한 번
실행했습니다.

| 검색 방식 | Hit@1 | Hit@3 | Hit@5 | MRR | 평균 지연 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.80 | 0.90 | 0.90 | 0.833 | 0.7ms |
| Chroma | 0.50 | 0.70 | 0.90 | 0.633 | 20.1ms |
| RRF | 0.80 | 0.90 | 0.90 | 0.833 | 17.2ms |
| BGE Reranker | 0.80 | 0.90 | 0.90 | **0.850** | 4324.8ms |

dev보다 BGE Hit@1과 MRR이 조금 낮아졌지만 설정을 다시 바꾸지 않습니다. test는
설정 선택 도구가 아니라 선택을 마지막에 확인하는 자료이기 때문입니다.

자동 평가는 모든 검색기의 `q026`을 실패로 계산했습니다. 그러나 BM25 1위에는
골든셋의 “모집기간” 문장과 표현만 다른 다음 근거가 있었습니다.

```text
제출기간 : 26. 2. 24.(화) 18:00
```

실제 답변은 가능하지만 단일 answer span 판정이 놓친 false negative입니다.
사후에 점수를 올리지 않고 Hit@5 0.90을 그대로 보존하며, 자동 metric과 사람
검토가 함께 필요한 사례로 기록했습니다. 전체 결과는
[test 검색 평가 리포트](../experiments/retrieval-evaluation-test.md)에 있습니다.

### 11-5. BM25가 예상보다 강했습니다

BM25는 Hit@1 0.70, Hit@10 1.00이면서 매우 빨랐습니다.

현재 공고문 질문에는 날짜, 이메일, 숫자, 사업명처럼 정확한 단어가 많아 BM25가
유리했습니다.

이 결과만 보고 “항상 BM25가 Chroma보다 좋다”고 결론 내리면 안 됩니다.
문서가 3개이고 질문 유형이 공고문 정보 검색에 집중돼 있기 때문입니다.

### 11-6. Chroma 단독은 정답 두 개를 후보 10개에 넣지 못했습니다

```text
q028
q031
```

하지만 RRF는 BM25 후보를 함께 사용해 두 문항을 모두 복구했고 Hit@10 1.00이 됐습니다.

이는 현재 파이프라인에서 Chroma를 단독 검색기로 쓰기보다 BM25와 후보를 합치는 이유가
됩니다.

### 11-7. RRF의 역할은 최종 1위보다 후보 회수였습니다

RRF Hit@1은 0.60으로 BM25보다 낮았지만 Hit@10은 1.00이었습니다.

```text
RRF의 현재 역할
→ 정답을 1위로 확정하기
아니라
→ 여러 검색기의 정답 후보를 잃지 않고 reranker에 전달하기
```

---

## 12. 지금 내릴 수 있는 결정

### 유지

- Kiwi BM25
- multilingual E5 + Chroma
- RRF 후보 결합
- 상위 후보 CrossEncoder 재정렬

### 아직 결정하지 않음

- RRF weight와 rank constant를 바꿀지

### 현재 선택

- 후보 수는 7개
- 품질 우선 기본 reranker는 BGE
- MiniLM은 속도 우선 선택지
- 외부 reranker API는 범위에서 제외하고 로컬 BGE로 고정
- test 한 번 완료, 검색 설정 잠금

### 이유

dev 결과는 설정을 고르는 자료입니다. 이 결과에 맞춰 여러 번 수정한 뒤 최종 성능이라고
말하면 안 됩니다.

test normal 10개는 이미 한 번 실행했습니다. 이제 이 결과에 맞춰 모델·후보 수를
다시 바꾸지 않고 LangGraph 답변 단계로 이동합니다.

---

## 13. 실행 방법

### 빠르게 BM25만 확인

```powershell
.venv\Scripts\python.exe src\run_retrieval_evaluation.py `
  --split dev `
  --systems bm25
```

### 네 검색기 비교

```powershell
.venv\Scripts\python.exe src\run_retrieval_evaluation.py `
  --split dev `
  --systems bm25,chroma,rrf,reranker `
  --ks 1,3,5,10 `
  --rerank-candidates 10
```

현재 기본 후보 7개를 평가할 때는 `--ks 1,3,5`를 사용합니다. 후보보다 큰
Hit@k를 요청하면 실제 후보 수가 늘어나므로 CLI가 이 설정을 거부합니다.

결과는 다음 위치에 저장됩니다.

```text
experiments/retrieval-evaluation-dev.json
experiments/retrieval-evaluation-dev.md
```

### test 실행

```powershell
.venv\Scripts\python.exe src\run_retrieval_evaluation.py --split test
```

이 명령은 설정 선택이 끝난 뒤 실행합니다.

---

## 14. 평가 전에 자동으로 검사하는 것

- JSONL 문법
- 중복 질문 ID
- `normal`과 `no_answer` 외 type
- dev/test 외 split
- normal 문항의 빈 정답 또는 문서 ID
- no-answer 문항에 잘못 들어간 정답
- 지정 문서의 존재
- 현재 Chunk에 정답 문장이 실제로 있는지
- 검색 결과의 중복 Chunk ID
- 검색기마다 같은 k를 사용했는지

검사가 실패하면 점수를 만들지 않고 오류를 냅니다.

잘못된 정답으로 그럴듯한 점수를 만드는 것보다 실행을 중단하는 편이 안전합니다.

---

# 면접 대비 기술 설명

## 15. 현재 relevance 판정 방식

normal 문항마다 다음 조건을 모두 만족하는 Chunk를 relevant로 봅니다.

```text
chunk.source_filename == question.doc_id
그리고
normalize(answer_span) in normalize(chunk.text)
```

문서 ID까지 확인하므로 다른 문서에 같은 숫자나 짧은 문장이 있어도 정답으로 잘못
판정할 가능성을 줄입니다.

공백과 줄바꿈은 정규화합니다. PDF 추출 과정에서 줄바꿈 위치가 달라질 수 있기
때문입니다.

---

## 16. metric 공식

### Hit@k

```text
Hit@k = 상위 k개 안에 정답이 있는 질문 수 / 전체 질문 수
```

### Reciprocal Rank

```text
RR = 1 / 첫 정답 순위
```

정답이 없으면 0입니다.

### MRR

```text
MRR = 모든 질문 RR의 평균
```

### DCG

binary relevance에서:

```text
DCG@k = Σ relevance(rank) / log2(rank + 1)
```

### nDCG

```text
nDCG@k = 실제 DCG@k / 완벽한 순서의 IDCG@k
```

---

## 17. 여러 relevant Chunk 처리

overlap 때문에 한 정답 문장이 두 Chunk에 들어갈 수 있습니다.

이 경우:

- Hit@k와 MRR은 가장 먼저 나온 정답 Chunk를 사용
- nDCG는 반환된 relevant Chunk 전체의 순서를 반영

중복 Chunk ID를 검색기가 두 번 반환하면 평가를 중단합니다.
같은 결과를 여러 번 보여줘 nDCG를 부풀리는 일을 막기 위해서입니다.

---

## 18. latency 측정의 한계

현재는 Python `perf_counter`로 각 `search()` 호출의 벽시계 시간을 측정합니다.

포함되는 것:

- query tokenization
- query embedding
- Chroma 조회
- RRF
- CrossEncoder 추론

상황에 따라 포함되지 않을 수 있는 것:

- 첫 모델 다운로드
- 앱 시작 전 준비
- 네트워크 API 지연
- 여러 사용자의 동시 요청 queue

현재 값은 단일 개발 PC와 warm cache의 참고값입니다. 운영 전에는 별도 부하 테스트가
필요합니다.

---

## 19. 데이터 누수와 과적합

dev 질문의 실패를 보고 설정을 바꾸는 것은 허용됩니다.

test 질문의 실패를 보고 다시 설정을 바꾸면 test가 사실상 dev가 됩니다.

안전한 순서:

```text
dev로 비교
→ 설정 선택
→ 선택을 고정
→ test 한 번 실행 완료
→ 결과를 기록하고 검색 설정 종료
```

문서나 사용자 유형이 크게 바뀌면 새 test 데이터를 추가하되, 기존 결과와 새 결과를
구분해 기록합니다.

---

## 20. Ragas와 이번 평가는 무엇이 다른가요?

이번 평가는 정답 원문 문장과 Chunk ID를 사용하는 **결정적인 검색 평가**입니다.

```text
질문 → 검색 결과
정답 Chunk가 몇 위인가?
```

Ragas는 이후 다음과 같은 RAG 전체 품질을 보는 데 사용합니다.

- 검색된 문맥이 질문과 관련 있는가?
- 필요한 근거를 충분히 가져왔는가?
- 최종 답변이 근거에 충실한가?
- 답변이 질문에 맞는가?

일부 Ragas metric은 LLM 판정을 사용하므로 비용과 비결정성이 생길 수 있습니다.
그래서 먼저 현재처럼 빠르고 설명 가능한 검색 metric을 기준선으로 둡니다.

---

## 21. 현재 평가의 한계

- 문서가 3개뿐입니다.
- dev normal 질문이 20개로 작습니다.
- 질문 작성자의 표현 습관이 들어갈 수 있습니다.
- relevance가 원문 문장 포함 여부인 binary 기준입니다.
- 의미상 맞지만 정답 문장을 포함하지 않은 Chunk는 오답으로 처리될 수 있습니다.
- LangGraph 근거 판정 node는 구현했지만 실제 답변 충분성 정확도는 아직 평가하지 않았습니다.
- latency는 개발 PC 단일 실행값입니다.
- 통계적 신뢰구간을 계산하지 않았습니다.
- test normal 문항도 10개뿐이라 일반화하기에는 작습니다.
- `q026`처럼 의미상 맞는 근거를 자동 판정이 놓치는 경우가 있습니다.
- 현재 reranker는 RRF 후보를 입력으로 사용합니다. `BM25 후보 → BGE`와 직접
  비교하지 않아 Embedding·RRF가 최종 성능에 추가로 기여하는지는 아직 확정하지
  않았습니다.

이 한계를 문서에 밝히고 “Hit@1 0.85가 일반적인 모든 공고문에서 보장된다”고 말하지
않습니다.

---

## 22. 30초 면접 설명

> 일반적인 Hybrid RAG를 초기 가설로 구현한 뒤 실제 공고문 3개에서 만든 36문항
> 골든셋을 dev와 test로 분리했습니다. normal 문항은
> answer span과 source document를 이용해 현재 Chunk ID에 다시 매핑했습니다. 같은
> dev 20문항으로 BM25, Chroma, RRF, CrossEncoder reranker를 평가해 Hit@k,
> MRR, binary nDCG와 latency를 비교했습니다. 최초 후보 10개 reranker는 Hit@1
> 0.80, 평균 6.28초였습니다. 후보 7개는 Hit@1 0.85, MRR 0.925를 기록하면서
> 평균 지연을 4.20초로 33.1% 줄여 현재 기본값으로 선택했습니다. 작은 MiniLM은
> 약 9.8배 빨랐지만 Hit@1이 0.70으로 떨어져 BGE를 유지했습니다. BM25는 Hit@1
> 0.70이면서 0.6ms였고, RRF는 BM25보다 항상 좋아지지는 않았습니다. 따라서
> `BM25→BGE`와 `RRF→BGE`를 dev에서 직접 비교하는 ablation을 마지막 설계 검증으로
> 남겼습니다.
> no-answer 문항은 검색 순위와 섞지 않고 이후 답변 거절 평가에 사용하며, 설정을
> 고른 뒤 test normal 10문항을 한 번 실행했습니다. BGE는 Hit@1 0.80, MRR
> 0.85였고, 결과를 본 뒤에는 후보 수나 모델을 다시 조정하지 않았습니다.

---

## 23. 자주 나오는 면접 질문

### Q1. 골든셋은 누가 만들었나요?

실제 사용자 질문을 기준으로 사람이 만들고 원문의 정답 문장을 사람이 확인합니다.
AI는 초안과 형식 검사를 도울 수 있지만 정답의 최종 책임은 사람에게 있습니다.

### Q2. 왜 Chunk ID를 정답으로 저장하지 않았나요?

Chunk 크기와 overlap을 바꾸면 ID가 달라지기 때문입니다. 문서 ID와 원문 answer
span을 저장하고 실행 시 현재 Chunk에 다시 매핑합니다.

### Q3. Hit@k와 MRR의 차이는 무엇인가요?

Hit@k는 k 안에 있기만 하면 성공입니다. MRR은 첫 정답이 1위에 가까울수록 더 높은
점수를 줍니다.

### Q4. nDCG를 왜 추가했나요?

overlap으로 relevant Chunk가 여러 개일 수 있어, 정답들이 위쪽에 얼마나 잘
배치됐는지 보기 위해서입니다.

### Q5. 왜 no-answer를 Hit@k에서 제외했나요?

검색기는 항상 후보를 반환할 수 있지만 답의 존재 여부는 별도 판단 문제입니다.
no-answer는 최종 답변이 안전하게 거절하는지 평가할 때 사용합니다.

### Q6. 왜 BM25가 Chroma보다 좋았나요?

현재 질문에는 날짜, 이메일, 숫자, 사업명처럼 정확한 단어가 많았습니다. 작은
공고문 데이터의 결과이며 다른 도메인까지 일반화하지 않습니다.

### Q7. RRF가 BM25보다 Hit@1이 낮은데 왜 유지하나요?

현재는 Hybrid 후보를 reranker에 넘기는 v0 기준선을 보존하기 위해 유지합니다.
하지만 이 결과만으로 RRF가 필수라고 결론 내리지는 않습니다. dev에서
`BM25 Top-7→BGE`와 `RRF Top-7→BGE`를 비교해, 품질 차이가 없으면 더 단순하고
빠른 BM25 경로를 기본값으로 선택합니다.

### Q8. reranker를 바로 운영에 쓰나요?

품질 우선 포트폴리오 설정으로 로컬 BGE를 사용합니다. CPU 평균 지연은 dev 약
4.20초, test 약 4.32초라 실시간 운영에는 느립니다. MiniLM은 빨랐지만 품질이
하락했습니다. 외부 reranker API는 생략했고 운영 최적화는 별도
요구사항이 생겼을 때 새 dev/holdout 질문으로 진행합니다.

### Q9. test는 왜 한 번만 실행했나요?

dev로 설정을 결정한 뒤 고정된 설정으로 한 번 실행했습니다. 이제 test 결과를 보고
후보 수나 모델을 다시 바꾸지 않습니다.

### Q10. 다음 단계는 무엇인가요?

현재 v0 검색 설정의 test 확인과 LangGraph 연결까지 끝났습니다. 다음은 dev에서
`BM25→BGE`와 `RRF→BGE`를 비교해 각 구성요소의 필요성을 확인합니다. 그 결정을
기록한 뒤 normal 답변의 관련성·근거 충실성과 no-answer 거절을 Ragas와 사람
검토로 평가합니다.

---

## 24. 공식 참고 자료

- [Stanford IR Book: ranked retrieval evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [scikit-learn nDCG 설명](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [Ragas Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Ragas 사용 가능한 평가 metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

---

## 25. 코드 위치

```text
data/golden_set.jsonl
  └── 질문, 정답 문장, 문서, dev/test

src/evaluate.py
  ├── 골든셋 검증
  ├── 정답 Chunk 매핑
  ├── Hit@k, MRR, nDCG, latency
  └── JSON·Markdown용 결과

src/run_retrieval_evaluation.py
  ├── 문서와 Chunk 준비
  ├── 네 검색기 구성
  ├── dev/test 실행
  └── experiments 리포트 저장

tests/test_evaluate.py
tests/test_retrieval_evaluation.py
```

테스트:

```powershell
.venv\Scripts\python.exe tests\test_evaluate.py
.venv\Scripts\python.exe tests\test_retrieval_evaluation.py
```
