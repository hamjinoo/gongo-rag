# 6번 작업: RRF 후보를 질문에 더 잘 맞는 순서로 다시 세우기

이 문서는 두 가지 목적으로 작성했습니다.

1. 처음 보는 사람도 **reranker가 왜 필요한지** 이해하기
2. 면접에서 모델 선택, 속도, 점수, 한계를 기술적으로 설명하기

---

## 0. 한눈에 보기

### 이번에 만든 것

```text
질문
  ↓
BM25 + Chroma
  ↓
RRF 후보 10개
  ↓
CrossEncoder가 질문과 후보 본문을 한 쌍씩 같이 읽음
  ↓
관련성 점수가 높은 순서로 다시 정렬
  ↓
최종 5개 + 원본 위치 metadata
```

### 왜 만들었나요?

RRF는 BM25와 Chroma가 만든 **순위표만** 봅니다.

본문을 직접 읽지 않기 때문에 이런 실수가 생길 수 있습니다.

- 질문과 같은 단어가 있지만 답은 아닌 Chunk가 위에 올라옴
- 의미가 비슷하지만 질문의 조건과 다른 Chunk가 위에 올라옴
- 두 검색기에서 모두 중간 순위인 진짜 답이 밀림

reranker는 질문과 본문을 같이 읽고 “이 본문이 이 질문에 얼마나 관련 있는가?”를
다시 판단합니다.

### 이번 단계에서 얻어야 하는 것

- embedding 모델과 CrossEncoder의 차이
- 검색과 재정렬을 두 단계로 나누는 이유
- 후보 수가 정확도와 속도에 어떤 영향을 주는지
- 모델 점수를 정답 확률로 보면 안 되는 이유
- 실제 결과가 좋아졌는지 고정 질문으로 평가해야 하는 이유

---

## 1. 12살도 이해할 수 있는 설명

### 도서관 비유

“어떤 회사가 지원받을 수 있나요?”라는 질문에 답할 책장을 찾는다고 생각해 봅시다.

#### BM25

책 제목과 본문에서 `회사`, `지원`, `대상` 같은 단어를 빨리 찾습니다.

#### Chroma

`어떤 회사`와 `참여기업 자격`처럼 표현이 달라도 뜻이 비슷한 글을 찾습니다.

#### RRF

BM25 추천 목록과 Chroma 추천 목록을 합쳐 후보를 만듭니다.

#### Reranker

후보 내용을 질문과 함께 직접 읽고 마지막 순서를 정합니다.

```text
BM25: 단어 탐정
Chroma: 뜻 탐정
RRF: 두 탐정의 추천 목록 합치기
Reranker: 후보 내용을 읽는 최종 심사위원
```

---

## 2. RRF만으로 끝내면 안 되나요?

끝낼 수도 있습니다. 단순한 서비스나 속도가 아주 중요한 환경에서는 RRF 결과를
바로 사용할 수 있습니다.

하지만 RRF 공식에는 본문이 들어가지 않습니다.

```text
RRF 점수 = Σ 1 / (상수 + 각 검색기의 순위)
```

RRF는 다음 정보만 압니다.

- BM25에서 몇 위인가?
- Chroma에서 몇 위인가?

RRF는 다음 내용은 직접 판단하지 않습니다.

- 이 문장이 질문의 답인가?
- 지원 대상인지 제외 대상인지?
- 질문의 조건을 모두 만족하는가?

CrossEncoder는 다음처럼 질문과 후보 본문을 동시에 입력받습니다.

```text
[질문, 후보 본문 1] → 관련성 점수
[질문, 후보 본문 2] → 관련성 점수
[질문, 후보 본문 3] → 관련성 점수
```

그 점수가 높은 순서로 후보를 다시 세웁니다.

---

## 3. 왜 모든 Chunk를 CrossEncoder로 읽지 않나요?

CrossEncoder는 정확한 대신 느립니다.

embedding은 문서를 미리 숫자로 바꿔 둘 수 있지만, CrossEncoder는 질문이 올 때마다
질문과 모든 본문의 조합을 다시 계산해야 합니다.

문서가 10,000개라면:

```text
모든 문서를 CrossEncoder로 읽기
→ 질문-본문 계산 10,000번

RRF 후보 10개만 CrossEncoder로 읽기
→ 질문-본문 계산 10번
```

그래서 역할을 나눕니다.

| 단계 | 역할 | 특징 |
|---|---|---|
| BM25·Chroma | 넓게 후보 찾기 | 빠름 |
| RRF | 두 후보 목록 합치기 | 매우 가벼움 |
| CrossEncoder | 소수 후보를 자세히 읽기 | 더 정확하지만 느림 |

이 구조를 흔히 **retrieve then rerank**, 즉 “먼저 찾고 나중에 정밀 심사하기”라고
설명합니다.

---

## 4. 현재 프로젝트에서 실제로 하는 일

### 4-1. RRF 후보 요청

기본 설정에서는 RRF 상위 10개를 요청합니다.

```python
candidates = hybrid_retriever.search(query, k=10)
```

### 4-2. 질문-본문 쌍 만들기

```python
pairs = [
    (query, candidate.chunk.text)
    for candidate in candidates
]
```

### 4-3. CrossEncoder 점수 계산

```python
scores = model.predict(pairs)
```

### 4-4. 점수가 높은 순서로 정렬

```python
점수가 높은 후보
→ 낮은 후보
```

점수가 같으면 기존 RRF 순위가 높은 후보를 먼저 둡니다.

### 4-5. 검색 근거 보존

재정렬 뒤에도 다음 정보를 버리지 않습니다.

- 최종 CrossEncoder 순위와 점수
- 이전 RRF 순위와 점수
- BM25 순위와 원점수
- Chroma 순위와 similarity
- 파일명, 페이지, 추출 방식, 문자 위치
- 원본 Chunk ID와 본문

따라서 “왜 이 결과가 올라왔나요?”라는 질문에 이전 검색 근거까지 보여줄 수 있습니다.

---

## 5. 화면에서 무엇을 보면 되나요?

앱을 실행합니다.

```powershell
Set-Location "C:\Users\mae\Desktop\260704\publish-worktrees\gongo-rag"
.venv\Scripts\Activate.ps1
streamlit run app.py
```

브라우저에서 다음 순서로 확인합니다.

1. PDF 또는 DOCX 업로드
2. `텍스트 추출`
3. `Chunk 만들기`
4. BM25와 Chroma 결과 확인
5. RRF 결과 확인
6. `재정렬 질문` 입력
7. `CrossEncoder 재정렬`

각 결과에서 다음 값을 비교합니다.

```text
최종 CrossEncoder 순위
이전 RRF 순위
순위 변화
CrossEncoder 점수
BM25와 Chroma의 원래 순위
원본 파일과 페이지
```

### 좋은 관찰 기록

```text
질문:
내가 생각한 정답 Chunk:
RRF 순위:
reranker 순위:
좋아졌나, 나빠졌나:
그 이유로 추측되는 문장:
다음에 확인할 것:
```

코드를 외우는 것보다 이 기록을 자기 말로 설명하는 것이 더 중요합니다.

---

## 6. 실제 한국어 PDF 결과

사용한 문서:

```text
붙임 1. 2026년 글로벌화 사전검증 지원사업 참여기업 모집 공고.pdf
5페이지
11 chunks
```

### 성공 사례

질문:

```text
지원대상 대전시 소재 기업
```

정답으로 본 Chunk:

```text
지원대상
글로벌 시장 진출을 희망하는 대전시 소재 기업
```

결과:

| 단계 | 정답 Chunk 순위 |
|---|---:|
| RRF | 1위 |
| CrossEncoder | 1위 |

CrossEncoder 점수는 `0.998472`였습니다.

정확한 핵심어가 들어간 질문에서는 RRF의 올바른 결과를 그대로 유지했습니다.

### 일부 개선됐지만 완벽하지 않은 사례

질문:

```text
어떤 회사가 지원받을 수 있나요?
```

결과:

| 후보 | RRF 순위 | CrossEncoder 순위 |
|---|---:|---:|
| 실제 지원 대상 Chunk | 4위 | 2위 |
| 참여 제한사항 Chunk | 6위 | 1위 |

실제 지원 대상 Chunk는 4위에서 2위로 올라왔습니다. 하지만 “참여 제한사항”도 회사가
지원할 수 있는 조건과 관련된 글이어서 1위가 됐습니다.

### 이 실패에서 배운 것

1. reranker는 관련성을 개선했지만 정답을 항상 1위로 만들지는 않습니다.
2. `지원 대상`과 `참여 제한`은 의미상 가까워 모델이 둘 다 높게 볼 수 있습니다.
3. 한두 질문만 보고 모델이 좋아졌다고 결론 내리면 안 됩니다.
4. 다음 단계에서 고정 질문과 정답 Chunk를 만들어 Hit@1, MRR, nDCG를 비교해야 합니다.
5. 최종 답변 단계에서는 여러 근거를 함께 읽고 대상과 제외 조건을 구분해야 합니다.

이 사례는 실패를 숨긴 것이 아니라, 다음 평가 작업이 왜 필요한지 보여주는 근거입니다.

---

## 7. 실제 실행 시간

현재 개발 PC의 CPU에서 측정한 참고값입니다.

| 작업 | 시간 |
|---|---:|
| 캐시된 모델을 새 Python 프로세스에 로드 | 약 10.6초 |
| 질문-본문 10쌍 점수 계산, batch size 2 | 약 4.3초 |
| 모델 첫 다운로드를 포함한 최초 통합 확인 | 약 304초 |

이 값은 CPU, 네트워크, 캐시, batch size, 입력 길이에 따라 달라집니다.
포트폴리오에서는 절대 성능 수치가 아니라 다음 트레이드오프를 설명할 때 사용합니다.

```text
후보 수 증가
→ 정답을 포함할 가능성 증가
→ CrossEncoder 계산량과 지연 증가
```

---

## 8. 이번 구현 완료 기준

- [x] RRF 후보만 재정렬
- [x] 한국어를 포함한 다국어 CrossEncoder 사용
- [x] 외부 API 키 없이 CPU에서 실행
- [x] 질문과 후보 본문을 쌍으로 batch 처리
- [x] 최종 순위와 이전 RRF 순위 함께 표시
- [x] BM25·Chroma 근거와 metadata 보존
- [x] 결과 JSON 다운로드
- [x] 모델과 Streamlit resource 캐시
- [x] 중복 Chunk ID 거부
- [x] 점수 개수 불일치, NaN, 무한대, 문자열 점수 거부
- [x] 작은 문서의 UI 경계 조건 처리
- [x] 실제 한국어 PDF 성공·실패 확인
- [ ] 고정 질문셋으로 RRF 전후 품질 평가
- [ ] Cohere Rerank와 품질·비용·속도 비교
- [ ] 최종 LLM 답변과 인용에 연결

---

## 9. 다음 작업

다음은 기능을 더 붙이는 것보다 **현재 기능이 실제로 좋아졌는지 측정하는 작업**입니다.

```text
같은 문서
같은 Chunk
같은 질문
같은 정답 Chunk

BM25
vs Chroma
vs RRF
vs CrossEncoder
```

최소한 다음을 기록합니다.

- Hit@1, Hit@3, Hit@5
- MRR 또는 정답의 평균 역순위
- 후보 Hit@10
- 질문당 검색·재정렬 시간
- 정답 없음 질문의 오탐
- 실패 원인: 추출, Chunk, 검색, 재정렬 중 어디인가?

고정 질문셋은 모델을 괴롭히기 위한 시험지가 아니라, 변경 전후를 같은 조건으로
비교하는 자입니다.

---

# 면접 대비 기술 설명

## 10. Bi-encoder와 CrossEncoder의 차이

### Bi-encoder

질문과 문서를 따로 embedding합니다.

```text
질문 → 질문 벡터
문서 → 문서 벡터
두 벡터의 거리 비교
```

문서 벡터를 미리 저장할 수 있어 대규모 검색에 적합합니다. 현재 프로젝트에서는
multilingual E5와 Chroma가 이 역할을 합니다.

### CrossEncoder

질문과 문서를 한 입력으로 넣습니다.

```text
[질문, 문서] → Transformer → 관련성 점수
```

질문 단어와 문서 단어가 Transformer 내부에서 직접 상호작용하므로 일반적으로 더
정밀하지만, 문서 점수를 미리 계산할 수 없습니다.

그래서 Bi-encoder는 후보 생성, CrossEncoder는 상위 후보 재정렬에 사용합니다.

---

## 11. 모델 선택 근거

기본 모델:

```text
BAAI/bge-reranker-v2-m3
```

선택 이유:

- 한국어를 포함하는 multilingual 모델
- 질문과 passage를 입력받는 reranker
- Apache-2.0 라이선스
- 로컬 실행 가능
- Sentence Transformers `CrossEncoder`로 연결 가능
- 외부 서비스 API 키와 문서 전송이 필요 없음

현재는 CPU 재현성을 우선해 `device="cpu"`를 사용합니다.

모델 로딩에는 다음 안전 설정을 적용합니다.

```python
CrossEncoder(
    model_name,
    device="cpu",
    max_length=512,
    trust_remote_code=False,
)
```

`trust_remote_code=False`로 Hub 저장소의 사용자 정의 코드를 실행하지 않습니다.

---

## 12. 점수는 확률인가요?

그렇게 단정하면 안 됩니다.

현재 CrossEncoder는 한 후보당 하나의 관련성 점수를 반환하고, 모델 설정에 따라
sigmoid가 적용된 값이 보일 수 있습니다. 하지만 `0.8`을 “정답일 확률 80%”라고
해석할 근거는 없습니다.

안전한 사용법:

```text
같은 질문
같은 모델
같은 후보 집합 안에서
점수가 더 높은 후보를 위에 둔다
```

절대 임계값으로 정답 여부를 판단하려면 별도 검증 데이터로 calibration 해야 합니다.

---

## 13. Candidate k가 중요한 이유

reranker는 RRF가 전달하지 않은 Chunk를 되살릴 수 없습니다.

```text
정답이 RRF 30위
reranker 후보는 10개
→ 정답을 읽어보지도 못함
```

이를 **candidate recall bottleneck**이라고 설명할 수 있습니다.

후보 수를 늘리면:

- 정답 포함 가능성이 커짐
- 계산 시간과 메모리 사용량도 커짐

따라서 먼저 `Hit@candidate_k`를 보고, 정답이 충분히 포함되는 가장 작은 후보 수를
선택해야 합니다.

---

## 14. 입력 길이와 잘림

현재 최대 입력 길이는 512 tokens입니다.

```text
질문 tokens + 본문 tokens + 특수 tokens ≤ 512
```

넘는 부분은 모델 입력에서 잘립니다.

현재 Chunk 기준선은 약 700자이므로 대부분 처리할 수 있지만 다음 경우는 확인해야 합니다.

- 영어와 숫자가 많아 token 수가 늘어난 문서
- 표를 한 줄로 길게 추출한 문서
- OCR 때문에 글자가 잘게 분리된 문서
- 중요한 답이 Chunk 맨 뒤에 있는 경우

Chunk 크기를 크게 바꾸면 reranker의 truncation도 함께 점검해야 합니다.

---

## 15. 재현성과 안전 처리

### 동점

CrossEncoder 점수가 같으면 다음 순서로 정렬합니다.

1. 기존 RRF 순위
2. 문서의 Chunk 순서
3. Chunk ID

### 중복 ID

동일한 Chunk ID가 두 번 오면 오류로 처리합니다. 같은 문서를 두 번 점수화해 결과가
왜곡되는 일을 막기 위해서입니다.

### 비정상 점수

다음 결과는 오류로 처리합니다.

- 후보 수와 점수 수가 다름
- NaN
- 양수 또는 음수 무한대
- 숫자로 바꿀 수 없는 문자열

### metadata

새 객체로 본문만 복사하지 않고 기존 `HybridSearchResult` 전체를 보존합니다.
따라서 재정렬 뒤에도 파일명과 페이지 인용을 잃지 않습니다.

---

## 16. 캐시와 운영 환경

두 종류의 캐시가 있습니다.

### Hugging Face 로컬 캐시

모델 파일을 디스크에 저장해 다음 실행에서 다시 다운로드하지 않습니다.

### 프로세스와 Streamlit resource 캐시

같은 모델 설정이면 메모리에 올라온 CrossEncoder 객체를 재사용합니다.

운영 환경에서는 추가로 고려해야 합니다.

- 모델 이미지 또는 볼륨에 사전 다운로드
- GPU 사용 여부
- ONNX 또는 OpenVINO 최적화
- batch 크기
- 동시 요청 queue
- timeout과 fallback
- 모델 버전 또는 revision 고정
- 관찰 가능한 latency와 오류 metric

---

## 17. 시간 복잡도

RRF 후보 수를 `N`, Transformer 한 번의 계산 비용을 `T`라고 단순화하면:

```text
재정렬 비용 ≈ O(N × T)
```

실제로 batch 처리를 사용하므로 벽시계 시간은 단순히 N배와 같지는 않지만, 후보 수가
늘수록 계산량이 거의 선형으로 증가합니다.

RRF 자체는 가볍지만 CrossEncoder가 전체 검색 latency의 큰 부분을 차지할 수 있습니다.

---

## 18. 로컬 CrossEncoder와 Cohere Rerank 비교

현재는 로컬 CrossEncoder만 구현했습니다.

| 항목 | 로컬 CrossEncoder | Cohere Rerank |
|---|---|---|
| API 키 | 불필요 | 필요 |
| 문서 외부 전송 | 없음 | API 정책 확인 필요 |
| 초기 준비 | 모델 다운로드·메모리 필요 | SDK와 API 설정 |
| 비용 | 장비 비용 | 요청량 기반 비용 |
| 운영 | 직접 최적화 | 관리형 서비스 |
| 모델 변경 통제 | 직접 고정 가능 | 제공 모델 정책에 따름 |

Cohere를 추가할 때도 `PairScorer`와 같은 인터페이스를 유지하면 검색 파이프라인을
바꾸지 않고 같은 질문셋으로 두 provider를 비교할 수 있습니다.

비교 없이 “API 모델이 더 좋다” 또는 “로컬이 더 좋다”고 결론 내리지 않습니다.

---

## 19. LangChain과 LangGraph에서는 어디에 있나요?

현재 Chunk는 LangChain `Document`로 변환되어 Chroma에 저장됩니다.

reranker는 검색 품질 실험이 라이브러리에 묶이지 않도록 작은 Python 인터페이스로
분리했습니다.

이후 LangGraph에서는 다음 노드로 사용할 수 있습니다.

```text
retrieve node
→ RRF candidates
→ rerank node
→ relevance check
→ 충분하면 answer
→ 부족하면 query rewrite 또는 정보 없음
```

즉 LangGraph는 검색 알고리즘을 대신하는 도구가 아니라, 검색·재검색·거절 흐름을
명확한 상태 전이로 연결하는 역할입니다.

---

## 20. 30초 면접 설명

> BM25와 multilingual E5/Chroma 결과를 RRF로 합친 뒤, 상위 10개 후보만
> `BAAI/bge-reranker-v2-m3` CrossEncoder에 질문-본문 쌍으로 입력해 재정렬했습니다.
> CrossEncoder는 bi-encoder보다 정밀하지만 후보마다 추론해야 하므로 전체 문서가 아닌
> RRF 상위 후보에만 사용했습니다. 재정렬 후에도 RRF, BM25, Chroma 순위와 원본
> metadata를 보존해 결과를 설명할 수 있게 했습니다. 실제 한국어 PDF에서 지원 대상
> Chunk가 자연어 질문 기준 RRF 4위에서 2위로 개선됐지만 제외 조건 Chunk가 1위가 된
> 실패도 확인했습니다. 그래서 다음 단계는 고정 질문셋으로 Hit@k, MRR, nDCG와
> latency를 함께 비교하는 것입니다.

---

## 21. 자주 나오는 면접 질문

### Q1. 왜 embedding 검색만 쓰지 않았나요?

embedding은 질문과 문서를 따로 압축하므로 빠르지만, 질문과 본문의 세부 단어 관계를
직접 비교하는 정보가 줄어듭니다. CrossEncoder는 둘을 같이 읽어 상위 후보를 더
정밀하게 비교합니다.

### Q2. 왜 CrossEncoder로 처음부터 검색하지 않았나요?

모든 문서와 질문의 쌍을 매 요청마다 계산해야 해 느립니다. BM25와 Chroma로 후보를
줄인 뒤 적용했습니다.

### Q3. RRF와 reranker의 차이는 무엇인가요?

RRF는 여러 검색기의 순위를 합칩니다. reranker는 질문과 각 후보 본문을 직접 읽고
새 관련성 점수를 계산합니다.

### Q4. 후보 수는 어떻게 정하나요?

고정 평가셋의 후보 Hit@k와 latency를 같이 측정해, 정답을 충분히 포함하면서 가장
작은 값을 선택합니다.

### Q5. 점수 0.9면 정답 확률 90%인가요?

아닙니다. calibration하지 않은 관련성 점수입니다. 현재는 같은 질문의 후보 순서를
정하는 데만 사용합니다.

### Q6. 왜 이 모델을 선택했나요?

한국어를 포함한 다국어 지원, reranking 목적, 로컬 실행, Apache-2.0 라이선스,
Sentence Transformers 호환성을 기준으로 선택했습니다.

### Q7. 모델이 정답을 1위로 못 올리면 실패인가요?

개별 사례만으로 판단하지 않습니다. 고정 질문 전체에서 순위 metric이 좋아지는지,
특정 질문 유형에서 나빠지는지 확인합니다.

### Q8. 속도를 줄이려면 무엇을 하나요?

후보 수와 max length를 조정하고, batch 처리, GPU, ONNX/OpenVINO, 더 작은 모델,
관리형 API를 같은 평가셋으로 비교할 수 있습니다.

### Q9. 정답이 후보에 없으면 어떻게 하나요?

reranker로 해결할 수 없습니다. BM25·embedding·Chunk 설정을 개선하거나 query
rewrite로 다시 검색해야 합니다.

### Q10. metadata를 왜 보존했나요?

최종 답변에서 파일명과 페이지를 인용하고, 어떤 검색 단계 때문에 결과가 선택됐는지
추적하기 위해서입니다.

### Q11. 한국어에서 특히 확인할 것은 무엇인가요?

조사와 어미, 띄어쓰기, 한영 혼합, 표, OCR 오류, 공고문의 긴 조건 문장을 실제
질문으로 확인해야 합니다.

### Q12. 다음 단계는 무엇인가요?

고정 질문과 정답 Chunk로 BM25, Chroma, RRF, reranker의 품질과 latency를 비교한 뒤,
LangGraph 답변·재검색·거절 흐름에 연결합니다.

---

## 22. 공식 참고 자료

- [Sentence Transformers CrossEncoder 문서](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
- [Sentence Transformers 빠른 시작: CrossEncoder](https://www.sbert.net/docs/quickstart.html#cross-encoder)
- [BAAI bge-reranker-v2-m3 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3)

---

## 23. 개발자용 테스트

reranker 핵심 테스트:

```powershell
.venv\Scripts\python.exe tests\test_reranker.py
```

Streamlit 전체 흐름 테스트:

```powershell
.venv\Scripts\python.exe tests\test_document_upload_ui.py
```

전체 문법 검사:

```powershell
.venv\Scripts\python.exe -m compileall -q app.py src tests
```

현재 검증 범위에는 재정렬 순서, 후보 창, 동점, metadata, JSON, 비정상 점수,
중복 ID, UI 흐름과 실제 한국어 PDF가 포함됩니다.
