# 작은 로컬 Reranker 비교

> 후보 수는 7개로 고정하고, 큰 심사위원과 작은 심사위원이 같은 한국어 질문을
> 얼마나 잘 정렬하는지 비교했습니다. 결론은 **BGE를 기본 모델로 유지**하는 것입니다.

## 12살도 이해할 수 있는 결론

두 명의 심사위원에게 같은 답안 7개를 보여줬습니다.

- BGE는 천천히 읽지만 정답을 더 잘 찾았습니다.
- MiniLM은 약 9.8배 빨랐지만 정답 순서를 더 자주 틀렸습니다.

지금 프로젝트는 정부 공고문의 숫자·이메일·지원 조건을 정확히 찾는 것이
중요합니다. 그래서 속도보다 품질이 좋은 BGE를 기본 심사위원으로 남겼습니다.
MiniLM은 빠른 미리보기처럼 속도가 더 중요한 환경에서 다시 검토할 수 있습니다.

## 무엇만 바꿨나요?

| 그대로 둔 것 | 값 |
|---|---|
| 문서와 Chunk | 공고문 TXT 3개, paragraph 700/120, 총 38개 |
| 질문 | dev normal 20개 |
| 검색 | Kiwi BM25 + E5/Chroma + RRF |
| Reranker 입력 | RRF 후보 7개 |
| 추론 설정 | CPU, batch 2, max length 512 |

모델만 바꿨습니다.

| 모델 | 파라미터 | 한국어 조건 | 라이선스 | 원격 코드 |
|---|---:|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | 다국어 | Apache-2.0 | 사용 안 함 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | 학습 15개 언어, 한국어 zero-shot 평가 | Apache-2.0 | 사용 안 함 |

두 모델은 실험 중 내용이 바뀌지 않도록 Hugging Face revision SHA를 코드에
고정했습니다.

## 실제 결과

| 모델 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | 평균 | p95 | 최대 프로세스 RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BGE** | **0.850** | **1.000** | **1.000** | **0.925** | **0.913** | 4513.5ms | 4743.5ms | 3041.2MB |
| MiniLM | 0.700 | 0.850 | 0.950 | 0.800 | 0.806 | **460.6ms** | **507.0ms** | **1969.8MB** |

MiniLM은 다음과 같았습니다.

- 파라미터 약 79.3% 감소
- 평균 지연 약 89.8% 감소, 약 9.8배 빠름
- 최대 프로세스 RSS 약 35.2% 감소
- Hit@1은 0.15, MRR은 0.125 감소
- `q028`은 정답을 최종 5개 안에 넣지 못함

## 어떤 질문에서 달랐나요?

| 질문 | BGE 정답 순위 | MiniLM 정답 순위 | 관찰 |
|---|---:|---:|---|
| q007 신청서는 어디로 보내면 되나요? | 1 | 4 | 이메일·제출처 구분이 약해짐 |
| q014 지원을 몇 년 동안 받을 수 있나요? | 2 | 1 | MiniLM이 한 건 개선 |
| q028 관련 기업도 운영기관으로 신청할 수 있나요? | 1 | 5위 밖 | 지원 조건과 주체 구분 실패 |
| q030 훈련로드맵은 몇 개 기업 대상인가요? | 1 | 4 | 숫자 조건 순위가 낮아짐 |
| q031 훈련과정 개발 목표는 몇 개소인가요? | 1 | 2 | 숫자 질문 순위가 낮아짐 |

작은 모델이 모든 질문에서 나쁜 것은 아닙니다. 하지만 현재 문서에 자주 나오는
숫자·이메일·신청 조건에서 품질 하락이 확인돼 기본값으로 바꾸지 않았습니다.

## 검토했지만 실험하지 않은 모델

| 모델 | 검토 결과 |
|---|---|
| `Alibaba-NLP/gte-multilingual-reranker-base` | 306M·75개 언어·Apache-2.0이라 유력했지만, 별도 원격 코드를 요구하고 현재 Python 3.14 / Transformers 5 환경의 첫 한국어 추론에서 위치 인덱스 오류가 발생해 제외 |
| `jinaai/jina-reranker-v2-base-multilingual` | 다국어 278M이지만 로컬 사용 라이선스가 CC-BY-NC-4.0이라 상업 포트폴리오의 기본 후보에서 제외 |

실행이 안 된 모델을 성능이 나쁘다고 평가한 것은 아닙니다. 현재 환경 호환성 또는
라이선스 기준 때문에 실제 비교 대상에서 제외했습니다.

## 결정

```text
기본 모델: BAAI/bge-reranker-v2-m3
후보 수: 7개
MiniLM: 속도 우선 선택지로만 기록
test split: 아직 실행하지 않음
```

외부 reranker API는 포트폴리오 범위에서 생략합니다. BGE 설정을 잠그고 test를
한 번만 실행합니다.

## 결과를 그대로 확인하는 파일

- [`reranker-model-bge-dev.json`](./reranker-model-bge-dev.json)
- [`reranker-model-minilm-dev.json`](./reranker-model-minilm-dev.json)

## 해석할 때 주의할 점

- 문서 3개와 dev 질문 20개의 작은 CPU 실험이므로 다른 도메인에 일반화하지 않습니다.
- 지연 시간과 메모리는 현재 컴퓨터에서 한 번 측정한 참고값입니다.
- 최대 RSS는 reranker만이 아니라 embedding·BM25·Chroma를 포함한 전체 평가
  프로세스의 최고 메모리입니다.
- 모델 로드 시간은 이미 다운로드된 캐시를 사용했으며 최초 다운로드 시간은
  포함하지 않습니다.
- 설정을 고르는 중이므로 잠가 둔 test 질문은 보지 않았습니다.

## 공식 모델 자료

- [BGE reranker v2 m3 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [MiniLM 다국어 CrossEncoder 모델 카드](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [Alibaba GTE 다국어 reranker 모델 카드](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
- [Jina 다국어 reranker v2 모델 카드](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
