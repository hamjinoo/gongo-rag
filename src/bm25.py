"""
bm25.py — BM25 키워드 검색 직접 구현  [✍️ Week 3: 이 프로젝트의 첫 번째 산]

먼저 읽기: ../../04-concepts/BM25-완전정복.md  ← 공식과 손계산 예제가 있음. 필수!
자가 채점:  python tests\\test_bm25.py  +  아래 데모의 기대값 비교
GPT 규칙:  TF-IDF/BM25 전체 구현 대행 금지 (WORKFLOW.md). 리뷰는 OK.

공식 (k1=1.5, b=0.75):
    IDF(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    score(D, Q) = Σ_{t∈Q}  IDF(t) * f(t,D) * (k1+1) / ( f(t,D) + k1*(1 - b + b*|D|/avgdl) )

    N=문서 수, df(t)=t가 등장하는 문서 수, f(t,D)=D 안의 t 등장 횟수,
    |D|=D의 토큰 수, avgdl=평균 토큰 수
"""
import math


def tokenize(text: str) -> list[str]:
    """v0 tokenizer: 소문자화 + 공백 분리.  [✅ 제공 — 단순함이 의도]

    Week 6 실험: 아래 kiwi 버전으로 교체했을 때 hit@k가 얼마나 오르는지 비교.
    (한국어 조사 문제는 BM25-완전정복.md '한국어 함정' 참고)
    """
    return text.lower().split()


# --- Week 6에 주석 해제하고 실험 (설치: pip install kiwipiepy) -----------------
# from kiwipiepy import Kiwi
# _kiwi = Kiwi()
#
# def tokenize_kiwi(text: str) -> list[str]:
#     """형태소 분석 tokenizer. 어떤 품사를 남길지 자체가 실험 변수다.
#     시작점: 명사(NN*), 외국어(SL), 숫자(SN)만 남기기. 동사(VV)를 넣으면? 직접 비교!
#     """
#     tokens = _kiwi.tokenize(text)
#     keep = ("NN", "SL", "SN")   # ← 이 목록을 바꿔가며 실험
#     return [t.form.lower() for t in tokens if t.tag.startswith(keep)]
# ------------------------------------------------------------------------------


class BM25:
    """사용법:
        bm25 = BM25([chunk["text"] for chunk in chunks])   # 색인 (1회, 비쌈)
        results = bm25.search("신청 자격", k=3)             # 검색 (여러 번, 쌈)
        # results = [(문서 인덱스, 점수), ...] 점수 내림차순
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75,  
                 tokenizer=tokenize):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.corpus = corpus

        # TODO(직접 구현) — 색인 단계. 단계 힌트:
        #  1. self.doc_tokens: 각 문서를 토큰화한 리스트의 리스트
        #  2. self.doc_lens:   각 문서의 토큰 수 리스트
        #  3. self.avgdl:      토큰 수의 평균
        #  4. self.df:         {단어: 그 단어가 '등장하는 문서 수'} dict
        #     ⚠️ 함정: 한 문서에 10번 나와도 df는 +1.  힌트: set(문서 토큰들)
        #  5. self.N:          문서 수            



        #  1. self.doc_tokens: 각 문서를 토큰화한 리스트의 리스트
        self.doc_tokens = []
        for text in corpus:
            self.doc_tokens.append(self.tokenizer(text))
        print(f"self.doc_tokens: {self.doc_tokens}")
        

        #  2. self.doc_lens:   각 문서의 토큰 수 리스트
        self.doc_lens = []
        for num in self.doc_tokens:
            num_len = len(num)
            self.doc_lens.append(num_len)

        print(f"self.doc_lens: {self.doc_lens}")

        #  3. self.avgdl:      토큰 수의 평균
        # self.avgdl = 0
        # for text in self.doc_tokens:
        #     sum += len(text)
        
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

        print(f"self.avgdl: {self.avgdl}")
            

        #  4. self.df:         {단어: 그 단어가 '등장하는 문서 수'} dict
        #     ⚠️ 함정: 한 문서에 10번 나와도 df는 +1.  힌트: set(문서 토큰들)
        self.df = {}
        for tokens in self.doc_tokens:
            for word in set(tokens):
                self.df[word] = self.df.get(word, 0) + 1
        print(f"self.df: {self.df}")

        #  5. self.N:          문서 수            
        self.N = len(self.doc_tokens)
        print(f"self.N: {self.N}")


    def idf(self, term: str) -> float:
        """단어의 희귀도. 문서에 없는 단어는 df=0으로 계산하면 됨.

        TODO(직접 구현): 위 공식 그대로.  힌트: math.log는 자연로그(ln).
        자가 검증: 희귀한 단어의 idf > 흔한 단어의 idf 여야 한다.
        """
        # IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        return math.log((self.N - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5) + 1)

        

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        """질문 토큰들에 대한 doc_idx번 문서의 BM25 점수.

        TODO(직접 구현) — 단계 힌트:
          1. 이 문서의 토큰 리스트에서 각 질문 토큰의 등장 횟수 f를 센다.
             힌트: 매번 .count()는 느리니 dict로 한 번에 세어두면 좋다 (일단은 count도 OK)
          2. f가 0인 토큰은 건너뛴다 (기여 0).
          3. 공식의 분모에 길이 보정: k1 * (1 - b + b * |D| / avgdl)
          4. 토큰별 기여(idf * f*(k1+1)/(f+보정분모))를 전부 더해 반환.
        """

        total = 0 

        tokens = self.doc_tokens[doc_idx]
        docs_len = self.doc_lens[doc_idx]
        norm = self.k1 * (1 - self.b + self.b * docs_len / self.avgdl)

        for token in query_tokens:
            f = tokens.count(token)
            if f == 0:
                continue
            total += self.idf(token) * f * (self.k1 + 1) / (f + norm)
        return total


    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """전 문서 점수 계산 → 내림차순 상위 k개 (인덱스, 점수) 반환.

        TODO(직접 구현) 힌트:
          query를 토큰화 → 모든 문서에 self.score → 정렬.
          정렬 힌트: sorted(..., key=lambda x: x[1], reverse=True)[:k]
        """
        query_tokens = self.tokenizer(query)
        print(f"{query_tokens=}")

        results = []                          # 밖: 짝 모을 빈 통
        for i in range(self.N):               # 문서 0, 1, 2 ... 돌기
            s = self.score(query_tokens, i)   # 이 문서 점수 (score 재사용!)
            results.append((i, s))            # (문서번호, 점수) 짝을 통에 담기
            print(f"  문서{i} score={s:.3f}")  # 계기판

        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
    
    
# ──────────────────────────────────────────────────────────────
# 완성 배관: 손계산 예제 데모 (BM25-완전정복.md의 코퍼스와 동일)
# 실행:  python src\bm25.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus = [
        "청년 창업 지원 사업 공고",   # D1
        "창업 기업 지원 금액 안내",   # D2
        "청년 주택 정책 안내",       # D3
    ]
    query = "청년 지원 금액"
    print(f"질문: {query!r}")
    print("기대값(문서 참고): D2≈1.41 > D1≈0.91 > D3≈0.50  (k1=1.5, b=0.75)\n")

    try:
        bm25 = BM25(corpus)
        for idx, s in bm25.search(query, k=3):
            print(f"  D{idx+1}  score={s:.3f}   {corpus[idx]}")
        print("\n위 기대값과 ±0.01 안에서 같으면 구현 성공. tests\\test_bm25.py 도 돌려보세요.")
    except NotImplementedError as e:
        print(f"아직 구현 전: {e}")
