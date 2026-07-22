"""BM25 기준선과 metadata가 보존된 한국어 chunk 검색기."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Protocol

from chunker import DocumentChunk


TokenizerName = Literal["simple", "kiwi"]
CONTENT_POS_TAGS = {
    "NR",
    "NP",
    "MM",
    "MAG",
    "MAJ",
    "XR",
    "SL",
    "SN",
    "SH",
}
CONTENT_POS_PREFIXES = ("NN", "VV", "VA", "VX")


class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[str]:
        """검색에 사용할 단어 목록을 반환한다."""


class KiwiUnavailableError(RuntimeError):
    """Kiwi 한국어 형태소 분석기를 불러올 수 없음."""


@dataclass(frozen=True)
class TokenizerStatus:
    name: TokenizerName
    ready: bool
    message: str


@dataclass(frozen=True)
class SearchResult:
    """순위·점수·원본 chunk를 함께 가진 BM25 검색 결과."""

    rank: int
    score: float
    chunk: DocumentChunk
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "chunk": self.chunk.to_dict(),
        }


def tokenize(text: str) -> list[str]:
    """기존 손계산 테스트를 위한 공백 분리 tokenizer."""

    return text.lower().split()


def tokenize_simple(text: str) -> list[str]:
    """문장부호를 제거하고 한글·영문·숫자 덩어리를 나누는 기준선."""

    return re.findall(r"[가-힣]+|[a-z]+|\d+(?:[.,]\d+)*", text.lower())


class KiwiTokenizer:
    """조사·어미를 제외하고 검색에 중요한 한국어 형태소만 남긴다."""

    def __init__(self) -> None:
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:
            raise KiwiUnavailableError(
                "kiwipiepy가 없습니다. requirements.txt를 설치해주세요."
            ) from exc

        self._kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in self._kiwi.tokenize(text):
            if (
                token.tag not in CONTENT_POS_TAGS
                and not token.tag.startswith(CONTENT_POS_PREFIXES)
            ):
                continue

            value = getattr(token, "lemma", None) or token.form
            normalized = value.lower().strip()
            if normalized:
                tokens.append(normalized)
        return tokens


def get_tokenizer_status(name: TokenizerName) -> TokenizerStatus:
    if name == "simple":
        return TokenizerStatus("simple", True, "기본 tokenizer 준비 완료")

    try:
        KiwiTokenizer()
    except KiwiUnavailableError as exc:
        return TokenizerStatus("kiwi", False, str(exc))
    except Exception as exc:
        return TokenizerStatus("kiwi", False, f"Kiwi 초기화에 실패했습니다: {exc}")

    return TokenizerStatus("kiwi", True, "Kiwi 한국어 형태소 분석 준비 완료")


@lru_cache(maxsize=2)
def create_tokenizer(name: TokenizerName) -> Tokenizer:
    if name == "simple":
        return tokenize_simple
    if name == "kiwi":
        return KiwiTokenizer()
    raise ValueError("tokenizer는 simple 또는 kiwi여야 합니다.")


class BM25:
    """BM25Okapi 공식을 직접 구현한 키워드 검색 기준선.

    IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    """

    def __init__(
        self,
        corpus: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
        debug: bool = False,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1은 0보다 커야 합니다.")
        if not 0 <= b <= 1:
            raise ValueError("b는 0~1 사이여야 합니다.")

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.debug = debug
        self.doc_tokens = [self.tokenizer(text) for text in corpus]
        self.doc_term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.N = len(self.doc_tokens)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0
        self.df: dict[str, int] = {}

        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        if self.debug:
            print(f"self.doc_tokens: {self.doc_tokens}")
            print(f"self.doc_lens: {self.doc_lens}")
            print(f"self.avgdl: {self.avgdl}")
            print(f"self.df: {self.df}")
            print(f"self.N: {self.N}")

    def idf(self, term: str) -> float:
        document_frequency = self.df.get(term, 0)
        return math.log(
            (self.N - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1
        )

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if not 0 <= doc_idx < self.N:
            raise IndexError("doc_idx가 corpus 범위를 벗어났습니다.")

        document_length = self.doc_lens[doc_idx]
        if self.avgdl:
            length_normalization = self.k1 * (
                1 - self.b + self.b * document_length / self.avgdl
            )
        else:
            length_normalization = self.k1

        frequencies = self.doc_term_frequencies[doc_idx]
        total = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            total += (
                self.idf(token)
                * frequency
                * (self.k1 + 1)
                / (frequency + length_normalization)
            )
        return total

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        if k <= 0 or not self.corpus:
            return []

        query_tokens = self.tokenizer(query)
        if self.debug:
            print(f"{query_tokens=}")

        results: list[tuple[int, float]] = []
        for index in range(self.N):
            score = self.score(query_tokens, index)
            results.append((index, score))
            if self.debug:
                print(f"  문서{index} score={score:.3f}")

        return sorted(results, key=lambda item: item[1], reverse=True)[:k]


class BM25ChunkRetriever:
    """DocumentChunk를 색인하고 출처가 포함된 검색 결과를 반환한다."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        tokenizer_name: TokenizerName = "kiwi",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("검색할 chunk가 없습니다.")

        self.chunks = chunks
        self.tokenizer_name = tokenizer_name
        self.tokenizer = create_tokenizer(tokenizer_name)
        self.index = BM25(
            [chunk.text for chunk in chunks],
            k1=k1,
            b=b,
            tokenizer=self.tokenizer,
        )

    def analyze_query(self, query: str) -> list[str]:
        return self.tokenizer(query)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0 or not query.strip():
            return []

        query_tokens = self.analyze_query(query)
        if not query_tokens:
            return []

        raw_results = self.index.search(query, k=len(self.chunks))
        positive_results = [
            (chunk_index, score)
            for chunk_index, score in raw_results
            if score > 0
        ][:k]

        results: list[SearchResult] = []
        unique_query_terms = list(dict.fromkeys(query_tokens))
        for rank, (chunk_index, score) in enumerate(positive_results, start=1):
            document_terms = set(self.index.doc_tokens[chunk_index])
            matched_terms = tuple(
                term for term in unique_query_terms if term in document_terms
            )
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[chunk_index],
                    matched_terms=matched_terms,
                )
            )
        return results

    def retrieve_texts(self, query: str, k: int = 5) -> list[str]:
        """기존 hit_rate_at_k 평가 함수와 연결할 수 있는 어댑터."""

        return [result.chunk.text for result in self.search(query, k=k)]


def main() -> None:
    corpus = [
        "청년 창업 지원 사업 공고",
        "창업 기업 지원 금액 안내",
        "청년 주택 정책 안내",
    ]
    query = "청년 지원 금액"

    bm25 = BM25(corpus, debug=True)
    print(f"\n질문: {query!r}")
    print("기대 순위: D2 > D1 > D3\n")
    for index, score in bm25.search(query, k=3):
        print(f"D{index + 1} · {score:.3f} · {corpus[index]}")


if __name__ == "__main__":
    main()
