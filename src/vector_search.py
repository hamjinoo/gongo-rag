"""LangChain Document와 Chroma를 사용하는 한국어 의미 검색기."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chunker import DocumentChunk

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DOCUMENT_PROMPT = "passage: "
QUERY_PROMPT = "query: "
EMBEDDING_CONFIG_VERSION = "e5-query-passage-normalized-v1"


class VectorSearchError(RuntimeError):
    """의미 검색을 준비하거나 실행할 수 없음."""


class VectorSearchDependencyError(VectorSearchError):
    """LangChain 또는 Chroma 의존성이 설치되지 않음."""


class EmbeddingModelUnavailableError(VectorSearchError):
    """로컬 embedding 모델을 불러올 수 없음."""


class VectorIndexMismatchError(VectorSearchError):
    """같은 Chroma collection에 다른 chunk 구성이 들어 있음."""


@dataclass(frozen=True)
class VectorSearchResult:
    """Chroma 순위와 원본 chunk를 함께 가진 의미 검색 결과."""

    rank: int
    similarity: float
    distance: float
    chunk: DocumentChunk
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "similarity": self.similarity,
            "distance": self.distance,
            "model_name": self.model_name,
            "chunk": self.chunk.to_dict(),
        }


def chunk_to_langchain_document(chunk: DocumentChunk) -> Document:
    """출처 metadata를 잃지 않고 LangChain Document로 변환한다."""

    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise VectorSearchDependencyError(
            "langchain-core가 없습니다. requirements.txt를 설치해주세요."
        ) from exc

    return Document(
        id=chunk.id,
        page_content=chunk.text,
        metadata={
            **chunk.metadata,
            "chunk_id": chunk.id,
        },
    )


def chunks_to_langchain_documents(
    chunks: list[DocumentChunk],
) -> list[Document]:
    """여러 DocumentChunk를 LangChain 표준 문서로 변환한다."""

    return [chunk_to_langchain_document(chunk) for chunk in chunks]


@lru_cache(maxsize=2)
def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Embeddings:
    """E5 검색 규칙을 적용한 로컬 Hugging Face embedding 모델을 만든다."""

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise VectorSearchDependencyError(
            "langchain-huggingface가 없습니다. requirements.txt를 설치해주세요."
        ) from exc

    try:
        return HuggingFaceEmbeddings(
            model=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "prompt": DOCUMENT_PROMPT,
                "normalize_embeddings": True,
            },
            query_encode_kwargs={
                "prompt": QUERY_PROMPT,
                "normalize_embeddings": True,
            },
            show_progress=False,
        )
    except Exception as exc:
        raise EmbeddingModelUnavailableError(
            f"embedding 모델을 불러오지 못했습니다: {model_name} ({exc})"
        ) from exc


class ChromaChunkRetriever:
    """DocumentChunk를 Chroma에 색인하고 의미가 가까운 chunk를 반환한다."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        embeddings: Embeddings | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("검색할 chunk가 없습니다.")

        chunk_ids = [chunk.id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk id는 중복될 수 없습니다.")

        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise VectorSearchDependencyError(
                "langchain-chroma가 없습니다. requirements.txt를 설치해주세요."
            ) from exc

        self.chunks = chunks
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.model_name = model_name
        self.embeddings = embeddings or create_embedding_model(model_name)
        self.documents = chunks_to_langchain_documents(chunks)
        self.collection_name = collection_name or build_collection_name(
            chunks,
            model_name=model_name,
        )
        self.persist_directory = (
            str(Path(persist_directory).resolve())
            if persist_directory is not None
            else None
        )

        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
            self._ensure_documents_indexed(chunk_ids)
        except VectorIndexMismatchError:
            raise
        except Exception as exc:
            raise VectorSearchError(f"Chroma 색인 생성에 실패했습니다: {exc}") from exc

    @property
    def index_size(self) -> int:
        return len(self.vector_store.get(include=[]).get("ids") or [])

    def search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        if k <= 0 or not query.strip():
            return []

        try:
            raw_results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=min(k, len(self.chunks)),
            )
        except Exception as exc:
            raise VectorSearchError(f"Chroma 의미 검색에 실패했습니다: {exc}") from exc

        results: list[VectorSearchResult] = []
        for rank, (document, raw_similarity) in enumerate(raw_results, start=1):
            chunk_id = str(document.metadata.get("chunk_id") or document.id or "")
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                raise VectorIndexMismatchError(
                    f"검색 결과의 원본 chunk를 찾을 수 없습니다: {chunk_id}"
                )

            similarity = max(-1.0, min(1.0, float(raw_similarity)))
            results.append(
                VectorSearchResult(
                    rank=rank,
                    similarity=similarity,
                    distance=1.0 - similarity,
                    chunk=chunk,
                    model_name=self.model_name,
                )
            )
        return results

    def retrieve_texts(self, query: str, k: int = 5) -> list[str]:
        """기존 Hit@k 평가 함수와 연결할 수 있는 어댑터."""

        return [result.chunk.text for result in self.search(query, k=k)]

    def _ensure_documents_indexed(self, expected_ids: list[str]) -> None:
        existing = self.vector_store.get(include=["documents", "metadatas"])
        existing_ids = list(existing.get("ids") or [])

        if not existing_ids:
            self.vector_store.add_documents(
                documents=self.documents,
                ids=expected_ids,
            )
            return

        existing_documents = list(existing.get("documents") or [])
        existing_metadatas = list(existing.get("metadatas") or [])
        existing_by_id = {
            chunk_id: (document, metadata)
            for chunk_id, document, metadata in zip(
                existing_ids,
                existing_documents,
                existing_metadatas,
                strict=True,
            )
        }
        expected_by_id = {
            chunk.id: (
                chunk.text,
                {
                    **chunk.metadata,
                    "chunk_id": chunk.id,
                },
            )
            for chunk in self.chunks
        }

        if existing_by_id != expected_by_id or set(existing_ids) != set(expected_ids):
            raise VectorIndexMismatchError(
                "같은 Chroma collection에 다른 chunk가 들어 있습니다. "
                "새 collection 이름을 사용해주세요."
            )


def build_collection_name(
    chunks: list[DocumentChunk],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """chunk 내용과 모델이 같으면 같은 안전한 collection 이름을 만든다."""

    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(EMBEDDING_CONFIG_VERSION.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
        digest.update(
            json.dumps(
                chunk.metadata,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
    return f"gongo-{digest.hexdigest()[:20]}"


def get_embedding_model_info() -> dict[str, Any]:
    return {
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "document_prompt": DOCUMENT_PROMPT,
        "query_prompt": QUERY_PROMPT,
        "config_version": EMBEDDING_CONFIG_VERSION,
        "normalized": True,
        "distance_metric": "cosine",
    }
