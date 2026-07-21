"""포트폴리오 데모에서 업로드 문서를 한 번에 검색 가능한 corpus로 준비한다."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from chunker import ChunkingConfig, DocumentChunk, chunk_documents
from document_ingestion import (
    ExtractedDocument,
    ExtractionConfig,
    extract_document,
)


@dataclass(frozen=True)
class PreparedCorpus:
    """업로드부터 Chunk까지 끝난 한 번의 검색 corpus."""

    documents: tuple[ExtractedDocument, ...]
    chunks: tuple[DocumentChunk, ...]
    signature: str


def prepare_uploaded_corpus(
    files: list[tuple[str, bytes]],
    *,
    extraction_config: ExtractionConfig | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> PreparedCorpus:
    """파일 바이트를 추출하고 고정한 포트폴리오 Chunk 설정으로 나눈다."""

    if not files:
        raise ValueError("준비할 업로드 문서가 없습니다.")

    documents = tuple(
        extract_document(
            filename,
            content,
            config=extraction_config or ExtractionConfig(),
        )
        for filename, content in files
    )
    chunks = tuple(
        chunk_documents(
            list(documents),
            config=chunking_config or ChunkingConfig(),
            deduplicate=True,
        )
    )
    if not chunks:
        raise ValueError("문서에서 검색할 수 있는 글자를 찾지 못했습니다.")

    signature_source = "|".join(document.source_sha256 for document in documents)
    signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()[:16]
    return PreparedCorpus(
        documents=documents,
        chunks=chunks,
        signature=signature,
    )


__all__ = ["PreparedCorpus", "prepare_uploaded_corpus"]
