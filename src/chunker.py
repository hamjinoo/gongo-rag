"""추출된 문서를 페이지와 출처 정보가 보존된 검색용 chunk로 나눈다."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from document_ingestion import ExtractedDocument


ChunkStrategy = Literal["fixed", "paragraph"]


@dataclass(frozen=True)
class ChunkingConfig:
    """문서를 나누는 방법과 크기 설정."""

    strategy: ChunkStrategy = "paragraph"
    chunk_size: int = 700
    overlap: int = 120

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "paragraph"):
            raise ValueError("strategy는 fixed 또는 paragraph여야 합니다.")
        _validate_split_options(self.chunk_size, self.overlap)


@dataclass(frozen=True)
class DocumentChunk:
    """검색과 인용에 필요한 본문 및 원본 위치 정보."""

    id: str
    text: str
    source_filename: str
    source_sha256: str
    file_type: str
    page_number: int
    page_label: str
    extraction_method: str
    chunk_index: int
    page_chunk_index: int
    start_char: int
    end_char: int
    strategy: ChunkStrategy

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "file_type": self.file_type,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "extraction_method": self.extraction_method,
            "chunk_index": self.chunk_index,
            "page_chunk_index": self.page_chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "strategy": self.strategy,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class _TextSpan:
    text: str
    start: int
    end: int


def chunk_fixed(
    text: str,
    doc_id: str = "doc",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """고정 글자 수로 자르는 가장 단순한 기준선.

    기존 BM25 코드와의 호환을 위해 dict 목록을 반환한다.
    """

    spans = _split_fixed_spans(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "id": f"{doc_id}-{index}",
            "text": span.text,
            "start": span.start,
        }
        for index, span in enumerate(spans)
    ]


def chunk_by_paragraph(
    text: str,
    doc_id: str = "doc",
    max_chars: int = 800,
    overlap: int | None = None,
) -> list[dict]:
    """문단·줄·문장 경계를 우선하여 자르는 방식.

    max_chars 안에서 가능한 한 의미 경계를 찾고, 경계를 찾지 못한 경우에만
    글자 수 기준으로 자른다. 기존 코드와의 호환을 위해 dict 목록을 반환한다.
    """

    effective_overlap = min(100, max_chars // 5) if overlap is None else overlap
    spans = _split_paragraph_spans(
        text,
        chunk_size=max_chars,
        overlap=effective_overlap,
    )
    return [
        {
            "id": f"{doc_id}-{index}",
            "text": span.text,
            "start": span.start,
        }
        for index, span in enumerate(spans)
    ]


def chunk_document(
    document: ExtractedDocument,
    *,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """문서의 페이지 경계를 넘지 않으면서 검색용 chunk를 만든다."""

    settings = config or ChunkingConfig()
    chunks: list[DocumentChunk] = []
    global_index = 0
    document_key = document.source_sha256[:12]

    for page in document.pages:
        if not page.text.strip():
            continue

        if settings.strategy == "fixed":
            spans = _split_fixed_spans(
                page.text,
                chunk_size=settings.chunk_size,
                overlap=settings.overlap,
            )
        else:
            spans = _split_paragraph_spans(
                page.text,
                chunk_size=settings.chunk_size,
                overlap=settings.overlap,
            )

        for page_chunk_index, span in enumerate(spans):
            chunks.append(
                DocumentChunk(
                    id=f"{document_key}-p{page.page_number}-c{page_chunk_index}",
                    text=span.text,
                    source_filename=document.filename,
                    source_sha256=document.source_sha256,
                    file_type=document.file_type,
                    page_number=page.page_number,
                    page_label=page.label,
                    extraction_method=page.method,
                    chunk_index=global_index,
                    page_chunk_index=page_chunk_index,
                    start_char=span.start,
                    end_char=span.end,
                    strategy=settings.strategy,
                )
            )
            global_index += 1

    return chunks


def chunk_documents(
    documents: list[ExtractedDocument],
    *,
    config: ChunkingConfig | None = None,
    deduplicate: bool = True,
) -> list[DocumentChunk]:
    """여러 문서를 나누며 같은 내용의 파일은 기본적으로 한 번만 처리한다."""

    chunks: list[DocumentChunk] = []
    occurrences: dict[str, int] = {}

    for document in documents:
        occurrence = occurrences.get(document.source_sha256, 0)
        if deduplicate and occurrence:
            continue

        document_chunks = chunk_document(document, config=config)
        if occurrence:
            document_chunks = [
                replace(chunk, id=f"{chunk.id}-copy{occurrence}")
                for chunk in document_chunks
            ]

        chunks.extend(document_chunks)
        occurrences[document.source_sha256] = occurrence + 1

    return chunks


def summarize_chunks(chunks: list[DocumentChunk]) -> dict[str, int | float]:
    """UI와 실험 기록에서 공통으로 사용하는 간단한 크기 통계."""

    if not chunks:
        return {
            "count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "average_chars": 0.0,
        }

    sizes = [chunk.char_count for chunk in chunks]
    return {
        "count": len(chunks),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "average_chars": sum(sizes) / len(sizes),
    }


def _split_fixed_spans(text: str, *, chunk_size: int, overlap: int) -> list[_TextSpan]:
    _validate_split_options(chunk_size, overlap)
    if not text:
        return []

    spans: list[_TextSpan] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if chunk_text:
            spans.append(_TextSpan(text=chunk_text, start=start, end=end))
        if end >= len(text):
            break
        start = end - overlap

    return spans


def _split_paragraph_spans(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[_TextSpan]:
    _validate_split_options(chunk_size, overlap)
    if not text or not text.strip():
        return []

    spans: list[_TextSpan] = []
    start = _skip_whitespace_forward(text, 0, len(text))

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        if hard_end >= len(text):
            end = len(text)
        else:
            end = _find_preferred_end(text, start, hard_end, chunk_size)

        content_start = _skip_whitespace_forward(text, start, end)
        content_end = _skip_whitespace_backward(text, content_start, end)
        if content_start < content_end:
            spans.append(
                _TextSpan(
                    text=text[content_start:content_end],
                    start=content_start,
                    end=content_end,
                )
            )

        if end >= len(text):
            break

        overlap_target = max(end - overlap, start + 1)
        next_start = _find_safe_overlap_start(text, overlap_target, end)
        if next_start <= start:
            next_start = end
        start = _skip_whitespace_forward(text, next_start, len(text))

    return spans


def _find_preferred_end(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """chunk 후반부에서 문단, 줄, 문장, 단어 순으로 안전한 경계를 찾는다."""

    minimum = min(start + max(int(chunk_size * 0.55), 1), hard_end)
    search_area = text[minimum:hard_end]

    for separator in ("\n\n", "\n"):
        position = search_area.rfind(separator)
        if position >= 0:
            return minimum + position + len(separator)

    sentence_matches = list(
        re.finditer(r"[.!?。！？](?:[\"'”’)]*)\s+", search_area)
    )
    if sentence_matches:
        return minimum + sentence_matches[-1].end()

    position = search_area.rfind(" ")
    if position >= 0:
        return minimum + position + 1

    return hard_end


def _find_safe_overlap_start(text: str, target: int, previous_end: int) -> int:
    """overlap 근처의 다음 단어·줄 시작점을 찾아 중간 절단을 줄인다."""

    search_end = min(previous_end, target + 80)
    area = text[target:search_end]

    for separator in ("\n\n", "\n", " "):
        position = area.find(separator)
        if position >= 0:
            return target + position + len(separator)

    return target


def _skip_whitespace_forward(text: str, start: int, limit: int) -> int:
    while start < limit and text[start].isspace():
        start += 1
    return start


def _skip_whitespace_backward(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _validate_split_options(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0:
        raise ValueError("overlap은 0 이상이어야 합니다.")
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")


def main() -> None:
    text_dir = Path(__file__).resolve().parents[1] / "docs" / "text"
    text_files = sorted(text_dir.glob("*.txt"))
    if not text_files:
        print("docs/text에 텍스트가 없습니다. 먼저 문서 추출을 실행하세요.")
        return

    text = text_files[0].read_text(encoding="utf-8")
    print(f"문서: {text_files[0].name} ({len(text):,}자)\n")

    for name, chunks in [
        ("고정 크기 500/100", chunk_fixed(text, chunk_size=500, overlap=100)),
        ("문단 우선 700/120", chunk_by_paragraph(text, max_chars=700, overlap=120)),
    ]:
        sizes = [len(chunk["text"]) for chunk in chunks]
        print(
            f"[{name}] {len(chunks)}개 · "
            f"평균 {sum(sizes) // max(len(sizes), 1)}자 · "
            f"최소 {min(sizes)}자 · 최대 {max(sizes)}자"
        )
        print(f"첫 chunk: {chunks[0]['text'][:100]!r}\n")


if __name__ == "__main__":
    main()
