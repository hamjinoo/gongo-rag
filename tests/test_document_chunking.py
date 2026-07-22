"""추출 문서 → metadata가 보존된 chunk 변환 테스트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import (  # noqa: E402
    ChunkingConfig,
    chunk_document,
    chunk_documents,
    summarize_chunks,
)
from document_ingestion import ExtractedDocument, ExtractedPage  # noqa: E402


def make_document() -> ExtractedDocument:
    return ExtractedDocument(
        filename="청년창업 공고.pdf",
        file_type="pdf",
        source_sha256="a" * 64,
        pages=[
            ExtractedPage(
                page_number=1,
                label="페이지 1",
                text=(
                    "신청 자격\n\n"
                    "신청 대상은 창업 3년 이내 기업입니다. "
                    "대표자는 만 39세 이하여야 합니다.\n\n"
                    "접수 기간은 7월 31일까지입니다."
                ),
                method="native",
            ),
            ExtractedPage(
                page_number=2,
                label="페이지 2",
                text=(
                    "지원 내용\n\n"
                    "사업화 자금을 최대 1억원까지 지원합니다. "
                    "선정 결과는 전자우편으로 안내합니다."
                ),
                method="ocr",
            ),
        ],
    )


def test_chunk_document_preserves_source_and_page_metadata():
    chunks = chunk_document(
        make_document(),
        config=ChunkingConfig(strategy="paragraph", chunk_size=80, overlap=15),
    )

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.source_filename == "청년창업 공고.pdf" for chunk in chunks)
    assert all(chunk.source_sha256 == "a" * 64 for chunk in chunks)
    assert all(chunk.file_type == "pdf" for chunk in chunks)
    assert {chunk.extraction_method for chunk in chunks} == {"native", "ocr"}
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_never_cross_page_boundaries():
    document = make_document()
    chunks = chunk_document(
        document,
        config=ChunkingConfig(strategy="fixed", chunk_size=50, overlap=10),
    )

    page_text = {page.page_number: page.text for page in document.pages}
    for chunk in chunks:
        original = page_text[chunk.page_number]
        assert original[chunk.start_char:chunk.end_char] == chunk.text
        assert len(chunk.text) <= 50


def test_paragraph_strategy_prefers_blank_line_boundary():
    document = ExtractedDocument(
        filename="paragraph.txt",
        file_type="text",
        source_sha256="b" * 64,
        pages=[
            ExtractedPage(
                page_number=1,
                label="문서 전체",
                text=("가" * 45) + "\n\n" + ("나" * 45),
                method="plain",
            )
        ],
    )
    chunks = chunk_document(
        document,
        config=ChunkingConfig(strategy="paragraph", chunk_size=80, overlap=0),
    )

    assert chunks[0].text == "가" * 45
    assert chunks[1].text == "나" * 45


def test_empty_pages_are_skipped():
    document = ExtractedDocument(
        filename="empty-page.pdf",
        file_type="pdf",
        source_sha256="c" * 64,
        pages=[
            ExtractedPage(1, "페이지 1", "", "empty"),
            ExtractedPage(2, "페이지 2", "두 번째 페이지 내용", "native"),
        ],
    )
    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_duplicate_document_content_is_chunked_once_by_default():
    document = make_document()
    chunks_once = chunk_document(document)
    chunks_many = chunk_documents([document, document])

    assert [chunk.to_dict() for chunk in chunks_many] == [
        chunk.to_dict() for chunk in chunks_once
    ]


def test_duplicate_document_can_be_kept_when_requested():
    document = make_document()
    chunks = chunk_documents([document, document], deduplicate=False)

    assert len(chunks) == len(chunk_document(document)) * 2
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_chunk_dict_has_text_and_rag_metadata():
    chunk = chunk_document(make_document())[0]
    payload = chunk.to_dict()

    assert payload["id"] == chunk.id
    assert payload["text"] == chunk.text
    assert payload["metadata"]["source_filename"] == "청년창업 공고.pdf"
    assert payload["metadata"]["page_number"] == 1
    assert payload["metadata"]["extraction_method"] == "native"
    assert payload["metadata"]["strategy"] == "paragraph"


def test_chunk_summary_reports_sizes():
    chunks = chunk_document(
        make_document(),
        config=ChunkingConfig(strategy="fixed", chunk_size=50, overlap=10),
    )
    summary = summarize_chunks(chunks)
    sizes = [len(chunk.text) for chunk in chunks]

    assert summary["count"] == len(chunks)
    assert summary["min_chars"] == min(sizes)
    assert summary["max_chars"] == max(sizes)
    assert summary["average_chars"] == sum(sizes) / len(sizes)


def test_invalid_chunk_settings_are_rejected():
    invalid = [
        {"chunk_size": 0, "overlap": 0},
        {"chunk_size": 100, "overlap": -1},
        {"chunk_size": 100, "overlap": 100},
        {"chunk_size": 100, "overlap": 101},
    ]

    for values in invalid:
        try:
            ChunkingConfig(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 설정을 거절해야 합니다: {values}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as error:
            print(f"  ❌ {test.__name__}: {error}")
    print(f"\n{passed}/{len(tests)} 통과")
    if passed != len(tests):
        raise SystemExit(1)
