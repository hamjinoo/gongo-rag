"""docs/raw의 PDF를 공통 문서 추출기로 읽어 docs/text에 저장한다."""

from pathlib import Path

from document_ingestion import ExtractedDocument, extract_document_from_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "docs" / "raw"
TEXT_DIR = PROJECT_ROOT / "docs" / "text"


def extract_pdf_result(pdf_path: Path) -> ExtractedDocument:
    """PDF 텍스트와 페이지별 추출 방식, 경고를 함께 반환한다."""
    return extract_document_from_path(pdf_path)


def extract_pdf(pdf_path: Path) -> str:
    """기존 호출 코드와 호환되도록 추출 텍스트만 반환한다."""
    return extract_pdf_result(pdf_path).text


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"docs/raw에 PDF가 없습니다: {RAW_DIR}")
        return

    for pdf_path in pdf_files:
        try:
            result = extract_pdf_result(pdf_path)
        except Exception as exc:  # 깨진 파일은 건너뛰고 다음 파일을 처리한다.
            print(f"[실패] {pdf_path.name}: {exc}")
            continue

        out_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        out_path.write_text(result.text, encoding="utf-8")
        print(f"[완료] {pdf_path.name} → {out_path.name} ({result.char_count:,}자)")
        for warning in result.warnings:
            print(f"  [경고] {warning}")

    print("\n다음 할 일: docs/text의 결과를 원본과 비교해 빠진 글자와 깨진 표를 확인하세요.")


if __name__ == "__main__":
    main()
