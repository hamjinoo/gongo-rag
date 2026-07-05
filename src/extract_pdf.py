"""
extract_pdf.py — 공고문 PDF → 텍스트 추출  [✅ 완성 배관: 실행만 하면 됨]

계획서 규칙: "PDF 추출 보일러플레이트"는 맡겨도 되는 배관 → 완성본 제공.

사용법:
    1) docs/raw/ 에 PDF 3개 넣기
    2) 02-gongo-rag 폴더에서:  python src\\extract_pdf.py
    3) docs/text/ 에 같은 이름의 .txt 생성됨
    4) 생성된 .txt를 눈으로 읽고 notes/관찰노트-템플릿.md 채우기  ← 진짜 과제는 이것

주의: 추출 품질에 매몰되지 말 것. 60점짜리 텍스트로 전체 흐름 완주가 목표 (계획서 금지사항 3).
"""
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "docs" / "raw"
TEXT_DIR = PROJECT_ROOT / "docs" / "text"


def extract_pdf(pdf_path: Path) -> str:
    """PDF 한 개의 전체 텍스트를 페이지 구분자와 함께 반환."""
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            parts.append(f"\n[페이지 {page_no}]\n{text}")
    return "\n".join(parts)


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"docs/raw/ 에 PDF가 없습니다: {RAW_DIR}")
        print("기업마당(bizinfo.go.kr)이나 K-Startup에서 공고문 PDF 3개를 받아 넣어주세요.")
        return

    for pdf_path in pdf_files:
        try:
            text = extract_pdf(pdf_path)
        except Exception as e:  # 깨진 PDF 등은 건너뛰고 계속
            print(f"[실패] {pdf_path.name}: {e}")
            continue

        out_path = TEXT_DIR / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")

        n_chars = len(text.strip())
        print(f"[완료] {pdf_path.name} → {out_path.name} ({n_chars:,}자)")
        if n_chars < 500:
            print("  ⚠️ 추출된 글자가 너무 적습니다. 스캔 이미지 PDF일 가능성이 높습니다.")
            print("     → OCR로 씨름하지 말고, 텍스트가 있는 다른 공고문으로 교체하는 게 빠릅니다.")

    print("\n다음 할 일: docs/text/ 의 파일을 열어 원본 PDF와 비교하며")
    print("notes/관찰노트-템플릿.md 를 채우세요. (표가 어떻게 깨졌는지가 관전 포인트)")


if __name__ == "__main__":
    main()
