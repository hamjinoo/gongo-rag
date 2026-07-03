from pathlib import Path
import fitz  # PyMuPDF


RAW_DIR = Path("docs/raw")


def diagnose_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)

    print("=" * 80)
    print(f"파일명: {pdf_path.name}")
    print(f"페이지 수: {len(doc)}")
    print("=" * 80)

    total_text_length = 0
    total_words = 0
    total_images = 0

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True).strip()
        words = page.get_text("words")
        images = page.get_images(full=True)
        blocks = page.get_text("blocks")

        text_length = len(text)
        word_count = len(words)
        image_count = len(images)

        total_text_length += text_length
        total_words += word_count
        total_images += image_count

        print(f"\n[PAGE {page_index}]")
        print(f"- text length: {text_length}")
        print(f"- word count: {word_count}")
        print(f"- image count: {image_count}")
        print(f"- block count: {len(blocks)}")

        if text_length > 0:
            print("\n미리보기:")
            print(text[:300])
        else:
            print("\n미리보기: 추출된 텍스트 없음")

    print("\n" + "=" * 80)
    print("전체 요약")
    print(f"- total text length: {total_text_length}")
    print(f"- total words: {total_words}")
    print(f"- total images: {total_images}")

    if total_text_length == 0 and total_images > 0:
        print("\n판단: 이미지/스캔 PDF일 가능성이 큼. OCR 또는 다른 원본 필요.")
    elif total_text_length == 0 and total_images == 0:
        print("\n판단: 텍스트 추출 실패 또는 특수 PDF 가능성. 다른 라이브러리 테스트 필요.")
    else:
        print("\n판단: 텍스트 추출 가능 PDF. 전처리/줄바꿈 정리 단계로 진행 가능.")

    doc.close()


def main():
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("docs/raw 폴더에 PDF 파일이 없습니다.")
        return

    for pdf_path in pdf_files:
        diagnose_pdf(pdf_path)


if __name__ == "__main__":
    main()