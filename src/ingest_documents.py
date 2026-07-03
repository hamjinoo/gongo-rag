from pathlib import Path
import fitz  # PyMuPDF


RAW_DIR = Path("docs/raw")
TEXT_DIR = Path("docs/text")


def extract_from_pdf(file_path: Path) -> str:
    doc = fitz.open(file_path)
    pages = []
    has_real_text = False

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True).strip()

        if text:
            has_real_text = True
            pages.append(f"\n\n===== PAGE {page_number} =====\n\n{text}")
        else:
            pages.append(f"\n\n===== PAGE {page_number} =====\n\n[NO TEXT EXTRACTED]")

    doc.close()

    if not has_real_text:
        return ""

    return "\n".join(pages).strip()


def extract_from_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8").strip()


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_from_pdf(file_path)

    if suffix == ".txt":
        return extract_from_txt(file_path)

    raise NotImplementedError(f"아직 지원하지 않는 형식입니다: {suffix}")


def main():
    if not RAW_DIR.exists():
        print("docs/raw 폴더가 없습니다.")
        return 
    
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    files = [p for p in RAW_DIR.iterdir() if p.is_file()]

    if not files:
        print("docs/raw 폴더에 파일이 없습니다.")
        return
    
    for file_path in files:
        print(f"처리 중: {file_path.name}")

        try:
            text = extract_text(file_path)

            if not text:
                print(f"추출 실패: {file_path.name} - 텍스트가 비어 있음")
                continue

            output_path = TEXT_DIR / f"{file_path.stem}.txt"
            output_path.write_text(text, encoding="utf-8")

            print(f"저장 완료: {output_path}")

        except NotImplementedError as e:
            print(f"보류: {file_path.name} - {e}")

        except Exception as e:
            print(f"실패: {file_path.name} - {e}")


if __name__ == "__main__":
    main()