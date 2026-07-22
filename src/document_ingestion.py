"""PDF, 스캔 PDF, DOCX, 이미지, 텍스트 파일을 공통 형식으로 추출한다."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol


PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | DOCX_EXTENSIONS | TEXT_EXTENSIONS | IMAGE_EXTENSIONS

ExtractionMethod = Literal["native", "ocr", "docx", "plain", "empty"]


class DocumentIngestionError(Exception):
    """사용자가 이해할 수 있는 문서 추출 오류."""


class UnsupportedDocumentError(DocumentIngestionError):
    """지원하지 않는 파일 형식."""


class InvalidDocumentError(DocumentIngestionError):
    """확장자와 실제 파일 내용이 다르거나 파일이 손상됨."""


class MissingDependencyError(DocumentIngestionError):
    """필수 Python 패키지가 설치되지 않음."""


class OCRUnavailableError(DocumentIngestionError):
    """OCR 엔진 또는 언어 모델을 사용할 수 없음."""


@dataclass(frozen=True)
class ExtractionConfig:
    """문서 추출의 안전 한도와 OCR 기준."""

    ocr_enabled: bool = True
    ocr_language: str = "kor+eng"
    min_native_chars: int = 30
    ocr_dpi: int = 250
    ocr_timeout_seconds: int = 45
    max_file_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.min_native_chars < 0:
            raise ValueError("min_native_chars는 0 이상이어야 합니다.")
        if not 150 <= self.ocr_dpi <= 400:
            raise ValueError("ocr_dpi는 150~400 사이여야 합니다.")
        if self.ocr_timeout_seconds <= 0:
            raise ValueError("ocr_timeout_seconds는 1 이상이어야 합니다.")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes는 1 이상이어야 합니다.")


@dataclass(frozen=True)
class OCRStatus:
    ready: bool
    engine: str
    message: str
    available_languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    label: str
    text: str
    method: ExtractionMethod

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    source_sha256: str
    pages: list[ExtractedPage]
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts: list[str] = []
        show_markers = len(self.pages) > 1 or self.file_type == "pdf"
        for page in self.pages:
            if not page.text.strip():
                continue
            if show_markers:
                parts.append(f"[{page.label}]\n{page.text.strip()}")
            else:
                parts.append(page.text.strip())
        return "\n\n".join(parts).strip()

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def used_ocr(self) -> bool:
        return any(page.method == "ocr" for page in self.pages)

    @property
    def empty_pages(self) -> list[int]:
        return [page.page_number for page in self.pages if not page.text.strip()]


class OCRBackend(Protocol):
    def status(self, language: str) -> OCRStatus:
        """OCR 엔진과 요청 언어를 사용할 수 있는지 확인."""

    def recognize(self, image_bytes: bytes, language: str, timeout_seconds: int) -> str:
        """이미지 바이트에서 텍스트를 추출."""


class TesseractOCRBackend:
    """pytesseract를 통해 로컬 Tesseract OCR을 사용한다."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.tesseract_cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        self._status_cache: dict[str, OCRStatus] = {}

    def _configure_command(self, pytesseract_module) -> None:
        command = self.tesseract_cmd
        if not command:
            command = shutil.which("tesseract")

        if not command and os.name == "nt":
            common_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if common_path.exists():
                command = str(common_path)

        if command:
            pytesseract_module.pytesseract.tesseract_cmd = command

    def status(self, language: str) -> OCRStatus:
        if language in self._status_cache:
            return self._status_cache[language]

        try:
            import pytesseract
        except ImportError:
            result = OCRStatus(
                ready=False,
                engine="tesseract",
                message="pytesseract가 없습니다. requirements.txt를 설치하세요.",
            )
            self._status_cache[language] = result
            return result

        self._configure_command(pytesseract)
        try:
            pytesseract.get_tesseract_version()
            languages = tuple(sorted(pytesseract.get_languages(config="")))
        except Exception:
            result = OCRStatus(
                ready=False,
                engine="tesseract",
                message=(
                    "Tesseract 실행 파일을 찾지 못했습니다. Tesseract 5와 kor/eng 언어 "
                    "데이터를 설치하거나 TESSERACT_CMD를 설정하세요."
                ),
            )
            self._status_cache[language] = result
            return result

        required = tuple(part for part in language.split("+") if part)
        missing = [lang for lang in required if lang not in languages]
        if missing:
            result = OCRStatus(
                ready=False,
                engine="tesseract",
                message=f"OCR 언어 데이터가 없습니다: {', '.join(missing)}",
                available_languages=languages,
            )
            self._status_cache[language] = result
            return result

        result = OCRStatus(
            ready=True,
            engine="tesseract",
            message=f"Tesseract OCR 준비 완료 ({language})",
            available_languages=languages,
        )
        self._status_cache[language] = result
        return result

    def recognize(self, image_bytes: bytes, language: str, timeout_seconds: int) -> str:
        status = self.status(language)
        if not status.ready:
            raise OCRUnavailableError(status.message)

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise MissingDependencyError(
                "OCR에 필요한 Pillow 또는 pytesseract가 설치되지 않았습니다."
            ) from exc

        self._configure_command(pytesseract)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
                return pytesseract.image_to_string(
                    rgb_image,
                    lang=language,
                    config="--oem 1 --psm 4",
                    timeout=timeout_seconds,
                )
        except RuntimeError as exc:
            raise OCRUnavailableError(f"OCR 시간이 초과됐습니다: {exc}") from exc
        except Exception as exc:
            raise OCRUnavailableError(f"OCR 실행에 실패했습니다: {exc}") from exc


def get_ocr_status(
    language: str = "kor+eng",
    backend: OCRBackend | None = None,
) -> OCRStatus:
    return (backend or TesseractOCRBackend()).status(language)


def extract_document(
    filename: str,
    data: bytes,
    *,
    config: ExtractionConfig | None = None,
    ocr_backend: OCRBackend | None = None,
) -> ExtractedDocument:
    """업로드된 파일 바이트를 파일 종류에 맞게 추출한다."""

    settings = config or ExtractionConfig()
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()

    if not safe_name:
        raise InvalidDocumentError("파일 이름이 없습니다.")
    if extension == ".doc":
        raise UnsupportedDocumentError(
            "옛날 Word 형식(.doc)은 지원하지 않습니다. Word에서 .docx로 다시 저장하세요."
        )
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(f"지원하지 않는 파일입니다. 지원 형식: {supported}")
    if not data:
        raise InvalidDocumentError("빈 파일은 추출할 수 없습니다.")
    if len(data) > settings.max_file_bytes:
        limit_mb = settings.max_file_bytes / (1024 * 1024)
        raise InvalidDocumentError(f"파일이 너무 큽니다. 최대 {limit_mb:g}MB까지 가능합니다.")

    source_hash = hashlib.sha256(data).hexdigest()
    backend = ocr_backend or TesseractOCRBackend()

    if extension in PDF_EXTENSIONS:
        pages, warnings = _extract_pdf(data, settings, backend)
        file_type = "pdf"
    elif extension in DOCX_EXTENSIONS:
        pages, warnings = _extract_docx(data)
        file_type = "docx"
    elif extension in IMAGE_EXTENSIONS:
        pages, warnings = _extract_image(data, settings, backend)
        file_type = "image"
    else:
        pages, warnings = _extract_plain_text(data)
        file_type = "text"

    result = ExtractedDocument(
        filename=safe_name,
        file_type=file_type,
        source_sha256=source_hash,
        pages=pages,
        warnings=warnings,
    )
    if not result.text:
        _append_warning_once(
            result.warnings,
            "추출된 텍스트가 없습니다. 스캔 문서라면 OCR 설치 상태와 원본 품질을 확인하세요.",
        )
    return result


def extract_document_from_path(
    path: str | Path,
    *,
    config: ExtractionConfig | None = None,
    ocr_backend: OCRBackend | None = None,
) -> ExtractedDocument:
    source = Path(path)
    return extract_document(
        source.name,
        source.read_bytes(),
        config=config,
        ocr_backend=ocr_backend,
    )


def _extract_pdf(
    data: bytes,
    config: ExtractionConfig,
    ocr_backend: OCRBackend,
) -> tuple[list[ExtractedPage], list[str]]:
    if not data.startswith(b"%PDF"):
        raise InvalidDocumentError("확장자는 PDF지만 실제 PDF 파일이 아닙니다.")

    try:
        import pymupdf
    except ImportError as exc:
        raise MissingDependencyError("PyMuPDF가 없습니다. requirements.txt를 설치하세요.") from exc

    pages: list[ExtractedPage] = []
    warnings: list[str] = []
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError(f"PDF를 열 수 없습니다: {exc}") from exc

    try:
        if document.needs_pass:
            raise InvalidDocumentError("암호로 잠긴 PDF는 읽을 수 없습니다. 암호를 해제한 뒤 다시 올려주세요.")

        for page_index, page in enumerate(document, start=1):
            native_text = _clean_text(page.get_text("text", sort=True))
            method: ExtractionMethod = "native" if native_text else "empty"
            final_text = native_text

            needs_ocr = (
                config.ocr_enabled
                and len(_compact_text(native_text)) < config.min_native_chars
            )
            if needs_ocr:
                try:
                    pixmap = page.get_pixmap(
                        dpi=config.ocr_dpi,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    ocr_text = _clean_text(
                        ocr_backend.recognize(
                            pixmap.tobytes("png"),
                            config.ocr_language,
                            config.ocr_timeout_seconds,
                        )
                    )
                    if len(_compact_text(ocr_text)) > len(_compact_text(native_text)):
                        final_text = ocr_text
                        method = "ocr"
                    elif not ocr_text:
                        _append_warning_once(
                            warnings,
                            f"{page_index}페이지에서 OCR 텍스트를 찾지 못했습니다.",
                        )
                except OCRUnavailableError as exc:
                    _append_warning_once(warnings, f"OCR을 사용할 수 없습니다: {exc}")

            pages.append(
                ExtractedPage(
                    page_number=page_index,
                    label=f"페이지 {page_index}",
                    text=final_text,
                    method=method,
                )
            )
    finally:
        document.close()

    return pages, warnings


def _extract_docx(data: bytes) -> tuple[list[ExtractedPage], list[str]]:
    if not data.startswith(b"PK"):
        raise InvalidDocumentError("확장자는 DOCX지만 실제 DOCX 파일이 아닙니다.")

    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise MissingDependencyError("python-docx가 없습니다. requirements.txt를 설치하세요.") from exc

    try:
        document = Document(BytesIO(data))
    except Exception as exc:
        raise InvalidDocumentError(f"DOCX를 열 수 없습니다: {exc}") from exc

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = _clean_text(Paragraph(child, document).text)
            if text:
                blocks.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            rows: list[str] = []
            for row in table.rows:
                cells = [_clean_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("[표]\n" + "\n".join(rows))

    warnings: list[str] = []
    if document.inline_shapes:
        warnings.append(
            "DOCX 안의 그림은 보존하지 않습니다. 현재 단계에서는 문단과 표의 글자만 추출합니다."
        )

    text = _clean_text("\n\n".join(blocks))
    return [
        ExtractedPage(
            page_number=1,
            label="문서 전체",
            text=text,
            method="docx" if text else "empty",
        )
    ], warnings


def _extract_image(
    data: bytes,
    config: ExtractionConfig,
    ocr_backend: OCRBackend,
) -> tuple[list[ExtractedPage], list[str]]:
    _validate_image(data)

    if not config.ocr_enabled:
        return [
            ExtractedPage(1, "이미지 1", "", "empty")
        ], ["이미지에서 글자를 읽으려면 OCR을 켜야 합니다."]

    warnings: list[str] = []
    try:
        text = _clean_text(
            ocr_backend.recognize(
                data,
                config.ocr_language,
                config.ocr_timeout_seconds,
            )
        )
        method: ExtractionMethod = "ocr" if text else "empty"
    except OCRUnavailableError as exc:
        text = ""
        method = "empty"
        warnings.append(f"OCR을 사용할 수 없습니다: {exc}")

    return [ExtractedPage(1, "이미지 1", text, method)], warnings


def _validate_image(data: bytes) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingDependencyError("Pillow가 없습니다. requirements.txt를 설치하세요.") from exc

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise InvalidDocumentError(f"이미지 파일을 읽을 수 없습니다: {exc}") from exc


def _extract_plain_text(data: bytes) -> tuple[list[ExtractedPage], list[str]]:
    warnings: list[str] = []
    decoded: str | None = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            decoded = data.decode(encoding)
            if encoding != "utf-8-sig":
                warnings.append(f"텍스트 인코딩을 {encoding}로 감지했습니다.")
            break
        except UnicodeDecodeError:
            continue

    if decoded is None:
        raise InvalidDocumentError("텍스트 인코딩을 읽을 수 없습니다.")

    text = _clean_text(decoded)
    return [
        ExtractedPage(
            page_number=1,
            label="문서 전체",
            text=text,
            method="plain" if text else "empty",
        )
    ], warnings


def _clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _append_warning_once(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="문서 파일에서 텍스트를 추출합니다.")
    parser.add_argument("input", type=Path, help="PDF, DOCX, 이미지 또는 텍스트 파일")
    parser.add_argument("-o", "--output", type=Path, help="저장할 TXT 경로")
    parser.add_argument("--no-ocr", action="store_true", help="OCR을 사용하지 않음")
    args = parser.parse_args()

    config = ExtractionConfig(ocr_enabled=not args.no_ocr)
    result = extract_document_from_path(args.input, config=config)
    output = args.output or args.input.with_suffix(".txt")
    output.write_text(result.text, encoding="utf-8")

    print(f"[완료] {result.filename} → {output} ({result.char_count:,}자)")
    for warning in result.warnings:
        print(f"[경고] {warning}")


if __name__ == "__main__":
    main()
