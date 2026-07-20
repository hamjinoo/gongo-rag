"""Streamlit 앱 어디서든 호출할 수 있는 문서 업로드 패널."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from document_ingestion import (
    DocumentIngestionError,
    ExtractedDocument,
    ExtractionConfig,
    extract_document,
    get_ocr_status,
)


UPLOAD_TYPES = [
    "pdf",
    "doc",
    "docx",
    "txt",
    "md",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
]


def render_document_upload(
    *,
    key_prefix: str = "document_ingestion",
    max_file_mb: int = 20,
) -> list[ExtractedDocument]:
    """업로드 → 추출 → 미리보기 → TXT 다운로드 UI를 렌더링한다."""

    st.subheader("문서 넣기")
    st.caption(
        "일반 PDF와 DOCX는 글자를 바로 읽고, 스캔 PDF와 이미지는 필요한 페이지만 OCR합니다."
    )

    with st.expander("추출 설정", expanded=False):
        ocr_enabled = st.checkbox(
            "스캔 문서 OCR 사용",
            value=True,
            key=f"{key_prefix}_ocr_enabled",
        )
        ocr_dpi = st.slider(
            "OCR 선명도",
            min_value=150,
            max_value=350,
            value=250,
            step=50,
            help="높을수록 작은 글자를 잘 읽을 수 있지만 느리고 메모리를 더 사용합니다.",
            key=f"{key_prefix}_ocr_dpi",
        )

        if ocr_enabled:
            status = get_ocr_status("kor+eng")
            if status.ready:
                st.success(status.message)
            else:
                st.warning(status.message)
                st.caption("일반 PDF와 DOCX는 OCR이 없어도 추출할 수 있습니다.")

    uploaded_files = st.file_uploader(
        "PDF, DOCX, 이미지 또는 텍스트 파일을 올려주세요.",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"파일 하나당 최대 {max_file_mb}MB",
        key=f"{key_prefix}_files",
    )

    state_key = f"{key_prefix}_results"
    if st.button(
        "텍스트 추출",
        type="primary",
        disabled=not uploaded_files,
        key=f"{key_prefix}_extract",
    ):
        config = ExtractionConfig(
            ocr_enabled=ocr_enabled,
            ocr_dpi=ocr_dpi,
            max_file_bytes=max_file_mb * 1024 * 1024,
        )
        results: list[ExtractedDocument] = []

        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"{uploaded_file.name} 읽는 중..."):
                    result = extract_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        config=config,
                    )
                results.append(result)
            except DocumentIngestionError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: 예상하지 못한 오류가 발생했습니다. ({exc})")

        st.session_state[state_key] = results

    results = st.session_state.get(state_key, [])
    if not results:
        return []

    st.success("텍스트 추출이 끝났습니다. 아래에서 검색용 chunk로 나눌 수 있습니다.")
    for index, result in enumerate(results):
        icon = "🖼️" if result.used_ocr else "📄"
        with st.expander(f"{icon} {result.filename}", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("글자 수", f"{result.char_count:,}")
            col2.metric("구역 수", len(result.pages))
            col3.metric("OCR", "사용" if result.used_ocr else "미사용")

            for warning in result.warnings:
                st.warning(warning)

            method_summary = ", ".join(
                f"{page.label}: {page.method}" for page in result.pages
            )
            st.caption(f"추출 방식 · {method_summary}")

            preview = result.text[:8000]
            result_key = f"{index}_{result.source_sha256[:12]}"
            st.text_area(
                "추출 결과 미리보기",
                value=preview,
                height=280,
                key=f"{key_prefix}_preview_{result_key}",
                disabled=True,
            )
            if result.char_count > len(preview):
                st.caption("미리보기는 앞 8,000자만 표시합니다. 전체 내용은 TXT로 받으세요.")

            output_name = f"{Path(result.filename).stem}.txt"
            st.download_button(
                "추출 텍스트 받기",
                data=result.text.encode("utf-8"),
                file_name=output_name,
                mime="text/plain",
                key=f"{key_prefix}_download_{result_key}",
            )

    return results
