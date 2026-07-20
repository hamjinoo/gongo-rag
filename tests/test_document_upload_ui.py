"""Streamlit 문서 업로드 패널의 최소 사용자 흐름 테스트."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def open_app() -> AppTest:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20)
    app.run()
    return app


def test_upload_screen_renders():
    app = open_app()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["1. 문서 넣기", "2. 질문하기"]
    assert len(app.get("file_uploader")) == 1
    assert any(button.label == "텍스트 추출" for button in app.button)


def test_text_file_can_be_uploaded_and_previewed():
    app = open_app()
    expected = "지원 대상은 창업 3년 이내 기업입니다."

    app.get("file_uploader")[0].upload(
        "sample.txt",
        expected.encode("utf-8"),
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert any(expected in area.value for area in app.text_area)
    assert [metric.value for metric in app.metric] == [str(len(expected)), "1", "미사용"]
    assert any(button.label == "추출 텍스트 받기" for button in app.get("download_button"))


def test_extracted_text_can_be_chunked_and_previewed():
    app = open_app()
    text = (
        "신청 자격은 창업 3년 이내 기업입니다.\n\n"
        "지원 금액은 최대 1억원입니다.\n\n"
    ) * 30

    app.get("file_uploader")[0].upload(
        "long-sample.txt",
        text.encode("utf-8"),
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert [button.label for button in app.button] == ["텍스트 추출", "Chunk 만들기"]
    app.button[1].click().run()

    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert "Chunk 수" in metric_labels
    assert "평균 크기" in metric_labels
    assert any(area.label == "Chunk 내용" and area.value for area in app.text_area)
    assert len(app.json) == 1
    assert any(
        button.label == "Chunk JSON 받기"
        for button in app.get("download_button")
    )


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
