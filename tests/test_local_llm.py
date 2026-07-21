"""Ollama 로컬 LLM 연결의 단위 테스트."""

import json
import sys
from pathlib import Path
from urllib import error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import local_llm  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_status_reports_installed_local_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(
        local_llm.request,
        "urlopen",
        lambda http_request, timeout: FakeResponse(
            {"models": [{"name": "qwen3:4b", "model": "qwen3:4b"}]}
        ),
    )

    status = local_llm.get_ollama_status()

    assert status.ready is True
    assert status.server_ready is True
    assert status.model == "qwen3:4b"


def test_status_explains_how_to_pull_missing_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(
        local_llm.request,
        "urlopen",
        lambda http_request, timeout: FakeResponse(
            {"models": [{"name": "gemma3:4b", "model": "gemma3:4b"}]}
        ),
    )

    status = local_llm.get_ollama_status()

    assert status.ready is False
    assert status.server_ready is True
    assert "ollama pull qwen3:4b" in status.message


def test_status_reports_stopped_ollama_server(monkeypatch):
    def raise_connection_error(http_request, timeout):
        raise error.URLError("connection refused")

    monkeypatch.setattr(local_llm.request, "urlopen", raise_connection_error)

    status = local_llm.get_ollama_status()

    assert status.ready is False
    assert status.server_ready is False
    assert "Ollama에 연결" in status.message


def test_chat_call_uses_local_non_streaming_api(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return FakeResponse({"message": {"role": "assistant", "content": "로컬 답변"}})

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(local_llm.request, "urlopen", fake_urlopen)

    answer = local_llm.call_ollama("근거만 사용하세요.")

    assert answer == "로컬 답변"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 12
    body = captured["body"]
    assert body["model"] == "qwen3:4b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"] == [{"role": "user", "content": "근거만 사용하세요."}]
