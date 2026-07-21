"""Ollama의 로컬 HTTP API로 한국어 LLM을 호출한다."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error, request


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 180.0


class LocalLLMError(RuntimeError):
    """사용자가 조치할 수 있는 로컬 LLM 오류."""


class OllamaUnavailableError(LocalLLMError):
    """Ollama 서버에 연결할 수 없음."""


class OllamaModelMissingError(LocalLLMError):
    """요청한 로컬 모델이 설치되지 않음."""


class OllamaResponseError(LocalLLMError):
    """Ollama가 올바른 답변을 반환하지 않음."""


@dataclass(frozen=True)
class OllamaStatus:
    """UI와 CLI가 공통으로 사용하는 Ollama 준비 상태."""

    ready: bool
    server_ready: bool
    model: str
    message: str
    installed_models: tuple[str, ...] = ()


def ollama_base_url() -> str:
    """환경 변수 또는 기본 로컬 주소를 반환한다."""

    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def ollama_model() -> str:
    """환경 변수 또는 포트폴리오 기본 모델을 반환한다."""

    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def ollama_timeout_seconds() -> float:
    """잘못된 환경 변수는 안전한 기본 제한 시간으로 되돌린다."""

    raw_value = os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_OLLAMA_TIMEOUT_SECONDS


def get_ollama_status(*, timeout_seconds: float = 1.0) -> OllamaStatus:
    """서버 연결과 선택 모델 설치 여부를 짧게 확인한다."""

    model = ollama_model()
    try:
        payload = _json_request(
            f"{ollama_base_url()}/api/tags",
            method="GET",
            timeout_seconds=timeout_seconds,
        )
    except OllamaUnavailableError as exc:
        return OllamaStatus(
            ready=False,
            server_ready=False,
            model=model,
            message=str(exc),
        )
    except LocalLLMError as exc:
        return OllamaStatus(
            ready=False,
            server_ready=True,
            model=model,
            message=str(exc),
        )

    raw_models = payload.get("models", [])
    installed = tuple(
        sorted(
            {
                str(item.get("model") or item.get("name") or "").strip()
                for item in raw_models
                if isinstance(item, dict)
            }
            - {""}
        )
    )
    if not _model_is_installed(model, installed):
        return OllamaStatus(
            ready=False,
            server_ready=True,
            model=model,
            message=f"로컬 모델이 없습니다. `ollama pull {model}`을 한 번 실행하세요.",
            installed_models=installed,
        )
    return OllamaStatus(
        ready=True,
        server_ready=True,
        model=model,
        message=f"로컬 LLM 준비됨 · {model}",
        installed_models=installed,
    )


def call_ollama(
    prompt: str,
    *,
    response_format: str | dict[str, object] | None = None,
    max_tokens: int = 512,
) -> str:
    """Qwen의 thinking 출력은 끄고 최종 답변만 동기식으로 받는다."""

    if not prompt.strip():
        raise ValueError("prompt는 비어 있을 수 없습니다.")
    if max_tokens < 1:
        raise ValueError("max_tokens는 1 이상이어야 합니다.")

    model = ollama_model()
    request_body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "추론 과정이나 작업 설명을 출력하지 말고 요청한 최종 결과만 "
                    "한국어로 반환하세요. /no_think"
                ),
            },
            {"role": "user", "content": f"{prompt.rstrip()}\n\n/no_think"},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    if response_format is not None:
        request_body["format"] = response_format
    payload = _json_request(
        f"{ollama_base_url()}/api/chat",
        method="POST",
        timeout_seconds=ollama_timeout_seconds(),
        body=request_body,
    )
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaResponseError("Ollama가 비어 있는 답변을 반환했습니다.")
    return content.strip()


def _json_request(
    url: str,
    *,
    method: str,
    timeout_seconds: float,
    body: dict[str, object] | None = None,
) -> dict[str, object]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    http_request = request.Request(
        url,
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "model" in detail.lower():
            model = ollama_model()
            raise OllamaModelMissingError(
                f"로컬 모델이 없습니다. `ollama pull {model}`을 한 번 실행하세요."
            ) from exc
        raise OllamaResponseError(
            f"Ollama 요청에 실패했습니다. HTTP {exc.code}: {detail[:200]}"
        ) from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise OllamaUnavailableError(
            "Ollama에 연결할 수 없습니다. Ollama 앱 또는 `ollama serve`를 실행하세요."
        ) from exc

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise OllamaResponseError("Ollama 응답이 올바른 JSON이 아닙니다.") from exc
    if not isinstance(payload, dict):
        raise OllamaResponseError("Ollama 응답 형식이 올바르지 않습니다.")
    if payload.get("error"):
        error_message = str(payload["error"])
        if "model" in error_message.lower() and "not found" in error_message.lower():
            model = ollama_model()
            raise OllamaModelMissingError(
                f"로컬 모델이 없습니다. `ollama pull {model}`을 한 번 실행하세요."
            )
        raise OllamaResponseError(f"Ollama 오류: {error_message}")
    return payload


def _model_is_installed(model: str, installed: tuple[str, ...]) -> bool:
    if model in installed:
        return True
    if ":" not in model and f"{model}:latest" in installed:
        return True
    return False


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "LocalLLMError",
    "OllamaModelMissingError",
    "OllamaResponseError",
    "OllamaStatus",
    "OllamaUnavailableError",
    "call_ollama",
    "get_ollama_status",
    "ollama_base_url",
    "ollama_model",
]
