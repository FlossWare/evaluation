"""Ollama chat backend implemented with the Python standard library."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluation_ai.types import ChatMessage, ChatResponse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_ALLOWED_OPTIONS = {
    "num_ctx",
    "num_predict",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "seed",
    "stop",
}


class OllamaBackendError(RuntimeError):
    """Raised when an Ollama request fails or returns an invalid response."""


class OllamaBackend:
    """Asynchronous LLMBackend adapter for an Ollama server.

    Blocking HTTP I/O runs in a worker thread. Requests are bounded by
    max_concurrency (one by default) to avoid competing with laptop workloads.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        timeout: float = 300.0,
        keep_alive: str | int = "5m",
        max_concurrency: int = 1,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str = "",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request to Ollama."""
        if not model.strip():
            raise ValueError("model must be specified for OllamaBackend")

        options: dict[str, Any] = {}
        supplied_options = kwargs.pop("options", {})
        if supplied_options:
            if not isinstance(supplied_options, dict):
                raise ValueError("options must be a dictionary")
            options.update(supplied_options)

        for key in tuple(kwargs):
            if key in _ALLOWED_OPTIONS:
                options[key] = kwargs.pop(key)
        options["temperature"] = temperature

        payload = {
            "model": model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "stream": False,
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            "options": options,
        }
        if kwargs:
            logger.debug("Ignoring unsupported Ollama chat arguments: %s", sorted(kwargs))

        async with self._semaphore:
            response = await asyncio.to_thread(self._post_chat, payload)

        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaBackendError("Ollama response did not contain message.content")

        usage = {
            key: response[key]
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
            if key in response
        }
        resolved_model = str(response.get("model") or model)
        # Resolve the mutable tag to an immutable digest after inference.
        # Failure does not discard a valid response, but is explicitly recorded.
        identity = await asyncio.to_thread(self._resolve_model_identity, resolved_model)
        return ChatResponse(
            content=message["content"],
            model=resolved_model,
            provider="ollama",
            usage=usage,
            model_identity=identity,
        )

    def _resolve_model_identity(self, model: str) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "runtime": "ollama",
            "runtime_version": None,
            "model_reference": model,
            "digest": None,
            "details": {},
            "identity_complete": False,
        }
        try:
            version_request = Request(f"{self.base_url}/api/version", method="GET")
            with urlopen(version_request, timeout=min(self.timeout, 15)) as result:
                version_data = json.loads(result.read().decode("utf-8"))
            tags_request = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(tags_request, timeout=min(self.timeout, 15)) as result:
                tags_data = json.loads(result.read().decode("utf-8"))
            identity["runtime_version"] = version_data.get("version")
            entries = tags_data.get("models", [])
            match = next(
                (
                    item for item in entries
                    if isinstance(item, dict)
                    and model in {item.get("name"), item.get("model")}
                ),
                None,
            )
            if match is None:
                identity["identity_error"] = "model_not_found_in_local_inventory"
                return identity
            identity["digest"] = match.get("digest")
            identity["modified_at"] = match.get("modified_at")
            identity["size_bytes"] = match.get("size")
            identity["details"] = match.get("details") or {}
            identity["identity_complete"] = bool(
                identity["runtime_version"] and identity["digest"]
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            identity["identity_error"] = str(exc)
        return identity

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as result:
                decoded = json.loads(result.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaBackendError(
                f"Ollama returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaBackendError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaBackendError("Ollama returned invalid JSON") from exc

        if not isinstance(decoded, dict):
            raise OllamaBackendError("Ollama returned an unexpected response shape")
        return decoded
