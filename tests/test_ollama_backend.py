"""Tests for the optional Ollama backend; no Ollama server is required."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from evaluation_ai import ChatMessage
from evaluation_ai.backends.ollama import OllamaBackend, OllamaBackendError


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_ollama_backend_maps_chat_response():
    backend = OllamaBackend()
    response_payload = {
        "model": "qwen3.5:2b",
        "message": {"role": "assistant", "content": '{"label":"success"}'},
        "prompt_eval_count": 42,
        "eval_count": 12,
        "total_duration": 1000,
    }

    def fake_urlopen(request, timeout=0):
        if request.full_url.endswith("/api/chat"):
            return FakeHTTPResponse(response_payload)
        if request.full_url.endswith("/api/version"):
            return FakeHTTPResponse({"version": "0.12.6"})
        if request.full_url.endswith("/api/tags"):
            return FakeHTTPResponse({
                "models": [{
                    "name": "qwen3.5:2b",
                    "model": "qwen3.5:2b",
                    "digest": "sha256:abc123",
                    "details": {"family": "qwen3", "parameter_size": "2B"},
                }]
            })
        raise AssertionError(request.full_url)

    with patch(
        "evaluation_ai.backends.ollama.urlopen",
        side_effect=fake_urlopen,
    ) as mocked:
        result = asyncio.run(
            backend.chat(
                [ChatMessage(role="user", content="Evaluate this turn")],
                model="qwen3.5:2b",
                temperature=0.0,
                num_ctx=4096,
            )
        )

    assert result.content == '{"label":"success"}'
    assert result.model == "qwen3.5:2b"
    assert result.provider == "ollama"
    assert result.usage["prompt_eval_count"] == 42
    assert result.model_identity["digest"] == "sha256:abc123"
    assert result.model_identity["runtime_version"] == "0.12.6"
    assert result.model_identity["identity_complete"] is True

    request = next(
        call.args[0] for call in mocked.call_args_list
        if call.args[0].full_url.endswith("/api/chat")
    )
    sent = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["num_ctx"] == 4096
    assert sent["keep_alive"] == "5m"


def test_ollama_backend_requires_model():
    backend = OllamaBackend()
    with pytest.raises(ValueError, match="model must be specified"):
        asyncio.run(backend.chat([ChatMessage(role="user", content="hello")]))


def test_ollama_backend_rejects_invalid_response_shape():
    backend = OllamaBackend()
    with patch(
        "evaluation_ai.backends.ollama.urlopen",
        return_value=FakeHTTPResponse({"model": "test"}),
    ):
        with pytest.raises(OllamaBackendError, match="message.content"):
            asyncio.run(
                backend.chat(
                    [ChatMessage(role="user", content="hello")],
                    model="test",
                )
            )


def test_ollama_backend_validates_resource_limits():
    with pytest.raises(ValueError, match="timeout"):
        OllamaBackend(timeout=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        OllamaBackend(max_concurrency=0)
