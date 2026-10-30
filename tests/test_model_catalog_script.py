"""Tests for the hand-run Ollama model catalog helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "model-catalog.py"
SPEC = importlib.util.spec_from_file_location("model_catalog_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
model_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_catalog)


def test_snapshot_records_digest_and_runtime_version(tmp_path, monkeypatch):
    def fake_request(_base_url, path, payload=None):
        if path == "/api/version":
            return {"version": "0.12.6"}
        if path == "/api/tags":
            return {
                "models": [{
                    "name": "qwen3.5:2b",
                    "model": "qwen3.5:2b",
                    "digest": "sha256:abc123",
                    "size": 123456,
                    "modified_at": "2026-10-30T00:00:00Z",
                    "details": {
                        "family": "qwen3",
                        "parameter_size": "2.3B",
                        "quantization_level": "Q4_K_M",
                    },
                }]
            }
        if path == "/api/show":
            assert payload == {"model": "qwen3.5:2b", "verbose": True}
            return {
                "details": {"family": "qwen3", "quantization_level": "Q4_K_M"},
                "model_info": {"general.architecture": "qwen3"},
            }
        raise AssertionError(path)

    monkeypatch.setattr(model_catalog, "request_json", fake_request)
    record = model_catalog.save_snapshot(
        "http://127.0.0.1:11434", tmp_path, "qwen3.5:2b"
    )

    assert record["runtime"]["version"] == "0.12.6"
    assert record["model"]["digest"] == "sha256:abc123"
    assert record["model"]["details"]["parameter_size"] == "2.3B"
    assert record["model"]["show"]["model_info"]["general.architecture"] == "qwen3"

    catalog_file = tmp_path / "catalog" / "model-catalog.json"
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    assert len(catalog["models"]) == 1


def test_snapshot_does_not_duplicate_same_digest_and_runtime(tmp_path, monkeypatch):
    def fake_request(_base_url, path, payload=None):
        if path == "/api/version":
            return {"version": "0.12.6"}
        if path == "/api/tags":
            return {"models": [{
                "name": "qwen3.5:2b",
                "digest": "sha256:abc123",
                "details": {"family": "qwen3"},
            }]}
        if path == "/api/show":
            return {"details": {"family": "qwen3"}}
        raise AssertionError(path)

    monkeypatch.setattr(model_catalog, "request_json", fake_request)
    model_catalog.save_snapshot("http://localhost:11434", tmp_path, "qwen3.5:2b")
    model_catalog.save_snapshot("http://localhost:11434", tmp_path, "qwen3.5:2b")

    catalog_file = tmp_path / "catalog" / "model-catalog.json"
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    assert len(catalog["models"]) == 1


def test_snapshot_requires_digest(tmp_path, monkeypatch):
    def fake_request(_base_url, path, payload=None):
        if path == "/api/version":
            return {"version": "0.12.6"}
        if path == "/api/tags":
            return {"models": [{"name": "untagged-model"}]}
        if path == "/api/show":
            return {}
        raise AssertionError(path)

    monkeypatch.setattr(model_catalog, "request_json", fake_request)
    try:
        model_catalog.save_snapshot("http://localhost:11434", tmp_path, "untagged-model")
    except model_catalog.OllamaError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("snapshot should reject models without a digest")
