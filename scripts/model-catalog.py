#!/usr/bin/env python3
"""Inspect Ollama models and maintain a reproducible FlossWare model catalog.

Uses only the Python standard library. It never downloads, deletes, or modifies
model artifacts. Ollama remains responsible for its own manifests and blobs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODELS_DIR = Path.home() / ".FlossWare" / "models"


class OllamaError(RuntimeError):
    pass


def request_json(
    base_url: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"Cannot reach Ollama at {base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned invalid JSON for {path}") from exc
    if not isinstance(result, dict):
        raise OllamaError(f"Ollama returned an unexpected response for {path}")
    return result


def get_inventory(base_url: str) -> tuple[str, list[dict[str, Any]]]:
    version_data = request_json(base_url, "/api/version")
    tags_data = request_json(base_url, "/api/tags")
    version = str(version_data.get("version") or "unknown")
    models = tags_data.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama /api/tags response did not contain a models list")
    return version, [item for item in models if isinstance(item, dict)]


def find_model(models: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    matches = [
        item for item in models
        if reference in {item.get("name"), item.get("model"), item.get("digest")}
    ]
    if not matches:
        raise OllamaError(
            f"Model {reference!r} is not present in this Ollama instance. "
            "Run 'ollama pull MODEL:TAG' first."
        )
    if len(matches) > 1:
        raise OllamaError(
            f"Model reference {reference!r} is ambiguous; use its full name or digest"
        )
    return matches[0]


def inspect_model(base_url: str, model: dict[str, Any]) -> dict[str, Any]:
    reference = str(model.get("name") or model.get("model") or "")
    detail = request_json(base_url, "/api/show", {"model": reference, "verbose": True})
    return {
        "reference": reference,
        "name": model.get("name"),
        "digest": model.get("digest"),
        "modified_at": model.get("modified_at"),
        "size_bytes": model.get("size"),
        "details": model.get("details") or {},
        "show": {
            key: detail[key]
            for key in (
                "details", "model_info", "capabilities", "parameters",
                "template", "modelfile", "modified_at",
            )
            if key in detail
        },
    }


def catalog_path(models_dir: Path) -> Path:
    return models_dir / "catalog" / "model-catalog.json"


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaError(f"Cannot read model catalog {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise OllamaError(f"Model catalog has an unsupported structure: {path}")
    return data


def save_snapshot(base_url: str, models_dir: Path, reference: str) -> dict[str, Any]:
    runtime_version, inventory = get_inventory(base_url)
    selected = find_model(inventory, reference)
    inspected = inspect_model(base_url, selected)
    digest = inspected.get("digest")
    if not isinstance(digest, str) or not digest:
        raise OllamaError("Ollama did not provide a model digest; refusing to catalog an unpinned model")

    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"name": "ollama", "version": runtime_version, "base_url": base_url},
        "model": inspected,
    }
    path = catalog_path(models_dir)
    catalog = load_catalog(path)
    records = catalog["models"]
    # Preserve history, but avoid adding an identical snapshot repeatedly.
    identity = (digest, runtime_version)
    if not any(
        (entry.get("model", {}).get("digest"), entry.get("runtime", {}).get("version")) == identity
        for entry in records
        if isinstance(entry, dict)
    ):
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Ollama model identities and maintain the FlossWare model catalog."
    )
    parser.add_argument("--url", default=os.environ.get("OLLAMA_HOST", DEFAULT_URL))
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("OLLAMA_MODELS", str(DEFAULT_MODELS_DIR))).expanduser(),
        help="Ollama model root; catalog is stored beneath its catalog/ directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List locally available models and their digests")
    inspect = sub.add_parser("inspect", help="Show detailed metadata for one installed model")
    inspect.add_argument("model", help="Exact Ollama model reference, e.g. qwen3.5:2b")
    snapshot = sub.add_parser("snapshot", help="Record exact model and runtime identity in the catalog")
    snapshot.add_argument("model", help="Exact Ollama model reference, e.g. qwen3.5:2b")
    sub.add_parser("catalog", help="Print the saved FlossWare model catalog")
    args = parser.parse_args()

    try:
        if args.command == "list":
            version, models = get_inventory(args.url)
            print_json({
                "runtime": {"name": "ollama", "version": version},
                "models": [
                    {
                        "name": item.get("name"),
                        "digest": item.get("digest"),
                        "size_bytes": item.get("size"),
                        "modified_at": item.get("modified_at"),
                        "details": item.get("details") or {},
                    }
                    for item in models
                ],
            })
        elif args.command == "inspect":
            _, models = get_inventory(args.url)
            print_json(inspect_model(args.url, find_model(models, args.model)))
        elif args.command == "snapshot":
            print_json(save_snapshot(args.url, args.models_dir, args.model))
            print(f"Catalog: {catalog_path(args.models_dir)}", file=sys.stderr)
        elif args.command == "catalog":
            print_json(load_catalog(catalog_path(args.models_dir)))
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
