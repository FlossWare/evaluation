#!/usr/bin/env python3
"""Export a private Obsidian vault into an experimental retrieval corpus.

This research utility uses a deliberately simple whitespace-window baseline.
It has no runtime dependency on another FlossWare repository and makes no
provider requests. It never modifies the source vault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("kbase-corpus-export.json")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"\S+")
CHUNKING_STRATEGY = "whitespace-window-baseline-v1"


def note_title(content: str, relative_path: Path) -> str:
    match = TITLE_RE.search(content)
    return match.group(1).strip() if match else relative_path.stem


def chunk_text(
    content: str,
    *,
    document_id: str,
    max_tokens: int = 512,
    overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split text on whitespace-token boundaries while preserving source offsets.

    Token counts are approximate word counts, not model-tokenizer counts. This
    intentionally modest baseline exists for comparative research only.
    """
    if max_tokens < 1 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("require max_tokens > 0 and 0 <= overlap < max_tokens")

    matches = list(TOKEN_RE.finditer(content))
    chunks: list[dict[str, Any]] = []
    step = max_tokens - overlap
    for sequence, start_index in enumerate(range(0, len(matches), step)):
        window = matches[start_index:start_index + max_tokens]
        if not window:
            break
        start_offset = window[0].start()
        end_offset = window[-1].end()
        chunks.append({
            "id": f"{document_id}:{sequence}",
            "document_id": document_id,
            "sequence": sequence,
            "content": content[start_offset:end_offset],
            "token_count": len(window),
            "start_offset": start_offset,
            "end_offset": end_offset,
            "metadata": dict(metadata or {}),
            "provenance": {"chunking_strategy": CHUNKING_STRATEGY},
        })
        if start_index + max_tokens >= len(matches):
            break
    return chunks


def export_vault(
    vault: Path,
    *,
    max_tokens: int = 512,
    overlap: int = 50,
    exclude_dirs: tuple[str, ...] = (".obsidian", ".git", ".trash"),
) -> dict[str, Any]:
    """Create stable chunk records from Markdown notes without modifying the vault."""
    root = vault.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Vault directory does not exist: {root}")
    excluded = set(exclude_dirs)
    documents: list[dict[str, Any]] = []
    for source in sorted(root.rglob("*.md")):
        relative_parts = source.relative_to(root).parts
        if source.is_symlink() or any(
            part in excluded or part.startswith(".") for part in relative_parts
        ):
            continue
        relative = source.relative_to(root)
        content = source.read_text(encoding="utf-8")
        if not content.strip():
            continue
        relative_name = relative.as_posix()
        document_id = "note-" + hashlib.sha256(relative_name.encode("utf-8")).hexdigest()[:24]
        title = note_title(content, relative)
        chunks = chunk_text(
            content,
            document_id=document_id,
            max_tokens=max_tokens,
            overlap=overlap,
            metadata={"source": "local-kbase-export", "relative_path": relative_name, "title": title},
        )
        for chunk in chunks:
            documents.append({
                "id": chunk["id"],
                "text": chunk["content"],
                "metadata": {
                    "document_id": chunk["document_id"],
                    "sequence": chunk["sequence"],
                    "relative_path": relative_name,
                    "title": title,
                    "token_count": chunk["token_count"],
                    "start_offset": chunk["start_offset"],
                    "end_offset": chunk["end_offset"],
                    "chunking_strategy": CHUNKING_STRATEGY,
                },
            })
    return {
        "name": "kbase-corpus-export",
        "source": "private-kbase-local-export",
        "version": 1,
        "chunking_strategy": CHUNKING_STRATEGY,
        "documents": documents,
        "queries": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault", type=Path, required=True,
        help="Path to the local private kbase Obsidian vault.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Local output JSON; keep corpus exports out of Git.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--exclude-dir", action="append", default=[],
        help="Additional directory name to exclude; may be repeated.",
    )
    args = parser.parse_args()
    if args.max_tokens < 1 or args.overlap < 0 or args.overlap >= args.max_tokens:
        parser.error("require max-tokens > 0 and 0 <= overlap < max-tokens")
    try:
        data = export_vault(
            args.vault,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            exclude_dirs=(".obsidian", ".git", ".trash", *args.exclude_dir),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    if not data["documents"]:
        parser.error("no Markdown chunks found; check the vault path and exclusions")
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    note_count = len({item["metadata"]["document_id"] for item in data["documents"]})
    print(f"Exported {len(data['documents'])} chunks from {note_count} notes")
    print(f"Chunking strategy: {CHUNKING_STRATEGY} (approximate word counts)")
    print(f"Local corpus export: {output}")
    print("No provider requests were made. Queries and relevance judgments are not included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
