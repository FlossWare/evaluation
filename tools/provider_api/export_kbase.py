#!/usr/bin/env python3
"""Export a private Obsidian kbase vault into a local embedding benchmark corpus.

The exporter reads Markdown files and uses FlossWare's canonical chunking
capability from a sibling checkout. It never calls a provider or writes vault
content into this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_CHUNKING_REPO = Path(__file__).resolve().parents[3] / "chunking"
DEFAULT_OUTPUT = Path("kbase-corpus-export.json")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def load_chunker(chunking_repo: Path):
    """Load TokenChunker from a local FlossWare/chunking checkout."""
    package_parent = chunking_repo.expanduser().resolve().parent
    if not (package_parent / "chunking" / "chunker.py").is_file():
        raise FileNotFoundError(
            f"Cannot find chunking package under {package_parent}; "
            "provide --chunking-repo pointing to the FlossWare/chunking checkout"
        )
    sys.path.insert(0, str(package_parent))
    try:
        from chunking import TokenChunker
    except ImportError as exc:
        raise RuntimeError(f"Could not import TokenChunker from {package_parent}") from exc
    return TokenChunker


def note_title(content: str, relative_path: Path) -> str:
    match = TITLE_RE.search(content)
    return match.group(1).strip() if match else relative_path.stem


def export_vault(
    vault: Path,
    chunker: Any,
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
        if source.is_symlink() or any(part in excluded or part.startswith(".") for part in source.relative_to(root).parts):
            continue
        relative = source.relative_to(root)
        content = source.read_text(encoding="utf-8")
        if not content.strip():
            continue
        relative_name = relative.as_posix()
        document_id = "note-" + hashlib.sha256(relative_name.encode("utf-8")).hexdigest()[:24]
        title = note_title(content, relative)
        chunks = chunker.chunk(
            content,
            document_id=document_id,
            max_tokens=max_tokens,
            overlap=overlap,
            metadata={"source": "kbase", "relative_path": relative_name, "title": title},
            provenance={"repository": "FlossWare/kbase", "format": "obsidian-markdown"},
        )
        for chunk in chunks:
            documents.append({
                "id": chunk.id,
                "text": chunk.content,
                "metadata": {
                    "document_id": chunk.document_id,
                    "sequence": chunk.sequence,
                    "relative_path": relative_name,
                    "title": title,
                    "token_count": chunk.token_count,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                },
            })
    return {
        "name": "flossware-kbase-corpus",
        "source": "private-kbase-local-export",
        "version": 1,
        "documents": documents,
        "queries": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True, help="Path to the local private kbase Obsidian vault.")
    parser.add_argument("--chunking-repo", type=Path, default=DEFAULT_CHUNKING_REPO, help="Local FlossWare/chunking checkout.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Local output JSON; keep private corpus exports out of Git.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--exclude-dir", action="append", default=[], help="Additional directory name to exclude; may be repeated.")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.overlap < 0 or args.overlap >= args.max_tokens:
        parser.error("require max-tokens > 0 and 0 <= overlap < max-tokens")
    try:
        TokenChunker = load_chunker(args.chunking_repo)
        data = export_vault(
            args.vault,
            TokenChunker(),
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            exclude_dirs=(".obsidian", ".git", ".trash", *args.exclude_dir),
        )
    except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if not data["documents"]:
        parser.error("no Markdown chunks found; check the vault path and exclusions")
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(f"Exported {len(data['documents'])} chunks from {len({item['metadata']['document_id'] for item in data['documents']})} notes")
    print(f"Local corpus export: {output}")
    print("No provider requests were made. Queries and relevance judgments are not included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
