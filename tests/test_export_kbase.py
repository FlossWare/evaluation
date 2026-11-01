from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.provider_api.export_kbase import (
    CHUNKING_STRATEGY,
    chunk_text,
    export_vault,
)


class ChunkTextTests(unittest.TestCase):
    def test_chunks_preserve_text_offsets_and_overlap(self) -> None:
        source = "one two three four five six"
        chunks = chunk_text(
            source,
            document_id="doc-1",
            max_tokens=3,
            overlap=1,
        )

        self.assertEqual([item["id"] for item in chunks], ["doc-1:0", "doc-1:1"])
        self.assertEqual([item["content"] for item in chunks], [
            "one two three",
            "three four five",
        ])
        self.assertEqual([item["token_count"] for item in chunks], [3, 3])
        for item in chunks:
            self.assertEqual(
                source[item["start_offset"]:item["end_offset"]],
                item["content"],
            )
            self.assertEqual(
                item["provenance"]["chunking_strategy"],
                CHUNKING_STRATEGY,
            )

    def test_rejects_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("one two", document_id="doc-1", max_tokens=2, overlap=2)


class ExportVaultTests(unittest.TestCase):
    def test_exports_markdown_and_excludes_hidden_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = Path(temporary)
            (vault / "topic.md").write_text(
                "# Topic\n\nalpha beta gamma delta", encoding="utf-8"
            )
            (vault / ".obsidian").mkdir()
            (vault / ".obsidian" / "settings.md").write_text(
                "must not export", encoding="utf-8"
            )
            (vault / "subdir").mkdir()
            (vault / "subdir" / "empty.md").write_text("  \n", encoding="utf-8")

            result = export_vault(vault, max_tokens=3, overlap=1)

        self.assertEqual(result["chunking_strategy"], CHUNKING_STRATEGY)
        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(
            [item["text"] for item in result["documents"]],
            ["# Topic\n\nalpha beta", "beta gamma delta"],
        )
        self.assertTrue(all(
            item["metadata"]["relative_path"] == "topic.md"
            for item in result["documents"]
        ))

    def test_note_ids_are_stable_for_same_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = Path(temporary)
            note = vault / "topic.md"
            note.write_text("one two three", encoding="utf-8")
            first = export_vault(vault, max_tokens=10, overlap=0)
            second = export_vault(vault, max_tokens=10, overlap=0)

        self.assertEqual(
            [item["id"] for item in first["documents"]],
            [item["id"] for item in second["documents"]],
        )


if __name__ == "__main__":
    unittest.main()
