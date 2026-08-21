#!/usr/bin/env python3
"""Claude Code hook: flag critical files for adversarial verification.

Usage as a post-tool-edit hook in .claude/hooks/post-tool-edit.py:
    python3 examples/claude_code_hook.py "$FILE_PATH"
"""
from __future__ import annotations

import sys

CRITICAL_PATTERNS = [
    "auth", "security", "payment", "credential", "secret",
    "password", "token", "encrypt", "decrypt", "permission",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: claude_code_hook.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    lower_path = file_path.lower()

    matches = [p for p in CRITICAL_PATTERNS if p in lower_path]
    if not matches:
        sys.exit(0)

    print(f"[evaluation-ai] CRITICAL FILE MODIFIED: {file_path}")
    print(f"[evaluation-ai] Matched patterns: {', '.join(matches)}")
    print("[evaluation-ai] Recommend adversarial verification before commit:")
    print("  from evaluation_ai import AdversarialVerifier, select_panel")
    print("  verifier = AdversarialVerifier(backend, models)")
    print('  result = await verifier.verify(claim="change is safe", evidence=diff)')


if __name__ == "__main__":
    main()
