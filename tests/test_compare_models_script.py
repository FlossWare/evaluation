"""Tests for the manual model comparison runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare-models.py"
SPEC = importlib.util.spec_from_file_location("compare_models_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_models)


def test_extract_json_accepts_expected_judgment():
    raw = (
        '{"outcome":"unknown","confidence":0.8,'
        '"evidence":[],"rationale":"No user feedback is present."}'
    )
    parsed = compare_models.extract_json(raw)
    assert parsed is not None
    assert parsed["outcome"] == "unknown"


def test_extract_json_accepts_markdown_fence():
    fence = chr(96) * 3
    raw = fence + 'json\n{"outcome":"success","confidence":0.9,"evidence":["Thanks"],"rationale":"Clear completion."}\n' + fence
    parsed = compare_models.extract_json(raw)
    assert parsed is not None
    assert parsed["outcome"] == "success"


def test_extract_json_rejects_invalid_confidence_and_schema():
    assert compare_models.extract_json(
        '{"outcome":"success","confidence":2,"evidence":[],"rationale":"x"}'
    ) is None
    assert compare_models.extract_json(
        '{"outcome":"success","confidence":0.5,"evidence":"not a list","rationale":"x"}'
    ) is None


def test_expected_label_is_not_sent_to_model():
    case = {
        "id": "private-label",
        "expected_outcome": "failure",
        "transcript": [{"role": "user", "content": "Do the task"}],
    }
    messages = compare_models.make_messages("Judge carefully.", case)
    combined = "\n".join(message.content for message in messages)
    assert "expected_outcome" not in combined
    assert "failure" not in combined
    assert "Do the task" in combined
