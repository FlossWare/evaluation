from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_entities", ROOT / "scripts" / "extract-entities.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixture_cases_load():
    cases = MODULE.load_cases(ROOT / "tests" / "fixtures" / "entity-extraction-cases.jsonl")
    assert len(cases) >= 5
    assert all("expected_entities" in case and "expected_relations" in case for case in cases)


def test_valid_extraction_requires_exact_offsets_and_evidence():
    source = "Apache Solr stores search documents."
    valid = {
        "entities": [{
            "text": "Apache Solr", "type": "SOFTWARE",
            "canonical_name": "Apache Solr", "attributes": {}, "start": 0, "end": 11
        }],
        "relations": [{
            "subject": "Apache Solr", "predicate": "stores",
            "object": "search documents", "evidence": "Apache Solr stores search documents"
        }]
    }
    assert MODULE.parse_output(__import__("json").dumps(valid), source) == valid
    valid["entities"][0]["start"] = 1
    assert MODULE.parse_output(__import__("json").dumps(valid), source) is None


def test_entity_and_relation_metrics():
    expected_entities = [{"text": "Redis", "type": "DATABASE", "canonical_name": "Redis", "attributes": {}}]
    expected_relations = [{"subject": "Redis", "predicate": "used_for", "object": "caching"}]
    actual = {
        "entities": [{
            "text": "Redis", "type": "DATABASE", "canonical_name": "Redis",
            "attributes": {}, "start": 0, "end": 5
        }],
        "relations": [{
            "subject": "Redis", "predicate": "used_for", "object": "caching",
            "evidence": "Redis is used for caching"
        }]
    }
    metrics = MODULE.score(expected_entities, expected_relations, actual)
    assert metrics["entities"]["f1"] == 1.0
    assert metrics["relations"]["f1"] == 1.0
    assert metrics["canonical_name_accuracy"] == 1.0


def test_malformed_or_unsupported_output_is_rejected():
    assert MODULE.parse_output('{"entities":[],"relations":[]}', "nothing") == {
        "entities": [], "relations": []
    }
    assert MODULE.parse_output('{"entities":[{"text":"X"}],"relations":[]}', "X") is None
