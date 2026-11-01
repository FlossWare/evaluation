"""Tests for the deterministic embedding retrieval benchmark."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.provider_api.embedding_benchmark import (
    embed_in_batches,
    embedding_batch_spec,
    extract_batch_vectors,
    load_dataset,
    ranking_metrics,
    select_account,
)


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_dataset_has_valid_gold_relevance(self):
        dataset = load_dataset()
        doc_ids = {item["id"] for item in dataset["documents"]}
        self.assertEqual(len(doc_ids), len(dataset["documents"]))
        self.assertEqual(len(dataset["queries"]), 8)
        self.assertTrue(all(set(item["relevant"]) <= doc_ids for item in dataset["queries"]))

    def test_dataset_rejects_unknown_relevance_id(self):
        bad = {
            "version": 1,
            "documents": [{"id": "d1", "text": "doc"}],
            "queries": [{"id": "q1", "text": "query", "relevant": ["missing"]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_external_dataset_metadata_and_validation(self):
        data = {
            "name": "flossware-corpus-v1",
            "source": "postgresql-export",
            "version": 1,
            "documents": [{"id": "chunk-1", "text": "A representative chunk."}],
            "queries": [{"id": "query-1", "text": "Find a representative chunk.", "relevant": ["chunk-1"]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_dataset(path)
        self.assertEqual(loaded["name"], "flossware-corpus-v1")
        self.assertEqual(loaded["source"], "postgresql-export")

    def test_dataset_rejects_empty_text_and_duplicate_query_ids(self):
        bad_datasets = [
            {
                "documents": [{"id": "d1", "text": "doc"}],
                "queries": [{"id": "q1", "text": " ", "relevant": ["d1"]}],
            },
            {
                "documents": [{"id": "d1", "text": "doc"}],
                "queries": [
                    {"id": "q1", "text": "query one", "relevant": ["d1"]},
                    {"id": "q1", "text": "query two", "relevant": ["d1"]},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, data in enumerate(bad_datasets):
                path = Path(directory) / f"bad-{index}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_embedding_batches_preserve_order_and_aggregate_metadata(self):
        with patch(
            "tools.provider_api.embedding_benchmark.post_embeddings",
            side_effect=[
                ([[1.0, 0.0], [0.0, 1.0]], {"latency_seconds": 0.2, "http_status": 200}),
                ([[-1.0, 0.0]], {"latency_seconds": 0.3, "http_status": 200}),
            ],
        ) as request:
            vectors, metadata = embed_in_batches(
                "COHERE", "embed-v4.0", "secret", ["one", "two", "three"],
                "document", 10.0, 2,
            )
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual(metadata["request_count"], 2)
        self.assertEqual(metadata["latency_seconds"], 0.5)
        self.assertEqual(request.call_count, 2)

    def test_provider_native_document_and_query_modes(self):
        url, payload = embedding_batch_spec(
            "COHERE", "embed-v4.0", ["text"], "document"
        )
        self.assertEqual(url, "https://api.cohere.com/v2/embed")
        self.assertEqual(payload["input_type"], "search_document")
        _, payload = embedding_batch_spec("COHERE", "embed-v4.0", ["text"], "query")
        self.assertEqual(payload["input_type"], "search_query")

        url, payload = embedding_batch_spec(
            "GOOGLE", "models/gemini-embedding-001", ["text"], "query"
        )
        self.assertTrue(url.endswith(":batchEmbedContents"))
        self.assertEqual(payload["requests"][0]["taskType"], "RETRIEVAL_QUERY")

        _, payload = embedding_batch_spec(
            "JINA", "jina-embeddings-v3", ["text"], "query"
        )
        self.assertEqual(payload["task"], "retrieval.query")

    def test_batch_response_envelopes(self):
        self.assertEqual(
            extract_batch_vectors("GOOGLE", {"embeddings": [{"values": [1, 0]}]}),
            [[1.0, 0.0]],
        )
        self.assertEqual(
            extract_batch_vectors("COHERE", {"embeddings": {"float": [[1, 0]]}}),
            [[1.0, 0.0]],
        )
        self.assertEqual(
            extract_batch_vectors("OPENAI", {"data": [{"embedding": [1, 0]}]}),
            [[1.0, 0.0]],
        )

    def test_batch_response_rejects_non_finite_or_inconsistent_vectors(self):
        with self.assertRaises(ValueError):
            extract_batch_vectors("OPENAI", {"data": [{"embedding": [1, float("nan")]}]})
        with self.assertRaises(ValueError):
            extract_batch_vectors("OPENAI", {
                "data": [{"embedding": [1, 0]}, {"embedding": [1]}]
            })

    def test_metrics_for_perfect_retrieval(self):
        docs = [[1, 0], [0, 1], [-1, 0]]
        queries = [[1, 0]]
        relevance = [{"id": "q1", "relevant": ["d1"]}]
        metrics = ranking_metrics(queries, docs, ["d1", "d2", "d3"], relevance, k=2)
        self.assertEqual(metrics["recall_at_2"], 1.0)
        self.assertEqual(metrics["mrr_at_2"], 1.0)
        self.assertEqual(metrics["ndcg_at_2"], 1.0)
        self.assertEqual(metrics["per_query"][0]["top_document_ids"][0], "d1")

    def test_metrics_penalize_lower_rank_and_missing_relevant_docs(self):
        docs = [[0, 1], [1, 0], [-1, 0]]
        queries = [[1, 0]]
        relevance = [{"id": "q1", "relevant": ["d1"]}]
        metrics = ranking_metrics(queries, docs, ["d1", "d2", "d3"], relevance, k=2)
        self.assertLess(metrics["mrr_at_2"], 1.0)
        self.assertEqual(metrics["recall_at_2"], 1.0)

    def test_metrics_reject_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            ranking_metrics([[1, 0]], [[1, 0, 0]], ["d1"], [{"id": "q", "relevant": ["d1"]}])

    def test_default_account_selection_is_single_preferred_account(self):
        with patch.dict("os.environ", {
            "COHERE_API_KEY_HOTMAIL": "hotmail-secret",
            "COHERE_API_KEY_FLOSSWARE": "flossware-secret",
        }, clear=True):
            accounts = select_account("COHERE")
            self.assertEqual(len(accounts), 1)
            self.assertEqual(accounts[0]["account"], "FLOSSWARE")
            self.assertEqual(accounts[0]["credential"], "flossware-secret")


if __name__ == "__main__":
    unittest.main()
