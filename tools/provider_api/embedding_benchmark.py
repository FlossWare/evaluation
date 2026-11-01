#!/usr/bin/env python3
"""Run a small, reproducible semantic retrieval benchmark across embedding APIs.

No vectors are written to disk. A run requires explicit --run consent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .smoke_test import (
        EMBEDDING_CASES,
        PAID_ACCESS,
        build_headers,
        classify_http_error,
        credential_accounts,
        error_summary,
        sanitize_error_detail,
    )
except ImportError:
    from smoke_test import (
        EMBEDDING_CASES,
        PAID_ACCESS,
        build_headers,
        classify_http_error,
        credential_accounts,
        error_summary,
        sanitize_error_detail,
    )

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "embedding-benchmark.json"
DEFAULT_OUTPUT = Path("embedding-benchmark-results.jsonl")
DEFAULT_PROVIDERS = ("COHERE", "GOOGLE")
TOP_K = 5


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Load and validate a benchmark dataset with binary relevance judgments."""
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    queries = data.get("queries", [])
    if not isinstance(documents, list) or not isinstance(queries, list):
        raise ValueError("Dataset documents and queries must be arrays")
    doc_ids = [item.get("id") for item in documents if isinstance(item, dict)]
    query_ids = [item.get("id") for item in queries if isinstance(item, dict)]
    if (
        not documents
        or not queries
        or len(doc_ids) != len(documents)
        or len(query_ids) != len(queries)
        or any(not isinstance(value, str) or not value.strip() for value in doc_ids + query_ids)
        or len(doc_ids) != len(set(doc_ids))
        or len(query_ids) != len(set(query_ids))
    ):
        raise ValueError("Dataset must contain non-empty unique document and query IDs")
    if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in documents + queries):
        raise ValueError("Every document and query must contain non-empty text")
    known = set(doc_ids)
    for query in queries:
        relevant = query.get("relevant", [])
        if (
            not isinstance(relevant, list)
            or not relevant
            or any(not isinstance(value, str) for value in relevant)
            or not set(relevant).issubset(known)
        ):
            raise ValueError(f"Query {query.get('id')} has missing or unknown relevance IDs")
    data.setdefault("name", path.stem)
    data.setdefault("source", "local-dataset")
    data.setdefault("version", 1)
    return data


def embedding_batch_spec(provider: str, model: str, texts: list[str], mode: str) -> tuple[str, dict[str, Any]]:
    """Build a provider-native batch request for document or query embeddings."""
    if mode not in {"document", "query"}:
        raise ValueError(f"Unsupported embedding mode: {mode}")
    if provider == "GOOGLE":
        task = "RETRIEVAL_DOCUMENT" if mode == "document" else "RETRIEVAL_QUERY"
        return (
            f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents",
            {"requests": [
                {"model": model, "content": {"parts": [{"text": text}]}, "taskType": task}
                for text in texts
            ]},
        )
    if provider == "COHERE":
        return (
            "https://api.cohere.com/v2/embed",
            {
                "model": model,
                "texts": texts,
                "input_type": "search_document" if mode == "document" else "search_query",
                "embedding_types": ["float"],
            },
        )
    if provider == "VOYAGEAI":
        return (
            "https://api.voyageai.com/v1/embeddings",
            {
                "model": model,
                "input": texts,
                "input_type": mode,
                "output_dtype": "float",
            },
        )
    if provider == "JINA":
        return (
            "https://api.jina.ai/v1/embeddings",
            {
                "model": model,
                "input": texts,
                "task": "retrieval.passage" if mode == "document" else "retrieval.query",
                "embedding_type": "float",
                "normalized": True,
            },
        )
    if provider in {"OPENAI", "MISTRAL"}:
        endpoint = (
            "https://api.openai.com/v1/embeddings"
            if provider == "OPENAI"
            else "https://api.mistral.ai/v1/embeddings"
        )
        return endpoint, {"model": model, "input": texts, "encoding_format": "float"}
    raise ValueError(f"No embedding batch adapter for provider: {provider}")


def extract_batch_vectors(provider: str, data: dict[str, Any]) -> list[list[float]]:
    """Normalize provider batch responses and validate every vector."""
    if provider == "GOOGLE":
        raw = [item.get("values") for item in data.get("embeddings", []) if isinstance(item, dict)]
    elif provider == "COHERE":
        raw = (data.get("embeddings") or {}).get("float") or []
    else:
        items = data.get("data", [])
        raw = [
            item.get("embedding") for item in items
            if isinstance(item, dict) and item.get("embedding") is not None
        ]
    vectors: list[list[float]] = []
    for vector in raw:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Provider returned an empty or invalid embedding")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
            raise ValueError("Provider returned a non-numeric embedding value")
        normalized = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError("Provider returned a non-finite embedding value")
        if not any(value != 0.0 for value in normalized):
            raise ValueError("Provider returned an all-zero embedding")
        vectors.append(normalized)
    if not vectors:
        raise ValueError("Provider response contained no embeddings")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Provider returned embeddings with inconsistent dimensions")
    return vectors


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare vectors with different dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return dot / (left_norm * right_norm)


def ranking_metrics(
    query_vectors: list[list[float]],
    document_vectors: list[list[float]],
    document_ids: list[str],
    queries: list[dict[str, Any]],
    k: int = TOP_K,
) -> dict[str, Any]:
    """Calculate macro Recall@K, MRR@K, and binary nDCG@K."""
    if len(query_vectors) != len(queries) or len(document_vectors) != len(document_ids):
        raise ValueError("Vector counts do not match benchmark records")
    dimensions = {len(vector) for vector in query_vectors + document_vectors}
    if len(dimensions) != 1:
        raise ValueError("Query and document vector dimensions do not match")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    details: list[dict[str, Any]] = []
    for query, vector in zip(queries, query_vectors):
        ranked = sorted(
            zip(document_ids, (cosine(vector, doc) for doc in document_vectors)),
            key=lambda item: (-item[1], item[0]),
        )
        relevant = set(query["relevant"])
        top = ranked[:k]
        hits = [rank for rank, (doc_id, _) in enumerate(ranked[:k], start=1) if doc_id in relevant]
        recalls.append(len({doc_id for doc_id, _ in top} & relevant) / len(relevant))
        reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        details.append({
            "query_id": query["id"],
            "relevant_count": len(relevant),
            "retrieved_relevant_at_k": len(hits),
            "reciprocal_rank_at_k": round(reciprocal_ranks[-1], 6),
            "ndcg_at_k": round(ndcgs[-1], 6),
            "top_document_ids": [doc_id for doc_id, _ in top],
        })
    return {
        f"recall_at_{k}": round(sum(recalls) / len(recalls), 6),
        f"mrr_at_{k}": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        f"ndcg_at_{k}": round(sum(ndcgs) / len(ndcgs), 6),
        "query_count": len(queries),
        "document_count": len(document_ids),
        "per_query": details,
    }


def post_embeddings(
    provider: str,
    model: str,
    key: str,
    texts: list[str],
    mode: str,
    timeout: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    url, payload = embedding_batch_spec(provider, model, texts, mode)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_headers(provider, key),
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            metadata = {
                "http_status": response.status,
                "latency_seconds": round(time.monotonic() - started, 3),
                "request_id": response.headers.get("x-request-id") or response.headers.get("request-id"),
                "usage": body.get("usage") or body.get("usageMetadata"),
            }
        vectors = extract_batch_vectors(provider, body)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} vectors for {mode}, received {len(vectors)}"
            )
        return vectors, metadata
    except urllib.error.HTTPError as exc:
        detail = sanitize_error_detail(exc.read().decode("utf-8", errors="replace"), key)
        raise ProviderRequestError(exc.code, detail, time.monotonic() - started) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderRequestError(None, str(reason), time.monotonic() - started) from exc


def embed_in_batches(
    provider: str,
    model: str,
    key: str,
    texts: list[str],
    mode: str,
    timeout: float,
    batch_size: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Embed a larger corpus in bounded batches without persisting vectors."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    vectors: list[list[float]] = []
    requests: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch_vectors, metadata = post_embeddings(
            provider, model, key, texts[start:start + batch_size], mode, timeout
        )
        vectors.extend(batch_vectors)
        requests.append(metadata)
    return vectors, {
        "request_count": len(requests),
        "latency_seconds": round(sum(item["latency_seconds"] for item in requests), 3),
        "requests": requests,
    }


class ProviderRequestError(RuntimeError):
    def __init__(self, status: int | None, detail: str, elapsed: float):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.elapsed = elapsed


def select_account(provider: str, all_accounts: bool = False) -> list[dict[str, str]]:
    accounts = credential_accounts(provider)
    if all_accounts:
        return accounts
    return accounts[:1]


def run_provider(
    provider: str,
    case: dict[str, str],
    account: dict[str, str],
    dataset: dict[str, Any],
    timeout: float,
    batch_size: int,
) -> dict[str, Any]:
    key = account["credential"]
    docs = dataset["documents"]
    queries = dataset["queries"]
    evidence: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "embedding-semantic-retrieval-v1",
        "provider": provider,
        "account": account["account"],
        "credential_env": account["credential_env"],
        "model": case["model"],
        "access_classification": case["access"],
        "dataset_name": dataset["name"],
        "dataset_source": dataset["source"],
        "dataset_version": dataset["version"],
        "top_k": TOP_K,
        "batch_size": batch_size,
        "document_count": len(docs),
        "query_count": len(queries),
        "status": "failed",
        "retry_performed": False,
        "vectors_persisted": False,
    }
    try:
        document_vectors, document_meta = embed_in_batches(
            provider, case["model"], key, [item["text"] for item in docs],
            "document", timeout, batch_size
        )
        query_vectors, query_meta = embed_in_batches(
            provider, case["model"], key, [item["text"] for item in queries],
            "query", timeout, batch_size
        )
        metrics = ranking_metrics(
            query_vectors, document_vectors, [item["id"] for item in docs], queries
        )
        dimensions = len(document_vectors[0])
        evidence.update({
            "status": "passed",
            "dimensions": dimensions,
            "document_request": document_meta,
            "query_request": query_meta,
            "total_latency_seconds": round(
                document_meta["latency_seconds"] + query_meta["latency_seconds"], 3
            ),
            "metrics": metrics,
        })
    except ProviderRequestError as exc:
        evidence.update({
            "status": "http-error" if exc.status is not None else "transport-error",
            "http_status": exc.status,
            "latency_seconds": round(exc.elapsed, 3),
            "http_error_category": classify_http_error(exc.status, exc.detail),
            "error": exc.detail,
        })
    except (ValueError, KeyError, TypeError) as exc:
        evidence.update({"status": "invalid-response", "error": str(exc)})
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=sorted(
        {item["provider"] for item in EMBEDDING_CASES}
    ), help="Provider to benchmark; repeat to select several. Defaults to Cohere and Google.")
    parser.add_argument("--run", action="store_true", help="Execute provider requests. Without this flag, only print the plan.")
    parser.add_argument("--include-paid", action="store_true", help="Explicitly allow paid/account-credit candidates.")
    parser.add_argument("--all-accounts", action="store_true", help="Run once per configured account instead of only the preferred account.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--batch-size", type=int, default=64, help="Maximum texts per provider request.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Benchmark JSON file; supports local corpus exports.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        dataset = load_dataset(args.dataset)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid dataset {args.dataset}: {exc}")
    selected_providers = args.provider or list(DEFAULT_PROVIDERS)
    cases = {item["provider"]: item for item in EMBEDDING_CASES}
    plans: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for provider in selected_providers:
        case = cases[provider]
        if case["access"] in PAID_ACCESS and not args.include_paid:
            print(f"{provider:10} {case['model']:32} skipped-paid-opt-in-required")
            continue
        accounts = select_account(provider, args.all_accounts)
        if not accounts:
            print(f"{provider:10} {'-':32} skipped-no-credential")
            continue
        for account in accounts:
            plans.append((provider, case, account))
            print(
                f"{provider:10} {account['account']:10} {case['model']:32} "
                f"planned ({len(dataset['documents'])} docs + {len(dataset['queries'])} queries; "
                f"{math.ceil(len(dataset['documents']) / args.batch_size) + math.ceil(len(dataset['queries']) / args.batch_size)} batch requests)"
            )
    if not args.run:
        print("Planning only: no provider requests were made. Add --run to execute.")
        return 0
    if not plans:
        print("No eligible configured provider/account combinations.", file=sys.stderr)
        return 2
    results = [
        run_provider(provider, case, account, dataset, args.timeout, args.batch_size)
        for provider, case, account in plans
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    for item in results:
        print(
            f"{item['provider']:10} {item['account']:10} {item['model']:32} "
            f"{item['status']:18} dimensions={item.get('dimensions', '-')}"
        )
        if item.get("metrics"):
            metrics = item["metrics"]
            print(
                f"  Recall@{TOP_K}={metrics[f'recall_at_{TOP_K}']:.3f} "
                f"MRR@{TOP_K}={metrics[f'mrr_at_{TOP_K}']:.3f} "
                f"nDCG@{TOP_K}={metrics[f'ndcg_at_{TOP_K}']:.3f} "
                f"latency={item['total_latency_seconds']}s"
            )
        elif item.get("error"):
            print(f"  detail [{item.get('http_error_category') or 'provider-error'}]: {error_summary(item['error'])}")
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
