"""Compare exact NumPy search, argpartition top-K, and tuned Chroma HNSW.

Uses synthetic normalized vectors and temporary Chroma collections only. Exact search is
used as ground truth for ANN recall. Run: python3 benchmarks/vector/run.py --help
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import package_versions, summarize_ms, write_result


def exact(matrix, query, k):
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]


def partial(matrix, query, k):
    scores = matrix @ query
    if k >= len(scores):
        candidates = np.arange(len(scores))
    else:
        candidates = np.argpartition(-scores, k - 1)[:k]
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order, scores[order]


def timed_searches(matrix, queries, collection, k, warmup_queries, rounds, seed):
    """Warm each path, then measure repeated rounds with deterministic query order."""
    warmup_count = min(warmup_queries, len(queries))
    for query in queries[:warmup_count]:
        exact(matrix, query, k)
        partial(matrix, query, k)
        collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            include=["distances"],
        )

    exact_samples, partial_samples, hnsw_samples = [], [], []
    recalls, agreements, round_results = [], [], []
    rng = np.random.default_rng(seed)

    for round_index in range(rounds):
        order = rng.permutation(len(queries))
        round_exact, round_partial, round_hnsw = [], [], []
        round_recalls, round_agreements = [], []

        for query_index in order:
            query = queries[query_index]

            started = time.perf_counter()
            exact_ids, _ = exact(matrix, query, k)
            exact_ms = (time.perf_counter() - started) * 1000
            round_exact.append(exact_ms)

            started = time.perf_counter()
            partial_ids, _ = partial(matrix, query, k)
            partial_ms = (time.perf_counter() - started) * 1000
            round_partial.append(partial_ms)
            round_agreements.append(
                len(set(exact_ids.tolist()) & set(partial_ids.tolist())) / k
            )

            started = time.perf_counter()
            hits = collection.query(
                query_embeddings=[query.tolist()],
                n_results=k,
                include=["distances"],
            )
            hnsw_ms = (time.perf_counter() - started) * 1000
            round_hnsw.append(hnsw_ms)
            ann = {int(value[1:]) for value in hits["ids"][0]}
            round_recalls.append(
                len(set(exact_ids.tolist()) & ann) / k
            )

        exact_samples.extend(round_exact)
        partial_samples.extend(round_partial)
        hnsw_samples.extend(round_hnsw)
        recalls.extend(round_recalls)
        agreements.extend(round_agreements)
        round_results.append(
            {
                "round": round_index + 1,
                "exact_numpy": summarize_ms(round_exact),
                "argpartition": summarize_ms(round_partial),
                "chroma_hnsw": summarize_ms(round_hnsw),
                "hnsw_recall_at_k": float(np.mean(round_recalls)),
                "hnsw_recall_min": float(np.min(round_recalls)),
                "argpartition_topk_agreement": float(np.mean(round_agreements)),
            }
        )

    return {
        "warmup_queries": warmup_count,
        "measurement_rounds": rounds,
        "measured_queries_per_round": len(queries),
        "exact_numpy": summarize_ms(exact_samples),
        "argpartition": summarize_ms(partial_samples),
        "chroma_hnsw": summarize_ms(hnsw_samples),
        "argpartition_topk_agreement": float(np.mean(agreements)),
        "hnsw_recall_at_k": float(np.mean(recalls)),
        "hnsw_recall_min": float(np.min(recalls)),
        "rounds": round_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 10000],
        help="include 100000 for the larger run",
    )
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="HNSW search effort values to compare; each value gets a fresh index",
    )
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=20,
        help="queries per index/path excluded from timing (default: 20)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="independent measured passes over the query set (default: 3)",
    )
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    if (
        min(args.sizes) < 1
        or args.dimensions < 1
        or args.queries < 1
        or min(args.top_k) < 1
        or min(args.ef_search) < 1
        or args.ef_construction < 1
        or args.max_neighbors < 1
        or args.warmup_queries < 0
        or args.rounds < 1
    ):
        parser.error("sizes, dimensions, queries, top-k, HNSW settings, and rounds "
                     "must be positive; warmup-queries may be zero")

    try:
        import chromadb
    except ImportError:
        parser.error("ChromaDB is required for the HNSW comparison")

    rng = np.random.default_rng(args.seed)
    all_cases = []

    for size in args.sizes:
        matrix = rng.normal(size=(size, args.dimensions)).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        queries = rng.normal(size=(args.queries, args.dimensions)).astype(np.float32)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

        with tempfile.TemporaryDirectory(prefix="poc-ce-eval-chroma-") as temp:
            client = chromadb.PersistentClient(path=temp)

            for ef_search in args.ef_search:
                collection = client.create_collection(
                    name=f"eval_vectors_ef{ef_search}",
                    configuration={
                        "hnsw": {
                            "space": "cosine",
                            "ef_search": ef_search,
                            "ef_construction": args.ef_construction,
                            "max_neighbors": args.max_neighbors,
                        }
                    },
                )
                ids = [f"v{i}" for i in range(size)]
                build_start = time.perf_counter()
                for start in range(0, size, 500):
                    end = min(start + 500, size)
                    collection.add(
                        ids=ids[start:end],
                        embeddings=matrix[start:end].tolist(),
                    )
                build_ms = (time.perf_counter() - build_start) * 1000

                for k in args.top_k:
                    k = min(k, size)
                    measurements = timed_searches(
                        matrix=matrix,
                        queries=queries,
                        collection=collection,
                        k=k,
                        warmup_queries=args.warmup_queries,
                        rounds=args.rounds,
                        seed=args.seed + size + ef_search + k,
                    )
                    all_cases.append(
                        {
                            "vectors": size,
                            "dimensions": args.dimensions,
                            "top_k": k,
                            "queries": args.queries,
                            "hnsw_ef_search": ef_search,
                            "hnsw_ef_construction": args.ef_construction,
                            "hnsw_max_neighbors": args.max_neighbors,
                            "hnsw_build_ms": build_ms,
                            **measurements,
                        }
                    )

    result = write_result(
        "vector",
        {
            "seed": args.seed,
            "runtime": {"packages": package_versions(["numpy", "chromadb"])},
            "cases": all_cases,
        },
        args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
