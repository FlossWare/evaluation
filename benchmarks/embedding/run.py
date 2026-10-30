"""Reproducible embedding backend benchmark.

Run from the repository root:
  python3 benchmarks/embedding/run.py --backend onnxruntime-cpu
  python3 benchmarks/embedding/run.py --backend openvino-gpu --variant model_fp16.onnx

Model artifacts are read locally; this runner never downloads models and never
imports or changes the live embedding service implementation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import package_versions, summarize_ms, write_result

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
MODEL_DIR = Path(os.path.expanduser(os.environ.get(
    "ENSEMBLE_EMBEDDING_MODEL_DIR",
    "~/.FlossWare/models/embeddings/nomic-ai/nomic-embed-text-v1.5/" + MODEL_REVISION,
)))
DIMENSIONS = 768
MAX_TOKENS = 512
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
TEXTS = [
    "How do I renew the TLS certificate on the load balancer?",
    "The JSON parser rejects trailing commas and reports the offending byte offset.",
    "The vector cache doubles its NumPy buffer capacity when the number of rows reaches the current allocation.",
    "Chunk IDs include the profile name and version, source ID, and character offsets.",
    "SQLite FTS5 provides lexical full-text search over indexed transcript text.",
    "The embedding service converts query and document text into normalized vectors.",
    "OpenVINO can execute compatible ONNX graphs on supported CPU and GPU devices.",
    "Reciprocal rank fusion combines ranked lists without requiring score calibration.",
]


class Encoder:
    def __init__(self, backend: str, graph: Path, threads: int, performance_hint: str):
        self.backend = backend
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.input_names: set[str]
        self.output = None
        self.session = None
        if backend == "onnxruntime-cpu":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            self.session = ort.InferenceSession(
                str(graph), options, providers=["CPUExecutionProvider"]
            )
            self.input_names = {item.name for item in self.session.get_inputs()}
            self.providers = self.session.get_providers()
            self.runtime_name = "onnxruntime"
        else:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(str(graph))
            self.compiled = core.compile_model(
                model, "GPU" if backend == "openvino-gpu" else "CPU",
                {"PERFORMANCE_HINT": performance_hint},
            )
            self.input_names = {item.get_any_name() for item in self.compiled.inputs}
            self.output = self.compiled.output(0)
            self.providers = [self.compiled.get_property("EXECUTION_DEVICES")[0]]
            self.runtime_name = "openvino"

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        width = max(len(item.ids) for item in encodings)
        ids = np.zeros((len(texts), width), dtype=np.int64)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, item in enumerate(encodings):
            ids[row, :len(item.ids)] = item.ids
            mask[row, :len(item.ids)] = item.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        if self.backend == "onnxruntime-cpu":
            hidden = self.session.run(None, feed)[0]
        else:
            hidden = self.compiled(feed)[self.output]
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden.astype(np.float32) * weights).sum(axis=1)
        pooled /= np.maximum(weights.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([DOC_PREFIX + text for text in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([QUERY_PREFIX + text for text in texts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["onnxruntime-cpu", "openvino-cpu", "openvino-gpu"],
                        default="onnxruntime-cpu")
    parser.add_argument("--variant", help="ONNX graph filename; defaults by backend")
    parser.add_argument("--threads", type=int, default=8, help="ONNX Runtime CPU intra-op threads")
    parser.add_argument("--performance-hint", choices=["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
                        default="THROUGHPUT", help="OpenVINO performance hint")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 0 or args.iterations < 1 or not args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("threads and batch sizes must be positive; warmups must be non-negative; iterations must be positive")

    default_graph = "model_fp16.onnx" if args.backend == "openvino-gpu" else "model.onnx"
    graph = MODEL_DIR / "onnx" / (args.variant or default_graph)
    if not graph.is_file():
        parser.error(f"local model graph not found: {graph}; evaluations never download models")
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not tokenizer_path.is_file():
        parser.error(f"local tokenizer not found: {tokenizer_path}")

    started = time.perf_counter()
    encoder = Encoder(args.backend, graph, args.threads, args.performance_hint)
    load_ms = (time.perf_counter() - started) * 1000

    cases = []
    for batch_size in args.batch_sizes:
        texts = (TEXTS * ((batch_size + len(TEXTS) - 1) // len(TEXTS)))[:batch_size]
        for _ in range(args.warmups):
            encoder.embed_documents(texts)
        samples = []
        vectors = None
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            vectors = encoder.embed_documents(texts)
            samples.append((time.perf_counter() - t0) * 1000)
        norms = np.linalg.norm(vectors, axis=1)
        cases.append({
            "batch_size": batch_size,
            "timing": summarize_ms(samples),
            "texts_per_second": batch_size / (sum(samples) / 1000 / len(samples)),
            "dimensions": int(vectors.shape[1]),
            "finite": bool(np.isfinite(vectors).all()),
            "mean_l2_norm": float(norms.mean()),
            "min_l2_norm": float(norms.min()),
            "max_l2_norm": float(norms.max()),
        })

    query_vectors = encoder.embed_queries(TEXTS[:4])
    doc_vectors = encoder.embed_documents(TEXTS)
    similarities = query_vectors @ doc_vectors.T
    correct_top1 = sum(int(similarities[i].argmax() == i) for i in range(4))

    if args.backend != "onnxruntime-cpu":
        reference_graph = MODEL_DIR / "onnx" / "model.onnx"
        if not reference_graph.is_file():
            parser.error(f"CPU FP32 reference graph not found: {reference_graph}")
        reference_encoder = Encoder("onnxruntime-cpu", reference_graph, args.threads, args.performance_hint)
        candidate = encoder.embed_documents(TEXTS)
        baseline = reference_encoder.embed_documents(TEXTS)
        cosine = np.sum(candidate * baseline, axis=1)
        similarity_check = {
            "reference_backend": "onnxruntime-cpu",
            "reference_graph": str(reference_graph),
            "mean_cosine_similarity": float(cosine.mean()),
            "min_cosine_similarity": float(cosine.min()),
            "max_cosine_similarity": float(cosine.max()),
            "per_text_cosine_similarity": cosine.tolist(),
        }
    else:
        similarity_check = None

    runtime_packages = ["numpy", "tokenizers"]
    runtime_packages.append("onnxruntime" if args.backend == "onnxruntime-cpu" else "openvino")
    result = write_result("embedding", {
        "model": {
            "id": MODEL_ID, "revision": MODEL_REVISION, "directory": str(MODEL_DIR),
            "graph": str(graph), "graph_bytes": graph.stat().st_size, "dimensions": DIMENSIONS,
        },
        "configuration": {
            "backend": args.backend,
            "device": "CPU" if args.backend.endswith("cpu") else "GPU",
            "precision": "FP16" if "fp16" in graph.name.lower() else "FP32",
            "performance_hint": args.performance_hint if args.backend.startswith("openvino") else None,
        },
        "runtime": {
            "name": encoder.runtime_name, "providers": encoder.providers,
            "threads": args.threads if args.backend == "onnxruntime-cpu" else None,
            "packages": package_versions(runtime_packages),
        },
        "load_ms": load_ms, "warmups": args.warmups, "iterations": args.iterations,
        "batch_cases": cases,
        "semantic_smoke": {
            "queries": 4, "correct_top1": correct_top1,
            "similarity_matrix": similarities.tolist(),
        },
        "fp32_reference_comparison": similarity_check,
    }, args.output)
    print(result)


if __name__ == "__main__":
    main()
