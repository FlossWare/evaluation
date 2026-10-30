"""Isolated end-to-end ingestion benchmark for the POC REST services.

The runner starts private Index, Knowledge, Chunking, and Vector service
processes with temporary SQLite/Chroma stores. It supplies the embedding
service contract from this repository's benchmark Encoder so CPU FP32 and
OpenVINO GPU FP16 can be compared without changing the POC implementation.

Run from the evaluation repository root:
  python3 benchmarks/ingestion/run.py --backends onnxruntime-cpu openvino-gpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import package_versions, resolve_sut_root, write_result
from embedding.run import DIMENSIONS, MODEL_DIR, MODEL_ID, MODEL_REVISION, Encoder

COLLECTION_PREFIX = "evaluation_knowledge_chunks"
CHUNK_PROFILE = "window-v1"
CHUNK_PROFILE_VERSION = "window-v1@1"
DOCS = [
    {
        "title": "TLS certificate rotation runbook",
        "topic": "TLS certificate rotation on the edge load balancer",
        "content": (
            "TLS certificate rotation requires creating a new key pair, obtaining a "
            "replacement certificate, validating the full certificate chain, and staging "
            "the certificate on the load balancer before switching listeners. The runbook "
            "checks the Subject Alternative Names, issuer, validity window, key algorithm, "
            "and private-key permissions. A health probe verifies the new certificate from "
            "an independent client before the old certificate is retired."
        ),
        "query": "How does the TLS certificate rotation runbook validate a replacement certificate?",
    },
    {
        "title": "SQLite manifest revision ordering",
        "topic": "SQLite manifest revision ordering and tombstones",
        "content": (
            "The SQLite manifest records the highest accepted revision sequence for each "
            "source. A later revision may replace an earlier revision, an identical revision "
            "is idempotent, and a different revision at the same sequence is a conflict. "
            "Tombstones prevent deleted sources from being reintroduced by delayed workers. "
            "Every update is committed transactionally so concurrent ingestion workers observe "
            "a consistent revision order."
        ),
        "query": "How does the SQLite manifest enforce revision ordering and tombstones?",
    },
    {
        "title": "Deterministic chunk overlap",
        "topic": "Deterministic text chunking with overlapping character windows",
        "content": (
            "The window-v1 chunking profile creates deterministic character windows of "
            "1800 characters with 200 characters of overlap. Chunk identifiers include the "
            "profile name, profile version, source identifier, and character offsets. The "
            "overlap preserves context across boundaries while stable identifiers make "
            "retries idempotent. The service reports exact start and end offsets for every "
            "chunk so source provenance can be checked."
        ),
        "query": "What are the window-v1 chunk size and overlap, and how are chunk identifiers made deterministic?",
    },
    {
        "title": "OpenVINO GPU FP16 embedding inference",
        "topic": "OpenVINO GPU FP16 embedding inference",
        "content": (
            "OpenVINO GPU FP16 inference executes the Nomic embedding ONNX graph on the "
            "Intel integrated graphics device. The embedding pipeline tokenizes document "
            "text with the pinned tokenizer, applies the search_document prefix, performs "
            "masked mean pooling, and L2-normalizes the resulting 768-dimensional vector. "
            "The GPU backend must report the actual execution device and must not silently "
            "fall back to CPU."
        ),
        "query": "How does OpenVINO GPU FP16 produce normalized Nomic document embeddings?",
    },
]
FILLER_TOPICS = [
    "HTTP retry policy and bounded exponential backoff",
    "ChromaDB collection metadata and cosine distance",
    "REST service readiness and loopback-only binding",
    "Document provenance and canonical source locators",
    "Embedding batch limits and request validation",
    "Vector identifier stability across content revisions",
    "SQLite write-ahead logging and transaction isolation",
    "Service health checks and startup dependency ordering",
]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    if not isinstance(body, dict) or body.get("ok") is False:
        raise RuntimeError(f"{url} returned an unsuccessful response: {body}")
    return body


def wait_ready(url: str, process: subprocess.Popen, log_path: Path, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not yet responding"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            request_json(url, timeout=2)
            return
        except Exception as exc:  # startup races and transient connection errors
            last_error = str(exc)
            time.sleep(0.2)
    tail = ""
    try:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    except OSError:
        pass
    raise RuntimeError(f"service did not become ready at {url}: {last_error}\n{tail}")


def make_embedding_handler(encoder: Encoder):
    class EmbeddingHandler(BaseHTTPRequestHandler):
        server_version = "EvaluationEmbeddingAdapter/1"

        def _send(self, status: int, payload: dict) -> None:
            body = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "service": "embedding", "model_loaded": True})
            elif self.path == "/model":
                self._send(200, {
                    "ok": True, "model": MODEL_ID, "dimensions": DIMENSIONS,
                    "max_texts": 64, "max_text_chars": 65536,
                    "modes": ["document", "query"],
                })
            else:
                self._send(404, {"ok": False, "error": {"code": "not_found"}})

        def do_POST(self) -> None:
            if self.path != "/embeddings":
                self._send(404, {"ok": False, "error": {"code": "not_found"}})
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > 1_048_576:
                self._send(413, {"ok": False, "error": {"code": "payload_too_large"}})
                return
            try:
                body = json.loads(self.rfile.read(length).decode())
                texts, mode = body.get("texts"), body.get("mode", "document")
                if mode not in {"document", "query"} or not isinstance(texts, list) or not texts or len(texts) > 64:
                    raise ValueError("texts must be a non-empty list of at most 64; mode must be document or query")
                if any(not isinstance(value, str) or not value.strip() or len(value) > 65536 for value in texts):
                    raise ValueError("each text must be a non-empty string of at most 65536 characters")
                vectors = encoder.embed_queries(texts) if mode == "query" else encoder.embed_documents(texts)
                self._send(200, {
                    "ok": True, "model": MODEL_ID, "dimensions": DIMENSIONS,
                    "embeddings": vectors.tolist(),
                })
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                self._send(400, {"ok": False, "error": {"code": "invalid_request", "message": str(exc)}})
            except Exception as exc:
                self._send(500, {"ok": False, "error": {"code": "inference_error", "message": str(exc)}})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return EmbeddingHandler


class IsolatedServices:
    def __init__(self, poc_root: Path, state_dir: Path, model_dir: Path):
        self.poc_root = poc_root
        self.state_dir = state_dir
        self.model_dir = model_dir
        self.processes: list[subprocess.Popen] = []
        self.log_handles = []
        self.embedding_server: ThreadingHTTPServer | None = None
        self.embedding_thread: threading.Thread | None = None
        self.urls: dict[str, str] = {}

    def __enter__(self):
        ports = {
            "index": free_port(), "knowledge": free_port(),
            "chunking": free_port(), "vector": free_port(),
        }
        self.urls = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
        sqlite_dir = self.state_dir / "sqlite"
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "ENSEMBLE_EMBEDDING_MODEL_DIR": str(self.model_dir),
            "ENSEMBLE_INDEX_HOST": "127.0.0.1",
            "ENSEMBLE_INDEX_PORT": str(ports["index"]),
            "ENSEMBLE_INDEX_DB": str(sqlite_dir / "manifest.sqlite3"),
            "ENSEMBLE_KNOWLEDGE_HOST": "127.0.0.1",
            "ENSEMBLE_KNOWLEDGE_PORT": str(ports["knowledge"]),
            "ENSEMBLE_KNOWLEDGE_DB": str(sqlite_dir / "knowledge.sqlite3"),
            "ENSEMBLE_INDEX_URL": self.urls["index"],
            "ENSEMBLE_CHUNKING_HOST": "127.0.0.1",
            "ENSEMBLE_CHUNKING_PORT": str(ports["chunking"]),
            "ENSEMBLE_VECTOR_HOST": "127.0.0.1",
            "ENSEMBLE_VECTOR_PORT": str(ports["vector"]),
            "ENSEMBLE_VECTOR_STORE": str(self.state_dir / "chromadb"),
        })
        services = [
            ("index", "services/index/index_service.py", "/ready"),
            ("knowledge", "services/knowledge/knowledge_service.py", "/ready"),
            ("chunking", "services/chunking/chunking_service.py", "/ready"),
            ("vector", "services/embeddings/vector_service.py", "/health"),
        ]
        try:
            for name, relative_path, health_path in services:
                log_path = self.state_dir / f"{name}.log"
                handle = log_path.open("w", encoding="utf-8")
                self.log_handles.append(handle)
                process = subprocess.Popen(
                    [sys.executable, str(self.poc_root / relative_path)],
                    cwd=self.poc_root, env=env, stdout=handle, stderr=subprocess.STDOUT,
                )
                self.processes.append(process)
                wait_ready(self.urls[name] + health_path, process, log_path)
            return self
        except Exception:
            self.close()
            raise

    def start_embedding_adapter(self, encoder: Encoder) -> str:
        self.embedding_server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_embedding_handler(encoder)
        )
        port = self.embedding_server.server_address[1]
        self.embedding_thread = threading.Thread(
            target=self.embedding_server.serve_forever, daemon=True
        )
        self.embedding_thread.start()
        self.urls["embedding"] = f"http://127.0.0.1:{port}"
        wait_ready(self.urls["embedding"] + "/health", self.processes[-1],
                   self.state_dir / "vector.log")
        return self.urls["embedding"]

    def close(self) -> None:
        if self.embedding_server is not None:
            self.embedding_server.shutdown()
            self.embedding_server.server_close()
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in self.log_handles:
            handle.close()

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def fixture_documents(count: int, run_number: int) -> list[dict]:
    documents = []
    for index in range(count):
        if index < len(DOCS):
            base = DOCS[index]
        else:
            topic = FILLER_TOPICS[(index - len(DOCS)) % len(FILLER_TOPICS)]
            base = {
                "title": f"Evaluation fixture {index + 1}: {topic}",
                "topic": topic,
                "content": (
                    f"This engineering note describes {topic}. It documents the service "
                    "contract, validation rules, failure handling, observability, and "
                    "operational constraints. Requests are bounded, state changes are "
                    "explicit, and retries must preserve idempotency. The implementation "
                    "records enough provenance to explain the result of each operation."
                ),
            }
        filler = (
            f"\n\nOperational detail for {base['topic']}: the implementation validates "
            "inputs before changing state, records a deterministic identity, and reports "
            "errors through its REST contract. The service is tested with repeated requests, "
            "invalid payloads, and dependency failures. Measurements distinguish cold startup "
            "from warm steady-state execution and retain per-stage timing data."
        )
        content = base["content"] + filler * (5 + index % 4)
        documents.append({
            "source_id": f"https://evaluation.invalid/run-{run_number}/document-{index + 1:03d}",
            "title": base["title"],
            "content": content,
            "origin": "evaluation-fixture",
            "query": base.get("query") if index < len(DOCS) else None,
        })
    return documents


def post_timed(url: str, payload: dict, timeout: float = 120) -> tuple[dict, float]:
    started = time.perf_counter()
    result = request_json(url, payload, timeout=timeout)
    return result, (time.perf_counter() - started) * 1000


def vector_id(chunk_id: str, content_sha256: str) -> str:
    return hashlib.sha256(f"{chunk_id}|{content_sha256}".encode()).hexdigest()


def ingest_run(
    services: IsolatedServices,
    encoder: Encoder,
    backend: str,
    batch_size: int,
    document_count: int,
    run_number: int,
) -> dict:
    urls = services.urls
    collection = f"{COLLECTION_PREFIX}_{run_number:02d}"
    profiles = {
        "chunking": CHUNK_PROFILE_VERSION,
        "embedding": f"nomic-embed-text-v1.5@{backend}",
        "vector": "knowledge_chunks@1",
    }
    totals = {
        "knowledge_ms": 0.0, "chunking_ms": 0.0, "existing_ms": 0.0,
        "embedding_ms": 0.0, "vector_upsert_ms": 0.0,
    }
    documents = fixture_documents(document_count, run_number)
    pending: list[tuple[str, str, dict]] = []
    chunk_count = 0
    embedding_requests = 0
    flush_sizes: list[int] = []
    started = time.perf_counter()

    def flush() -> None:
        nonlocal embedding_requests
        if not pending:
            return
        texts = [text for _, text, _ in pending]
        embedded, elapsed = post_timed(
            urls["embedding"] + "/embeddings", {"texts": texts, "mode": "document"}
        )
        totals["embedding_ms"] += elapsed
        embedding_requests += 1
        vectors = embedded.get("embeddings", [])
        if len(vectors) != len(pending):
            raise RuntimeError(f"embedding count mismatch: expected {len(pending)}, got {len(vectors)}")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(pending), DIMENSIONS) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid embedding matrix: {matrix.shape}")
        upsert, elapsed = post_timed(
            f"{urls['vector']}/collections/{collection}/upsert",
            {
                "ids": [item[0] for item in pending],
                "embeddings": matrix.tolist(),
                "documents": texts,
                "metadatas": [item[2] for item in pending],
            },
        )
        totals["vector_upsert_ms"] += elapsed
        if upsert.get("upserted") != len(pending):
            raise RuntimeError(f"vector upsert count mismatch: {upsert}")
        flush_sizes.append(len(pending))
        pending.clear()

    for doc in documents:
        content_sha = hashlib.sha256(doc["content"].encode()).hexdigest()
        revision = {
            "source_id": doc["source_id"],
            "revision_sequence": 1,
            "revision_id": content_sha,
            "title": doc["title"],
            "content": doc["content"],
            "provenance": {"origin": doc["origin"], "locator": doc["source_id"]},
            "profiles": profiles,
        }
        stored, elapsed = post_timed(urls["knowledge"] + "/records", revision)
        totals["knowledge_ms"] += elapsed
        if stored.get("decision") not in {"accepted", "duplicate"} or not stored.get("stored"):
            raise RuntimeError(f"Index/Knowledge refused fixture {doc['source_id']}: {stored}")

        chunked, elapsed = post_timed(
            urls["chunking"] + "/chunk",
            {
                "profile": CHUNK_PROFILE, "source_id": doc["source_id"],
                "media_type": "text/plain", "text": doc["content"],
            },
        )
        totals["chunking_ms"] += elapsed
        chunks = chunked.get("chunks", [])
        chunk_count += len(chunks)
        ids = [vector_id(chunk["chunk_id"], content_sha) for chunk in chunks]
        found, elapsed = post_timed(
            f"{urls['vector']}/collections/{collection}/existing", {"ids": ids}
        )
        totals["existing_ms"] += elapsed
        present = set(found.get("found", []))
        for chunk, record_id in zip(chunks, ids):
            if record_id in present:
                continue
            pending.append((record_id, chunk["text"], {
                "kind": "knowledge_chunk", "source": doc["origin"],
                "url": doc["source_id"], "title": doc["title"],
                "chunk": chunk["index"], "chunks": len(chunks),
                "chunk_id": chunk["chunk_id"], "content_sha256": content_sha,
            }))
            if len(pending) >= batch_size:
                flush()
    flush()
    elapsed_s = time.perf_counter() - started

    manifest = request_json(urls["index"] + "/manifest?limit=500")
    manifest_count = len(manifest.get("sources", []))
    collections = request_json(urls["vector"] + "/collections")
    vector_count = next(
        (item["count"] for item in collections.get("collections", []) if item["name"] == collection),
        0,
    )
    if manifest_count != document_count:
        raise RuntimeError(f"manifest count mismatch: expected {document_count}, got {manifest_count}")
    if vector_count != chunk_count:
        raise RuntimeError(f"vector count mismatch: expected {chunk_count}, got {vector_count}")

    retrieval_checks = []
    for doc in documents[:len(DOCS)]:
        embedded, elapsed = post_timed(
            urls["embedding"] + "/embeddings", {"texts": [doc["query"]], "mode": "query"}
        )
        query_vector = embedded["embeddings"][0]
        result, query_ms = post_timed(
            f"{urls['vector']}/collections/{collection}/query",
            {"embedding": query_vector, "top_k": 5},
        )
        hits = result.get("results", [])
        top = hits[0] if hits else None
        retrieval_checks.append({
            "query": doc["query"],
            "expected_source_id": doc["source_id"],
            "top_source_id": top["metadata"].get("url") if top else None,
            "correct_top1": bool(top and top["metadata"].get("url") == doc["source_id"]),
            "embedding_ms": elapsed,
            "search_ms": query_ms,
        })

    return {
        "run": run_number,
        "backend": backend,
        "batch_size": batch_size,
        "documents": document_count,
        "chunks": chunk_count,
        "vectors_stored": vector_count,
        "embedding_requests": embedding_requests,
        "embedding_batch_sizes": flush_sizes,
        "manifest_records": manifest_count,
        "elapsed_seconds": elapsed_s,
        "documents_per_second": document_count / elapsed_s if elapsed_s else None,
        "chunks_per_second": chunk_count / elapsed_s if elapsed_s else None,
        "stage_ms": totals,
        "retrieval_smoke": {
            "queries": len(retrieval_checks),
            "correct_top1": sum(item["correct_top1"] for item in retrieval_checks),
            "checks": retrieval_checks,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+",
        choices=["onnxruntime-cpu", "openvino-cpu", "openvino-gpu"],
        default=["onnxruntime-cpu", "openvino-gpu"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--performance-hint", choices=["LATENCY", "THROUGHPUT"], default="THROUGHPUT")
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()
    if args.runs < 1 or args.documents < len(DOCS) or args.documents > 500:
        parser.error(f"--runs must be positive and --documents must be between {len(DOCS)} and 500")
    if not args.batch_sizes or any(size < 1 or size > 64 for size in args.batch_sizes):
        parser.error("--batch-sizes must contain values from 1 through 64")
    if args.threads < 1:
        parser.error("--threads must be positive")

    poc_root = resolve_sut_root()
    model_dir = MODEL_DIR.resolve()
    results = []
    for backend in args.backends:
        graph_name = "model_fp16.onnx" if backend == "openvino-gpu" else "model.onnx"
        graph = model_dir / "onnx" / graph_name
        if not graph.is_file() or not (model_dir / "tokenizer.json").is_file():
            parser.error(f"required local model artifacts are missing under {model_dir}; no downloads are attempted")
        for batch_size in args.batch_sizes:
            with tempfile.TemporaryDirectory(prefix=f"flossware-ingestion-{backend}-") as temp:
                state_dir = Path(temp)
                with IsolatedServices(poc_root, state_dir, model_dir) as services:
                    model_load_started = time.perf_counter()
                    encoder = Encoder(backend, graph, args.threads, args.performance_hint)
                    model_load_ms = (time.perf_counter() - model_load_started) * 1000
                    actual_device = str(encoder.providers[0]) if encoder.providers else "unknown"
                    if backend == "openvino-gpu" and "GPU" not in actual_device.upper():
                        raise RuntimeError(f"OpenVINO GPU requested but runtime reports {actual_device!r}")
                    services.start_embedding_adapter(encoder)
                    for run_number in range(1, args.runs + 1):
                        results.append(ingest_run(
                            services, encoder, backend, batch_size, args.documents, run_number
                        ) | {"device": actual_device, "graph": graph.name})
                # Temporary service state and stores are removed only after all child processes stop.

    result = write_result("ingestion", {
        "model": {
            "id": MODEL_ID, "revision": MODEL_REVISION, "directory": str(model_dir),
            "dimensions": DIMENSIONS,
        },
        "configuration": {
            "backends": args.backends, "batch_sizes": args.batch_sizes,
            "runs_per_configuration": args.runs, "documents_per_run": args.documents,
            "threads": args.threads, "performance_hint": args.performance_hint,
            "chunk_profile": CHUNK_PROFILE, "chunk_size_chars": 1800, "chunk_overlap_chars": 200,
            "stores": "temporary SQLite and ChromaDB; no POC user data is accessed",
        },
        "runtime_packages": package_versions(["numpy", "onnxruntime", "openvino", "chromadb", "tokenizers"]),
        "system_under_test_services": [
            "services/index/index_service.py",
            "services/knowledge/knowledge_service.py",
            "services/chunking/chunking_service.py",
            "services/embeddings/vector_service.py",
        ],
        "embedding_endpoint": "evaluation-owned adapter implementing the POC /embeddings contract",
        "runs": results,
    }, args.output)
    print(result)


if __name__ == "__main__":
    main()
