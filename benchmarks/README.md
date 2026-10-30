# Service benchmarks

Repeatable correctness, quality, and performance evaluations for FlossWare
services. This repository owns the benchmark harness; the implementation under
test remains in the separate `FlossWare/poc-claude-ensemble` checkout.

## Repository boundary

- `evaluation`: benchmark runners, fixed fixtures, raw result artifacts, and
  reports.
- `poc-claude-ensemble`: service implementations under test.
- Benchmark runs do not modify production service code or live CE data.

The runners locate a sibling `poc-claude-ensemble` checkout by default. If it
lives elsewhere, set:

```bash
export POC_CLAUDE_ENSEMBLE_ROOT="$HOME/Development/github/FlossWare/poc-claude-ensemble"
```

Run commands from the `evaluation` repository root.

## Current runners

| Track | Command | Measures |
|---|---|---|
| Embeddings | `python3 benchmarks/embedding/run.py` | Nomic ONNX Runtime CPU FP32, OpenVINO CPU, and OpenVINO GPU backends; batch sizes, latency, throughput, vector validity, semantic smoke, and optional FP32 cosine comparison |\n| Ingestion | `python3 benchmarks/ingestion/run.py` | Isolated Index → Knowledge → Chunking → Embedding → Vector REST pipeline; CPU/OpenVINO backends, batch sizes, per-stage timings, vector counts, and retrieval smoke checks |
| Vector retrieval | `python3 benchmarks/vector/run.py` | Exact NumPy vs argpartition vs Chroma HNSW, Recall@K |
| Reranking | `python3 benchmarks/reranking/run.py` | Pinned local MiniLM quality (MRR/NDCG) and batch-size timing |
| Hybrid retrieval | `python3 benchmarks/hybrid/run.py` | Production RRF fusion against fixed lexical/vector relevance judgments |
| Chunking | `python3 benchmarks/chunking/run.py` | Existing profiles, throughput, coverage, offsets, determinism |
| Tokenization | `python3 benchmarks/tokenization/run.py` | lexical-v1 throughput, determinism, source-offset correctness |
| Local LLM smoke | `python3 benchmarks/llm/run_ollama.py --models qwen3.5:2b` | Already-installed Ollama models, prompt/generation rates, latency, JSON validity |

Results are written to `benchmarks/results/` and are ignored by Git. Every
run creates a unique JSON artifact containing host metadata, both repository
revisions, configuration, timings, and correctness/quality metrics. The vector
runner excludes configurable warm-up queries from timing and records each
measurement round separately alongside aggregate statistics. No runner
downloads models. The Ollama runner only evaluates exact tags already returned
by the local `/api/tags` endpoint.

The evaluation repository also has dedicated model-comparison tools:
`scripts/compare-models.py` for Thompson outcome judgments and
`scripts/extract-entities.py` for entity/relationship extraction. Those tools
capture Ollama model digests and inference provenance; this benchmark suite's
LLM runner is a lightweight throughput/JSON smoke test, not a replacement.

## Commands

```bash
python3 -m compileall -q benchmarks
python3 benchmarks/chunking/run.py
python3 benchmarks/tokenization/run.py
python3 benchmarks/hybrid/run.py
python3 benchmarks/vector/run.py --sizes 1000 10000 --warmup-queries 20 --rounds 3
python3 benchmarks/embedding/run.py
python3 benchmarks/reranking/run.py
python3 benchmarks/llm/run_ollama.py --models qwen3.5:2b
```

Embedding and reranking require the pinned local model artifacts used by the
system under test. Embedding defaults to ONNX Runtime CPU FP32 and accepts
`--backend onnxruntime-cpu`, `--backend openvino-cpu`, or
`--backend openvino-gpu`. The GPU backend defaults to the local
`model_fp16.onnx` graph and compares its document vectors with the ONNX Runtime
CPU FP32 reference. OpenVINO runs can select `--performance-hint LATENCY` or
`THROUGHPUT`; install the OpenVINO Python package and ensure its GPU plugin and
Intel graphics runtime are available. Use identical batch sizes, warmups,
iterations, and model revision when comparing backends. Vector comparison requires NumPy and ChromaDB. The Ollama
runner requires a local Ollama server and installed model tags. Missing
prerequisites should be reported as unavailable; no model downloads occur.

These scripts are standalone evaluations, not part of the package's normal
unit-test command. Run the system-under-test tests from its own checkout.

## Interpretation

The current relevance fixtures are deliberately small smoke datasets, not a
statistically credible acceptance corpus. Expand them with representative,
privacy-safe queries and independently judged relevance labels before using
quality deltas to make production decisions. Keep quality and speed separate:
a faster candidate is not an improvement if retrieval quality regresses.

See [PLAN.md](PLAN.md) for implementation status, acceptance gates, and deferred
work.


The ingestion runner starts the POC Index, Knowledge, Chunking, and Vector
services on dynamically allocated loopback ports. It supplies an evaluation-owned
embedding adapter with the same `/embeddings` contract, backed by the selected
local inference runtime. SQLite databases and ChromaDB use a fresh temporary
directory for every backend/batch configuration; the runner does not read the
scrape ledger, raw corpus, live service ports, or existing POC stores. It creates
deterministic synthetic documents, records per-stage REST timings, verifies
manifest/vector counts, and checks retrieval against four known queries. The
runner requires the POC service dependencies and local Nomic model artifacts;
it never downloads models. The default run uses three repetitions and batch size
64. Use the expanded command above to compare batch sizes 16, 32, and 64.
