# evaluation

**Reusable evaluation and verification capabilities for FlossWare.**

Evaluation answers: *Did the result satisfy the intended outcome, and how strong is the supporting evidence?*

## Boundary

```text
Intent + result + evidence
          |
          v
       evaluation
          |
          +-- deterministic tests and rules
          +-- model-based judges
          +-- adversarial verification
          +-- human feedback
          +-- external evaluators
          |
          v
EvaluationResult
```

Evaluation is independent of model invocation. It may use `model-gateway` when a model-based judge is appropriate, but it does not own provider selection or model routing.

## Relationship to Loom

Loom defines execution and evidence semantics. `evaluation` supplies reusable evaluation behavior and result evidence that Loom Workers, Arbiters, or other compatible callers can consume. It does not replace the Arbiter or become an orchestration layer.

Evaluation results should distinguish measured facts, evaluator judgments, confidence, and provenance. A successful transport call is not proof that the intended outcome was achieved.

## Provider API research utilities

Experimental provider-discovery and smoke-test scripts live under `tools/provider_api/`. They support the cross-project investigation documented in [FlossWare/research/provider-api-inference](https://github.com/FlossWare/research/tree/research/provider-api-inference-2026-11/provider-api-inference).

These scripts are **research utilities, not part of the published `evaluation-ai` package**:

- `catalogue_inventory.sh` reads authenticated provider model catalogues and writes a normalized local JSON report. It does not invoke models.
- `smoke_test.py` sends a tiny synthetic prompt to selected catalogue-listed models and records transport/response evidence as JSONL.

The harness does not own provider selection, routing, credential storage, retries, or production invocation. Those responsibilities remain with `model-gateway`. Both utilities prefer FlossWare-scoped environment variables and fall back to conventional provider/SDK names. They never accept credentials as command-line arguments or print/write credential values.

### Run

From the repository root, with the relevant provider credentials exported. The harness first checks each `*_API_KEY_FLOSSWARE` name, then falls back to conventional provider/SDK names. The built-in catalogue includes Cerebras, Cohere, DeepInfra, DeepSeek, Google/Gemini, Groq, Hugging Face, Mistral, NVIDIA, OpenAI, OpenRouter, and Anthropic. Hugging Face accepts `HUGGINGFACE_API_KEY`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN`; Google also accepts `GEMINI_API_KEY`. In addition, the Python harness discovers non-empty environment variables ending in `_API_KEY`, `_API_KEY_FLOSSWARE`, `_API_KEY_HOTMAIL`, `_API_KEY_NCRR`, `_API_TOKEN`, or `_AUTH_TOKEN`, and reports providers without an adapter as unsupported rather than silently omitting them. Account suffixes `FLOSSWARE`, `HOTMAIL`, and `NCRR` are treated as separate credential identities, never merged. The all-configured batch creates a separate inventory row for each account and tests each account independently where a supported, eligible candidate exists. The FlossWare account is preferred only by single-credential convenience lookups; it does not suppress other account-scoped keys in the batch. Non-credential settings such as `GROQ_ORG_ID_FLOSSWARE` are not treated as API keys. This is a discovery inventory, not a claim that every discovered service has an inference integration. Only variable names are recorded, never values:

```bash
bash tools/provider_api/catalogue_inventory.sh
python3 tools/provider_api/smoke_test.py
```

The catalogue helper writes `provider-model-catalogue.json`. The unified runner writes `provider-smoke-results.jsonl`. Both are local evidence artifacts and should be reviewed before sharing.

### Unified runner inventory and reporting

The Python runner is the single command-line entry point for provider/account discovery, capability selection, inference execution, and evidence reporting. Its static adapter metadata lives in `tools/provider_api/provider-catalogue.json`; that catalogue records endpoints, authentication schemes, implemented capabilities, and operations. Model-level price/eligibility remains attached to individual model cases, because provider-wide pricing labels are misleading.

```bash
# Inventory known and environment-discovered providers/accounts, without requests.
python3 tools/provider_api/smoke_test.py --inventory

# Run all currently implemented chat adapters, preserving separate account identities.
python3 tools/provider_api/smoke_test.py --all-configured

# Record the current embeddings adapter backlog; this makes no provider requests.
python3 tools/provider_api/smoke_test.py --capability embeddings

# Summarize accumulated JSONL evidence, without requests.
python3 tools/provider_api/smoke_test.py --report
```

Capability families are declared in the provider catalogue: `chat`, `embeddings`, `reranking`, `image-generation`, and `document-processing`. Only adapters listed under `implemented_capabilities` execute network requests. Unsupported capability selections append explicit `skipped-no-capability-adapter` planning records, not false inference failures. Chat and text embeddings have initial adapters; reranking, image generation, and document processing remain planned. Adding an adapter means declaring its endpoint/authentication/operations in the catalogue, implementing its request/response contract, and adding deterministic tests. One runner does not mean forcing incompatible APIs through a chat-completions payload.

#### Embedding adapter evaluation

The initial embedding adapters use native provider request/response contracts for Voyage AI, Jina, Cohere, Google Gemini, OpenAI, and Mistral. Initial account testing found Jina credentials had insufficient balance on all four configured accounts, so Jina is conservatively classified as paid/account-credit and excluded from default execution. The bounded smoke request embeds one synthetic FlossWare sentence and validates that the response contains a finite, non-zero vector. Evidence stores model, dimensions, norm, latency, usage, status, and request ID; it deliberately does not persist vector contents.

```bash
# Plan capability coverage only. No provider requests are made.
python3 tools/provider_api/smoke_test.py --capability embeddings

# Execute candidates classified as free-tier/trial, across configured accounts. Jina is excluded after all four configured accounts reported no balance.
python3 tools/provider_api/smoke_test.py --all-configured --capability embeddings

# Jina is classified as paid/account-credit after the initial four-account test.
# It requires explicit opt-in if you later add credits and want to retest.
python3 tools/provider_api/smoke_test.py --capability embeddings --provider JINA --include-paid

# Explicitly opt into paid/account-credit candidates for one provider.
python3 tools/provider_api/smoke_test.py --capability embeddings --provider VOYAGEAI --include-paid

# Explicitly opt into all configured embedding candidates, including paid candidates.
python3 tools/provider_api/smoke_test.py --all-configured --capability embeddings --include-paid
```

Access labels are conservative candidate classifications, not a promise that a particular account is free or has quota. The default execution excludes `paid` and `paid-or-account-credit` candidates. The explicit `--include-paid` flag may incur charges. There are no retries. The adapter validates vector shape and numeric values without retaining raw embeddings.

#### Semantic retrieval benchmark

`tools/provider_api/embedding_benchmark.py` evaluates actual retrieval ranking rather than endpoint reachability. Its checked-in dataset contains 16 short passages and 8 natural-language queries with explicit gold-relevant passage IDs. The runner embeds passages with the provider's document task mode and queries with its query task mode, ranks passages using cosine similarity, and reports macro Recall@5, MRR@5, and binary nDCG@5.

```bash
# Plan only. Defaults to Cohere and Google, one preferred credential per provider.
python3 tools/provider_api/embedding_benchmark.py

# Execute the two batch requests per provider (one for passages, one for queries).
python3 tools/provider_api/embedding_benchmark.py --run

# Select providers explicitly.
python3 tools/provider_api/embedding_benchmark.py --provider COHERE --provider GOOGLE --run

# Include every configured account, or paid/account-credit providers, only deliberately.
python3 tools/provider_api/embedding_benchmark.py --all-accounts --run
python3 tools/provider_api/embedding_benchmark.py --provider JINA --include-paid --run
```

The benchmark requires `--run` before making network requests. By default it selects only Cohere and Google, which passed the initial embedding smoke test, and uses one preferred credential per provider (FlossWare first). Paid/account-credit candidates are excluded unless `--include-paid` is supplied. `--all-accounts` multiplies requests by configured accounts and should be used deliberately. Each provider/account run makes exactly two batch requests when successful. Vectors remain in memory for ranking and are never written to evidence; JSONL records dimensions, latency, usage, per-query top document IDs, and aggregate metrics. No automatic retries are performed.

The dataset is a small engineering-focused diagnostic, not a statistically representative benchmark or a substitute for a production corpus. Gold relevance judgments are hand-curated. Compare provider rankings on this same dataset, then validate promising candidates against representative FlossWare documents and queries before selecting a production embedding model. Vector spaces remain model-specific: do not mix Cohere and Gemini vectors in one fixed-dimension index.

#### Benchmarking an exported FlossWare corpus

The semantic benchmark accepts an external JSON dataset through `--dataset`. This keeps the evaluation harness independent of the corpus storage implementation.

Dataset format:

```json
{
  "name": "flossware-corpus-v1",
  "source": "local-corpus-export",
  "version": 1,
  "documents": [
    {
      "id": "stable-chunk-id",
      "text": "The canonical chunk text used for retrieval.",
      "metadata": {
        "document_id": "source-document-id",
        "collection": "collection-name"
      }
    }
  ],
  "queries": [
    {
      "id": "q-001",
      "text": "A realistic search query.",
      "relevant": ["stable-chunk-id"]
    }
  ]
}
```

Document metadata is optional and is not sent to embedding providers or written to result evidence. Document IDs are used to map relevance judgments and ranked results. Each query must have at least one manually reviewed relevant chunk ID. Keep this dataset local and out of Git when it contains private corpus text.

Run planning and execution:

```bash
# Validate and plan only; no provider requests.
python3 tools/provider_api/embedding_benchmark.py --dataset /path/to/kbase-corpus-export.json --queries /path/to/reviewed-queries.json

# Execute after reviewing the dataset and provider data-sharing implications.
python3 tools/provider_api/embedding_benchmark.py --dataset /path/to/kbase-corpus-export.json --queries /path/to/reviewed-queries.json --run

# Larger datasets are sent in bounded batches (default 64 texts per request).
python3 tools/provider_api/embedding_benchmark.py --dataset /path/to/kbase-corpus-export.json --queries /path/to/reviewed-queries.json --batch-size 32
```

The benchmark sends the actual document chunks and query text to each selected external embedding provider when `--run` is supplied. Do not include credentials, secrets, personal data, Red Hat confidential material, or content that is not approved for that provider. Only aggregate metrics, stable IDs, and request metadata are written to JSONL evidence; raw vectors and source text are not persisted by the runner. The dataset file itself remains under the caller's control.

For a credible corpus benchmark, sample representative chunks from the PostgreSQL knowledge corpus, create realistic queries, and have a human review the relevant chunk IDs. Do not infer relevance from an existing embedding model's nearest-neighbor results alone; that would bake the incumbent model's bias into the gold labels. Use a stable corpus snapshot and keep document/query sets identical across providers.


#### Exporting the private Obsidian kbase

The canonical FlossWare corpus is the private, Git-backed Obsidian vault in `FlossWare/kbase`. Do not copy its content into this public repository, fixtures, or release artifacts.

The local exporter reads Markdown files from a vault path and delegates segmentation to the canonical `FlossWare/chunking` implementation. It excludes hidden directories plus `.obsidian`, `.git`, and `.trash` by default. The output contains stable note/chunk IDs, relative source paths, offsets, and title metadata. It makes no provider requests and does not modify the vault.

```bash
python3 tools/provider_api/export_kbase.py \\
  --vault ~/Development/github/FlossWare/kbase \\
  --output ~/kbase-corpus-export.json

# Validate and plan only after preparing reviewed query judgments.
python3 tools/provider_api/embedding_benchmark.py \\
  --dataset ~/kbase-corpus-export.json \\
  --queries ~/kbase-reviewed-queries.json
```

The query file may be either a JSON array or an object containing a `queries` array. Each entry has a unique query ID, query text, and a non-empty `relevant` list of exported chunk IDs. Relevance judgments must be reviewed by a human. Do not use an existing embedding model alone to generate the gold labels.

Keep both exported corpus and query files outside Git. The benchmark transmits their text to selected external providers only when `--run` is supplied. Review the export and provider data-sharing rules before executing.

Provider API references: [Voyage embeddings](https://docs.voyageai.com/docs/embeddings), [Jina embeddings](https://docs.jina.ai/), [Cohere Embed v2](https://docs.cohere.com/reference/embed), [Google embedContent](https://ai.google.dev/api/embeddings), [OpenAI embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create), [Mistral embeddings](https://docs.mistral.ai/api/endpoint/embeddings).

The non-paid candidate set includes Google Gemini 3.5 Flash-Lite, Mistral Small (free API mode), Cohere Command R7B (trial key), catalogue-marked OpenRouter `:free` models (including DeepSeek V4 Flash 0731), and the DeepInfra-hosted DeepSeek V4 Flash 0731 route through Hugging Face's monthly included credits. The explicit paid-comparison batch includes Cerebras, DeepSeek, Groq, and a DeepInfra representative (`deepseek-ai/DeepSeek-V4-Flash-0731`) using DeepInfra's OpenAI-compatible chat endpoint. Hugging Face Inference Providers is also tested through its routed chat-completions endpoint. NVIDIA NIM's previously tested `openai/gpt-oss-120b` endpoint is marked deprecated and is excluded from execution. Paid or promotional-credit comparison models require explicit opt-in:

```bash
python3 tools/provider_api/smoke_test.py --include-paid
```

Run the complete provider inventory in one batch:

```bash
python3 tools/provider_api/smoke_test.py --all-configured
```

This iterates every known provider, not merely the providers with a ready model. It runs at most one representative candidate for each provider with a configured credential and a free/trial/zero-price candidate, including Gemini 3.5 Flash-Lite, Mistral Small, Cohere Command R7B, Hugging Face's monthly included-credit route, and OpenRouter free candidates. Every provider discovered from the environment is written to the evidence file with an explicit status: no credential, paid candidate requiring opt-in, deprecated endpoint, no supported chat candidate, or no provider adapter. OpenAI is included as a paid-only representative and is not called by default. Anthropic is included in catalogue credential discovery; inference remains skipped until an appropriate message API candidate/adapter is explicitly added. The final summary reports the complete discovered provider count and skip categories. Paid/promotional-credit candidates are never invoked by default; `--include-paid` is a separate explicit opt-in and may incur charges. Deprecated endpoints are always skipped. To explicitly include those candidates:

```bash
python3 tools/provider_api/smoke_test.py --all-configured --include-paid
```

The batch never fans out across every model in a provider catalogue. It selects at most one representative model per provider, continues after individual failures, and appends all outcomes to the same JSONL evidence file. `--all-configured` cannot be combined with `--provider` or `--model`.

Select providers with repeatable `--provider` arguments. The script makes one non-streaming request per selected model, defaults to a 512-token output limit (configurable with `--max-output-tokens`), and performs no automatic retries. It captures the requested output-token budget, total latency, HTTP status, request IDs, rate-limit headers, provider-native usage, normalized reasoning-token usage, finish reason, visible output length, and whether the exact synthetic response was returned. HTTP 402 responses are classified as account-balance, account-credit/billing, or organization-subscription issues when provider details support that distinction. HTTP 429 responses are classified separately, including upstream shared-pool limits and BYOK provider limits when metadata is available. HTTP 403 responses explicitly identifying agentic-harness-only access are classified as `agentic-harness-required`; other 403 responses remain generic provider errors. Error evidence removes provider-supplied user identifiers and redacts the active credential before persistence; compact diagnostics prefer the provider's underlying error detail so account restrictions and shared-pool throttling are distinguishable from model/transport failures. Any response ending with `length` (OpenAI-compatible APIs) or `MAX_TOKENS` (Google) is classified as incomplete, even when partial or apparently matching text is present. It is classified as `incomplete-reasoning` when reasoning tokens were consumed, or `incomplete-output` otherwise; partial output is never mislabeled as a complete semantic mismatch. Other non-empty incorrect responses are `response-mismatch`, while empty responses without a length stop are `empty-response`. JSONL evidence is appended across invocations so successive experiments are retained. Router-selected model IDs are recorded separately from the requested router alias. It does not measure time-to-first-token; streaming belongs in the later capability suite.

Inspect which credential variable name is selected for each provider without displaying values or making network requests:

```bash
python3 tools/provider_api/smoke_test.py --show-credential-sources
```

Inspect the latest recorded HTTP errors for the supported diagnostic providers, including OpenRouter, without making any network requests:

```bash
python3 tools/provider_api/smoke_test.py --show-latest-errors
```

Run the deterministic classification tests with `python3 -m unittest discover -s tests -v`.

A zero price in a model catalogue is not a guarantee of availability or account eligibility. Check provider billing and account terms independently. Never send real user data, proprietary code, or secrets in these tests.

## Packaging note

The repository is named `evaluation`. The current distribution/import identifiers remain `evaluation-ai` and `evaluation_ai`; repository naming does not automatically rename a published package or Python imports. Change those identifiers only through an explicit compatibility and release decision.
