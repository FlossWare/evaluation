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

The harness does not own provider selection, routing, credential storage, retries, or production invocation. Those responsibilities remain with `model-gateway`. The harness reads only FlossWare-scoped environment variables and never accepts or prints credential values.

### Run

From the repository root, with the relevant `*_API_KEY_FLOSSWARE` variables exported:

```bash
bash tools/provider_api/catalogue_inventory.sh
python3 tools/provider_api/smoke_test.py
```

The catalogue command writes `provider-model-catalogue.json`. The smoke test writes `provider-smoke-results.jsonl`. Both are local evidence artifacts and should be reviewed before sharing.

The non-paid candidate set includes Google Gemini 3.5 Flash-Lite, Mistral Small (free API mode), Cohere Command R7B (trial key), catalogue-marked OpenRouter `:free` models (including DeepSeek V4 Flash 0731), and the DeepInfra-hosted DeepSeek V4 Flash 0731 route through Hugging Face's monthly included credits. The explicit paid-comparison batch includes Cerebras, DeepSeek, Groq, and a DeepInfra representative (`deepseek-ai/DeepSeek-V4-Flash-0731`) using DeepInfra's OpenAI-compatible chat endpoint. Hugging Face Inference Providers is also tested through its routed chat-completions endpoint. NVIDIA NIM's previously tested `openai/gpt-oss-120b` endpoint is marked deprecated and is excluded from execution. Paid or promotional-credit comparison models require explicit opt-in:

```bash
python3 tools/provider_api/smoke_test.py --include-paid
```

Run one representative candidate for every provider in a single batch:

```bash
python3 tools/provider_api/smoke_test.py --all-configured
```

This checks one non-paid candidate per provider where available, including Gemini 3.5 Flash-Lite, Mistral Small, Cohere Command R7B, and OpenRouter free candidates. Providers without credentials are recorded as skipped; providers whose only configured candidate is paid or promotional-credit based are also skipped unless `--include-paid` is supplied. Deprecated endpoints are always skipped. To explicitly include those candidates:

```bash
python3 tools/provider_api/smoke_test.py --all-configured --include-paid
```

The batch never fans out across every model in a provider catalogue. It selects at most one representative model per provider, continues after individual failures, and appends all outcomes to the same JSONL evidence file. `--all-configured` cannot be combined with `--provider` or `--model`.

Select providers with repeatable `--provider` arguments. The script makes one non-streaming request per selected model, defaults to a 128-token output limit (configurable with `--max-output-tokens`), and performs no automatic retries. It captures the requested output-token budget, total latency, HTTP status, request IDs, rate-limit headers, provider-native usage, normalized reasoning-token usage, finish reason, visible output length, and whether the exact synthetic response was returned. HTTP 402 responses are additionally classified as account-balance, account-credit/billing, or organization-subscription issues when the provider response supports that distinction; a compact sanitized error detail is printed so account restrictions are distinguishable from model/transport failures. Any response ending with `length` (OpenAI-compatible APIs) or `MAX_TOKENS` (Google) is classified as incomplete, even when partial or apparently matching text is present. It is classified as `incomplete-reasoning` when reasoning tokens were consumed, or `incomplete-output` otherwise; partial output is never mislabeled as a complete semantic mismatch. Other non-empty incorrect responses are `response-mismatch`, while empty responses without a length stop are `empty-response`. JSONL evidence is appended across invocations so successive experiments are retained. Router-selected model IDs are recorded separately from the requested router alias. It does not measure time-to-first-token; streaming belongs in the later capability suite.

Inspect the latest recorded HTTP errors without making any network requests:

```bash
python3 tools/provider_api/smoke_test.py --show-latest-errors
```

Run the deterministic classification tests with `python3 -m unittest discover -s tests -v`.

A zero price in a model catalogue is not a guarantee of availability or account eligibility. Check provider billing and account terms independently. Never send real user data, proprietary code, or secrets in these tests.

## Packaging note

The repository is named `evaluation`. The current distribution/import identifiers remain `evaluation-ai` and `evaluation_ai`; repository naming does not automatically rename a published package or Python imports. Change those identifiers only through an explicit compatibility and release decision.
