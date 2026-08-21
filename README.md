# evaluation-ai

Multi-model evaluation harness, adversarial verification panels, and feedback-loop detection for LLM outputs.

Zero external dependencies at runtime -- uses only the Python standard library.

## Installation

```bash
pip install evaluation-ai
```

For development (includes pytest):

```bash
pip install evaluation-ai[dev]
```

## Quickstart

### Multi-model evaluation

```python
import asyncio
from evaluation_ai import SimpleEvaluationHarness, ChatMessage, ChatResponse

# Without a backend (returns default ACCEPT for testing)
harness = SimpleEvaluationHarness()
result = asyncio.run(harness.evaluate("2+2=4", task="Compute 2+2", models=[]))
print(result.verdict)  # "ACCEPT"

# With a real backend (any object satisfying the LLMBackend protocol)
harness = SimpleEvaluationHarness(backend=my_backend)
result = asyncio.run(harness.evaluate(
    output="The answer is 42.",
    task="What is the meaning of life?",
    models=["gpt-4o", "claude-sonnet", "gemini-pro"],
))
print(result.verdict, result.scores)
```

### Adversarial verification

```python
from evaluation_ai import AdversarialVerifier

verifier = AdversarialVerifier(
    backend=my_backend,
    available_models=["gpt-4o", "claude-sonnet", "gemini-pro", "llama-3"],
    panel_size=3,
)
result = asyncio.run(verifier.verify(
    candidate="Paris is the capital of France.",
    task="What is the capital of France?",
    candidate_model="gpt-4o",
))
print(result.verdict, result.confidence)
```

### Feedback-loop detection

```python
from evaluation_ai import SimpleFeedbackLoopDetector

detector = SimpleFeedbackLoopDetector()
detector.record_usage("gpt-4o", "generator")
detector.record_usage("gpt-4o", "evaluator")
detector.record_usage("gpt-4o", "generator")

report = asyncio.run(detector.analyze(window_days=7))
print(report.is_healthy, report.risks)
```

## API Overview

### Types

- `ChatMessage(role, content)` -- a single chat message
- `ChatResponse(content, model, provider, usage)` -- response from an LLM
- `EvaluationResult(verdict, scores, reasoning, evaluator_models)` -- evaluation outcome
- `FeedbackLoopRisk(layer, severity, description, metric_value, threshold)` -- detected risk
- `FeedbackLoopReport(is_healthy, risks, analyzed_at, window_days)` -- analysis report

### Protocol

- `LLMBackend` -- runtime-checkable protocol requiring an async `chat()` method

### Classes

- `SimpleEvaluationHarness` -- fans out evaluation to multiple models, averages scores, derives verdict
- `AdversarialVerifier` -- assembles independent panel to critique a candidate response
- `SimpleFeedbackLoopDetector` -- detects model dominance, eval-generator coupling, reward hacking, and concept collapse

### Helper functions

- `select_panel(available_models, candidate_model, panel_size)` -- choose independent panel members
- `aggregate_verdicts(panel_results)` -- majority-vote aggregation

### Decorators (ADR-0006)

```python
from evaluation_ai import adversarial_verify, detect_feedback_loops

@adversarial_verify(backend=my_backend, available_models=["a", "b", "c"], panel_size=3)
async def generate_answer(question: str) -> str:
    return "42"

@detect_feedback_loops(window_days=7, usage_data=my_usage_data)
async def run_pipeline() -> dict:
    return {"status": "ok"}
```

## FlossWare Engineering Standards

This package complies with [FlossWare/engineering-standards](https://github.com/FlossWare/engineering-standards):

| ADR | Title | How |
|-----|-------|-----|
| ADR-0001 | Explicit Opt-In | Nothing activates automatically; all capabilities require explicit instantiation |
| ADR-0006 | Cross-Cutting Decorators | `@adversarial_verify`, `@detect_feedback_loops` |
| ADR-0008 | Free-First | Zero external runtime dependencies (stdlib only) |
| ADR-0009 | Core Principles | Modular, composable, contracts over implementations |
| ADR-0012 | Multi-Model Consensus Quality Gates | `SimpleEvaluationHarness` with parallel fan-out and score averaging |
| ADR-0017 | Agent-Neutral | Works with any agent runtime via `LLMBackend` Protocol |
| ADR-0020 | Capability-Protocol Separation | Transport-independent evaluation capabilities |

See [STANDARDS.md](STANDARDS.md) for detailed compliance notes.

## License

MIT
