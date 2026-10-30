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

## Local model evaluation

Optional local LLM backends integrate through the existing `LLMBackend` protocol.
The first adapter targets Ollama and uses the local HTTP API without adding a
runtime dependency to the core package.

Model assets use the established `~/.FlossWare/models` directory. See
[Local model evaluation](docs/local-models.md) for setup, resource limits, and
usage. The initial laptop profile uses one loaded model, one inference request,
and a modest context window.

## Relationship to Loom

Loom defines execution and evidence semantics. `evaluation` supplies reusable evaluation behavior and result evidence that Loom Workers, Arbiters, or other compatible callers can consume. It does not replace the Arbiter or become an orchestration layer.

Evaluation results should distinguish measured facts, evaluator judgments, confidence, and provenance. A successful transport call is not proof that the intended outcome was achieved.

## Packaging note

The repository is named `evaluation`. The current distribution/import identifiers remain `evaluation-ai` and `evaluation_ai`; repository naming does not automatically rename a published package or Python imports. Change those identifiers only through an explicit compatibility and release decision.
