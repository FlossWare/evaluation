# FlossWare Engineering Standards Compliance

This package adheres to the following ADRs from [FlossWare/engineering-standards](https://github.com/FlossWare/engineering-standards):

## ADR-0001: Explicit Opt-In

Evaluation, adversarial verification, and feedback-loop detection never activate automatically.
All capabilities require explicit instantiation or decorator application by the developer.

- `SimpleEvaluationHarness` must be instantiated and `evaluate()` called explicitly.
- `AdversarialVerifier` must be instantiated with a backend and models.
- `SimpleFeedbackLoopDetector` requires explicit `analyze()` calls.
- `@adversarial_verify` and `@detect_feedback_loops` decorators are opt-in.
- When no backend is configured, harnesses degrade gracefully (default ACCEPT / no-op).

## ADR-0006: Cross-Cutting Decorators

Convenience decorators in `evaluation_ai.decorators`:

- `@adversarial_verify(panel_size=3)` -- wraps an async function with adversarial panel verification.
- `@detect_feedback_loops(window_days=7)` -- monitors for self-referential evaluation patterns before each call.

## ADR-0008: Free-First

Zero external dependencies at runtime. The package uses only the Python standard library (`asyncio`, `re`, `dataclasses`, `collections`, `datetime`, `logging`, `typing`).

Development dependencies (pytest, pytest-asyncio) are optional.

## ADR-0009: Core Principles

- **Modular**: Each concern (evaluation, adversarial, feedback-loop) is a separate module.
- **Composable**: Components can be used independently or combined.
- **Contracts over implementations**: The `LLMBackend` Protocol defines the interface; any conforming object works.

## ADR-0012: Multi-Model Consensus Quality Gates

`SimpleEvaluationHarness` implements the quality gates described in ADR-0012:

- Fans out evaluation to multiple models in parallel.
- Each model scores on correctness, completeness, and quality (1-5).
- Scores are averaged across all responding models.
- Verdict thresholds: ACCEPT (>=4.0), ACCEPT_WITH_RESERVATIONS (>=2.5), REJECT (<2.5).
- Models that fail or return unparseable responses are excluded from consensus.

## ADR-0017: Agent-Neutral

The package works with any agent runtime. The `LLMBackend` Protocol is the only integration point -- any agent framework that can provide an async `chat()` method is compatible.

No assumptions are made about the calling agent's architecture, event loop, or lifecycle.

## ADR-0020: Capability-Protocol Separation

Evaluation capabilities are transport-independent:

- `LLMBackend` Protocol defines what is needed (async chat), not how it is delivered.
- No HTTP, gRPC, or other transport assumptions baked in.
- The same evaluation logic works whether the backend is a local mock, an API client, or an agent-internal router.
