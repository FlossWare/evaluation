# evaluation-ai

**Evaluation and verification capabilities for FlossWare.**

Evaluation answers: *did the result satisfy the intended outcome, and how confident are we?*

## Boundary

```text
Intent + result + evidence
          │
          ▼
      Evaluator
          │
          ├── deterministic tests/rules
          ├── model judges
          ├── adversarial verification
          ├── human feedback
          └── external evaluators
          │
          ▼
      EvaluationResult
```

Evaluation is independent of model invocation. Model-backed evaluators should consume `model-gateway`; they should not define another provider abstraction.

Evaluation must distinguish authoritative verification from untrusted/model-generated reward. Learned reward signals may inform optimization, but they cannot override authoritative tests, policy, or verification.

## Feedback and learning

Evaluation results can feed Loom's adaptive strategies and Knowledge system with explicit provenance. Immediate and delayed outcomes should remain associated with the Intent, run, Worker/Arbiter, model/resource selection, tools, and relevant Knowledge versions.

## Relationship to Loom

`loom-ai` owns execution and orchestration. `evaluation-ai` provides reusable evaluation implementations. `model-gateway` owns model access. `knowledge` owns durable validated knowledge.

## License

MIT
