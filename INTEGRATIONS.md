# evaluation-ai Integrations

Install from GitHub:

```bash
pip install "git+https://github.com/FlossWare/evaluation-ai.git"
```

---

## Claude Code

### CLAUDE.md Snippet

```markdown
## Adversarial Evaluation (evaluation-ai)

This project uses `evaluation-ai` for adversarial verification and feedback loop detection.

- Adversarial verify: `from evaluation_ai import adversarial_verify, AdversarialVerifier`
- Feedback loops: `from evaluation_ai import detect_feedback_loops, SimpleFeedbackLoopDetector`
- Evaluation harness: `from evaluation_ai import SimpleEvaluationHarness, select_panel`
- Use `@adversarial_verify` decorator to automatically verify LLM outputs
- Use `@detect_feedback_loops` decorator to monitor for self-referential evaluation
- Zero external dependencies (stdlib only)
```

### Hook Example

Create `.claude/hooks/post-tool-edit.py` to verify critical changes:

```python
#!/usr/bin/env python3
"""Post-edit hook: flag files that should go through adversarial verification."""
from __future__ import annotations

import sys

CRITICAL_PATHS = ["auth", "security", "payment", "credential", "secret"]


def main():
    if len(sys.argv) < 2:
        sys.exit(0)

    file_path = sys.argv[1].lower()
    for keyword in CRITICAL_PATHS:
        if keyword in file_path:
            print(f"[evaluation-ai] CRITICAL FILE: {sys.argv[1]}")
            print("[evaluation-ai] Consider running adversarial verification:")
            print("  from evaluation_ai import adversarial_verify, AdversarialVerifier")
            break


if __name__ == "__main__":
    main()
```

### Skill Example

Create `.claude/skills/adversarial-review.md`:

```markdown
---
name: adversarial-review
description: Adversarially verify a claim or code output
---

Use evaluation-ai to adversarially verify outputs:

\```python
import asyncio
from evaluation_ai import AdversarialVerifier, select_panel

verifier = AdversarialVerifier(backend=backend, available_models=models)
panel = select_panel(available_models=models, panel_size=3)
result = await verifier.verify(
    claim="The function handles all edge cases",
    evidence="<code here>",
    panel=panel,
)
print(f"Verdict: {result.verdict} (confidence: {result.confidence})")
\```
```

---

## OpenAI Codex

### AGENTS.md Snippet

```markdown
## Adversarial Evaluation

Use evaluation-ai for multi-model adversarial verification:
- Install: `pip install "git+https://github.com/FlossWare/evaluation-ai.git"`
- Verifier: `AdversarialVerifier(backend, available_models)`
- Decorator: `@adversarial_verify(backend, available_models, panel_size=3)`
- Feedback loops: `SimpleFeedbackLoopDetector` monitors for self-referential evaluation
- Panel selection: `select_panel(available_models, panel_size)` for diverse model selection
```

### Tool Definition

```python
from evaluation_ai import AdversarialVerifier, select_panel, aggregate_verdicts

# Select a diverse panel
panel = select_panel(available_models=["gpt-4o", "claude-sonnet", "gemini-flash"], panel_size=3)

# Verify a claim
verifier = AdversarialVerifier(backend=my_backend, available_models=panel)
result = await verifier.verify(
    claim="This code is thread-safe",
    evidence=code_snippet,
    panel=panel,
)
print(f"Verdict: {result.verdict}, Confidence: {result.confidence}")
```

---

## Cursor

### .cursorrules Snippet

```
When verifying critical code or claims, use evaluation-ai:

- Import: from evaluation_ai import adversarial_verify, AdversarialVerifier, SimpleFeedbackLoopDetector
- Decorator: @adversarial_verify(backend=b, available_models=models, panel_size=3)
- Verifier: AdversarialVerifier(backend, available_models) for manual verification
- Feedback loops: SimpleFeedbackLoopDetector to detect self-referential evaluation
- Panel: select_panel(available_models, panel_size) for diverse model selection
- Zero dependencies - stdlib only
- Install: pip install "git+https://github.com/FlossWare/evaluation-ai.git"
```

---

## Crush

### Configuration

```python
# crush.config.py
from evaluation_ai import AdversarialVerifier, SimpleFeedbackLoopDetector, select_panel

async def verify_output(claim: str, evidence: str, backend, models: list[str]):
    """Adversarially verify a claim with a diverse model panel."""
    panel = select_panel(available_models=models, panel_size=3)
    verifier = AdversarialVerifier(backend=backend, available_models=models)
    return await verifier.verify(claim=claim, evidence=evidence, panel=panel)

def check_feedback_loops(execution_log: list[dict]) -> bool:
    """Check if recent executions show feedback loop risks."""
    detector = SimpleFeedbackLoopDetector()
    report = detector.analyze(execution_log)
    return not any(r.severity > 0.6 for r in report.risks)
```

---

## Generic Python Agent

### Adversarial Verification

```python
import asyncio
from evaluation_ai import (
    AdversarialVerifier,
    SimpleEvaluationHarness,
    SimpleFeedbackLoopDetector,
    select_panel,
    aggregate_verdicts,
)


async def main():
    # 1. Select a diverse panel from available models
    all_models = ["gpt-4o", "claude-sonnet", "gemini-flash", "llama-70b"]
    panel = select_panel(available_models=all_models, panel_size=3)
    print(f"Selected panel: {panel}")

    # 2. Create verifier
    verifier = AdversarialVerifier(backend=my_backend, available_models=all_models)

    # 3. Verify a claim
    result = await verifier.verify(
        claim="This sorting algorithm is O(n log n) in all cases",
        evidence="def sort(arr): return sorted(arr)",
        panel=panel,
    )
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence}")
    for member in result.panel_results:
        print(f"  {member.model}: {member.verdict} ({member.reasoning})")

asyncio.run(main())
```

### Feedback Loop Detection

```python
from evaluation_ai import SimpleFeedbackLoopDetector, FeedbackLoopReport

# Analyze execution history for self-referential patterns
detector = SimpleFeedbackLoopDetector()

execution_log = [
    {"model": "gpt-4o", "role": "generator", "task": "write code"},
    {"model": "gpt-4o", "role": "evaluator", "task": "review code"},  # same model!
    {"model": "claude-sonnet", "role": "generator", "task": "write docs"},
    {"model": "gemini-flash", "role": "evaluator", "task": "review docs"},
]

report: FeedbackLoopReport = detector.analyze(execution_log)
print(f"Risks found: {len(report.risks)}")
for risk in report.risks:
    print(f"  [{risk.severity:.1f}] {risk.description}")
```

### Decorator Pattern

```python
from evaluation_ai import adversarial_verify, detect_feedback_loops

@adversarial_verify(backend=eval_backend, available_models=models, panel_size=3)
async def generate_analysis(prompt: str, *, model: str = "default"):
    """Output is automatically verified by a diverse panel."""
    return await backend.chat([{"role": "user", "content": prompt}], model=model)

@detect_feedback_loops(detector=detector)
async def evaluated_generation(prompt: str, *, model: str = "default"):
    """Execution is monitored for self-referential patterns."""
    return await backend.chat([{"role": "user", "content": prompt}], model=model)
```

---

## Cross-Package Integration

### evaluation-ai + consensus-ai

Verify consensus results with adversarial panel:

```python
from evaluation_ai import adversarial_verify
from consensus_ai import with_consensus

@adversarial_verify(backend=eval_backend, available_models=["m4", "m5"], panel_size=3)
@with_consensus(strategy="majority_vote", models=["m1", "m2", "m3"])
async def verified_consensus(prompt: str, *, model: str = "default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=model)
```

### evaluation-ai + structured-output-ai

Structured adversarial verdicts:

```python
from evaluation_ai import adversarial_verify
from structured_output_ai import structured_output

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_safe": {"type": "boolean"},
        "risk_level": {"type": "string"},
        "evidence": {"type": "string"},
    },
}

@structured_output(schema=VERDICT_SCHEMA)
@adversarial_verify(backend=eval_backend, available_models=verifier_models)
async def safety_check(prompt: str, *, model: str = "default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=model)
```

### Full Stack: All Packages

```python
from evaluation_ai import adversarial_verify
from consensus_ai import with_consensus
from structured_output_ai import structured_output
from resilience_ai import with_retry, with_circuit_breaker
from observability_ai import track_execution

@structured_output(schema=SCHEMA)       # parse into typed object
@track_execution(telemetry=t)           # track timing and cost
@adversarial_verify(backend=eval_b)     # verify correctness
@with_consensus(models=models)          # multi-model vote
@with_retry(max_attempts=3)             # retry on failure
@with_circuit_breaker(provider="llm")   # circuit break per provider
async def production_query(prompt, *, model="default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=model)
```
