#!/usr/bin/env python3
"""Basic evaluation-ai usage: adversarial verification and feedback loop detection."""
from __future__ import annotations

from evaluation_ai import (
    SimpleFeedbackLoopDetector,
    select_panel,
)


def main():
    # 1. Select a diverse evaluation panel
    all_models = ["gpt-4o", "claude-sonnet", "gemini-flash", "llama-70b", "mistral-large"]
    panel = select_panel(available_models=all_models, panel_size=3)
    print(f"Selected panel: {panel}")
    print(f"  Panel size: {len(panel)}")
    print(f"  From pool of: {len(all_models)} models")

    # 2. Feedback loop detection
    detector = SimpleFeedbackLoopDetector()

    execution_log = [
        {"model": "gpt-4o", "role": "generator", "task": "write code"},
        {"model": "gpt-4o", "role": "evaluator", "task": "review code"},
        {"model": "claude-sonnet", "role": "generator", "task": "write docs"},
        {"model": "gemini-flash", "role": "evaluator", "task": "review docs"},
        {"model": "gpt-4o", "role": "generator", "task": "fix bugs"},
        {"model": "gpt-4o", "role": "evaluator", "task": "verify fix"},
    ]

    report = detector.analyze(execution_log)
    print(f"\nFeedback Loop Analysis:")
    print(f"  Executions analyzed: {len(execution_log)}")
    print(f"  Risks found: {len(report.risks)}")
    for risk in report.risks:
        print(f"  [{risk.severity:.1f}] {risk.description}")

    # 3. Adversarial verification (requires async backend)
    print("\nAdversarial Verification (requires LLMBackend):")
    print("  verifier = AdversarialVerifier(backend, available_models)")
    print("  result = await verifier.verify(claim=..., evidence=..., panel=panel)")
    print("  result.verdict  -> 'CONFIRMED' | 'REFUTED' | 'INCONCLUSIVE'")
    print("  result.confidence -> 0.0 to 1.0")


if __name__ == "__main__":
    main()
