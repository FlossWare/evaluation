#!/usr/bin/env python3
"""Verify evaluation-ai installation and run a quick smoke test."""
import sys


def main():
    try:
        from evaluation_ai import (
            AdversarialVerifier,
            ChatMessage,
            ChatResponse,
            EvaluationResult,
            FeedbackLoopReport,
            FeedbackLoopRisk,
            LLMBackend,
            PanelMemberResult,
            SimpleEvaluationHarness,
            SimpleFeedbackLoopDetector,
            VerificationResult,
            adversarial_verify,
            aggregate_verdicts,
            detect_feedback_loops,
            select_panel,
        )
    except ImportError as e:
        print(f"FAIL: Could not import evaluation_ai: {e}")
        print("Install: pip install 'git+https://github.com/FlossWare/evaluation-ai.git'")
        sys.exit(1)

    import evaluation_ai

    print(f"evaluation-ai v{evaluation_ai.__version__} installed successfully")
    print(f"Exports: {len(evaluation_ai.__all__)} public symbols")

    # Smoke test: panel selection
    models = ["gpt-4o", "claude-sonnet", "gemini-flash", "llama-70b"]
    panel = select_panel(available_models=models, panel_size=3)
    print(f"Smoke test: select_panel({len(models)} models, size=3) -> {panel}")

    # Smoke test: feedback loop detector
    detector = SimpleFeedbackLoopDetector()
    print(f"Smoke test: SimpleFeedbackLoopDetector created: {detector}")

    # Smoke test: decorator is callable
    assert callable(adversarial_verify), "adversarial_verify must be callable"
    assert callable(detect_feedback_loops), "detect_feedback_loops must be callable"
    print("Smoke test: decorators are callable")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
