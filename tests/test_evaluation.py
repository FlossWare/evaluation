"""Tests for evaluation-ai package.

Covers types, evaluation harness, adversarial verification, and
feedback-loop detection -- all without external LLM calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
from evaluation_ai.evaluation import _parse_scores, _verdict_from_average


# -- Helpers ----------------------------------------------------------------


class FakeBackend:
    """Minimal backend that returns a fixed response for every call."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def chat(self, messages, *, model="", temperature=1.0, **kwargs):
        return ChatResponse(content=self._content, model=model)


class FailingBackend:
    """Backend that always raises."""

    async def chat(self, messages, *, model="", temperature=1.0, **kwargs):
        raise RuntimeError("boom")


# -- Type tests -------------------------------------------------------------


def test_chat_message_fields():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_response_defaults():
    resp = ChatResponse(content="hi")
    assert resp.model == ""
    assert resp.provider == ""
    assert resp.usage == {}


def test_evaluation_result_defaults():
    r = EvaluationResult(verdict="ACCEPT")
    assert r.scores == {}
    assert r.reasoning == ""
    assert r.evaluator_models == []


def test_feedback_loop_risk_fields():
    risk = FeedbackLoopRisk(
        layer="model_dominance",
        severity=0.9,
        description="too dominant",
        metric_value=0.9,
        threshold=0.7,
    )
    assert risk.layer == "model_dominance"
    assert risk.severity == 0.9


def test_feedback_loop_report_defaults():
    report = FeedbackLoopReport(is_healthy=True)
    assert report.risks == []
    assert report.window_days == 7


# -- Protocol test ----------------------------------------------------------


def test_llm_backend_protocol():
    """FakeBackend satisfies the LLMBackend protocol at runtime."""
    backend = FakeBackend("ok")
    assert isinstance(backend, LLMBackend)


# -- Evaluation harness tests -----------------------------------------------


def test_evaluation_no_backend():
    """Without a backend, harness returns default ACCEPT."""
    harness = SimpleEvaluationHarness()
    result = asyncio.run(
        harness.evaluate("output", task="task", models=["m1"])
    )
    assert result.verdict == "ACCEPT"
    assert result.scores["correctness"] == 5


def test_evaluation_with_perfect_scores():
    backend = FakeBackend(
        "correctness: 5\ncompleteness: 5\nquality: 5"
    )
    harness = SimpleEvaluationHarness(backend=backend)
    result = asyncio.run(
        harness.evaluate("output", task="task", models=["m1", "m2"])
    )
    assert result.verdict == "ACCEPT"
    assert len(result.evaluator_models) == 2


def test_evaluation_with_low_scores():
    backend = FakeBackend(
        "correctness: 1\ncompleteness: 1\nquality: 1"
    )
    harness = SimpleEvaluationHarness(backend=backend)
    result = asyncio.run(
        harness.evaluate("output", task="task", models=["m1"])
    )
    assert result.verdict == "REJECT"


def test_evaluation_unparseable_response():
    backend = FakeBackend("I have no scores to give.")
    harness = SimpleEvaluationHarness(backend=backend)
    result = asyncio.run(
        harness.evaluate("output", task="task", models=["m1"])
    )
    assert result.verdict == "REJECT"
    assert result.scores == {}


def test_evaluation_backend_failure():
    backend = FailingBackend()
    harness = SimpleEvaluationHarness(backend=backend)
    result = asyncio.run(
        harness.evaluate("output", task="task", models=["m1"])
    )
    assert result.verdict == "REJECT"


def test_parse_scores():
    text = "correctness: 4\ncompleteness: 3\nquality: 5"
    scores = _parse_scores(text)
    assert scores == {"correctness": 4, "completeness": 3, "quality": 5}


def test_verdict_thresholds():
    assert _verdict_from_average(4.0) == "ACCEPT"
    assert _verdict_from_average(3.0) == "ACCEPT_WITH_RESERVATIONS"
    assert _verdict_from_average(2.0) == "REJECT"


# -- Adversarial tests ------------------------------------------------------


def test_select_panel_excludes_candidate():
    panel = select_panel(["a", "b", "c"], "a", panel_size=2)
    assert "a" not in panel
    assert len(panel) == 2


def test_select_panel_empty():
    assert select_panel([], "a") == []


def test_select_panel_prefers_different_family():
    models = ["openai/gpt-4o", "openai/gpt-3.5", "google/gemini", "anthropic/claude"]
    panel = select_panel(models, "openai/gpt-4o", panel_size=2)
    # Should prefer google and anthropic over openai/gpt-3.5
    families = [m.split("/")[0] for m in panel]
    assert "openai" not in families


def test_aggregate_verdicts_majority():
    results = [
        PanelMemberResult(model="a", verdict="CONFIRMED", raw_response="ok"),
        PanelMemberResult(model="b", verdict="CONFIRMED", raw_response="ok"),
        PanelMemberResult(model="c", verdict="REFUTED", raw_response="no"),
    ]
    verdict, confidence = aggregate_verdicts(results)
    assert verdict == "CONFIRMED"
    assert abs(confidence - 2 / 3) < 0.01


def test_aggregate_verdicts_empty():
    verdict, confidence = aggregate_verdicts([])
    assert verdict == "UNCERTAIN"
    assert confidence == 0.0


def test_adversarial_verify_confirmed():
    backend = FakeBackend(
        "ANALYSIS: looks good\nERRORS: none\nVERDICT: CONFIRMED"
    )
    verifier = AdversarialVerifier(
        backend=backend,
        available_models=["a", "b", "c"],
        panel_size=3,
    )
    result = asyncio.run(
        verifier.verify("answer", task="question", candidate_model="x")
    )
    assert result.verdict == "CONFIRMED"
    assert result.confidence == 1.0


def test_adversarial_verify_no_panel():
    backend = FakeBackend("ok")
    verifier = AdversarialVerifier(
        backend=backend,
        available_models=["a"],
        panel_size=3,
    )
    result = asyncio.run(
        verifier.verify("answer", task="question", candidate_model="a")
    )
    assert result.verdict == "UNCERTAIN"


# -- Feedback loop tests ----------------------------------------------------


def test_feedback_loop_healthy():
    detector = SimpleFeedbackLoopDetector(usage_data=[
        {"model": "a", "role": "generator"},
        {"model": "b", "role": "evaluator"},
        {"model": "c", "role": "generator"},
    ])
    report = asyncio.run(detector.analyze())
    assert report.is_healthy is True


def test_feedback_loop_model_dominance():
    data = [{"model": "gpt-4o", "role": "generator"}] * 8 + [
        {"model": "claude", "role": "generator"},
        {"model": "gemini", "role": "generator"},
    ]
    detector = SimpleFeedbackLoopDetector(usage_data=data)
    report = asyncio.run(detector.analyze())
    assert report.is_healthy is False
    assert any(r.layer == "model_dominance" for r in report.risks)


def test_feedback_loop_eval_coupling():
    data = [
        {"model": "gpt-4o", "role": "generator"},
        {"model": "gpt-4o", "role": "evaluator"},
        {"model": "gpt-4o", "role": "evaluator"},
        {"model": "gpt-4o", "role": "evaluator"},
        {"model": "claude", "role": "evaluator"},
    ]
    detector = SimpleFeedbackLoopDetector(usage_data=data, coupling_threshold=0.40)
    report = asyncio.run(detector.analyze())
    coupling_risks = [r for r in report.risks if r.layer == "eval_coupling"]
    assert len(coupling_risks) >= 1


def test_feedback_loop_reward_hacking():
    data = [
        {"model": "a", "role": "generator", "quality": 0.5, "diversity": 0.8},
        {"model": "b", "role": "generator", "quality": 0.6, "diversity": 0.7},
        {"model": "c", "role": "generator", "quality": 0.9, "diversity": 0.3},
    ]
    detector = SimpleFeedbackLoopDetector(usage_data=data)
    report = asyncio.run(detector.analyze())
    assert any(r.layer == "reward_hacking" for r in report.risks)


def test_feedback_loop_empty_data():
    detector = SimpleFeedbackLoopDetector()
    report = asyncio.run(detector.analyze())
    assert report.is_healthy is True
    assert report.risks == []


def test_feedback_loop_record_usage():
    detector = SimpleFeedbackLoopDetector()
    detector.record_usage("gpt-4o", "generator")
    detector.record_usage("claude", "evaluator")
    report = asyncio.run(detector.analyze())
    assert report.is_healthy is True


def test_no_loom_ai_imports():
    """Verify that no source files contain loom_ai imports."""
    import pathlib
    pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "evaluation_ai"
    for py_file in pkg_dir.rglob("*.py"):
        content = py_file.read_text()
        assert "loom_ai" not in content, (
            f"{py_file.name} still contains a loom_ai reference"
        )


# -- Decorator tests --------------------------------------------------------


def test_adversarial_verify_no_backend_passthrough():
    """Without a backend, @adversarial_verify is a no-op."""

    @adversarial_verify(backend=None, available_models=None)
    async def my_func():
        return "hello"

    result = asyncio.run(my_func())
    assert result == "hello"


def test_adversarial_verify_with_backend():
    backend = FakeBackend(
        "ANALYSIS: looks good\nERRORS: none\nVERDICT: CONFIRMED"
    )

    @adversarial_verify(
        backend=backend,
        available_models=["a", "b", "c"],
        panel_size=2,
        task="test task",
    )
    async def my_func():
        return "answer"

    result = asyncio.run(my_func())
    assert result == "answer"


def test_adversarial_verify_refuted_raises():
    backend = FakeBackend(
        "ANALYSIS: wrong\nERRORS: everything\nVERDICT: REFUTED"
    )

    @adversarial_verify(
        backend=backend,
        available_models=["a", "b", "c"],
        panel_size=2,
        on_refuted="raise",
    )
    async def my_func():
        return "bad answer"

    try:
        asyncio.run(my_func())
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "refuted" in str(e).lower()


def test_detect_feedback_loops_healthy():
    @detect_feedback_loops(window_days=7, usage_data=[
        {"model": "a", "role": "generator"},
        {"model": "b", "role": "evaluator"},
    ])
    async def my_func():
        return "ok"

    result = asyncio.run(my_func())
    assert result == "ok"


def test_detect_feedback_loops_unhealthy_raises():
    data = [{"model": "x", "role": "generator"}] * 10 + [
        {"model": "y", "role": "generator"},
    ]

    @detect_feedback_loops(window_days=7, usage_data=data, on_unhealthy="raise")
    async def my_func():
        return "ok"

    try:
        asyncio.run(my_func())
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "feedback-loop" in str(e).lower() or "dominance" in str(e).lower()
