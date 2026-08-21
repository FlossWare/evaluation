"""evaluation-ai -- Multi-model evaluation, adversarial verification, and feedback-loop detection.

Public API
----------
Types:
    ChatMessage, ChatResponse, EvaluationResult,
    FeedbackLoopRisk, FeedbackLoopReport

Protocol:
    LLMBackend

Evaluation:
    SimpleEvaluationHarness

Adversarial:
    AdversarialVerifier, VerificationResult, PanelMemberResult,
    select_panel, aggregate_verdicts

Feedback Loop:
    SimpleFeedbackLoopDetector

Decorators (ADR-0006):
    adversarial_verify, detect_feedback_loops
"""

from __future__ import annotations

from evaluation_ai.adversarial import (
    AdversarialVerifier,
    PanelMemberResult,
    VerificationResult,
    aggregate_verdicts,
    select_panel,
)
from evaluation_ai.decorators import adversarial_verify, detect_feedback_loops
from evaluation_ai.evaluation import SimpleEvaluationHarness
from evaluation_ai.feedback_loop import SimpleFeedbackLoopDetector
from evaluation_ai.protocol import LLMBackend
from evaluation_ai.types import (
    ChatMessage,
    ChatResponse,
    EvaluationResult,
    FeedbackLoopReport,
    FeedbackLoopRisk,
)

__all__ = [
    "AdversarialVerifier",
    "ChatMessage",
    "ChatResponse",
    "EvaluationResult",
    "FeedbackLoopReport",
    "FeedbackLoopRisk",
    "LLMBackend",
    "PanelMemberResult",
    "SimpleEvaluationHarness",
    "SimpleFeedbackLoopDetector",
    "VerificationResult",
    "adversarial_verify",
    "aggregate_verdicts",
    "detect_feedback_loops",
    "select_panel",
]

__version__ = "0.1"
