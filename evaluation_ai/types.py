"""Shared data types for the evaluation-ai package.

All types are plain ``dataclasses`` with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """A single chat message exchanged with a language model."""

    role: str
    content: str


@dataclass
class ChatResponse:
    """Response returned by an LLM backend."""

    content: str
    model: str = ""
    provider: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation result from the multi-model harness."""

    verdict: str
    scores: dict = field(default_factory=dict)
    reasoning: str = ""
    evaluator_models: list[str] = field(default_factory=list)


@dataclass
class FeedbackLoopRisk:
    """A single feedback-loop risk detected during analysis."""

    layer: str
    severity: float
    description: str
    metric_value: float
    threshold: float


@dataclass
class FeedbackLoopReport:
    """Complete feedback-loop analysis report."""

    is_healthy: bool
    risks: list[FeedbackLoopRisk] = field(default_factory=list)
    analyzed_at: str = ""
    window_days: int = 7
