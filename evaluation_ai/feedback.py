"""Conversation feedback detection extracted from the retired learning package.

Feedback is an evaluation signal, not a separate learning subsystem.  This
module deliberately detects observations only.  Interpretation, reward
assignment, persistence, and knowledge extraction belong to their respective
components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evaluation_ai.types import ChatMessage


@dataclass(frozen=True)
class FeedbackPattern:
    """A feedback pattern with a signal type and base confidence."""

    pattern: re.Pattern[str]
    feedback_type: str
    confidence: float


@dataclass(frozen=True)
class FeedbackSignal:
    """An observed feedback signal from a user message."""

    type: str
    content: str
    confidence: float
    source_message: str


DEFAULT_FEEDBACK_PATTERNS: tuple[FeedbackPattern, ...] = (
    FeedbackPattern(re.compile(r"\byou\s+should\s+have\b", re.I), "correction", 0.9),
    FeedbackPattern(re.compile(r"\bthat'?s\s+not\s+(right|correct|what)\b", re.I), "correction", 0.9),
    FeedbackPattern(re.compile(r"\bthat\s+is\s+not\s+(right|correct|what)\b", re.I), "correction", 0.9),
    FeedbackPattern(re.compile(r"\bwhy\s+didn'?t\s+you\b", re.I), "correction", 0.85),
    FeedbackPattern(re.compile(r"\bwhy\s+did\s+not\s+you\b", re.I), "correction", 0.85),
    FeedbackPattern(re.compile(r"\binstead\s+of\b", re.I), "correction", 0.7),
    FeedbackPattern(re.compile(r"\balways\s+\w+", re.I), "preference", 0.85),
    FeedbackPattern(re.compile(r"\bnever\s+\w+", re.I), "preference", 0.85),
    FeedbackPattern(re.compile(r"\bdon'?t\s+\w+", re.I), "preference", 0.8),
    FeedbackPattern(re.compile(r"\bdo\s+not\s+\w+", re.I), "preference", 0.8),
    FeedbackPattern(re.compile(r"\bprefer\s+\w+\s+over\s+\w+", re.I), "preference", 0.9),
    FeedbackPattern(re.compile(r"\bprefer\s+\w+", re.I), "preference", 0.8),
    FeedbackPattern(re.compile(r"\bavoid\s+\w+", re.I), "preference", 0.75),
    FeedbackPattern(re.compile(r"\bthat\s+worked\s+well\b", re.I), "confirmation", 0.8),
    FeedbackPattern(re.compile(r"\bgood\s+job\b", re.I), "confirmation", 0.7),
    FeedbackPattern(re.compile(r"\bperfect\b", re.I), "confirmation", 0.65),
    FeedbackPattern(re.compile(r"\bexactly\s+what\s+I\s+wanted\b", re.I), "confirmation", 0.85),
    FeedbackPattern(re.compile(r"\bthat'?s\s+(great|correct|right)\b", re.I), "confirmation", 0.75),
    FeedbackPattern(re.compile(r"\bthat\s+is\s+(great|correct|right)\b", re.I), "confirmation", 0.75),
)


class FeedbackPatternMatcher:
    """Detect corrections, preferences, and confirmations in user messages."""

    def __init__(
        self, patterns: tuple[FeedbackPattern, ...] = DEFAULT_FEEDBACK_PATTERNS
    ) -> None:
        self._patterns = patterns

    @property
    def patterns(self) -> tuple[FeedbackPattern, ...]:
        return self._patterns

    def match(self, messages: list[ChatMessage]) -> list[FeedbackSignal]:
        """Return at most one signal of each type per user message."""
        signals: list[FeedbackSignal] = []
        for message in messages:
            if message.role != "user":
                continue
            seen_types: set[str] = set()
            for feedback_pattern in self._patterns:
                if feedback_pattern.feedback_type in seen_types:
                    continue
                if feedback_pattern.pattern.search(message.content):
                    seen_types.add(feedback_pattern.feedback_type)
                    signals.append(
                        FeedbackSignal(
                            type=feedback_pattern.feedback_type,
                            content=message.content,
                            confidence=feedback_pattern.confidence,
                            source_message=message.content,
                        )
                    )
        return signals

    async def match_async(self, messages: list[ChatMessage]) -> list[FeedbackSignal]:
        """Async adapter for callers using async evaluation protocols."""
        return self.match(messages)
