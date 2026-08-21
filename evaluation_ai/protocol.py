"""LLM backend protocol for evaluation-ai.

Defines the structural interface that any backend must satisfy in order
to be used with the evaluation and adversarial verification harnesses.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from evaluation_ai.types import ChatMessage, ChatResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for asynchronous LLM backends.

    Any object that provides an async ``chat`` method with the
    following signature satisfies this protocol via structural
    subtyping.
    """

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str = "",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send *messages* to the specified *model* and return a response."""
        ...
