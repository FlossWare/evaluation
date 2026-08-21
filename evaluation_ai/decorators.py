"""Convenience decorators for evaluation-ai (ADR-0006).

Provides ``@adversarial_verify`` and ``@detect_feedback_loops``
decorators that wrap async functions with cross-cutting evaluation
concerns.  Both are explicit opt-in (ADR-0001) -- nothing activates
unless a developer deliberately applies a decorator.

Zero external dependencies -- uses only the standard library.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

from evaluation_ai.adversarial import AdversarialVerifier, VerificationResult
from evaluation_ai.feedback_loop import SimpleFeedbackLoopDetector
from evaluation_ai.types import FeedbackLoopReport

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def adversarial_verify(
    *,
    backend: Any = None,
    available_models: list[str] | None = None,
    panel_size: int = 3,
    temperature: float = 0.3,
    task: str = "",
    candidate_model: str = "",
    on_refuted: str = "warn",
) -> Callable[[F], F]:
    """Decorator that wraps an async function with adversarial panel verification.

    The decorated function's return value (converted to ``str``) is
    submitted to an :class:`AdversarialVerifier` panel for independent
    critique.

    Parameters
    ----------
    backend:
        An ``LLMBackend``-compatible object.  Required for actual
        verification; when ``None``, the decorator is a no-op pass-through
        (ADR-0001: explicit opt-in, graceful degradation).
    available_models:
        Model ids available for panel selection.
    panel_size:
        Number of panel members (default ``3``).
    temperature:
        Temperature for panel member calls (default ``0.3``).
    task:
        Task description for the verification prompt.  When empty, the
        decorated function's docstring is used as a fallback.
    candidate_model:
        The model that generated the candidate output.
    on_refuted:
        Action when the panel refutes the output:
        ``"warn"`` (default) -- log a warning and return the result anyway;
        ``"raise"`` -- raise ``ValueError``.

    Returns
    -------
    The original function's return value.  A ``_verification`` attribute
    is attached to the return value (if it supports attribute assignment)
    containing the :class:`VerificationResult`.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await fn(*args, **kwargs)

            if backend is None or not available_models:
                return result

            verifier = AdversarialVerifier(
                backend=backend,
                available_models=available_models,
                panel_size=panel_size,
                temperature=temperature,
            )

            effective_task = task or (fn.__doc__ or fn.__name__)
            verification = await verifier.verify(
                str(result),
                task=effective_task,
                candidate_model=candidate_model,
            )

            if verification.verdict == "REFUTED":
                msg = (
                    f"Adversarial panel refuted output of {fn.__name__!r} "
                    f"(confidence={verification.confidence:.0%})"
                )
                if on_refuted == "raise":
                    raise ValueError(msg)
                logger.warning(msg)

            # Attach verification metadata when possible.
            try:
                result._verification = verification  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                pass

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def detect_feedback_loops(
    *,
    window_days: int = 7,
    usage_data: list[dict[str, Any]] | None = None,
    detector: SimpleFeedbackLoopDetector | None = None,
    on_unhealthy: str = "warn",
) -> Callable[[F], F]:
    """Decorator that monitors an async function for feedback-loop risks.

    Runs :class:`SimpleFeedbackLoopDetector` analysis before each
    invocation using the supplied *usage_data*.

    Parameters
    ----------
    window_days:
        Analysis window in days (default ``7``).
    usage_data:
        Pre-loaded usage records.  When ``None``, the detector starts
        empty (no risks detected -- ADR-0001 safe default).
    detector:
        A pre-existing detector instance.  When provided, state persists
        across decorated function invocations.
    on_unhealthy:
        Action when feedback loops are detected:
        ``"warn"`` (default) -- log a warning and continue;
        ``"raise"`` -- raise ``RuntimeError``.

    Returns
    -------
    The original function's return value.  A ``_feedback_loop_report``
    attribute is attached when possible.
    """
    active_detector = detector or SimpleFeedbackLoopDetector(usage_data=usage_data)

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            report = await active_detector.analyze(window_days=window_days)

            if not report.is_healthy:
                risk_summary = "; ".join(r.description for r in report.risks)
                msg = (
                    f"Feedback-loop risks detected before {fn.__name__!r}: "
                    f"{risk_summary}"
                )
                if on_unhealthy == "raise":
                    raise RuntimeError(msg)
                logger.warning(msg)

            result = await fn(*args, **kwargs)

            try:
                result._feedback_loop_report = report  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                pass

            return result

        return wrapper  # type: ignore[return-value]

    return decorator
