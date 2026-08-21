"""Multi-model evaluation harness for evaluation-ai.

Fans out evaluation prompts to multiple models in parallel using
``asyncio.gather``, parses numeric scores from responses, and
derives a verdict based on score thresholds.

Zero external dependencies -- uses only the standard library.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from evaluation_ai.types import ChatMessage, EvaluationResult

if TYPE_CHECKING:
    from evaluation_ai.protocol import LLMBackend

# Score dimensions each evaluator model is asked to assess.
_DIMENSIONS = ("correctness", "completeness", "quality")

_EVALUATION_PROMPT_TEMPLATE = """\
You are an evaluation judge. Score the following output against the task.

Task: {task}

Output to evaluate:
{output}

Score each dimension on a 1-5 integer scale (1=poor, 5=excellent):
- correctness: Does the output correctly address the task?
- completeness: Does the output fully cover what was asked?
- quality: Is the output well-structured and clear?

Respond with EXACTLY three lines in this format (no extra text):
correctness: <score>
completeness: <score>
quality: <score>
"""

# Pattern to extract "dimension: score" lines from model responses.
_SCORE_PATTERN = re.compile(
    r"^(correctness|completeness|quality)\s*:\s*([1-5])\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_scores(text: str) -> dict[str, int]:
    """Extract dimension scores from a model response.

    Returns a dict mapping dimension name to integer score (1-5).
    Only recognised dimensions with valid scores are included.
    """
    scores: dict[str, int] = {}
    for match in _SCORE_PATTERN.finditer(text):
        dimension = match.group(1).lower()
        score = int(match.group(2))
        scores[dimension] = score
    return scores


def _verdict_from_average(avg: float) -> str:
    """Map a numeric average to a verdict string."""
    if avg >= 4.0:
        return "ACCEPT"
    if avg >= 2.5:
        return "ACCEPT_WITH_RESERVATIONS"
    return "REJECT"


class SimpleEvaluationHarness:
    """Multi-model evaluation harness using an ``LLMBackend``.

    Parameters
    ----------
    backend:
        Any object satisfying the ``LLMBackend`` protocol.  When
        ``None``, :meth:`evaluate` returns a default ACCEPT result
        (useful for testing without an API).
    """

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    async def evaluate(
        self,
        output: Any,
        *,
        task: str,
        models: list[str],
    ) -> EvaluationResult:
        """Evaluate *output* against *task* using the specified *models*.

        Each model is prompted in parallel to score the output on
        correctness, completeness, and quality (1-5 scale).  Scores are
        averaged across all models that return parseable responses.

        Verdict thresholds:

        - avg >= 4.0 -- ``"ACCEPT"``
        - avg >= 2.5 -- ``"ACCEPT_WITH_RESERVATIONS"``
        - avg < 2.5  -- ``"REJECT"``
        """
        if self._backend is None:
            return EvaluationResult(
                verdict="ACCEPT",
                scores=dict.fromkeys(_DIMENSIONS, 5),
                reasoning="No backend configured; returning default ACCEPT.",
                evaluator_models=[],
            )

        prompt_content = _EVALUATION_PROMPT_TEMPLATE.format(
            task=task,
            output=output,
        )
        messages = [ChatMessage(role="user", content=prompt_content)]

        async def _evaluate_with_model(model: str) -> dict[str, int] | None:
            try:
                response = await self._backend.chat(
                    messages,
                    model=model,
                    temperature=0.3,
                )
                scores = _parse_scores(response.content)
                if scores:
                    return scores
            except Exception:
                pass
            return None

        raw_results = await asyncio.gather(*[_evaluate_with_model(m) for m in models])

        # Collect only successful parses.
        per_model_scores: list[dict[str, int]] = []
        successful_models: list[str] = []
        for model, result in zip(models, raw_results):
            if result is not None:
                per_model_scores.append(result)
                successful_models.append(model)

        if not per_model_scores:
            return EvaluationResult(
                verdict="REJECT",
                scores={},
                reasoning="No evaluator models returned parseable scores.",
                evaluator_models=[],
            )

        # Average each dimension across models.
        averaged: dict[str, float] = {}
        for dim in _DIMENSIONS:
            values = [s[dim] for s in per_model_scores if dim in s]
            if values:
                averaged[dim] = sum(values) / len(values)

        if not averaged:
            return EvaluationResult(
                verdict="REJECT",
                scores={},
                reasoning="No valid dimension scores found.",
                evaluator_models=successful_models,
            )

        overall_avg = sum(averaged.values()) / len(averaged)
        verdict = _verdict_from_average(overall_avg)

        reasoning_parts = [
            f"{dim}={averaged[dim]:.1f}" for dim in _DIMENSIONS if dim in averaged
        ]
        reasoning = (
            f"Average scores: {', '.join(reasoning_parts)}. "
            f"Overall average: {overall_avg:.2f}."
        )

        return EvaluationResult(
            verdict=verdict,
            scores=averaged,
            reasoning=reasoning,
            evaluator_models=successful_models,
        )
