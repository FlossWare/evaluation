"""Feedback-loop detection for evaluation-ai.

Analyses model usage data for self-referential feedback loops across
four layers: model dominance, evaluator-generator coupling, reward
hacking, and concept collapse.  Uses only the standard library.

Classes
-------
SimpleFeedbackLoopDetector -- in-memory feedback-loop detector
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from evaluation_ai.types import FeedbackLoopReport, FeedbackLoopRisk

# -- defaults ---------------------------------------------------------------

_DEFAULT_DOMINANCE_THRESHOLD = 0.70
_DEFAULT_COUPLING_THRESHOLD = 0.40
_SEVERITY_HIGH = 0.8
_SEVERITY_MEDIUM = 0.5


class SimpleFeedbackLoopDetector:
    """In-memory feedback-loop detector.

    Parameters
    ----------
    usage_data:
        Optional pre-loaded list of usage records.  Each record is a
        ``dict`` with at least ``"model"`` and ``"role"`` keys.
    usage_fetcher:
        Optional async callable that returns a list of usage records.
        Called by :meth:`analyze` to obtain fresh data when provided.
    dominance_threshold:
        Fraction above which a single model is flagged for dominance
        (default 0.70).
    coupling_threshold:
        Fraction above which a model evaluating its own output is
        flagged for eval-generator coupling (default 0.40).
    """

    def __init__(
        self,
        usage_data: list[dict[str, Any]] | None = None,
        usage_fetcher: Callable[..., Any] | None = None,
        *,
        dominance_threshold: float = _DEFAULT_DOMINANCE_THRESHOLD,
        coupling_threshold: float = _DEFAULT_COUPLING_THRESHOLD,
    ) -> None:
        self._usage_data: list[dict[str, Any]] = list(usage_data or [])
        self._usage_fetcher = usage_fetcher
        self._dominance_threshold = dominance_threshold
        self._coupling_threshold = coupling_threshold

    # -- public helpers ------------------------------------------------------

    def record_usage(self, model: str, role: str) -> None:
        """Append a usage record for later analysis.

        Parameters
        ----------
        model:
            Identifier of the model (e.g. ``"gpt-4o"``).
        role:
            Role the model played: ``"generator"``, ``"evaluator"``, etc.
        """
        self._usage_data.append({"model": model, "role": role})

    # -- protocol methods ----------------------------------------------------

    async def analyze(self, *, window_days: int = 7) -> FeedbackLoopReport:
        """Run feedback-loop analysis over the given time window.

        If a *usage_fetcher* was supplied at construction time it is
        awaited first to refresh the data.

        Checks four layers (see class docstring).
        """
        if self._usage_fetcher is not None:
            try:
                fetched = await self._usage_fetcher()
            except Exception:
                fetched = None
            if fetched:
                self._usage_data = list(fetched)

        data = self._usage_data
        risks: list[FeedbackLoopRisk] = []

        risks.extend(self._check_model_dominance(data))
        risks.extend(self._check_eval_coupling(data))
        risks.extend(self._check_reward_hacking(data))
        risks.extend(self._check_concept_collapse(data))

        is_healthy = all(r.severity <= 0.6 for r in risks)
        return FeedbackLoopReport(
            is_healthy=is_healthy,
            risks=risks,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            window_days=window_days,
        )

    async def is_healthy(self) -> bool:
        """Return ``True`` if no risks have severity > 0.6."""
        report = await self.analyze()
        return report.is_healthy

    # -- internal checks -----------------------------------------------------

    def _check_model_dominance(
        self, data: list[dict[str, Any]]
    ) -> list[FeedbackLoopRisk]:
        """Layer 1: any single model exceeding the dominance threshold."""
        if not data:
            return []

        counts: Counter[str] = Counter(r.get("model", "unknown") for r in data)
        total = sum(counts.values())
        risks: list[FeedbackLoopRisk] = []

        for model, count in counts.most_common():
            ratio = count / total
            if ratio > self._dominance_threshold:
                risks.append(
                    FeedbackLoopRisk(
                        layer="model_dominance",
                        severity=min(ratio, 1.0),
                        description=(
                            f"Model '{model}' accounts for {ratio:.0%} of "
                            f"usage (threshold {self._dominance_threshold:.0%})"
                        ),
                        metric_value=ratio,
                        threshold=self._dominance_threshold,
                    )
                )

        return risks

    def _check_eval_coupling(
        self, data: list[dict[str, Any]]
    ) -> list[FeedbackLoopRisk]:
        """Layer 2: same model acting as both generator and evaluator."""
        generators: set[str] = set()
        evaluators: set[str] = set()
        role_counts: Counter[tuple[str, str]] = Counter()

        for record in data:
            model = record.get("model", "unknown")
            role = record.get("role", "unknown")
            role_counts[(model, role)] += 1
            if role == "generator":
                generators.add(model)
            elif role == "evaluator":
                evaluators.add(model)

        overlap = generators & evaluators
        if not overlap or not data:
            return []

        total_evals = sum(
            c for (_, role), c in role_counts.items() if role == "evaluator"
        )
        if total_evals == 0:
            return []

        risks: list[FeedbackLoopRisk] = []
        for model in overlap:
            self_evals = role_counts.get((model, "evaluator"), 0)
            ratio = self_evals / total_evals
            if ratio > self._coupling_threshold:
                risks.append(
                    FeedbackLoopRisk(
                        layer="eval_coupling",
                        severity=min(_SEVERITY_HIGH, ratio + 0.2),
                        description=(
                            f"Model '{model}' evaluates its own outputs "
                            f"({ratio:.0%} of evaluations, threshold "
                            f"{self._coupling_threshold:.0%})"
                        ),
                        metric_value=ratio,
                        threshold=self._coupling_threshold,
                    )
                )

        return risks

    def _check_reward_hacking(
        self, data: list[dict[str, Any]]
    ) -> list[FeedbackLoopRisk]:
        """Layer 3: quality metrics rising while diversity drops.

        This layer requires records to carry ``"quality"`` and
        ``"diversity"`` numeric fields.  When absent, no risk is
        reported (insufficient data).
        """
        quality_vals = [r["quality"] for r in data if "quality" in r]
        diversity_vals = [r["diversity"] for r in data if "diversity" in r]

        if len(quality_vals) < 2 or len(diversity_vals) < 2:
            return []

        quality_trend = quality_vals[-1] - quality_vals[0]
        diversity_trend = diversity_vals[-1] - diversity_vals[0]

        if quality_trend > 0 and diversity_trend < 0:
            return [
                FeedbackLoopRisk(
                    layer="reward_hacking",
                    severity=_SEVERITY_MEDIUM,
                    description=(
                        "Quality increasing while diversity decreasing "
                        "-- possible reward hacking"
                    ),
                    metric_value=abs(diversity_trend),
                    threshold=0.0,
                )
            ]
        return []

    def _check_concept_collapse(
        self, _data: list[dict[str, Any]]
    ) -> list[FeedbackLoopRisk]:
        """Layer 4: placeholder -- always passes.

        Full implementation requires embedding similarity computation
        which is deferred to a future release.
        """
        return []
