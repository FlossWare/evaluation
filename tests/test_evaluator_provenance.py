"""Tests for evaluator provenance in aggregate results."""

from __future__ import annotations

import asyncio

from evaluation_ai.evaluation import SimpleEvaluationHarness
from evaluation_ai.types import ChatResponse


class FakeBackend:
    async def chat(self, messages, *, model="", temperature=1.0, **kwargs):
        return ChatResponse(
            content="correctness: 5\ncompleteness: 4\nquality: 5",
            model=model,
            provider="ollama",
            usage={"eval_count": 11},
            model_identity={
                "runtime": "ollama",
                "runtime_version": "0.12.6",
                "model_reference": model,
                "digest": "sha256:" + model,
                "identity_complete": True,
            },
        )


def test_evaluation_result_preserves_model_identity():
    result = asyncio.run(
        SimpleEvaluationHarness(FakeBackend()).evaluate(
            "The answer",
            task="Answer the question",
            models=["qwen3.5:2b"],
        )
    )
    assert result.evaluator_models == ["qwen3.5:2b"]
    assert result.evaluator_metadata[0]["model_identity"]["digest"] == "sha256:qwen3.5:2b"
    assert result.evaluator_metadata[0]["usage"]["eval_count"] == 11
