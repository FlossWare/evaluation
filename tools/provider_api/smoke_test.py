#!/usr/bin/env python3
"""Minimal, non-streaming provider inference smoke tests.

Uses only Python's standard library. Credentials are read from *_API_KEY_FLOSSWARE
environment variables, never accepted as command-line arguments, and never emitted.
No automatic retries are performed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT = "Reply with exactly this token and nothing else: FLOSSWARE_PROVIDER_SMOKE_OK"
EXPECTED = "FLOSSWARE_PROVIDER_SMOKE_OK"

# The default batch is restricted to catalogue-marked OpenRouter :free models and
# Gemini models visible in the authenticated catalogue. Paid/promotional-credit
# comparisons require --include-paid.
CASES: list[dict[str, str]] = [
    {"provider": "GOOGLE", "model": "models/gemini-2.5-flash-lite", "access": "free-tier-candidate"},
    {"provider": "GOOGLE", "model": "models/gemini-2.5-flash", "access": "free-tier-candidate"},
    {"provider": "OPENROUTER", "model": "nvidia/nemotron-3.5-lightning:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "google/gemma-4-31b-it:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "thinkingmachines/inkling:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "thinkingmachines/inkling-small:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "openrouter/free", "access": "catalogue-zero-price-router"},
    {"provider": "CEREBRAS", "model": "gpt-oss-120b", "access": "promotional-or-paid"},
    {"provider": "CEREBRAS", "model": "qwen-3.8-27b", "access": "promotional-or-paid"},
    {"provider": "DEEPSEEK", "model": "deepseek-flash", "access": "paid-unless-account-credit"},
    {"provider": "GROQ", "model": "openai/gpt-oss-120b", "access": "paid"},
]

OPENAI_COMPATIBLE = {
    "CEREBRAS": "https://api.cerebras.ai/v1/chat/completions",
    "DEEPSEEK": "https://api.deepseek.com/chat/completions",
    "GROQ": "https://api.groq.com/openai/v1/chat/completions",
    "OPENROUTER": "https://openrouter.ai/api/v1/chat/completions",
}
ENV_VARS = {
    "CEREBRAS": "CEREBRAS_API_KEY_FLOSSWARE",
    "DEEPSEEK": "DEEPSEEK_API_KEY_FLOSSWARE",
    "GOOGLE": "GOOGLE_API_KEY_FLOSSWARE",
    "GROQ": "GROQ_API_KEY_FLOSSWARE",
    "OPENROUTER": "OPENROUTER_API_KEY_FLOSSWARE",
}


def safe_text(value: Any, secret: str) -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:2000]


def request_case(case: dict[str, str], timeout: float, max_tokens: int) -> dict[str, Any]:
    provider, model = case["provider"], case["model"]
    env_name = ENV_VARS[provider]
    key = os.environ.get(env_name, "")
    result: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model_requested": model,
        "access_classification": case["access"],
        "credential_env": env_name,
        "credential_configured": bool(key),
        "http_status": None,
        "latency_seconds": None,
        "provider_request_id": None,
        "model_returned": None,
        "finish_reason": None,
        "usage": None,
        "response_text": None,
        "semantic_match": None,
        "rate_limit_headers": {},
        "error": None,
        "retry_performed": False,
    }
    if not key:
        result["status"] = "skipped-no-credential"
        return result

    if provider == "GOOGLE":
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    else:
        url = OPENAI_COMPATIBLE[provider]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if provider == "OPENROUTER":
            headers["HTTP-Referer"] = "https://flossware.org"
            headers["X-Title"] = "FlossWare provider API smoke test"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body_bytes = response.read()
            result["http_status"] = response.status
            result["rate_limit_headers"] = {
                name.lower(): value
                for name, value in response.headers.items()
                if "ratelimit" in name.lower() or name.lower() in {
                    "retry-after", "x-request-id", "request-id", "cf-ray"
                }
            }
            result["provider_request_id"] = (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or response.headers.get("cf-ray")
            )
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        data = json.loads(body_bytes.decode("utf-8"))

        if provider == "GOOGLE":
            candidates = data.get("candidates") or []
            candidate = candidates[0] if candidates else {}
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            result["model_returned"] = data.get("modelVersion") or model
            result["finish_reason"] = candidate.get("finishReason")
            result["usage"] = data.get("usageMetadata")
        else:
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            text = message.get("content") or ""
            result["model_returned"] = data.get("model") or model
            result["finish_reason"] = choice.get("finish_reason")
            result["usage"] = data.get("usage")
            result["provider_request_id"] = result["provider_request_id"] or data.get("id")

        result["response_text"] = safe_text(text, key)
        result["semantic_match"] = text.strip() == EXPECTED

        # Preserve provider-native usage while exposing a normalized reasoning count.
        usage = result["usage"] or {}
        if provider == "GOOGLE":
            reasoning_tokens = usage.get("thoughtsTokenCount", 0)
        else:
            completion_details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = completion_details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
        result["reasoning_tokens"] = reasoning_tokens or 0
        result["visible_output_chars"] = len(text.strip())

        if result["semantic_match"]:
            result["response_completeness"] = "complete"
            result["status"] = "passed"
        elif not text.strip() and result["finish_reason"] in {"length", "MAX_TOKENS"}:
            result["response_completeness"] = "incomplete"
            result["status"] = (
                "incomplete-reasoning" if result["reasoning_tokens"] else "incomplete-output"
            )
        elif not text.strip():
            result["response_completeness"] = "empty"
            result["status"] = "empty-response"
        else:
            result["response_completeness"] = "complete"
            result["status"] = "response-mismatch"
    except urllib.error.HTTPError as exc:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        result["http_status"] = exc.code
        result["provider_request_id"] = (
            exc.headers.get("x-request-id")
            or exc.headers.get("request-id")
            or exc.headers.get("cf-ray")
        )
        result["rate_limit_headers"] = {
            name.lower(): value
            for name, value in exc.headers.items()
            if "ratelimit" in name.lower() or name.lower() in {"retry-after", "x-request-id", "request-id", "cf-ray"}
        }
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        result["error"] = safe_text(detail, key)
        result["status"] = "http-error"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        result["error"] = safe_text(exc, key)
        result["status"] = "transport-error"
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        result["error"] = safe_text(f"Response parsing failed: {type(exc).__name__}: {exc}", key)
        result["status"] = "response-parse-error"
    finally:
        # Avoid retaining the key in local references longer than necessary.
        key = ""
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=sorted(ENV_VARS),
                        help="Run only this provider; repeat to select several.")
    parser.add_argument("--include-paid", action="store_true",
                        help="Also run paid/promotional-credit comparison candidates.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-output-tokens", type=int, default=128,
                        help="Maximum generated tokens per request (default: 128).")
    parser.add_argument("--output", type=Path, default=Path("provider-smoke-results.jsonl"))
    args = parser.parse_args()

    if args.timeout <= 0 or args.max_output_tokens < 1:
        parser.error("--timeout must be positive and --max-output-tokens must be at least 1")

    selected = [
        case for case in CASES
        if (not args.provider or case["provider"] in args.provider)
        and (args.include_paid or case["access"] not in {
            "promotional-or-paid", "paid-unless-account-credit", "paid"
        })
    ]
    if not selected:
        parser.error("No test cases selected.")

    results = [request_case(case, args.timeout, args.max_output_tokens) for case in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    for item in results:
        status = item["status"]
        http = item["http_status"] if item["http_status"] is not None else "-"
        latency = item["latency_seconds"] if item["latency_seconds"] is not None else "-"
        print(f'{item["provider"]:10} {item["model_requested"]:46} {status:24} HTTP={http} latency={latency}s')

    passed = sum(item["status"] == "passed" for item in results)
    skipped = sum(item["status"] == "skipped-no-credential" for item in results)
    failed = len(results) - passed - skipped
    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped; {len(results)} selected.")
    print(f"Evidence: {args.output}")
    print("No retries performed. Review provider billing/usage independently; catalogue price is not a bill.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
