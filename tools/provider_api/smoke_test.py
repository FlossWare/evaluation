#!/usr/bin/env python3
"""Minimal, non-streaming provider inference smoke tests.

Uses only Python's standard library. Credentials prefer *_API_KEY_FLOSSWARE and
fall back to conventional provider/SDK environment names. They are never accepted
as command-line arguments or emitted. No automatic retries are performed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    {"provider": "GOOGLE", "model": "models/gemini-3.5-flash-lite", "access": "free-tier-candidate"},
    {"provider": "MISTRAL", "model": "mistral-small-latest", "access": "free-tier-candidate"},
    {"provider": "COHERE", "model": "command-r7b-12-2024", "access": "free-trial-candidate"},
    {"provider": "OPENROUTER", "model": "nvidia/nemotron-3.5-lightning:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "deepseek/deepseek-v4-flash-0731:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "google/gemma-4-31b-it:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "thinkingmachines/inkling:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "thinkingmachines/inkling-small:free", "access": "catalogue-zero-price"},
    {"provider": "OPENROUTER", "model": "openrouter/free", "access": "catalogue-zero-price-router"},
    {"provider": "DEEPINFRA", "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "access": "paid"},
    {"provider": "NVIDIA", "model": "openai/gpt-oss-120b", "access": "deprecated-endpoint"},
    {"provider": "HUGGINGFACE", "model": "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", "access": "free-monthly-credit"},
    {"provider": "HUGGINGFACE", "model": "openai/gpt-oss-120b:fastest", "access": "paid-or-account-credit"},
    {"provider": "CEREBRAS", "model": "gpt-oss-120b", "access": "promotional-or-paid"},
    {"provider": "CEREBRAS", "model": "qwen-3.8-27b", "access": "promotional-or-paid"},
    {"provider": "DEEPSEEK", "model": "deepseek-flash", "access": "paid-unless-account-credit"},
    {"provider": "GROQ", "model": "openai/gpt-oss-120b", "access": "paid"},
    {"provider": "OPENAI", "model": "gpt-4.1-mini", "access": "paid"},
]

OPENAI_COMPATIBLE = {
    "CEREBRAS": "https://api.cerebras.ai/v1/chat/completions",
    "DEEPINFRA": "https://api.deepinfra.com/v1/openai/chat/completions",
    "NVIDIA": "https://integrate.api.nvidia.com/v1/chat/completions",
    "HUGGINGFACE": "https://router.huggingface.co/v1/chat/completions",
    "DEEPSEEK": "https://api.deepseek.com/chat/completions",
    "MISTRAL": "https://api.mistral.ai/v1/chat/completions",
    "GROQ": "https://api.groq.com/openai/v1/chat/completions",
    "OPENROUTER": "https://openrouter.ai/api/v1/chat/completions",
    "OPENAI": "https://api.openai.com/v1/chat/completions",
}
ENV_VARS = {
    "CEREBRAS": "CEREBRAS_API_KEY_FLOSSWARE",
    "DEEPINFRA": "DEEPINFRA_API_KEY_FLOSSWARE",
    "NVIDIA": "NVIDIA_API_KEY_FLOSSWARE",
    "HUGGINGFACE": "HUGGINGFACE_API_KEY_FLOSSWARE",
    "DEEPSEEK": "DEEPSEEK_API_KEY_FLOSSWARE",
    "GOOGLE": "GOOGLE_API_KEY_FLOSSWARE",
    "GROQ": "GROQ_API_KEY_FLOSSWARE",
    "OPENROUTER": "OPENROUTER_API_KEY_FLOSSWARE",
    "MISTRAL": "MISTRAL_API_KEY_FLOSSWARE",
    "COHERE": "COHERE_API_KEY_FLOSSWARE",
    "OPENAI": "OPENAI_API_KEY_FLOSSWARE",
    "ANTHROPIC": "ANTHROPIC_API_KEY_FLOSSWARE",
}

# Accept standard provider names and common SDK conventions.
# The FlossWare-specific variable takes precedence when both are present.
ENV_VAR_ALIASES = {
    "CEREBRAS": ("CEREBRAS_API_KEY",),
    "DEEPINFRA": ("DEEPINFRA_API_KEY",),
    "NVIDIA": ("NVIDIA_API_KEY",),
    "HUGGINGFACE": ("HUGGINGFACE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "DEEPSEEK": ("DEEPSEEK_API_KEY",),
    "GOOGLE": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "GROQ": ("GROQ_API_KEY",),
    "OPENROUTER": ("OPENROUTER_API_KEY",),
    "MISTRAL": ("MISTRAL_API_KEY",),
    "COHERE": ("COHERE_API_KEY",),
    "OPENAI": ("OPENAI_API_KEY",),
    "ANTHROPIC": ("ANTHROPIC_API_KEY",),
}


def provider_from_env_name(env_name: str) -> str | None:
    """Map a provider credential variable name to its canonical provider."""
    explicit = {
        "GEMINI_API_KEY": "GOOGLE",
        "GEMINI_API_KEY_FLOSSWARE": "GOOGLE",
        "HF_TOKEN": "HUGGINGFACE",
        "HUGGING_FACE_HUB_TOKEN": "HUGGINGFACE",
    }
    if env_name in explicit:
        return explicit[env_name]
    match = re.fullmatch(r"([A-Z][A-Z0-9_]*)_(?:API_KEY(?:_FLOSSWARE)?|API_TOKEN|AUTH_TOKEN)", env_name)
    if not match:
        return None
    provider = match.group(1)
    return {"HF": "HUGGINGFACE", "HUGGING_FACE": "HUGGINGFACE", "GEMINI": "GOOGLE"}.get(provider, provider)


def discovered_provider_names() -> list[str]:
    """Return known providers plus providers inferred from non-empty credential variables."""
    discovered = {
        provider
        for name, value in os.environ.items()
        if value.strip() and (provider := provider_from_env_name(name))
    }
    return sorted(set(ENV_VARS) | discovered)


def credential_variable_names(provider: str) -> tuple[str, ...]:
    """Return ordered credential variable names for a provider, without reading values."""
    primary = ENV_VARS.get(provider, f"{provider}_API_KEY_FLOSSWARE")
    names = [primary, *ENV_VAR_ALIASES.get(provider, ())]
    names.extend(
        name for name in os.environ
        if provider_from_env_name(name) == provider
    )
    return tuple(dict.fromkeys(names))


def resolve_credential(provider: str) -> tuple[str, str | None]:
    """Return credential and source variable name without exposing its value."""
    for env_name in credential_variable_names(provider):
        value = os.environ.get(env_name, "")
        if value.strip():
            return value.strip(), env_name
    return "", None


def safe_text(value: Any, secret: str) -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:2000]


def classify_http_error(status: int | None, detail: str | None) -> str | None:
    """Classify provider HTTP errors without confusing billing with inference failure."""
    if status != 402:
        return None
    message = (detail or "").lower()
    if "team" in message and ("subscription" in message or "credit" in message):
        return "organization-subscription-or-credit-required"
    if "insufficient balance" in message or "no balance" in message:
        return "insufficient-account-balance"
    if any(term in message for term in ("credit", "billing", "payment required", "payment_required")):
        return "account-credit-or-billing-required"
    return "payment-required-unclassified"


def error_summary(detail: str | None) -> str | None:
    """Extract a compact provider message for console output; never print credentials."""
    if not detail:
        return None
    message = detail
    try:
        parsed = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("detail") or error)
        elif isinstance(error, str):
            message = error
        else:
            message = str(parsed.get("message") or parsed.get("detail") or detail)
    return " ".join(message.split())[:240]


def build_headers(provider: str, key: str) -> dict[str, str]:
    """Build explicit application-identifying headers for provider API requests."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Cloudflare Error 1010 can reject urllib's default Python-urllib user agent.
        # Identify the client honestly; do not impersonate a browser.
        "User-Agent": "FlossWare-ProviderSmoke/0.1",
    }
    if provider == "GOOGLE":
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    if provider == "COHERE":
        headers["X-Client-Name"] = "FlossWare provider API smoke test"
    if provider == "OPENROUTER":
        headers["HTTP-Referer"] = "https://flossware.org"
        headers["X-Title"] = "FlossWare provider API smoke test"
    return headers


def classify_response(text: str, finish_reason: str | None, reasoning_tokens: int) -> tuple[str, str, bool]:
    """Return (completeness, status, semantic_match) for one model response."""
    semantic_match = text.strip() == EXPECTED
    hit_output_limit = str(finish_reason or "").upper() in {"LENGTH", "MAX_TOKENS"}

    # A length stop takes precedence over text matching: partial output must
    # never be reported as complete, even when it contains some visible text.
    if hit_output_limit:
        status = "incomplete-reasoning" if reasoning_tokens else "incomplete-output"
        return "incomplete", status, semantic_match
    if semantic_match:
        return "complete", "passed", True
    if not text.strip():
        return "empty", "empty-response", False
    return "complete", "response-mismatch", False


DEPRECATED_ACCESS = {"deprecated-endpoint"}


PAID_ACCESS = {
    "paid", "paid-or-account-credit", "paid-unless-account-credit",
    "promotional-or-paid",
}


def representative_cases(include_paid: bool) -> dict[str, dict[str, str]]:
    """Choose at most one eligible catalogue candidate per provider."""
    selected: dict[str, dict[str, str]] = {}
    for case in CASES:
        if case["access"] not in DEPRECATED_ACCESS and (include_paid or case["access"] not in PAID_ACCESS):
            selected.setdefault(case["provider"], case)
    return selected


def plan_all_configured(include_paid: bool = False) -> list[dict[str, Any]]:
    """Build a complete provider plan without making network requests."""
    selected = representative_cases(include_paid)
    plan: list[dict[str, Any]] = []
    for provider in discovered_provider_names():
        _, env_name = resolve_credential(provider)
        candidates = [case for case in CASES if case["provider"] == provider]
        case = selected.get(provider)
        item: dict[str, Any] = {
            "provider": provider,
            "credential_env": env_name,
            "credential_configured": bool(env_name),
            "model_requested": case["model"] if case else None,
            "access_classification": case["access"] if case else None,
        }
        if not env_name:
            item["status"] = "skipped-no-credential"
        elif case:
            item["status"] = "ready"
        elif candidates and all(c["access"] in DEPRECATED_ACCESS for c in candidates):
            item["status"] = "skipped-deprecated-endpoint"
            item["model_requested"] = candidates[0]["model"]
            item["access_classification"] = candidates[0]["access"]
        elif provider not in {"GOOGLE", "COHERE", *OPENAI_COMPATIBLE}:
            item["status"] = "skipped-no-provider-adapter"
        elif candidates and not include_paid and all(c["access"] in PAID_ACCESS or c["access"] in DEPRECATED_ACCESS for c in candidates):
            item["status"] = "skipped-paid-opt-in-required"
            eligible = next((c for c in candidates if c["access"] in PAID_ACCESS), candidates[0])
            item["model_requested"] = eligible["model"]
            item["access_classification"] = eligible["access"]
        else:
            item["status"] = "skipped-no-chat-candidate"
        plan.append(item)
    return plan


def request_case(case: dict[str, str], timeout: float, max_tokens: int) -> dict[str, Any]:
    provider, model = case["provider"], case["model"]
    key, env_name = resolve_credential(provider)
    result: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model_requested": model,
        "requested_max_output_tokens": max_tokens,
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
    elif provider == "COHERE":
        url = "https://api.cohere.com/v2/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    else:
        url = OPENAI_COMPATIBLE[provider]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    headers = build_headers(provider, key)

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
        elif provider == "COHERE":
            message = data.get("message") or {}
            content = message.get("content") or []
            text = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            result["model_returned"] = data.get("model") or model
            result["finish_reason"] = data.get("finish_reason")
            result["usage"] = data.get("usage")
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

        # Preserve provider-native usage while exposing normalized reasoning usage.
        usage = result["usage"] or {}
        if provider == "GOOGLE":
            reasoning_tokens = usage.get("thoughtsTokenCount", 0)
        elif provider == "COHERE":
            reasoning_tokens = 0
        else:
            completion_details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = completion_details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
        result["reasoning_tokens"] = reasoning_tokens or 0
        result["visible_output_chars"] = len(text.strip())

        (
            result["response_completeness"],
            result["status"],
            result["semantic_match"],
        ) = classify_response(
            text,
            result["finish_reason"],
            result["reasoning_tokens"],
        )
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
        result["http_error_category"] = classify_http_error(exc.code, result["error"])
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
    parser.add_argument("--provider", action="append", choices=discovered_provider_names(),
                        help="Run only this provider; repeat to select several.")
    parser.add_argument("--all-configured", action="store_true",
                        help="Iterate every known provider; run one free/trial candidate where available and report all others as skipped.")
    parser.add_argument("--model",
                        help="Run only this exact configured model ID; combine with --provider to validate its provider.")
    parser.add_argument("--include-paid", action="store_true",
                        help="Also run paid/promotional-credit comparison candidates.")
    parser.add_argument("--show-latest-errors", action="store_true",
                        help="Print the latest recorded errors for Cerebras, DeepInfra, DeepSeek, and Hugging Face without making API calls.")
    parser.add_argument("--show-credential-sources", action="store_true",
                        help="Show which credential variable name is selected for each provider, without printing values or making API calls.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-output-tokens", type=int, default=128,
                        help="Maximum generated tokens per request (default: 128).")
    parser.add_argument("--output", type=Path, default=Path("provider-smoke-results.jsonl"))
    args = parser.parse_args()

    if args.timeout <= 0 or args.max_output_tokens < 1:
        parser.error("--timeout must be positive and --max-output-tokens must be at least 1")

    if args.show_credential_sources:
        if args.show_latest_errors or args.all_configured or args.provider or args.model or args.include_paid:
            parser.error("--show-credential-sources cannot be combined with other selection options")
        for provider in discovered_provider_names():
            _, env_name = resolve_credential(provider)
            print(f"{provider:10} {env_name or 'not configured'}")
        print("Read-only: no credential values were displayed and no provider requests were made.")
        return 0

    if args.show_latest_errors:
        if args.all_configured or args.provider or args.model or args.include_paid or args.show_credential_sources:
            parser.error("--show-latest-errors cannot be combined with inference-selection options")
        if not args.output.exists():
            parser.error(f"No evidence file found: {args.output}")
        wanted = {"CEREBRAS", "DEEPINFRA", "DEEPSEEK", "HUGGINGFACE"}
        latest: dict[str, dict[str, Any]] = {}
        with args.output.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                provider = item.get("provider")
                if provider in wanted and (item.get("error") or item.get("http_status") == 402):
                    latest[provider] = item
        for provider in sorted(wanted):
            item = latest.get(provider)
            if not item:
                print(f"{provider:10} no recorded error")
                continue
            detail = error_summary(item.get("error")) or "No response body recorded"
            category = item.get("http_error_category") or classify_http_error(
                item.get("http_status"), item.get("error")
            ) or "provider-error"
            print(
                f'{provider:10} {item.get("model_requested") or "-"} '
                f'HTTP={item.get("http_status") or "-"} [{category}]'
            )
            print(f"  {detail}")
        print("Read-only: no provider requests were made.")
        return 0

    if args.all_configured and (args.provider or args.model):
        parser.error("--all-configured cannot be combined with --provider or --model")

    # Validate exact model selection against the configured case catalogue first.
    # This keeps --model narrow and prevents arbitrary model IDs from reaching providers.
    if args.model:
        matching_models = [case for case in CASES if case["model"] == args.model]
        if not matching_models:
            parser.error(f"Model is not configured in this smoke-test catalogue: {args.model}")
        if args.provider and not any(case["provider"] in args.provider for case in matching_models):
            configured = ", ".join(sorted({case["provider"] for case in matching_models}))
            parser.error(f"Model {args.model!r} is configured for {configured}, not the selected provider(s).")

    if args.all_configured:
        # Plan every provider first. Only free/trial/zero-price candidates run
        # by default; paid-only and deprecated providers are explicitly recorded.
        # At most one candidate per provider is ever invoked in this batch.
        results = []
        for item in plan_all_configured(args.include_paid):
            if item["status"] == "ready":
                case = next(
                    c for c in CASES
                    if c["provider"] == item["provider"] and c["model"] == item["model_requested"]
                )
                results.append(request_case(case, args.timeout, args.max_output_tokens))
            else:
                item.update({
                    "http_status": None,
                    "latency_seconds": None,
                    "retry_performed": False,
                })
                results.append(item)
    else:
        selected = [
            case for case in CASES
            if (not args.provider or case["provider"] in args.provider)
            and (not args.model or case["model"] == args.model)
            and case["access"] not in DEPRECATED_ACCESS
            and (args.include_paid or case["access"] not in PAID_ACCESS)
        ]
        if not selected and args.model:
            parser.error(f"Model {args.model!r} is excluded from the default batch; use --include-paid if it is a paid/promotional candidate.")
        if not selected:
            parser.error("No test cases selected.")
        results = [request_case(case, args.timeout, args.max_output_tokens) for case in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    for item in results:
        status = item["status"]
        http = item["http_status"] if item["http_status"] is not None else "-"
        latency = item["latency_seconds"] if item["latency_seconds"] is not None else "-"
        model_label = item["model_requested"] or "-"
        print(f'{item["provider"]:10} {model_label:46} {status:30} HTTP={http} latency={latency}s')
        detail = error_summary(item.get("error"))
        if detail:
            category = item.get("http_error_category") or "provider-error"
            print(f"  detail [{category}]: {detail}")

    passed = sum(item["status"] == "passed" for item in results)
    skipped = sum(item["status"].startswith("skipped-") for item in results)
    failed = len(results) - passed - skipped
    if args.all_configured:
        by_reason = {
            reason: sum(item["status"] == reason for item in results)
            for reason in (
                "skipped-no-credential",
                "skipped-paid-opt-in-required",
                "skipped-deprecated-endpoint",
                "skipped-no-chat-candidate",
                "skipped-no-provider-adapter",
            )
        }
        print(f"\nProvider inventory: {len(results)} providers examined.")
        print(f"Eligible requests: {len(results) - skipped}; passed: {passed}; failed: {failed}.")
        for reason, count in by_reason.items():
            if count:
                print(f"  {reason}: {count}")
    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped; {len(results)} selected.")
    print(f"Evidence: {args.output}")
    print("No retries performed. Review provider billing/usage independently; catalogue price is not a bill.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
