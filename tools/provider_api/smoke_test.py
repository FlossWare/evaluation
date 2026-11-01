#!/usr/bin/env python3
"""Minimal, non-streaming provider inference smoke tests.

Uses only Python's standard library. Credentials prefer *_API_KEY_FLOSSWARE and
fall back to conventional provider/SDK environment names. They are never accepted
as command-line arguments or emitted. No automatic retries are performed.
"""
from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_MAX_OUTPUT_TOKENS = 512

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


EMBEDDING_TEXT = (
    "FlossWare evaluates embedding APIs for semantic retrieval and vector search."
)
EMBEDDING_CASES: list[dict[str, str]] = [
    {"provider": "VOYAGEAI", "model": "voyage-4-lite", "access": "paid-or-account-credit"},
    {"provider": "JINA", "model": "jina-embeddings-v3", "access": "free-tier-candidate"},
    {"provider": "COHERE", "model": "embed-v4.0", "access": "free-trial-candidate"},
    {"provider": "GOOGLE", "model": "models/gemini-embedding-001", "access": "free-tier-candidate"},
    {"provider": "OPENAI", "model": "text-embedding-3-small", "access": "paid"},
    {"provider": "MISTRAL", "model": "mistral-embed", "access": "paid"},
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

PROVIDER_CATALOGUE_PATH = Path(__file__).with_name("provider-catalogue.json")
with PROVIDER_CATALOGUE_PATH.open(encoding="utf-8") as catalogue_stream:
    PROVIDER_CATALOGUE = json.load(catalogue_stream)


def provider_catalogue_entry(provider: str) -> dict[str, Any]:
    """Return metadata for a provider, or an explicit unknown-adapter record."""
    return PROVIDER_CATALOGUE.get("providers", {}).get(provider, {
        "adapter": None,
        "implemented_capabilities": [],
        "known_capabilities": [],
        "endpoint": None,
        "authentication": None,
        "operations": [],
        "status": "provider-not-catalogued",
    })


def inventory_rows() -> list[dict[str, Any]]:
    """Describe every known/discovered provider and each configured account without secrets."""
    rows: list[dict[str, Any]] = []
    for provider in discovered_provider_names():
        entry = provider_catalogue_entry(provider)
        accounts = credential_accounts(provider) or [{
            "account": "DEFAULT",
            "credential_env": None,
            "credential": "",
        }]
        for account in accounts:
            rows.append({
                "provider": provider,
                "account": account["account"],
                "credential_env": account["credential_env"],
                "credential_configured": bool(account["credential_env"]),
                "adapter": entry.get("adapter"),
                "implemented_capabilities": entry.get("implemented_capabilities", []),
                "known_capabilities": entry.get("known_capabilities", []),
                "endpoint": entry.get("endpoint"),
                "authentication": entry.get("authentication"),
                "status": (
                    "adapter-available" if account["credential_env"] and entry.get("adapter")
                    else "skipped-no-credential" if not account["credential_env"]
                    else entry.get("status", "skipped-no-provider-adapter")
                ),
            })
    return rows


def capability_plan(capability: str, providers: list[str] | None = None) -> list[dict[str, Any]]:
    """Build a no-network plan for a capability family not yet backed by adapters."""
    selected = set(providers or discovered_provider_names())
    rows: list[dict[str, Any]] = []
    for provider in sorted(selected):
        entry = provider_catalogue_entry(provider)
        accounts = credential_accounts(provider) or [{
            "account": "DEFAULT", "credential_env": None, "credential": ""
        }]
        for account in accounts:
            configured = bool(account["credential_env"])
            supported = capability in entry.get("implemented_capabilities", [])
            rows.append({
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "account": account["account"],
                "credential_env": account["credential_env"],
                "credential_configured": configured,
                "capability": capability,
                "adapter": entry.get("adapter"),
                "status": (
                    "skipped-no-credential" if not configured
                    else "ready" if supported
                    else "skipped-no-capability-adapter"
                ),
                "http_status": None,
                "latency_seconds": None,
                "retry_performed": False,
            })
    return rows


def print_inventory() -> None:
    """Print a safe provider/account inventory; never print credential values."""
    rows = inventory_rows()
    print(f"{'PROVIDER':16} {'ACCOUNT':10} {'CREDENTIAL':9} {'ADAPTER':24} CAPABILITIES")
    for item in rows:
        credential = "yes" if item["credential_configured"] else "no"
        adapter = item["adapter"] or "-"
        capabilities = ",".join(item["implemented_capabilities"]) or "-"
        print(
            f'{item["provider"]:16} {item["account"]:10} {credential:9} '
            f'{adapter:24} {capabilities}'
        )
        if item["credential_env"]:
            print(f'  source: {item["credential_env"]}')
        if item["endpoint"]:
            print(f'  endpoint: {item["endpoint"]}')
        if item["status"] not in {"ready"}:
            print(f'  status: {item["status"]}')
    print(f"\n{len(rows)} provider/account rows. Inventory only; no network requests or credential values.")


def print_report(path: Path) -> int:
    """Summarize accumulated JSONL evidence without making provider requests."""
    if not path.exists():
        print(f"No evidence file found: {path}", file=sys.stderr)
        return 2
    totals: dict[tuple[str, str, str, str], int] = {}
    errors: dict[tuple[str, str, str], int] = {}
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            provider = str(item.get("provider") or "UNKNOWN")
            account = str(item.get("account") or "DEFAULT")
            capability = str(item.get("capability") or "chat")
            status = str(item.get("status") or "unknown")
            totals[(provider, account, capability, status)] = totals.get((provider, account, capability, status), 0) + 1
            if status not in {"passed"} and not status.startswith("skipped-"):
                category = str(item.get("http_error_category") or status)
                errors[(provider, account, category)] = errors.get((provider, account, category), 0) + 1
            count += 1
    print(f"Evidence report: {path}")
    print(f"Records: {count}")
    print(f"{'PROVIDER':16} {'ACCOUNT':10} {'CAPABILITY':20} {'STATUS':34} COUNT")
    for (provider, account, capability, status), total in sorted(totals.items()):
        print(f"{provider:16} {account:10} {capability:20} {status:34} {total}")
    if errors:
        print("\nFailure categories:")
        for (provider, account, category), total in sorted(errors.items()):
            print(f"  {provider}/{account}: {category} ({total})")
    print("Read-only: no provider requests were made.")
    return 0



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
    match = re.fullmatch(
        r"([A-Z][A-Z0-9_]*)_(?:API_KEY(?:_(?:FLOSSWARE|HOTMAIL|NCRR))?|API_TOKEN|AUTH_TOKEN)",
        env_name,
    )
    if not match:
        return None
    provider = match.group(1)
    return {
        "HF": "HUGGINGFACE",
        "HUGGING_FACE": "HUGGINGFACE",
        "GEMINI": "GOOGLE",
        "THINKINGMACHINES": "THINKMACHINES",
    }.get(provider, provider)


def credential_account_from_env_name(env_name: str) -> str:
    """Return the account namespace encoded in a credential variable name."""
    match = re.fullmatch(r"[A-Z][A-Z0-9_]*_API_KEY_(FLOSSWARE|HOTMAIL|NCRR)", env_name)
    return match.group(1) if match else "DEFAULT"


def discovered_provider_names() -> list[str]:
    """Return known providers plus providers inferred from non-empty credential variables."""
    discovered = {
        provider
        for name, value in os.environ.items()
        if value.strip() and (provider := provider_from_env_name(name))
    }
    return sorted(set(ENV_VARS) | discovered)


def credential_variable_names(provider: str) -> tuple[str, ...]:
    """Return all configured credential variable names, with FlossWare first."""
    names = [name for name in os.environ if provider_from_env_name(name) == provider]
    aliases = ENV_VAR_ALIASES.get(provider, ())
    names.extend(name for name in aliases if name in os.environ)
    primary = ENV_VARS.get(provider)
    if primary and primary in os.environ:
        names.append(primary)
    rank = {"FLOSSWARE": 0, "HOTMAIL": 1, "NCRR": 2, "DEFAULT": 3}
    unique = dict.fromkeys(names)
    return tuple(sorted(
        unique,
        key=lambda name: (rank.get(credential_account_from_env_name(name), 3), name),
    ))


def credential_accounts(provider: str) -> list[dict[str, str]]:
    """Return each configured account credential separately; never merge accounts."""
    accounts: list[dict[str, str]] = []
    for env_name in credential_variable_names(provider):
        value = os.environ.get(env_name, "").strip()
        if value:
            accounts.append({
                "account": credential_account_from_env_name(env_name),
                "credential_env": env_name,
                "credential": value,
            })
    return accounts


def resolve_credential(provider: str) -> tuple[str, str | None]:
    """Return the preferred credential and source variable name."""
    accounts = credential_accounts(provider)
    if accounts:
        return accounts[0]["credential"], accounts[0]["credential_env"]
    return "", None


def safe_text(value: Any, secret: str) -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:2000]


def sanitize_error_detail(detail: str | None, secret: str = "") -> str | None:
    """Remove provider user identifiers and redact secrets before evidence is persisted."""
    if detail is None:
        return None
    try:
        parsed = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        return safe_text(detail, secret)

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, nested in value.items():
                normalized_key = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized_key in {"userid", "useridentifier"}:
                    continue
                cleaned[key] = scrub(nested)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str) and secret:
            return value.replace(secret, "[REDACTED]")
        return value

    sanitized = json.dumps(scrub(parsed), ensure_ascii=False, separators=(",", ":"))
    return safe_text(sanitized, secret)


def classify_http_error(status: int | None, detail: str | None) -> str | None:
    """Classify billing and rate-limit responses using provider metadata where available."""
    try:
        parsed = json.loads(detail or "")
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
    metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    if status == 403:
        message = str(metadata.get("raw") or error.get("message") or detail or "").lower()
        if (
            "only available on agentic harnesses" in message
            or "agentic harness" in message and "coding agent" in message
        ):
            return "agentic-harness-required"
        return None

    if status == 429:
        limit_source = str(metadata.get("limit_source") or "").lower()
        if limit_source == "upstream_provider_shared_pool":
            return "upstream-shared-pool-rate-limit"
        if metadata.get("is_byok") is True:
            return "byok-provider-rate-limit"
        if "upstream" in limit_source:
            return "upstream-provider-rate-limit"
        if limit_source:
            return "provider-rate-limit"
        return "rate-limit-unclassified"

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
            metadata = error.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            message = str(
                metadata.get("raw")
                or error.get("message")
                or error.get("detail")
                or error
            )
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


def openrouter_free_cases() -> list[dict[str, str]]:
    """Return explicit, catalogue-zero-price OpenRouter models (not the dynamic router alias)."""
    return [
        case for case in CASES
        if case["provider"] == "OPENROUTER"
        and case["access"] == "catalogue-zero-price"
    ]


def plan_all_configured(include_paid: bool = False) -> list[dict[str, Any]]:
    """Build a provider/account plan without making network requests."""
    selected = representative_cases(include_paid)
    plan: list[dict[str, Any]] = []
    for provider in discovered_provider_names():
        accounts = credential_accounts(provider)
        account_rows = accounts or [{
            "account": "DEFAULT",
            "credential_env": None,
            "credential": "",
        }]
        candidates = [case for case in CASES if case["provider"] == provider]
        case = selected.get(provider)
        for account in account_rows:
            env_name = account["credential_env"]
            item: dict[str, Any] = {
                "provider": provider,
                "account": account["account"],
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



def embedding_request_spec(case: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Build a provider-native embedding request without performing network I/O."""
    provider = case["provider"]
    model = case["model"]
    if provider == "GOOGLE":
        return (
            f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent",
            {
                "content": {"parts": [{"text": EMBEDDING_TEXT}]},
                "taskType": "RETRIEVAL_DOCUMENT",
            },
        )
    if provider == "COHERE":
        return (
            "https://api.cohere.com/v2/embed",
            {
                "model": model,
                "texts": [EMBEDDING_TEXT],
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
        )
    if provider == "VOYAGEAI":
        return (
            "https://api.voyageai.com/v1/embeddings",
            {
                "model": model,
                "input": [EMBEDDING_TEXT],
                "input_type": "document",
                "output_dtype": "float",
            },
        )
    if provider == "JINA":
        return (
            "https://api.jina.ai/v1/embeddings",
            {
                "model": model,
                "input": [EMBEDDING_TEXT],
                "task": "retrieval.passage",
                "embedding_type": "float",
                "normalized": True,
            },
        )
    if provider in {"OPENAI", "MISTRAL"}:
        endpoint = (
            "https://api.openai.com/v1/embeddings"
            if provider == "OPENAI"
            else "https://api.mistral.ai/v1/embeddings"
        )
        return (
            endpoint,
            {"model": model, "input": [EMBEDDING_TEXT], "encoding_format": "float"},
        )
    raise ValueError(f"No embedding request adapter for provider: {provider}")


def extract_embedding_vectors(provider: str, data: dict[str, Any]) -> list[list[float]]:
    """Normalize supported provider embedding response envelopes to float vectors."""
    if provider == "GOOGLE":
        values = (data.get("embedding") or {}).get("values")
        raw_vectors = [values] if values is not None else []
    elif provider == "COHERE":
        raw_vectors = (data.get("embeddings") or {}).get("float") or []
    else:
        raw_vectors = [
            item.get("embedding")
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("embedding") is not None
        ]

    vectors: list[list[float]] = []
    for vector in raw_vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding response contained an empty or invalid vector")
        normalized: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Embedding vector contained a non-numeric value")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Embedding vector contained a non-finite value")
            normalized.append(number)
        vectors.append(normalized)
    if not vectors:
        raise ValueError("Embedding response did not contain any vectors")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Embedding response contained inconsistent vector dimensions")
    if not any(value != 0.0 for value in vectors[0]):
        raise ValueError("Embedding response contained an all-zero vector")
    return vectors


def request_embedding_case(
    case: dict[str, str],
    timeout: float,
    credential_env: str | None = None,
    account: str = "DEFAULT",
) -> dict[str, Any]:
    """Run one bounded embedding smoke request and retain only vector metadata."""
    provider, model = case["provider"], case["model"]
    if credential_env:
        key = os.environ.get(credential_env, "").strip()
        env_name = credential_env
    else:
        key, env_name = resolve_credential(provider)
        if env_name:
            account = credential_account_from_env_name(env_name)

    result: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "capability": "embeddings",
        "account": account,
        "model_requested": model,
        "access_classification": case["access"],
        "credential_env": env_name,
        "credential_configured": bool(key),
        "http_status": None,
        "latency_seconds": None,
        "provider_request_id": None,
        "model_returned": None,
        "embedding_count": 0,
        "embedding_dimensions": None,
        "embedding_norm": None,
        "usage": None,
        "error": None,
        "retry_performed": False,
    }
    if not key:
        result["status"] = "skipped-no-credential"
        return result

    url, payload = embedding_request_spec(case)
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
        vectors = extract_embedding_vectors(provider, data)
        vector = vectors[0]
        result["model_returned"] = data.get("model") or data.get("modelVersion") or model
        result["embedding_count"] = len(vectors)
        result["embedding_dimensions"] = len(vector)
        result["embedding_norm"] = round(math.sqrt(sum(value * value for value in vector)), 8)
        result["usage"] = data.get("usage") or data.get("usageMetadata")
        result["status"] = "passed"
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
            if "ratelimit" in name.lower() or name.lower() in {
                "retry-after", "x-request-id", "request-id", "cf-ray"
            }
        }
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        result["error"] = sanitize_error_detail(detail, key)
        result["http_error_category"] = classify_http_error(exc.code, result["error"])
        result["status"] = "http-error"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        result["error"] = safe_text(exc, key)
        result["status"] = "transport-error"
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        result["error"] = safe_text(
            f"Embedding response validation failed: {type(exc).__name__}: {exc}", key
        )
        result["status"] = "response-parse-error"
    finally:
        key = ""
    return result


def plan_capability_all_configured(
    capability: str,
    include_paid: bool = False,
    providers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Plan each account for a capability, selecting at most one candidate per provider."""
    candidates = EMBEDDING_CASES if capability == "embeddings" else []
    selected_providers = sorted(set(providers or discovered_provider_names()))
    results: list[dict[str, Any]] = []
    for provider in selected_providers:
        entry = provider_catalogue_entry(provider)
        case = next((item for item in candidates if item["provider"] == provider), None)
        accounts = credential_accounts(provider) or [{
            "account": "DEFAULT", "credential_env": None, "credential": ""
        }]
        for account in accounts:
            env_name = account["credential_env"]
            row: dict[str, Any] = {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "account": account["account"],
                "credential_env": env_name,
                "credential_configured": bool(env_name),
                "capability": capability,
                "model_requested": case["model"] if case else None,
                "access_classification": case["access"] if case else None,
                "adapter": entry.get("capability_adapters", {}).get(capability),
                "http_status": None,
                "latency_seconds": None,
                "retry_performed": False,
            }
            if not env_name:
                row["status"] = "skipped-no-credential"
            elif not case:
                row["status"] = (
                    "skipped-no-provider-adapter"
                    if capability not in entry.get("implemented_capabilities", [])
                    else "skipped-no-model-candidate"
                )
            elif case["access"] in PAID_ACCESS and not include_paid:
                row["status"] = "skipped-paid-opt-in-required"
            else:
                row["status"] = "ready"
            results.append(row)
    return results


def execute_embedding_selection(
    capability: str,
    timeout: float,
    include_paid: bool,
    providers: list[str] | None,
) -> list[dict[str, Any]]:
    """Execute eligible embedding cases while preserving account boundaries."""
    results: list[dict[str, Any]] = []
    for item in plan_capability_all_configured(capability, include_paid, providers):
        if item["status"] == "ready":
            case = next(
                candidate for candidate in EMBEDDING_CASES
                if candidate["provider"] == item["provider"]
            )
            results.append(request_embedding_case(
                case,
                timeout,
                credential_env=item.get("credential_env"),
                account=item.get("account", "DEFAULT"),
            ))
        else:
            results.append(item)
    return results

def request_case(
    case: dict[str, str],
    timeout: float,
    max_tokens: int,
    credential_env: str | None = None,
    account: str = "DEFAULT",
) -> dict[str, Any]:
    provider, model = case["provider"], case["model"]
    if credential_env:
        key = os.environ.get(credential_env, "").strip()
        env_name = credential_env
    else:
        key, env_name = resolve_credential(provider)
        if env_name:
            # Direct single-provider runs must record the account namespace
            # encoded in the selected credential, not the default parameter.
            account = credential_account_from_env_name(env_name)
    result: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "capability": "chat",
        "account": account,
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
        result["error"] = sanitize_error_detail(detail, key)
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
    parser.add_argument("--inventory", action="store_true",
                        help="Show all known/discovered provider accounts, credential source names, endpoints, and implemented capabilities without network requests.")
    parser.add_argument("--report", action="store_true",
                        help="Summarize accumulated JSONL evidence without network requests.")
    parser.add_argument("--capability", choices=PROVIDER_CATALOGUE["capability_families"], default="chat",
                        help="Capability family to evaluate (only implemented adapters execute requests).")
    parser.add_argument("--provider", action="append", choices=discovered_provider_names(),
                        help="Run only this provider; repeat to select several.")
    parser.add_argument("--all-configured", action="store_true",
                        help="Iterate every known provider; run one free/trial candidate where available and report all others as skipped.")
    parser.add_argument("--openrouter-free-matrix", action="store_true",
                        help="Run every explicit catalogue-zero-price OpenRouter model once for each configured OpenRouter credential account; excludes the dynamic router alias and all paid candidates.")
    parser.add_argument("--model",
                        help="Run only this exact configured model ID; combine with --provider to validate its provider.")
    parser.add_argument("--include-paid", action="store_true",
                        help="Also run paid/promotional-credit comparison candidates.")
    parser.add_argument("--show-latest-errors", action="store_true",
                        help="Print the latest recorded errors for Cerebras, DeepInfra, DeepSeek, Hugging Face, and OpenRouter without making API calls.")
    parser.add_argument("--show-credential-sources", action="store_true",
                        help="Show which credential variable name is selected for each provider, without printing values or making API calls.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                        help=f"Maximum generated tokens per request (default: {DEFAULT_MAX_OUTPUT_TOKENS}).")
    parser.add_argument("--output", type=Path, default=Path("provider-smoke-results.jsonl"))
    args = parser.parse_args()

    if args.inventory:
        if args.report or args.provider or args.model or args.all_configured or args.include_paid or args.openrouter_free_matrix or args.show_latest_errors or args.show_credential_sources or args.capability != "chat":
            parser.error("--inventory cannot be combined with inference, report, or capability-selection options")
        print_inventory()
        return 0

    if args.report:
        if args.inventory or args.provider or args.model or args.all_configured or args.include_paid or args.openrouter_free_matrix or args.show_latest_errors or args.show_credential_sources or args.capability != "chat":
            parser.error("--report cannot be combined with inference, inventory, or capability-selection options")
        return print_report(args.output)

    if args.timeout <= 0 or args.max_output_tokens < 1:
        parser.error("--timeout must be positive and --max-output-tokens must be at least 1")

    if args.show_credential_sources:
        if args.show_latest_errors or args.all_configured or args.provider or args.model or args.include_paid or args.report or args.capability != "chat":
            parser.error("--show-credential-sources cannot be combined with other selection options")
        for provider in discovered_provider_names():
            names = [
                name for name in credential_variable_names(provider)
                if os.environ.get(name, "").strip()
            ]
            print(f"{provider:16} {', '.join(names) if names else 'not configured'}")
        print("Read-only: no credential values were displayed and no provider requests were made.")
        return 0

    if args.show_latest_errors:
        if args.all_configured or args.provider or args.model or args.include_paid or args.show_credential_sources or args.inventory or args.report or args.capability != "chat":
            parser.error("--show-latest-errors cannot be combined with inference-selection options")
        if not args.output.exists():
            parser.error(f"No evidence file found: {args.output}")
        wanted = {"CEREBRAS", "DEEPINFRA", "DEEPSEEK", "HUGGINGFACE", "OPENROUTER"}
        latest: dict[str, dict[str, Any]] = {}
        with args.output.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                provider = item.get("provider")
                if provider in wanted and (
                    item.get("error") or item.get("http_status") in {402, 429}
                ):
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

    if args.capability != "chat":
        if args.model or args.include_paid or args.openrouter_free_matrix:
            parser.error("--model, --include-paid, and --openrouter-free-matrix currently apply only to chat")
        results = capability_plan(args.capability, args.provider)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            for item in results:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        unsupported = sum(item["status"] == "skipped-no-capability-adapter" for item in results)
        missing = sum(item["status"] == "skipped-no-credential" for item in results)
        print(f"Capability: {args.capability}")
        print(f"Provider/account rows: {len(results)}; adapters not implemented: {unsupported}; no credentials: {missing}.")
        print("No provider requests were made. These rows are explicit adapter backlog, not inference failures.")
        print(f"Evidence: {args.output}")
        return 0

    if args.openrouter_free_matrix and (
        args.all_configured or args.provider or args.model or args.include_paid
        or args.show_latest_errors or args.show_credential_sources
    ):
        parser.error("--openrouter-free-matrix cannot be combined with other selection options")

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

    if args.openrouter_free_matrix:
        accounts = credential_accounts("OPENROUTER")
        cases = openrouter_free_cases()
        if not accounts:
            parser.error("No configured OpenRouter credential accounts found.")
        if not cases:
            parser.error("No explicit catalogue-zero-price OpenRouter models are configured.")
        results = [
            request_case(
                case,
                args.timeout,
                args.max_output_tokens,
                credential_env=account["credential_env"],
                account=account["account"],
            )
            for account in accounts
            for case in cases
        ]
    elif args.all_configured:
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
                results.append(request_case(
                    case,
                    args.timeout,
                    args.max_output_tokens,
                    credential_env=item.get("credential_env"),
                    account=item.get("account", "DEFAULT"),
                ))
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
        account_label = item.get("account", "DEFAULT")
        print(f'{item["provider"]:10} {account_label:10} {model_label:46} {status:30} HTTP={http} latency={latency}s')
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
        print(f"\nProvider/account inventory: {len(results)} rows examined.")
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
