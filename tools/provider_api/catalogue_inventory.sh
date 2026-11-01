#!/usr/bin/env bash
set -Eeuo pipefail

# Read authenticated model catalogues and normalize public model metadata.
# This script does NOT invoke models. It never prints or writes credential values.

OUT="${1:-provider-model-catalogue.json}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }

declare -A KEYS=(
  [CEREBRAS]=CEREBRAS_API_KEY_FLOSSWARE
  [COHERE]=COHERE_API_KEY_FLOSSWARE
  [DEEPINFRA]=DEEPINFRA_API_KEY_FLOSSWARE
  [DEEPSEEK]=DEEPSEEK_API_KEY_FLOSSWARE
  [GOOGLE]=GOOGLE_API_KEY_FLOSSWARE
  [GROQ]=GROQ_API_KEY_FLOSSWARE
  [HUGGINGFACE]=HUGGINGFACE_API_KEY_FLOSSWARE
  [MISTRAL]=MISTRAL_API_KEY_FLOSSWARE
  [NVIDIA]=NVIDIA_API_KEY_FLOSSWARE
  [OPENAI]=OPENAI_API_KEY_FLOSSWARE
  [OPENROUTER]=OPENROUTER_API_KEY_FLOSSWARE
)

declare -A URLS=(
  [CEREBRAS]="https://api.cerebras.ai/v1/models"
  [COHERE]="https://api.cohere.com/v1/models"
  [DEEPINFRA]="https://api.deepinfra.com/v1/openai/models"
  [DEEPSEEK]="https://api.deepseek.com/models"
  [GOOGLE]="https://generativelanguage.googleapis.com/v1beta/models"
  [GROQ]="https://api.groq.com/openai/v1/models"
  [HUGGINGFACE]="https://router.huggingface.co/v1/models"
  [MISTRAL]="https://api.mistral.ai/v1/models"
  [NVIDIA]="https://integrate.api.nvidia.com/v1/models"
  [OPENAI]="https://api.openai.com/v1/models"
  [OPENROUTER]="https://openrouter.ai/api/v1/models"
)

RESULTS="$TMP/results.jsonl"
: > "$RESULTS"

for provider in CEREBRAS COHERE DEEPINFRA DEEPSEEK GOOGLE GROQ                 HUGGINGFACE MISTRAL NVIDIA OPENAI OPENROUTER; do
  var="${KEYS[$provider]}"
  key="${!var:-}"
  body="$TMP/$provider.json"
  status="not_configured"
  elapsed=""
  reason=""

  if [[ -n "$key" ]]; then
    if [[ "$provider" == GOOGLE ]]; then
      config="$(printf 'header = "x-goog-api-key: %s"\n' "$key")"
    else
      config="$(printf 'header = "Authorization: Bearer %s"\n' "$key")"
    fi

    result="$(
      printf '%s' "$config" |
      curl --config - -sS         --connect-timeout 8 --max-time 30         -o "$body"         -w '%{http_code} %{time_total}'         "${URLS[$provider]}" 2>/dev/null || true
    )"

    read -r status elapsed <<< "$result"
    status="${status:-000}"
    unset config
  else
    reason="Credential not configured"
  fi

  python3 - "$provider" "$var" "$status" "$elapsed" "$body" "$reason" >> "$RESULTS" <<'PY'
import json
import re
import sys

provider, envvar, status, elapsed, path, reason = sys.argv[1:]

def first(obj, *keys):
    for key in keys:
        value = obj.get(key) if isinstance(obj, dict) else None
        if value is not None:
            return value
    return None

def as_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def is_zero(value):
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False

models = []
parse_error = None

if status == "200":
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            raw_models = payload
        else:
            raw_models = first(payload, "data", "models", "modelData") or []

        if not isinstance(raw_models, list):
            raw_models = []

        for item in raw_models:
            if not isinstance(item, dict):
                continue

            model_id = first(item, "id", "name", "model", "model_id")
            if not model_id:
                continue

            pricing = first(item, "pricing", "prices")
            if not isinstance(pricing, dict):
                pricing = None

            prompt_price = first(pricing, "prompt", "input") if pricing else None
            completion_price = first(pricing, "completion", "output") if pricing else None

            free = None
            if prompt_price is not None and completion_price is not None:
                free = is_zero(prompt_price) and is_zero(completion_price)
            elif str(model_id).endswith(":free"):
                free = True

            methods = first(item, "supportedGenerationMethods", "supported_generation_methods")
            endpoints = first(item, "endpoints", "supported_endpoints", "supportedEndpoints")
            capabilities = first(item, "capabilities", "tasks")

            model = {
                "id": str(model_id),
                "display_name": first(item, "displayName", "display_name", "name"),
                "owned_by": first(item, "owned_by", "publisher", "organization"),
                "context_length": as_number(first(item, "context_length", "context_window", "inputTokenLimit", "max_input_tokens")),
                "max_output_tokens": as_number(first(item, "max_output_tokens", "max_completion_tokens", "outputTokenLimit")),
                "input_modalities": first(item, "input_modalities", "inputModalities"),
                "output_modalities": first(item, "output_modalities", "outputModalities"),
                "generation_methods": methods,
                "endpoints": endpoints,
                "capabilities": capabilities,
                "pricing": pricing,
                "free_by_catalogue_metadata": free
            }
            models.append(model)
    except Exception as exc:
        parse_error = type(exc).__name__

# Remove obvious non-chat-only entries from the initial general-inference shortlist.
non_chat = re.compile(r"(embed|rerank|whisper|transcri|text-to-speech|tts|moderation|prompt.guard)", re.I)

def chat_candidate(model):
    if provider == "GOOGLE":
        methods = model.get("generation_methods") or []
        return "generateContent" in methods
    text = " ".join(str(model.get(k) or "") for k in ("id", "endpoints", "capabilities"))
    return not non_chat.search(text)

candidates = [m for m in models if chat_candidate(m)]

def rank(model):
    free = model.get("free_by_catalogue_metadata")
    ctx = model.get("context_length") or 0
    return (0 if free is True else 1 if free is None else 2, -ctx, model["id"])

candidates.sort(key=rank)

record = {
    "provider": provider,
    "environment_variable": envvar,
    "http_status": status,
    "response_seconds": float(elapsed) if elapsed else None,
    "catalogue_count": len(models),
    "models": models,
    "chat_candidates": candidates[:20],
    "parse_error": parse_error,
    "reason": reason or None
}
print(json.dumps(record))
PY

  printf '%-14s HTTP %-3s catalogue=%s\n' "$provider" "$status"     "$(python3 -c 'import json,sys; x=json.load(sys.stdin); print(x["catalogue_count"])' < <(tail -n 1 "$RESULTS"))"
done

python3 - "$RESULTS" "$OUT" <<'PY'
import json
import sys
from datetime import datetime, timezone

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as f:
    providers = [json.loads(line) for line in f if line.strip()]

providers.sort(key=lambda p: p["provider"])
report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "Model catalogue inventory only; no inference requests made.",
    "providers": providers
}

with open(destination, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("\nCandidate shortlist (up to 20 per provider):")
for provider in providers:
    if provider["http_status"] != "200":
        print(f'{provider["provider"]}: HTTP {provider["http_status"]}')
        continue
    print(f'\n{provider["provider"]} ({provider["catalogue_count"]} models)')
    for model in provider["chat_candidates"]:
        free = model["free_by_catalogue_metadata"]
        marker = "catalogue-$0" if free is True else "price-unknown" if free is None else "priced"
        print(f'  {model["id"]} [{marker}]')

print(f"\nFull normalized catalogue written to: {destination}")
print("No credential values or inference outputs were written.")
PY
