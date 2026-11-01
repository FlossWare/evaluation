"""Deterministic tests for provider smoke-test response classification."""
import json
import unittest

from unittest.mock import patch

from tools.provider_api.smoke_test import CASES, DEFAULT_MAX_OUTPUT_TOKENS, DEPRECATED_ACCESS, ENV_VARS, EXPECTED, OPENAI_COMPATIBLE, build_headers, classify_response, discovered_provider_names, plan_all_configured, provider_from_env_name, representative_cases, resolve_credential



class CredentialResolutionTests(unittest.TestCase):
    def test_standard_provider_environment_names_are_supported(self):
        cases = {
            "CEREBRAS": "CEREBRAS_API_KEY",
            "DEEPINFRA": "DEEPINFRA_API_KEY",
            "DEEPSEEK": "DEEPSEEK_API_KEY",
            "HUGGINGFACE": "HF_TOKEN",
            "GOOGLE": "GEMINI_API_KEY",
            "GROQ": "GROQ_API_KEY",
            "OPENROUTER": "OPENROUTER_API_KEY",
            "MISTRAL": "MISTRAL_API_KEY",
            "COHERE": "COHERE_API_KEY",
            "NVIDIA": "NVIDIA_API_KEY",
            "OPENAI": "OPENAI_API_KEY",
            "ANTHROPIC": "ANTHROPIC_API_KEY",
        }
        for provider, env_name in cases.items():
            with self.subTest(provider=provider):
                with patch.dict("os.environ", {env_name: "test-secret"}, clear=True):
                    self.assertEqual(resolve_credential(provider), ("test-secret", env_name))

    def test_flossware_variable_has_precedence(self):
        with patch.dict("os.environ", {
            "CEREBRAS_API_KEY_FLOSSWARE": "preferred-secret",
            "CEREBRAS_API_KEY": "fallback-secret",
        }, clear=True):
            self.assertEqual(
                resolve_credential("CEREBRAS"),
                ("preferred-secret", "CEREBRAS_API_KEY_FLOSSWARE"),
            )

    def test_huggingface_sdk_alias_is_supported(self):
        with patch.dict("os.environ", {"HUGGING_FACE_HUB_TOKEN": "hf-secret"}, clear=True):
            self.assertEqual(
                resolve_credential("HUGGINGFACE"),
                ("hf-secret", "HUGGING_FACE_HUB_TOKEN"),
            )

    def test_blank_variables_are_ignored(self):
        with patch.dict("os.environ", {
            "DEEPSEEK_API_KEY_FLOSSWARE": "  ",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        }, clear=True):
            self.assertEqual(
                resolve_credential("DEEPSEEK"),
                ("deepseek-secret", "DEEPSEEK_API_KEY"),
            )

    def test_missing_credentials_return_no_value_or_source(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_credential("GROQ"), ("", None))


class AllConfiguredPlanningTests(unittest.TestCase):
    def test_plan_includes_every_provider_and_never_opts_into_paid_candidates(self):
        aliases = {
            "CEREBRAS_API_KEY": "CEREBRAS_API_KEY",
            "DEEPINFRA_API_KEY": "DEEPINFRA_API_KEY",
            "NVIDIA_API_KEY": "NVIDIA_API_KEY",
            "HF_TOKEN": "HF_TOKEN",
            "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY": "GEMINI_API_KEY",
            "GROQ_API_KEY": "GROQ_API_KEY",
            "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
            "MISTRAL_API_KEY": "MISTRAL_API_KEY",
            "COHERE_API_KEY": "COHERE_API_KEY",
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
        }
        with patch.dict("os.environ", {name: "test-secret" for name in aliases}, clear=True):
            plan = plan_all_configured()
        self.assertEqual([item["provider"] for item in plan], sorted(ENV_VARS))
        by_provider = {item["provider"]: item for item in plan}
        self.assertEqual(by_provider["GOOGLE"]["status"], "ready")
        self.assertEqual(by_provider["HUGGINGFACE"]["status"], "ready")
        self.assertEqual(by_provider["OPENROUTER"]["status"], "ready")
        self.assertEqual(by_provider["CEREBRAS"]["status"], "skipped-paid-opt-in-required")
        self.assertEqual(by_provider["DEEPINFRA"]["status"], "skipped-paid-opt-in-required")
        self.assertEqual(by_provider["DEEPSEEK"]["status"], "skipped-paid-opt-in-required")
        self.assertEqual(by_provider["GROQ"]["status"], "skipped-paid-opt-in-required")
        self.assertEqual(by_provider["NVIDIA"]["status"], "skipped-deprecated-endpoint")
        self.assertTrue(all(item["credential_configured"] for item in plan))

    def test_plan_reports_every_missing_credential(self):
        with patch.dict("os.environ", {}, clear=True):
            plan = plan_all_configured()
        self.assertEqual(len(plan), len(ENV_VARS))
        self.assertTrue(all(item["status"] == "skipped-no-credential" for item in plan))
        self.assertTrue(all(item["credential_env"] is None for item in plan))


    def test_unlisted_provider_credentials_are_discovered_without_secret_values(self):
        with patch.dict("os.environ", {
            "TOGETHER_API_KEY": "together-secret",
            "FIREWORKS_API_KEY": "fireworks-secret",
        }, clear=True):
            self.assertEqual(provider_from_env_name("TOGETHER_API_KEY"), "TOGETHER")
            self.assertEqual(
                resolve_credential("TOGETHER"),
                ("together-secret", "TOGETHER_API_KEY"),
            )
            self.assertIn("FIREWORKS", discovered_provider_names())
            plan = {item["provider"]: item for item in plan_all_configured()}
            self.assertEqual(plan["TOGETHER"]["status"], "skipped-no-provider-adapter")
            self.assertEqual(plan["FIREWORKS"]["status"], "skipped-no-provider-adapter")

    def test_account_scoped_api_keys_are_discovered_separately(self):
        from tools.provider_api.smoke_test import credential_accounts
        with patch.dict("os.environ", {
            "CEREBRAS_API_KEY_FLOSSWARE": "fw-secret",
            "CEREBRAS_API_KEY_HOTMAIL": "hotmail-secret",
            "CEREBRAS_API_KEY_NCRR": "ncrr-secret",
            "GROQ_ORG_ID_FLOSSWARE": "org-id",
        }, clear=True):
            self.assertEqual(provider_from_env_name("CEREBRAS_API_KEY_HOTMAIL"), "CEREBRAS")
            self.assertEqual(provider_from_env_name("CEREBRAS_API_KEY_NCRR"), "CEREBRAS")
            accounts = credential_accounts("CEREBRAS")
            self.assertEqual(
                [(a["account"], a["credential_env"]) for a in accounts],
                [
                    ("FLOSSWARE", "CEREBRAS_API_KEY_FLOSSWARE"),
                    ("HOTMAIL", "CEREBRAS_API_KEY_HOTMAIL"),
                    ("NCRR", "CEREBRAS_API_KEY_NCRR"),
                ],
            )
            plan = [p for p in plan_all_configured() if p["provider"] == "CEREBRAS"]
            self.assertEqual([p["account"] for p in plan], ["FLOSSWARE", "HOTMAIL", "NCRR"])
            self.assertTrue(all(p["status"] == "skipped-paid-opt-in-required" for p in plan))

        with patch.dict("os.environ", {
            "THINKMACHINES_API_KEY_FLOSSWARE": "fw-secret",
            "THINKMACHINES_API_KEY_HOTMAIL": "hotmail-secret",
            "THINKINGMACHINES_API_KEY_NCRR": "ncrr-secret",
        }, clear=True):
            self.assertEqual(
                provider_from_env_name("THINKINGMACHINES_API_KEY_NCRR"),
                "THINKMACHINES",
            )
            aliases = credential_accounts("THINKMACHINES")
            self.assertEqual(
                [(a["account"], a["credential_env"]) for a in aliases],
                [
                    ("FLOSSWARE", "THINKMACHINES_API_KEY_FLOSSWARE"),
                    ("HOTMAIL", "THINKMACHINES_API_KEY_HOTMAIL"),
                    ("NCRR", "THINKINGMACHINES_API_KEY_NCRR"),
                ],
            )

        self.assertIsNone(provider_from_env_name("GITHUB_TOKEN"))
        self.assertIsNone(provider_from_env_name("AWS_SESSION_TOKEN"))
        self.assertIsNone(provider_from_env_name("GROQ_ORG_ID_FLOSSWARE"))

    def test_paid_opt_in_only_changes_the_execution_plan(self):
        with patch.dict("os.environ", {
            "CEREBRAS_API_KEY": "test-secret",
            "DEEPINFRA_API_KEY": "test-secret",
            "DEEPSEEK_API_KEY": "test-secret",
            "GROQ_API_KEY": "test-secret",
        }, clear=True):
            plan = {item["provider"]: item for item in plan_all_configured(include_paid=True)}
        self.assertEqual(plan["CEREBRAS"]["status"], "ready")
        self.assertEqual(plan["DEEPINFRA"]["status"], "ready")
        self.assertEqual(plan["DEEPSEEK"]["status"], "ready")
        self.assertEqual(plan["GROQ"]["status"], "ready")

class RequestHeaderTests(unittest.TestCase):
    def test_provider_requests_identify_the_application(self):
        headers = build_headers("CEREBRAS", "test-key")
        self.assertEqual(headers["User-Agent"], "FlossWare-ProviderSmoke/0.1")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    def test_google_uses_its_api_key_header(self):
        headers = build_headers("GOOGLE", "test-key")
        self.assertEqual(headers["x-goog-api-key"], "test-key")
        self.assertNotIn("Authorization", headers)

    def test_deepinfra_uses_openai_compatible_chat_endpoint(self):
        self.assertEqual(
            OPENAI_COMPATIBLE["DEEPINFRA"],
            "https://api.deepinfra.com/v1/openai/chat/completions",
        )

    def test_mistral_uses_openai_compatible_chat_endpoint(self):
        self.assertEqual(
            OPENAI_COMPATIBLE["MISTRAL"],
            "https://api.mistral.ai/v1/chat/completions",
        )

    def test_nvidia_candidate_is_marked_deprecated(self):
        self.assertIn("deprecated-endpoint", DEPRECATED_ACCESS)


class HttpErrorClassificationTests(unittest.TestCase):
    def test_deepseek_insufficient_balance_is_not_called_model_failure(self):
        from tools.provider_api.smoke_test import classify_http_error
        self.assertEqual(
            classify_http_error(402, '{"error":{"message":"Insufficient Balance"}}'),
            "insufficient-account-balance",
        )

    def test_cerebras_team_subscription_error_is_identified(self):
        from tools.provider_api.smoke_test import classify_http_error
        self.assertEqual(
            classify_http_error(402, "Team org requires active subscription or available credit"),
            "organization-subscription-or-credit-required",
        )

    def test_huggingface_credit_error_is_identified(self):
        from tools.provider_api.smoke_test import classify_http_error
        self.assertEqual(
            classify_http_error(402, "Inference Providers credits exhausted"),
            "account-credit-or-billing-required",
        )

    def test_unknown_402_is_not_overclassified(self):
        from tools.provider_api.smoke_test import classify_http_error
        self.assertEqual(
            classify_http_error(402, "Payment required"),
            "account-credit-or-billing-required",
        )

    def test_non_402_is_not_a_billing_failure(self):
        from tools.provider_api.smoke_test import classify_http_error
        self.assertIsNone(classify_http_error(403, "Forbidden"))

    def test_openrouter_agentic_only_403_is_classified(self):
        from tools.provider_api.smoke_test import classify_http_error

        detail = json.dumps({
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": (
                        "thinkingmachines/inkling-small:free is only available "
                        "on agentic harnesses. Try plugging it into a coding agent."
                    ),
                    "provider_name": "Thinking Machines",
                },
            }
        })
        self.assertEqual(
            classify_http_error(403, detail),
            "agentic-harness-required",
        )

    def test_generic_403_remains_unclassified(self):
        from tools.provider_api.smoke_test import classify_http_error

        self.assertIsNone(classify_http_error(403, "Forbidden"))


    def test_openrouter_shared_pool_429_is_classified_as_upstream_rate_limit(self):
        from tools.provider_api.smoke_test import classify_http_error

        detail = json.dumps({
            "error": {
                "metadata": {
                    "provider_name": "Google AI Studio",
                    "is_byok": False,
                    "limit_source": "upstream_provider_shared_pool",
                }
            }
        })
        self.assertEqual(
            classify_http_error(429, detail),
            "upstream-shared-pool-rate-limit",
        )

    def test_byok_429_is_distinguished_from_shared_pool_limit(self):
        from tools.provider_api.smoke_test import classify_http_error

        detail = json.dumps({"error": {"metadata": {"is_byok": True}}})
        self.assertEqual(
            classify_http_error(429, detail),
            "byok-provider-rate-limit",
        )

    def test_unclassified_429_is_still_identified_as_rate_limit(self):
        from tools.provider_api.smoke_test import classify_http_error

        self.assertEqual(
            classify_http_error(429, "Too many requests"),
            "rate-limit-unclassified",
        )

    def test_provider_error_sanitizer_removes_user_ids_and_redacts_secrets(self):
        from tools.provider_api.smoke_test import sanitize_error_detail

        detail = json.dumps({
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": "request rejected using secret-value",
                    "user_id": "provider-user-123",
                    "provider_name": "Google AI Studio",
                },
            },
            "userId": "top-level-user-456",
        })
        sanitized = sanitize_error_detail(detail, "secret-value")

        self.assertNotIn("provider-user-123", sanitized)
        self.assertNotIn("top-level-user-456", sanitized)
        self.assertNotIn("user_id", sanitized)
        self.assertNotIn("userId", sanitized)
        self.assertNotIn("secret-value", sanitized)
        self.assertIn("[REDACTED]", sanitized)
        self.assertIn("Google AI Studio", sanitized)

    def test_error_summary_prefers_provider_diagnostic_detail(self):
        from tools.provider_api.smoke_test import error_summary

        detail = json.dumps({
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": "temporarily rate-limited upstream",
                    "provider_name": "Google AI Studio",
                },
            }
        })
        self.assertEqual(error_summary(detail), "temporarily rate-limited upstream")


class BatchSelectionTests(unittest.TestCase):
    def test_batch_selects_at_most_one_candidate_per_provider(self):
        selected = representative_cases(include_paid=False)
        self.assertEqual(len(selected), len({case["provider"] for case in selected.values()}))
        self.assertEqual(selected["GOOGLE"]["model"], "models/gemini-3.5-flash-lite")
        self.assertEqual(selected["OPENROUTER"]["model"], "nvidia/nemotron-3.5-lightning:free")
        self.assertIn(
            {"provider": "OPENROUTER", "model": "deepseek/deepseek-v4-flash-0731:free", "access": "catalogue-zero-price"},
            CASES,
        )
        self.assertEqual(selected["MISTRAL"]["model"], "mistral-small-latest")
        self.assertEqual(selected["COHERE"]["model"], "command-r7b-12-2024")
        self.assertEqual(
            selected["HUGGINGFACE"]["model"],
            "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
        )
        self.assertNotIn("CEREBRAS", selected)
        self.assertNotIn("NVIDIA", selected)

    def test_paid_opt_in_adds_paid_only_providers(self):
        selected = representative_cases(include_paid=True)
        self.assertIn("CEREBRAS", selected)
        self.assertIn("DEEPINFRA", selected)
        self.assertNotIn("NVIDIA", selected)


class OpenRouterFreeMatrixTests(unittest.TestCase):
    def test_matrix_contains_only_explicit_zero_price_models(self):
        from tools.provider_api.smoke_test import openrouter_free_cases

        cases = openrouter_free_cases()
        self.assertEqual(
            [case["model"] for case in cases],
            [
                "nvidia/nemotron-3.5-lightning:free",
                "deepseek/deepseek-v4-flash-0731:free",
                "google/gemma-4-31b-it:free",
                "thinkingmachines/inkling:free",
                "thinkingmachines/inkling-small:free",
            ],
        )
        self.assertTrue(all(case["provider"] == "OPENROUTER" for case in cases))
        self.assertTrue(all(case["access"] == "catalogue-zero-price" for case in cases))
        self.assertNotIn("openrouter/free", [case["model"] for case in cases])


class DirectRequestAccountTests(unittest.TestCase):
    def test_direct_request_records_the_selected_credential_account(self):
        import json
        from tools.provider_api.smoke_test import request_case

        class FakeResponse:
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "id": "test-request",
                    "model": "nvidia/nemotron-3.5-lightning:free",
                    "choices": [{
                        "message": {"content": EXPECTED},
                        "finish_reason": "stop",
                    }],
                    "usage": {},
                }).encode("utf-8")

        case = {
            "provider": "OPENROUTER",
            "model": "nvidia/nemotron-3.5-lightning:free",
            "access": "catalogue-zero-price",
        }
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY_FLOSSWARE": "test-secret",
        }, clear=True), patch(
            "tools.provider_api.smoke_test.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            result = request_case(case, timeout=1, max_tokens=512)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["credential_env"], "OPENROUTER_API_KEY_FLOSSWARE")
        self.assertEqual(result["account"], "FLOSSWARE")


class DefaultOutputBudgetTests(unittest.TestCase):
    def test_default_output_budget_is_512_tokens(self):
        import inspect
        from tools.provider_api.smoke_test import main

        self.assertEqual(DEFAULT_MAX_OUTPUT_TOKENS, 512)
        source = inspect.getsource(main)
        self.assertIn('default=DEFAULT_MAX_OUTPUT_TOKENS', source)
        self.assertIn('default: {DEFAULT_MAX_OUTPUT_TOKENS}', source)


class ClassifyResponseTests(unittest.TestCase):
    def test_exact_response_with_stop_passes(self):
        self.assertEqual(
            classify_response(EXPECTED, "stop", 0),
            ("complete", "passed", True),
        )

    def test_partial_text_with_length_is_incomplete_reasoning(self):
        self.assertEqual(
            classify_response("FLOSS", "length", 316),
            ("incomplete", "incomplete-reasoning", False),
        )

    def test_partial_text_with_length_without_reasoning_is_incomplete_output(self):
        self.assertEqual(
            classify_response("FLOSS", "length", 0),
            ("incomplete", "incomplete-output", False),
        )

    def test_exact_text_with_length_is_still_incomplete(self):
        self.assertEqual(
            classify_response(EXPECTED, "length", 4),
            ("incomplete", "incomplete-reasoning", True),
        )

    def test_empty_response_without_length_is_empty(self):
        self.assertEqual(
            classify_response("", "stop", 0),
            ("empty", "empty-response", False),
        )

    def test_nonmatching_complete_text_is_semantic_mismatch(self):
        self.assertEqual(
            classify_response("some other answer", "stop", 0),
            ("complete", "response-mismatch", False),
        )

    def test_google_max_tokens_is_incomplete(self):
        self.assertEqual(
            classify_response("partial", "MAX_TOKENS", 12),
            ("incomplete", "incomplete-reasoning", False),
        )


class UnifiedRunnerInventoryTests(unittest.TestCase):
    def test_inventory_lists_each_account_without_exposing_credentials(self):
        import io
        from contextlib import redirect_stdout
        from tools.provider_api.smoke_test import inventory_rows, print_inventory

        secrets = {
            "OPENROUTER_API_KEY_FLOSSWARE": "flossware-secret",
            "OPENROUTER_API_KEY_HOTMAIL": "hotmail-secret",
            "VOYAGEAI_API_KEY_NCRR": "voyage-secret",
        }
        with patch.dict("os.environ", secrets, clear=True):
            rows = inventory_rows()
            openrouter = [row for row in rows if row["provider"] == "OPENROUTER"]
            self.assertEqual(
                [row["account"] for row in openrouter],
                ["FLOSSWARE", "HOTMAIL"],
            )
            self.assertTrue(all(row["credential_configured"] for row in openrouter))
            output = io.StringIO()
            with redirect_stdout(output):
                print_inventory()

        rendered = output.getvalue()
        for secret in secrets.values():
            self.assertNotIn(secret, rendered)
        self.assertIn("OPENROUTER_API_KEY_FLOSSWARE", rendered)
        self.assertIn("OPENROUTER_API_KEY_HOTMAIL", rendered)
        self.assertIn("VOYAGEAI", rendered)

    def test_unimplemented_capability_is_reported_without_requests(self):
        from tools.provider_api.smoke_test import capability_plan

        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY_FLOSSWARE": "test-secret",
        }, clear=True):
            rows = capability_plan("embeddings", ["OPENROUTER"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["capability"], "embeddings")
        self.assertEqual(rows[0]["status"], "skipped-no-capability-adapter")
        self.assertIsNone(rows[0]["http_status"])
        self.assertFalse(rows[0]["retry_performed"])

    def test_evidence_report_groups_by_provider_account_capability_and_status(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from tools.provider_api.smoke_test import print_report

        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.jsonl"
            evidence.write_text(
                json.dumps({
                    "provider": "OPENROUTER",
                    "account": "FLOSSWARE",
                    "capability": "chat",
                    "status": "passed",
                }) + "\n" +
                json.dumps({
                    "provider": "OPENROUTER",
                    "account": "FLOSSWARE",
                    "capability": "embeddings",
                    "status": "skipped-no-capability-adapter",
                }) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                status = print_report(evidence)

        self.assertEqual(status, 0)
        self.assertIn("Records: 2", output.getvalue())
        self.assertIn("chat", output.getvalue())
        self.assertIn("embeddings", output.getvalue())
        self.assertIn("Read-only", output.getvalue())



class EmbeddingAdapterTests(unittest.TestCase):
    def test_provider_native_embedding_request_specs(self):
        from tools.provider_api.smoke_test import embedding_request_spec

        cases = {
            "GOOGLE": "models/gemini-embedding-001",
            "COHERE": "embed-v4.0",
            "VOYAGEAI": "voyage-4-lite",
            "JINA": "jina-embeddings-v3",
            "OPENAI": "text-embedding-3-small",
            "MISTRAL": "mistral-embed",
        }
        for provider, model in cases.items():
            with self.subTest(provider=provider):
                url, payload = embedding_request_spec({
                    "provider": provider,
                    "model": model,
                    "access": "test",
                })
                self.assertTrue(url.startswith("https://"))
                if provider != "GOOGLE":
                    self.assertEqual(payload["model"], model)
                else:
                    self.assertIn(model, url)
                self.assertTrue(
                    payload.get("content") or payload.get("texts") or payload.get("input")
                )
        self.assertEqual(
            embedding_request_spec({
                "provider": "JINA", "model": "jina-embeddings-v3", "access": "test"
            })[1]["task"],
            "retrieval.passage",
        )
        self.assertEqual(
            embedding_request_spec({
                "provider": "GOOGLE", "model": "models/gemini-embedding-001", "access": "test"
            })[1]["embedContentConfig"]["taskType"],
            "RETRIEVAL_DOCUMENT",
        )
        self.assertEqual(
            embedding_request_spec({
                "provider": "COHERE", "model": "embed-v4.0", "access": "test"
            })[1]["input_type"],
            "search_document",
        )

    def test_embedding_response_envelopes_normalize(self):
        from tools.provider_api.smoke_test import extract_embedding_vectors

        fixtures = {
            "OPENAI": {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            "MISTRAL": {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            "VOYAGEAI": {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            "JINA": {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            "COHERE": {"embeddings": {"float": [[0.1, 0.2, 0.3]]}},
            "GOOGLE": {"embedding": {"values": [0.1, 0.2, 0.3]}},
        }
        for provider, fixture in fixtures.items():
            with self.subTest(provider=provider):
                self.assertEqual(
                    extract_embedding_vectors(provider, fixture),
                    [[0.1, 0.2, 0.3]],
                )

    def test_embedding_response_rejects_invalid_vectors(self):
        from tools.provider_api.smoke_test import extract_embedding_vectors

        invalid = [
            {"data": [{"embedding": []}]},
            {"data": [{"embedding": [0.0, 0.0]}]},
            {"data": [{"embedding": [0.1, float("nan")]}]},
            {"data": [{"embedding": [0.1, "bad"]}]},
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.1]}]},
        ]
        for fixture in invalid:
            with self.subTest(fixture=fixture):
                with self.assertRaises(ValueError):
                    extract_embedding_vectors("OPENAI", fixture)

    def test_embedding_request_records_dimensions_without_persisting_vectors(self):
        import json
        from tools.provider_api.smoke_test import request_embedding_case

        class FakeResponse:
            status = 200
            headers = {"x-request-id": "embedding-test-id"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "model": "text-embedding-3-small",
                    "data": [{"embedding": [0.25, 0.5, 0.75]}],
                    "usage": {"total_tokens": 8},
                }).encode("utf-8")

        case = {
            "provider": "OPENAI",
            "model": "text-embedding-3-small",
            "access": "paid",
        }
        with patch.dict("os.environ", {
            "OPENAI_API_KEY_FLOSSWARE": "test-secret",
        }, clear=True), patch(
            "tools.provider_api.smoke_test.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            result = request_embedding_case(
                case,
                timeout=1,
                credential_env="OPENAI_API_KEY_FLOSSWARE",
                account="FLOSSWARE",
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["account"], "FLOSSWARE")
        self.assertEqual(result["capability"], "embeddings")
        self.assertEqual(result["embedding_count"], 1)
        self.assertEqual(result["embedding_dimensions"], 3)
        self.assertGreater(result["embedding_norm"], 0)
        self.assertNotIn("embedding", result)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")

    def test_embedding_plan_keeps_paid_candidates_opt_in_only(self):
        from tools.provider_api.smoke_test import plan_capability_all_configured

        env = {
            "VOYAGE_API_KEY_FLOSSWARE": "voyage-secret",
            "JINA_API_KEY_FLOSSWARE": "jina-secret",
            "OPENAI_API_KEY_FLOSSWARE": "openai-secret",
        }
        with patch.dict("os.environ", env, clear=True):
            plan = plan_capability_all_configured(
                "embeddings",
                include_paid=False,
                providers=["VOYAGEAI", "JINA", "OPENAI"],
            )

        statuses = {item["provider"]: item["status"] for item in plan}
        self.assertEqual(statuses["JINA"], "ready")
        self.assertEqual(statuses["VOYAGEAI"], "skipped-paid-opt-in-required")
        self.assertEqual(statuses["OPENAI"], "skipped-paid-opt-in-required")

    def test_embedding_capability_plan_recognizes_embedding_only_providers(self):
        from tools.provider_api.smoke_test import capability_plan

        with patch.dict("os.environ", {
            "JINA_API_KEY_FLOSSWARE": "jina-secret",
            "VOYAGE_API_KEY_FLOSSWARE": "voyage-secret",
        }, clear=True):
            rows = capability_plan("embeddings", ["JINA", "VOYAGEAI"])

        by_provider = {item["provider"]: item for item in rows}
        self.assertEqual(by_provider["JINA"]["status"], "ready")
        self.assertEqual(by_provider["JINA"]["adapter"], "jina-embeddings")
        self.assertEqual(by_provider["VOYAGEAI"]["status"], "ready")
        self.assertEqual(by_provider["VOYAGEAI"]["adapter"], "voyage-embeddings")

    def test_embedding_plan_includes_paid_candidates_only_with_opt_in(self):
        from tools.provider_api.smoke_test import plan_capability_all_configured

        with patch.dict("os.environ", {
            "VOYAGE_API_KEY_FLOSSWARE": "voyage-secret",
        }, clear=True):
            plan = plan_capability_all_configured(
                "embeddings",
                include_paid=True,
                providers=["VOYAGEAI"],
            )

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["status"], "ready")



if __name__ == "__main__":
    unittest.main()
