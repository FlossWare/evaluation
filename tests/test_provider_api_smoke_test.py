"""Deterministic tests for provider smoke-test response classification."""
import unittest

from unittest.mock import patch

from tools.provider_api.smoke_test import CASES, DEPRECATED_ACCESS, ENV_VARS, EXPECTED, OPENAI_COMPATIBLE, build_headers, classify_response, discovered_provider_names, plan_all_configured, provider_from_env_name, representative_cases, resolve_credential



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

    def test_non_provider_tokens_are_not_misclassified_as_api_credentials(self):
        self.assertIsNone(provider_from_env_name("GITHUB_TOKEN"))
        self.assertIsNone(provider_from_env_name("AWS_SESSION_TOKEN"))

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


if __name__ == "__main__":
    unittest.main()
