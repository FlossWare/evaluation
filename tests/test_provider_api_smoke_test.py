"""Deterministic tests for provider smoke-test response classification."""
import unittest

from tools.provider_api.smoke_test import CASES, DEPRECATED_ACCESS, EXPECTED, OPENAI_COMPATIBLE, build_headers, classify_response, representative_cases


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
        self.assertEqual(selected["MISTRAL"]["model"], "mistral-small-latest")
        self.assertEqual(selected["COHERE"]["model"], "command-r7b-12-2024")
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
