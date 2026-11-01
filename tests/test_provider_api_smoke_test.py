"""Deterministic tests for provider smoke-test response classification."""
import unittest

from tools.provider_api.smoke_test import CASES, EXPECTED, OPENAI_COMPATIBLE, build_headers, classify_response, representative_cases


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

    def test_nvidia_uses_nim_chat_completions_endpoint(self):
        self.assertEqual(
            OPENAI_COMPATIBLE["NVIDIA"],
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )


class BatchSelectionTests(unittest.TestCase):
    def test_batch_selects_at_most_one_candidate_per_provider(self):
        selected = representative_cases(include_paid=False)
        self.assertEqual(len(selected), len({case["provider"] for case in selected.values()}))
        self.assertEqual(selected["GOOGLE"]["model"], "models/gemini-2.5-flash-lite")
        self.assertEqual(selected["OPENROUTER"]["model"], "nvidia/nemotron-3.5-lightning:free")
        self.assertNotIn("CEREBRAS", selected)
        self.assertNotIn("NVIDIA", selected)

    def test_paid_opt_in_adds_paid_only_providers(self):
        selected = representative_cases(include_paid=True)
        self.assertIn("CEREBRAS", selected)
        self.assertIn("DEEPINFRA", selected)
        self.assertIn("NVIDIA", selected)
        self.assertEqual(selected["NVIDIA"]["model"], "openai/gpt-oss-120b")


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
