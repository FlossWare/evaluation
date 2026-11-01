"""Deterministic tests for provider smoke-test response classification."""
import unittest

from tools.provider_api.smoke_test import EXPECTED, build_headers, classify_response


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
