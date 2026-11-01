"""Deterministic tests for provider smoke-test response classification."""
import unittest

from tools.provider_api.smoke_test import EXPECTED, classify_response


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
