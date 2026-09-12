from evaluation_ai.feedback import FeedbackPatternMatcher
from evaluation_ai.types import ChatMessage


def test_feedback_matcher_detects_correction_preference_and_confirmation():
    matcher = FeedbackPatternMatcher()
    signals = matcher.match(
        [
            ChatMessage(role="assistant", content="irrelevant"),
            ChatMessage(role="user", content="You should have used the gateway instead of the client."),
            ChatMessage(role="user", content="I prefer Debian over Ubuntu."),
            ChatMessage(role="user", content="That worked well."),
        ]
    )

    assert [signal.type for signal in signals] == [
        "correction",
        "preference",
        "confirmation",
    ]
    assert signals[0].confidence == 0.9


def test_feedback_matcher_emits_one_signal_per_type_per_message():
    matcher = FeedbackPatternMatcher()
    signals = matcher.match(
        [ChatMessage(role="user", content="Don't use X. Never use X. Avoid X.")]
    )

    assert [signal.type for signal in signals] == ["preference"]


def test_feedback_matcher_ignores_non_user_messages():
    matcher = FeedbackPatternMatcher()
    signals = matcher.match(
        [ChatMessage(role="assistant", content="Perfect. You should have done this.")]
    )

    assert signals == []
