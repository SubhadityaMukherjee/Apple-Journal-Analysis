import pytest

from journal.analyze import Question, QuestionRegistry


def _dummy_question(qid: str = "dummy") -> Question:
    return Question(
        id=qid,
        system_prompt="sys",
        user_template="Process: {text}",
        validate=lambda d: "x" in d,
    )


def test_registry_register_and_get():
    reg = QuestionRegistry()
    q = _dummy_question()
    reg.register(q)
    assert reg.get("dummy") is q


def test_registry_list_ids():
    reg = QuestionRegistry()
    reg.register(_dummy_question("a"))
    reg.register(_dummy_question("b"))
    assert set(reg.list_ids()) == {"a", "b"}


def test_registry_get_missing_raises():
    reg = QuestionRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


from journal.questions.sentiment import SENTIMENT, SENTIMENT_LEVELS


def test_sentiment_has_well_formed_template():
    assert "{text}" in SENTIMENT.user_template
    assert SENTIMENT.id == "sentiment"


def test_sentiment_validate_accepts_known_levels():
    for level in SENTIMENT_LEVELS:
        assert SENTIMENT.validate({"level": level, "score": 3, "confidence": 0.5})


def test_sentiment_validate_rejects_unknown_level():
    assert not SENTIMENT.validate({"level": "meh", "score": 3, "confidence": 0.5})


def test_sentiment_validate_rejects_out_of_range_score():
    assert not SENTIMENT.validate({"level": "neutral", "score": 9, "confidence": 0.5})
