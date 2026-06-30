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
