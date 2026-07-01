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


from journal.questions.topics import TOPICS
from journal.questions import default_registry


def test_topics_template_and_id():
    assert "{text}" in TOPICS.user_template
    assert TOPICS.id == "topics"


def test_topics_validate_accepts_three_strings():
    assert TOPICS.validate({"topics": ["a", "b", "c"]})


def test_topics_validate_rejects_too_few():
    assert not TOPICS.validate({"topics": ["a", "b"]})


def test_topics_validate_rejects_non_string():
    assert not TOPICS.validate({"topics": ["a", "b", 3]})


def test_default_registry_has_both_questions():
    reg = default_registry()
    assert set(reg.list_ids()) == {"sentiment", "topics"}


from datetime import datetime
from pathlib import Path

from journal.analyze import BatchAnalyzer, AnalyzeReport
from journal.store import Store
from journal.questions.sentiment import SENTIMENT


class FakeLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def complete_json(self, system: str, user: str):
        return dict(self.payload)


def _seed_store(tmp_path: Path, n: int = 4) -> Store:
    s = Store(tmp_path / "lance")
    s.create_tables()
    rows = [
        {
            "id": f"e{i}",
            "file_path": f"/tmp/f{i}.html",
            "date": datetime(2020, 1, 1 + i).date(),
            "timestamp": datetime(2020, 1, 1 + i, 10, 0),
            "time_of_day": "morning",
            "day_of_week": "Wed",
            "text": f"body {i}",
            "category": None,
            "embedding": [0.0] * 768,
            "ingested_at": datetime(2026, 6, 30),
        }
        for i in range(n)
    ]
    s.add_entries(rows)
    return s


def test_batch_analyzer_runs_sentiment_over_all_entries(tmp_path: Path):
    store = _seed_store(tmp_path, n=4)
    llm = FakeLLM({"level": "neutral", "score": 3, "confidence": 0.7})
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    analyzer = BatchAnalyzer(store=store, llm=llm, registry=reg)

    report = analyzer.run(question_id="sentiment")

    assert report.processed == 4
    assert report.failed == 0
    adf = store.analyses_to_pandas()
    assert len(adf) == 4
    assert all(adf["parsed_ok"])
    assert all(adf["question_id"] == "sentiment")


def test_batch_analyzer_skips_already_done(tmp_path: Path):
    store = _seed_store(tmp_path, n=4)
    llm = FakeLLM({"level": "neutral", "score": 3, "confidence": 0.7})
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    analyzer = BatchAnalyzer(store=store, llm=llm, registry=reg)

    analyzer.run(question_id="sentiment")
    report2 = analyzer.run(question_id="sentiment")

    assert report2.processed == 0
    assert store.analyses_to_pandas().shape[0] == 4


def test_batch_analyzer_counts_failures(tmp_path: Path):
    store = _seed_store(tmp_path, n=2)

    class BadLLM:
        def complete_json(self, system, user):
            return None

    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    analyzer = BatchAnalyzer(store=store, llm=BadLLM(), registry=reg)
    report = analyzer.run(question_id="sentiment")
    assert report.processed == 0
    assert report.failed == 2


def test_batch_analyzer_persists_partial_progress_on_abort(tmp_path: Path):
    """If a run aborts mid-way (kernel interrupt or a raised exception), entries
    processed before the abort must already be persisted so the next run resumes
    from there instead of redoing them. Regression for the 'not keeping state' bug
    where add_analyses was called once at the very end of run()."""
    store = _seed_store(tmp_path, n=6)

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        def complete_json(self, system, user):
            self.n += 1
            if self.n == 5:
                raise RuntimeError("ollama exploded")
            return {"level": "neutral", "score": 3, "confidence": 0.7}

    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    analyzer = BatchAnalyzer(
        store=store, llm=FlakyLLM(), registry=reg,
        max_workers=1, flush_every=2,
    )

    with pytest.raises(RuntimeError):
        analyzer.run(question_id="sentiment")

    persisted = store.analyses_to_pandas()
    assert len(persisted) > 0, "partial progress should be persisted before the abort"


def test_run_many_runs_all_questions(tmp_path: Path):
    store = _seed_store(tmp_path, n=3)
    reg = QuestionRegistry()
    reg.register(_dummy_question("a"))
    reg.register(_dummy_question("b"))
    analyzer = BatchAnalyzer(
        store=store, llm=FakeLLM({"x": 1}), registry=reg,
    )

    result = analyzer.run_many(["a", "b"])

    assert [r.question_id for r in result.reports] == ["a", "b"]
    assert all(r.processed == 3 for r in result.reports)
    adf = store.analyses_to_pandas()
    assert len(adf) == 6  # 3 entries × 2 questions
    assert set(adf["question_id"]) == {"a", "b"}


def test_run_many_resumes_per_question(tmp_path: Path):
    store = _seed_store(tmp_path, n=3)
    reg = QuestionRegistry()
    reg.register(_dummy_question("a"))
    reg.register(_dummy_question("b"))
    analyzer = BatchAnalyzer(
        store=store, llm=FakeLLM({"x": 1}), registry=reg,
    )

    analyzer.run_many(["a", "b"])
    result2 = analyzer.run_many(["a", "b"])

    assert len(result2.reports) == 2
    assert all(r.processed == 0 for r in result2.reports)
    # store content unchanged: still exactly one row per (entry, question)
    assert store.analyses_to_pandas().shape[0] == 6


def test_run_many_rejects_unknown_id_before_any_work(tmp_path: Path):
    store = _seed_store(tmp_path, n=3)
    reg = QuestionRegistry()
    reg.register(_dummy_question("a"))
    analyzer = BatchAnalyzer(
        store=store, llm=FakeLLM({"x": 1}), registry=reg,
    )

    with pytest.raises(KeyError):
        analyzer.run_many(["a", "bogus"])

    # validation happens up front, so even question "a" never writes anything
    assert len(store.analyses_to_pandas()) == 0
