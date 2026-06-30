from datetime import datetime
from pathlib import Path

import pytest

from journal.query import answer_question
from journal.store import Store


@pytest.fixture
def seeded_store(tmp_path: Path) -> Store:
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
            "text": f"body about topic{i}",
            "category": None,
            "embedding": [float(i)] * 768,
            "ingested_at": datetime(2026, 6, 30),
        }
        for i in range(5)
    ]
    s.add_entries(rows)
    return s


class FakeEmbedder:
    def embed(self, text):
        return [0.5] * 768

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeLLM:
    def __init__(self):
        self.received = None

    def complete_json(self, system, user):
        return None

    def complete(self, system, user):
        self.received = (system, user)
        return "ANSWER"


def test_query_returns_answer_and_sources(seeded_store):
    llm = FakeLLM()
    result = answer_question(
        store=seeded_store,
        embedder=FakeEmbedder(),
        llm=llm,
        question="what is topic2?",
        k=3,
    )
    assert result.answer == "ANSWER"
    assert len(result.sources) == 3
    assert any("topic" in t for t in result.sources["text"])
