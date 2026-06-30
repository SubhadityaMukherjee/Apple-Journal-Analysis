from datetime import datetime
from pathlib import Path

import pytest

from journal.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "lance")
    s.create_tables()
    return s


def _fake_entries(n: int = 3):
    return [
        {
            "id": f"id-{i}",
            "file_path": f"/tmp/f{i}.html",
            "date": datetime(2020, 6, 28 + i).date(),
            "timestamp": datetime(2020, 6, 28 + i, 10, i),
            "time_of_day": "morning",
            "day_of_week": "Tue",
            "text": f"entry body {i}",
            "category": None,
            "embedding": [float(i)] * 768,
            "ingested_at": datetime(2026, 6, 30),
        }
        for i in range(n)
    ]


def test_create_tables(store: Store):
    assert "entries" in store.table_names()
    assert "analyses" in store.table_names()


def test_add_entries_and_to_pandas(store: Store):
    store.add_entries(_fake_entries(3))
    df = store.entries_to_pandas()
    assert len(df) == 3
    assert set(["id", "file_path", "date", "text", "embedding"]).issubset(df.columns)


def test_delete_by_file(store: Store):
    store.add_entries(_fake_entries(3))
    store.delete_by_file("/tmp/f1.html")
    df = store.entries_to_pandas()
    assert len(df) == 2
    assert "/tmp/f1.html" not in set(df["file_path"])


def test_search_returns_nearest(store: Store):
    store.add_entries(_fake_entries(3))
    query = [float(i) for i in range(768)]
    results = store.search_entries(query_vec=query, k=2)
    assert len(results) == 2


def test_add_analysis(store: Store):
    store.add_entries(_fake_entries(1))
    store.add_analysis({
        "entry_id": "id-0",
        "question_id": "sentiment",
        "question_text": "...",
        "result_json": '{"level":"negative","score":2}',
        "parsed_ok": True,
        "model": "gemma3:4b",
        "created_at": datetime(2026, 6, 30),
    })
    df = store.analyses_to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["question_id"] == "sentiment"
