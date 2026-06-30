import json
from datetime import datetime
from pathlib import Path

import pytest

from journal.ingest import Ingestor
from journal.store import Store


class FakeEmbedder:
    def embed(self, text: str):
        return [0.0] * 768

    def embed_batch(self, texts: list[str]):
        return [[0.0] * 768 for _ in texts]


@pytest.fixture
def fake_entries_dir(tmp_path: Path) -> Path:
    src = Path("tests/fixtures/2020-06-29.html")
    d = tmp_path / "entries"
    d.mkdir()
    (d / "2020-06-29.html").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return d


def test_ingest_processes_new_files(tmp_path: Path, fake_entries_dir: Path):
    store = Store(tmp_path / "lance")
    store.create_tables()
    ing = Ingestor(store=store, embedder=FakeEmbedder(), state_path=tmp_path / "seen.json")

    report = ing.scan_and_ingest(fake_entries_dir)

    assert report.files_processed == 1
    assert report.entries_added > 5
    df = store.entries_to_pandas()
    assert len(df) == report.entries_added
    assert (tmp_path / "seen.json").exists()


def test_ingest_skips_unchanged_files(tmp_path: Path, fake_entries_dir: Path):
    store = Store(tmp_path / "lance")
    store.create_tables()
    ing = Ingestor(store=store, embedder=FakeEmbedder(), state_path=tmp_path / "seen.json")

    first = ing.scan_and_ingest(fake_entries_dir)
    second = ing.scan_and_ingest(fake_entries_dir)

    assert second.files_processed == 0
    assert second.entries_added == 0
    assert first.entries_added == store.entries_to_pandas().shape[0]


def test_ingest_reprocesses_changed_files(tmp_path: Path, fake_entries_dir: Path):
    store = Store(tmp_path / "lance")
    store.create_tables()
    ing = Ingestor(store=store, embedder=FakeEmbedder(), state_path=tmp_path / "seen.json")

    ing.scan_and_ingest(fake_entries_dir)
    target = fake_entries_dir / "2020-06-29.html"
    # bump mtime
    content = target.read_text(encoding="utf-8") + "\n<!-- edited -->"
    target.write_text(content, encoding="utf-8")

    second = ing.scan_and_ingest(fake_entries_dir)
    assert second.files_processed == 1
