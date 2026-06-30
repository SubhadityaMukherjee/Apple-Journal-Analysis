import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import time_of_day
from .parse import Entry, parse_file
from .store import Store


@dataclass
class IngestReport:
    files_processed: int = 0
    entries_added: int = 0
    files_skipped: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _entry_id(file_path: str, index: int) -> str:
    h = hashlib.sha256(f"{file_path}:{index}".encode("utf-8"))
    return h.hexdigest()[:16]


def _day_of_week(ts: datetime) -> str:
    return ts.strftime("%a")


def _to_row(entry: Entry, index: int, embedding: list[float]) -> dict:
    return {
        "id": _entry_id(entry.file_path, index),
        "file_path": entry.file_path,
        "date": entry.date,
        "timestamp": entry.timestamp,
        "time_of_day": time_of_day(entry.timestamp.hour),
        "day_of_week": _day_of_week(entry.timestamp),
        "text": entry.text,
        "category": entry.category,
        "embedding": embedding,
        "ingested_at": datetime.now(),
    }


class Ingestor:
    def __init__(self, store: Store, embedder, state_path: Path):
        self.store = store
        self.embedder = embedder
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2))

    def scan_and_ingest(self, directory: Path) -> IngestReport:
        directory = Path(directory)
        state = self._load_state()
        report = IngestReport()

        for html_path in sorted(directory.glob("*.html")):
            key = str(html_path)
            stat = html_path.stat()
            fingerprint = {"mtime": stat.st_mtime, "size": stat.st_size}

            if state.get(key) == fingerprint:
                report.files_skipped += 1
                continue

            try:
                entries = parse_file(html_path)
            except Exception as exc:
                report.errors.append(f"{key}: {exc}")
                continue

            self.store.delete_by_file(key)
            embeddings = self.embedder.embed_batch([e.text for e in entries]) if entries else []
            rows = [
                _to_row(e, i, embeddings[i]) for i, e in enumerate(entries)
            ]
            if rows:
                self.store.add_entries(rows)

            state[key] = fingerprint
            report.files_processed += 1
            report.entries_added += len(rows)

        self._save_state(state)
        return report
