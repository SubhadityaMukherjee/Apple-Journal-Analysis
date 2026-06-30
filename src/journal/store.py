from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .config import EMBED_DIM


def _entries_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("time_of_day", pa.string()),
        pa.field("day_of_week", pa.string()),
        pa.field("text", pa.string()),
        pa.field("category", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("ingested_at", pa.timestamp("us")),
    ])


def _analyses_schema() -> pa.Schema:
    return pa.schema([
        pa.field("entry_id", pa.string()),
        pa.field("question_id", pa.string()),
        pa.field("question_text", pa.string()),
        pa.field("result_json", pa.string()),
        pa.field("parsed_ok", pa.bool_()),
        pa.field("model", pa.string()),
        pa.field("created_at", pa.timestamp("us")),
    ])


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def create_tables(self) -> None:
        if "entries" not in self.db.list_tables().tables:
            self.db.create_table("entries", schema=_entries_schema())
        if "analyses" not in self.db.list_tables().tables:
            self.db.create_table("analyses", schema=_analyses_schema())

    def table_names(self) -> list[str]:
        return self.db.list_tables().tables

    def add_entries(self, rows: list[dict]) -> None:
        self.db.open_table("entries").add(rows)

    def delete_by_file(self, file_path: str) -> None:
        safe = file_path.replace("'", "\\'")
        self.db.open_table("entries").delete(f"file_path = '{safe}'")

    def entries_to_pandas(self):
        return self.db.open_table("entries").to_pandas()

    def search_entries(self, query_vec: list[float], k: int = 10, where: str | None = None):
        q = self.db.open_table("entries").search(query_vec).limit(k)
        if where:
            q = q.where(where)
        return q.to_pandas()

    def add_analysis(self, row: dict) -> None:
        self.db.open_table("analyses").add([row])

    def add_analyses(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.db.open_table("analyses").add(rows)

    def analyses_to_pandas(self):
        return self.db.open_table("analyses").to_pandas()

    def entries_missing_analysis(self, question_id: str, model: str) -> list[str]:
        df_e = self.entries_to_pandas()
        df_a = self.analyses_to_pandas()
        done: set[str] = set()
        if len(df_a) > 0:
            mask = (df_a["question_id"] == question_id) & (df_a["model"] == model)
            done = set(df_a.loc[mask, "entry_id"].tolist())
        return [i for i in df_e["id"].tolist() if i not in done]
