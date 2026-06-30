# Journal Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local pipeline that ingests ~1,831 Apple Journal HTML files into a persistent LanceDB vector store, runs batch LLM analyses (sentiment, topics) over timestamped entries, and exposes both ad-hoc Q&A and Jupyter dashboards.

**Architecture:** Two-stage pipeline. Stage 1 (fast): parse HTML → embed entries with `nomic-embed-text` → store in LanceDB. Stage 2 (slow): run registered LLM questions over entries with `gemma3:4b`, store strict-JSON results in a separate `analyses` table. Ad-hoc Q&A is a separate retrieval path.

**Tech Stack:** Python 3.12+, uv, Ollama (gemma3:4b, nomic-embed-text), LanceDB, BeautifulSoup4, pandas, Plotly, JupyterLab, pytest.

**Spec:** `docs/superpowers/specs/2026-06-30-journal-analysis-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv-managed deps and project metadata |
| `src/journal/__init__.py` | package marker, version |
| `src/journal/config.py` | all paths, model names, time-of-day buckets |
| `src/journal/parse.py` | HTML → list of `Entry` dataclasses |
| `src/journal/store.py` | LanceDB wrapper: tables, upsert, search, to_pandas |
| `src/journal/embed.py` | `OllamaEmbedder` (Protocol + concrete impl) |
| `src/journal/llm.py` | `OllamaLLM` with strict-JSON completion + retries |
| `src/journal/ingest.py` | dir scan + state diff + parse/embed/write |
| `src/journal/analyze.py` | `Question` protocol, `QuestionRegistry`, `BatchAnalyzer` |
| `src/journal/query.py` | ad-hoc retrieval + LLM Q&A |
| `src/journal/questions/sentiment.py` | sentiment question def |
| `src/journal/questions/topics.py` | topics question def |
| `tests/conftest.py` | fixtures: sample HTML path, temp DB |
| `tests/fixtures/2020-06-29.html` | real HTML fixture |
| `tests/test_parse.py` | parser tests |
| `tests/test_store.py` | LanceDB wrapper tests |
| `tests/test_llm.py` | JSON-parsing/retry tests with fake client |
| `tests/test_ingest.py` | end-to-end ingest with fake embedder |
| `tests/test_analyze.py` | registry + batch analyzer with fakes |
| `tests/test_query.py` | ad-hoc query with fakes |
| `notebooks/01_ingest_and_explore.ipynb` | first-run ingest + sanity stats |
| `notebooks/02_sentiment.ipynb` | run sentiment batch + day-of-week / time-of-day graphs |
| `notebooks/03_topics.ipynb` | run topics batch + frequency graphs |
| `notebooks/04_ad_hoc_qa.ipynb` | interactive Q&A |
| `notebooks/05_dashboard.ipynb` | combined dashboard |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/journal/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `README.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "journal"
version = "0.1.0"
description = "Local analysis pipeline for Apple Journal entries"
requires-python = ">=3.12"
dependencies = [
    "ollama>=0.4",
    "lancedb>=0.17",
    "pyarrow>=17",
    "beautifulsoup4>=4.12",
    "pandas>=2.2",
    "plotly>=5.24",
    "matplotlib>=3.9",
    "tqdm>=4.66",
    "python-dateutil>=2.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/journal"]

[dependency-groups]
dev = [
    "pytest>=8",
    "jupyterlab>=4",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `src/journal/__init__.py`**

```python
"""Local analysis pipeline for Apple Journal entries."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write `tests/__init__.py` (empty)**

```python
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_html_path() -> Path:
    return FIXTURES / "2020-06-29.html"
```

- [ ] **Step 5: Write `README.md`**

```markdown
# journal_stata

Local analysis pipeline for Apple Journal HTML exports.

See `docs/superpowers/specs/2026-06-30-journal-analysis-design.md` for the design.

## Setup

```bash
uv sync
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Use

Open notebooks in `notebooks/`. The pipeline modules live in `src/journal/`.
```

- [ ] **Step 6: Install deps**

Run: `uv sync`
Expected: deps install into `.venv/`, lockfile created.

- [ ] **Step 7: Verify pytest runs (no tests yet)**

Run: `uv run pytest -q`
Expected: `no tests ran` (or similar) — no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/ README.md
git commit -m "Scaffold project: pyproject, package, tests skeleton"
```

---

## Task 2: Config module

**Files:**
- Create: `src/journal/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test `tests/test_config.py`**

```python
from journal.config import (
    ENTRIES_DIR,
    LANCE_ROOT,
    LLM_MODEL,
    EMBED_MODEL,
    EMBED_DIM,
    time_of_day,
)


def test_paths_are_absolute():
    assert ENTRIES_DIR.is_absolute()
    assert LANCE_ROOT.is_absolute()


def test_model_names():
    assert LLM_MODEL == "gemma3:4b"
    assert EMBED_MODEL == "nomic-embed-text"
    assert EMBED_DIM == 768


def test_time_of_day_buckets():
    assert time_of_day(5) == "morning"
    assert time_of_day(11) == "morning"
    assert time_of_day(12) == "afternoon"
    assert time_of_day(16) == "afternoon"
    assert time_of_day(17) == "evening"
    assert time_of_day(20) == "evening"
    assert time_of_day(21) == "night"
    assert time_of_day(0) == "night"
    assert time_of_day(4) == "night"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_config.py -v`
Expected: ImportError on `journal.config`.

- [ ] **Step 3: Write `src/journal/config.py`**

```python
from pathlib import Path

ENTRIES_DIR = Path("/Users/smukherjee/Documents/Archives/Backups/AppleJournalEntries/Entries")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANCE_ROOT = PROJECT_ROOT / "data" / "lance"
STATE_DIR = PROJECT_ROOT / "state"
SEEN_FILES_PATH = STATE_DIR / "seen_files.json"
INGEST_ERRORS_PATH = STATE_DIR / "ingest_errors.json"

LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/journal/config.py tests/test_config.py
git commit -m "Add config module: paths, model names, time-of-day buckets"
```

---

## Task 3: HTML parser

**Files:**
- Create: `tests/fixtures/2020-06-29.html` (copy of real file)
- Create: `src/journal/parse.py`
- Create: `tests/test_parse.py`

- [ ] **Step 1: Copy a real HTML file as fixture**

Run:
```bash
mkdir -p tests/fixtures
cp /Users/smukherjee/Documents/Archives/Backups/AppleJournalEntries/Entries/2020-06-29.html tests/fixtures/
```

Expected: file exists at `tests/fixtures/2020-06-29.html`.

- [ ] **Step 2: Write failing test `tests/test_parse.py`**

```python
from datetime import datetime
from pathlib import Path

from journal.parse import parse_file


def test_parse_file_returns_entries(sample_html_path: Path):
    entries = parse_file(sample_html_path)
    assert len(entries) > 5
    first = entries[0]
    assert first.date.isoformat() == "2020-06-30"
    assert first.timestamp == datetime(2020, 6, 30, 0, 0)
    assert "I don't even like their way of life" in first.text
    assert first.category == "Stress"


def test_parse_file_extracts_timestamps(sample_html_path: Path):
    entries = parse_file(sample_html_path)
    timestamps = [e.timestamp for e in entries]
    assert datetime(2020, 6, 30, 12, 44) in timestamps
    assert datetime(2020, 6, 30, 22, 34) in timestamps


def test_parse_file_gratitude_category(sample_html_path: Path):
    entries = parse_file(sample_html_path)
    gratitude = [e for e in entries if e.category == "Gratitude"]
    assert len(gratitude) >= 3


def test_parse_file_strips_html(sample_html_path: Path):
    entries = parse_file(sample_html_path)
    for e in entries:
        assert "<" not in e.text
        assert "Cocoa HTML Writer" not in e.text
```

- [ ] **Step 3: Run test, expect FAIL**

Run: `uv run pytest tests/test_parse.py -v`
Expected: ImportError on `journal.parse`.

- [ ] **Step 4: Write `src/journal/parse.py`**

```python
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

TIMESTAMP_RE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2}),\s+(\d{4})\s+-\s+(\d{1,2}):(\d{2})\s+(AM|PM)$"
)

KNOWN_CATEGORIES = {
    "Gratitude", "Stress", "Notes", "Joy", "Sadness", "Anger",
    "Fear", "Pride", "Hope", "Love", "Calm", "Excitement",
    "Anxiety", "Frustration", "Confusion", "Reflection",
}


@dataclass
class Entry:
    file_path: str
    date: date
    timestamp: datetime
    text: str
    category: str | None


def _parse_filename_date(path: Path) -> date:
    stem = path.stem.split("_")[0]
    return datetime.strptime(stem, "%Y-%m-%d").date()


def _parse_timestamp(line: str) -> datetime | None:
    m = TIMESTAMP_RE.match(line.strip())
    if not m:
        return None
    _, month, day, year, hour, minute, ampm = m.groups()
    month_num = datetime.strptime(month, "%b").month
    hour = int(hour) % 12
    if ampm == "PM":
        hour += 12
    return datetime(int(year), month_num, int(day), hour, int(minute))


def _extract_category(text: str) -> tuple[str, str | None]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return text, None
    last = lines[-1]
    if last in KNOWN_CATEGORIES and len(lines) > 1:
        body = "\n".join(lines[:-1])
        return body, last
    return text, None


def parse_file(path: Path) -> list[Entry]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for style in soup(["style", "script"]):
        style.decompose()
    file_date = _parse_filename_date(path)

    raw_lines: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ").strip()
        if text:
            raw_lines.append(text)

    entries: list[Entry] = []
    current_ts: datetime | None = None
    current_buf: list[str] = []

    for line in raw_lines:
        ts = _parse_timestamp(line)
        if ts is not None:
            if current_ts is not None and current_buf:
                body, cat = _extract_category("\n".join(current_buf))
                entries.append(Entry(
                    file_path=str(path),
                    date=file_date,
                    timestamp=current_ts.replace(),
                    text=body,
                    category=cat,
                ))
            current_ts = ts
            current_buf = []
        else:
            if current_ts is not None:
                current_buf.append(line)

    if current_ts is not None and current_buf:
        body, cat = _extract_category("\n".join(current_buf))
        entries.append(Entry(
            file_path=str(path),
            date=file_date,
            timestamp=current_ts,
            text=body,
            category=cat,
        ))

    return entries
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_parse.py -v`
Expected: 4 passed.

If any fail, inspect actual output by running:
```bash
uv run python -c "from journal.parse import parse_file; from pathlib import Path; es = parse_file(Path('tests/fixtures/2020-06-29.html')); [print(repr(e.timestamp), repr(e.category), e.text[:60]) for e in es[:8]]"
```
Adjust parser accordingly.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/2020-06-29.html tests/test_parse.py src/journal/parse.py
git commit -m "Add HTML parser: extract timestamped entries from Apple Journal exports"
```

---

## Task 4: LanceDB store wrapper

**Files:**
- Create: `src/journal/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing test `tests/test_store.py`**

```python
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
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
            "date": datetime(2020, 6, 30 + i).date(),
            "timestamp": datetime(2020, 6, 30 + i, 10, i),
            "time_of_day": "morning",
            "day_of_week": "Tue",
            "text": f"entry body {i}",
            "category": None,
            "embedding": [float(i)] * 8,
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
    results = store.search_entries(query_vec=[0.0, 1.0, 2.0, 0, 0, 0, 0, 0], k=2)
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_store.py -v`
Expected: ImportError on `journal.store`.

- [ ] **Step 3: Write `src/journal/store.py`**

```python
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
        if "entries" not in self.db.table_names():
            self.db.create_table("entries", schema=_entries_schema())
        if "analyses" not in self.db.table_names():
            self.db.create_table("analyses", schema=_analyses_schema())

    def table_names(self) -> list[str]:
        return self.db.table_names()

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
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/journal/store.py tests/test_store.py
git commit -m "Add LanceDB store wrapper: entries + analyses tables, search, to_pandas"
```

---

## Task 5: Ollama embedder and LLM client

**Files:**
- Create: `src/journal/embed.py`
- Create: `src/journal/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test `tests/test_llm.py`**

```python
import pytest

from journal.llm import OllamaLLM


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, model: str, messages: list, format=None):
        self.calls += 1
        return {"message": {"content": self.responses.pop(0)}}


def test_complete_json_returns_parsed_dict():
    client = FakeClient(['{"level":"negative","score":2}'])
    llm = OllamaLLM(model="gemma3:4b", client=client)
    result = llm.complete_json(system="sys", user="usr")
    assert result == {"level": "negative", "score": 2}
    assert client.calls == 1


def test_complete_json_strips_code_fences():
    client = FakeClient(['```json\n{"x": 1}\n```'])
    llm = OllamaLLM(model="m", client=client)
    assert llm.complete_json("s", "u") == {"x": 1}


def test_complete_json_retries_on_invalid_then_succeeds():
    client = FakeClient(["not json", '{"x": 1}'])
    llm = OllamaLLM(model="m", client=client)
    assert llm.complete_json("s", "u") == {"x": 1}
    assert client.calls == 2


def test_complete_json_returns_none_after_max_retries():
    client = FakeClient(["no", "no", "no"])
    llm = OllamaLLM(model="m", client=client, max_retries=3)
    assert llm.complete_json("s", "u") is None
    assert client.calls == 3
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_llm.py -v`
Expected: ImportError on `journal.llm`.

- [ ] **Step 3: Write `src/journal/embed.py`**

```python
from typing import Protocol

import ollama

from .config import EMBED_MODEL


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, client=None):
        self.model = model
        self._client = client or ollama.Client()

    def embed(self, text: str) -> list[float]:
        resp = self._client.embed(model=self.model, input=text)
        return resp["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(model=self.model, input=texts)
        return resp["embeddings"]
```

- [ ] **Step 4: Write `src/journal/llm.py`**

```python
import json
import re
from typing import Protocol

import ollama

from .config import LLM_MODEL


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict | None: ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


class OllamaLLM:
    def __init__(self, model: str = LLM_MODEL, client=None, max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self._client = client or ollama.Client()

    def complete_json(self, system: str, user: str) -> dict | None:
        for _ in range(self.max_retries):
            resp = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
            )
            content = resp["message"]["content"]
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
        return None
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/journal/embed.py src/journal/llm.py tests/test_llm.py
git commit -m "Add Ollama embedder and LLM client with JSON-extraction + retries"
```

---

## Task 6: Pull Ollama models

**Files:** (none — environment setup only)

- [ ] **Step 1: Verify Ollama is running**

Run: `ollama list`
Expected: lists currently installed models (may be empty).

If command not found, install from https://ollama.com first.
If `ollama list` errors with connection refused, run `ollama serve` in another terminal.

- [ ] **Step 2: Pull embedding model**

Run: `ollama pull nomic-embed-text`
Expected: downloads ~270 MB; completes with `success`.

- [ ] **Step 3: Pull LLM**

Run: `ollama pull gemma3:4b`
Expected: downloads ~2.5 GB; completes with `success`.

- [ ] **Step 4: Verify both are available**

Run: `ollama list`
Expected: output contains `nomic-embed-text` and `gemma3:4b`.

- [ ] **Step 5: Smoke-test embedding**

Run:
```bash
uv run python -c "
from journal.embed import OllamaEmbedder
e = OllamaEmbedder()
v = e.embed('hello world')
print('dim:', len(v))
assert len(v) == 768
print('ok')
"
```
Expected: prints `dim: 768` and `ok`.

- [ ] **Step 6: Smoke-test LLM**

Run:
```bash
uv run python -c "
from journal.llm import OllamaLLM
llm = OllamaLLM()
r = llm.complete_json('You return JSON.', 'Return {\"ok\": true}.')
print(r)
assert r == {'ok': True}
print('ok')
"
```
Expected: prints `{'ok': True}` and `ok`.

(No commit — this task only pulls models.)

---

## Task 7: Ingestion pipeline

**Files:**
- Create: `src/journal/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing test `tests/test_ingest.py`**

```python
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
    (d / "2020-06-29.html").write_text(src.read_text(), encoding="utf-8")
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: ImportError on `journal.ingest`.

- [ ] **Step 3: Write `src/journal/ingest.py`**

```python
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
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/journal/ingest.py tests/test_ingest.py
git commit -m "Add ingestion pipeline: dir scan, state diff, embed, write"
```

---

## Task 8: Question registry

**Files:**
- Create: `src/journal/questions/__init__.py`
- Modify: `src/journal/analyze.py` (create)
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Write failing test `tests/test_analyze.py`**

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: ImportError on `journal.analyze`.

- [ ] **Step 3: Write `src/journal/questions/__init__.py` (empty)**

```python
"""Built-in question definitions."""
```

- [ ] **Step 4: Write `src/journal/analyze.py`**

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Question:
    id: str
    system_prompt: str
    user_template: str
    validate: Callable[[dict], bool] = lambda d: True


class QuestionRegistry:
    def __init__(self):
        self._questions: dict[str, Question] = {}

    def register(self, question: Question) -> None:
        self._questions[question.id] = question

    def get(self, qid: str) -> Question:
        return self._questions[qid]

    def list_ids(self) -> list[str]:
        return list(self._questions.keys())
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/journal/analyze.py src/journal/questions/__init__.py tests/test_analyze.py
git commit -m "Add Question type and QuestionRegistry"
```

---

## Task 9: Sentiment question

**Files:**
- Create: `src/journal/questions/sentiment.py`
- Modify: `src/journal/questions/__init__.py`

- [ ] **Step 1: Add test to `tests/test_analyze.py`**

Append:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: ImportError on `journal.questions.sentiment`.

- [ ] **Step 3: Write `src/journal/questions/sentiment.py`**

```python
from ..analyze import Question

SENTIMENT_LEVELS = {
    "extreme_negative",
    "negative",
    "neutral",
    "positive",
    "extreme_positive",
}

SYSTEM = (
    "You are analyzing a personal journal entry. Classify the overall emotional tone. "
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = """Respond with ONLY a JSON object of the form:
{"level": "<one of: extreme_negative, negative, neutral, positive, extreme_positive>", "score": <integer 1-5>, "confidence": <float 0.0-1.0>}

Scale:
1 / extreme_negative: anguish, crisis language, overwhelming distress
2 / negative: frustrated, sad, upset, but contained
3 / neutral: factual, no strong affect
4 / positive: happy, grateful, content
5 / extreme_positive: ecstatic, overjoyed, deep fulfillment

Journal entry:
{text}"""


def _validate(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    level = d.get("level")
    score = d.get("score")
    confidence = d.get("confidence")
    if level not in SENTIMENT_LEVELS:
        return False
    if not isinstance(score, int) or not (1 <= score <= 5):
        return False
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False
    return True


SENTIMENT = Question(
    id="sentiment",
    system_prompt=SYSTEM,
    user_template=USER_TEMPLATE,
    validate=_validate,
)
```

- [ ] **Step 4: Update `src/journal/questions/__init__.py`**

```python
"""Built-in question definitions."""

from ..analyze import QuestionRegistry
from .sentiment import SENTIMENT


def default_registry() -> QuestionRegistry:
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    return reg
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: 7 passed (3 prior + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/journal/questions/sentiment.py src/journal/questions/__init__.py tests/test_analyze.py
git commit -m "Add sentiment question with 5-level scale + validator"
```

---

## Task 10: Topics question

**Files:**
- Create: `src/journal/questions/topics.py`
- Modify: `src/journal/questions/__init__.py`
- Modify: `tests/test_analyze.py`

- [ ] **Step 1: Add test to `tests/test_analyze.py`**

Append:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: ImportError on `journal.questions.topics`.

- [ ] **Step 3: Write `src/journal/questions/topics.py`**

```python
from ..analyze import Question

SYSTEM = (
    "You are analyzing a personal journal entry. Identify the most prominent topics. "
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = """Identify the top 3 topics in this journal entry.
Topics must be short phrases (1-3 words), lowercase, and concrete (not "life" or "feelings").

Respond with ONLY a JSON object of the form:
{"topics": ["topic_one", "topic_two", "topic_three"]}

Journal entry:
{text}"""


def _validate(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    topics = d.get("topics")
    if not isinstance(topics, list):
        return False
    if len(topics) != 3:
        return False
    if not all(isinstance(t, str) and t.strip() for t in topics):
        return False
    return True


TOPICS = Question(
    id="topics",
    system_prompt=SYSTEM,
    user_template=USER_TEMPLATE,
    validate=_validate,
)
```

- [ ] **Step 4: Update `src/journal/questions/__init__.py`**

```python
"""Built-in question definitions."""

from ..analyze import QuestionRegistry
from .sentiment import SENTIMENT
from .topics import TOPICS


def default_registry() -> QuestionRegistry:
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    reg.register(TOPICS)
    return reg
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: 12 passed (7 prior + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/journal/questions/topics.py src/journal/questions/__init__.py tests/test_analyze.py
git commit -m "Add topics question (top-3 short phrases) and default registry"
```

---

## Task 11: Batch analyzer

**Files:**
- Modify: `src/journal/analyze.py`
- Modify: `tests/test_analyze.py`

- [ ] **Step 1: Append test to `tests/test_analyze.py`**

```python
from datetime import datetime
from pathlib import Path

from journal.analyze import BatchAnalyzer, AnalyzeReport
from journal.store import Store


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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: ImportError on `BatchAnalyzer`.

- [ ] **Step 3: Extend `src/journal/analyze.py`**

Append at end of file:

```python
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AnalyzeReport:
    question_id: str
    processed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchAnalyzer:
    store: object
    llm: object
    registry: QuestionRegistry
    model: str = "gemma3:4b"
    max_workers: int = 4

    def run(self, question_id: str) -> AnalyzeReport:
        question = self.registry.get(question_id)
        report = AnalyzeReport(question_id=question_id)
        pending_ids = self.store.entries_missing_analysis(question_id, self.model)
        if not pending_ids:
            return report

        df = self.store.entries_to_pandas().set_index("id")
        rows_to_write: list[dict] = []

        def _one(entry_id: str) -> dict:
            text = df.loc[entry_id, "text"]
            user_msg = question.user_template.format(text=text)
            parsed = self.llm.complete_json(
                system=question.system_prompt, user=user_msg
            )
            ok = parsed is not None and question.validate(parsed)
            if ok:
                result_json = json.dumps(parsed)
            elif parsed is not None:
                result_json = json.dumps({"_invalid": parsed})
            else:
                result_json = json.dumps({"_error": "no_json"})
            return {
                "entry_id": entry_id,
                "question_id": question_id,
                "question_text": user_msg,
                "result_json": result_json,
                "parsed_ok": bool(ok),
                "model": self.model,
                "created_at": datetime.now(),
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_one, eid): eid for eid in pending_ids}
            for fut in as_completed(futures):
                row = fut.result()
                rows_to_write.append(row)
                if row["parsed_ok"]:
                    report.processed += 1
                else:
                    report.failed += 1

        self.store.add_analyses(rows_to_write)
        return report
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/journal/analyze.py tests/test_analyze.py
git commit -m "Add BatchAnalyzer: parallel LLM runs, resumable via analyses table"
```

---

## Task 12: Ad-hoc query

**Files:**
- Create: `src/journal/query.py`
- Create: `tests/test_query.py`

- [ ] **Step 1: Write failing test `tests/test_query.py`**

```python
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
    assert "topic2" in result.sources.iloc[0]["text"] or any(
        "topic" in t for t in result.sources["text"]
    )
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_query.py -v`
Expected: ImportError on `journal.query`.

- [ ] **Step 3: Write `src/journal/query.py`**

```python
from dataclasses import dataclass

import pandas as pd


@dataclass
class QueryAnswer:
    answer: str
    sources: pd.DataFrame


def answer_question(store, embedder, llm, question: str, k: int = 10) -> QueryAnswer:
    qvec = embedder.embed(question)
    sources = store.search_entries(query_vec=qvec, k=k)
    context = "\n\n---\n\n".join(f"[{i}] {row['text']}" for i, (_, row) in enumerate(sources.iterrows()))

    system = (
        "You answer questions about a user's personal journal. "
        "Use the provided excerpts. Cite excerpts by their [N] index when relevant."
    )
    user = f"Question: {question}\n\nExcerpts:\n{context}"
    answer = llm.complete(system=system, user=user)
    return QueryAnswer(answer=answer, sources=sources)
```

- [ ] **Step 4: Extend `src/journal/llm.py` with a non-JSON completion method**

Add to the `LLM` Protocol and `OllamaLLM` class:

```python
class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict | None: ...
    def complete(self, system: str, user: str) -> str: ...
```

Add method to `OllamaLLM`:

```python
    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp["message"]["content"]
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/test_query.py tests/test_llm.py -v`
Expected: all pass (4 prior LLM + 1 query).

- [ ] **Step 6: Commit**

```bash
git add src/journal/query.py src/journal/llm.py tests/test_query.py
git commit -m "Add ad-hoc Q&A: embed question, retrieve, LLM answer with sources"
```

---

## Task 13: Notebook 01 — ingest and explore

**Files:**
- Create: `notebooks/01_ingest_and_explore.ipynb`

- [ ] **Step 1: Create directory**

Run: `mkdir -p notebooks`

- [ ] **Step 2: Write notebook as JSON**

Create `notebooks/01_ingest_and_explore.ipynb` with this content (4 cells: markdown intro, imports + run, sanity stats, sample rows):

```bash
uv run python -c "
import nbformat as nb
from pathlib import Path

cells = []
cells.append(nb.v4.new_markdown_cell('''# 01 — Ingest & Explore

First-run notebook. Scans the entries directory, embeds all new entries, and writes them to LanceDB. After this, the data is ready for analysis notebooks.'''))

cells.append(nb.v4.new_code_cell('''import sys
sys.path.insert(0, \"../src\")

from journal.config import ENTRIES_DIR, LANCE_ROOT
from journal.embed import OllamaEmbedder
from journal.ingest import Ingestor
from journal.store import Store

LANCE_ROOT.mkdir(parents=True, exist_ok=True)
store = Store(LANCE_ROOT)
store.create_tables()
ingestor = Ingestor(store=store, embedder=OllamaEmbedder(), state_path=LANCE_ROOT.parent.parent / \"state\" / \"seen_files.json\")
report = ingestor.scan_and_ingest(ENTRIES_DIR)
print(report)'''))

cells.append(nb.v4.new_code_cell('''df = store.entries_to_pandas()
print(f\"total entries: {len(df):,}\")
print(f\"date range: {df[\\\"date\\\"].min()} → {df[\\\"date\\\"].max()}\")
print()
print(\"entries per year:\")
print(df.groupby(df[\\\"date\\\"].dt.year).size())'''))

cells.append(nb.v4.new_code_cell('''df.head(3)'''))

nbnode = nb.v4.new_notebook()
nbnode[\"cells\"] = cells
Path(\"notebooks/01_ingest_and_explore.ipynb\").write_text(nb.writes(nbnode))
print(\"ok\")
"
```

Expected: prints `ok`, file exists.

- [ ] **Step 3: Add `nbformat` to dev deps**

Edit `pyproject.toml`, add `"nbformat>=5.10"` to `[dependency-groups].dev`. Run: `uv sync`.

- [ ] **Step 4: Execute notebook to verify it runs**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_ingest_and_explore.ipynb --output 01_ingest_and_explore.ipynb
```
Expected: completes without error. Total entries should be 5,000–20,000 (this run embeds everything for the first time; expect ~10–15 minutes).

- [ ] **Step 5: Spot-check the LanceDB output**

Run:
```bash
uv run python -c "
from pathlib import Path
from journal.store import Store
from journal.config import LANCE_ROOT
df = Store(LANCE_ROOT).entries_to_pandas()
print('rows:', len(df))
print('cols:', list(df.columns))
print('first row text:', df.iloc[0]['text'][:120])
"
```
Expected: prints row count > 1,000 and a sample of clean text.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock notebooks/
git commit -m "Add ingest+explore notebook; run first full ingest"
```

---

## Task 14: Notebook 02 — sentiment batch + day/time graphs

**Files:**
- Create: `notebooks/02_sentiment.ipynb`

- [ ] **Step 1: Write notebook generator script**

Run:
```bash
uv run python -c "
import nbformat as nb
from pathlib import Path

cells = []
cells.append(nb.v4.new_markdown_cell('''# 02 — Sentiment Analysis

Runs the sentiment question over every entry (resumable), then plots:
- sentiment distribution by day of week
- sentiment distribution by time of day
- average sentiment score over time (weekly)'''))

cells.append(nb.v4.new_code_cell('''import sys, json
sys.path.insert(0, \"../src\")

import pandas as pd
import plotly.express as px

from journal.config import LANCE_ROOT, LLM_MODEL
from journal.analyze import BatchAnalyzer
from journal.llm import OllamaLLM
from journal.questions.sentiment import SENTIMENT
from journal.analyze import QuestionRegistry
from journal.store import Store

store = Store(LANCE_ROOT)
reg = QuestionRegistry(); reg.register(SENTIMENT)
analyzer = BatchAnalyzer(store=store, llm=OllamaLLM(), registry=reg, model=LLM_MODEL, max_workers=4)
report = analyzer.run(\"sentiment\")
print(report)'''))

cells.append(nb.v4.new_code_cell('''adf = store.analyses_to_pandas()
adf = adf[adf[\"question_id\"] == \"sentiment\"].copy()
adf = adf[adf[\"parsed_ok\"]].copy()
adf[\"level\"] = adf[\"result_json\"].apply(lambda s: json.loads(s)[\"level\"])
adf[\"score\"] = adf[\"result_json\"].apply(lambda s: json.loads(s)[\"score\"])

edf = store.entries_to_pandas()[[\"id\", \"date\", \"day_of_week\", \"time_of_day\"]]
df = adf.merge(edf, left_on=\"entry_id\", right_on=\"id\")
df.head()'''))

cells.append(nb.v4.new_code_cell('''order = [\"extreme_negative\", \"negative\", \"neutral\", \"positive\", \"extreme_positive\"]
day_order = [\"Mon\", \"Tue\", \"Wed\", \"Thu\", \"Fri\", \"Sat\", \"Sun\"]
ct = df.groupby([\"day_of_week\", \"level\"]).size().reset_index(name=\"count\")
ct[\"day_of_week\"] = pd.Categorical(ct[\"day_of_week\"], categories=day_order, ordered=True)
fig = px.bar(ct, x=\"day_of_week\", y=\"count\", color=\"level\",
             category_orders={\"level\": order},
             title=\"Sentiment distribution by day of week\")
fig.show()'''))

cells.append(nb.v4.new_code_cell('''tod_order = [\"morning\", \"afternoon\", \"evening\", \"night\"]
ct = df.groupby([\"time_of_day\", \"level\"]).size().reset_index(name=\"count\")
ct[\"time_of_day\"] = pd.Categorical(ct[\"time_of_day\"], categories=tod_order, ordered=True)
fig = px.bar(ct, x=\"time_of_day\", y=\"count\", color=\"level\",
             category_orders={\"level\": order},
             title=\"Sentiment distribution by time of day\")
fig.show()'''))

cells.append(nb.v4.new_code_cell('''df[\"date\"] = pd.to_datetime(df[\"date\"])
weekly = df.set_index(\"date\")[\"score\"].resample(\"W\").mean().reset_index()
fig = px.line(weekly, x=\"date\", y=\"score\", title=\"Average sentiment score over time (weekly)\")
fig.show()'''))

nbnode = nb.v4.new_notebook()
nbnode[\"cells\"] = cells
Path(\"notebooks/02_sentiment.ipynb\").write_text(nb.writes(nbnode))
print(\"ok\")
"
```

Expected: prints `ok`.

- [ ] **Step 2: Execute notebook**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_sentiment.ipynb --output 02_sentiment.ipynb --ExecutePreprocessor.timeout=36000
```

Expected: completes without error. This is the slow step (~4–8h for full corpus on first run). To validate the pipeline end-to-end first, run on a date-sliced subset:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from journal.config import LANCE_ROOT, LLM_MODEL
from journal.analyze import BatchAnalyzer, QuestionRegistry
from journal.llm import OllamaLLM
from journal.questions.sentiment import SENTIMENT
from journal.store import Store

store = Store(LANCE_ROOT)
df = store.entries_to_pandas()
recent_ids = set(df.sort_values('date').tail(200)['id'])
# scope analyzer by temporarily filtering store — quick check uses raw run on the tail
reg = QuestionRegistry(); reg.register(SENTIMENT)
a = BatchAnalyzer(store=store, llm=OllamaLLM(), registry=reg, model=LLM_MODEL, max_workers=4)
print(a.run('sentiment'))
"
```
Expected: reports processed + failed counts; 0 failed ideally.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_sentiment.ipynb
git commit -m "Add sentiment notebook: batch run + day-of-week / time-of-day graphs"
```

---

## Task 15: Notebook 03 — topics batch + frequency graphs

**Files:**
- Create: `notebooks/03_topics.ipynb`

- [ ] **Step 1: Write notebook generator script**

Run:
```bash
uv run python -c "
import nbformat as nb
from pathlib import Path

cells = []
cells.append(nb.v4.new_markdown_cell('''# 03 — Topic Analysis

Runs the topics question over every entry (resumable), then plots topic frequency.'''))

cells.append(nb.v4.new_code_cell('''import sys, json
sys.path.insert(0, \"../src\")

import pandas as pd
import plotly.express as px

from journal.config import LANCE_ROOT, LLM_MODEL
from journal.analyze import BatchAnalyzer, QuestionRegistry
from journal.llm import OllamaLLM
from journal.questions.topics import TOPICS
from journal.store import Store

store = Store(LANCE_ROOT)
reg = QuestionRegistry(); reg.register(TOPICS)
analyzer = BatchAnalyzer(store=store, llm=OllamaLLM(), registry=reg, model=LLM_MODEL, max_workers=4)
print(analyzer.run(\"topics\"))'''))

cells.append(nb.v4.new_code_cell('''adf = store.analyses_to_pandas()
adf = adf[(adf[\"question_id\"] == \"topics\") & adf[\"parsed_ok\"]].copy()
adf[\"topics\"] = adf[\"result_json\"].apply(lambda s: json.loads(s)[\"topics\"])
adf.head()'''))

cells.append(nb.v4.new_code_cell('''exploded = adf.explode(\"topics\")
top = exploded[\"topics\"].value_counts().head(30).reset_index()
top.columns = [\"topic\", \"count\"]
fig = px.bar(top, x=\"count\", y=\"topic\", orientation=\"h\",
             title=\"Top 30 topics across all entries\")
fig.update_layout(yaxis={\"categoryorder\": \"total ascending\"})
fig.show()'''))

cells.append(nb.v4.new_code_cell('''edf = store.entries_to_pandas()[[\"id\", \"date\"]]
m = adf.merge(edf, left_on=\"entry_id\", right_on=\"id\")
m[\"date\"] = pd.to_datetime(m[\"date\"])
m = m.explode(\"topics\")
top_topics = m[\"topics\"].value_counts().head(8).index.tolist()
sub = m[m[\"topics\"].isin(top_topics)]
freq = sub.groupby([sub[\"date\"].dt.to_period(\"M\"), \"topics\"]).size().reset_index(name=\"count\")
freq[\"date\"] = freq[\"date\"].dt.to_timestamp()
fig = px.line(freq, x=\"date\", y=\"count\", color=\"topics\", title=\"Top topics over time (monthly)\")
fig.show()'''))

nbnode = nb.v4.new_notebook()
nbnode[\"cells\"] = cells
Path(\"notebooks/03_topics.ipynb\").write_text(nb.writes(nbnode))
print(\"ok\")
"
```

Expected: prints `ok`.

- [ ] **Step 2: Execute notebook (resumable; can be run incrementally)**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/03_topics.ipynb --output 03_topics.ipynb --ExecutePreprocessor.timeout=36000
```

Expected: completes without error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_topics.ipynb
git commit -m "Add topics notebook: batch run + top-topic and over-time graphs"
```

---

## Task 16: Notebook 04 — ad-hoc Q&A

**Files:**
- Create: `notebooks/04_ad_hoc_qa.ipynb`

- [ ] **Step 1: Write notebook generator script**

Run:
```bash
uv run python -c "
import nbformat as nb
from pathlib import Path

cells = []
cells.append(nb.v4.new_markdown_cell('''# 04 — Ad-hoc Q&A

Ask free-form questions over the journal. Uses vector retrieval to find relevant entries, then LLM to answer with citations.'''))

cells.append(nb.v4.new_code_cell('''import sys
sys.path.insert(0, \"../src\")

from IPython.display import Markdown, display
from journal.config import LANCE_ROOT, LLM_MODEL
from journal.embed import OllamaEmbedder
from journal.llm import OllamaLLM
from journal.query import answer_question
from journal.store import Store

store = Store(LANCE_ROOT)

def ask(question, k=10):
    result = answer_question(
        store=store,
        embedder=OllamaEmbedder(),
        llm=OllamaLLM(),
        question=question,
        k=k,
    )
    display(Markdown(f\"### Q: {question}\\n\\n{result.answer}\"))
    display(Markdown(f\"**Sources ({len(result.sources)}):**\"))
    for i, row in result.sources.iterrows():
        display(Markdown(f\"- [{i}] {row['date']} — {row['text'][:200]}...\"))'''))

cells.append(nb.v4.new_code_cell('''ask(\"When did I feel most grateful, and what about?\")'''))

cells.append(nb.v4.new_code_cell('''ask(\"What were the recurring sources of stress in 2021?\")'''))

cells.append(nb.v4.new_code_cell('''ask(\"How did my relationship with my parents come up over the years?\")'''))

nbnode = nb.v4.new_notebook()
nbnode[\"cells\"] = cells
Path(\"notebooks/04_ad_hoc_qa.ipynb\").write_text(nb.writes(nbnode))
print(\"ok\")
"
```

Expected: prints `ok`.

- [ ] **Step 2: Execute notebook**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/04_ad_hoc_qa.ipynb --output 04_ad_hoc_qa.ipynb --ExecutePreprocessor.timeout=600
```

Expected: completes without error; the three sample questions produce non-empty answers with sources.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_ad_hoc_qa.ipynb
git commit -m "Add ad-hoc Q&A notebook with sample questions"
```

---

## Task 17: Notebook 05 — combined dashboard

**Files:**
- Create: `notebooks/05_dashboard.ipynb`

- [ ] **Step 1: Write notebook generator script**

Run:
```bash
uv run python -c "
import nbformat as nb
from pathlib import Path

cells = []
cells.append(nb.v4.new_markdown_cell('''# 05 — Dashboard

Combined view. Assumes notebooks 02 and 03 have already been run so the analyses table has sentiment + topics for the corpus.'''))

cells.append(nb.v4.new_code_cell('''import sys, json
sys.path.insert(0, \"../src\")

import pandas as pd
import plotly.express as px

from journal.config import LANCE_ROOT
from journal.store import Store

store = Store(LANCE_ROOT)
e = store.entries_to_pandas()
a = store.analyses_to_pandas()
print(f\"{len(e):,} entries, {len(a):,} analyses\")'''))

cells.append(nb.v4.new_code_cell('''s = a[(a[\"question_id\"]==\"sentiment\") & a[\"parsed_ok\"]].copy()
s[\"level\"] = s[\"result_json\"].apply(lambda x: json.loads(x)[\"level\"])
s[\"score\"] = s[\"result_json\"].apply(lambda x: json.loads(x)[\"score\"])
df = s.merge(e[[\"id\",\"date\",\"day_of_week\",\"time_of_day\"]], left_on=\"entry_id\", right_on=\"id\")
df[\"date\"] = pd.to_datetime(df[\"date\"])

order = [\"extreme_negative\",\"negative\",\"neutral\",\"positive\",\"extreme_positive\"]
day_order = [\"Mon\",\"Tue\",\"Wed\",\"Thu\",\"Fri\",\"Sat\",\"Sun\"]
tod_order = [\"morning\",\"afternoon\",\"evening\",\"night\"]

ct = df.groupby([\"day_of_week\",\"level\"]).size().reset_index(name=\"n\")
ct[\"day_of_week\"] = pd.Categorical(ct[\"day_of_week\"], categories=day_order, ordered=True)
px.bar(ct, x=\"day_of_week\", y=\"n\", color=\"level\", category_orders={\"level\": order},
       title=\"Sentiment by day of week\").show()

ct = df.groupby([\"time_of_day\",\"level\"]).size().reset_index(name=\"n\")
ct[\"time_of_day\"] = pd.Categorical(ct[\"time_of_day\"], categories=tod_order, ordered=True)
px.bar(ct, x=\"time_of_day\", y=\"n\", color=\"level\", category_orders={\"level\": order},
       title=\"Sentiment by time of day\").show()

wk = df.set_index(\"date\")[\"score\"].resample(\"W\").mean().reset_index()
px.line(wk, x=\"date\", y=\"score\", title=\"Weekly average sentiment score\").show()'''))

cells.append(nb.v4.new_code_cell('''t = a[(a[\"question_id\"]==\"topics\") & a[\"parsed_ok\"]].copy()
t[\"topics\"] = t[\"result_json\"].apply(lambda x: json.loads(x)[\"topics\"])
exploded = t.explode(\"topics\")
top = exploded[\"topics\"].value_counts().head(25).reset_index()
top.columns = [\"topic\",\"n\"]
fig = px.bar(top, x=\"n\", y=\"topic\", orientation=\"h\", title=\"Top 25 topics\")
fig.update_layout(yaxis={\"categoryorder\":\"total ascending\"})
fig.show()'''))

cells.append(nb.v4.new_code_cell('''activity = e.copy()
activity[\"date\"] = pd.to_datetime(activity[\"date\"])
per_week = activity.set_index(\"date\").resample(\"W\").size().reset_index(name=\"entries\")
px.line(per_week, x=\"date\", y=\"entries\", title=\"Entries per week\").show()'''))

nbnode = nb.v4.new_notebook()
nbnode[\"cells\"] = cells
Path(\"notebooks/05_dashboard.ipynb\").write_text(nb.writes(nbnode))
print(\"ok\")
"
```

Expected: prints `ok`.

- [ ] **Step 2: Execute notebook**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/05_dashboard.ipynb --output 05_dashboard.ipynb --ExecutePreprocessor.timeout=600
```

Expected: completes without error; all plots render.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_dashboard.ipynb
git commit -m "Add combined dashboard notebook"
```

---

## Task 18: End-to-end verification on full corpus

**Files:** (none — verification only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Verify ingest completeness**

Run:
```bash
uv run python -c "
from journal.config import ENTRIES_DIR, LANCE_ROOT
from journal.store import Store
import os
file_count = len([f for f in os.listdir(ENTRIES_DIR) if f.endswith('.html')])
df = Store(LANCE_ROOT).entries_to_pandas()
print(f'HTML files: {file_count}')
print(f'entries stored: {len(df):,}')
print(f'distinct file_paths: {df[\"file_path\"].nunique()}')
"
```
Expected: distinct `file_path` count is within ~1 of the file_count (some files may fail to parse — those are logged in `state/ingest_errors.json`). Investigate any large gap.

- [ ] **Step 3: Run a 200-entry sentiment smoke slice (if not already done in Task 14)**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from journal.config import LANCE_ROOT, LLM_MODEL
from journal.analyze import BatchAnalyzer, QuestionRegistry
from journal.llm import OllamaLLM
from journal.questions.sentiment import SENTIMENT
from journal.store import Store
store = Store(LANCE_ROOT)
reg = QuestionRegistry(); reg.register(SENTIMENT)
print(BatchAnalyzer(store=store, llm=OllamaLLM(), registry=reg, model=LLM_MODEL, max_workers=4).run('sentiment'))
"
```

Expected: after a few minutes, a report with `failed=0` (or a small number). If `failed` is high, inspect `result_json` for `_error` / `_invalid` rows and adjust the prompt in `src/journal/questions/sentiment.py`.

- [ ] **Step 4: Verify a sample of sentiment results look sensible**

Run:
```bash
uv run python -c "
import json
from journal.config import LANCE_ROOT
from journal.store import Store
store = Store(LANCE_ROOT)
e = store.entries_to_pandas()
a = store.analyses_to_pandas()
s = a[(a['question_id']=='sentiment') & a['parsed_ok']].head(20)
for _, r in s.iterrows():
    row = e[e['id']==r['entry_id']].iloc[0]
    print(r['created_at'], '|', json.loads(r['result_json']), '|', row['text'][:80].replace(chr(10), ' '))
"
```

Expected: sentiment labels generally match the entry tone. If many are obviously wrong, revisit `src/journal/questions/sentiment.py` prompt and re-run (delete the analyses rows for the question first to force re-analysis).

- [ ] **Step 5: Commit (no code changes; tag the verification)**

Run:
```bash
git tag v0.1-e2e-verified
```

Expected: tag created.

---

## Self-review notes

**Spec coverage check:**
- §1 Goals — covered by Tasks 1–18.
- §2 Non-goals — respected (no image extraction, no UI beyond notebooks).
- §3 Source data — Task 3 parser handles timestamped entries, file-name dates, `_(1)` suffixes (split on `_`).
- §4 Tech stack — pinned in `pyproject.toml` (Task 1).
- §5 Architecture — Tasks 7, 11, 12 implement the three paths.
- §6 Data model — Task 4 implements both tables exactly per spec; `parsed_ok` boolean replaces the spec's nullable struct column (simpler, equivalent information).
- §7 Built-in questions — Tasks 9, 10.
- §8 Time-of-day buckets — Task 2 `time_of_day()`.
- §9 Standardized output / error handling — `complete_json` retries + parse-failure envelope in Task 11's `_one()`.
- §10 Project layout — File Map above matches.
- §11 Ingestion / change detection — Task 7 `scan_and_ingest` + state file.
- §12 Testing — Tasks 3, 4, 5, 7, 8, 9, 10, 11, 12 include TDD.
- §13 Performance — Task 14 flags the long run; resumable; `max_workers=4` default.

**Type/name consistency:**
- `Entry`, `parse_file` — defined Task 3, used Task 7.
- `Store` methods (`add_entries`, `delete_by_file`, `entries_to_pandas`, `add_analyses`, `analyses_to_pandas`, `entries_missing_analysis`, `search_entries`) — defined Task 4, used Tasks 7, 11, 12.
- `Embedder.embed` / `embed_batch` — defined Task 5, used Tasks 7, 12.
- `LLM.complete_json` / `complete` — defined Task 5, extended Task 12.
- `Question`, `QuestionRegistry`, `BatchAnalyzer`, `AnalyzeReport` — defined Tasks 8/11.
- `SENTIMENT`, `TOPICS`, `default_registry` — defined Tasks 9/10.
- `answer_question`, `QueryAnswer` — defined Task 12.

**Placeholder scan:** none.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-journal-analysis.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
