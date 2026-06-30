# Journal Analysis — Design Spec

**Date:** 2026-06-30
**Status:** Approved (pending spec review)
**Working dir:** `/Users/smukherjee/Documents/Projects/Personal/journal_stata`
**Source data:** `/Users/smukherjee/Documents/Archives/Backups/AppleJournalEntries/Entries`

## 1. Goals

Build a local, persistent system that:

1. Ingests Apple Journal HTML exports from a fixed directory, detects new/changed files since the last run, and extracts timestamped entries.
2. Embeds each entry with a local model and stores it in a persistent vector store that is directly queryable as a pandas DataFrame.
3. Runs batch LLM analyses over entries (sentiment, topics, and arbitrary user-defined questions) with strict, parseable JSON output.
4. Supports ad-hoc natural-language Q&A over the journal via vector retrieval + LLM.
5. Provides Jupyter dashboards for visual analysis — sentiment by day-of-week and time-of-day, topic frequency, activity, mood trends.

## 2. Non-goals (this iteration)

- Real-time / always-on ingestion. Ingestion runs on demand.
- Multi-user, networked, or hosted deployment. Single-user, local only.
- Image / audio / video asset extraction. Text only (the bodyText portion of each entry).
- A custom UI outside Jupyter. Dashboards live in notebooks.
- Cross-device sync.

## 3. Source data

- ~1,831 HTML files named `YYYY-MM-DD.html`. Some dates have multiple files suffixed `_(1).html`, `_(2).html`, …
- Each file is an Apple Journal export containing a header date (`Tuesday, 30 June 2020`) followed by multiple timestamped entries. Each entry begins with a timestamp line matching `^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), \w{3} \d+, \d{4} - \d{1,2}:\d{2} [AP]M$`.
- Some entries end with a short tag word (e.g., `Gratitude`, `Stress`) which we extract into a `category` column.
- Span: 2020-06-29 → 2026-06-22.
- Estimated total entries: ~10,000–18,000.

## 4. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| Package manager | `uv` |
| LLM (chat / analysis) | `gemma3:4b` via Ollama |
| Embedding model | `nomic-embed-text` via Ollama (768-dim) |
| Vector store | LanceDB (local, Arrow-native, persistent, `.to_pandas()` first-class) |
| Notebooks | JupyterLab |
| HTML parsing | BeautifulSoup4 |
| Plotting | Plotly (interactive) + matplotlib fallback |
| Data | pandas, pyarrow |

## 5. Architecture

Two-stage pipeline plus an ad-hoc query path:

```
[Entries dir] → parse → embed → [LanceDB: entries]
                                         │
                          batch analyzer → [LanceDB: analyses]
                                         │
                          notebooks ──── dashboard / ad-hoc Q&A
```

**Stage 1 — ingest (fast, minutes):** scan directory, diff against `state/seen_files.json`, parse new/changed HTML files, embed each entry, write to `entries` table. Idempotent.

**Stage 2 — analyze (slow, hours):** iterate registered questions, find entries lacking a stored result for `(question_id, model)`, call LLM with strict-JSON prompt, parse + store in `analyses` table. Resumable and parallelizable.

**Ad-hoc path:** embed question → top-k vector search → LLM answer over retrieved context. Does not write to `analyses`.

### Why two stages

Embedding is cheap; LLM analysis is expensive. Decoupling lets us add a new question and run it across history without re-embedding. It also lets the slow stage resume without re-paying the embedding cost.

## 6. Data model

### Table `entries` — one row per timestamped entry

| field | type | notes |
|---|---|---|
| `id` | string | deterministic hash of `file_path + ":" + entry_index` |
| `file_path` | string | absolute path of source HTML file |
| `date` | date | parsed from filename |
| `timestamp` | datetime | parsed from entry header; tz-naive local |
| `time_of_day` | string | `morning` / `afternoon` / `evening` / `night` |
| `day_of_week` | string | `Mon`…`Sun` |
| `text` | string | entry body, stripped of HTML |
| `category` | string | nullable; trailing tag word if present (e.g., `Gratitude`) |
| `embedding` | vector<float, 768> | `nomic-embed-text` |
| `ingested_at` | timestamp | wall-clock when row was written |

Primary key: `id`. Indexed for filter on `date`, `day_of_week`, `time_of_day`, `category`.

### Table `analyses` — one row per (entry, question, model)

| field | type | notes |
|---|---|---|
| `entry_id` | string | FK → `entries.id` |
| `question_id` | string | e.g., `sentiment`, `topics_v1` |
| `question_text` | string | exact prompt template applied |
| `result_json` | string | raw LLM JSON response |
| `parsed` | struct | typed fields for known question schemas (nullable on parse failure) |
| `model` | string | e.g., `gemma3:4b` |
| `created_at` | timestamp | |

Logical primary key: `(entry_id, question_id, model)`.

## 7. Built-in questions

Adding a new question = one Python file under `src/journal/questions/` exporting `{id, prompt_template, parse_fn}`. No other code changes.

### `sentiment`

Returns one of five ordinal levels plus a numeric score so averages are meaningful.

```json
{"level": "negative", "score": 2, "confidence": 0.8}
```

Scale:

| score | level | meaning |
|---|---|---|
| 1 | `extreme_negative` | anguish, crisis language, overwhelming distress |
| 2 | `negative` | frustrated, sad, upset, but contained |
| 3 | `neutral` | factual, no strong affect |
| 4 | `positive` | happy, grateful, content |
| 5 | `extreme_positive` | ecstatic, overjoyed, deep fulfillment |

### `topics`

Top three short-phrase topics discussed in the entry.

```json
{"topics": ["family conflict", "GRE stress", "sleep"]}
```

### Future candidates (not in v1)

- `mood_tags` — emotion labels (`anxious`, `grateful`, `angry`, …)
- `summary` — one-sentence summary per entry
- `people_mentioned` — named entities

## 8. Time-of-day buckets

| bucket | hours (local) |
|---|---|
| morning | 05:00–11:59 |
| afternoon | 12:00–16:59 |
| evening | 17:00–20:59 |
| night | 21:00–04:59 (wraps midnight) |

## 9. Standardized output & error handling

- LLM is prompted to return strict JSON only. We parse with `json.loads`; on failure we retry up to 3 times with an increasingly explicit prompt.
- After 3 failures, we store `result_json = raw_text` and `parsed = null`, mark the row, and continue. A later re-run can target `parsed IS NULL` rows.
- Ingestion failures (malformed HTML, unreadable file) are logged and the file is skipped; the path is recorded in `state/ingest_errors.json` so it isn't retried blindly on next run.
- Ollama unreachable → raise with a clear "run `ollama serve`" message; do not silently swallow.

## 10. Project layout

```
journal_stata/
├── pyproject.toml              # uv-managed
├── README.md
├── .gitignore
├── data/lance/                 # LanceDB root (gitignored)
├── state/seen_files.json       # ingestion checkpoint (gitignored)
├── state/ingest_errors.json    # parse failures (gitignored)
├── src/journal/
│   ├── __init__.py
│   ├── config.py               # paths, model names, buckets
│   ├── parse.py                # HTML → entries
│   ├── ingest.py               # dir scan + LanceDB writer
│   ├── embed.py                # ollama embedding client
│   ├── store.py                # LanceDB wrapper + .to_pandas()
│   ├── analyze.py              # batch LLM + question registry
│   ├── query.py                # ad-hoc retrieval + LLM Q&A
│   └── questions/
│       ├── __init__.py
│       ├── sentiment.py
│       └── topics.py
├── notebooks/
│   ├── 01_ingest_and_explore.ipynb
│   ├── 02_sentiment.ipynb
│   ├── 03_topics.ipynb
│   ├── 04_ad_hoc_qa.ipynb
│   └── dashboard.ipynb
└── tests/
    ├── conftest.py
    ├── fixtures/2020-06-29.html
    └── test_parse.py
```

Notebooks import from `src/journal/` and contain no logic — only dashboards and interactive exploration.

## 11. Ingestion / change detection

- State file: `state/seen_files.json` — map of `file_path → {mtime, size, entry_count}`.
- On each run:
  1. Walk source dir for `*.html`.
  2. For each file: if path not in state, or mtime/size changed → (re)process.
  3. Before writing new rows for a file, delete any existing rows with that `file_path` (handles re-parsing on file change).
  4. Update state.
- This makes ingestion safe to re-run; only the delta is paid for.

## 12. Testing

- `test_parse.py` — feed a saved sample HTML fixture, assert extracted entries match expected `(timestamp, text, category)` tuples.
- Roundtrip test for LanceDB: write entries, read back via `.to_pandas()`, assert row count + key fields.
- LLM JSON parse test using a mock client returning valid/invalid JSON.
- Live Ollama integration is exercised by notebooks, not the test suite (CI-free local project).

## 13. Performance expectations

| step | per-unit | total (≈15k entries) |
|---|---|---|
| HTML parse | ~1 ms | seconds |
| Embed (`nomic-embed-text`) | ~50 ms | ~12 min |
| Sentiment (`gemma3:4b`) | ~1–2 s | ~4–8 h |
| Topics (`gemma3:4b`) | ~1–2 s | ~4–8 h |

Mitigations for the slow stage:
- Resumable via `analyses` table (skip rows with existing `(entry_id, question_id, model)`).
- Parallelizable: Ollama supports concurrent requests; default to `min(8, n_cpus)` workers, configurable.
- Date-range scoping available at the API so you can validate on a slice first.

First-run plan: full ingest + embeddings end-to-end, then sentiment + topics across all entries.

## 14. Out of scope (deferred)

- Image/audio extraction from entries.
- Streaming / live updates to the source folder.
- A non-Jupyter UI.
- Topic modeling alternative to LLM (e.g., BERTopic) — LLM-derived topics are sufficient for v1.
- Cross-encoder reranking for ad-hoc retrieval.
