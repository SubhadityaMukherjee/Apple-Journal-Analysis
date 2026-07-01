# Multi-question batch + reusable plotting helpers

**Date:** 2026-07-01
**Status:** Approved, ready for implementation plan

## Goal

Make the batch analyzer runnable for multiple questions in one call, and lift the notebook's plotting code into importable, reusable helpers. Also fix a latent bug in `00_run_all.ipynb` where plotting cells reference `sdf` and `a` that no cell constructs.

## Background

Today `BatchAnalyzer.run(question_id)` runs one question at a time. The notebook works around this by constructing a `BatchAnalyzer` twice and calling `run("sentiment")` then `run("topics")` in separate cells. The underlying storage already supports multi-question cleanly — each row in the `analyses` table is tagged with `question_id`, and `entries_missing_analysis(question_id, model)` makes resumability per-question, not per-batch. Only the entry point is single-question.

Separately, the notebook contains ~80 lines of plotting code (sentiment distribution diverging bars, weekly score line, top topics bar, topics-over-time lines) that is a natural candidate for reuse but currently lives only in the notebook.

## Non-goals

- Interleaving questions in a single thread pool. The local Ollama server is the bottleneck, not orchestration; sequential-per-question delivers the same end-to-end runtime with simpler resumability semantics.
- End-to-end plotting helpers that consume raw `(analyses_df, entries_df)` and do their own join/parse. The helpers take tidy DataFrames; prep stays in the notebook so the helpers generalize across future questions.
- New analyses, new chart types, or visual redesign. The plots produced will look identical to today.

## Design

### 1. `BatchAnalyzer.run_many(question_ids)`

In `src/journal/analyze.py`:

```python
@dataclass
class AnalyzeReportSet:
    reports: list[AnalyzeReport]
```

```python
def run_many(self, question_ids: list[str]) -> AnalyzeReportSet:
    for qid in question_ids:
        self.registry.get(qid)  # raises KeyError before any work if missing
    return AnalyzeReportSet(reports=[self.run(qid) for qid in question_ids])
```

- Validates all ids against the registry up front, then calls `run()` once per id in order. A missing id raises `KeyError` before any work begins — no partial-batch surprises.
- `run()` is unchanged. Existing tests for `run()` stay green.
- Resumability, partial-progress-on-abort, and per-question parallelism all carry through unchanged because they live inside `run()`.

### 2. `src/journal/plots.py` — new module

Four functions plus two exported constants. All return `plotly.graph_objects.Figure`; none call `.show()`. All operate on **tidy DataFrames** the caller constructs.

**Constants:**

```python
SENTIMENT_LEVELS = ["extreme_negative", "negative", "neutral", "positive", "extreme_positive"]
SENTIMENT_COLORS = {
    "extreme_negative": "#8B0000",
    "negative": "#E57373",
    "neutral": "#BDBDBD",
    "positive": "#81C784",
    "extreme_positive": "#1B5E20",
}
```

**Functions:**

```python
def diverging_sentiment_bar(
    df, group_col, group_order, title
) -> go.Figure
```
Diverging stacked bar: negatives stacked left of zero, positives right of zero, neutral split half-and-half across the center. One trace per level in `SENTIMENT_LEVELS`, colored from `SENTIMENT_COLORS`. Requires `df` to have columns `group_col` and `level` (where `level` ∈ `SENTIMENT_LEVELS`). Behavior is identical to the existing notebook function of the same name — this is a lift-and-move, not a redesign.

```python
def score_over_time(
    df, date_col="date", value_col="score", freq="W", title="Score over time"
) -> go.Figure
```
Resamples `value_col` over `date_col` at pandas frequency `freq` using `.mean()`, plots as a line with markers. Generalizes the weekly sentiment score line.

```python
def top_n_bar(items, n=30, title="Top items") -> go.Figure
```
Takes a 1D iterable (typically an exploded list column, e.g., `exploded["topics"]`). Computes `value_counts().head(n)`, plots as an ascending horizontal bar.

```python
def top_n_over_time(
    df, item_col, date_col, n=8, freq="M", title="Top items over time"
) -> go.Figure
```
Expects `df` already exploded to one row per item. Picks the top-N most frequent items, filters to those, groups by `(date.to_period(freq), item_col)`, plots count lines colored by item. Generalizes the topics-over-time chart.

**Dependencies:** `pandas` and `plotly` are already declared in `pyproject.toml`; no new deps.

### 3. Notebook rewrite (`notebooks/00_run_all.ipynb`)

Cell-level changes (cell indices refer to current notebook):

- **Merge cells 3–6** (markdown "Run sentiment batch" + code; markdown "Run topics batch" + code) into one section: build `default_registry()`, construct `BatchAnalyzer` once, call `analyzer.run_many(["sentiment", "topics"])`, print each report.
- **Add a new cell** after the batch that loads analyses and builds the plot-ready DataFrames:
  ```python
  a = store.analyses_to_pandas()

  sdf = (a[(a["question_id"] == "sentiment") & a["parsed_ok"]]
         .assign(
             level=lambda d: d["result_json"].apply(lambda x: json.loads(x)["level"]),
             score=lambda d: d["result_json"].apply(lambda x: json.loads(x)["score"]),
         )
         .merge(df[["id", "date", "day_of_week", "time_of_day"]],
                left_on="entry_id", right_on="id"))
  ```
  This is the cell that was silently missing — the notebook NameErrors today without it.
- **Cell 8** (sentiment plots): drop the inline `diverging_sentiment_bar` definition and the redefined `order`/`colors` constants; import from `journal.plots`. Calls become one-liners.
- **Cell 10** (topics plots): keep the `result_json` parsing and `merge` with entries for the date, but swap the plotting calls for `top_n_bar(exploded["topics"], n=30, ...)` and `top_n_over_time(m_exploded, item_col="topics", date_col="date", n=8, freq="M", ...)`.

The visual output of each plot is unchanged from today.

### 4. Tests

- `tests/test_analyze.py` — add:
  - `test_run_many_runs_all_questions`: seed store, register both sentiment and a trivial dummy question, run `run_many`, assert two reports returned with the right `question_id`s and the right processed counts.
  - `test_run_many_resumes_per_question`: run `run_many(["sentiment", "topics"])` twice; second call returns two reports each with `processed == 0`. Verifies resumability is still keyed per question, not per batch.
  - `test_run_many_rejects_unknown_id_before_any_work`: register only `sentiment`, call `run_many(["sentiment", "bogus"])`, assert it raises `KeyError` and that no analysis rows were written for `sentiment` either (proves validation happens up front).
- `tests/test_plots.py` — new file, smoke tests only:
  - `diverging_sentiment_bar`: tiny DataFrame with two groups and known level counts → assert trace count is 5 and the figure's x-axis `categoryarray` matches `group_order`.
  - `score_over_time`: two rows, two dates → assert one trace, x length matches resample output.
  - `top_n_bar`: list with known duplicates → assert bar count equals `n` (or unique count if smaller), y-axis categoryorder is `"total ascending"`.
  - `top_n_over_time`: exploded DataFrame with two items across two months → assert one line trace per top item.

No image / pixel comparison. The contract is "right shape of Figure out, given tidy data in."

## Risks

- **Notebook outputs are stale.** The committed notebook still shows old outputs from the two-`run()`-call era. After rewriting, the executed cells' outputs will be out of sync until someone re-runs end-to-end (~1.5hr on the full corpus). The plan should clear stale outputs from rewritten cells rather than re-executing — the user will re-run when convenient.
- **`run_many` validates ids up front** via `registry.get()` before the loop. A missing id raises `KeyError` before any work begins; questions listed before it never run. Same contract as `run()` on a missing id.

## Out of scope (explicit)

- Interleaved execution in a shared thread pool.
- End-to-end plotting helpers that absorb the notebook prep.
- Re-executing the notebook against the full corpus as part of this change.
- Any change to `run()`'s signature or the store/analyses schema.
