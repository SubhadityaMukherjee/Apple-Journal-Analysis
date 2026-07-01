# Multi-question batch + reusable plotting helpers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `BatchAnalyzer.run_many(...)` so the notebook runs sentiment + topics in one call, lift the notebook's plotting code into importable helpers in `src/journal/plots.py`, and fix the latent `sdf`/`a` NameError in `00_run_all.ipynb`.

**Architecture:** `run_many` is a thin sequential loop over the existing `run()`, with up-front registry validation. `plots.py` exposes four plot-only helpers that consume tidy DataFrames and return `plotly.graph_objects.Figure` — the notebook keeps the per-question prep (parse `result_json`, merge entries) and swaps inline plot code for one-liner helper calls.

**Tech Stack:** Python 3.12, pandas, plotly, pytest, nbformat. All already declared in `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-07-01-multi-question-batch-and-plot-helpers-design.md`

---

## File map

- **Modify** `src/journal/analyze.py` — add `AnalyzeReportSet` dataclass and `BatchAnalyzer.run_many` method.
- **Modify** `tests/test_analyze.py` — add three tests for `run_many`.
- **Create** `src/journal/plots.py` — module constants `SENTIMENT_LEVELS` and `SENTIMENT_COLORS`, plus four functions: `diverging_sentiment_bar`, `score_over_time`, `top_n_bar`, `top_n_over_time`.
- **Create** `tests/test_plots.py` — five smoke tests for the helpers and constants.
- **Modify** `notebooks/00_run_all.ipynb` — merge the two batch cells into one `run_many` call, add a missing "load analyses into DataFrames" cell, swap the two plotting cells to use the helpers.

---

## Task 1: Add `AnalyzeReportSet` and `BatchAnalyzer.run_many`

**Files:**
- Modify: `src/journal/analyze.py`
- Modify: `tests/test_analyze.py`

- [ ] **Step 1: Write the three failing tests**

Append to `tests/test_analyze.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_analyze.py -k run_many -v`
Expected: 3 FAIL with `AttributeError: 'BatchAnalyzer' object has no attribute 'run_many'` (and `AnalyzeReportSet` not found).

- [ ] **Step 3: Add `AnalyzeReportSet` and `run_many` to `analyze.py`**

In `src/journal/analyze.py`, add a new dataclass immediately after the `AnalyzeReport` definition (around line 42):

```python
@dataclass
class AnalyzeReportSet:
    reports: list[AnalyzeReport] = field(default_factory=list)
```

Then add the method to the `BatchAnalyzer` class, immediately after the existing `run` method (around line 105):

```python
    def run_many(self, question_ids: list[str]) -> AnalyzeReportSet:
        for qid in question_ids:
            self.registry.get(qid)  # raises KeyError before any work if missing
        return AnalyzeReportSet(reports=[self.run(qid) for qid in question_ids])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_analyze.py -v`
Expected: all tests PASS, including the three new `run_many` tests and all existing `BatchAnalyzer`/registry/question tests.

- [ ] **Step 5: Commit**

```bash
git add src/journal/analyze.py tests/test_analyze.py
git commit -m "Add BatchAnalyzer.run_many for multi-question batch runs"
```

---

## Task 2: Create `src/journal/plots.py` with four reusable helpers

**Files:**
- Create: `src/journal/plots.py`
- Create: `tests/test_plots.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plots.py`:

```python
import pandas as pd

from journal.plots import (
    SENTIMENT_COLORS,
    SENTIMENT_LEVELS,
    diverging_sentiment_bar,
    score_over_time,
    top_n_bar,
    top_n_over_time,
)


def test_constants_exported():
    assert SENTIMENT_LEVELS == [
        "extreme_negative", "negative", "neutral", "positive", "extreme_positive",
    ]
    assert SENTIMENT_COLORS["neutral"] == "#BDBDBD"


def test_diverging_sentiment_bar_has_one_trace_per_level():
    df = pd.DataFrame({
        "group": ["g1", "g1", "g2", "g2"],
        "level": ["negative", "positive", "neutral", "extreme_positive"],
    })
    fig = diverging_sentiment_bar(df, "group", ["g1", "g2"], "title")
    # one Bar trace per sentiment level
    assert len(fig.data) == len(SENTIMENT_LEVELS)
    # x axis uses group_order
    assert list(fig.layout.xaxis.categoryarray) == ["g1", "g2"]


def test_score_over_time_returns_single_line():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-02-01"]),
        "score": [1.0, 3.0, 5.0],
    })
    fig = score_over_time(df, title="t")
    assert len(fig.data) == 1  # one line trace
    assert len(fig.data[0].x) >= 1


def test_top_n_bar_caps_at_n():
    items = ["a", "a", "a", "a", "b", "b", "b", "c", "c", "d", "e", "f"]
    fig = top_n_bar(items, n=3, title="t")
    assert len(fig.data) == 1  # single Bar trace
    assert len(fig.data[0].y) == 3  # only 3 items shown
    assert fig.layout.yaxis.categoryorder == "total ascending"


def test_top_n_over_time_one_trace_per_top_item():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-05", "2024-01-05", "2024-01-05", "2024-02-05",
            "2024-02-05", "2024-02-05", "2024-02-05",
            "2024-01-05",
        ]),
        "item": ["x", "x", "x", "x",
                 "y", "y", "y",
                 "z"],
    })
    fig = top_n_over_time(df, item_col="item", date_col="date", n=2, title="t")
    # top 2 items (x=4, y=3) → one line trace each
    assert len(fig.data) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'journal.plots'`.

- [ ] **Step 3: Write `src/journal/plots.py`**

Create `src/journal/plots.py`:

```python
"""Reusable Plotly figures built from tidy DataFrames.

Each helper takes data the caller has already prepared (filtered to the
relevant question, result_json parsed, entries merged for date columns) and
returns a plotly Figure. The caller decides when to call .show().
"""
from collections.abc import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SENTIMENT_LEVELS = [
    "extreme_negative",
    "negative",
    "neutral",
    "positive",
    "extreme_positive",
]

SENTIMENT_COLORS = {
    "extreme_negative": "#8B0000",
    "negative": "#E57373",
    "neutral": "#BDBDBD",
    "positive": "#81C784",
    "extreme_positive": "#1B5E20",
}


def diverging_sentiment_bar(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title: str,
) -> go.Figure:
    """Diverging stacked bar: negatives stacked left of zero, positives right,
    neutral split half-and-half across the center. Requires `df` to have a
    `level` column whose values are in SENTIMENT_LEVELS."""
    ct = df.groupby([group_col, "level"]).size().reset_index(name="count")
    piv = (
        ct.pivot(index=group_col, columns="level", values="count")
          .reindex(index=group_order, columns=SENTIMENT_LEVELS, fill_value=0)
    )

    half_neutral = piv["neutral"] / 2

    bases = {
        "extreme_negative": -(half_neutral + piv["negative"] + piv["extreme_negative"]),
        "negative": -(half_neutral + piv["negative"]),
        "neutral": -half_neutral,
        "positive": half_neutral,
        "extreme_positive": half_neutral + piv["positive"],
    }
    counts = {lvl: piv[lvl] for lvl in SENTIMENT_LEVELS}

    fig = go.Figure()
    for level in SENTIMENT_LEVELS:
        c = counts[level]
        labels = c.apply(lambda v: str(v) if v > 0 else "")
        fig.add_trace(go.Bar(
            x=group_order, y=c, base=bases[level],
            name=level, marker_color=SENTIMENT_COLORS[level],
            customdata=c,
            text=labels, textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=11),
            hovertemplate="%{x}<br>" + level + ": %{customdata}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay", title=title,
        xaxis=dict(categoryorder="array", categoryarray=group_order),
        yaxis_title="count (negative ← → positive)",
        uniformtext_minsize=9, uniformtext_mode="hide",
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    return fig


def score_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "score",
    freq: str = "W",
    title: str = "Score over time",
) -> go.Figure:
    """Resample `value_col` over `date_col` at pandas frequency `freq` using
    .mean(), plot as a line with markers."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    series = d.set_index(date_col)[value_col].resample(freq).mean().reset_index()
    return px.line(series, x=date_col, y=value_col, title=title, markers=True)


def top_n_bar(
    items: Iterable,
    n: int = 30,
    title: str = "Top items",
) -> go.Figure:
    """Horizontal bar of the top-N most frequent items in `items` (typically
    an exploded list column). Ascending order so the largest is on top."""
    counts = pd.Series(items).value_counts().head(n).reset_index()
    counts.columns = ["item", "count"]
    fig = px.bar(counts, x="count", y="item", orientation="h", title=title)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def top_n_over_time(
    df: pd.DataFrame,
    item_col: str,
    date_col: str,
    n: int = 8,
    freq: str = "M",
    title: str = "Top items over time",
) -> go.Figure:
    """For a DataFrame already exploded to one row per item, pick the top-N
    most frequent items and plot their per-period counts as colored lines."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    top_items = d[item_col].value_counts().head(n).index.tolist()
    sub = d[d[item_col].isin(top_items)]
    freq_df = (
        sub.groupby([sub[date_col].dt.to_period(freq), item_col])
           .size()
           .reset_index(name="count")
    )
    freq_df[date_col] = freq_df[date_col].dt.to_timestamp()
    return px.line(freq_df, x=date_col, y="count", color=item_col, title=title)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `.venv/bin/pytest -v`
Expected: all tests pass, including the new ones from Task 1.

- [ ] **Step 6: Commit**

```bash
git add src/journal/plots.py tests/test_plots.py
git commit -m "Add reusable plot helpers (diverging bar, time series, top-N)"
```

---

## Task 3: Rewrite `00_run_all.ipynb` to use `run_many` and the helpers

**Files:**
- Modify: `notebooks/00_run_all.ipynb`

The notebook has no unit tests. We do an atomic content-based rewrite via a one-shot `nbformat` script, then verify by re-parsing and spot-checking. The script is deleted at the end of the task — it is not part of the codebase.

- [ ] **Step 1: Write the one-shot rewrite script**

Create `scripts/rewrite_run_all_nb.py`:

```python
"""One-shot rewrite of notebooks/00_run_all.ipynb.

Run once, then delete. Edits are content-based (match on the first line of
each cell's source) so they survive minor reformatting.
"""
from pathlib import Path

import nbformat

NB_PATH = Path("notebooks/00_run_all.ipynb")


def md(src: str) -> dict:
    return nbformat.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbformat.v4.new_code_cell(src)


nb = nbformat.read(NB_PATH, as_version=4)

new_cells: list = []
i = 0
while i < len(nb["cells"]):
    c = nb["cells"][i]
    src = "".join(c["source"]).strip()

    # 1) Merge the "Run sentiment batch" + "Run topics batch" sections into
    #    one "Run batch analyses" section that uses run_many.
    if src.startswith("### 2. Run sentiment batch"):
        new_cells.append(md(
            "## 2. Run batch analyses (sentiment + topics)\n\n"
            "Resumable: entries already analyzed for `LLM_MODEL` are skipped "
            "automatically, per question."
        ))
        new_cells.append(code(
            "import json\n"
            "\n"
            "from journal.analyze import BatchAnalyzer\n"
            "from journal.config import LLM_MODEL\n"
            "from journal.llm import OllamaLLM\n"
            "from journal.questions import default_registry\n"
            "\n"
            "analyzer = BatchAnalyzer(\n"
            "    store=store, llm=OllamaLLM(), registry=default_registry(),\n"
            "    model=LLM_MODEL, max_workers=6,\n"
            ")\n"
            "result = analyzer.run_many([\"sentiment\", \"topics\"])\n"
            "for report in result.reports:\n"
            "    print(report)"
        ))
        # Skip the four original cells: this markdown, sentiment code,
        # the "## 3. Run topics batch" markdown, and the topics code.
        i += 4
        continue

    # 2) Insert the missing "Load analyses into DataFrames" section before
    #    "## 4. Sentiment plots".
    if src.startswith("## 4. Sentiment plots"):
        new_cells.append(md("## 3. Load analyses into DataFrames"))
        new_cells.append(code(
            "a = store.analyses_to_pandas()\n"
            "\n"
            "sdf = (\n"
            "    a[(a[\"question_id\"] == \"sentiment\") & a[\"parsed_ok\"]]\n"
            "    .assign(\n"
            "        level=lambda d: d[\"result_json\"].apply(lambda x: json.loads(x)[\"level\"]),\n"
            "        score=lambda d: d[\"result_json\"].apply(lambda x: json.loads(x)[\"score\"]),\n"
            "    )\n"
            "    .merge(df[[\"id\", \"date\", \"day_of_week\", \"time_of_day\"]],\n"
            "           left_on=\"entry_id\", right_on=\"id\")\n"
            ")\n"
            "\n"
            "topics_df = (\n"
            "    a[(a[\"question_id\"] == \"topics\") & a[\"parsed_ok\"]]\n"
            "    .assign(topics=lambda d: d[\"result_json\"].apply(lambda x: json.loads(x)[\"topics\"]))\n"
            "    .merge(df[[\"id\", \"date\"]], left_on=\"entry_id\", right_on=\"id\")\n"
            ")\n"
            "topics_df[\"date\"] = pd.to_datetime(topics_df[\"date\"])\n"
            "topics_exploded = topics_df.explode(\"topics\")"
        ))
        # Keep the original "## 4. Sentiment plots" markdown too.
        new_cells.append(c)
        i += 1
        continue

    # 3) Replace the inline sentiment plot code (which defined its own
    #    diverging_sentiment_bar) with helper-based one-liners.
    if src.startswith("import plotly.graph_objects as go"):
        new_cells.append(code(
            "from journal.plots import diverging_sentiment_bar, score_over_time\n"
            "\n"
            "day_order = [\"Mon\", \"Tue\", \"Wed\", \"Thu\", \"Fri\", \"Sat\", \"Sun\"]\n"
            "tod_order = [\"morning\", \"afternoon\", \"evening\", \"night\"]\n"
            "\n"
            "diverging_sentiment_bar(sdf, \"day_of_week\", day_order, \"Sentiment distribution by day of week\").show()\n"
            "diverging_sentiment_bar(sdf, \"time_of_day\", tod_order, \"Sentiment distribution by time of day\").show()\n"
            "score_over_time(sdf, title=\"Average sentiment score over time (weekly)\").show()"
        ))
        i += 1
        continue

    # 4) Replace the inline topics plot code with helper-based one-liners.
    if src.startswith("t = a["):
        new_cells.append(code(
            "from journal.plots import top_n_bar, top_n_over_time\n"
            "\n"
            "top_n_bar(topics_exploded[\"topics\"], n=30, title=\"Top 30 topics across all entries\").show()\n"
            "top_n_over_time(\n"
            "    topics_exploded, item_col=\"topics\", date_col=\"date\", n=8, freq=\"M\",\n"
            "    title=\"Top topics over time (monthly)\",\n"
            ").show()"
        ))
        i += 1
        continue

    # 5) The ask() helper cell calls display(...) but only imported Markdown.
    #    The old cell 4 used to import display for it; the new cell 4 doesn't,
    #    so add display to this cell's IPython import.
    if src.startswith("from IPython.display import Markdown"):
        new_cells.append(code(
            "from IPython.display import Markdown, display\n"
            "from journal.query import answer_question\n"
            "\n"
            "def ask(question, k=10):\n"
            "    result = answer_question(\n"
            "        store=store,\n"
            "        embedder=OllamaEmbedder(),\n"
            "        llm=OllamaLLM(),\n"
            "        question=question,\n"
            "        k=k,\n"
            "    )\n"
            "    display(Markdown(f\"### Q: {question}\\n\\n{result.answer}\"))\n"
            "    display(Markdown(f\"**Sources ({len(result.sources)}):**\"))\n"
            "    for i, row in result.sources.iterrows():\n"
            "        snippet = row[\"text\"][:200].replace(\"\\n\", \" \")\n"
            "        display(Markdown(f\"- [{i}] {row['date']} — {snippet}...\"))"
        ))
        i += 1
        continue

    # Default: keep the cell verbatim.
    new_cells.append(c)
    i += 1

nb["cells"] = new_cells
nbformat.write(nb, NB_PATH)
print(f"Rewrote {NB_PATH}: {len(new_cells)} cells")
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python scripts/rewrite_run_all_nb.py`
Expected: prints `Rewrote notebooks/00_run_all.ipynb: 16 cells` (original 16; the batch merge replaces 4 cells with 2 for net -2, the DataFrame-section insert replaces 1 cell with 3 for net +2, the two plot-cell replacements are 1:1).

- [ ] **Step 3: Verify the rewrite by re-parsing and dumping the new outline**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('notebooks/00_run_all.ipynb', as_version=4)
for i, c in enumerate(nb['cells']):
    first = ''.join(c['source']).split('\n')[0][:90]
    print(f'{i:2d} [{c[\"cell_type\"][:2]}] {first}')
"
```

Expected output:

```
 0 [md] # 00 — Run All
 1 [md] ## 1. Setup & ingest
 2 [co] import sys
 3 [md] ## 2. Run batch analyses (sentiment + topics)
 4 [co] import json
 5 [md] ## 3. Load analyses into DataFrames
 6 [co] a = store.analyses_to_pandas()
 7 [md] ## 4. Sentiment plots
 8 [co] from journal.plots import diverging_sentiment_bar, score_over_time
 9 [md] ## 5. Topics plots
10 [co] from journal.plots import top_n_bar, top_n_over_time
11 [md] ## 6. Ad-hoc Q&A
12 [co] from IPython.display import Markdown, display
13 [co] ask("When did I feel most grateful, and what about?")
14 [co] ask("What were the recurring sources of stress in 2021?")
15 [co] ask("How did my relationship with my parents come up over the years?")
```

If the output does not match (especially the section headers and the four replaced code cells), do not proceed — investigate which content-based match failed.

- [ ] **Step 4: Spot-check that rewritten cells have no outputs**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('notebooks/00_run_all.ipynb', as_version=4)
for i in [4, 6, 8, 10]:
    c = nb['cells'][i]
    print(f'cell {i}: outputs={len(c.get(\"outputs\", []))}, execution_count={c.get(\"execution_count\")}')
"
```

Expected: each line shows `outputs=0, execution_count=None`. (`nbformat.v4.new_code_cell` produces cells with empty outputs and no execution count — this is what we want, since these cells will be re-run by the user against the live store.)

- [ ] **Step 5: Delete the throwaway script**

Run: `rm scripts/rewrite_run_all_nb.py`
If `scripts/` is now empty, also: `rmdir scripts`

- [ ] **Step 6: Run the full test suite to confirm nothing else broke**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS (notebooks are not part of the pytest suite, so this is a sanity check on Tasks 1 and 2).

- [ ] **Step 7: Commit**

```bash
git add notebooks/00_run_all.ipynb
git commit -m "Rewrite 00_run_all to use run_many and plot helpers

Merges the two batch cells into a single run_many(['sentiment','topics'])
call, adds the missing analyses-to-DataFrame cell that previously caused
NameErrors on sdf/a, and swaps inline plot code for calls into
journal.plots."
```

---

## Self-review notes

- **Spec coverage:** spec §1 (`run_many`) → Task 1. spec §2 (`plots.py` with the four named helpers and two constants) → Task 2. spec §3 (notebook rewrite covering the merged batch cell, the inserted DataFrame cell, and the two swapped plotting cells) → Task 3. spec §4 (tests) → Tasks 1 and 2. No spec section is left without a task.
- **Type consistency:** `AnalyzeReportSet.reports: list[AnalyzeReport]` matches the access pattern `for r in result.reports` used in the notebook rewrite. `run_many(question_ids: list[str])` matches the call signature in the notebook. Helper signatures in `plots.py` match both the tests and the notebook calls.
- **Cross-cell import dependencies (notebook):** the original cell 4 imported `from IPython.display import display` even though cell 4 itself didn't use it — cell 12's `ask()` helper relied on it. The rewrite drops the unused import from cell 4 and adds `display` to cell 12's own IPython import. Verified that `OllamaEmbedder` (cell 2), `OllamaLLM` (new cell 4), `store` (cell 2), and `answer_question` (new cell 12) are all in scope when cell 12 runs.
- **Cell count math:** original notebook has 16 cells. Batch-merge step: 4 → 2 (net -2). DataFrame-section insert: 1 → 3 (net +2). Plot replacements and the cell-12 import fix are 1:1. Final: 16 cells.
- **Risks (from spec):** stale notebook outputs handled by `new_code_cell` producing empty-output cells (verified in Step 4). `run_many` up-front validation covered by `test_run_many_rejects_unknown_id_before_any_work`.
