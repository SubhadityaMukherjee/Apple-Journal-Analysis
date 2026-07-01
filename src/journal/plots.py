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
