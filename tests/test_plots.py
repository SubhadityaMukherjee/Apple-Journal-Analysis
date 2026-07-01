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
