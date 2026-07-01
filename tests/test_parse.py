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


def test_parse_file_handles_new_format_no_timestamps():
    """Post-2024 Apple Journal exports have no per-entry timestamps.
    The parser should fall back to the page header date and treat the body as one entry."""
    path = Path(__file__).parent / "fixtures" / "2025-07-07.html"
    entries = parse_file(path)
    assert len(entries) == 1
    e = entries[0]
    assert e.date.isoformat() == "2025-07-08"  # header says "Tuesday, 8 July 2025"
    assert e.timestamp == datetime(2025, 7, 8, 0, 0)  # midnight fallback
    assert "new phone" in e.text.lower()
