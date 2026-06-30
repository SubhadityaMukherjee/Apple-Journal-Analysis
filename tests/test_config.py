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
