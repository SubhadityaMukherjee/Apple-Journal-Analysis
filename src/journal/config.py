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
