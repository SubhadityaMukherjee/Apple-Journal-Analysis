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
    cat_indices = [i for i, l in enumerate(lines) if l in KNOWN_CATEGORIES]
    if not cat_indices:
        return text, None
    category = lines[cat_indices[-1]]
    remaining = [l for i, l in enumerate(lines) if i not in cat_indices]
    body = "\n".join(remaining).strip()
    return body, category


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
                if body:
                    entries.append(Entry(
                        file_path=str(path),
                        date=file_date,
                        timestamp=current_ts,
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
        if body:
            entries.append(Entry(
                file_path=str(path),
                date=file_date,
                timestamp=current_ts,
                text=body,
                category=cat,
            ))

    return entries
