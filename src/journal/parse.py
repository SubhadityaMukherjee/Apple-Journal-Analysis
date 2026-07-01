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

HEADER_DATE_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{4})$"
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


def _parse_header_date(line: str) -> date | None:
    m = HEADER_DATE_RE.match(line.strip())
    if not m:
        return None
    _, day, month_name, year = m.groups()
    month_num = datetime.strptime(month_name, "%B").month
    return date(int(year), month_num, int(day))


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

    page_header_date: date | None = None
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
                        date=current_ts.date(),
                        timestamp=current_ts,
                        text=body,
                        category=cat,
                    ))
            current_ts = ts
            current_buf = []
            continue
        hd = _parse_header_date(line)
        if hd is not None:
            if page_header_date is None:
                page_header_date = hd
            continue
        current_buf.append(line)

    if current_ts is not None and current_buf:
        body, cat = _extract_category("\n".join(current_buf))
        if body:
            entries.append(Entry(
                file_path=str(path),
                date=current_ts.date(),
                timestamp=current_ts,
                text=body,
                category=cat,
            ))

    if not entries and page_header_date is not None and current_buf:
        body, cat = _extract_category("\n".join(current_buf))
        if body:
            fallback_ts = datetime.combine(page_header_date, datetime.min.time())
            entries.append(Entry(
                file_path=str(path),
                date=page_header_date,
                timestamp=fallback_ts,
                text=body,
                category=cat,
            ))

    return entries
