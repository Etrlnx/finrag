"""SEC section-heading patterns shared by the text loader and the table extractor.

Lives in its own module to avoid a circular import between those two.
"""

from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(PART\s+[IVX]+|Item\s+\d+[A-Z]?)", re.IGNORECASE)
ITEM_NUM_RE = re.compile(r"[Ii]tem\s+(\d+[A-Z]?)", re.IGNORECASE)
PART_RE = re.compile(r"^(PART\s+[IVX]+)", re.IGNORECASE)

# Table-of-contents entries are short one-liners; real headings are longer.
TOC_THRESHOLD = 25


def normalize_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def parse_item_number(heading: str | None) -> str | None:
    if not heading:
        return None
    m = ITEM_NUM_RE.match(heading)
    return m.group(1) if m else None


def parse_part(heading: str | None) -> str | None:
    if not heading:
        return None
    m = PART_RE.match(heading)
    return m.group(1) if m else None
