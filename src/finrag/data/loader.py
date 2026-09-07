from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_core.documents import Document

from finrag.config import config
from finrag.data.table_extractor import extract_tables
from finrag.data.xbrl import (  # noqa: F401  (re-exported for compatibility)
    NAVIGATION_SELECTORS,
    XBRL_TAGS,
    XBRL_UNWRAP_TAGS,
    clean_xbrl,
)
from finrag.data.sec_headings import (  # noqa: F401
    HEADING_RE,
    ITEM_NUM_RE,
    PART_RE,
    TOC_THRESHOLD,
    normalize_heading as _normalize_heading,
    parse_item_number as _parse_item_number,
    parse_part as _parse_part,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def _is_toc_line(line: str) -> bool:
    return len(line) < TOC_THRESHOLD


def extract_text_with_structure(
    soup: BeautifulSoup, already_cleaned: bool = False
) -> tuple[str, list[tuple[int, str]]]:
    if not already_cleaned:
        clean_xbrl(soup)
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.split("\n")

    raw_heading_lines: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if HEADING_RE.match(stripped) and len(stripped) < 120:
            raw_heading_lines.append((i, stripped))

    toc_end = 0
    if raw_heading_lines:
        first_line_idx = raw_heading_lines[0][0]
        consecutive_short = 0
        for idx, (line_no, line_text) in enumerate(raw_heading_lines):
            if line_no - (raw_heading_lines[idx - 1][0] if idx > 0 else first_line_idx) <= 5:
                if _is_toc_line(line_text):
                    consecutive_short += 1
                    toc_end = idx
                else:
                    break
            else:
                break
        if consecutive_short >= 3:
            raw_heading_lines = raw_heading_lines[toc_end + 1:]

    seen_keys: set[str] = set()
    headings: list[tuple[int, str]] = []
    for line_no, line_text in raw_heading_lines:
        normalized = _normalize_heading(line_text)
        if PART_RE.match(normalized):
            key = normalized.upper()
        else:
            item_match = ITEM_NUM_RE.match(normalized)
            if item_match:
                key = f"item_{item_match.group(1).upper()}"
            else:
                key = normalized.lower()

        if key not in seen_keys:
            seen_keys.add(key)
            headings.append((line_no, normalized))

    return text, headings


def _parse_item_number(heading: str | None) -> str | None:
    if not heading:
        return None
    m = ITEM_NUM_RE.match(heading)
    return m.group(1) if m else None


def _parse_part(heading: str | None) -> str | None:
    if not heading:
        return None
    m = PART_RE.match(heading)
    return m.group(1) if m else None


def split_by_sections(
    text: str, headings: list[tuple[int, str]]
) -> list[tuple[str | None, str, str | None, str | None]]:
    if not headings:
        return [(None, text, None, None)]

    lines = text.split("\n")

    sections: list[tuple[str | None, str, str | None, str | None]] = []
    current_part: str | None = None

    for i, (start_line, heading) in enumerate(headings):
        end_line = headings[i + 1][0] if i + 1 < len(headings) else len(lines)

        section_lines = lines[start_line:end_line]
        section_text = "\n".join(section_lines).strip()

        part_from_heading = _parse_part(heading)
        if part_from_heading:
            current_part = part_from_heading

        is_part_only = part_from_heading is not None and _parse_item_number(heading) is None

        if len(section_text) > 100:
            item_num = _parse_item_number(heading)
            sections.append((heading, section_text, item_num, current_part))
        elif is_part_only:
            pass

    if not sections:
        return [(None, text, None, None)]

    return sections


def load_filing(
    file_path: str | Path,
    metadata: dict[str, Any],
    include_tables: bool = True,
) -> list[Document]:
    path = Path(file_path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Clean once and share the soup between the text and table passes, so both
    # see the same unwrapped inline-XBRL values.
    clean_xbrl(soup)

    tables: list[Document] = []
    if include_tables:
        base_meta = metadata.copy()
        base_meta["source_file"] = str(path)
        try:
            tables = extract_tables(soup, base_meta, clean=False)
        except Exception as e:
            print(f"Table extraction failed for {path.name}: {e}")

        # Now remove the tables before building the prose stream. Otherwise the
        # same figures appear twice: once as a clean markdown table and once
        # flattened into surrounding text. The flattened copy is noisier, yet
        # it competes for the same queries and frequently outranks the table.
        for tbl in soup.find_all("table"):
            tbl.decompose()

    text, headings = extract_text_with_structure(soup, already_cleaned=True)
    sections = split_by_sections(text, headings)

    documents = []
    for i, (section_name, section_text, item_num, part) in enumerate(sections):
        if not section_text or len(section_text.strip()) < 100:
            continue

        doc_metadata = metadata.copy()
        doc_metadata.update({
            "source_file": str(path),
            "section": section_name,
            "section_index": i,
            "item_number": item_num,
            "part": part,
            "content_type": "text",
            "is_table": False,
            "char_count": len(section_text),
        })
        documents.append(Document(page_content=section_text, metadata=doc_metadata))

    documents.extend(tables)
    return documents


def load_all_filings(
    manifest_path: str | Path | None = None,
    include_tables: bool = True,
) -> list[Document]:
    manifest_path = Path(manifest_path) if manifest_path else config.paths.manifest_path
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_docs = []
    for entry in manifest:
        file_path = entry["file_path"]
        try:
            docs = load_filing(file_path, {
                "ticker": entry["ticker"],
                "company_name": entry["company_name"],
                "cik": entry["cik"],
                "form": entry["form"],
                "filing_date": entry["filing_date"],
                "report_date": entry["report_date"],
                "accession_number": entry["accession_number"],
            }, include_tables=include_tables)
            all_docs.extend(docs)
            n_tables = sum(1 for d in docs if d.metadata.get("is_table"))
            print(
                f"Loaded {len(docs) - n_tables} sections + {n_tables} tables "
                f"from {entry['ticker']} {entry['form']} ({entry['filing_date']})"
            )
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    total_tables = sum(1 for d in all_docs if d.metadata.get("is_table"))
    print(f"Total documents loaded: {len(all_docs)} ({total_tables} tables)")
    return all_docs
