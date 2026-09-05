"""Phase 6: HTML table extraction for SEC filings.

SEC filings are HTML, so financial statements live in <table> elements. The
plain-text loader flattens those tables into a stream of unlabelled numbers,
which destroys the row/column relationships that make the figures meaningful
(revenue by segment, assets vs liabilities, year-over-year columns).

This module extracts each <table> as its own Document, rendered as a markdown
table with a context header, so that:

  * a chunk is never cut through the middle of a row,
  * every table chunk carries its ticker / form / date for metadata filtering,
  * BM25 has the row labels and the company name as exact-match anchors.

Usage:
    from finrag.data.table_extractor import extract_tables
    tables = extract_tables(soup, metadata)
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document

from finrag.data.sec_headings import (
    HEADING_RE,
    normalize_heading as _normalize_heading,
    parse_item_number as _parse_item_number,
    parse_part as _parse_part,
)
from finrag.data.xbrl import clean_xbrl

# --- Quality thresholds -------------------------------------------------------
# SEC HTML is full of layout tables, checkbox grids on the cover page, and
# single-cell spacers. These filters keep only tables that carry real content.
MIN_ROWS = 2
MIN_COLS = 2
MIN_ALNUM_CHARS = 40
MIN_FILLED_CELL_RATIO = 0.25

# Financial tables are, by definition, mostly numbers. Cover-page boilerplate
# (registrant name, exchange listing, officer signatures) is prose in a grid,
# so requiring a handful of numeric cells filters it out cheaply.
MIN_NUMERIC_CELLS = 3

# Tables appearing before any recognised section heading are cover-page or
# boilerplate in SEC filings. Require a section unless the table is dense enough
# with figures to be worth keeping regardless.
NUMERIC_CELLS_TO_ALLOW_NO_SECTION = 12

# Cover-page checkbox tables render as these glyphs and nothing else.
CHECKBOX_GLYPHS = set("☒☐\u2610\u2611\u2612")

# A caption is the short text immediately above a table ("CONDENSED CONSOLIDATED
# STATEMENTS OF OPERATIONS"). It is the single most useful lexical anchor for
# retrieval, so we look for it explicitly.
MIN_CAPTION_CHARS = 4
MAX_CAPTION_CHARS = 150


def _clean_cell(text: str) -> str:
    """Collapse whitespace and normalise common SEC cell artefacts."""
    text = text.replace("\xa0", " ").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("|", "/")  # keep markdown table syntax valid
    return text.strip()


def _cell_is_checkbox(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return all(ch in CHECKBOX_GLYPHS or ch.isspace() for ch in stripped)


NUMBER_RE = re.compile(r"\d")

# SEC filings split the currency glyph into its own cell: <td>$</td><td>78,678</td>.
# Joining them back makes the figure readable and matchable as "$78,678".
CURRENCY_ONLY_RE = re.compile(r"^[\$\u20ac\u00a3]$")

# Tables that are navigation/index rather than data.
JUNK_CAPTION_RE = re.compile(
    r"table\s+of\s+contents|index\s+to\s|cross[- ]reference\s+sheet", re.IGNORECASE
)


def _merge_currency_cells(grid: list[list[str]]) -> list[list[str]]:
    """Attach a lone currency symbol to the number that follows it."""
    out: list[list[str]] = []
    for row in grid:
        new_row: list[str] = []
        pending = ""
        for cell in row:
            text = cell.strip()
            if CURRENCY_ONLY_RE.match(text):
                pending = text
                continue
            if pending:
                if text:
                    new_row.append(f"{pending}{text}")
                    pending = ""
                else:
                    new_row.append(pending)
                    pending = ""
                continue
            new_row.append(cell)
        if pending:
            new_row.append(pending)
        out.append(new_row)
    return out


def _prune_empty_columns(grid: list[list[str]]) -> list[list[str]]:
    """Drop columns that are blank in every row.

    colspan padding leaves large stretches of empty columns; removing them is
    what turns a 24-column financial statement into its real 5-column shape.
    """
    if not grid:
        return grid

    width = max(len(r) for r in grid)
    keep = [
        j
        for j in range(width)
        if any(j < len(r) and r[j].strip() for r in grid)
    ]
    if len(keep) == width:
        return grid

    return [[r[j] if j < len(r) else "" for j in keep] for r in grid]


def postprocess_grid(grid: list[list[str]]) -> list[list[str]]:
    """Apply the cleanup passes that run after a grid is parsed."""
    grid = _merge_currency_cells(grid)
    grid = _prune_empty_columns(grid)
    return grid


def count_numeric_cells(grid: list[list[str]]) -> int:
    """Count cells containing at least one digit."""
    return sum(1 for row in grid for c in row if c and NUMBER_RE.search(c))


def table_to_grid(table: Tag) -> list[list[str]]:
    """Flatten a <table> into a rectangular grid of cleaned strings.

    colspan is handled by placing the value in the first covered column and
    leaving the rest blank. Repeating the value (the obvious approach) triples
    the size of SEC tables, because filings lean on colspan heavily for
    presentation, and it swamps the embedding with duplicated text.

    rowspan is not expanded -- SEC financial statements use it rarely, and
    guessing at it introduces more errors than it fixes.
    """
    grid: list[list[str]] = []

    for tr in table.find_all("tr"):
        row: list[str] = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            text = _clean_cell(cell.get_text(" ", strip=True))
            try:
                span = int(cell.get("colspan", 1) or 1)
            except (TypeError, ValueError):
                span = 1
            span = max(1, min(span, 12))  # guard against absurd colspan values
            row.append(text)
            row.extend([""] * (span - 1))
        if row:
            grid.append(row)

    return grid


def _grid_is_worthwhile(
    grid: list[list[str]], has_section: bool = True
) -> bool:
    if len(grid) < MIN_ROWS:
        return False

    cells = [c for row in grid for c in row]
    if not cells:
        return False

    filled = [c for c in cells if c and not _cell_is_checkbox(c)]
    if not filled:
        return False

    # Blanking colspan padding means most rows carry empty trailing columns, so
    # raw cell counts overstate the sparsity of the table. Measure against the
    # columns that actually hold content.
    width = max(len(r) for r in grid)
    effective_cols = sum(
        1
        for j in range(width)
        if any(j < len(r) and r[j] and not _cell_is_checkbox(r[j]) for r in grid)
    )
    if effective_cols < MIN_COLS:
        return False

    density = len(filled) / (effective_cols * len(grid))
    if density < MIN_FILLED_CELL_RATIO:
        return False

    alnum = sum(len(c) for c in filled)
    if alnum < MIN_ALNUM_CHARS:
        return False

    numeric = count_numeric_cells(grid)
    if numeric < MIN_NUMERIC_CELLS:
        return False

    # No section heading means this sits on the cover page or in boilerplate.
    # Keep it only if it is unmistakably a dense financial statement.
    if not has_section and numeric < NUMERIC_CELLS_TO_ALLOW_NO_SECTION:
        return False

    return True


def grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a grid as a markdown table.

    The first row becomes the header when it is predominantly non-numeric.
    Financial statements often put period labels ("2026", "2025") in the header
    row, so we only treat it as a header when it looks like a label row.
    """
    if not grid:
        return ""

    width = max(len(r) for r in grid)

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (width - len(row))

    header = pad(grid[0])
    body = [pad(r) for r in grid[1:]]

    def looks_numeric(row: list[str]) -> bool:
        cells = [c for c in row if c]
        if not cells:
            return False
        numeric = sum(
            1 for c in cells if re.fullmatch(r"[\s$\u20ac\u00a3()\-–—,.\d]+", c)
        )
        return numeric / len(cells) > 0.5

    # If the first row is mostly numbers it is data, not a header -- synthesise
    # a blank header so the markdown still parses.
    if looks_numeric(header):
        header = [f"col_{i + 1}" for i in range(width)]
        body = [pad(r) for r in grid]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _find_caption(table: Tag) -> str | None:
    """Find the short descriptive text immediately preceding a table."""
    node = table
    for _ in range(6):
        node = node.previous_sibling
        while node is not None and getattr(node, "name", None) is None:
            node = node.previous_sibling
        if node is None:
            # Walk up one level and retry from the parent's previous sibling.
            parent = table.parent
            if parent is None:
                return None
            node = parent.previous_sibling
            while node is not None and getattr(node, "name", None) is None:
                node = node.previous_sibling
            if node is None:
                return None

        text = _clean_cell(node.get_text(" ", strip=True)) if hasattr(node, "get_text") else ""
        if not text:
            continue
        if len(text) < MIN_CAPTION_CHARS or len(text) > MAX_CAPTION_CHARS:
            if len(text) > MAX_CAPTION_CHARS:
                return None
            continue
        if HEADING_RE.match(text):
            return None
        return text

    return None


def _build_content(
    meta: dict[str, Any], caption: str | None, markdown: str
) -> str:
    """Prepend a context header so every table chunk is self-describing.

    Without this a retrieved table is an anonymous grid of numbers -- the reader
    (and the embedding) has no idea which company or period it belongs to.
    """
    parts = [
        f"TABLE from {meta.get('company_name', meta.get('ticker', 'Unknown'))} "
        f"({meta.get('ticker', '?')}) {meta.get('form', '?')} "
        f"filed {meta.get('filing_date', '?')}"
    ]
    section = meta.get("section")
    if section:
        parts.append(f"Section: {section}")
    if caption:
        parts.append(f"Table: {caption}")
    return "\n".join(parts) + "\n\n" + markdown


def _has_tag_child(el: Tag) -> bool:
    """True when the element has at least one child element (i.e. is not a leaf)."""
    for child in el.children:
        if isinstance(child, Tag):
            return True
    return False


def iter_elements(root: Tag) -> Iterator[tuple[Tag, bool, bool]]:
    """Walk the tree in document order, yielding (element, inside_table, inside_anchor).

    Done as one DFS with explicit state instead of per-element find_parent()
    calls. A 10-K runs to several hundred thousand elements, and find_parent is
    O(depth) each, which pushes extraction past tens of minutes per filing.

    Tables are yielded but not descended into, so nested tables are neither
    double-counted nor emitted out of order.
    """
    stack: list[tuple[Tag, bool, bool]] = [(root, False, False)]

    while stack:
        node, inside_table, inside_a = stack.pop()

        is_tag = isinstance(node, Tag)
        if is_tag:
            yield node, inside_table, inside_a
            if node.name == "table":
                continue

        child_table = inside_table or (is_tag and node.name == "table")
        child_a = inside_a or (is_tag and node.name == "a")

        children = getattr(node, "children", None)
        if children is None:
            continue
        for child in reversed(list(children)):
            if isinstance(child, Tag):
                stack.append((child, child_table, child_a))


def extract_tables(
    soup: BeautifulSoup,
    base_metadata: dict[str, Any],
    clean: bool = True,
) -> list[Document]:
    """Extract every content-bearing <table> in a filing as its own Document.

    Walks the document in order so each table inherits the section heading that
    most recently preceded it -- the same section labels the text loader uses.
    """
    if clean:
        soup = clean_xbrl(soup)

    documents: list[Document] = []

    current_section: str | None = None
    current_item: str | None = None
    current_part: str | None = None

    for el, inside_table, inside_a in iter_elements(soup):
        # Skip anything inside a table; tables are handled at the top level and
        # nested tables would otherwise be emitted twice or out of order.
        if inside_table:
            continue

        if el.name == "table":
            grid = postprocess_grid(table_to_grid(el))
            if not _grid_is_worthwhile(grid, has_section=current_section is not None):
                continue

            caption = _find_caption(el)
            if caption and JUNK_CAPTION_RE.search(caption):
                continue

            markdown = grid_to_markdown(grid)
            if not markdown:
                continue

            meta = dict(base_metadata)
            meta.update(
                {
                    "section": current_section,
                    "item_number": current_item,
                    "part": current_part,
                    "content_type": "table",
                    "is_table": True,
                    "table_caption": caption,
                    "n_rows": len(grid),
                    "n_cols": max(len(r) for r in grid) if grid else 0,
                }
            )

            content = _build_content(meta, caption, markdown)
            meta["char_count"] = len(content)
            documents.append(Document(page_content=content, metadata=meta))
            continue

        # Heading tracking: only leaf elements whose text matches a SEC heading
        # pattern. Anchors are skipped because table-of-contents entries are
        # links and would otherwise masquerade as real section headings.
        if inside_a or _has_tag_child(el):
            continue

        text = _normalize_heading(el.get_text(" ", strip=True) or "")
        if not text or len(text) > 120 or not HEADING_RE.match(text):
            continue

        part = _parse_part(text)
        if part:
            current_part = part
        item = _parse_item_number(text)
        if item:
            current_item = item
        current_section = text

    return documents
