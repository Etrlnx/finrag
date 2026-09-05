from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from finrag.config import config, ChunkingConfig


SEC_SEPARATORS = [
    "\nPART I\n", "\nPART II\n", "\nPART III\n", "\nPART IV\n",
    "\nPart I\n", "\nPart II\n", "\nPart III\n", "\nPart IV\n",
    "\nItem 1.\n", "\nItem 1A.\n", "\nItem 1B.\n", "\nItem 1C.\n",
    "\nItem 2.\n", "\nItem 3.\n", "\nItem 4.\n",
    "\nItem 5.\n", "\nItem 6.\n", "\nItem 7.\n", "\nItem 7A.\n",
    "\nItem 8.\n", "\nItem 9.\n", "\nItem 9A.\n", "\nItem 9B.\n", "\nItem 9C.\n",
    "\nItem 10.\n", "\nItem 11.\n", "\nItem 12.\n", "\nItem 13.\n",
    "\nItem 14.\n", "\nItem 15.\n", "\nItem 16.\n",
    "\nITEM 1.\n", "\nITEM 1A.\n", "\nITEM 1B.\n", "\nITEM 1C.\n",
    "\nITEM 2.\n", "\nITEM 3.\n", "\nITEM 4.\n",
    "\nITEM 5.\n", "\nITEM 6.\n", "\nITEM 7.\n", "\nITEM 7A.\n",
    "\nITEM 8.\n", "\nITEM 9.\n", "\nITEM 9A.\n", "\nITEM 9B.\n", "\nITEM 9C.\n",
    "\nITEM 10.\n", "\nITEM 11.\n", "\nITEM 12.\n", "\nITEM 13.\n",
    "\nITEM 14.\n", "\nITEM 15.\n", "\nITEM 16.\n",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

MIN_SECTION_CHARS = 150
MERGE_THRESHOLD = 500


@dataclass
class ChunkStats:
    strategy: str
    total_chunks: int
    avg_chars: float
    min_chars: int
    max_chars: int
    median_chars: float

    def __str__(self) -> str:
        return (
            f"[{self.strategy}] chunks={self.total_chunks} | "
            f"avg={self.avg_chars:.0f} | min={self.min_chars} | "
            f"max={self.max_chars} | median={self.median_chars:.0f}"
        )


def _extract_fiscal_year(filing_date: str | None) -> str | None:
    if not filing_date:
        return None
    m = re.match(r"(\d{4})", filing_date)
    return m.group(1) if m else None


def _enrich_metadata(meta: dict) -> dict:
    enriched = dict(meta)
    if "filing_date" in enriched and enriched.get("filing_date"):
        enriched.setdefault("fiscal_year", _extract_fiscal_year(enriched["filing_date"]))
    return enriched


def split_table_documents(
    documents: list[Document],
    chunk_size: int = 1000,
) -> list[Document]:
    """Chunk extracted tables without cutting through a row.

    A table is kept whole when it fits. When it does not, it is split on row
    boundaries and the context header plus the markdown column header are
    repeated on every part, so no fragment is an orphaned grid of numbers.
    """
    chunks: list[Document] = []

    for doc in documents:
        content = doc.page_content.strip()
        if not content:
            continue

        base_meta = _enrich_metadata(
            {k: v for k, v in doc.metadata.items() if k != "chunking_strategy"}
        )
        base_meta["chunking_strategy"] = "table"

        if len(content) <= chunk_size:
            chunks.append(Document(page_content=content, metadata=dict(base_meta)))
            continue

        lines = content.split("\n")
        # The context header is everything before the first markdown row.
        table_start = next(
            (i for i, ln in enumerate(lines) if ln.startswith("| ")), 0
        )
        header_block = lines[:table_start]
        header_row = lines[table_start] if table_start < len(lines) else ""
        separator = (
            lines[table_start + 1] if table_start + 1 < len(lines) else ""
        )
        body_rows = lines[table_start + 2 :]

        overhead = len("\n".join(header_block)) + len(header_row) + len(separator) + 3
        budget = max(chunk_size - overhead, 200)

        part: list[str] = []
        part_len = 0
        for row in body_rows:
            row_len = len(row) + 1
            if part and part_len + row_len > budget:
                chunk_text = "\n".join(header_block + [header_row, separator] + part)
                chunks.append(
                    Document(page_content=chunk_text, metadata=dict(base_meta))
                )
                part, part_len = [], 0
            part.append(row)
            part_len += row_len

        if part:
            chunk_text = "\n".join(header_block + [header_row, separator] + part)
            chunks.append(Document(page_content=chunk_text, metadata=dict(base_meta)))

    print(f"[table] {len(chunks)} chunks from {len(documents)} tables")
    return chunks


def fixed_size_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )


def split_fixed(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = fixed_size_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata = _enrich_metadata(chunk.metadata)
        chunk.metadata["chunking_strategy"] = "fixed"
        chunk.metadata["chunk_index"] = i
    print(f"[fixed] {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def recursive_sec_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEC_SEPARATORS,
        keep_separator=True,
    )


def split_recursive(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = recursive_sec_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata = _enrich_metadata(chunk.metadata)
        chunk.metadata["chunking_strategy"] = "recursive"
        chunk.metadata["chunk_index"] = i
    print(f"[recursive] {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def _group_by_filing(documents: list[Document]) -> dict[str, list[Document]]:
    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        key = f"{doc.metadata.get('ticker', '')}_{doc.metadata.get('form', '')}_{doc.metadata.get('filing_date', '')}"
        groups[key].append(doc)
    for docs in groups.values():
        docs.sort(key=lambda d: d.metadata.get("section_index", 0))
    return groups


def _merge_small_fragments(
    docs: list[Document], chunk_size: int
) -> list[Document]:
    merged: list[Document] = []
    buffer_text = ""
    buffer_meta: dict | None = None

    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < MIN_SECTION_CHARS:
            continue

        if len(text) < MERGE_THRESHOLD and buffer_meta is not None:
            candidate = buffer_text + "\n\n" + text
            if len(candidate) <= chunk_size:
                buffer_text = candidate
                continue
            else:
                merged.append(Document(page_content=buffer_text, metadata=dict(buffer_meta)))
                buffer_text = text
                buffer_meta = _enrich_metadata(doc.metadata)
                continue

        if buffer_meta is not None and buffer_text:
            merged.append(Document(page_content=buffer_text, metadata=dict(buffer_meta)))

        buffer_text = text
        buffer_meta = _enrich_metadata(doc.metadata)

    if buffer_meta is not None and buffer_text:
        merged.append(Document(page_content=buffer_text, metadata=dict(buffer_meta)))

    return merged


def split_section_aware(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEC_SEPARATORS,
        keep_separator=True,
    )

    filing_groups = _group_by_filing(documents)

    all_chunks: list[Document] = []
    for filing_key, filing_docs in filing_groups.items():
        merged_docs = _merge_small_fragments(filing_docs, chunk_size)

        for doc in merged_docs:
            section_text = doc.page_content.strip()
            if not section_text:
                continue

            base_meta = {k: v for k, v in doc.metadata.items() if k != "chunking_strategy"}

            if len(section_text) <= chunk_size:
                chunk = Document(
                    page_content=section_text,
                    metadata={**base_meta, "chunking_strategy": "section"},
                )
                all_chunks.append(chunk)
            else:
                sub_chunks = sub_splitter.create_documents(
                    [section_text],
                    metadatas=[{**base_meta, "chunking_strategy": "section"}],
                )
                all_chunks.extend(sub_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i

    print(f"[section] {len(all_chunks)} chunks from {len(documents)} section-documents")
    return all_chunks


ChunkingStrategy = Literal["fixed", "recursive", "section"]

STRATEGY_FUNCS = {
    "fixed": split_fixed,
    "recursive": split_recursive,
    "section": split_section_aware,
}


def split_documents_with_strategy(
    documents: list[Document],
    strategy: ChunkingStrategy = "recursive",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    cfg = config.chunking
    size = chunk_size or cfg.chunk_size
    overlap = chunk_overlap or cfg.chunk_overlap

    func = STRATEGY_FUNCS.get(strategy)
    if func is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_FUNCS)}")

    # Tables and prose need different splitting. Route each through the splitter
    # that preserves its structure, then merge.
    table_docs = [d for d in documents if d.metadata.get("is_table")]
    text_docs = [d for d in documents if not d.metadata.get("is_table")]

    chunks = func(text_docs, chunk_size=size, chunk_overlap=overlap)
    if table_docs:
        chunks = chunks + split_table_documents(table_docs, chunk_size=size)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def compute_chunk_stats(chunks: list[Document], strategy: str) -> ChunkStats:
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        return ChunkStats(strategy, 0, 0, 0, 0, 0)
    return ChunkStats(
        strategy=strategy,
        total_chunks=len(lengths),
        avg_chars=statistics.mean(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        median_chars=statistics.median(lengths),
    )
