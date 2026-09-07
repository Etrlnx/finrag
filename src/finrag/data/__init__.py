from __future__ import annotations

from finrag.data.loader import load_filing, load_all_filings
from finrag.data.splitter import get_text_splitter, split_documents
from finrag.data.chunking import (
    split_documents_with_strategy,
    split_fixed,
    split_recursive,
    split_section_aware,
    compute_chunk_stats,
    ChunkingStrategy,
    ChunkStats,
)

__all__ = [
    "load_filing",
    "load_all_filings",
    "get_text_splitter",
    "split_documents",
    "split_documents_with_strategy",
    "split_fixed",
    "split_recursive",
    "split_section_aware",
    "compute_chunk_stats",
    "ChunkingStrategy",
    "ChunkStats",
]
