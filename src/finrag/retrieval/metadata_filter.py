"""Pre-Retrieval Metadata Filtering for Financial SEC Filings.

Extracts query metadata constraints (Ticker, Form, Fiscal Year, Item/Section)
and enforces pre-retrieval filtering on Dense, BM25, and Hybrid retrievers
to eliminate cross-company contamination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS


# Mapping from natural language company names / aliases to official SEC ticker symbols
COMPANY_ALIASES: Dict[str, str] = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "iphone": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "azure": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "aws": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "youtube": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "instagram": "META",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "chase": "JPM",
    "goldman": "GS",
    "gs": "GS",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "bofa": "BAC",
    "bac": "BAC",
    "walmart": "WMT",
    "wmt": "WMT",
    "costco": "COST",
    "cost": "COST",
    "johnson & johnson": "JNJ",
    "jnj": "JNJ",
    "unitedhealth": "UNH",
    "unh": "UNH",
    "optum": "UNH",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "xom": "XOM",
}

# Section / Item intent mappings
SECTION_KEYWORDS: Dict[str, str] = {
    "risk": "Item 1A",
    "risks": "Item 1A",
    "risk factors": "Item 1A",
    "md&a": "Item 7",
    "management's discussion": "Item 7",
    "results of operations": "Item 7",
    "operating results": "Item 7",
    "financial statements": "Item 8",
    "balance sheet": "Item 8",
    "income statement": "Item 8",
    "cash flows": "Item 8",
    "business overview": "Item 1",
    "market risk": "Item 7A",
    "controls": "Item 9A",
}


@dataclass
class MetadataFilter:
    """Explicit metadata filter specification."""
    tickers: Optional[List[str]] = None
    forms: Optional[List[str]] = None
    fiscal_years: Optional[List[str]] = None
    item_numbers: Optional[List[str]] = None
    parts: Optional[List[str]] = None

    def is_empty(self) -> bool:
        return not any([
            self.tickers,
            self.forms,
            self.fiscal_years,
            self.item_numbers,
            self.parts
        ])

    def matches(self, meta: Dict[str, Any]) -> bool:
        """Check if document metadata satisfies filter criteria."""
        if self.tickers:
            doc_ticker = meta.get("ticker", "").upper()
            if doc_ticker not in [t.upper() for t in self.tickers]:
                return False

        if self.forms:
            doc_form = meta.get("form", "").upper()
            if doc_form not in [f.upper() for f in self.forms]:
                return False

        if self.fiscal_years:
            doc_fy = str(meta.get("fiscal_year", ""))
            if doc_fy not in [str(y) for y in self.fiscal_years]:
                return False

        if self.item_numbers:
            doc_item = str(meta.get("item_number", "") or "").upper()
            target_items = [str(it).upper() for it in self.item_numbers]
            if not any(it in doc_item for it in target_items):
                return False

        if self.parts:
            doc_part = str(meta.get("part", "") or "").upper()
            if doc_part not in [p.upper() for p in self.parts]:
                return False

        return True


def extract_metadata_filter(query: str) -> MetadataFilter:
    """Extract metadata filter constraints from a natural language query."""
    query_lower = query.lower()
    tickers: List[str] = []
    forms: List[str] = []
    years: List[str] = []
    item_numbers: List[str] = []

    # 1. Tickers / Company Names
    # Sort aliases by length descending to match multi-word names first (e.g. "goldman sachs")
    for alias, ticker in sorted(COMPANY_ALIASES.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, query_lower):
            if ticker not in tickers:
                tickers.append(ticker)

    # 2. Forms (10-K, 10-Q, annual, quarterly)
    if re.search(r"\b10-k\b|\bannual\b", query_lower):
        forms.append("10-K")
    if re.search(r"\b10-q\b|\bquarterly\b|\bq[1-4]\b", query_lower):
        forms.append("10-Q")

    # 3. Fiscal Years (2024, 2025, 2026, FY24, FY25, FY26)
    year_matches = re.findall(r"\b(202[4-7])\b", query)
    for ym in year_matches:
        if ym not in years:
            years.append(ym)

    fy_matches = re.findall(r"\bfy\s*(20)?(2[4-7])\b", query_lower)
    for _, yr in fy_matches:
        full_yr = f"20{yr}"
        if full_yr not in years:
            years.append(full_yr)

    # 4. Item / Section
    for kw, item_val in SECTION_KEYWORDS.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, query_lower):
            if item_val not in item_numbers:
                item_numbers.append(item_val)

    return MetadataFilter(
        tickers=tickers if tickers else None,
        forms=forms if forms else None,
        fiscal_years=years if years else None,
        item_numbers=item_numbers if item_numbers else None
    )


class FilteredRetriever(BaseRetriever):
    """Retriever wrapper that applies a metadata filter to underlying vector/BM25 retrievers."""
    base_retriever: Any
    metadata_filter: Optional[MetadataFilter] = None
    auto_extract_filter: bool = True
    k: int = 5
    fetch_k: int = 30  # Number of candidates to fetch before filtering

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Determine filter
        active_filter = self.metadata_filter
        if self.auto_extract_filter and (active_filter is None or active_filter.is_empty()):
            active_filter = extract_metadata_filter(query)

        # If no filter or cross-document query with multiple companies, run standard retrieval
        if active_filter is None or active_filter.is_empty():
            docs = self.base_retriever.invoke(query)
            return docs[:self.k]

        # Fetch candidate documents with over-fetch factor
        fetch_k = self.fetch_k
        if hasattr(self.base_retriever, "search_kwargs"):
            old_k = self.base_retriever.search_kwargs.get("k", self.k)
            self.base_retriever.search_kwargs["k"] = fetch_k
            candidate_docs = self.base_retriever.invoke(query)
            self.base_retriever.search_kwargs["k"] = old_k
        else:
            candidate_docs = self.base_retriever.invoke(query)

        filtered_docs = []
        for doc in candidate_docs:
            if active_filter.matches(doc.metadata):
                filtered_docs.append(doc)
                if len(filtered_docs) >= self.k:
                    break

        # Fallback: if strict filter returned fewer than k, backfill with top unfiltered
        if len(filtered_docs) < self.k:
            for doc in candidate_docs:
                if doc not in filtered_docs:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= self.k:
                        break

        return filtered_docs[:self.k]