"""Build a FAISS index from the SEC corpus.

Phase 6 introduced two corpus changes that both need measuring:
  1. the inline-XBRL fix (numbers no longer deleted during cleaning)
  2. table-aware extraction (financial statements as markdown tables)

To separate them, this script can build either variant:

    python scripts/build_index.py --variant tables    # XBRL fix + tables
    python scripts/build_index.py --variant textonly  # XBRL fix, no tables
    python scripts/build_index.py --variant legacy    # neither (Phase 5 corpus)

Usage:
    python scripts/build_index.py --variant tables
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_community.vectorstores import FAISS  # noqa: E402

from finrag.config import config  # noqa: E402
from finrag.data import load_all_filings, split_documents_with_strategy  # noqa: E402
from finrag.embeddings import get_embeddings  # noqa: E402
from finrag.data.chunking import compute_chunk_stats  # noqa: E402

VARIANTS = {
    "tables": ("data/vector_stores/phase6_tables_bge_base", True, True),
    "textonly": ("data/vector_stores/phase6_textonly_bge_base", True, False),
    "legacy": ("data/vector_stores/phase6_legacy_bge_base", False, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default="tables",
        help="Which corpus variant to build (default: tables)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load the first N filings (for quick smoke tests)",
    )
    args = parser.parse_args()

    out_dir, use_xbrl_fix, include_tables = VARIANTS[args.variant]

    print(f"Building '{args.variant}' index -> {out_dir}")
    print(f"  XBRL value-preservation fix: {use_xbrl_fix}")
    print(f"  table extraction:            {include_tables}")

    if not use_xbrl_fix:
        # Restore the legacy decompose-everything behaviour to reproduce the
        # Phase 5 corpus exactly.
        from finrag.data import xbrl as xbrl_mod

        xbrl_mod.XBRL_UNWRAP_TAGS = []
        xbrl_mod.LEGACY_MODE = True
        print("  (legacy mode: unwrapping disabled, ix:* tags decomposed)")

    t0 = time.time()
    docs = load_all_filings(include_tables=include_tables)
    if args.limit:
        docs = docs[: args.limit]
    print(f"Loaded {len(docs)} documents in {time.time() - t0:.1f}s")

    t0 = time.time()
    chunks = split_documents_with_strategy(docs, strategy="recursive")
    print(f"Split into {len(chunks)} chunks in {time.time() - t0:.1f}s")
    print(compute_chunk_stats(chunks, "recursive"))
    print(f"  table chunks: {sum(1 for c in chunks if c.metadata.get('is_table'))}")

    t0 = time.time()
    embeddings = get_embeddings(config.embedding)
    store = FAISS.from_documents(chunks, embeddings)
    print(f"Embedded in {time.time() - t0:.1f}s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(out_dir)
    print(f"Saved index to {out_dir} ({store.index.ntotal} vectors)")


if __name__ == "__main__":
    main()
