from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

from finrag.config import config
from finrag.data import load_all_filings, split_documents_with_strategy, compute_chunk_stats
from finrag.embeddings import get_embeddings
from finrag.vectorstore import build_vector_store, load_vector_store, get_retriever

warnings.filterwarnings("ignore", message="Created a chunk of size")

TEST_QUESTIONS = [
    "What are Microsoft's main risk factors?",
    "What business segments does Amazon operate in?",
    "What is Google's primary source of revenue?",
    "What was Apple's total net revenue in fiscal year 2025?",
    "How much did NVIDIA spend on research and development?",
    "What was JPMorgan Chase's net income for 2025?",
    "How did Meta's advertising revenue change year over year?",
    "What is Walmart's revenue growth trend?",
    "Which companies mention AI infrastructure as a risk factor?",
    "What is Tesla's revenue forecast for 2027?",
]

STRATEGIES = ["fixed", "recursive", "section"]
K = 5

RESULTS_DIR = Path("eval/results")


def get_store_dir(strategy: str) -> Path:
    return Path("data/vector_stores") / strategy


def build_or_load_index(strategy: str, rebuild: bool):
    store_dir = get_store_dir(strategy)
    embeddings = get_embeddings(config.embedding)
    stats = None

    if not rebuild and (store_dir / "index.faiss").exists():
        print(f"\n[{strategy}] Loading existing index from {store_dir}...")
        store = load_vector_store(store_dir, embeddings)
        return store, stats

    print(f"\n[{strategy}] Building index...")
    docs = load_all_filings()
    chunks = split_documents_with_strategy(docs, strategy=strategy)

    stats = compute_chunk_stats(chunks, strategy)
    print(f"  Stats: {stats}")

    store = build_vector_store(chunks, store_dir, embeddings)
    return store, stats


def retrieve_and_display(question: str, retrievers: dict) -> dict:
    results = {}
    for strategy, retriever in retrievers.items():
        try:
            docs = retriever.invoke(question)
            sections = [d.metadata.get("section", "N/A") or "N/A" for d in docs]
            tickers = [d.metadata.get("ticker", "?") for d in docs]
            item_numbers = [d.metadata.get("item_number", "N/A") for d in docs]
            avg_len = sum(len(d.page_content) for d in docs) / max(len(docs), 1)
            previews = [d.page_content[:150].replace("\n", " ") for d in docs]
            results[strategy] = {
                "count": len(docs),
                "tickers": tickers,
                "sections": sections,
                "item_numbers": item_numbers,
                "avg_chunk_len": round(avg_len, 1),
                "previews": previews,
            }
        except Exception as e:
            results[strategy] = {"error": str(e)}
    return results


def print_comparison_table(question: str, results: dict):
    print(f"\n{'─' * 80}")
    print(f"Q: {question}")
    print(f"{'─' * 80}")
    print(f"  {'Strategy':<12} {'Docs':>4}  {'Avg len':>7}  {'Tickers':<30}  {'Top Section'}")
    print(f"  {'─' * 12} {'─' * 4}  {'─' * 7}  {'─' * 30}  {'─' * 25}")

    for strategy in STRATEGIES:
        r = results.get(strategy, {})
        if "error" in r:
            print(f"  {strategy:<12}  ERROR: {r['error']}")
            continue

        tickers_str = ", ".join(sorted(set(r["tickers"])))[:28]
        top_section = (r["sections"][0] if r["sections"] else "N/A") or "N/A"
        top_section = top_section[:40]

        print(
            f"  {strategy:<12} {r['count']:>4}  {r['avg_chunk_len']:>7.0f}  "
            f"{tickers_str:<30}  {top_section}"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Chunking strategy comparison")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild all indexes")
    args = parser.parse_args()

    print("=" * 80)
    print("  FinRAG Phase 2 — Chunking Strategy Comparison")
    print("=" * 80)

    index_stats = {}
    retrievers = {}
    for strategy in STRATEGIES:
        store, stats = build_or_load_index(strategy, args.rebuild)
        retrievers[strategy] = get_retriever(store, k=K)

        if stats:
            index_stats[strategy] = {
                "total_chunks": stats.total_chunks,
                "avg_chars": round(stats.avg_chars, 1),
                "min_chars": stats.min_chars,
                "max_chars": stats.max_chars,
                "median_chars": round(stats.median_chars, 1),
            }

    print(f"\n{'=' * 80}")
    print(f"  Retrieval Comparison  (k={K} per strategy)")
    print(f"{'=' * 80}")

    all_results = []
    for question in TEST_QUESTIONS:
        results = retrieve_and_display(question, retrievers)
        print_comparison_table(question, results)
        all_results.append({"question": question, "results": results})
        time.sleep(0.1)

    print(f"\n{'=' * 80}")
    print("  Phase 2 Complete")
    print(f"{'=' * 80}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "k": K,
        "strategies": STRATEGIES,
        "index_stats": index_stats,
        "questions": all_results,
    }
    output_path = RESULTS_DIR / "phase2_chunking_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
