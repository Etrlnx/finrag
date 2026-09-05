"""Phase 7: cross-encoder reranking benchmark.

The reranker itself already exists (finrag.retrieval.get_reranker + LangChain's
ContextualCompressionRetriever). What Phase 7 adds is the measurement: does
rescoring a wide candidate set actually beat the Phase 5 filtered-hybrid
baseline, and what does it cost in latency?

Configurations compared:
  * Filtered Hybrid, no rerank (Phase 5 champion) -- baseline
  * Filtered Hybrid + rerank, fetching 20 candidates and keeping 5
  * Filtered Hybrid + rerank, fetching 50 candidates and keeping 5

Metrics: Hit@5, MRR@5, Recall@5, Precision@5, NDCG@5, latency.

Usage:
    python eval/evaluate_reranking.py --store data/vector_stores/phase6_tables_bge_base
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

from finrag.config import config  # noqa: E402
from finrag.data import load_all_filings, split_documents_with_strategy  # noqa: E402
from finrag.embeddings import get_embeddings  # noqa: E402
from finrag.retrieval import (  # noqa: E402
    get_bm25_retriever,
    get_dense_retriever,
    get_ensemble_retriever,
    get_filtered_retriever,
    get_reranker,
)
from langchain_classic.retrievers import ContextualCompressionRetriever  # noqa: E402

CATEGORIES = ["factual", "numerical", "temporal", "cross-document", "unanswerable"]


def load_eval_dataset(eval_path: str = "eval/eval_set.json") -> List[Dict[str, Any]]:
    return json.loads(Path(eval_path).read_text(encoding="utf-8"))


def _relevance(doc: Document, item: Dict[str, Any], is_cross_doc: bool,
               keywords: List[str], sections: List[str]) -> float:
    """Graded relevance: how many ground-truth keywords this chunk carries.

    unwrap()ed numeric cells mean a keyword such as "416,161" now appears in
    real table chunks, so this grades tables and prose on the same scale.
    """
    content = doc.page_content.lower()
    doc_ticker = (doc.metadata.get("ticker") or "").upper()
    doc_section = (doc.metadata.get("section") or "").lower()

    if not is_cross_doc and doc_ticker != (item.get("ticker") or "").upper():
        return 0.0

    matched = sum(1 for kw in keywords if kw in content)
    section_hit = any(s.lower() in doc_section or s.lower() in content for s in sections)
    if matched == 0 and not section_hit:
        return 0.0
    return float(max(matched, 1))


def ndcg_at_k(rels: List[float], k: int) -> float:
    def dcg(scores: List[float]) -> float:
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))

    actual = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0


def evaluate_item(
    item: Dict[str, Any], retrieved: List[Document], k: int
) -> Dict[str, Any]:
    category = item.get("category", "factual")
    if category == "unanswerable":
        return {
            "hit": True, "reciprocal_rank": 1.0, "precision": 1.0,
            "recall": 1.0, "ndcg": 1.0, "contamination": 0.0,
            "retrieved_tickers": [d.metadata.get("ticker") for d in retrieved[:k]],
        }

    ticker = item.get("ticker", "")
    keywords = [kw.lower() for kw in item.get("ground_truth_keywords", [])]
    sections = item.get("target_sections", [])
    is_cross_doc = ticker in ["TECH", "BANKS"]

    top = retrieved[:k]
    rels = [_relevance(d, item, is_cross_doc, keywords, sections) for d in top]

    relevant_ranks = [i + 1 for i, r in enumerate(rels) if r > 0]
    matched_keywords = set()
    for d in top:
        content = d.page_content.lower()
        for kw in keywords:
            if kw in content:
                matched_keywords.add(kw)

    contaminated = 0
    if not is_cross_doc:
        contaminated = sum(
            1
            for d in top
            if (d.metadata.get("ticker") or "").upper() != ticker.upper()
        )

    hit = len(relevant_ranks) > 0
    return {
        "hit": hit,
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "precision": sum(1 for r in rels if r > 0) / k if k else 0.0,
        "recall": (
            len(matched_keywords) / len(keywords) if keywords else (1.0 if hit else 0.0)
        ),
        "ndcg": ndcg_at_k(rels, k),
        "contamination": contaminated / k if k and not is_cross_doc else 0.0,
        "retrieved_tickers": [d.metadata.get("ticker") for d in top],
    }


def evaluate_config(
    name: str, retriever: Any, eval_set: List[Dict[str, Any]], k: int
) -> Dict[str, Any]:
    print(f"\n{'=' * 62}\n Evaluating: {name} (k={k})\n{'=' * 62}")

    per_question = []
    latencies, hits, mrrs, precs, recs, ndcgs, contams = [], [], [], [], [], [], []
    by_cat: Dict[str, Dict[str, List[float]]] = {}
    by_table: Dict[str, List[float]] = {}

    for item in eval_set:
        q = item["question"]
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        latencies.append((time.perf_counter() - t0) * 1000)

        res = evaluate_item(item, docs, k)
        hits.append(1.0 if res["hit"] else 0.0)
        mrrs.append(res["reciprocal_rank"])
        precs.append(res["precision"])
        recs.append(res["recall"])
        ndcgs.append(res["ndcg"])
        contams.append(res["contamination"])

        cat = item.get("category", "factual")
        by_cat.setdefault(
            cat, {"hit": [], "mrr": [], "precision": [], "recall": [], "ndcg": []}
        )
        by_cat[cat]["hit"].append(1.0 if res["hit"] else 0.0)
        by_cat[cat]["mrr"].append(res["reciprocal_rank"])
        by_cat[cat]["precision"].append(res["precision"])
        by_cat[cat]["recall"].append(res["recall"])
        by_cat[cat]["ndcg"].append(res["ndcg"])

        bucket = "table" if item.get("requires_table") else "text"
        by_table.setdefault(bucket, {"recall": [], "mrr": [], "ndcg": []})
        by_table[bucket]["recall"].append(res["recall"])
        by_table[bucket]["mrr"].append(res["reciprocal_rank"])
        by_table[bucket]["ndcg"].append(res["ndcg"])

        per_question.append(
            {
                "id": item["id"],
                "question": q,
                "category": cat,
                "requires_table": bool(item.get("requires_table")),
                "latency_ms": round(latencies[-1], 2),
                "metrics": res,
            }
        )

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "config_name": name,
        "k": k,
        "avg_latency_ms": mean(latencies),
        "hit_rate": mean(hits),
        "mrr": mean(mrrs),
        "precision": mean(precs),
        "recall": mean(recs),
        "ndcg": mean(ndcgs),
        "contamination": mean(contams),
        "by_category": {
            c: {
                "hit_rate": mean(v["hit"]),
                "mrr": mean(v["mrr"]),
                "precision": mean(v["precision"]),
                "recall": mean(v["recall"]),
                "ndcg": mean(v["ndcg"]),
                "count": len(v["hit"]),
            }
            for c, v in by_cat.items()
        },
        "by_table_dependency": {
            b: {
                "recall": mean(v["recall"]),
                "mrr": mean(v["mrr"]),
                "ndcg": mean(v["ndcg"]),
                "count": len(v["recall"]),
            }
            for b, v in by_table.items()
        },
        "per_question_results": per_question,
    }


def write_report(results: List[Dict[str, Any]], out_path: str) -> None:
    md = [
        "# FinRAG Phase 7: Cross-Encoder Reranking Benchmark\n",
        "Compares the Phase 5 filtered-hybrid baseline against cross-encoder "
        "reranking over a wider candidate pool. Model: "
        "`cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API key).\n",
        "## 1. Overall (at k=5)\n",
        "| Configuration | Latency (ms) | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md.append(
            f"| **{r['config_name']}** | {r['avg_latency_ms']:.1f} | "
            f"{r['hit_rate'] * 100:.1f}% | {r['mrr']:.3f} | {r['recall'] * 100:.1f}% | "
            f"{r['precision'] * 100:.1f}% | {r['ndcg']:.3f} |"
        )

    md.append("\n## 2. Effect on table-dependent questions\n")
    md.append("| Configuration | Table Q recall | Table Q MRR | Table Q NDCG | Text Q recall |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in results:
        t = r["by_table_dependency"].get("table", {})
        x = r["by_table_dependency"].get("text", {})
        md.append(
            f"| **{r['config_name']}** | {t.get('recall', 0) * 100:.1f}% | "
            f"{t.get('mrr', 0):.3f} | {t.get('ndcg', 0):.3f} | "
            f"{x.get('recall', 0) * 100:.1f}% |"
        )

    md.append("\n## 3. Category breakdown\n")
    for cat in CATEGORIES:
        rows = [r for r in results if cat in r["by_category"]]
        if not rows:
            continue
        md.append(f"### `{cat.upper()}`")
        md.append("| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for r in rows:
            c = r["by_category"][cat]
            md.append(
                f"| {r['config_name']} | {c['hit_rate'] * 100:.1f}% | {c['mrr']:.3f} | "
                f"{c['recall'] * 100:.1f}% | {c['ndcg']:.3f} |"
            )
        md.append("")

    best_ndcg = max(results, key=lambda r: r["ndcg"])
    best_mrr = max(results, key=lambda r: r["mrr"])
    fastest = min(results, key=lambda r: r["avg_latency_ms"])
    md.append("## 4. Findings\n")
    md.append(
        f"1. **Best NDCG@5**: {best_ndcg['config_name']} "
        f"({best_ndcg['ndcg']:.3f}); best MRR@5: {best_mrr['config_name']} "
        f"({best_mrr['mrr']:.3f})."
    )
    md.append(
        f"2. **Latency cost**: reranking moves average query latency from "
        f"{fastest['avg_latency_ms']:.0f} ms to "
        f"{max(r['avg_latency_ms'] for r in results):.0f} ms."
    )
    md.append(
        "3. Cross-encoders score the query and passage jointly, so they help most "
        "where a bi-encoder cannot see the interaction between a question and a "
        "specific figure."
    )

    content = "\n".join(md)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(content, encoding="utf-8")
    Path("knowledge-graph/phase7_summary.md").write_text(content, encoding="utf-8")
    print(f"\n[ok] Report written to {out_path} and knowledge-graph/phase7_summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--store",
        default="data/vector_stores/phase6_tables_bge_base",
        help="FAISS index directory",
    )
    ap.add_argument("--k", type=int, default=5, help="final results to score")
    ap.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[20, 50],
        help="candidate depths to rerank from",
    )
    ap.add_argument(
        "--no-tables",
        action="store_true",
        help="build the corpus without table extraction (for ablation)",
    )
    ap.add_argument(
        "--legacy-xbrl",
        action="store_true",
        help=(
            "reproduce the pre-Phase-6 corpus: decompose ix:* tags instead of "
            "unwrapping them, so every filing number is stripped. Pair this with "
            "--store data/vector_stores/embeddings_bge_base to reproduce the "
            "Phase 5 baseline exactly."
        ),
    )
    args = ap.parse_args()

    if args.legacy_xbrl:
        from finrag.data import xbrl as xbrl_mod

        xbrl_mod.XBRL_UNWRAP_TAGS = []
        xbrl_mod.LEGACY_MODE = True
        print("!! legacy XBRL mode: ix:* tags decomposed, filing numbers stripped")

    eval_set = load_eval_dataset()
    print(f"Loaded {len(eval_set)} benchmark questions.")

    chunks = split_documents_with_strategy(
        load_all_filings(include_tables=not args.no_tables), strategy="recursive"
    )
    store = FAISS.load_local(
        args.store, get_embeddings(config.embedding), allow_dangerous_deserialization=True
    )
    print(f"Loaded index from {args.store} ({store.index.ntotal} vectors)")

    def build(fetch: int, rerank_top_n: int | None):
        dense = get_dense_retriever(store, k=fetch)
        bm25 = get_bm25_retriever(chunks, k=fetch)
        hybrid = get_ensemble_retriever(dense, bm25, bm25_weight=0.3, dense_weight=0.7)
        filtered = get_filtered_retriever(hybrid, auto_extract_filter=True, k=fetch)
        if rerank_top_n is None:
            return filtered
        compressor = get_reranker()
        compressor.top_n = rerank_top_n
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=filtered
        )

    configs = [("Filtered Hybrid (no rerank)", build(args.k, None))]
    for depth in args.candidates:
        configs.append(
            (
                f"Filtered Hybrid + rerank (top {depth} -> {args.k})",
                build(depth, args.k),
            )
        )

    results = [evaluate_config(name, ret, eval_set, args.k) for name, ret in configs]

    suffix = "_notables" if args.no_tables else ""
    json_path = f"eval/results/phase7_reranking_comparison{suffix}.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] Results saved to {json_path}")

    write_report(results, f"eval/results/phase7_summary{suffix}.md")


if __name__ == "__main__":
    main()
