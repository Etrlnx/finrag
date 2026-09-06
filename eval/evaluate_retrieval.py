"""Phase 9: Retrieval Evaluation on evaluation set."""

import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.config import config
from finrag.data import load_all_filings, split_documents_with_strategy
from finrag.embeddings import get_embeddings
from finrag.retrieval import (
    get_bm25_retriever,
    get_dense_retriever,
    get_ensemble_retriever,
    get_filtered_retriever,
    get_reranker,
    build_retrieval_pipeline,
)

warnings.filterwarnings("ignore")


def load_eval_dataset(path: str = "eval/eval_set.json") -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relevance(doc: Document, item: Dict[str, Any], is_cross_doc: bool,
               keywords: List[str], sections: List[str]) -> float:
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
    import math
    def dcg(scores: List[float]) -> float:
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))

    actual = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    item: Dict[str, Any],
    retrieved: List[Document],
    k_values: List[int]
) -> Dict[str, Any]:
    category = item.get("category", "factual")
    target_ticker = item.get("ticker", "")
    target_sections = item.get("target_sections", [])
    ground_truth_keywords = [kw.lower() for kw in item.get("ground_truth_keywords", [])]
    is_cross_doc = target_ticker in ["TECH", "BANKS", "AUTO", "ENERGY", "RETAIL"]

    results = {}
    for k in k_values:
        top_k = retrieved[:k]
        rels = [_relevance(d, item, False, [kw.lower() for kw in item.get("ground_truth_keywords", [])], target_sections) for d in top_k]

        relevant_ranks = [i + 1 for i, r in enumerate(rels) if r > 0]
        hit = len(relevant_ranks) > 0
        reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        precision = sum(1 for r in rels if r > 0) / k if k else 0.0
        recall = len([r for r in rels if r > 0]) / max(1, len([kw for kw in item.get("ground_truth_keywords", [])]))
        ndcg = ndcg_at_k(rels, k)

        contaminated = 0
        if not target_ticker in ["TECH", "BANKS"]:
            contaminated = sum(1 for d in top_k if (d.metadata.get("ticker") or "").upper() != (item.get("ticker") or "").upper())

        results[f"k_{k}"] = {
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "contamination": contaminated / k if k else 0.0,
        }

    return {
        "id": item["id"],
        "category": item.get("category"),
        "ticker": item.get("ticker"),
        "question": item["question"],
        "k_metrics": results,
    }


def run_retrieval_evaluation(
    eval_set: List[Dict],
    store: FAISS,
    chunks: List[Document],
    cfg,
    use_reranker: bool = True,
    use_filtering: bool = True,
    k_values: List[int] = [3, 5, 10]
) -> List[Dict]:
    retriever = build_retrieval_pipeline(
        documents=chunks,
        dense_retriever=store.as_retriever(search_kwargs={"k": 20}),
        cfg=cfg.retrieval,
        use_bm25=True,
        use_reranker=use_reranker,
        use_filtering=use_filtering,
    )

    results = []
    for item in eval_set:
        q = item["question"]
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        latency = (time.perf_counter() - t0) * 1000

        res = evaluate_retrieval(item, docs, [3, 5, 10])
        res["latency_ms"] = round(latency, 2)
        res["retrieved_count"] = len(docs)
        results.append(res)

    return results


def aggregate_metrics(results: List[Dict], k_values: List[int]) -> Dict[str, Any]:
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {k: {"hit": [], "mrr": [], "precision": [], "recall": [], "ndcg": [], "contamination": []} for k in [3, 5, 10]}

        for k in k_values:
            m = r["k_metrics"][f"k_{k}"]
            by_cat[cat][k]["hit"].append(1.0 if m["hit"] else 0.0)
            by_cat[cat][k]["mrr"].append(m["reciprocal_rank"])
            by_cat[cat][k]["precision"].append(m["precision"])
            by_cat[cat][k]["recall"].append(m["recall"])
            by_cat[cat][k]["ndcg"].append(m["ndcg"])
            by_cat[cat][k]["contamination"].append(m["contamination"])

    return {cat: {k: {m: sum(v) / len(v) if v else 0.0 for m, v in d.items()} for k, d in d_items.items()} for cat, d_items in by_cat.items()}


def main():
    eval_set = json.loads(Path("eval/eval_set.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(eval_set)} evaluation questions.")

    chunks = split_documents_with_strategy(
        load_all_filings(include_tables=True), strategy="recursive"
    )
    print(f"Loaded {len(chunks)} chunks.")

    emb = get_embeddings(config.embedding)
    store = FAISS.load_local("data/vector_stores/phase6_table_aware", emb, allow_dangerous_deserialization=True)
    print(f"Loaded index: {store.index.ntotal} vectors")

    print("\n=== Retrieval Evaluation (with reranker) ===")
    results_rerank = run_retrieval_evaluation(eval_set, store, chunks, config, use_reranker=True, use_filtering=True)

    print("\n=== Retrieval Evaluation (no reranker) ===")
    results_no_rerank = run_retrieval_evaluation(eval_set, store, chunks, config, use_reranker=False, use_filtering=True)

    # Aggregate and print metrics
    for name, results in [("With Reranker", results_rerank), ("Without Reranker", results_no_rerank)]:
        agg = aggregate_metrics(results, [3, 5, 10])
        print(f"\n=== {name} ===")
        for k in [3, 5, 10]:
            # aggregate_metrics returns {category: {k: metrics}} with already-aggregated means
            # Aggregate across all categories (weighted by number of questions per category)
            total_questions = sum(len([r for r in results if r["category"] == cat]) for cat in agg)
            hit = sum(agg[cat][k]["hit"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            mrr = sum(agg[cat][k]["mrr"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            precision = sum(agg[cat][k]["precision"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            recall = sum(agg[cat][k]["recall"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            ndcg = sum(agg[cat][k]["ndcg"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            contamination = sum(agg[cat][k]["contamination"] * len([r for r in results if r["category"] == cat]) for cat in agg) / total_questions
            print(f"  k={k}: Hit@k={hit*100:.1f}%, MRR={mrr:.3f}, Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, NDCG={ndcg:.3f}, Contamination={contamination*100:.1f}%")
        print("  By category:")
        for cat, metrics in agg.items():
            print(f"    {cat}: " + ", ".join(f"k={k}: hit={v['hit']*100:.1f}%, mrr={v['mrr']:.3f}, prec={v['precision']*100:.1f}%, rec={v['recall']*100:.1f}%, ndcg={v['ndcg']:.3f}, contam={v['contamination']*100:.1f}%" for k, v in metrics.items()))

    output = {
        "with_reranker": results_rerank,
        "without_reranker": results_no_rerank,
        "config": {
            "corpus": "phase6_table_aware",
            "embedding": "BAAI/bge-base-en-v1.5",
            "hybrid_weights": {"dense": 0.7, "bm25": 0.3},
            "filtering": True,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }
    }

    Path("eval/results/retrieval_metrics.json").parent.mkdir(parents=True, exist_ok=True)
    Path("eval/results/retrieval_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("[ok] Results saved to eval/results/retrieval_metrics.json")


if __name__ == "__main__":
    main()