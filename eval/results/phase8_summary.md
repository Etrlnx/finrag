# FinRAG Phase 8: Grounded Generation Evaluation Summary

## Executive Summary

Completed full 20-question generation evaluation using Ollama (llama3.2) with 4 configuration comparisons.

## Configuration Comparison

| Configuration | Accuracy | Format % | Citation % | Grounded % | Refusal % | Latency |
|--------------|----------|----------|------------|------------|-----------|---------|
| **Hardened + Rerank** | **45%** (9/20) | 70% | 85% | 90% | 80% | 5398ms |
| Hardened + No Rerank | 40% | 65% | 85% | 90% | 80% | 7061ms |
| Original + Rerank | 0% | 0% | 55% | 90% | 85% | 5401ms |
| Original + No Rerank | 0% | 0% | 20% | 100% | 55% | 5837ms |

## Key Findings

### 1. Hardened Prompt Significantly Outperforms Original
- **45% vs 0% accuracy** — the hardened prompt with per-claim citations, mandatory refusal, and structured output format is essential
- Original prompt fails format compliance completely (0% format adherence)

### 2. Reranker Provides Modest Improvement
- **+5% accuracy** (45% vs 40%) with reranker
- **+5% format compliance** (70% vs 65%)
- Latency increase: +1663ms (reranked vs non-reranked)

### 3. Category Breakdown (Hardened + Rerank)

| Category | Accuracy | Questions |
|----------|----------|-----------|
| temporal | 100% | 2 |
| cross_document | 100% | 1 |
| supported_numerical | 60% | 5 |
| supported_factual | 50% | 2 |
| supported_table_dependent | 50% | 4 |
| unanswerable_future | 0% | 1 |
| unanswerable_absent_company | 0% | 1 |
| unanswerable_unsupported_causal | 0% | 1 |
| unanswerable_outside_corpus | 0% | 1 |
| unanswerable_specific_metric | 0% | 1 |
| ambiguous_contradictory | 0% | 1 |

### 4. Refusal Performance
- **80% correct** on unanswerable cases (4/5 correct refusals)
- **Over-refusal issue**: Model refuses on some supported questions (particularly table-dependent)
- False refusal rate: ~20% on supported questions

### 4. Citation & Grounding Quality
- **85% citation format validity** (when citations present)
- **90% grounding accuracy** (citations match retrieved docs)
- **Format compliance**: 70% (needs improvement on Answer:/Evidence: structure)

## Root Causes of Failures

1. **Format non-compliance** — Model doesn't consistently use `Answer:` / `Evidence:` structure
2. **Over-refusal** — Model refuses on answerable questions, especially table-dependent
3. **Citation omission** — Many answers lack proper `[TICKER, FORM, DATE, SECTION]` citations
4. **Retrieval gaps** — Some questions retrieve irrelevant or wrong-company chunks

## Production Recommendation

**Use: Hardened Prompt + Reranker (top-20→5)**
- Best overall accuracy: 45%
- Best format compliance: 70%
- Acceptable latency: ~5.4s per query
- Reranker worth the +1.6s latency cost

## Files

- Full results: `eval/results/phase8_generation_comparison.json`
- Configuration tested: 4 configurations × 20 questions = 80 total generations
- Model: Ollama llama3.2 (local, unlimited quota)
- Corpus: Phase 6 table-aware (16,626 chunks, 4,678 tables)
- Retrieval: Filtered Hybrid (Dense 0.7 / BM25 0.3) + Reranker top-20→5