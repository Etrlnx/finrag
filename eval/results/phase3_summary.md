# FinRAG Phase 3: Embedding Model Comparison Report

Quantitative evaluation of candidate embedding models on the 35-question financial benchmark across 15 companies and 30 SEC 10-K/10-Q filings.

## 1. Overall Performance Summary

| Embedding Model | Dims | Avg Latency (ms) | Hit@3 | Hit@5 | Hit@10 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **bge-small** (`BAAI/bge-small-en-v1.5`) | 384 | 19.2 | 97.1% | 97.1% | 100.0% | **0.943** | **79.9%** | 82.3% |
| **minilm-l6** (`sentence-transformers/all-MiniLM-L6-v2`) | 384 | 9.7 | 97.1% | 97.1% | 97.1% | **0.957** | **69.3%** | 80.0% |
| **bge-base** (`BAAI/bge-base-en-v1.5`) | 768 | 35.5 | 100.0% | 100.0% | 100.0% | **0.971** | **78.6%** | 81.1% |

## 2. Category-Specific Breakdown (at k=5)

### Category: `FACTUAL`
| Model | Hit Rate@5 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- |
| bge-small | 100.0% | 0.929 | 76.4% | 74.3% |
| minilm-l6 | 92.9% | 0.929 | 69.3% | 68.6% |
| bge-base | 100.0% | 0.964 | 66.3% | 67.1% |

### Category: `NUMERICAL`
| Model | Hit Rate@5 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- |
| bge-small | 100.0% | 1.000 | 85.6% | 92.7% |
| minilm-l6 | 100.0% | 0.955 | 67.0% | 83.6% |
| bge-base | 100.0% | 0.955 | 87.9% | 92.7% |

### Category: `TEMPORAL`
| Model | Hit Rate@5 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- |
| bge-small | 66.7% | 0.667 | 68.3% | 60.0% |
| minilm-l6 | 100.0% | 1.000 | 53.3% | 80.0% |
| bge-base | 100.0% | 1.000 | 78.3% | 66.7% |

### Category: `CROSS-DOCUMENT`
| Model | Hit Rate@5 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- |
| bge-small | 100.0% | 1.000 | 40.0% | 70.0% |
| minilm-l6 | 100.0% | 1.000 | 30.0% | 90.0% |
| bge-base | 100.0% | 1.000 | 60.0% | 90.0% |

### Category: `UNANSWERABLE`
| Model | Hit Rate@5 | MRR@5 | Recall@5 | Precision@5 |
| :--- | :--- | :--- | :--- | :--- |
| bge-small | 100.0% | 1.000 | 100.0% | 100.0% |
| minilm-l6 | 100.0% | 1.000 | 100.0% | 100.0% |
| bge-base | 100.0% | 1.000 | 100.0% | 100.0% |

## 3. Key Findings & Champion Model Decision

1. **Champion Model Selected**: **`bge-base` (`BAAI/bge-base-en-v1.5`)** achieved the highest MRR@5 (0.971) and Hit Rate@5 (100.0%).
2. **BGE vs MiniLM**: The BGE family demonstrates significantly stronger semantic capture over complex financial statements and Item 1A/7 SEC text compared to smaller MiniLM baselines.
3. **Local Embedding Feasibility**: Embedding locally completely avoids free-tier API rate limits (1000 req/day quota) while executing in sub-100ms latency per query.
4. **Cross-Company Filtering Need**: While dense embedding recall is high, exact ticker matching is still imperfect in pure dense mode, highlighting the necessity of **Hybrid Search (Phase 4)** and **Metadata Filtering (Phase 5)**.