# FinRAG Phase 7: Cross-Encoder Reranking Benchmark

Compares the Phase 5 filtered-hybrid baseline against cross-encoder reranking over a wider candidate pool. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API key).

## 1. Overall (at k=5)

| Configuration | Latency (ms) | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Filtered Hybrid (no rerank)** | 77.6 | 97.8% | 0.919 | 77.1% | 82.2% | 0.847 |
| **Filtered Hybrid + rerank (top 20 -> 5)** | 349.3 | 97.8% | 0.978 | 79.7% | 84.4% | 0.872 |
| **Filtered Hybrid + rerank (top 50 -> 5)** | 745.2 | 100.0% | 0.953 | 79.7% | 82.7% | 0.886 |

## 2. Effect on table-dependent questions

| Configuration | Table Q recall | Table Q MRR | Table Q NDCG | Text Q recall |
| :--- | :---: | :---: | :---: | :---: |
| **Filtered Hybrid (no rerank)** | 55.0% | 0.833 | 0.785 | 83.4% |
| **Filtered Hybrid + rerank (top 20 -> 5)** | 56.7% | 1.000 | 0.891 | 86.3% |
| **Filtered Hybrid + rerank (top 50 -> 5)** | 61.7% | 0.933 | 0.890 | 84.8% |

## 3. Category breakdown

### `FACTUAL`
| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| Filtered Hybrid (no rerank) | 92.9% | 0.893 | 78.2% | 0.827 |
| Filtered Hybrid + rerank (top 20 -> 5) | 92.9% | 0.929 | 81.4% | 0.858 |
| Filtered Hybrid + rerank (top 50 -> 5) | 100.0% | 0.943 | 81.4% | 0.896 |

### `NUMERICAL`
| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| Filtered Hybrid (no rerank) | 100.0% | 0.921 | 71.4% | 0.817 |
| Filtered Hybrid + rerank (top 20 -> 5) | 100.0% | 1.000 | 73.0% | 0.871 |
| Filtered Hybrid + rerank (top 50 -> 5) | 100.0% | 0.937 | 73.8% | 0.870 |

### `TEMPORAL`
| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| Filtered Hybrid (no rerank) | 100.0% | 1.000 | 85.0% | 0.963 |
| Filtered Hybrid + rerank (top 20 -> 5) | 100.0% | 1.000 | 85.0% | 0.806 |
| Filtered Hybrid + rerank (top 50 -> 5) | 100.0% | 1.000 | 85.0% | 0.911 |

### `CROSS-DOCUMENT`
| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| Filtered Hybrid (no rerank) | 100.0% | 0.750 | 60.0% | 0.751 |
| Filtered Hybrid + rerank (top 20 -> 5) | 100.0% | 1.000 | 80.0% | 0.755 |
| Filtered Hybrid + rerank (top 50 -> 5) | 100.0% | 1.000 | 70.0% | 0.655 |

### `UNANSWERABLE`
| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| Filtered Hybrid (no rerank) | 100.0% | 1.000 | 100.0% | 1.000 |
| Filtered Hybrid + rerank (top 20 -> 5) | 100.0% | 1.000 | 100.0% | 1.000 |
| Filtered Hybrid + rerank (top 50 -> 5) | 100.0% | 1.000 | 100.0% | 1.000 |

## 4. Findings

1. **Best NDCG@5**: Filtered Hybrid + rerank (top 50 -> 5) (0.886); best MRR@5: Filtered Hybrid + rerank (top 20 -> 5) (0.978).
2. **Latency cost**: reranking moves average query latency from 78 ms to 745 ms.
3. Cross-encoders score the query and passage jointly, so they help most where a bi-encoder cannot see the interaction between a question and a specific figure.