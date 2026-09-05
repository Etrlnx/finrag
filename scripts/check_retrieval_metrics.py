import json

with open("eval/results/retrieval_metrics.json") as f:
    data = json.load(f)

rerank = data['with_reranker']
no_rerank = data['without_reranker']

def avg_metric(results, key):
    return sum(r['k_metrics']['k_5'][key] for r in results) / len(results)

print('=== WITH RERANKER ===')
print(f'Hit@5: {avg_metric(rerank, "hit")*100:.1f}%')
print(f'MRR@5: {avg_metric(rerank, "reciprocal_rank"):.3f}')
print(f'Precision@5: {avg_metric(rerank, "precision")*100:.1f}%')
print(f'Recall@5: {avg_metric(rerank, "recall")*100:.1f}%')
print(f'NDCG@5: {avg_metric(rerank, "ndcg")*100:.1f}%')
print(f'Contamination: {avg_metric(rerank, "contamination")*100:.1f}%')

print()
print('=== WITHOUT RERANKER ===')
print(f'Hit@5: {avg_metric(no_rerank, "hit")*100:.1f}%')
print(f'MRR@5: {avg_metric(no_rerank, "reciprocal_rank"):.3f}')
print(f'Precision@5: {avg_metric(no_rerank, "precision")*100:.1f}%')
print(f'Recall@5: {avg_metric(no_rerank, "recall")*100:.1f}%')
print(f'NDCG@5: {avg_metric(no_rerank, "ndcg")*100:.1f}%')
print(f'Contamination: {avg_metric(no_rerank, "contamination")*100:.1f}%')