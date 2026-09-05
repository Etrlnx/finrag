import argparse
from finrag.pipeline import load_baseline_pipeline, load_hybrid_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--hybrid", action="store_true", help="Run in hybrid (Dense + BM25) mode")
args = parser.parse_args()

if args.hybrid:
    print("Loading Hybrid Pipeline (bge-base Dense + BM25 Sparse)...")
    pipeline = load_hybrid_pipeline()
else:
    print("Loading Dense-only Pipeline...")
    pipeline = load_baseline_pipeline()
print("Pipeline loaded successfully")

test_questions = [
    "What was Apple's revenue in Q3 2026?",
    "What are Microsoft's main risk factors?",
    "How much did NVIDIA spend on R&D in 2025?",
    "What is Amazon's operating income for 2025?",
    "What are Google's key financial highlights?",
]

for q in test_questions:
    print(f"\nQuestion: {q}")
    answer = pipeline.query(q)
    print(f"Answer: {answer[:800]}...")
    print("=" * 80)