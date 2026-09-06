#!/usr/bin/env python
"""CLI tool for inspecting FinRAG explainable results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.pipeline import load_production_pipeline


def cmd_inspect(args):
    """Run a query and show full explainable trace."""
    print("Loading production pipeline...")
    pipeline = load_production_pipeline()
    print("Pipeline loaded.\n")

    print(f"Question: {args.question}")
    print("Generating explainable result...")
    result = pipeline.query_explainable(args.question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.to_string(verbose=not args.compact))


def cmd_batch(args):
    """Run multiple questions from a file."""
    print("Loading production pipeline...")
    pipeline = load_production_pipeline()
    print("Pipeline loaded.\n")

    questions = []
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    questions.append(line)
    else:
        questions = args.questions

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}: {q}")
        print(f"{'='*80}")
        result = pipeline.query_explainable(q)
        print(result.to_string(verbose=not args.compact))


def main():
    parser = argparse.ArgumentParser(description="FinRAG Explainability Inspector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a single question")
    inspect_parser.add_argument("question", help="Question to ask")
    inspect_parser.add_argument("--json", action="store_true", help="Output JSON")
    inspect_parser.add_argument("--compact", action="store_true", help="Compact output")
    inspect_parser.set_defaults(func=cmd_inspect)

    batch_parser = subparsers.add_parser("batch", help="Inspect multiple questions")
    batch_parser.add_argument("--file", help="File with questions (one per line)")
    batch_parser.add_argument("questions", nargs="*", help="Questions to ask")
    batch_parser.add_argument("--compact", action="store_true", help="Compact output")
    batch_parser.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()