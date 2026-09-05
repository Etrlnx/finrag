#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_api_key() -> bool:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PLACEHOLDER_REPLACE_WITH_REAL_KEY":
        print("ERROR: GEMINI_API_KEY not set in .env")
        print("Get a free API key from https://aistudio.google.com/app/apikey")
        return False
    return True


def get_embedding_model_arg(args):
    if args.embedding_model and args.embedding_model != "auto":
        return args.embedding_model
    return None


def cmd_build(args):
    if not check_api_key():
        sys.exit(1)

    from finrag.pipeline import build_baseline_pipeline
    build_baseline_pipeline()


def cmd_query(args):
    if not check_api_key():
        sys.exit(1)

    from finrag.pipeline import load_baseline_pipeline
    pipeline = load_baseline_pipeline()

    if args.question:
        print(f"\nQuestion: {args.question}")
        answer = pipeline.query(args.question)
        print(f"Answer:\n{answer}")
    else:
        print("Interactive mode (Ctrl+C to exit)")
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q:
                    print(f"Question: {q}")
                    answer = pipeline.query(q)
                    print(f"Answer:\n{answer}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break


def cmd_test(args):
    if not check_api_key():
        sys.exit(1)

    from finrag.pipeline import load_baseline_pipeline
    pipeline = load_baseline_pipeline()

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
        print(f"Answer:\n{answer}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="FinRAG - Financial Document Intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build baseline vector store and index")
    build_parser.set_defaults(func=cmd_build)

    test_parser = subparsers.add_parser("test", help="Run test queries on loaded index")
    test_parser.set_defaults(func=cmd_test)

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", nargs="?", help="Question to ask (interactive if omitted)")
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()