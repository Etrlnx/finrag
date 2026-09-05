"""Add Phase 6 table-dependent questions to the benchmark.

Every figure below was read directly out of the extracted HTML tables (not from
memory), using the same table_extractor that Phase 6 introduces.

Run:  python eval/add_table_questions.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

EVAL_PATH = Path("eval/eval_set.json")

# Ground truth verified against data/raw filings:
#   AAPL 10-K  filed 2025-10-31 (FY2025): net sales 416,161 / 391,035 / 383,285
#   AAPL 10-Q  filed 2026-07-31 (Q3 FY26): products 78,678, services 30,739,
#              total 109,417, net income 29,789, cash 39,544, inventories 11,092
#   NVDA 10-Q  filed 2026-08-26: net income 59,688 / 118,010, R&D 7,054
#   MSFT 10-Q  filed 2026-04-29: net income 31,778 / 97,983
#   WMT  10-K  filed 2026-03-13: revenues 713,163, assets 284,668
NEW_QUESTIONS = [
    {
        "id": "eval_36",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-K",
        "question": "In Apple's consolidated statements of operations, what were total net sales for fiscal 2025 versus fiscal 2024?",
        "target_sections": ["Item 8", "Item 7"],
        "ground_truth_keywords": ["416,161", "391,035", "Total net sales"],
        "expected_answer": "Total net sales were $416,161 million in fiscal 2025 versus $391,035 million in fiscal 2024.",
        "requires_table": True,
    },
    {
        "id": "eval_37",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-K",
        "question": "What were Apple's total assets at the end of fiscal 2025 according to the consolidated balance sheet?",
        "target_sections": ["Item 8"],
        "ground_truth_keywords": ["359,241", "Total assets"],
        "expected_answer": "Total assets were $359,241 million at September 27, 2025.",
        "requires_table": True,
    },
    {
        "id": "eval_38",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-K",
        "question": "What were Apple's total liabilities at the end of fiscal 2025?",
        "target_sections": ["Item 8"],
        "ground_truth_keywords": ["285,508", "Total liabilities"],
        "expected_answer": "Total liabilities were $285,508 million at the end of fiscal 2025.",
        "requires_table": True,
    },
    {
        "id": "eval_39",
        "category": "numerical",
        "ticker": "NVDA",
        "form": "10-Q",
        "question": "What was NVIDIA's net income for the three months ended in the most recent quarter reported in this 10-Q?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["59,688", "Net income"],
        "expected_answer": "Net income was $59,688 million for the three months ended in the most recent quarter.",
        "requires_table": True,
    },
    {
        "id": "eval_40",
        "category": "numerical",
        "ticker": "NVDA",
        "form": "10-Q",
        "question": "How much did NVIDIA spend on research and development in the most recent quarter versus the prior-year quarter?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["7,054", "4,291", "Research and development"],
        "expected_answer": "Research and development expense was $7,054 million versus $4,291 million in the prior-year quarter.",
        "requires_table": True,
    },
    {
        "id": "eval_41",
        "category": "numerical",
        "ticker": "MSFT",
        "form": "10-Q",
        "question": "What was Microsoft's net income for the three months ended March 31, 2026?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["31,778", "Net income"],
        "expected_answer": "Net income was $31,778 million for the three months ended March 31, 2026.",
        "requires_table": True,
    },
    {
        "id": "eval_42",
        "category": "numerical",
        "ticker": "WMT",
        "form": "10-K",
        "question": "What were Walmart's total revenues in the most recent fiscal year, and what were they the year before?",
        "target_sections": ["Item 8", "Item 7", "Item 1A"],
        "ground_truth_keywords": ["713,163", "680,985", "Total revenues"],
        "expected_answer": "Total revenues were $713,163 million in the most recent fiscal year versus $680,985 million in the prior year.",
        "requires_table": True,
    },
    {
        "id": "eval_43",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-Q",
        "question": "In the most recent quarter, how did Apple's Products net sales compare to Services net sales?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["78,678", "30,739", "Products", "Services"],
        "expected_answer": "Products net sales were $78,678 million and Services net sales were $30,739 million.",
        "requires_table": True,
    },
    {
        "id": "eval_44",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-Q",
        "question": "How much cash and cash equivalents did Apple hold at the end of the most recent quarter?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["39,544", "Cash and cash equivalents"],
        "expected_answer": "Cash and cash equivalents were $39,544 million.",
        "requires_table": True,
    },
    {
        "id": "eval_45",
        "category": "numerical",
        "ticker": "AAPL",
        "form": "10-Q",
        "question": "What was the value of Apple's inventories at the end of the most recent quarter?",
        "target_sections": ["Item 1"],
        "ground_truth_keywords": ["11,092", "Inventories"],
        "expected_answer": "Inventories were $11,092 million at the end of the most recent quarter.",
        "requires_table": True,
    },
]


def main() -> None:
    existing = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    have = {q["id"] for q in existing}
    to_add = [q for q in NEW_QUESTIONS if q["id"] not in have]

    if not to_add:
        print("All table questions already present; nothing to do.")
        return

    backup = EVAL_PATH.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy(EVAL_PATH, backup)
        print(f"Backup written to {backup}")

    existing.extend(to_add)
    EVAL_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    n_table = sum(1 for q in existing if q.get("requires_table"))
    print(f"Added {len(to_add)} questions. Benchmark now has {len(existing)} "
          f"({n_table} table-dependent).")


if __name__ == "__main__":
    main()
