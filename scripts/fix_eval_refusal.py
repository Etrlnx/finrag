#!/usr/bin/env python
"""Fix eval set: add expected_refusal: true to unanswerable and ambiguous questions."""

import json
from pathlib import Path

with open("eval/phase9_eval_set.json") as f:
    data = json.load(f)

# Categories that should have expected_refusal: true
refusal_categories = {"unanswerable", "ambiguous_contradictory"}

for item in data:
    if item["category"] in {"unanswerable", "ambiguous_contradictory"}:
        item["expected_refusal"] = True
    else:
        item["expected_refusal"] = False

output_path = Path("eval/phase9_eval_set.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

print(f"Updated {len(data)} questions")
cats = {}
for q in data:
    cats[q['category']] = cats.get(q['category'], 0) + 1
print('Categories:', cats)
refusal_count = sum(1 for q in data if q.get('expected_refusal'))
print(f"Questions with expected_refusal=true: {refusal_count}")