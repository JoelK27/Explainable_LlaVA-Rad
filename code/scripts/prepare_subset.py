"""
Data Preparation Component
--------------------------
Purpose: 
    Constructs the formalized evaluation subsets from the MIMIC-CXR/LLaVA-Rad 
    metadata. This script ensures that case identifiers, study references, and 
    image paths are preserved for later downstream linkage.

Input: 
    Raw MIMIC-CXR-JPG metadata and text annotations.
Output: 
    Structured query arrays (e.g., subset_600.json, subset_1500.json) containing 
    the necessary prompts and study-level ground truths.
"""

import argparse
import json
import os
import random
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def get_image_name(row):
    return row.get("image") or row.get("image_name") or row.get("img") or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["first", "random"], default="random")
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    valid = []

    for row in rows:
        image_name = get_image_name(row)
        image_path = os.path.join(args.image_folder, image_name)
        if image_name and os.path.exists(image_path):
            valid.append(row)

    if args.mode == "random":
        random.seed(args.seed)
        random.shuffle(valid)

    subset = valid[:args.n]
    write_jsonl(subset, args.output_jsonl)

    print(f"input_rows={len(rows)}")
    print(f"valid_rows={len(valid)}")
    print(f"written_rows={len(subset)}")
    print(f"output={args.output_jsonl}")

if __name__ == "__main__":
    main()
