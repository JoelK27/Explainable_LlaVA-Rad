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

import json
from pathlib import Path

src = Path("/workspace/thesis/data/refs/./physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")
dst = Path("/workspace/thesis/data/queries/subset_1500.json")

data = json.loads(src.read_text(encoding="utf-8"))

if isinstance(data, list):
    subset = data[:1500]
elif isinstance(data, dict):
    if "data" in data and isinstance(data["data"], list):
        subset = data["data"][:1500]
    else:
        raise ValueError("JSON ist ein dict, aber keine Liste unter 'data' gefunden.")
else:
    raise ValueError("Unerwartete JSON-Struktur.")

dst.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved_to =", dst)
print("subset_len =", len(subset))
