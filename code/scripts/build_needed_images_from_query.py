"""
Image Path Resolution Component
-------------------------------
Purpose: 
    Parses the generated JSON subsets to isolate the specific chest X-ray 
    image paths required for LLaVA-Rad inference. 

Methodology:
    Ensures that the computationally expensive inference stage only runs on 
    the targeted cohort (e.g., 1500 cases), reducing API/GPU overhead while 
    maintaining strict identifier coherence.
"""

import json
import re
from pathlib import Path

query_file = Path("/workspace/thesis/data/queries/subset_1500.json")
out_file = Path("/workspace/thesis/data/lists/needed_images_1500.txt")

pattern = re.compile(r"(p\d{2}/p\d{8}/s\d+/[A-Za-z0-9._-]+\.(jpg|jpeg))", re.I)
found = set()

def walk(x):
    if isinstance(x, dict):
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
    elif isinstance(x, str):
        m = pattern.search(x.strip())
        if m:
            found.add("files/" + m.group(1))

text = query_file.read_text(encoding="utf-8").strip()

if text.startswith("[") or text.startswith("{"):
    data = json.loads(text)
    walk(data)
else:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        walk(json.loads(line))

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text("\n".join(sorted(found)) + ("\n" if found else ""), encoding="utf-8")

print("unique_images =", len(found))
print("saved_to =", out_file)