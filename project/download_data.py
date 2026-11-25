"""Download the BANKING77 dataset by PolyAI (CC BY 4.0).

Fetches train.csv (10,003 rows) and test.csv (3,080 rows) with columns
(text, category) from PolyAI's official GitHub dataset repository, with
Hugging Face Datasets Server metadata+rows as fallback.
"""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

POLYAI_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}

HF_INFO_URL = "https://datasets-server.huggingface.co/info?dataset=PolyAI%2Fbanking77"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=PolyAI%2Fbanking77&config=default&split={split}&offset={offset}&length={length}"


def _download_from_polyai_github(split: str, dest_path: str) -> int:
    """Download official train.csv / test.csv directly from PolyAI-LDN GitHub."""
    url = POLYAI_URLS[split]
    req = urllib.request.Request(url, headers={"User-Agent": "alfred-ai-assignment/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_text = resp.read().decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(raw_text)))
    if not rows or "text" not in rows[0] or "category" not in rows[0]:
        raise ValueError(f"Unexpected CSV schema from {url}")

    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"text": r["text"], "category": r["category"]})
    return len(rows)


def _download_from_hf(split: str, dest_path: str) -> int:
    """Fallback downloader using Hugging Face Datasets Server with live label names."""
    req_info = urllib.request.Request(HF_INFO_URL, headers={"User-Agent": "alfred-ai-assignment/1.0"})
    with urllib.request.urlopen(req_info, timeout=30) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    label_names = info["dataset_info"]["default"]["features"]["label"]["names"]
    if len(label_names) != 77:
        raise ValueError(f"Expected 77 BANKING77 labels, got {len(label_names)}")

    rows: list[dict[str, str]] = []
    offset = 0
    length = 100
    while True:
        url = HF_ROWS_URL.format(split=split, offset=offset, length=length)
        req = urllib.request.Request(url, headers={"User-Agent": "alfred-ai-assignment/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item["row"]
            rows.append({"text": row["text"], "category": label_names[int(row["label"])]})
        offset += len(batch)
        if len(batch) < length:
            break

    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def download_data(force: bool = False) -> None:
    """Download BANKING77 train.csv and test.csv into project/data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train", "test"):
        dest = os.path.join(DATA_DIR, f"{split}.csv")
        if os.path.exists(dest) and not force:
            print(f"{split}.csv already exists at {dest}, skipping.")
            continue
        print(f"Downloading {split}.csv from PolyAI-LDN GitHub...")
        try:
            n = _download_from_polyai_github(split, dest)
        except Exception as exc:
            print(f"  GitHub failed ({exc}); falling back to Hugging Face Datasets Server...")
            n = _download_from_hf(split, dest)
        print(f"  Saved {n} rows to {dest}")

    for split in ("train", "test"):
        dest = os.path.join(DATA_DIR, f"{split}.csv")
        with open(dest, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cats = {r["category"] for r in rows}
        print(f"Verified {split}.csv: {len(rows)} rows, {len(cats)} unique categories")


if __name__ == "__main__":
    download_data()
