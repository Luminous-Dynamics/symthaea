#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Download the real Social Chemistry 101 dataset (292K examples) for moral prototype training.

Three fallback strategies:
1. HuggingFace `datasets` library -> allenai/social-chem-101
2. Direct TSV download from Google Cloud Storage
3. Keep existing synthetic examples (no-op)

Output: data/moral_datasets/social_chemistry_292k.json
Format: {"metadata": {...}, "examples": [{"action": "...", "rot": "...", "rot_judgment": "-1"|"0"|"1", "split": "train"|"test"}]}

Outputs a balanced subset of up to 50K examples (<=16,667 per class).
"""

import json
import os
import random
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "moral_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "social_chemistry_292k.json"

MAX_PER_CLASS = 16_667  # Balanced subset: 50K total max


def balance_examples(examples):
    """Balance examples across 3 classes (Good/Neutral/Bad)."""
    by_class = {"-1": [], "0": [], "1": []}
    for ex in examples:
        j = ex.get("rot_judgment", "0")
        if j in by_class:
            by_class[j].append(ex)

    random.seed(42)
    balanced = []
    for label, items in by_class.items():
        random.shuffle(items)
        balanced.extend(items[:MAX_PER_CLASS])
        print(f"  Class {label:>2}: {len(items)} total -> {min(len(items), MAX_PER_CLASS)} sampled")

    random.shuffle(balanced)
    return balanced


def try_huggingface():
    """Strategy 1: Load from HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  HuggingFace datasets library not installed.")
        print("  Install with: pip install datasets")
        return None

    print("Strategy 1: Loading from HuggingFace...")
    try:
        # Try tasksource mirror first (original allenai/ requires loading script)
        try:
            ds = load_dataset("tasksource/social-chemestry-101")
            print("  Using tasksource/social-chemestry-101 mirror")
        except Exception:
            ds = load_dataset("allenai/social-chem-101")
        examples = []

        for split_name in ["train", "test", "validation"]:
            if split_name not in ds:
                continue
            split = ds[split_name]
            for row in split:
                # Use action-moral-judgment (numeric: -1, 0, 1) as primary signal
                # Fall back to rot-judgment (text: "it's bad", "it's ok", "it's good")
                amj = row.get("action-moral-judgment")
                rot_judgment_text = str(row.get("rot-judgment", "")).strip().lower()

                if amj is not None and amj == amj:  # not NaN
                    val = float(amj)
                    if val > 0.5:
                        rot_judgment = "1"
                    elif val < -0.5:
                        rot_judgment = "-1"
                    else:
                        rot_judgment = "0"
                elif "bad" in rot_judgment_text or "wrong" in rot_judgment_text:
                    rot_judgment = "-1"
                elif "good" in rot_judgment_text or "expected" in rot_judgment_text:
                    rot_judgment = "1"
                elif "ok" in rot_judgment_text or "fine" in rot_judgment_text:
                    rot_judgment = "0"
                else:
                    rot_judgment = "0"

                action = str(row.get("action", "")).strip()
                rot = str(row.get("rot", "")).strip()

                if not rot and not action:
                    continue

                examples.append({
                    "action": action,
                    "rot": rot,
                    "rot_judgment": rot_judgment,
                    "split": str(row.get("split", "train")).strip(),
                })

        print(f"  Loaded {len(examples)} examples from HuggingFace.")
        return examples

    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        return None


def try_direct_download():
    """Strategy 2: Direct ZIP download from Google Cloud Storage."""
    import urllib.request
    import zipfile
    import csv
    import io

    url = "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip"
    print(f"Strategy 2: Direct download from {url}...")

    try:
        print("  Downloading (this may take a minute)...")
        response = urllib.request.urlopen(url, timeout=300)
        zip_bytes = response.read()
        print(f"  Downloaded {len(zip_bytes) / 1024 / 1024:.1f} MB")

        # Extract TSV from zip
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
        print(f"  Extracting {tsv_name}...")
        tsv_bytes = zf.read(tsv_name)
        reader = csv.DictReader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t")

        examples = []
        for row in reader:
            rot_judgment = row.get("rot-judgment", "0").strip()
            try:
                val = float(rot_judgment)
                if val > 0.5:
                    rot_judgment = "1"
                elif val < -0.5:
                    rot_judgment = "-1"
                else:
                    rot_judgment = "0"
            except (ValueError, TypeError):
                rot_judgment = "0"

            action = row.get("action", "").strip()
            rot = row.get("rot", "").strip()

            if not rot and not action:
                continue

            split_name = row.get("split", "train").strip()

            examples.append({
                "action": action,
                "rot": rot,
                "rot_judgment": rot_judgment,
                "split": "test" if split_name == "test" else "train",
            })

        print(f"  Parsed {len(examples)} examples from TSV.")
        return examples

    except Exception as e:
        print(f"  Direct download failed: {e}")
        return None


def main():
    print("=" * 60)
    print("Social Chemistry 101 Dataset Downloader")
    print("=" * 60)

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
        print(f"\nDataset already exists at {OUTPUT_FILE} ({size_mb:.1f} MB)")
        print("Delete the file to re-download.")
        return

    # Try strategies in order
    examples = try_huggingface()
    if examples is None:
        examples = try_direct_download()

    if examples is None:
        print("\nAll download strategies failed.")
        print("The benchmark will fall back to the existing synthetic examples.")
        print("To download manually, visit: https://github.com/mbforbes/social-chemistry-101")
        sys.exit(1)

    # Balance classes
    print(f"\nBalancing {len(examples)} examples across 3 classes...")
    balanced = balance_examples(examples)

    # Save
    output = {
        "metadata": {
            "source": "allenai/social-chem-101",
            "url": "https://github.com/mbforbes/social-chemistry-101",
            "description": "Social Chemistry 101: 292K rules-of-thumb for social norms (balanced subset)",
            "total_raw": len(examples),
            "total_balanced": len(balanced),
        },
        "examples": balanced,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(balanced)} balanced examples to {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
