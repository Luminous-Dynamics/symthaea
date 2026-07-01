#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Merge multiple JSONL training data files, normalizing to 24 channels.

Usage:
    python3 merge_training_data.py -o data/train-v5-combined.jsonl \
        data/train-v3-24ch.jsonl \
        data/train-v4-llm.jsonl \
        data/train-v4-llm-b2.jsonl

Handles:
- Files with different channel counts (pads to 24 with zeros)
- Deduplication by target_text
- Shuffling with deterministic seed
"""

import argparse
import json
import random
from pathlib import Path


def normalize_channels(channels: list, target: int = 24) -> list:
    """Pad or truncate channels to target length."""
    if len(channels) >= target:
        return channels[:target]
    return channels + [0.0] * (target - len(channels))


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL training data files")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle output")
    parser.add_argument("--no-dedup", action="store_true", help="Don't deduplicate")
    args = parser.parse_args()

    samples = []
    seen_texts = set()
    per_file = {}

    for path in args.inputs:
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text = d.get("target_text", "")

                # Deduplicate by text
                if not args.no_dedup and text in seen_texts:
                    continue
                seen_texts.add(text)

                # Normalize channels
                d["channels"] = normalize_channels(d["channels"])

                # Remove target_ids (will be re-tokenized by training pipeline)
                d.pop("target_ids", None)

                samples.append(d)
                count += 1
        per_file[path] = count
        print(f"  {path}: {count} samples")

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(samples)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nTotal: {len(samples)} samples → {output_path}")
    if not args.no_dedup:
        total_raw = sum(per_file.values())
        dupes = total_raw - len(samples)
        if dupes > 0:
            print(f"  ({dupes} duplicates removed)")


if __name__ == "__main__":
    main()
