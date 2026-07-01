#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Pre-compute sentence embeddings for ETHICS benchmark dataset.

This script uses sentence-transformers to generate semantic embeddings
for all ethics scenarios, saving them in a format that Rust can load.

Usage:
    python scripts/precompute_ethics_embeddings.py

Output:
    datasets/ethics/embeddings/ethics_embeddings.json
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple
import csv

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Will use fallback.")

# Configuration
ETHICS_BASE = Path("datasets/ethics/raw/ethics")
OUTPUT_DIR = Path("datasets/ethics/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality

# Category configurations
CATEGORIES = {
    "justice": {"prefix": "justice", "format": "label,scenario"},
    "deontology": {"prefix": "deontology", "format": "label,scenario"},
    "virtue": {"prefix": "virtue", "format": "label,scenario"},
    "utilitarianism": {"prefix": "util", "format": "comparison"},
    "commonsense": {"prefix": "cm", "format": "label,input,is_short,edited"},
}


def hash_text(text: str) -> int:
    """Hash text for cache lookup (same algorithm as Rust)."""
    normalized = text.lower().strip()
    # Use first 8 bytes of MD5 as u64
    h = hashlib.md5(normalized.encode()).digest()[:8]
    return int.from_bytes(h, byteorder='little')


def parse_csv_with_quotes(filepath: Path) -> List[Tuple[int, str]]:
    """Parse CSV file handling quoted multi-line fields."""
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple state machine parser for quoted CSV
    current_line = ""
    in_quotes = False
    is_header = True

    for char in content:
        if char == '"':
            in_quotes = not in_quotes
            current_line += char
        elif char == '\n' and not in_quotes:
            if is_header:
                is_header = False
                current_line = ""
                continue

            if current_line.strip():
                try:
                    # Parse the line
                    parts = parse_csv_line(current_line)
                    if len(parts) >= 2:
                        label = int(parts[0])
                        text = parts[1].strip('"')
                        results.append((label, text))
                except (ValueError, IndexError):
                    pass  # Skip malformed lines

            current_line = ""
        elif char == '\n' and in_quotes:
            current_line += ' '  # Replace newline with space inside quotes
        else:
            current_line += char

    return results


def parse_csv_line(line: str) -> List[str]:
    """Parse a single CSV line handling quotes."""
    fields = []
    current = ""
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            fields.append(current.strip())
            current = ""
        else:
            current += char

    fields.append(current.strip())
    return fields


def load_ethics_scenarios() -> Dict[str, List[Tuple[int, str]]]:
    """Load all ethics scenarios from the dataset."""
    all_scenarios = {}

    for category, config in CATEGORIES.items():
        prefix = config["prefix"]
        category_path = ETHICS_BASE / category / f"{prefix}_test.csv"

        if not category_path.exists():
            print(f"  Warning: {category_path} not found")
            continue

        scenarios = parse_csv_with_quotes(category_path)
        all_scenarios[category] = scenarios
        print(f"  Loaded {len(scenarios)} scenarios from {category}")

    return all_scenarios


def compute_embeddings_transformer(texts: List[str], model) -> List[List[float]]:
    """Compute embeddings using sentence-transformers."""
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def compute_embeddings_fallback(texts: List[str]) -> List[List[float]]:
    """Fallback: Generate random embeddings (for testing without transformers)."""
    import random
    random.seed(42)

    embeddings = []
    for text in texts:
        # Use hash of text to seed random generator for determinism
        seed = hash_text(text) % (2**32)
        random.seed(seed)
        emb = [random.gauss(0, 1) for _ in range(384)]
        # Normalize
        norm = sum(x*x for x in emb) ** 0.5
        emb = [x / norm for x in emb]
        embeddings.append(emb)

    return embeddings


def main():
    print("=" * 60)
    print("ETHICS Embedding Pre-computation")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load scenarios
    print("\n1. Loading scenarios...")
    all_scenarios = load_ethics_scenarios()

    if not all_scenarios:
        print("Error: No scenarios loaded!")
        return

    # Collect all unique texts
    print("\n2. Collecting unique texts...")
    all_texts = []
    text_to_idx = {}

    for category, scenarios in all_scenarios.items():
        for label, text in scenarios:
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)

    print(f"  Total unique texts: {len(all_texts)}")

    # Compute embeddings
    print("\n3. Computing embeddings...")

    if HAS_TRANSFORMERS:
        print(f"  Using model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        embeddings = compute_embeddings_transformer(all_texts, model)
    else:
        print("  Using fallback (random embeddings)")
        embeddings = compute_embeddings_fallback(all_texts)

    print(f"  Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

    # Create output format
    print("\n4. Saving embeddings...")

    output_data = []
    for text, embedding in zip(all_texts, embeddings):
        output_data.append({
            "text_hash": hash_text(text),
            "embedding": embedding
        })

    output_path = OUTPUT_DIR / "ethics_embeddings.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    # Also save a mapping file for reference
    mapping_path = OUTPUT_DIR / "ethics_text_mapping.json"
    mapping = {hash_text(text): text[:100] + "..." if len(text) > 100 else text
               for text in all_texts}
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Saved: {mapping_path}")

    # Statistics
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  File size: {file_size:.2f} MB")

    print("\n" + "=" * 60)
    print("Done! Embeddings ready for Rust loading.")
    print("=" * 60)


if __name__ == "__main__":
    main()
