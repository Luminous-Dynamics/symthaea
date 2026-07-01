#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Download priority moral reasoning datasets for Symthaea benchmarks.

Downloads 5 datasets from HuggingFace and converts to JSON format for Rust consumption.
"""

import json
import os
from pathlib import Path

# Try to import datasets, provide helpful error if missing
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed.")
    print("Install with: pip install datasets")
    exit(1)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "moral_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_dataset(name: str, data: list, metadata: dict):
    """Save dataset as JSON with metadata."""
    output_path = OUTPUT_DIR / f"{name}.json"
    output = {
        "metadata": metadata,
        "examples": data
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(data)} examples to {output_path}")

def download_ethics():
    """Download ETHICS benchmark (hendrycks/ethics)."""
    print("\n1. Downloading ETHICS benchmark...")

    categories = ["commonsense", "deontology", "justice", "virtue"]
    all_examples = []

    for category in categories:
        try:
            # Load the specific category
            ds = load_dataset("hendrycks/ethics", category)

            # Process train and test splits
            for split in ["train", "test"]:
                if split in ds:
                    for item in ds[split]:
                        example = {
                            "category": category,
                            "split": split,
                            "text": item.get("input", item.get("scenario", item.get("text", ""))),
                            "label": item.get("label", item.get("is_ethical", None)),
                        }
                        # Handle different field names across categories
                        if "scenario" in item:
                            example["text"] = item["scenario"]
                        if "excuse" in item:
                            example["excuse"] = item["excuse"]
                        all_examples.append(example)

            print(f"  Loaded {category}: {len(ds.get('train', []))} train, {len(ds.get('test', []))} test")
        except Exception as e:
            print(f"  Warning: Could not load {category}: {e}")

    save_dataset("ethics", all_examples, {
        "source": "hendrycks/ethics",
        "url": "https://huggingface.co/datasets/hendrycks/ethics",
        "categories": categories,
        "description": "ETHICS benchmark for commonsense moral reasoning"
    })

def download_moral_stories():
    """Download Moral Stories dataset."""
    print("\n2. Downloading Moral Stories...")

    try:
        ds = load_dataset("demelin/moral_stories", )
        all_examples = []

        for split in ds:
            for item in ds[split]:
                example = {
                    "split": split,
                    "norm": item.get("norm", ""),
                    "situation": item.get("situation", ""),
                    "intention": item.get("intention", ""),
                    "moral_action": item.get("moral_action", ""),
                    "moral_consequence": item.get("moral_consequence", ""),
                    "immoral_action": item.get("immoral_action", ""),
                    "immoral_consequence": item.get("immoral_consequence", ""),
                }
                all_examples.append(example)

        save_dataset("moral_stories", all_examples, {
            "source": "demelin/moral_stories",
            "url": "https://huggingface.co/datasets/demelin/moral_stories",
            "description": "Structured moral narratives with norm/action/consequence"
        })
    except Exception as e:
        print(f"  Error: {e}")

def download_scruples():
    """Download SCRUPLES dataset."""
    print("\n3. Downloading SCRUPLES...")

    try:
        # Try the anecdotes subset
        ds = load_dataset("metaeval/scruples", "anecdotes", )
        all_examples = []

        for split in ds:
            for item in ds[split]:
                example = {
                    "split": split,
                    "text": item.get("text", item.get("post_text", "")),
                    "title": item.get("title", item.get("post_title", "")),
                    "label": item.get("label", item.get("binarized_label", None)),
                    "label_distribution": item.get("label_scores", None),
                }
                all_examples.append(example)

        save_dataset("scruples", all_examples, {
            "source": "metaeval/scruples",
            "url": "https://huggingface.co/datasets/metaeval/scruples",
            "description": "Reddit AITA posts with crowdsourced moral judgments"
        })
    except Exception as e:
        print(f"  Error with metaeval/scruples: {e}")
        print("  Trying alternative source...")
        try:
            # Try alternative
            ds = load_dataset("allenai/scruples", )
            all_examples = []
            for split in ds:
                for item in ds[split]:
                    all_examples.append(dict(item))
            save_dataset("scruples", all_examples, {
                "source": "allenai/scruples",
                "description": "Reddit AITA posts with crowdsourced moral judgments"
            })
        except Exception as e2:
            print(f"  Error: Could not download SCRUPLES: {e2}")

def download_social_chemistry():
    """Download Social Chemistry 101 dataset."""
    print("\n4. Downloading Social Chemistry 101...")

    try:
        ds = load_dataset("allenai/social_i_qa", )
        all_examples = []

        for split in ds:
            for item in ds[split]:
                example = {
                    "split": split,
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "answerA": item.get("answerA", ""),
                    "answerB": item.get("answerB", ""),
                    "answerC": item.get("answerC", ""),
                    "label": item.get("label", ""),
                }
                all_examples.append(example)

        save_dataset("social_chemistry", all_examples, {
            "source": "allenai/social_i_qa",
            "url": "https://huggingface.co/datasets/allenai/social_i_qa",
            "description": "Social intelligence QA with norms and rules-of-thumb"
        })
    except Exception as e:
        print(f"  Error: {e}")
        print("  Trying social_chem_101...")
        try:
            ds = load_dataset("allenai/social_chem_101", )
            all_examples = []
            for split in ds:
                count = 0
                for item in ds[split]:
                    if count >= 50000:  # Limit to 50K for manageable size
                        break
                    example = {
                        "split": split,
                        "action": item.get("action", ""),
                        "rot": item.get("rot", ""),  # Rule of thumb
                        "rot_judgment": item.get("rot-judgment", ""),
                        "rot_action": item.get("rot-action", ""),
                    }
                    all_examples.append(example)
                    count += 1
            save_dataset("social_chemistry", all_examples, {
                "source": "allenai/social_chem_101",
                "description": "Rules-of-thumb for social/moral norms"
            })
        except Exception as e2:
            print(f"  Error: Could not download Social Chemistry: {e2}")

def download_moral_exceptqa():
    """Download MoralExceptQA dataset."""
    print("\n5. Downloading MoralExceptQA...")

    try:
        ds = load_dataset("feradauto/MoralExceptQA", )
        all_examples = []

        for split in ds:
            for item in ds[split]:
                example = dict(item)
                example["split"] = split
                all_examples.append(example)

        save_dataset("moral_exceptqa", all_examples, {
            "source": "feradauto/MoralExceptQA",
            "url": "https://huggingface.co/datasets/feradauto/MoralExceptQA",
            "description": "Moral exception scenarios testing when rules can be broken"
        })
    except Exception as e:
        print(f"  Error: {e}")

def main():
    print("=" * 60)
    print("Downloading Priority Moral Reasoning Datasets")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    download_ethics()
    download_moral_stories()
    download_scruples()
    download_social_chemistry()
    download_moral_exceptqa()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    total_size = 0
    for f in OUTPUT_DIR.glob("*.json"):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")

    print(f"\nTotal: {total_size / 1024 / 1024:.1f} MB")
    print("Done!")

if __name__ == "__main__":
    main()
