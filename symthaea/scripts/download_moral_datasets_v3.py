#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Download REAL external moral reasoning datasets for Symthaea benchmarks.

Replaces synthetic placeholders with actual research datasets:
  1. Moral Stories (Emelin et al., 2021) — ~12K norm/action/consequence triples
  2. SCRUPLES (Lourie et al., 2021) — ~32K Reddit AITA anecdotes
  3. MoralExceptQA (Jin et al., 2022) — ~3.2K moral exception scenarios
  4. MoralChoice (Scherrer et al., 2023) — ~1.7K low/high ambiguity scenarios

Usage: python3 scripts/download_moral_datasets_v3.py
"""

import csv
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "moral_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EXAMPLES = 10000


def save_dataset(name: str, data: list, metadata: dict):
    """Save dataset as JSON with metadata."""
    output_path = OUTPUT_DIR / f"{name}.json"
    output = {"metadata": metadata, "examples": data}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(data)} examples to {output_path}")


def download_url(url: str) -> bytes | None:
    """Download URL content, return bytes or None on failure."""
    try:
        print(f"  Downloading {url[:90]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Symthaea-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_moral_stories():
    """
    Moral Stories (Emelin et al., 2021)
    ~12K structured moral narratives with norm/action/consequence triples.
    Source: HuggingFace demelin/moral_stories
    """
    print("\n" + "=" * 60)
    print("1. Moral Stories (Emelin et al., 2021)")
    print("=" * 60)

    # Full dataset as single JSONL
    url = "https://huggingface.co/datasets/demelin/moral_stories/resolve/main/data/moral_stories_full.jsonl"
    data = download_url(url)

    if data is None:
        # Fallback: try split files
        for split in ["test", "train", "valid"]:
            alt = f"https://huggingface.co/datasets/demelin/moral_stories/resolve/main/data/generation/action%7Ccontext/norm_distance/{split}.jsonl"
            data = download_url(alt)
            if data:
                break

    if data is None:
        print("  Failed to download Moral Stories")
        return False

    all_examples = []
    for line in data.decode("utf-8", errors="replace").strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            all_examples.append({
                "norm": obj.get("norm", ""),
                "situation": obj.get("situation", ""),
                "intention": obj.get("intention", ""),
                "moral_action": obj.get("moral_action", ""),
                "moral_consequence": obj.get("moral_consequence", ""),
                "immoral_action": obj.get("immoral_action", ""),
                "immoral_consequence": obj.get("immoral_consequence", ""),
                "label": obj.get("label", None),
            })
        except json.JSONDecodeError:
            continue
        if len(all_examples) >= MAX_EXAMPLES:
            break

    if all_examples:
        save_dataset("moral_stories", all_examples, {
            "source": "demelin/moral_stories (HuggingFace)",
            "url": "https://huggingface.co/datasets/demelin/moral_stories",
            "description": "Structured moral narratives with norm/action/consequence triples (Emelin et al., 2021)"
        })
        return True

    print("  No examples parsed from Moral Stories")
    return False


def download_scruples():
    """
    SCRUPLES (Lourie et al., 2021)
    Reddit AITA posts with community moral judgments.
    Source: Google Cloud Storage (AllenAI)
    """
    print("\n" + "=" * 60)
    print("2. SCRUPLES (Lourie et al., 2021)")
    print("=" * 60)

    # Anecdotes corpus as tar.gz
    url = "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz"
    data = download_url(url)

    if data is None:
        print("  Failed to download SCRUPLES anecdotes archive")
        return False

    all_examples = []
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for member in tar.getmembers():
            if not member.name.endswith(".jsonl"):
                continue
            split = "test" if "test" in member.name else ("dev" if "dev" in member.name else "train")
            print(f"  Extracting {member.name} ({split})...")
            f = tar.extractfile(member)
            if f is None:
                continue
            for line in f.read().decode("utf-8", errors="replace").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    # label_scores has AUTHOR/OTHER/EVERYBODY/NOBODY/INFO counts
                    label_scores = obj.get("label_scores", {})
                    # AUTHOR = poster is wrong (YTA=0), OTHER = poster is right (NTA=1)
                    label_raw = obj.get("label", "")
                    if label_raw == "AUTHOR":
                        label = 0
                    elif label_raw == "OTHER":
                        label = 1
                    else:
                        label = -1  # EVERYBODY/NOBODY/INFO

                    all_examples.append({
                        "title": obj.get("title", ""),
                        "text": obj.get("text", "")[:2000],
                        "label": label,
                        "label_raw": label_raw,
                        "label_scores": label_scores,
                        "split": split,
                    })
                except json.JSONDecodeError:
                    continue
                if len(all_examples) >= MAX_EXAMPLES:
                    break
            if len(all_examples) >= MAX_EXAMPLES:
                break
        tar.close()
    except Exception as e:
        print(f"  Error extracting tar.gz: {e}")
        return False

    if all_examples:
        save_dataset("scruples", all_examples, {
            "source": "allenai/scruples (Google Cloud Storage)",
            "url": "https://github.com/allenai/scruples",
            "description": "Reddit AITA posts with community moral judgments (Lourie et al., 2021)"
        })
        return True

    print("  No examples parsed from SCRUPLES")
    return False


def download_moral_exceptqa():
    """
    MoralExceptQA (Jin et al., 2022)
    Moral rules + exception scenarios.
    Source: HuggingFace feradauto/MoralExceptQA
    """
    print("\n" + "=" * 60)
    print("3. MoralExceptQA (Jin et al., 2022)")
    print("=" * 60)

    url = "https://huggingface.co/datasets/feradauto/MoralExceptQA/resolve/main/data/complete_file.json"
    data = download_url(url)

    if data is None:
        # Try alternative paths
        for alt in [
            "https://huggingface.co/datasets/feradauto/MoralExceptQA/resolve/main/complete_file.json",
            "https://huggingface.co/datasets/feradauto/MoralExceptQA/resolve/main/data/test.json",
        ]:
            data = download_url(alt)
            if data:
                break

    if data is None:
        print("  Failed to download MoralExceptQA")
        return False

    text = data.decode("utf-8", errors="replace").strip()

    # Try JSON array first, then JSONL
    all_examples = []
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            all_examples = raw[:MAX_EXAMPLES]
        elif isinstance(raw, dict):
            keys = list(raw.keys())
            if keys and isinstance(raw[keys[0]], list):
                n = len(raw[keys[0]])
                for i in range(min(n, MAX_EXAMPLES)):
                    example = {k: raw[k][i] if i < len(raw[k]) else None for k in keys}
                    all_examples.append(example)
            else:
                all_examples = [raw]
    except json.JSONDecodeError:
        # Try JSONL (one JSON object per line)
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                all_examples.append(obj)
            except json.JSONDecodeError:
                continue
            if len(all_examples) >= MAX_EXAMPLES:
                break

    if all_examples:
        save_dataset("moral_exceptqa", all_examples, {
            "source": "feradauto/MoralExceptQA (HuggingFace)",
            "url": "https://huggingface.co/datasets/feradauto/MoralExceptQA",
            "description": "Moral exception scenarios — when is it OK to break a moral rule? (Jin et al., 2022)"
        })
        return True

    print("  No examples parsed from MoralExceptQA")
    return False


def download_moral_choice():
    """
    MoralChoice (Scherrer et al., 2023)
    Paired moral choice scenarios at low and high ambiguity.
    Source: GitHub ninodimontalcino/moralchoice
    """
    print("\n" + "=" * 60)
    print("4. MoralChoice (Scherrer et al., 2023)")
    print("=" * 60)

    urls = [
        ("low", "https://raw.githubusercontent.com/ninodimontalcino/moralchoice/main/data/scenarios/moralchoice_low_ambiguity.csv"),
        ("high", "https://raw.githubusercontent.com/ninodimontalcino/moralchoice/main/data/scenarios/moralchoice_high_ambiguity.csv"),
    ]

    # Fallback to HuggingFace
    hf_urls = [
        ("low", "https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/scenarios/moralchoice_low_ambiguity.csv"),
        ("high", "https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/scenarios/moralchoice_high_ambiguity.csv"),
    ]

    all_examples = []
    for ambiguity, url in urls:
        data = download_url(url)
        if data is None:
            # Try HuggingFace fallback
            for hf_amb, hf_url in hf_urls:
                if hf_amb == ambiguity:
                    data = download_url(hf_url)
                    break
        if data is None:
            continue

        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            example = {k.strip(): v.strip() if v else "" for k, v in row.items()}
            example["ambiguity"] = ambiguity
            all_examples.append(example)
            if len(all_examples) >= MAX_EXAMPLES:
                break

    if all_examples:
        save_dataset("moral_choice", all_examples, {
            "source": "ninodimontalcino/moralchoice (GitHub)",
            "url": "https://github.com/ninodimontalcino/moralchoice",
            "description": "Paired moral choice scenarios at low and high ambiguity (Scherrer et al., 2023)"
        })
        return True

    print("  Failed to download MoralChoice")
    return False


def main():
    print("=" * 60)
    print("Downloading REAL External Moral Reasoning Datasets")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max examples per dataset: {MAX_EXAMPLES}")

    results = {}
    results["moral_stories"] = download_moral_stories()
    results["scruples"] = download_scruples()
    results["moral_exceptqa"] = download_moral_exceptqa()
    results["moral_choice"] = download_moral_choice()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        path = OUTPUT_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                count = len(data.get("examples", []))
            size = path.stat().st_size
            print(f"  [{status}] {name}: {count:,} examples ({size / 1024:.1f} KB)")
        else:
            print(f"  [{status}] {name}: not downloaded")

    print("\n  Existing datasets:")
    for name in ["ethics", "social_chemistry_292k"]:
        path = OUTPUT_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                count = len(data.get("examples", []))
            size = path.stat().st_size
            print(f"  [OK] {name}: {count:,} examples ({size / 1024:.1f} KB)")

    total_examples = 0
    for f in OUTPUT_DIR.glob("*.json"):
        if "prototypes" in f.name:
            continue
        with open(f) as jf:
            data = json.load(jf)
            total_examples += len(data.get("examples", []))
    print(f"\n  Total examples across all datasets: {total_examples:,}")
    print("Done!")


if __name__ == "__main__":
    main()
