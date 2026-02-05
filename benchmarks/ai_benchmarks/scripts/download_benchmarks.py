#!/usr/bin/env python3
"""
Standard AI Benchmark Dataset Downloader

Downloads and prepares the following benchmark datasets:
- MMLU: Massive Multitask Language Understanding (57 subjects)
- HumanEval: Code generation (164 problems)
- GSM8K: Grade school math (8,500 problems)
- HellaSwag: Commonsense reasoning (10,042 problems)
- TruthfulQA: Factual accuracy (817 questions)
- ARC: AI2 Reasoning Challenge (7,787 questions)

Usage:
    python download_benchmarks.py --all
    python download_benchmarks.py --mmlu --gsm8k
    python download_benchmarks.py --check
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile
import tarfile

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Benchmark configurations
BENCHMARKS = {
    "mmlu": {
        "name": "Massive Multitask Language Understanding",
        "url": "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        "size_mb": 145,
        "format": "tar",
        "subjects": 57,
        "samples": 14042,
        "description": "Tests knowledge across 57 subjects from STEM to humanities"
    },
    "humaneval": {
        "name": "HumanEval (OpenAI)",
        "url": "https://github.com/openai/human-eval/archive/refs/heads/master.zip",
        "size_mb": 0.5,
        "format": "zip",
        "problems": 164,
        "description": "Evaluates code generation capabilities"
    },
    "gsm8k": {
        "name": "Grade School Math 8K",
        "hf_dataset": "gsm8k",
        "hf_config": "main",  # gsm8k requires 'main' or 'socratic' config
        "size_mb": 10,
        "train_samples": 7473,
        "test_samples": 1319,
        "description": "Grade school math word problems requiring multi-step reasoning"
    },
    "hellaswag": {
        "name": "HellaSwag",
        "hf_dataset": "Rowan/hellaswag",
        "size_mb": 50,
        "samples": 10042,
        "description": "Commonsense reasoning via sentence completion"
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "hf_dataset": "truthful_qa",
        "hf_config": "multiple_choice",  # Options: generation, multiple_choice
        "size_mb": 5,
        "samples": 817,
        "description": "Tests whether model avoids generating false answers"
    },
    "arc": {
        "name": "AI2 Reasoning Challenge",
        "hf_dataset": "ai2_arc",
        "hf_config": "ARC-Challenge",  # Options: ARC-Challenge, ARC-Easy
        "size_mb": 25,
        "easy_samples": 5197,
        "challenge_samples": 2590,
        "description": "Grade-school science questions requiring reasoning"
    },
    "winogrande": {
        "name": "WinoGrande",
        "hf_dataset": "winogrande",
        "hf_config": "winogrande_xl",  # Options: winogrande_xs, winogrande_s, winogrande_m, winogrande_l, winogrande_xl, winogrande_debiased
        "size_mb": 15,
        "samples": 44000,
        "description": "Large-scale Winograd schema challenge"
    }
}


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indication."""
    print(f"{desc}: {url}")
    print(f"  -> {dest}")

    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print("\n  ✅ Complete")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        return False


def extract_archive(archive: Path, dest: Path) -> bool:
    """Extract tar or zip archive."""
    print(f"Extracting: {archive}")

    try:
        if archive.suffix == ".tar" or ".tar." in archive.name:
            with tarfile.open(archive) as tar:
                tar.extractall(dest)
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(dest)
        else:
            print(f"  ❌ Unknown format: {archive.suffix}")
            return False

        print("  ✅ Extracted")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def download_hf_dataset(name: str, dataset_id: str, dest: Path, config: Optional[str] = None) -> bool:
    """Download dataset from Hugging Face."""
    try:
        from datasets import load_dataset

        config_str = f" (config: {config})" if config else ""
        print(f"Downloading from Hugging Face: {dataset_id}{config_str}")
        dataset = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)

        # Save to JSON
        dest.mkdir(parents=True, exist_ok=True)
        for split in dataset:
            output_file = dest / f"{name}_{split}.json"
            dataset[split].to_json(output_file)
            print(f"  Saved: {output_file}")

        print("  ✅ Complete")
        return True
    except ImportError:
        print("  ⚠️  Install 'datasets' package: pip install datasets")
        return create_hf_stub(name, dataset_id, dest)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return create_hf_stub(name, dataset_id, dest)


def create_hf_stub(name: str, dataset_id: str, dest: Path) -> bool:
    """Create a stub file with download instructions."""
    dest.mkdir(parents=True, exist_ok=True)
    stub_file = dest / f"{name}_DOWNLOAD_INSTRUCTIONS.md"

    instructions = f"""# {name} Dataset Download Instructions

## Option 1: Using Python (Recommended)

```python
from datasets import load_dataset

dataset = load_dataset("{dataset_id}", trust_remote_code=True)

# Save locally
for split in dataset:
    dataset[split].to_json(f"{name}_{{split}}.json")
```

## Option 2: Manual Download

Visit: https://huggingface.co/datasets/{dataset_id}

Download the files and place them in this directory.

## Option 3: CLI

```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('{dataset_id}')"
```
"""

    with open(stub_file, 'w') as f:
        f.write(instructions)

    print(f"  📝 Created instructions: {stub_file}")
    return False


def download_mmlu() -> bool:
    """Download MMLU dataset."""
    dest_dir = DATA_DIR / "mmlu"
    dest_dir.mkdir(parents=True, exist_ok=True)

    config = BENCHMARKS["mmlu"]
    archive = dest_dir / "mmlu.tar"

    if (dest_dir / "data").exists():
        print("MMLU: Already downloaded ✅")
        return True

    if not download_file(config["url"], archive, "Downloading MMLU"):
        return False

    if not extract_archive(archive, dest_dir):
        return False

    # Clean up archive
    archive.unlink()
    return True


def download_humaneval() -> bool:
    """Download HumanEval dataset."""
    dest_dir = DATA_DIR / "humaneval"
    dest_dir.mkdir(parents=True, exist_ok=True)

    config = BENCHMARKS["humaneval"]
    archive = dest_dir / "humaneval.zip"

    if (dest_dir / "human-eval-master").exists():
        print("HumanEval: Already downloaded ✅")
        return True

    if not download_file(config["url"], archive, "Downloading HumanEval"):
        return False

    if not extract_archive(archive, dest_dir):
        return False

    archive.unlink()
    return True


def download_gsm8k() -> bool:
    """Download GSM8K dataset."""
    config = BENCHMARKS["gsm8k"].get("hf_config")
    return download_hf_dataset("gsm8k", BENCHMARKS["gsm8k"]["hf_dataset"], DATA_DIR / "gsm8k", config=config)


def download_hellaswag() -> bool:
    """Download HellaSwag dataset."""
    return download_hf_dataset("hellaswag", BENCHMARKS["hellaswag"]["hf_dataset"], DATA_DIR / "hellaswag")


def download_truthfulqa() -> bool:
    """Download TruthfulQA dataset."""
    config = BENCHMARKS["truthfulqa"].get("hf_config")
    return download_hf_dataset("truthfulqa", BENCHMARKS["truthfulqa"]["hf_dataset"], DATA_DIR / "truthfulqa", config=config)


def download_arc() -> bool:
    """Download ARC dataset."""
    config = BENCHMARKS["arc"].get("hf_config")
    return download_hf_dataset("arc", BENCHMARKS["arc"]["hf_dataset"], DATA_DIR / "arc", config=config)


def download_winogrande() -> bool:
    """Download WinoGrande dataset."""
    config = BENCHMARKS["winogrande"].get("hf_config")
    return download_hf_dataset("winogrande", BENCHMARKS["winogrande"]["hf_dataset"], DATA_DIR / "winogrande", config=config)


def check_status():
    """Check which datasets are downloaded."""
    print("\n" + "=" * 60)
    print("BENCHMARK DATASET STATUS")
    print("=" * 60 + "\n")

    for name, config in BENCHMARKS.items():
        dest_dir = DATA_DIR / name
        exists = dest_dir.exists() and any(dest_dir.iterdir())

        status = "✅ Downloaded" if exists else "❌ Not downloaded"
        print(f"  {name:15} {config['name']:40} {status}")

    print("\n" + "-" * 60)
    print("Total datasets:", len(BENCHMARKS))
    print("Data directory:", DATA_DIR)
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download standard AI benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_benchmarks.py --all       # Download everything
    python download_benchmarks.py --mmlu      # Just MMLU
    python download_benchmarks.py --check     # Check status
        """
    )

    parser.add_argument("--all", action="store_true", help="Download all benchmarks")
    parser.add_argument("--check", action="store_true", help="Check download status")
    parser.add_argument("--mmlu", action="store_true", help="Download MMLU")
    parser.add_argument("--humaneval", action="store_true", help="Download HumanEval")
    parser.add_argument("--gsm8k", action="store_true", help="Download GSM8K")
    parser.add_argument("--hellaswag", action="store_true", help="Download HellaSwag")
    parser.add_argument("--truthfulqa", action="store_true", help="Download TruthfulQA")
    parser.add_argument("--arc", action="store_true", help="Download ARC")
    parser.add_argument("--winogrande", action="store_true", help="Download WinoGrande")

    args = parser.parse_args()

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_status()
        return

    if not any([args.all, args.mmlu, args.humaneval, args.gsm8k,
                args.hellaswag, args.truthfulqa, args.arc, args.winogrande]):
        parser.print_help()
        return

    print("\n" + "=" * 60)
    print("SYMTHAEA AI BENCHMARK DOWNLOADER")
    print("=" * 60 + "\n")

    downloaders = {
        "mmlu": download_mmlu,
        "humaneval": download_humaneval,
        "gsm8k": download_gsm8k,
        "hellaswag": download_hellaswag,
        "truthfulqa": download_truthfulqa,
        "arc": download_arc,
        "winogrande": download_winogrande,
    }

    results = {}

    for name, downloader in downloaders.items():
        if args.all or getattr(args, name, False):
            print(f"\n{'='*40}")
            print(f"Downloading: {name.upper()}")
            print(f"{'='*40}")
            results[name] = downloader()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name:15} {status}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
