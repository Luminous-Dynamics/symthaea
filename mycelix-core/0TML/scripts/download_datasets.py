#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Download academic benchmark datasets for Phase 11 experiments.

Datasets:
- MNIST: 60K training images, 10K test images (~50MB)
- CIFAR-10: 50K training images, 10K test images (~170MB)
- Shakespeare: Character-level text corpus (~5MB)

All datasets are free and standard in FL literature.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_mnist():
    """Download MNIST dataset (digits 0-9, 28x28 grayscale)."""
    print("\n" + "="*70)
    print("📦 Downloading MNIST Dataset")
    print("="*70)

    try:
        from torchvision import datasets

        data_dir = project_root / "datasets" / "mnist"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Download location: {data_dir}")
        print("Downloading training set...")
        datasets.MNIST(data_dir, train=True, download=True)

        print("Downloading test set...")
        datasets.MNIST(data_dir, train=False, download=True)

        print("✅ MNIST downloaded successfully!")
        print(f"   - Training: 60,000 images")
        print(f"   - Test: 10,000 images")
        print(f"   - Size: ~50MB")
        return True

    except Exception as e:
        print(f"❌ MNIST download failed: {e}")
        return False


def download_cifar10():
    """Download CIFAR-10 dataset (10 classes, 32x32 color)."""
    print("\n" + "="*70)
    print("📦 Downloading CIFAR-10 Dataset")
    print("="*70)

    try:
        from torchvision import datasets

        data_dir = project_root / "datasets" / "cifar10"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Download location: {data_dir}")
        print("Downloading training set...")
        datasets.CIFAR10(data_dir, train=True, download=True)

        print("Downloading test set...")
        datasets.CIFAR10(data_dir, train=False, download=True)

        print("✅ CIFAR-10 downloaded successfully!")
        print(f"   - Training: 50,000 images")
        print(f"   - Test: 10,000 images")
        print(f"   - Size: ~170MB")
        return True

    except Exception as e:
        print(f"❌ CIFAR-10 download failed: {e}")
        return False


def download_shakespeare():
    """Download Shakespeare dataset for NLP experiments."""
    print("\n" + "="*70)
    print("📦 Downloading Shakespeare Dataset")
    print("="*70)

    try:
        import urllib.request

        data_dir = project_root / "datasets" / "shakespeare"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download from TensorFlow Federated datasets
        url = "https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2"
        target_file = data_dir / "shakespeare.tar.bz2"

        print(f"Download location: {data_dir}")
        print(f"Downloading from: {url}")

        if target_file.exists():
            print("⚠️  File already exists, skipping download")
        else:
            urllib.request.urlretrieve(url, target_file)
            print(f"Downloaded to: {target_file}")

        # Extract if needed
        extract_dir = data_dir / "data"
        if not extract_dir.exists():
            print("Extracting archive...")
            import tarfile
            with tarfile.open(target_file, 'r:bz2') as tar:
                tar.extractall(data_dir)

        print("✅ Shakespeare downloaded successfully!")
        print(f"   - 715 clients (characters)")
        print(f"   - Character-level prediction")
        print(f"   - Size: ~5MB")
        return True

    except Exception as e:
        print(f"❌ Shakespeare download failed: {e}")
        print(f"   This is optional - can proceed with MNIST/CIFAR-10 only")
        return False


def check_datasets():
    """Check which datasets are already available."""
    print("\n" + "="*70)
    print("🔍 Checking Existing Datasets")
    print("="*70)

    datasets_info = {
        "MNIST": project_root / "datasets" / "mnist",
        "CIFAR-10": project_root / "datasets" / "cifar10",
        "Shakespeare": project_root / "datasets" / "shakespeare"
    }

    available = []
    for name, path in datasets_info.items():
        if path.exists() and any(path.iterdir()):
            print(f"✅ {name}: Available at {path}")
            available.append(name)
        else:
            print(f"❌ {name}: Not found")

    return available


def main():
    """Download all datasets for Phase 11."""
    print("\n" + "="*70)
    print("🎓 Phase 11 Dataset Preparation")
    print("="*70)
    print("Downloading academic benchmark datasets for federated learning research")
    print()

    # Check what's already available
    existing = check_datasets()

    # Download missing datasets
    results = {}

    if "MNIST" not in existing:
        results["MNIST"] = download_mnist()
    else:
        print("\n⏭️  Skipping MNIST (already available)")
        results["MNIST"] = True

    if "CIFAR-10" not in existing:
        results["CIFAR-10"] = download_cifar10()
    else:
        print("\n⏭️  Skipping CIFAR-10 (already available)")
        results["CIFAR-10"] = True

    if "Shakespeare" not in existing:
        results["Shakespeare"] = download_shakespeare()
    else:
        print("\n⏭️  Skipping Shakespeare (already available)")
        results["Shakespeare"] = True

    # Summary
    print("\n" + "="*70)
    print("📊 Download Summary")
    print("="*70)

    for name, success in results.items():
        status = "✅ Ready" if success else "❌ Failed"
        print(f"  {name}: {status}")

    successful = sum(results.values())
    total = len(results)

    print()
    print(f"Status: {successful}/{total} datasets ready")

    if successful >= 2:  # MNIST + CIFAR-10 are essential
        print("\n✅ Core datasets ready! Can proceed with Phase 11 experiments.")
        print("   (Shakespeare is optional - can add later for NLP validation)")
        return 0
    else:
        print("\n⚠️  Some downloads failed. Try running again or check network connection.")
        return 1


if __name__ == "__main__":
    exit(main())
