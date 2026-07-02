#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test All Three Datasets Before Full Launch
============================================

Verify MNIST, EMNIST, and CIFAR-10 all work correctly with their
proper model architectures:

- MNIST: SimpleCNN (1-channel grayscale, 10 classes)
- EMNIST: SimpleCNN (1-channel grayscale, 62 classes)
- CIFAR-10: ResNet9 (3-channel RGB, 10 classes)

This catches configuration issues before launching the full 96-experiment matrix!
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from experiments.runner import ExperimentRunner

# Suppress CUDA errors for cleaner output
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def create_test_config(dataset_name: str, model_type: str, num_classes: int) -> dict:
    """Create minimal test config for a dataset with proper architecture"""
    return {
        'experiment_name': f'test_{dataset_name}',
        'seed': 42,
        'dataset': {
            'name': dataset_name,
            'data_dir': 'datasets'
        },
        'model': {
            'type': model_type,  # CRITICAL: Must match dataset image format!
            'params': {
                'num_classes': num_classes  # CRITICAL: Must match dataset!
            }
        },
        'federated': {
            'num_clients': 5,  # Small for fast test
            'batch_size': 32,
            'learning_rate': 0.01,
            'local_epochs': 1,
            'fraction_clients': 1.0
        },
        'attack': {
            'type': 'sign_flip',
            'byzantine_ratio': 0.20
        },
        'baselines': ['fedavg'],
        'training': {
            'num_rounds': 2,  # Just 2 rounds for quick test
            'eval_every': 2
        },
        'output_dir': 'results',
        'data_split': {
            'type': 'iid',
            'seed': 42
        }
    }

def test_dataset(dataset_name: str, model_type: str, num_classes: int):
    """Test if a dataset works correctly with proper architecture"""
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name.upper()}")
    print(f"  Model: {model_type}")
    print(f"  Classes: {num_classes}")
    print(f"{'='*70}")

    try:
        # Create config
        config = create_test_config(dataset_name, model_type, num_classes)

        # Save to temp file
        temp_path = f"/tmp/test_{dataset_name}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)

        print(f"✓ Config created: {temp_path}")

        # Try to run experiment
        print(f"Running 2-round test...")
        runner = ExperimentRunner(temp_path)
        runner.run()

        print(f"✅ {dataset_name.upper()} WORKS!")
        return True

    except Exception as e:
        print(f"❌ {dataset_name.upper()} FAILED:")
        print(f"   Error: {str(e)[:200]}")
        if "assert" in str(e).lower():
            print(f"   Likely cause: num_classes mismatch!")
        return False

def main():
    print("="*70)
    print("🧪 DATASET VERIFICATION TEST SUITE")
    print("="*70)
    print("\nTesting all three datasets with proper architectures before launching full matrix...")

    # Dataset configurations matching FINAL_working_3datasets.yaml
    datasets = [
        # (name, model_type, num_classes)
        ('mnist', 'simple_cnn', 10),    # MNIST: grayscale, SimpleCNN
        ('emnist', 'simple_cnn', 62),   # EMNIST (FEMNIST): grayscale, SimpleCNN
        ('cifar10', 'resnet9', 10),     # CIFAR-10: RGB, ResNet9
    ]

    results = {}

    for dataset_name, model_type, num_classes in datasets:
        try:
            success = test_dataset(dataset_name, model_type, num_classes)
            results[dataset_name] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error testing {dataset_name}: {e}")
            results[dataset_name] = False

    # Print summary
    print("\n" + "="*70)
    print("📊 DATASET TEST SUMMARY")
    print("="*70)

    for dataset_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {dataset_name.upper():12s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ ALL DATASETS WORKING!")
        print("\n🚀 Safe to launch full matrix with these configurations:")
        print("   - MNIST: simple_cnn, 10 classes (grayscale)")
        print("   - EMNIST: simple_cnn, 62 classes (grayscale)")
        print("   - CIFAR10: resnet9, 10 classes (RGB)")
        print("\n📋 Total experiments: 3 datasets × 4 attacks × 4 defenses × 2 seeds = 96 experiments")
        print("⏱️  Estimated runtime: ~20-24 hours")
    else:
        print("\n❌ SOME DATASETS FAILED!")
        print("\n⚠️  DO NOT launch full matrix until all datasets pass!")
        failed = [name for name, success in results.items() if not success]
        print(f"\n   Failed datasets: {', '.join(failed)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
