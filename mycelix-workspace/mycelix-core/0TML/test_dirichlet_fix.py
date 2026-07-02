#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Test the Dirichlet split fix for 100 clients."""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from torchvision import datasets, transforms
from experiments.utils.data_splits import create_dirichlet_split, analyze_split

print("=" * 70)
print("Testing Dirichlet Split Fix")
print("=" * 70)

# Load EMNIST (the dataset that failed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1751,), (0.3332,))
])

print("\nLoading EMNIST dataset...")
train_dataset = datasets.EMNIST(
    root='datasets/emnist',
    split='balanced',
    train=True,
    transform=transform,
    download=False  # Should already be downloaded
)

print(f"Dataset size: {len(train_dataset)} samples")

# Test with 100 clients and alpha=0.3 (exact config that failed)
print("\nCreating Dirichlet split with 100 clients, alpha=0.3...")
try:
    client_indices = create_dirichlet_split(
        train_dataset,
        num_clients=100,
        alpha=0.3,
        seed=1337
    )
    print("✅ Split created successfully!")

    # Analyze split
    stats = analyze_split(train_dataset, client_indices)
    print(f"\nSplit statistics:")
    print(f"  Total clients: {stats['num_clients']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Samples per client: {stats['min_samples']} to {stats['max_samples']}")
    print(f"  Mean samples: {stats['mean_samples']:.1f}")
    print(f"  Mean class imbalance: {stats['mean_class_imbalance']:.3f}")

    # Check all clients have data
    empty_clients = sum(1 for indices in client_indices if len(indices) == 0)
    if empty_clients > 0:
        print(f"\n⚠️  Warning: {empty_clients} clients have no data")
    else:
        print(f"\n✅ All {len(client_indices)} clients have data")

    print("\n" + "=" * 70)
    print("✅ Test PASSED - Dirichlet split fix works!")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Test FAILED with error:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
