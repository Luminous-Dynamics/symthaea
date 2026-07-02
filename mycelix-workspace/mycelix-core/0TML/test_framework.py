# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick validation test for experiment framework.
Tests each component step-by-step with immediate output.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path.cwd()))

print("=" * 70)
print("Experiment Framework Validation Test")
print("=" * 70)

# Test 1: Imports
print("\n[1/7] Testing imports...")
try:
    from experiments.models.cnn_models import create_model
    from experiments.utils.data_splits import create_iid_split, create_dataloaders
    from baselines.fedavg import create_fedavg_experiment, FedAvgConfig
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Model creation
print("\n[2/7] Testing model creation...")
try:
    model = create_model('simple_cnn', num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ SimpleCNN created ({param_count:,} parameters)")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

# Test 3: Dataset loading
print("\n[3/7] Testing dataset loading...")
try:
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='datasets/mnist',
        train=True,
        transform=transform,
        download=False
    )
    print(f"✓ MNIST loaded ({len(train_dataset)} samples)")
except Exception as e:
    print(f"✗ Dataset loading error: {e}")
    sys.exit(1)

# Test 4: Data splitting
print("\n[4/7] Testing data splitting...")
try:
    client_indices = create_iid_split(train_dataset, num_clients=5, seed=42)
    print(f"✓ IID split created (5 clients, {len(client_indices[0])} samples each)")
except Exception as e:
    print(f"✗ Data splitting error: {e}")
    sys.exit(1)

# Test 5: DataLoader creation
print("\n[5/7] Testing DataLoader creation...")
try:
    client_loaders = create_dataloaders(train_dataset, client_indices, batch_size=32)
    print(f"✓ DataLoaders created ({len(client_loaders)} clients)")
except Exception as e:
    print(f"✗ DataLoader creation error: {e}")
    sys.exit(1)

# Test 6: Baseline initialization
print("\n[6/7] Testing baseline initialization...")
try:
    model_fn = lambda: create_model('simple_cnn', num_classes=10)

    test_dataset = datasets.MNIST(
        root='datasets/mnist',
        train=False,
        transform=transform,
        download=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    config = FedAvgConfig(
        learning_rate=0.01,
        local_epochs=1,
        batch_size=32,
        num_clients=5,
        fraction_clients=1.0
    )

    device = torch.device('cpu')
    server, clients, test_loader = create_fedavg_experiment(
        model_fn, client_loaders, test_loader, config, device
    )
    print(f"✓ FedAvg baseline initialized ({len(clients)} clients)")
except Exception as e:
    print(f"✗ Baseline initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Single training round
print("\n[7/7] Testing single training round...")
try:
    stats = server.train_round(clients)
    print(f"✓ Training round completed")
    print(f"  - Train loss: {stats['train_loss']:.4f}")
    print(f"  - Train accuracy: {stats['train_accuracy']:.4f}")
except Exception as e:
    print(f"✗ Training round error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All validation tests passed!")
print("=" * 70)
print("\nThe experiment framework is working correctly.")
print("Ready to run full experiments with runner.py")
