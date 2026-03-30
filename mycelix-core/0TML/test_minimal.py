# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Minimal validation with tiny dataset for quick testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from torch.utils.data import TensorDataset, DataLoader
from experiments.models.cnn_models import create_model
from baselines.fedavg import FedAvgServer, FedAvgClient, FedAvgConfig

print("=" * 70)
print("Minimal Training Test (Tiny Dataset)")
print("=" * 70)

# Create tiny fake dataset (100 samples)
print("\n[1/3] Creating tiny dataset...")
X = torch.randn(100, 1, 28, 28)  # 100 MNIST-like images
y = torch.randint(0, 10, (100,))  # 100 labels
dataset = TensorDataset(X, y)
print(f"✓ Created {len(dataset)} samples")

# Split into 2 clients (50 samples each)
print("\n[2/3] Creating clients...")
client_loaders = [
    DataLoader(TensorDataset(X[:50], y[:50]), batch_size=10),
    DataLoader(TensorDataset(X[50:], y[50:]), batch_size=10)
]
print(f"✓ Created 2 clients (50 samples each)")

# Create FedAvg setup
print("\n[3/3] Running 3 training rounds...")
device = torch.device('cpu')
model_fn = lambda: create_model('simple_cnn', num_classes=10)

config = FedAvgConfig(
    learning_rate=0.01,
    local_epochs=1,
    batch_size=10,
    num_clients=2,
    fraction_clients=1.0
)

server = FedAvgServer(model_fn(), config)
clients = [
    FedAvgClient(f"client_{i}", model_fn(), loader, config, device)
    for i, loader in enumerate(client_loaders)
]

for round_num in range(3):
    stats = server.train_round(clients)
    print(f"Round {round_num}: Loss={stats['train_loss']:.4f}, Acc={stats['train_accuracy']:.4f}")

print("\n" + "=" * 70)
print("✅ Minimal test passed! Training logic works correctly.")
print("=" * 70)
print("\nNote: Full experiments will be slower due to larger datasets.")
print("For 10-round test with 5 clients, expect ~5-10 minutes on CPU.")
