#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
CIFAR-10 Federated Learning Benchmark

Tests the Mycelix FL system with real ML training:
- 10 federated clients
- CNN model for CIFAR-10
- Byzantine attack simulation (30% malicious)
- Measures detection accuracy and model convergence

Author: Luminous Dynamics
Date: December 30, 2025
"""

import sys
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Import mycelix_fl components
from mycelix_fl import (
    MycelixFL,
    FLConfig,
    has_rust_backend,
    get_backend_info,
)


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_gradients(self) -> np.ndarray:
        """Extract all gradients as a flat numpy array."""
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().flatten())
        return np.concatenate(grads) if grads else np.array([])

    def apply_gradients(self, flat_grads: np.ndarray, lr: float = 0.01):
        """Apply flat gradients to model parameters."""
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                numel = param.numel()
                grad_chunk = flat_grads[offset:offset + numel]
                param -= lr * torch.from_numpy(grad_chunk.reshape(param.shape)).to(param.device)
                offset += numel


def load_cifar10(num_clients: int = 10, samples_per_client: int = 500):
    """Load CIFAR-10 and split into client datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=False, download=True, transform=transform
    )

    # Split training data among clients
    total_samples = len(trainset)
    indices = list(range(total_samples))
    random.shuffle(indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_indices = indices[start:end]
        client_datasets.append(Subset(trainset, client_indices))

    test_loader = DataLoader(testset, batch_size=100, shuffle=False)

    return client_datasets, test_loader


def train_one_batch(model: nn.Module, data: torch.Tensor, target: torch.Tensor):
    """Train model on one batch and return gradients."""
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return model.get_gradients(), loss.item()


def evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    model.train()
    return correct / total


def create_byzantine_gradient(honest_grad: np.ndarray, attack_type: str = "sign_flip") -> np.ndarray:
    """Create a Byzantine (malicious) gradient."""
    if attack_type == "sign_flip":
        return -honest_grad * 2  # Flip signs and amplify
    elif attack_type == "random":
        return np.random.randn(*honest_grad.shape).astype(np.float32) * np.std(honest_grad) * 10
    elif attack_type == "zero":
        return np.zeros_like(honest_grad)
    elif attack_type == "scaling":
        return honest_grad * 100  # Scale up massively
    else:
        return honest_grad


def run_benchmark(
    num_clients: int = 10,
    num_byzantine: int = 3,
    num_rounds: int = 10,
    samples_per_client: int = 500,
    attack_type: str = "sign_flip",
):
    """Run the CIFAR-10 FL benchmark."""
    print("=" * 70)
    print("CIFAR-10 Federated Learning Benchmark")
    print("=" * 70)

    # Backend info
    info = get_backend_info()
    print(f"\nBackend: {'Rust' if info['rust_available'] else 'Python'}")
    if info['rust_available']:
        print(f"Rust version: {info.get('rust_version', 'N/A')}")
        print(f"Paradigm shifts: {info.get('rust_paradigm_shifts', 'N/A')}")

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Byzantine: {num_byzantine} ({100*num_byzantine/num_clients:.0f}%)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Attack: {attack_type}")
    print(f"  Samples/client: {samples_per_client}")

    # Load data
    client_datasets, test_loader = load_cifar10(num_clients, samples_per_client)
    print(f"\nData loaded: {len(client_datasets)} clients")

    # Initialize models
    global_model = SimpleCNN()
    client_models = [SimpleCNN() for _ in range(num_clients)]

    # Sync client models with global
    for client_model in client_models:
        client_model.load_state_dict(global_model.state_dict())

    # Select Byzantine clients
    byzantine_clients = set(random.sample(range(num_clients), num_byzantine))
    print(f"Byzantine clients: {sorted(byzantine_clients)}")

    # Initialize Mycelix FL
    fl_config = FLConfig(
        num_rounds=num_rounds,
        byzantine_threshold=0.45,
        use_compression=True,
        use_detection=True,
        use_healing=True,
        aggregation_method="shapley",
    )
    fl_system = MycelixFL(config=fl_config)

    # Initial accuracy
    initial_acc = evaluate(global_model, test_loader)
    print(f"\nInitial accuracy: {100*initial_acc:.2f}%")

    # Metrics tracking
    accuracies = [initial_acc]
    detection_counts = []
    round_times = []

    print("\n" + "-" * 70)
    print("Starting Federated Learning...")
    print("-" * 70)

    for round_num in range(num_rounds):
        round_start = time.time()

        # Sync client models
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # Local training and gradient collection
        gradients = {}
        avg_loss = 0.0

        for client_id in range(num_clients):
            # Get one batch from client
            loader = DataLoader(client_datasets[client_id], batch_size=32, shuffle=True)
            data, target = next(iter(loader))

            # Train and get gradients
            grad, loss = train_one_batch(client_models[client_id], data, target)
            avg_loss += loss

            # Byzantine clients send malicious gradients
            if client_id in byzantine_clients:
                grad = create_byzantine_gradient(grad, attack_type)

            gradients[f"client-{client_id}"] = grad.astype(np.float32)

        avg_loss /= num_clients

        # Execute FL round with Byzantine detection
        try:
            result = fl_system.execute_round(gradients, round_num=round_num)

            # Track detections
            detected = len(result.byzantine_nodes)
            detection_counts.append(detected)

            # Get aggregated gradient
            agg_grad = result.aggregated_gradient

            # Apply to global model
            global_model.apply_gradients(agg_grad, lr=0.01)

        except Exception as e:
            print(f"  Round {round_num+1} error: {e}")
            # Fallback: simple average
            agg_grad = np.mean(list(gradients.values()), axis=0)
            global_model.apply_gradients(agg_grad, lr=0.01)
            detected = 0
            detection_counts.append(0)

        # Evaluate
        accuracy = evaluate(global_model, test_loader)
        accuracies.append(accuracy)

        round_time = time.time() - round_start
        round_times.append(round_time)

        # Progress
        print(f"Round {round_num+1:2d}/{num_rounds}: "
              f"acc={100*accuracy:5.2f}%, "
              f"loss={avg_loss:.4f}, "
              f"detected={detected}/{num_byzantine}, "
              f"time={round_time:.2f}s")

    # Final results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    final_acc = accuracies[-1]
    best_acc = max(accuracies)
    avg_detection = np.mean(detection_counts) if detection_counts else 0
    total_time = sum(round_times)

    print(f"\nAccuracy:")
    print(f"  Initial: {100*initial_acc:.2f}%")
    print(f"  Final:   {100*final_acc:.2f}%")
    print(f"  Best:    {100*best_acc:.2f}%")
    print(f"  Gain:    {100*(final_acc-initial_acc):+.2f}pp")

    print(f"\nByzantine Detection:")
    print(f"  Avg detected: {avg_detection:.1f}/{num_byzantine}")
    print(f"  Detection rate: {100*avg_detection/num_byzantine:.1f}%")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg round:  {np.mean(round_times):.2f}s")

    print("\n" + "=" * 70)

    return {
        "accuracies": accuracies,
        "detection_counts": detection_counts,
        "round_times": round_times,
        "final_accuracy": final_acc,
        "detection_rate": avg_detection / num_byzantine if num_byzantine > 0 else 1.0,
    }


if __name__ == "__main__":
    # Run benchmark with default settings
    results = run_benchmark(
        num_clients=10,
        num_byzantine=3,  # 30% Byzantine
        num_rounds=10,
        samples_per_client=500,
        attack_type="sign_flip",
    )
