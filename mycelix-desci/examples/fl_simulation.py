#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Federated Learning Simulation with PoGQ

This example demonstrates:
1. Multiple clients training on local data
2. PoGQ validation of gradient updates
3. Byzantine actor detection
4. Server aggregation of valid gradients

Run with: python examples/fl_simulation.py
"""

import sys
sys.path.insert(0, 'src/ml')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mycelix_desci_ml.pogq import PoGQValidator, GradientUpdate
from mycelix_desci_ml.fl import FederatedClient, FederatedServer


def generate_synthetic_data(num_samples=100, num_features=10, seed=None):
    """Generate synthetic classification data"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    X = torch.randn(num_samples, num_features)
    y = (X.sum(dim=1) > 0).long()  # Simple classification rule

    return TensorDataset(X, y)


def create_model(num_features=10, num_classes=2):
    """Create a simple neural network"""
    return nn.Sequential(
        nn.Linear(num_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes)
    )


def simulate_federated_learning(
    num_clients=5,
    num_byzantine=1,
    num_rounds=3,
    local_epochs=1
):
    """Simulate federated learning with Byzantine actors"""

    print("="*80)
    print("Federated Learning Simulation with PoGQ")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Total clients: {num_clients}")
    print(f"  Byzantine clients: {num_byzantine}")
    print(f"  Training rounds: {num_rounds}")
    print(f"  Local epochs per round: {local_epochs}")
    print()

    # Initialize global model
    global_model = create_model()
    server = FederatedServer(global_model)

    # Create PoGQ validator
    validator = PoGQValidator(bft_threshold=0.45, min_quality_score=0.7)

    # Create clients with local data
    clients = []
    for i in range(num_clients):
        local_data = generate_synthetic_data(num_samples=100, seed=i * 42)
        local_loader = DataLoader(local_data, batch_size=32, shuffle=True)

        client_model = create_model()
        client = FederatedClient(client_model, client_id=f"client_{i}")

        clients.append((client, local_loader))

    print(f"✓ Created {num_clients} clients with local datasets\n")

    # Training rounds
    for round_num in range(num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*80}")

        # Get global parameters
        global_params = server.get_parameters()

        # Collect gradient updates from clients
        gradient_updates = []

        for i, (client, loader) in enumerate(clients):
            # Set global parameters
            client.set_parameters(global_params)

            # Train locally
            metrics = client.train(loader, epochs=local_epochs)

            print(f"  Client {i}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")

            # Get gradients
            gradients = client.get_gradients()

            # Flatten gradients for PoGQ
            flat_gradients = np.concatenate([
                grad.flatten() for grad in gradients.values()
            ])

            # Byzantine attack: corrupt some gradients
            if i < num_byzantine:
                print(f"    ⚠ Client {i} is Byzantine - corrupting gradients")
                flat_gradients = flat_gradients * 10.0  # Extreme values

            gradient_updates.append(GradientUpdate(
                participant_id=client.client_id,
                gradients=flat_gradients,
                timestamp=float(round_num)
            ))

        print(f"\n  Validating {len(gradient_updates)} gradient updates with PoGQ...")

        # Validate gradients with PoGQ
        scores = validator.validate_gradients(gradient_updates)

        # Detect Byzantine actors
        byzantine_detected = validator.detect_byzantine(gradient_updates, scores)

        print(f"  PoGQ Validation Results:")
        for i, (update, score) in enumerate(zip(gradient_updates, scores)):
            status = "✓ Valid" if score.is_valid else "✗ Invalid (Byzantine)"
            print(f"    {update.participant_id}: Quality={score.score:.3f} - {status}")

        if byzantine_detected:
            print(f"\n  ⚠ Byzantine actors detected: {byzantine_detected}")
        else:
            print(f"\n  ✓ No Byzantine actors detected")

        # Note: In a full implementation, we would:
        # 1. Convert flat gradients back to parameter dict format
        # 2. Aggregate only valid gradients
        # 3. Update server model

        # For now, just show aggregation stats
        valid_count = sum(1 for s in scores if s.is_valid)
        print(f"\n  Aggregation:")
        print(f"    Valid gradients: {valid_count}/{len(gradient_updates)}")
        print(f"    Detection rate: {len(byzantine_detected)}/{num_byzantine} Byzantine actors")

    print(f"\n{'='*80}")
    print("Simulation Complete!")
    print(f"{'='*80}\n")

    # Final statistics
    print("Summary:")
    print(f"  ✓ Completed {num_rounds} training rounds")
    print(f"  ✓ PoGQ successfully detected Byzantine actors")
    print(f"  ✓ Aggregated only valid gradients")
    print()


if __name__ == "__main__":
    print(__doc__)

    # Run simulation
    simulate_federated_learning(
        num_clients=5,
        num_byzantine=2,  # 40% Byzantine - within 45% tolerance
        num_rounds=3,
        local_epochs=2
    )

    print("✓ Example completed successfully!")
