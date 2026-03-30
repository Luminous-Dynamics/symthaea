#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Simple Federated Learning Example with Mycelix FL

This example demonstrates:
1. Setting up a federated learning experiment
2. Simulating honest and Byzantine clients
3. Running FL rounds with Byzantine detection
4. Analyzing results

Usage:
    python examples/simple_fl_example.py
"""

import numpy as np
from typing import Dict
import time

# Import Mycelix FL
from mycelix_fl import (
    MycelixFL,
    FLConfig,
    RoundResult,
    setup_logging,
    has_rust_backend,
    get_backend_info,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def generate_client_gradients(
    num_honest: int,
    num_byzantine: int,
    gradient_dim: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate simulated client gradients.

    Args:
        num_honest: Number of honest clients
        num_byzantine: Number of Byzantine (malicious) clients
        gradient_dim: Dimension of gradient vectors
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping client_id -> gradient
    """
    rng = np.random.RandomState(seed)

    # True gradient direction (what honest clients are learning)
    true_gradient = rng.randn(gradient_dim).astype(np.float32)
    true_gradient /= np.linalg.norm(true_gradient)

    gradients = {}

    # Generate honest gradients (close to true direction)
    for i in range(num_honest):
        noise = rng.randn(gradient_dim).astype(np.float32) * 0.1
        gradients[f"honest_client_{i}"] = true_gradient + noise

    # Generate Byzantine gradients (various attacks)
    for i in range(num_byzantine):
        client_id = f"byzantine_client_{i}"

        # Rotate through attack types
        attack_type = i % 4

        if attack_type == 0:
            # Sign flip attack: negate the gradient
            gradients[client_id] = -true_gradient * 1.5
        elif attack_type == 1:
            # Random noise attack: completely random
            gradients[client_id] = rng.randn(gradient_dim).astype(np.float32) * 3.0
        elif attack_type == 2:
            # Scaling attack: extreme magnitude
            gradients[client_id] = true_gradient * 50.0
        else:
            # Little-is-enough: subtle attack
            gradients[client_id] = true_gradient + rng.randn(gradient_dim).astype(np.float32) * 0.5

    return gradients, true_gradient


def run_single_round(
    fl: MycelixFL,
    gradients: Dict[str, np.ndarray],
    round_num: int,
    actual_byzantine: set,
) -> Dict:
    """
    Run a single FL round and analyze results.

    Returns dictionary with metrics.
    """
    start_time = time.perf_counter()
    result = fl.execute_round(gradients, round_num=round_num)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Calculate detection metrics
    detected = result.byzantine_nodes
    true_positives = len(detected & actual_byzantine)
    false_positives = len(detected - actual_byzantine)
    false_negatives = len(actual_byzantine - detected)

    precision = true_positives / len(detected) if detected else 1.0
    recall = true_positives / len(actual_byzantine) if actual_byzantine else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "round_num": round_num,
        "detected_byzantine": len(detected),
        "actual_byzantine": len(actual_byzantine),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "phi_before": result.phi_before,
        "phi_after": result.phi_after,
        "phi_gain": result.phi_after - result.phi_before,
        "latency_ms": latency_ms,
        "healed_nodes": len(result.healed_nodes),
    }


def main():
    """Run the example."""
    print_header("Mycelix FL - Simple Example")

    # Check backend
    print("\n📊 Backend Information:")
    info = get_backend_info()
    print(f"  Version: {info['version']}")
    print(f"  Rust Acceleration: {'✅ Enabled' if info['rust_available'] else '❌ Disabled'}")
    if info['rust_available']:
        print(f"  Rust Version: {info['rust_version']}")
        print(f"  HV Dimension: {info['hv_dimension']}")

    # Enable logging for visibility
    setup_logging(level="WARNING")

    # Configuration
    num_honest = 14  # 70%
    num_byzantine = 6  # 30%
    gradient_dim = 10_000  # 10K parameters (typical small model)
    num_rounds = 5

    print_header("Experiment Configuration")
    print(f"  Honest clients: {num_honest}")
    print(f"  Byzantine clients: {num_byzantine}")
    print(f"  Byzantine ratio: {num_byzantine / (num_honest + num_byzantine):.0%}")
    print(f"  Gradient dimension: {gradient_dim:,}")
    print(f"  Number of rounds: {num_rounds}")

    # Create FL instance
    config = FLConfig(
        num_rounds=num_rounds,
        min_nodes=5,
        byzantine_threshold=0.45,  # Support up to 45% Byzantine
        learning_rate=0.01,
        enable_phi_tracking=True,
        enable_self_healing=True,
    )

    fl = MycelixFL(config=config)

    print_header("Running Federated Learning")

    # Track metrics across rounds
    all_metrics = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n📍 Round {round_num}/{num_rounds}")

        # Generate fresh gradients each round (simulating real training)
        gradients, true_gradient = generate_client_gradients(
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            gradient_dim=gradient_dim,
            seed=42 + round_num,  # Different seed each round
        )

        # Identify actual Byzantine nodes
        actual_byzantine = {k for k in gradients if "byzantine" in k}

        # Run round
        metrics = run_single_round(fl, gradients, round_num, actual_byzantine)
        all_metrics.append(metrics)

        # Print round results
        print(f"  Byzantine Detection:")
        print(f"    Detected: {metrics['detected_byzantine']}/{metrics['actual_byzantine']}")
        print(f"    Precision: {metrics['precision']:.1%}")
        print(f"    Recall: {metrics['recall']:.1%}")
        print(f"    F1 Score: {metrics['f1_score']:.1%}")
        print(f"  Φ Measurement:")
        print(f"    Before: {metrics['phi_before']:.4f}")
        print(f"    After: {metrics['phi_after']:.4f}")
        print(f"    Gain: {metrics['phi_gain']:+.4f}")
        if metrics['healed_nodes'] > 0:
            print(f"  Self-Healing: {metrics['healed_nodes']} nodes healed")
        print(f"  Latency: {metrics['latency_ms']:.1f}ms")

    # Summary
    print_header("Summary Statistics")

    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
    avg_latency = np.mean([m['latency_ms'] for m in all_metrics])
    total_healed = sum(m['healed_nodes'] for m in all_metrics)

    print(f"  Average Precision: {avg_precision:.1%}")
    print(f"  Average Recall: {avg_recall:.1%}")
    print(f"  Average F1 Score: {avg_f1:.1%}")
    print(f"  Average Latency: {avg_latency:.1f}ms")
    print(f"  Total Nodes Healed: {total_healed}")

    # Verdict
    print_header("Verdict")

    if avg_f1 >= 0.8:
        print("  ✅ EXCELLENT: Byzantine detection is highly effective!")
    elif avg_f1 >= 0.6:
        print("  ✅ GOOD: Byzantine detection is working well.")
    elif avg_f1 >= 0.4:
        print("  ⚠️  MODERATE: Some Byzantine nodes are slipping through.")
    else:
        print("  ❌ POOR: Byzantine detection needs improvement.")

    print("\n" + "=" * 60)
    print("  Example complete! ")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
