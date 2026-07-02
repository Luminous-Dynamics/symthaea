# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Threshold Parameter Sweep - Finding Optimal Ensemble Threshold

This test systematically evaluates different ensemble threshold values to find
the optimal balance between detection rate and false positive rate.

Thresholds to Test:
- 0.6 (Current): Known to produce 0% detection
- 0.5
- 0.4
- 0.3
- 0.2
- 0.1

Goal: Find threshold that achieves >80% detection with <10% FPR

Author: Zero-TrustML Research Team
Date: November 5, 2025
Status: Critical calibration for heterogeneous data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import pandas as pd

from src.byzantine_detection.hybrid_detector import HybridByzantineDetector
from src.byzantine_detection.gradient_analyzer import GradientDimensionalityAnalyzer
from test_mode1_boundaries import (
    SimpleCNN,
    create_synthetic_mnist_data,
    generate_honest_gradient,
    pretrain_model
)
from src.byzantine_attacks import create_attack, AttackType


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"🔬 {title}")
    print("=" * 80)


def run_threshold_test(
    threshold: float,
    num_clients: int = 20,
    num_byzantine: int = 6,  # 30% BFT
    seed: int = 42
) -> Dict:
    """
    Run detection test with specified ensemble threshold.

    Args:
        threshold: Ensemble confidence threshold (0.0-1.0)
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine clients
        seed: Random seed for reproducibility

    Returns:
        Dictionary with detection metrics
    """
    num_honest = num_clients - num_byzantine

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize model
    device = "cpu"
    global_model = SimpleCNN().to(device)

    # Pre-train (suppress output)
    import io
    import contextlib

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        pretrain_model(global_model, num_epochs=5, seed=seed, device=device)

    # Initialize detector with specified threshold
    detector = HybridByzantineDetector(
        temporal_window_size=5,
        temporal_cosine_var_threshold=0.1,
        temporal_magnitude_var_threshold=0.5,
        temporal_min_observations=1,
        magnitude_z_score_threshold=3.0,
        magnitude_min_samples=3,
        similarity_weight=0.5,
        temporal_weight=0.3,
        magnitude_weight=0.2,
        ensemble_threshold=threshold  # TEST THIS VALUE
    )

    gradient_analyzer = GradientDimensionalityAnalyzer()

    # Run 3 rounds
    total_byzantine_detected = 0
    total_honest_flagged = 0
    total_byzantine = 0
    total_honest = 0

    for round_num in range(3):
        # Generate gradients
        gradients_dict = {}

        # Honest gradients
        for i in range(num_honest):
            client_seed = seed + i * 1000 + round_num * 100000
            client_data = create_synthetic_mnist_data(100, client_seed, True)
            gradients_dict[i] = generate_honest_gradient(global_model, device, client_data)

        # Byzantine gradients
        for i in range(num_honest, num_clients):
            client_seed = seed + i * 1000 + round_num * 100000
            client_data = create_synthetic_mnist_data(100, client_seed, True)
            honest_grad = generate_honest_gradient(global_model, device, client_data)

            # Sign flip attack
            attack = create_attack(AttackType.SIGN_FLIP, flip_intensity=1.0)
            flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])
            flat_byz = attack.generate(flat_honest, round_num)

            # Reshape
            byz_grad = {}
            idx = 0
            for name, param in global_model.named_parameters():
                size = param.numel()
                byz_grad[name] = flat_byz[idx:idx+size].reshape(param.shape)
                idx += size

            gradients_dict[i] = byz_grad

        # Convert to tensors
        gradient_tensors = []
        node_ids = []
        for node_id in range(num_clients):
            grad_dict = gradients_dict[node_id]
            flat_grad = np.concatenate([g.flatten() for g in grad_dict.values()])
            gradient_tensors.append(torch.from_numpy(flat_grad).float())
            node_ids.append(node_id)

        # Analyze and detect
        profile = gradient_analyzer.analyze_gradients(gradient_tensors)
        decisions = detector.batch_analyze(gradient_tensors, node_ids, profile)

        # Compute metrics
        byzantine_detected = sum(
            1 for i in range(num_honest, num_clients)
            if decisions[i].is_byzantine
        )
        honest_flagged = sum(
            1 for i in range(num_honest)
            if decisions[i].is_byzantine
        )

        total_byzantine_detected += byzantine_detected
        total_honest_flagged += honest_flagged
        total_byzantine += num_byzantine
        total_honest += num_honest

    # Calculate rates
    detection_rate = total_byzantine_detected / total_byzantine if total_byzantine > 0 else 0.0
    fpr = total_honest_flagged / total_honest if total_honest > 0 else 0.0

    return {
        'threshold': threshold,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'byzantine_detected': total_byzantine_detected,
        'total_byzantine': total_byzantine,
        'honest_flagged': total_honest_flagged,
        'total_honest': total_honest
    }


def run_threshold_sweep():
    """
    Run parameter sweep across multiple threshold values.
    """
    print_header("THRESHOLD PARAMETER SWEEP")

    print("\nObjective: Find optimal ensemble threshold for heterogeneous data")
    print("Target: >80% detection rate with <10% false positive rate")
    print("\nTest Configuration:")
    print("  - Clients: 20 (14 honest, 6 Byzantine = 30% BFT)")
    print("  - Attack: Sign flip with heterogeneous data")
    print("  - Rounds: 3 per threshold")
    print("  - Seed: 42")
    print()

    # Thresholds to test
    thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    print(f"Testing {len(thresholds)} threshold values: {thresholds}")
    print()

    results = []

    for i, threshold in enumerate(thresholds):
        print(f"\n[{i+1}/{len(thresholds)}] Testing threshold = {threshold:.2f}...")
        result = run_threshold_test(threshold)
        results.append(result)

        print(f"  Detection Rate: {result['detection_rate']*100:>5.1f}% ({result['byzantine_detected']}/{result['total_byzantine']})")
        print(f"  False Positive Rate: {result['fpr']*100:>5.1f}% ({result['honest_flagged']}/{result['total_honest']})")

        # Check if this meets criteria
        if result['detection_rate'] >= 0.8 and result['fpr'] <= 0.1:
            print(f"  ✅ MEETS CRITERIA!")
        elif result['detection_rate'] >= 0.8:
            print(f"  ⚠️  Good detection but FPR too high")
        elif result['fpr'] <= 0.1:
            print(f"  ⚠️  Good FPR but detection too low")

    # Print summary table
    print_header("RESULTS SUMMARY")

    print("\n" + "-" * 80)
    print(f"{'Threshold':<12} {'Detection Rate':<18} {'FPR':<18} {'Status':<20}")
    print("-" * 80)

    best_threshold = None
    best_score = -float('inf')

    for result in results:
        threshold = result['threshold']
        detection = result['detection_rate']
        fpr = result['fpr']

        # Scoring: prioritize detection but penalize high FPR
        # Score = detection - 2*fpr (FPR is twice as bad)
        score = detection - 2 * fpr

        if detection >= 0.8 and fpr <= 0.1:
            status = "✅ OPTIMAL"
            if score > best_score:
                best_score = score
                best_threshold = threshold
        elif detection >= 0.7 and fpr <= 0.15:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ INSUFFICIENT"

        print(f"{threshold:<12.2f} {detection*100:>6.1f}% ({result['byzantine_detected']:>2}/{result['total_byzantine']:>2})"
              f"     {fpr*100:>6.1f}% ({result['honest_flagged']:>2}/{result['total_honest']:>2})"
              f"     {status:<20}")

    print("-" * 80)

    # Recommendation
    print_header("RECOMMENDATION")

    if best_threshold is not None:
        best_result = [r for r in results if r['threshold'] == best_threshold][0]
        print(f"\n✅ OPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")
        print(f"\nPerformance at threshold = {best_threshold:.2f}:")
        print(f"  - Detection Rate: {best_result['detection_rate']*100:.1f}%")
        print(f"  - False Positive Rate: {best_result['fpr']*100:.1f}%")
        print(f"\nRecommendation:")
        print(f"  Update `ensemble_threshold` in HybridByzantineDetector from 0.6 to {best_threshold:.2f}")
        print(f"  This configuration achieves excellent balance for heterogeneous data.")
    else:
        # Find best compromise
        print("\n⚠️  NO THRESHOLD MET STRICT CRITERIA (>80% detection, <10% FPR)")
        print("\nBest compromise:")

        # Sort by score
        sorted_results = sorted(results, key=lambda r: r['detection_rate'] - 2 * r['fpr'], reverse=True)
        best_compromise = sorted_results[0]

        print(f"  Threshold: {best_compromise['threshold']:.2f}")
        print(f"  Detection Rate: {best_compromise['detection_rate']*100:.1f}%")
        print(f"  False Positive Rate: {best_compromise['fpr']*100:.1f}%")
        print(f"\nThis represents the best balance achievable with current signal configuration.")

    print("\n" + "="*80 + "\n")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 THRESHOLD PARAMETER SWEEP - DETECTOR CALIBRATION")
    print("="*80 + "\n")

    print("This test will systematically evaluate multiple ensemble thresholds")
    print("to find the optimal configuration for heterogeneous federated learning.\n")

    results = run_threshold_sweep()

    print("Next Step: Update detector configuration with optimal threshold")
    print("Then re-run validation tests at 30% and 35% BFT\n")
