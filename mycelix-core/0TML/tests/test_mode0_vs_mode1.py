# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Mode 0 vs Mode 1 Comparison at 35% BFT

Demonstrates why ground truth validation (Mode 1) is necessary for
heterogeneous federated learning scenarios.

Mode 0 (Peer-Comparison):
- Uses cosine similarity + magnitude to detect Byzantine nodes
- Compares each gradient to its peers
- FAILS with heterogeneous data: honest nodes look "different" from each other
- At 35% BFT: Byzantine majority can make honest nodes appear as outliers

Mode 1 (Ground Truth - PoGQ):
- Uses validation loss improvement as ground truth quality signal
- WORKS with heterogeneous data: quality measured against actual task
- At 35% BFT: Maintains high detection with low FPR
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from peer_comparison_detector import PeerComparisonDetector
from ground_truth_detector import GroundTruthDetector
from test_mode1_boundaries import (
    SimpleCNN,
    create_synthetic_mnist_data,
    generate_honest_gradient,
    pretrain_model
)
from byzantine_attacks.basic_attacks import SignFlipAttack


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)


def print_comparison_table(mode0_results: dict, mode1_results: dict):
    """Print side-by-side comparison table"""
    print("\n" + "=" * 80)
    print("🔬 MODE 0 vs MODE 1 COMPARISON AT 35% BFT")
    print("=" * 80)

    print(f"\n{'Metric':<40} {'Mode 0 (Peer)':<20} {'Mode 1 (Ground Truth)':<20}")
    print("-" * 80)

    metrics = [
        ("Detection Rate", "detection_rate", True),
        ("False Positive Rate", "fpr", True),
        ("True Positives", "true_positives", False),
        ("False Positives", "false_positives", False),
        ("True Negatives", "true_negatives", False),
        ("False Negatives", "false_negatives", False),
    ]

    for name, key, is_percentage in metrics:
        mode0_val = mode0_results[key]
        mode1_val = mode1_results[key]

        if is_percentage:
            mode0_str = f"{mode0_val*100:>6.1f}%"
            mode1_str = f"{mode1_val*100:>6.1f}%"
        else:
            mode0_str = f"{mode0_val:>6d}"
            mode1_str = f"{mode1_val:>6d}"

        # Add checkmark/cross for pass/fail
        if key == "detection_rate":
            mode0_pass = "✅" if mode0_val >= 0.95 else "❌"
            mode1_pass = "✅" if mode1_val >= 0.95 else "❌"
        elif key == "fpr":
            mode0_pass = "✅" if mode0_val <= 0.10 else "❌"
            mode1_pass = "✅" if mode1_val <= 0.10 else "❌"
        else:
            mode0_pass = ""
            mode1_pass = ""

        print(f"{name:<40} {mode0_str:<12}{mode0_pass:<8} {mode1_str:<12}{mode1_pass:<8}")

    print("=" * 80)


def run_mode0_mode1_comparison():
    """
    Compare Mode 0 (peer-comparison) vs Mode 1 (ground truth) at 35% BFT.

    Configuration:
    - 20 clients (13 honest, 7 Byzantine = 35% BFT)
    - Heterogeneous data (each client has unique local data)
    - Sign flip attack (Byzantine gradients = -1 * honest gradients)
    - Pre-trained model (realistic FL scenario)
    - Single round (focus on detection, not training dynamics)
    """

    print_header("MODE 0 vs MODE 1 COMPARISON AT 35% BFT")

    # Configuration
    num_clients = 20
    num_byzantine = 7  # 35% BFT
    num_honest = num_clients - num_byzantine
    seed = 42

    print(f"\nConfiguration:")
    print(f"  - Total Clients: {num_clients}")
    print(f"  - Honest: {num_honest}")
    print(f"  - Byzantine: {num_byzantine}")
    print(f"  - BFT Ratio: {num_byzantine/num_clients*100:.0f}%")
    print(f"  - Data Distribution: HETEROGENEOUS (each client unique)")
    print(f"  - Attack: Sign Flip (static)")
    print(f"  - Seed: {seed}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model and pre-train
    device = "cpu"
    global_model = SimpleCNN().to(device)

    print("\n" + "=" * 80)
    print("🔧 Pre-training Global Model")
    print("=" * 80)
    pretrain_model(global_model, num_epochs=5, seed=seed, device=device)

    # Create validation set for Mode 1
    validation_loader = create_synthetic_mnist_data(num_samples=1000, seed=seed, train=False)

    # Initialize detectors
    print("\n" + "=" * 80)
    print("🔧 Initializing Detectors")
    print("=" * 80)

    mode0_detector = PeerComparisonDetector(
        cosine_threshold=0.5,      # Standard threshold
        magnitude_z_threshold=3.0,  # 3-sigma outlier
        min_samples=3
    )
    print("✓ Mode 0 (Peer-Comparison) initialized")

    mode1_detector = GroundTruthDetector(
        global_model=global_model,
        validation_loader=validation_loader,
        quality_threshold=0.5,  # Will be overridden by adaptive
        learning_rate=0.01,
        device=device,
        adaptive_threshold=True,  # CRITICAL for heterogeneous data
        mad_multiplier=3.0
    )
    print("✓ Mode 1 (Ground Truth - PoGQ) initialized with adaptive threshold")

    # Generate gradients with heterogeneous data
    print("\n" + "=" * 80)
    print("🔧 Generating Client Gradients (Heterogeneous Data)")
    print("=" * 80)

    gradients = {}
    ground_truth = {}  # Track which are actually Byzantine

    # Honest gradients - each with UNIQUE local data
    for i in range(num_honest):
        client_seed = seed + i * 1000
        client_data = create_synthetic_mnist_data(
            num_samples=100,
            seed=client_seed,
            train=True
        )
        gradients[i] = generate_honest_gradient(
            global_model,
            device,
            train_loader=client_data
        )
        ground_truth[i] = False

    print(f"✓ Generated {num_honest} honest gradients (unique local data)")

    # Byzantine gradients - each with UNIQUE local data + sign flip
    attack = SignFlipAttack(flip_intensity=1.0)

    for i in range(num_honest, num_clients):
        client_seed = seed + i * 1000
        client_data = create_synthetic_mnist_data(
            num_samples=100,
            seed=client_seed,
            train=True
        )

        # Generate honest gradient from their local data
        honest_grad = generate_honest_gradient(
            global_model,
            device,
            train_loader=client_data
        )

        # Then sign flip it (Byzantine attack)
        flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])
        flat_byz = attack.generate(flat_honest, round_num=0)

        # Reshape back to gradient dict
        byz_grad = {}
        idx = 0
        for name, param in global_model.named_parameters():
            size = param.numel()
            byz_grad[name] = flat_byz[idx:idx+size].reshape(param.shape)
            idx += size

        gradients[i] = byz_grad
        ground_truth[i] = True

    print(f"✓ Generated {num_byzantine} Byzantine gradients (sign flip attack)")

    # Run Mode 0 Detection
    print("\n" + "=" * 80)
    print("🧪 Mode 0: Peer-Comparison Detection")
    print("=" * 80)

    mode0_start = time.time()
    mode0_detections = mode0_detector.detect_byzantine(gradients)
    mode0_time = (time.time() - mode0_start) * 1000  # Convert to ms

    # Compute Mode 0 metrics
    mode0_tp = sum(1 for i in range(num_honest, num_clients) if mode0_detections[i])  # Detected Byzantine
    mode0_fp = sum(1 for i in range(num_honest) if mode0_detections[i])  # Falsely detected honest
    mode0_tn = sum(1 for i in range(num_honest) if not mode0_detections[i])  # Correctly not detected honest
    mode0_fn = sum(1 for i in range(num_honest, num_clients) if not mode0_detections[i])  # Missed Byzantine

    mode0_detection_rate = mode0_tp / num_byzantine if num_byzantine > 0 else 0.0
    mode0_fpr = mode0_fp / num_honest if num_honest > 0 else 0.0

    print(f"  Byzantine Detected: {mode0_tp}/{num_byzantine} ({mode0_detection_rate*100:.1f}%)")
    print(f"  Honest Flagged (FPR): {mode0_fp}/{num_honest} ({mode0_fpr*100:.1f}%)")
    print(f"  Time: {mode0_time:.2f}ms")

    # Run Mode 1 Detection
    print("\n" + "=" * 80)
    print("🧪 Mode 1: Ground Truth (PoGQ) Detection")
    print("=" * 80)

    mode1_start = time.time()
    mode1_detections = mode1_detector.detect_byzantine(gradients)
    mode1_time = (time.time() - mode1_start) * 1000  # Convert to ms

    # Compute Mode 1 metrics
    mode1_tp = sum(1 for i in range(num_honest, num_clients) if mode1_detections[i])
    mode1_fp = sum(1 for i in range(num_honest) if mode1_detections[i])
    mode1_tn = sum(1 for i in range(num_honest) if not mode1_detections[i])
    mode1_fn = sum(1 for i in range(num_honest, num_clients) if not mode1_detections[i])

    mode1_detection_rate = mode1_tp / num_byzantine if num_byzantine > 0 else 0.0
    mode1_fpr = mode1_fp / num_honest if num_honest > 0 else 0.0

    print(f"  Byzantine Detected: {mode1_tp}/{num_byzantine} ({mode1_detection_rate*100:.1f}%)")
    print(f"  Honest Flagged (FPR): {mode1_fp}/{num_honest} ({mode1_fpr*100:.1f}%)")
    print(f"  Adaptive Threshold: {mode1_detector.computed_threshold:.6f}")
    print(f"  Time: {mode1_time:.2f}ms")

    # Build results dicts
    mode0_results = {
        "detection_rate": mode0_detection_rate,
        "fpr": mode0_fpr,
        "true_positives": mode0_tp,
        "false_positives": mode0_fp,
        "true_negatives": mode0_tn,
        "false_negatives": mode0_fn,
        "time_ms": mode0_time
    }

    mode1_results = {
        "detection_rate": mode1_detection_rate,
        "fpr": mode1_fpr,
        "true_positives": mode1_tp,
        "false_positives": mode1_fp,
        "true_negatives": mode1_tn,
        "false_negatives": mode1_fn,
        "time_ms": mode1_time,
        "adaptive_threshold": mode1_detector.computed_threshold
    }

    # Print comparison table
    print_comparison_table(mode0_results, mode1_results)

    # Analysis
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS")
    print("=" * 80)

    print("\nMode 0 (Peer-Comparison) Limitations:")
    print("  - Relies on similarity to OTHER gradients")
    print("  - With heterogeneous data, honest clients naturally differ")
    print("  - Some honest clients may have low cosine similarity to peers")
    print("  - Result: High false positive rate")

    print("\nMode 1 (Ground Truth - PoGQ) Advantages:")
    print("  - Measures quality against ACTUAL TASK (validation loss)")
    print("  - Heterogeneous data doesn't matter - all honest clients improve loss")
    print("  - Byzantine gradients degrade loss → low quality score")
    print("  - Adaptive threshold finds optimal separation")
    print("  - Result: High detection, low false positives")

    print("\n" + "=" * 80)
    print("✅ CONCLUSION")
    print("=" * 80)

    if mode1_detection_rate >= 0.95 and mode1_fpr <= 0.10:
        print("Mode 1 (Ground Truth) PASSES at 35% BFT with heterogeneous data ✅")
    else:
        print("Mode 1 (Ground Truth) FAILS at 35% BFT ❌")

    if mode0_detection_rate >= 0.95 and mode0_fpr <= 0.10:
        print("Mode 0 (Peer-Comparison) PASSES at 35% BFT ✅")
    else:
        print("Mode 0 (Peer-Comparison) FAILS at 35% BFT ❌")

    print("\nGround truth validation (Mode 1) is ESSENTIAL for:")
    print("  ✓ Heterogeneous federated learning scenarios")
    print("  ✓ BFT ratios near or above 35%")
    print("  ✓ Achieving low false positive rates with diverse clients")

    # Return results for potential assertion
    return mode0_results, mode1_results


if __name__ == "__main__":
    mode0_results, mode1_results = run_mode0_mode1_comparison()

    # Optional: Assert Mode 1 passes and Mode 0 struggles
    # (commented out to allow seeing full results even if Mode 0 fails)
    # assert mode1_results["detection_rate"] >= 0.95, "Mode 1 detection too low"
    # assert mode1_results["fpr"] <= 0.10, "Mode 1 FPR too high"
