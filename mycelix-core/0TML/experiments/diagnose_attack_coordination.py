#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Diagnostic: Why does Median maintain 87% at 45% Byzantine?

Hypothesis: Model replacement attack is UNCOORDINATED - each Byzantine client
pushes toward a different random target, causing their malicious gradients to
cancel out when median is taken.

This test verifies the hypothesis by analyzing gradient directions.
"""

import numpy as np
from pathlib import Path
import sys

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.datasets.common import make_emnist
from experiments.simulator import FLScenario, softmax_gradient


def diagnose_attack_coordination():
    """Analyze why Median is robust at 45% Byzantine."""

    print("\n" + "=" * 80)
    print("🔬 DIAGNOSIS: Attack Coordination Analysis")
    print("=" * 80)
    print()

    # Setup
    seed = 101
    n_clients = 50
    byz_frac = 0.45
    n_byz = int(n_clients * byz_frac)  # 22 Byzantine clients

    print(f"Configuration:")
    print(f"  Total clients: {n_clients}")
    print(f"  Byzantine fraction: {byz_frac} ({n_byz} clients)")
    print(f"  Attack: Model replacement with Λ=10")
    print()

    # Load EMNIST
    dataset = make_emnist(
        n_clients=n_clients,
        noniid_alpha=1.0,
        n_train=6000,
        n_test=1000,
        seed=seed
    )

    # Initialize global model
    W_global = np.zeros((dataset.n_features, dataset.n_classes))

    # Simulate one round of training
    print("Simulating round 1 local training...")

    # Honest gradients (train normally)
    honest_grads = []
    for client_idx in range(n_byz, n_clients):  # Honest clients
        X, y = dataset.train_splits[client_idx]
        W_local = W_global.copy()

        # 5 local epochs
        for _ in range(5):
            grad_W = softmax_gradient(X, y, W_local)
            W_local = W_local - 0.05 * grad_W

        client_grad = W_local - W_global
        honest_grads.append(client_grad.flatten())

    honest_grads = np.array(honest_grads)

    # Byzantine gradients (model replacement attack)
    print("\nGenerating Byzantine gradients with current attack...")
    byzantine_grads = []
    byzantine_targets = []

    for client_idx in range(n_byz):  # Byzantine clients
        # Current attack: DIFFERENT random target per client
        rng = np.random.default_rng(seed + client_idx)
        W_target = rng.standard_normal(W_global.shape) * 0.1
        byzantine_targets.append(W_target.flatten())

        # Malicious gradient: λ * (W_target - W_global)
        attack_grad = 10.0 * (W_target - W_global)
        byzantine_grads.append(attack_grad.flatten())

    byzantine_grads = np.array(byzantine_grads)
    byzantine_targets = np.array(byzantine_targets)

    # Analysis 1: Cosine similarity between Byzantine targets
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Byzantine Target Coordination")
    print("=" * 80)

    # Compute pairwise cosine similarities between Byzantine targets
    from sklearn.metrics.pairwise import cosine_similarity

    target_similarities = cosine_similarity(byzantine_targets)

    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(n_byz, dtype=bool)
    off_diag_similarities = target_similarities[mask]

    print(f"\nByzantine target pairwise cosine similarities:")
    print(f"  Mean: {off_diag_similarities.mean():.4f}")
    print(f"  Std:  {off_diag_similarities.std():.4f}")
    print(f"  Min:  {off_diag_similarities.min():.4f}")
    print(f"  Max:  {off_diag_similarities.max():.4f}")

    if off_diag_similarities.mean() < 0.1:
        print("\n❌ PROBLEM CONFIRMED: Byzantine targets are UNCOORDINATED!")
        print("   Each Byzantine client pushes toward a different random direction.")
        print("   Their malicious gradients cancel out when median is taken.")
    else:
        print("\n✅ Byzantine targets are coordinated.")

    # Analysis 2: Effect on median
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Effect on Median Aggregation")
    print("=" * 80)

    # Combine all gradients
    all_grads = np.vstack([honest_grads, byzantine_grads])

    # Compute median
    median_grad = np.median(all_grads, axis=0)

    # Compare median to honest mean
    honest_mean = np.mean(honest_grads, axis=0)

    # Cosine similarity between median and honest mean
    median_honest_sim = cosine_similarity(
        median_grad.reshape(1, -1),
        honest_mean.reshape(1, -1)
    )[0, 0]

    print(f"\nMedian vs Honest Mean:")
    print(f"  Cosine similarity: {median_honest_sim:.4f}")

    if median_honest_sim > 0.9:
        print("\n❌ ATTACK INEFFECTIVE: Median is ~identical to honest mean!")
        print("   Uncoordinated Byzantine gradients don't affect the median.")
        print("   This explains why Median maintains 87% accuracy.")
    else:
        print("\n✅ Attack is affecting the median.")

    # Analysis 3: Proposed fix - Coordinated attack
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Proposed Fix - Coordinated Attack")
    print("=" * 80)

    print("\nGenerating coordinated Byzantine gradients...")

    # FIXED attack: SAME target for all Byzantine clients
    rng_coord = np.random.default_rng(seed)  # Same seed for all
    W_target_coord = rng_coord.standard_normal(W_global.shape) * 0.1

    byzantine_grads_coord = []
    for _ in range(n_byz):
        attack_grad_coord = 10.0 * (W_target_coord - W_global)
        byzantine_grads_coord.append(attack_grad_coord.flatten())

    byzantine_grads_coord = np.array(byzantine_grads_coord)

    # Compute median with coordinated attack
    all_grads_coord = np.vstack([honest_grads, byzantine_grads_coord])
    median_grad_coord = np.median(all_grads_coord, axis=0)

    # Compare to honest mean
    median_honest_sim_coord = cosine_similarity(
        median_grad_coord.reshape(1, -1),
        honest_mean.reshape(1, -1)
    )[0, 0]

    print(f"\nCoordinated attack - Median vs Honest Mean:")
    print(f"  Cosine similarity: {median_honest_sim_coord:.4f}")

    if median_honest_sim_coord < 0.7:
        print("\n✅ COORDINATED ATTACK IS EFFECTIVE!")
        print("   Median is corrupted when Byzantine clients coordinate.")
        print("   This should cause Median to fail while AEGIS detects the attack.")
    else:
        print("\n⚠️  Even coordinated attack doesn't corrupt median much.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n🔍 Root Cause:")
    print("   The model replacement attack uses `seed + client_idx`, causing")
    print("   each Byzantine client to push toward a DIFFERENT random target.")
    print("   These uncoordinated malicious gradients cancel out in the median.")
    print()

    print("💡 Recommended Fix:")
    print("   Change attack to use SAME seed for all Byzantine clients:")
    print()
    print("   # BEFORE (uncoordinated):")
    print("   rng = np.random.default_rng(scenario.seed + idx)")
    print()
    print("   # AFTER (coordinated):")
    print("   rng = np.random.default_rng(scenario.seed)  # Same for all")
    print()

    print("📊 Expected Results After Fix:")
    print("   - Median accuracy at 45% Byzantine: <70% (should fail)")
    print("   - AEGIS accuracy at 45% Byzantine: ≥70% (should succeed)")
    print("   - This validates the 45% BFT claim")
    print()

    print("=" * 80)
    print()


if __name__ == "__main__":
    diagnose_attack_coordination()
