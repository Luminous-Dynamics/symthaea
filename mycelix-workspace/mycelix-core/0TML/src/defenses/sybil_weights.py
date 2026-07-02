# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 3: Sybil Resistance via FoolsGold + Cluster Penalty
===========================================================

Implements:
1. FoolsGold-style trust weights based on L-round gradient history
2. Simple cluster penalty (cosine similarity → scale by 1/√k)

Usage:
    from src.defenses.sybil_weights import compute_sybil_weights

    # In your aggregation defense
    weights = compute_sybil_weights(
        client_updates,
        history,  # List of past L rounds
        cosine_threshold=0.95,
        history_length=5,
        enabled=True  # Set to False to disable (no-op)
    )
    weighted_avg = sum(w * u for w, u in zip(weights, client_updates))

Reference:
    Fung et al. "The Limitations of Federated Learning in Sybil Settings"
    NeurIPS 2020 Workshop on Scalability, Privacy, and Security

Author: Luminous Dynamics
Date: November 8, 2025
Status: Scaffolding (Phase 3 preparation)
"""

import numpy as np
from typing import List, Optional, Dict
from collections import deque


def compute_sybil_weights(
    client_updates: List[np.ndarray],
    history: Optional[List[List[np.ndarray]]] = None,
    cosine_threshold: float = 0.95,
    history_length: int = 5,
    enabled: bool = False  # NO-OP flag for current run
) -> np.ndarray:
    """
    Compute client weights based on FoolsGold + cluster penalty.

    Args:
        client_updates: List of client gradient vectors (N × D)
        history: Past L rounds of updates [[round_0_clients], [round_1_clients], ...]
        cosine_threshold: Similarity threshold for cluster detection (default 0.95)
        history_length: Number of past rounds to consider (L)
        enabled: If False, returns uniform weights (no-op mode)

    Returns:
        weights: Array of shape (N,) with client weights

    Algorithm:
        1. FoolsGold: Track pairwise cosine similarities over history
           - Penalize clients with high repeated similarity
           - Trust = 1 / (1 + sum of cosine similarities with others)

        2. Cluster Penalty: Detect current-round clusters
           - Group clients with cosine > threshold
           - Scale cluster members by 1/√k

        3. Combine: weights = foolsgold_trust × cluster_penalty
    """
    num_clients = len(client_updates)

    # NO-OP mode: return uniform weights
    if not enabled:
        return np.ones(num_clients) / num_clients

    # Initialize weights
    weights = np.ones(num_clients)

    # ==================================================================
    # Part 1: FoolsGold Trust Scores (based on history)
    # ==================================================================
    if history and len(history) >= 2:
        foolsgold_trust = compute_foolsgold_trust(
            client_updates,
            history,
            history_length
        )
        weights *= foolsgold_trust

    # ==================================================================
    # Part 2: Cluster Penalty (current round)
    # ==================================================================
    cluster_penalty = compute_cluster_penalty(
        client_updates,
        cosine_threshold
    )
    weights *= cluster_penalty

    # Normalize to sum to 1
    weights /= weights.sum()

    return weights


def compute_foolsgold_trust(
    current_updates: List[np.ndarray],
    history: List[List[np.ndarray]],
    history_length: int = 5
) -> np.ndarray:
    """
    FoolsGold: Penalize clients with repeated high similarity.

    Returns:
        trust_scores: Array of shape (N,) with values in (0, 1]
    """
    num_clients = len(current_updates)

    # Take last L rounds
    recent_history = history[-history_length:] if len(history) > history_length else history

    if len(recent_history) < 2:
        # Not enough history, return uniform trust
        return np.ones(num_clients)

    # Compute pairwise cosine similarities over history
    cumulative_similarity = np.zeros((num_clients, num_clients))

    for past_round in recent_history:
        # Ensure past_round has same number of clients
        if len(past_round) != num_clients:
            continue

        # Compute cosine similarity matrix for this round
        cos_sim = compute_cosine_matrix(past_round)
        cumulative_similarity += np.abs(cos_sim)  # Absolute to catch anti-correlated sybils

    # Trust score: inverse of average similarity with others
    # trust_i = 1 / (1 + mean(similarity_i_j for all j != i))
    trust_scores = np.ones(num_clients)

    for i in range(num_clients):
        # Sum similarities with all other clients (exclude self)
        sim_sum = cumulative_similarity[i, :].sum() - cumulative_similarity[i, i]
        avg_sim = sim_sum / (num_clients - 1) if num_clients > 1 else 0

        # Trust inversely proportional to similarity
        trust_scores[i] = 1.0 / (1.0 + avg_sim)

    return trust_scores


def compute_cluster_penalty(
    client_updates: List[np.ndarray],
    cosine_threshold: float = 0.95
) -> np.ndarray:
    """
    Detect clusters in current round and penalize by 1/√k.

    Args:
        client_updates: List of client gradients
        cosine_threshold: Similarity threshold for cluster membership

    Returns:
        penalty: Array of shape (N,) with cluster penalties
    """
    num_clients = len(client_updates)

    # Compute cosine similarity matrix
    cos_sim = compute_cosine_matrix(client_updates)

    # Find clusters: greedy clustering based on threshold
    clusters = []
    assigned = [False] * num_clients

    for i in range(num_clients):
        if assigned[i]:
            continue

        # Start new cluster with client i
        cluster = [i]
        assigned[i] = True

        # Find all clients similar to i
        for j in range(i + 1, num_clients):
            if not assigned[j] and cos_sim[i, j] > cosine_threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    # Compute penalty: 1/√k for clusters of size k
    penalty = np.ones(num_clients)

    for cluster in clusters:
        k = len(cluster)
        if k > 1:  # Only penalize actual clusters
            cluster_penalty = 1.0 / np.sqrt(k)
            for client_id in cluster:
                penalty[client_id] = cluster_penalty

    return penalty


def compute_cosine_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        vectors: List of N vectors

    Returns:
        cos_sim: N × N matrix where cos_sim[i, j] = cosine(v_i, v_j)
    """
    num_vectors = len(vectors)

    # Stack into matrix (N × D)
    matrix = np.vstack(vectors)

    # Normalize to unit vectors
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / norms

    # Cosine similarity = dot product of normalized vectors
    cos_sim = matrix_normalized @ matrix_normalized.T

    return cos_sim


class SybilWeightTracker:
    """
    Stateful tracker for FoolsGold history across rounds.

    Usage:
        tracker = SybilWeightTracker(history_length=5)

        # Each round
        weights = tracker.compute_weights(client_updates, enabled=True)
        tracker.update_history(client_updates)
    """

    def __init__(self, history_length: int = 5):
        self.history_length = history_length
        self.history = deque(maxlen=history_length)

    def compute_weights(
        self,
        client_updates: List[np.ndarray],
        cosine_threshold: float = 0.95,
        enabled: bool = False
    ) -> np.ndarray:
        """Compute weights using current history."""
        return compute_sybil_weights(
            client_updates,
            history=list(self.history),
            cosine_threshold=cosine_threshold,
            history_length=self.history_length,
            enabled=enabled
        )

    def update_history(self, client_updates: List[np.ndarray]):
        """Add current round to history."""
        # Store copies to avoid mutation
        self.history.append([u.copy() for u in client_updates])

    def reset(self):
        """Clear history."""
        self.history.clear()


# ==================================================================
# Integration Helper: Drop-in for Existing Defenses
# ==================================================================

def apply_sybil_weighting(
    aggregated_gradient: np.ndarray,
    client_updates: List[np.ndarray],
    history: Optional[List[List[np.ndarray]]] = None,
    cosine_threshold: float = 0.95,
    history_length: int = 5,
    enabled: bool = False
) -> np.ndarray:
    """
    Re-weight an existing aggregated gradient with Sybil weights.

    This is a post-processing step that can be added to any defense:

    Example:
        # In your defense's aggregate() method
        aggregated = compute_median(client_updates)  # or any aggregation

        # Apply Sybil re-weighting
        aggregated = apply_sybil_weighting(
            aggregated,
            client_updates,
            history,
            enabled=config.get('sybil_defense', False)
        )

    Args:
        aggregated_gradient: Current aggregated result
        client_updates: List of client gradients
        history: Past rounds
        enabled: Sybil defense flag

    Returns:
        reweighted_gradient: Aggregated with Sybil weights applied
    """
    if not enabled:
        return aggregated_gradient

    # Compute Sybil weights
    weights = compute_sybil_weights(
        client_updates,
        history,
        cosine_threshold,
        history_length,
        enabled=True
    )

    # Re-weight: weighted average
    reweighted = sum(w * u for w, u in zip(weights, client_updates))

    return reweighted


# ==================================================================
# Testing & Verification
# ==================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sybil Weight Module: Unit Tests")
    print("=" * 70)

    # Test 1: NO-OP mode
    print("\nTest 1: NO-OP mode (enabled=False)")
    updates = [np.random.randn(10) for _ in range(5)]
    weights = compute_sybil_weights(updates, enabled=False)
    assert np.allclose(weights, np.ones(5) / 5), "NO-OP should return uniform weights"
    print("✅ NO-OP mode works (uniform weights)")

    # Test 2: Cluster detection
    print("\nTest 2: Cluster penalty")
    # Create a cluster: 3 identical gradients + 2 unique
    cluster = [np.array([1, 0, 0]) for _ in range(3)]
    unique = [np.array([0, 1, 0]), np.array([0, 0, 1])]
    updates = cluster + unique

    weights = compute_sybil_weights(updates, enabled=True, cosine_threshold=0.99)

    print(f"Weights: {weights}")
    print(f"Cluster (first 3) weight: {weights[0]:.4f}")
    print(f"Unique weight: {weights[3]:.4f}")

    # Cluster should be penalized
    assert weights[0] < weights[3], "Cluster should have lower weight"
    print("✅ Cluster penalty working")

    # Test 3: FoolsGold with history
    print("\nTest 3: FoolsGold trust scores")
    tracker = SybilWeightTracker(history_length=3)

    # Simulate 5 rounds with repeated sybil pattern
    sybil_gradient = np.array([1, 0, 0])
    for round_num in range(5):
        updates = [
            sybil_gradient + np.random.randn(3) * 0.1,  # Sybil 1
            sybil_gradient + np.random.randn(3) * 0.1,  # Sybil 2
            np.random.randn(3)  # Honest
        ]

        weights = tracker.compute_weights(updates, enabled=True)
        tracker.update_history(updates)

        print(f"Round {round_num}: weights = {weights}")

    print("✅ FoolsGold trust scores computed")

    print("\n" + "=" * 70)
    print("All tests passed! Sybil weight module ready for Phase 3.")
    print("=" * 70)
