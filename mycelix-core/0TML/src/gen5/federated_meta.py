# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Federated Meta-Learning - Gen 5 Layer 1 Extension
==================================================

Enables privacy-preserving, distributed optimization of ensemble weights
via local gradient computation and Byzantine-robust aggregation.

Key Features:
- Differential privacy via Gaussian mechanism
- Byzantine-robust aggregation (Krum, TrimmedMean, Median)
- Heterogeneity-aware federated optimization
- Reputation-weighted aggregation

Components:
1. LocalMetaLearner - Agent-side local gradient computation
2. FederatedMetaLearningEnsemble - Coordinator-side aggregation

Author: Luminous Dynamics
Date: November 11, 2025
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class LocalMetaLearnerConfig:
    """Configuration for local meta-learning."""

    learning_rate: float = 0.01
    privacy_budget: float = 8.0  # DP epsilon
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.privacy_budget > 0, "Privacy budget must be positive"
        assert self.gradient_clip_norm > 0, "Clip norm must be positive"


class LocalMetaLearner:
    """
    Agent-side local meta-learning with differential privacy.

    Computes local gradient updates for ensemble weights based on local
    view of detection performance, without sharing raw data.

    Privacy Guarantee:
        Provides (ε, 0)-differential privacy via Gaussian mechanism:
        - Gradient clipping bounds sensitivity: Δf ≤ clip_norm
        - Gaussian noise: σ = clip_norm / ε
        - Each update consumes ε privacy budget

    Example:
        >>> # Agent setup
        >>> local_learner = LocalMetaLearner(
        ...     method_names=["pogq", "fltrust", "krum"],
        ...     learning_rate=0.01,
        ...     privacy_budget=8.0
        ... )
        >>>
        >>> # Agent observes local signals and labels
        >>> local_signals = np.array([
        ...     [0.8, 0.9, 0.7],  # Sample 1: honest-like
        ...     [0.2, 0.3, 0.4],  # Sample 2: Byzantine-like
        ... ])
        >>> local_labels = np.array([1.0, 0.0])
        >>>
        >>> # Compute DP-noised gradient
        >>> current_weights = np.array([0.33, 0.33, 0.34])
        >>> meta_gradient = local_learner.compute_local_gradient(
        ...     signals=local_signals,
        ...     labels=local_labels,
        ...     current_weights=current_weights
        ... )
        >>> # Returns: Noised gradient vector for privacy
        >>> # Shape: (3,) - one gradient per method
    """

    def __init__(
        self,
        method_names: List[str],
        learning_rate: float = 0.01,
        privacy_budget: float = 8.0,
        gradient_clip_norm: float = 1.0,
    ):
        """
        Initialize local meta-learner.

        Args:
            method_names: Detection method names (must match ensemble)
            learning_rate: Local gradient descent learning rate
            privacy_budget: DP epsilon for gradient noise
            gradient_clip_norm: Gradient clipping threshold
        """
        self.method_names = method_names
        self.n_methods = len(method_names)

        self.config = LocalMetaLearnerConfig(
            learning_rate=learning_rate,
            privacy_budget=privacy_budget,
            gradient_clip_norm=gradient_clip_norm,
        )

        # Privacy tracking
        self.privacy_consumed = 0.0
        self.n_updates = 0

    def compute_local_gradient(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute DP-noised local meta-gradient.

        Algorithm:
            1. Compute ensemble scores: s = signals @ weights
            2. Binary cross-entropy loss: L = -[y log(s) + (1-y) log(1-s)]
            3. Gradient: ∇L/∂w = Σ[(s - y) × signal] / n
            4. Clip: grad = grad / max(1, ||grad|| / clip_norm)
            5. Add noise: grad += N(0, σ²) where σ² = (clip_norm / ε)²

        Privacy Guarantee:
            Each call consumes privacy_budget ε via Gaussian mechanism.
            Total privacy: ε_total = n_updates × ε

        Args:
            signals: Local detection signals (n_samples, n_methods)
            labels: Local ground truth (n_samples,) - 1.0 for honest, 0.0 for Byzantine
            current_weights: Current global ensemble weights (n_methods,)

        Returns:
            Noised meta-gradient (n_methods,)

        Raises:
            ValueError: If shapes don't match
        """
        if signals.ndim != 2:
            raise ValueError(f"Signals must be 2D, got shape {signals.shape}")

        if signals.shape[1] != self.n_methods:
            raise ValueError(
                f"Signals must have {self.n_methods} methods, "
                f"got {signals.shape[1]}"
            )

        if len(labels) != len(signals):
            raise ValueError(
                f"Labels ({len(labels)}) must match signals ({len(signals)})"
            )

        if len(current_weights) != self.n_methods:
            raise ValueError(
                f"Weights must have {self.n_methods} methods, "
                f"got {len(current_weights)}"
            )

        # 1. Compute ensemble scores
        ensemble_scores = signals @ current_weights

        # 2. Clip scores for numerical stability
        eps = 1e-7
        ensemble_scores = np.clip(ensemble_scores, eps, 1.0 - eps)

        # 3. Compute gradient of BCE loss
        # L = -[y log(s) + (1-y) log(1-s)]
        # ∇L/∂w = Σ[(s - y) × signal] / n
        residuals = ensemble_scores - labels  # (n_samples,)
        gradient = signals.T @ residuals / len(signals)  # (n_methods,)

        # 4. Gradient clipping (for bounded sensitivity)
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.config.gradient_clip_norm:
            gradient = gradient * (self.config.gradient_clip_norm / gradient_norm)

        # 5. Add Gaussian noise for DP
        noise_std = self.config.gradient_clip_norm / self.config.privacy_budget
        noise = np.random.normal(0, noise_std, size=gradient.shape)
        noised_gradient = gradient + noise

        # Track privacy consumption
        self.privacy_consumed += self.config.privacy_budget
        self.n_updates += 1

        return noised_gradient

    def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Get privacy consumption statistics.

        Returns:
            Dictionary with:
                - privacy_budget: Per-update ε
                - privacy_consumed: Total ε used
                - n_updates: Number of gradient computations
                - privacy_remaining: Remaining budget (if total set)
        """
        return {
            "privacy_budget": self.config.privacy_budget,
            "privacy_consumed": self.privacy_consumed,
            "n_updates": self.n_updates,
            "gradient_clip_norm": self.config.gradient_clip_norm,
        }

    def reset_privacy_tracking(self) -> None:
        """Reset privacy consumption counters."""
        self.privacy_consumed = 0.0
        self.n_updates = 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LocalMetaLearner(\n"
            f"  methods={self.n_methods},\n"
            f"  ε={self.config.privacy_budget:.2f},\n"
            f"  clip_norm={self.config.gradient_clip_norm:.2f},\n"
            f"  privacy_consumed={self.privacy_consumed:.2f},\n"
            f"  updates={self.n_updates}\n"
            f")"
        )


# Byzantine-robust aggregation methods
def aggregate_gradients_krum(
    gradients: List[np.ndarray],
    f: Optional[int] = None,
) -> np.ndarray:
    """
    Multi-Krum aggregation on meta-gradients.

    Krum selects the gradient with minimum sum of distances to its
    f nearest neighbors, where f is the Byzantine tolerance.

    Algorithm:
        For each gradient g_i:
            1. Compute distances to all other gradients
            2. Sum distances to f nearest neighbors
            3. Krum score = sum of f nearest distances
        Select gradient with minimum Krum score

    Byzantine Tolerance:
        Tolerates f < n/3 Byzantine agents.
        Honest agents' gradients cluster together.
        Byzantine gradients are outliers → high Krum scores.

    Args:
        gradients: List of meta-gradients from agents (each: n_methods,)
        f: Byzantine tolerance (default: len(gradients) // 3)

    Returns:
        Selected gradient (n_methods,)

    Example:
        >>> # 7 honest + 3 Byzantine agents
        >>> honest_grads = [np.array([0.1, 0.2, 0.15])] * 7
        >>> byzantine_grads = [np.array([10.0, -5.0, 8.0])] * 3
        >>> all_grads = honest_grads + byzantine_grads
        >>>
        >>> aggregated = aggregate_gradients_krum(all_grads, f=3)
        >>> # Returns gradient from honest cluster
    """
    if len(gradients) == 0:
        raise ValueError("No gradients to aggregate")

    n = len(gradients)
    if f is None:
        f = n // 3

    if f >= n:
        raise ValueError(f"f ({f}) must be < n ({n})")

    # Convert to array for easier computation
    grad_array = np.array(gradients)  # (n, n_methods)

    # Compute pairwise distances
    krum_scores = []
    for i in range(n):
        # Euclidean distance to all other gradients
        distances = np.linalg.norm(grad_array - grad_array[i], axis=1)

        # Sort and sum f nearest distances
        sorted_distances = np.sort(distances)
        krum_score = np.sum(sorted_distances[1:f+1])  # Exclude self (distance=0)

        krum_scores.append(krum_score)

    # Select gradient with minimum Krum score
    best_idx = np.argmin(krum_scores)

    return gradients[best_idx]


def aggregate_gradients_trimmed_mean(
    gradients: List[np.ndarray],
    trim_ratio: float = 0.1,
) -> np.ndarray:
    """
    Trimmed mean aggregation (coordinate-wise).

    For each coordinate:
        1. Sort values across all gradients
        2. Remove top and bottom trim_ratio fraction
        3. Average remaining values

    Byzantine Tolerance:
        Tolerates up to trim_ratio fraction of Byzantine agents.
        Robust to outliers in any direction.

    Args:
        gradients: List of meta-gradients (each: n_methods,)
        trim_ratio: Fraction to trim from each end (default: 10%)

    Returns:
        Aggregated gradient (n_methods,)

    Example:
        >>> grads = [
        ...     np.array([0.1, 0.2]),  # Honest
        ...     np.array([0.12, 0.18]),  # Honest
        ...     np.array([0.09, 0.21]),  # Honest
        ...     np.array([10.0, -5.0]),  # Byzantine outlier
        ... ]
        >>> aggregated = aggregate_gradients_trimmed_mean(grads, trim_ratio=0.25)
        >>> # Trims top/bottom 25%, averages middle 50%
        >>> # Removes Byzantine outlier
    """
    if len(gradients) == 0:
        raise ValueError("No gradients to aggregate")

    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0.0, 0.5)")

    # Stack gradients (n, n_methods)
    grad_array = np.array(gradients)
    n, n_methods = grad_array.shape

    # Number to trim from each end
    n_trim = int(np.floor(n * trim_ratio))

    # Sort along agent axis (axis=0) for each method
    sorted_grads = np.sort(grad_array, axis=0)

    # Trim top and bottom
    if n_trim > 0:
        trimmed_grads = sorted_grads[n_trim:-n_trim, :]
    else:
        trimmed_grads = sorted_grads

    # Average remaining gradients
    aggregated = np.mean(trimmed_grads, axis=0)

    return aggregated


def aggregate_gradients_median(gradients: List[np.ndarray]) -> np.ndarray:
    """
    Coordinate-wise median aggregation.

    For each coordinate, compute median across all agents.

    Byzantine Tolerance:
        Tolerates up to 50% Byzantine agents (breakdown point).
        Extremely robust to outliers.

    Args:
        gradients: List of meta-gradients (each: n_methods,)

    Returns:
        Aggregated gradient (n_methods,)

    Example:
        >>> grads = [
        ...     np.array([0.1, 0.2]),
        ...     np.array([0.12, 0.19]),
        ...     np.array([10.0, -5.0]),  # Byzantine outlier
        ... ]
        >>> aggregated = aggregate_gradients_median(grads)
        >>> # Returns: [0.12, 0.19] - median ignores outlier
    """
    if len(gradients) == 0:
        raise ValueError("No gradients to aggregate")

    # Stack gradients (n, n_methods)
    grad_array = np.array(gradients)

    # Median along agent axis (axis=0)
    aggregated = np.median(grad_array, axis=0)

    return aggregated


def aggregate_gradients_reputation_weighted(
    gradients: List[np.ndarray],
    reputations: np.ndarray,
) -> np.ndarray:
    """
    Reputation-weighted average aggregation.

    Aggregated = Σ(reputation_i × gradient_i) / Σ(reputation_i)

    Gives higher weight to trusted agents (high reputation).

    Byzantine Tolerance:
        Depends on reputation system quality.
        Assumes Byzantine agents have low reputation.

    Args:
        gradients: List of meta-gradients (each: n_methods,)
        reputations: Agent reputation scores (n_agents,)

    Returns:
        Weighted aggregated gradient (n_methods,)

    Example:
        >>> grads = [
        ...     np.array([0.1, 0.2]),   # High-reputation agent
        ...     np.array([10.0, -5.0]),  # Low-reputation Byzantine
        ... ]
        >>> reps = np.array([0.9, 0.1])  # Reputations
        >>> aggregated = aggregate_gradients_reputation_weighted(grads, reps)
        >>> # Returns: mostly first gradient (weighted 90%)
    """
    if len(gradients) == 0:
        raise ValueError("No gradients to aggregate")

    if len(reputations) != len(gradients):
        raise ValueError(
            f"Reputations ({len(reputations)}) must match "
            f"gradients ({len(gradients)})"
        )

    if np.any(reputations < 0):
        raise ValueError("Reputations must be non-negative")

    # Stack gradients (n, n_methods)
    grad_array = np.array(gradients)

    # Normalize reputations
    rep_sum = np.sum(reputations)
    if rep_sum == 0:
        # All zero reputations → uniform average
        return np.mean(grad_array, axis=0)

    normalized_reps = reputations / rep_sum

    # Weighted average: (n_methods,)
    aggregated = grad_array.T @ normalized_reps

    return aggregated
