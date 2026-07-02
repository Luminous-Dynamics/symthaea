# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine-resistant gradient aggregation algorithms for federated learning.

This module implements multiple aggregation strategies:
- FedAvg: Standard weighted averaging
- Krum: Select most representative gradient
- Trimmed Mean: Remove outliers and average
- Coordinate-wise Median: Median per parameter
- Hierarchical aggregation: Tree-structured application of a base algorithm
"""

import numpy as np
from typing import List


class FedAvg:
    """
    Federated Averaging (FedAvg) - Standard aggregation algorithm

    Simple weighted average of gradients based on reputation scores.
    Fast but vulnerable to Byzantine attacks.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float]
    ) -> np.ndarray:
        """
        Weighted average aggregation

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores for weighting

        Returns:
            Averaged gradient
        """
        weights = np.array(reputations) / np.sum(reputations)
        stacked = np.stack([g.flatten() for g in gradients])
        result = np.average(stacked, axis=0, weights=weights)
        return result.reshape(gradients[0].shape)


class Krum:
    """
    Krum aggregation - Byzantine-resistant selection

    Selects the gradient with smallest distance to its neighbors.
    Resistant to coordinated attacks.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float],
        num_byzantine: int = 2
    ) -> np.ndarray:
        """
        Krum aggregation: Select most representative gradient

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores
            num_byzantine: Estimated number of Byzantine nodes

        Returns:
            Selected gradient (most representative)
        """
        n = len(gradients)
        k = n - num_byzantine - 2  # Number of neighbors to consider

        if k <= 0:
            # Fallback to weighted average
            return FedAvg.aggregate(gradients, reputations)

        # Flatten gradients for distance calculation
        flat_gradients = [g.flatten() for g in gradients]

        # Calculate pairwise distances
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(flat_gradients[i] - flat_gradients[j])
                    distances.append(dist)

            # Sum of k smallest distances (weighted by reputation)
            distances = np.array(sorted(distances)[:k])
            score = np.sum(distances) * reputations[i]
            scores.append(score)

        # Select gradient with minimum score
        best_idx = np.argmin(scores)
        return gradients[best_idx]


class TrimmedMean:
    """
    Trimmed Mean aggregation - Remove outliers

    Removes extreme values before averaging.
    Robust to both large and small outliers.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float],
        trim_ratio: float = 0.2
    ) -> np.ndarray:
        """
        Trimmed mean: Remove extreme values and average

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores
            trim_ratio: Fraction to trim from each end (default 0.2 = 20%)

        Returns:
            Trimmed mean gradient
        """
        # Stack gradients
        stacked = np.stack([g.flatten() for g in gradients])

        # Weight by reputation
        weights = np.array(reputations) / np.sum(reputations)

        # Calculate trim size
        n = len(gradients)
        trim_size = int(n * trim_ratio)

        if trim_size == 0:
            # No trimming, just weighted average
            return FedAvg.aggregate(gradients, reputations)

        # For trimmed mean, we trim per-coordinate
        # This is simpler: sort all gradients by norm and trim
        norms = np.array([np.linalg.norm(g) for g in stacked])
        sorted_by_norm = np.argsort(norms)

        # Trim based on norms
        keep_indices = sorted_by_norm[trim_size:-trim_size if trim_size > 0 else None]
        trimmed = stacked[keep_indices]
        trimmed_weights = weights[keep_indices]

        # Normalize weights
        trimmed_weights = trimmed_weights / np.sum(trimmed_weights)

        # Weighted average
        result = np.average(trimmed, axis=0, weights=trimmed_weights)
        return result.reshape(gradients[0].shape)


class ReputationWeighted:
    """
    Reputation-weighted aggregation

    Simple weighted average using reputation scores.
    Alias for FedAvg for API consistency.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float]
    ) -> np.ndarray:
        """Reputation-weighted average (same as FedAvg)"""
        return FedAvg.aggregate(gradients, reputations)


class CoordinateMedian:
    """
    Coordinate-wise Median aggregation

    Computes median independently for each parameter.
    Very robust but can be slow for large models.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float]
    ) -> np.ndarray:
        """
        Coordinate-wise median aggregation

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores (for weighted median)

        Returns:
            Median gradient
        """
        # Stack gradients
        stacked = np.stack([g.flatten() for g in gradients])

        # Weighted median per coordinate
        weights = np.array(reputations) / np.sum(reputations)

        result = np.array([
            CoordinateMedian._weighted_median(stacked[:, i], weights)
            for i in range(stacked.shape[1])
        ])

        return result.reshape(gradients[0].shape)

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted median"""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)

        return sorted_values[median_idx]


class HierarchicalAggregator:
    """
    Hierarchical aggregation wrapper.

    Applies a base aggregation algorithm recursively in a tree structure.
    This reduces the effective complexity of Krum-like methods from
    O(n²) pairwise distance computations toward O(n logₖ n), where k is
    the branching factor.

    The grouping strategy is simple and deterministic: gradients are
    partitioned into contiguous groups of size up to branching_factor,
    each group is aggregated with the base algorithm, and the results are
    aggregated again until a single root gradient remains.
    """

    @staticmethod
    def aggregate(
        gradients: List[np.ndarray],
        reputations: List[float],
        base_algorithm: str = "krum",
        branching_factor: int = 10,
        **kwargs,
    ) -> np.ndarray:
        """
        Hierarchical aggregation using a base algorithm at each tree level.

        Args:
            gradients: List of gradient arrays.
            reputations: Reputation scores for weighting.
            base_algorithm: Base algorithm name understood by
                ``aggregate_gradients`` (e.g., ``"krum"``).
            branching_factor: Maximum number of children per internal node.
            **kwargs: Additional parameters forwarded to the base algorithm.

        Returns:
            Aggregated gradient computed at the root of the tree.
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")

        if branching_factor < 2:
            raise ValueError("branching_factor must be >= 2")

        if len(gradients) != len(reputations):
            raise ValueError("gradients and reputations must have same length")

        current_gradients = list(gradients)
        current_reputations = list(reputations)

        # Reduce in levels until we are within a single group.
        while len(current_gradients) > branching_factor:
            grouped_gradients: List[np.ndarray] = []
            grouped_reputations: List[float] = []
            total = len(current_gradients)

            for start in range(0, total, branching_factor):
                end = min(start + branching_factor, total)
                group_gradients = current_gradients[start:end]
                group_reputations = current_reputations[start:end]

                if not group_gradients:
                    continue

                # Derive sensible defaults for the base algorithm.
                base_kwargs = dict(kwargs)
                if base_algorithm == "krum" and "num_byzantine" not in base_kwargs:
                    # Local Byzantine tolerance per group.
                    max_byzantine = max(0, len(group_gradients) - 2)
                    base_kwargs["num_byzantine"] = max_byzantine

                aggregated_group = aggregate_gradients(
                    group_gradients,
                    group_reputations,
                    algorithm=base_algorithm,
                    **base_kwargs,
                )

                # Represent the group's reputation as the arithmetic mean of
                # its members. This keeps the API simple while preserving the
                # relative weighting between groups.
                averaged_reputation = float(np.mean(group_reputations))
                grouped_gradients.append(aggregated_group)
                grouped_reputations.append(averaged_reputation)

            current_gradients = grouped_gradients
            current_reputations = grouped_reputations

        # Final aggregation at the root level.
        final_kwargs = dict(kwargs)
        if base_algorithm == "krum" and "num_byzantine" not in final_kwargs:
            max_byzantine = max(0, len(current_gradients) - 2)
            final_kwargs["num_byzantine"] = max_byzantine

        return aggregate_gradients(
            current_gradients,
            current_reputations,
            algorithm=base_algorithm,
            **final_kwargs,
        )


# Convenience function for dynamic algorithm selection
def aggregate_gradients(
    gradients: List[np.ndarray],
    reputations: List[float],
    algorithm: str = "krum",
    **kwargs
) -> np.ndarray:
    """
    Aggregate gradients using specified algorithm

    Args:
        gradients: List of gradient arrays
        reputations: Reputation scores
        algorithm: One of "fedavg", "krum", "trimmed_mean", "median"
        **kwargs: Algorithm-specific parameters

    Returns:
        Aggregated gradient

    Examples:
        >>> agg = aggregate_gradients(grads, reps, "krum", num_byzantine=2)
        >>> agg = aggregate_gradients(grads, reps, "trimmed_mean", trim_ratio=0.1)
    """
    if algorithm == "fedavg":
        return FedAvg.aggregate(gradients, reputations)
    elif algorithm == "krum":
        return Krum.aggregate(gradients, reputations, **kwargs)
    elif algorithm == "trimmed_mean":
        return TrimmedMean.aggregate(gradients, reputations, **kwargs)
    elif algorithm == "median":
        return CoordinateMedian.aggregate(gradients, reputations)
    elif algorithm == "reputation_weighted":
        return ReputationWeighted.aggregate(gradients, reputations)
    elif algorithm == "hierarchical" or algorithm == "hierarchical_krum":
        # Default hierarchical mode uses Krum at each level.
        return HierarchicalAggregator.aggregate(
            gradients,
            reputations,
            base_algorithm="krum",
            **kwargs,
        )
    elif algorithm.startswith("hierarchical_"):
        # Allow forms like "hierarchical_trimmed_mean".
        base_algorithm = algorithm[len("hierarchical_") :]
        return HierarchicalAggregator.aggregate(
            gradients,
            reputations,
            base_algorithm=base_algorithm,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown aggregation algorithm: {algorithm}")
