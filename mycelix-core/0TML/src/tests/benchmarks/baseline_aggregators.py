# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Baseline FL Aggregation Methods for Comparison

Implements standard aggregation methods from FL literature:
- FedAvg: Simple averaging (McMahan et al., 2017)
- Median: Coordinate-wise median (Yin et al., 2018)
- TrimmedMean: Remove extreme values then average (Yin et al., 2018)
- MultiKrum: Byzantine-resilient aggregation (Blanchard et al., 2017)
- GeoMed: Geometric median (Chen et al., 2017)
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AggregationResult:
    """Result from an aggregation method."""
    aggregated_gradient: np.ndarray
    excluded_nodes: Set[str]
    weights: Dict[str, float]
    method_name: str


class BaseAggregator(ABC):
    """Abstract base class for aggregators."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        pass


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg: Simple averaging of all gradients.

    No Byzantine resilience - used as baseline.

    Reference: McMahan et al., "Communication-Efficient Learning
    of Deep Networks from Decentralized Data" (2017)
    """

    @property
    def name(self) -> str:
        return "FedAvg"

    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        if not gradients:
            return AggregationResult(
                aggregated_gradient=np.array([]),
                excluded_nodes=set(),
                weights={},
                method_name=self.name,
            )

        node_ids = list(gradients.keys())
        grad_array = np.stack([gradients[k] for k in node_ids])

        # Simple average
        aggregated = np.mean(grad_array, axis=0)

        # Equal weights
        n = len(node_ids)
        weights = {k: 1.0 / n for k in node_ids}

        return AggregationResult(
            aggregated_gradient=aggregated,
            excluded_nodes=set(),
            weights=weights,
            method_name=self.name,
        )


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise Median aggregation.

    Byzantine-resilient for < 50% attackers.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates" (2018)
    """

    @property
    def name(self) -> str:
        return "Median"

    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        if not gradients:
            return AggregationResult(
                aggregated_gradient=np.array([]),
                excluded_nodes=set(),
                weights={},
                method_name=self.name,
            )

        node_ids = list(gradients.keys())
        grad_array = np.stack([gradients[k] for k in node_ids])

        # Coordinate-wise median
        aggregated = np.median(grad_array, axis=0)

        # No explicit weights for median
        weights = {k: 1.0 / len(node_ids) for k in node_ids}

        return AggregationResult(
            aggregated_gradient=aggregated,
            excluded_nodes=set(),
            weights=weights,
            method_name=self.name,
        )


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean: Remove extreme values then average.

    Removes top and bottom trim_ratio of values per coordinate.
    Byzantine-resilient for < trim_ratio attackers.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning" (2018)
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction to trim from each end (default 10%)
        """
        self.trim_ratio = trim_ratio

    @property
    def name(self) -> str:
        return f"TrimmedMean({self.trim_ratio})"

    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        if not gradients:
            return AggregationResult(
                aggregated_gradient=np.array([]),
                excluded_nodes=set(),
                weights={},
                method_name=self.name,
            )

        node_ids = list(gradients.keys())
        grad_array = np.stack([gradients[k] for k in node_ids])
        n = len(node_ids)

        # Number to trim from each end
        k = int(n * self.trim_ratio)

        if k == 0 or 2 * k >= n:
            # Fall back to mean if can't trim
            aggregated = np.mean(grad_array, axis=0)
        else:
            # Sort and trim per coordinate
            sorted_grads = np.sort(grad_array, axis=0)
            trimmed = sorted_grads[k:n-k, :]
            aggregated = np.mean(trimmed, axis=0)

        weights = {k: 1.0 / len(node_ids) for k in node_ids}

        return AggregationResult(
            aggregated_gradient=aggregated,
            excluded_nodes=set(),
            weights=weights,
            method_name=self.name,
        )


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum: Select subset of gradients closest to others.

    Byzantine-resilient for < (n - m - 2) / (2n - m - 2) attackers,
    where m is the number of selected gradients.

    Reference: Blanchard et al., "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent" (2017)
    """

    def __init__(self, num_select: int = None, num_byzantine: int = None):
        """
        Args:
            num_select: Number of gradients to select (default: n - f - 2)
            num_byzantine: Assumed number of Byzantine nodes
        """
        self.num_select = num_select
        self.num_byzantine = num_byzantine

    @property
    def name(self) -> str:
        return "MultiKrum"

    def _compute_scores(
        self,
        grad_array: np.ndarray,
        num_byzantine: int,
    ) -> np.ndarray:
        """Compute Krum score for each gradient."""
        n = len(grad_array)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(grad_array[i] - grad_array[j])
                distances[i, j] = d
                distances[j, i] = d

        # For each gradient, sum distances to n - f - 2 closest others
        num_closest = n - num_byzantine - 2
        num_closest = max(1, num_closest)

        scores = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            # Exclude self (distance 0) and sum closest
            scores[i] = np.sum(sorted_dists[1:num_closest + 1])

        return scores

    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        if not gradients:
            return AggregationResult(
                aggregated_gradient=np.array([]),
                excluded_nodes=set(),
                weights={},
                method_name=self.name,
            )

        node_ids = list(gradients.keys())
        grad_array = np.stack([gradients[k] for k in node_ids])
        n = len(node_ids)

        # Estimate Byzantine count if not provided
        num_byzantine = self.num_byzantine or int(n * 0.3)

        # Number to select
        num_select = self.num_select or max(1, n - num_byzantine - 2)
        num_select = min(num_select, n)

        # Compute scores
        scores = self._compute_scores(grad_array, num_byzantine)

        # Select gradients with lowest scores
        selected_indices = np.argsort(scores)[:num_select]

        # Average selected
        aggregated = np.mean(grad_array[selected_indices], axis=0)

        # Record excluded
        excluded_indices = set(range(n)) - set(selected_indices)
        excluded_nodes = {node_ids[i] for i in excluded_indices}

        # Weights
        weights = {}
        for i, node_id in enumerate(node_ids):
            if i in selected_indices:
                weights[node_id] = 1.0 / num_select
            else:
                weights[node_id] = 0.0

        return AggregationResult(
            aggregated_gradient=aggregated,
            excluded_nodes=excluded_nodes,
            weights=weights,
            method_name=self.name,
        )


class GeoMedAggregator(BaseAggregator):
    """
    Geometric Median: Point minimizing sum of distances.

    Byzantine-resilient for < 50% attackers.
    Uses Weiszfeld's algorithm for computation.

    Reference: Chen et al., "Distributed Statistical Machine Learning
    in Adversarial Settings" (2017)
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-5):
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "GeoMed"

    def _weiszfeld(self, points: np.ndarray) -> np.ndarray:
        """Compute geometric median using Weiszfeld's algorithm."""
        # Initialize with coordinate-wise median
        estimate = np.median(points, axis=0)

        for _ in range(self.max_iter):
            # Compute distances
            distances = np.linalg.norm(points - estimate, axis=1)

            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)

            # Compute weights
            weights = 1.0 / distances
            weights /= np.sum(weights)

            # Update estimate
            new_estimate = np.sum(weights[:, np.newaxis] * points, axis=0)

            # Check convergence
            if np.linalg.norm(new_estimate - estimate) < self.tol:
                break

            estimate = new_estimate

        return estimate

    def aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> AggregationResult:
        if not gradients:
            return AggregationResult(
                aggregated_gradient=np.array([]),
                excluded_nodes=set(),
                weights={},
                method_name=self.name,
            )

        node_ids = list(gradients.keys())
        grad_array = np.stack([gradients[k] for k in node_ids])

        # Compute geometric median
        aggregated = self._weiszfeld(grad_array)

        weights = {k: 1.0 / len(node_ids) for k in node_ids}

        return AggregationResult(
            aggregated_gradient=aggregated,
            excluded_nodes=set(),
            weights=weights,
            method_name=self.name,
        )


# =============================================================================
# AGGREGATOR FACTORY
# =============================================================================

def create_aggregator(name: str, **kwargs) -> BaseAggregator:
    """Factory function to create aggregators."""
    aggregators = {
        "fedavg": FedAvgAggregator,
        "median": MedianAggregator,
        "trimmed_mean": TrimmedMeanAggregator,
        "multi_krum": MultiKrumAggregator,
        "geomed": GeoMedAggregator,
    }

    name_lower = name.lower().replace("-", "_")
    if name_lower not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}")

    return aggregators[name_lower](**kwargs)


def get_all_aggregators() -> List[BaseAggregator]:
    """Get all available aggregators for comparison."""
    return [
        FedAvgAggregator(),
        MedianAggregator(),
        TrimmedMeanAggregator(trim_ratio=0.1),
        TrimmedMeanAggregator(trim_ratio=0.2),
        MultiKrumAggregator(),
        GeoMedAggregator(),
    ]
