# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Data Partitioning Strategies for Non-IID Federated Learning
============================================================

This module implements various data partitioning strategies that simulate
real-world non-IID data distributions in federated learning.

Partitioning Strategies:
1. Dirichlet Distribution: Controls class imbalance via alpha parameter
2. Pathological Partitioning: Each node gets only k classes
3. Natural Partitioning: Simulates real-world scenarios (hospitals, phones, etc.)
4. Quantity Skew: Different nodes have different amounts of data
5. Feature Skew: Different nodes have different feature distributions
6. Temporal Drift: Data distribution changes over time

Each partitioner takes a centralized dataset and splits it among clients
according to the specified non-IID pattern.

Author: Luminous Dynamics
Date: January 2026
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .experiment_config import (
    LabelSkewConfig,
    QuantitySkewConfig,
    FeatureSkewConfig,
    TemporalSkewConfig,
)


@dataclass
class PartitionStats:
    """Statistics about a data partition for analysis."""
    n_clients: int
    samples_per_client: List[int]
    classes_per_client: List[int]
    class_distribution: Dict[int, List[float]]  # class -> [fraction per client]
    effective_alpha: float  # Estimated Dirichlet alpha
    imbalance_ratio: float  # max/min samples

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Partition Stats:\n"
            f"  Clients: {self.n_clients}\n"
            f"  Samples/client: {np.mean(self.samples_per_client):.0f} +/- {np.std(self.samples_per_client):.0f}\n"
            f"  Classes/client: {np.mean(self.classes_per_client):.1f} +/- {np.std(self.classes_per_client):.1f}\n"
            f"  Imbalance ratio: {self.imbalance_ratio:.1f}x\n"
            f"  Effective alpha: {self.effective_alpha:.2f}"
        )


class BasePartitioner(ABC):
    """Abstract base class for data partitioners."""

    def __init__(self, n_clients: int, seed: int = 42):
        self.n_clients = n_clients
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data into client datasets.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Label vector [n_samples]

        Returns:
            List of (X_client, y_client) tuples, one per client
        """
        pass

    def compute_stats(
        self,
        partitions: List[Tuple[np.ndarray, np.ndarray]],
        n_classes: int,
    ) -> PartitionStats:
        """Compute statistics about the partition."""
        samples_per_client = [len(y) for _, y in partitions]
        classes_per_client = [len(np.unique(y)) for _, y in partitions]

        # Class distribution per client
        class_distribution = {c: [] for c in range(n_classes)}
        for _, y in partitions:
            class_counts = np.bincount(y.astype(int), minlength=n_classes)
            class_fractions = class_counts / (len(y) + 1e-10)
            for c in range(n_classes):
                class_distribution[c].append(float(class_fractions[c]))

        # Estimate effective alpha from variance
        # Higher variance in class fractions = lower alpha
        variances = [np.var(class_distribution[c]) for c in range(n_classes)]
        avg_variance = np.mean(variances)
        # Rough approximation: alpha ~ 1 / (variance * n_classes)
        effective_alpha = 1.0 / (avg_variance * n_classes + 0.01)
        effective_alpha = min(effective_alpha, 100.0)  # Cap at 100

        imbalance_ratio = max(samples_per_client) / (min(samples_per_client) + 1)

        return PartitionStats(
            n_clients=self.n_clients,
            samples_per_client=samples_per_client,
            classes_per_client=classes_per_client,
            class_distribution=class_distribution,
            effective_alpha=effective_alpha,
            imbalance_ratio=imbalance_ratio,
        )


class DirichletPartitioner(BasePartitioner):
    """Dirichlet distribution partitioner for label skew.

    The Dirichlet distribution controls how imbalanced class distributions
    are across clients:
    - alpha -> infinity: All clients have identical class distributions (IID)
    - alpha = 1.0: Moderate imbalance (mild non-IID)
    - alpha = 0.5: Significant imbalance (moderate non-IID)
    - alpha = 0.1: Severe imbalance (each client dominated by few classes)
    - alpha -> 0: Each client has only one class (extreme non-IID)
    """

    def __init__(
        self,
        n_clients: int,
        alpha: float = 1.0,
        min_samples_per_client: int = 10,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data using Dirichlet distribution."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Group samples by class
        class_indices = {c: np.where(y == c)[0] for c in range(n_classes)}

        # Sample Dirichlet proportions for each class
        # Shape: [n_classes, n_clients]
        proportions = self.rng.dirichlet(
            [self.alpha] * self.n_clients,
            size=n_classes
        )

        # Initialize client data
        client_indices = [[] for _ in range(self.n_clients)]

        # Distribute samples from each class according to proportions
        for class_idx in range(n_classes):
            class_samples = class_indices[class_idx]
            n_class_samples = len(class_samples)

            # Shuffle samples within class
            self.rng.shuffle(class_samples)

            # Compute number of samples per client for this class
            client_counts = (proportions[class_idx] * n_class_samples).astype(int)

            # Distribute any remainder
            remainder = n_class_samples - client_counts.sum()
            for i in range(int(remainder)):
                client_counts[i % self.n_clients] += 1

            # Assign samples to clients
            start_idx = 0
            for client_idx in range(self.n_clients):
                end_idx = start_idx + client_counts[client_idx]
                client_indices[client_idx].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx

        # Ensure minimum samples per client
        for client_idx in range(self.n_clients):
            if len(client_indices[client_idx]) < self.min_samples_per_client:
                # Steal samples from largest client
                largest_client = max(range(self.n_clients), key=lambda i: len(client_indices[i]))
                needed = self.min_samples_per_client - len(client_indices[client_idx])
                if len(client_indices[largest_client]) > needed + self.min_samples_per_client:
                    stolen = client_indices[largest_client][-needed:]
                    client_indices[largest_client] = client_indices[largest_client][:-needed]
                    client_indices[client_idx].extend(stolen)

        # Build partitions
        partitions = []
        for client_idx in range(self.n_clients):
            indices = np.array(client_indices[client_idx])
            if len(indices) == 0:
                # Empty client - give it one sample from each class if possible
                indices = [class_indices[c][0] for c in range(n_classes) if len(class_indices[c]) > 0]
                indices = np.array(indices[:1])  # At least one sample

            X_client = X[indices]
            y_client = y[indices]
            partitions.append((X_client, y_client))

        return partitions


class PathologicalPartitioner(BasePartitioner):
    """Pathological partitioner where each client gets only k classes.

    This is an extreme form of label skew used in many FL papers.
    Each client sees only a subset of classes, making local learning
    biased and global aggregation challenging.

    Args:
        n_clients: Number of clients
        classes_per_client: Number of classes each client can have (e.g., 2)
        shuffle_classes: Whether to randomly assign which classes to which client
    """

    def __init__(
        self,
        n_clients: int,
        classes_per_client: int = 2,
        shuffle_classes: bool = True,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.classes_per_client = classes_per_client
        self.shuffle_classes = shuffle_classes

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data so each client gets only k classes."""
        n_classes = len(np.unique(y))

        # Group samples by class
        class_indices = {c: np.where(y == c)[0].tolist() for c in range(n_classes)}

        # Assign classes to clients
        # Each class should appear in roughly the same number of clients
        classes_per_client_list = []
        class_list = list(range(n_classes))

        if self.shuffle_classes:
            self.rng.shuffle(class_list)

        # Round-robin assignment ensuring each client gets k classes
        for client_idx in range(self.n_clients):
            assigned_classes = []
            for k in range(self.classes_per_client):
                class_idx = (client_idx * self.classes_per_client + k) % n_classes
                assigned_classes.append(class_list[class_idx])
            classes_per_client_list.append(assigned_classes)

        # Count how many clients have each class
        class_client_count = {c: 0 for c in range(n_classes)}
        for classes in classes_per_client_list:
            for c in classes:
                class_client_count[c] += 1

        # Distribute samples from each class among clients that have it
        client_indices = [[] for _ in range(self.n_clients)]

        for class_idx in range(n_classes):
            class_samples = class_indices[class_idx].copy()
            self.rng.shuffle(class_samples)

            # Find clients that have this class
            clients_with_class = [
                i for i, classes in enumerate(classes_per_client_list)
                if class_idx in classes
            ]

            if not clients_with_class:
                continue

            # Split samples among these clients
            samples_per_client = len(class_samples) // len(clients_with_class)
            remainder = len(class_samples) % len(clients_with_class)

            start_idx = 0
            for i, client_idx in enumerate(clients_with_class):
                n_samples = samples_per_client + (1 if i < remainder else 0)
                end_idx = start_idx + n_samples
                client_indices[client_idx].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx

        # Build partitions
        partitions = []
        for client_idx in range(self.n_clients):
            indices = np.array(client_indices[client_idx])
            X_client = X[indices] if len(indices) > 0 else X[:1]
            y_client = y[indices] if len(indices) > 0 else y[:1]
            partitions.append((X_client, y_client))

        return partitions


class NaturalPartitioner(BasePartitioner):
    """Natural partitioner simulating real-world federated scenarios.

    This partitioner models realistic data distributions found in:
    - Healthcare (hospitals have different patient populations)
    - Mobile devices (users have different usage patterns)
    - IoT sensors (devices in different environments)

    Each "region" or "demographic" has a characteristic class distribution,
    and clients within a region have similar distributions.
    """

    def __init__(
        self,
        n_clients: int,
        n_regions: int = 5,
        region_specialization: float = 0.7,  # 0-1, how specialized regions are
        client_variance: float = 0.2,  # Variance within region
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.n_regions = n_regions
        self.region_specialization = region_specialization
        self.client_variance = client_variance

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data into natural regions with realistic distributions."""
        n_classes = len(np.unique(y))

        # Assign clients to regions
        clients_per_region = self.n_clients // self.n_regions
        client_regions = []
        for region in range(self.n_regions):
            n_in_region = clients_per_region
            if region < self.n_clients % self.n_regions:
                n_in_region += 1
            client_regions.extend([region] * n_in_region)
        self.rng.shuffle(client_regions)

        # Create region-specific class preferences
        # Each region specializes in certain classes
        region_preferences = np.zeros((self.n_regions, n_classes))
        for region in range(self.n_regions):
            # Base uniform distribution
            base = np.ones(n_classes) / n_classes

            # Add specialization in a few classes
            n_specialized = max(1, n_classes // self.n_regions)
            specialized_classes = self.rng.choice(
                n_classes, n_specialized, replace=False
            )
            specialization = np.zeros(n_classes)
            specialization[specialized_classes] = self.region_specialization / n_specialized

            # Combine base and specialization
            prefs = (1 - self.region_specialization) * base + specialization
            prefs /= prefs.sum()
            region_preferences[region] = prefs

        # Generate client-specific preferences (with variance around region)
        client_preferences = np.zeros((self.n_clients, n_classes))
        for client_idx in range(self.n_clients):
            region = client_regions[client_idx]
            base = region_preferences[region]

            # Add client-specific variance
            noise = self.rng.normal(0, self.client_variance, n_classes)
            prefs = base + noise
            prefs = np.maximum(prefs, 0.01)  # Ensure non-negative
            prefs /= prefs.sum()
            client_preferences[client_idx] = prefs

        # Group samples by class
        class_indices = {c: np.where(y == c)[0].tolist() for c in range(n_classes)}
        for c in range(n_classes):
            self.rng.shuffle(class_indices[c])

        # Distribute samples according to client preferences
        samples_per_client = len(y) // self.n_clients
        client_indices = [[] for _ in range(self.n_clients)]
        class_pointers = {c: 0 for c in range(n_classes)}

        for client_idx in range(self.n_clients):
            prefs = client_preferences[client_idx]

            # Sample from each class according to preference
            samples_to_add = []
            target_samples = samples_per_client

            for class_idx in range(n_classes):
                n_from_class = int(prefs[class_idx] * target_samples)
                available = len(class_indices[class_idx]) - class_pointers[class_idx]
                n_from_class = min(n_from_class, available)

                start = class_pointers[class_idx]
                end = start + n_from_class
                samples_to_add.extend(class_indices[class_idx][start:end])
                class_pointers[class_idx] = end

            client_indices[client_idx] = samples_to_add

        # Build partitions
        partitions = []
        for client_idx in range(self.n_clients):
            indices = np.array(client_indices[client_idx])
            if len(indices) == 0:
                # Fallback: give some samples
                indices = np.arange(min(10, len(y)))
            X_client = X[indices]
            y_client = y[indices]
            partitions.append((X_client, y_client))

        return partitions


class QuantitySkewPartitioner(BasePartitioner):
    """Partitioner that creates quantity imbalance across clients.

    Some clients have much more data than others, simulating real-world
    scenarios where data collection varies significantly.
    """

    def __init__(
        self,
        n_clients: int,
        config: QuantitySkewConfig,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.config = config

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition with quantity skew."""
        n_samples = len(y)

        # Get sample counts per client
        sample_counts = self.config.get_sample_counts(
            self.n_clients, n_samples, self.seed
        )

        # Shuffle all data
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        # Distribute to clients
        partitions = []
        start_idx = 0
        for client_idx in range(self.n_clients):
            n = sample_counts[client_idx]
            end_idx = min(start_idx + n, n_samples)
            client_indices = indices[start_idx:end_idx]

            X_client = X[client_indices]
            y_client = y[client_indices]
            partitions.append((X_client, y_client))

            start_idx = end_idx

        return partitions


class FeatureSkewPartitioner(BasePartitioner):
    """Partitioner that adds feature-level heterogeneity.

    Different clients see features with:
    - Different noise levels
    - Different scales
    - Different feature subsets
    """

    def __init__(
        self,
        n_clients: int,
        config: FeatureSkewConfig,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.config = config

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition with feature skew applied."""
        n_samples = len(y)
        n_features = X.shape[1]
        samples_per_client = n_samples // self.n_clients

        # Shuffle and split uniformly first
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        partitions = []
        for client_idx in range(self.n_clients):
            start_idx = client_idx * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_idx == self.n_clients - 1:
                end_idx = n_samples

            client_indices = indices[start_idx:end_idx]
            X_client = X[client_indices].copy()
            y_client = y[client_indices]

            # Apply feature transformations
            # 1. Add noise
            noise_level = self.config.get_node_noise_level(client_idx, self.n_clients)
            if noise_level > 0:
                noise = self.rng.normal(0, noise_level, X_client.shape)
                X_client = X_client + noise

            # 2. Apply scaling
            scale = self.config.get_node_scale(client_idx, self.n_clients, self.seed)
            X_client = X_client * scale

            # 3. Apply covariate shift
            if self.config.covariate_shift_intensity > 0:
                shift = self.rng.normal(
                    0,
                    self.config.covariate_shift_intensity,
                    n_features
                )
                X_client = X_client + shift

            partitions.append((X_client, y_client))

        return partitions


class TemporalDriftPartitioner(BasePartitioner):
    """Partitioner that simulates temporal drift in data.

    Data distribution changes over rounds:
    - Gradual drift: Slow continuous change
    - Sudden drift: Abrupt change at certain point
    - Periodic drift: Oscillating changes
    """

    def __init__(
        self,
        n_clients: int,
        config: TemporalSkewConfig,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.config = config
        self._base_partitions: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._drift_vectors: Optional[np.ndarray] = None

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create base partition (call once at start)."""
        n_samples = len(y)
        n_features = X.shape[1]
        samples_per_client = n_samples // self.n_clients

        # Shuffle and split uniformly
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        partitions = []
        for client_idx in range(self.n_clients):
            start_idx = client_idx * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_idx == self.n_clients - 1:
                end_idx = n_samples

            client_indices = indices[start_idx:end_idx]
            X_client = X[client_indices].copy()
            y_client = y[client_indices].copy()
            partitions.append((X_client, y_client))

        # Store base partitions for temporal drift
        self._base_partitions = partitions

        # Generate drift vectors for each client
        self._drift_vectors = self.rng.normal(0, 1, (self.n_clients, n_features))
        self._drift_vectors /= np.linalg.norm(self._drift_vectors, axis=1, keepdims=True)

        return partitions

    def get_drifted_data(
        self,
        round_idx: int,
        total_rounds: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get data with temporal drift applied for a specific round."""
        if self._base_partitions is None:
            raise ValueError("Must call partition() first")

        drift_factor = self.config.get_drift_factor(round_idx, total_rounds)

        drifted_partitions = []
        for client_idx, (X_base, y_base) in enumerate(self._base_partitions):
            X_drifted = X_base.copy()

            # Apply feature drift
            drift = self._drift_vectors[client_idx] * drift_factor
            X_drifted = X_drifted + drift

            # Concept drift: swap some labels (if enabled)
            y_drifted = y_base.copy()
            if self.config.concept_drift:
                n_classes = len(np.unique(y_base))
                n_swap = int(len(y_base) * drift_factor * 0.1)  # Swap up to 10%
                swap_indices = self.rng.choice(len(y_base), n_swap, replace=False)
                y_drifted[swap_indices] = (y_drifted[swap_indices] + 1) % n_classes

            drifted_partitions.append((X_drifted, y_drifted))

        return drifted_partitions


class CombinedPartitioner(BasePartitioner):
    """Combines multiple types of non-IID partitioning.

    Applies partitioners in sequence to create realistic combined scenarios.
    """

    def __init__(
        self,
        n_clients: int,
        label_skew: Optional[LabelSkewConfig] = None,
        quantity_skew: Optional[QuantitySkewConfig] = None,
        feature_skew: Optional[FeatureSkewConfig] = None,
        temporal_skew: Optional[TemporalSkewConfig] = None,
        seed: int = 42,
    ):
        super().__init__(n_clients, seed)
        self.label_skew = label_skew
        self.quantity_skew = quantity_skew
        self.feature_skew = feature_skew
        self.temporal_skew = temporal_skew

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply combined partitioning strategy."""
        n_samples = len(y)
        n_features = X.shape[1]

        # Step 1: Apply label skew (Dirichlet partitioning)
        if self.label_skew:
            partitioner = DirichletPartitioner(
                self.n_clients,
                alpha=self.label_skew.alpha,
                seed=self.seed,
            )
            partitions = partitioner.partition(X, y)
        else:
            # Uniform split
            samples_per_client = n_samples // self.n_clients
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            partitions = []
            for i in range(self.n_clients):
                start = i * samples_per_client
                end = start + samples_per_client if i < self.n_clients - 1 else n_samples
                client_indices = indices[start:end]
                partitions.append((X[client_indices], y[client_indices]))

        # Step 2: Apply quantity skew (redistribute samples)
        if self.quantity_skew:
            target_counts = self.quantity_skew.get_sample_counts(
                self.n_clients,
                sum(len(p[1]) for p in partitions),
                self.seed + 1,
            )

            # Collect all samples
            all_X = np.vstack([p[0] for p in partitions])
            all_y = np.concatenate([p[1] for p in partitions])

            # Redistribute
            indices = np.arange(len(all_y))
            self.rng.shuffle(indices)

            new_partitions = []
            start = 0
            for i, count in enumerate(target_counts):
                end = min(start + count, len(all_y))
                client_indices = indices[start:end]
                new_partitions.append((all_X[client_indices], all_y[client_indices]))
                start = end

            partitions = new_partitions

        # Step 3: Apply feature skew (transform features)
        if self.feature_skew:
            transformed_partitions = []
            for client_idx, (X_client, y_client) in enumerate(partitions):
                X_transformed = X_client.copy()

                # Add noise
                noise_level = self.feature_skew.get_node_noise_level(
                    client_idx, self.n_clients
                )
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level, X_transformed.shape)
                    X_transformed = X_transformed + noise

                # Apply scaling
                scale = self.feature_skew.get_node_scale(
                    client_idx, self.n_clients, self.seed + 2
                )
                X_transformed = X_transformed * scale

                # Covariate shift
                if self.feature_skew.covariate_shift_intensity > 0:
                    shift = self.rng.normal(
                        0,
                        self.feature_skew.covariate_shift_intensity,
                        n_features
                    )
                    X_transformed = X_transformed + shift

                transformed_partitions.append((X_transformed, y_client))

            partitions = transformed_partitions

        return partitions


def create_partitioner(
    n_clients: int,
    label_skew: Optional[LabelSkewConfig] = None,
    quantity_skew: Optional[QuantitySkewConfig] = None,
    feature_skew: Optional[FeatureSkewConfig] = None,
    temporal_skew: Optional[TemporalSkewConfig] = None,
    seed: int = 42,
) -> BasePartitioner:
    """Factory function to create appropriate partitioner."""
    # Count how many skew types are specified
    skew_count = sum([
        label_skew is not None,
        quantity_skew is not None,
        feature_skew is not None,
        temporal_skew is not None,
    ])

    if skew_count == 0:
        # IID case - use Dirichlet with high alpha
        return DirichletPartitioner(n_clients, alpha=100.0, seed=seed)

    elif skew_count == 1:
        # Single skew type
        if label_skew:
            return DirichletPartitioner(n_clients, alpha=label_skew.alpha, seed=seed)
        elif quantity_skew:
            return QuantitySkewPartitioner(n_clients, quantity_skew, seed=seed)
        elif feature_skew:
            return FeatureSkewPartitioner(n_clients, feature_skew, seed=seed)
        elif temporal_skew:
            return TemporalDriftPartitioner(n_clients, temporal_skew, seed=seed)

    # Multiple skew types - use combined partitioner
    return CombinedPartitioner(
        n_clients,
        label_skew=label_skew,
        quantity_skew=quantity_skew,
        feature_skew=feature_skew,
        temporal_skew=temporal_skew,
        seed=seed,
    )
