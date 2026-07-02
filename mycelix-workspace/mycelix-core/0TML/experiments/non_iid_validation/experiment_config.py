# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Configuration Classes for Non-IID Data Distribution Experiments
================================================================

This module defines configuration classes for various types of non-IID
data distributions encountered in real-world federated learning:

1. Label Skew: Different nodes have different class distributions
   - Controlled by Dirichlet alpha parameter
   - alpha=1.0 (IID), alpha=0.5 (moderate), alpha=0.1 (severe)

2. Quantity Skew: Nodes have vastly different dataset sizes
   - Some nodes have 10x more data than others
   - Simulates real-world resource inequality

3. Feature Skew: Nodes have different feature distributions
   - Different noise levels, scaling, or feature subsets
   - Simulates different sensor calibrations, data collection methods

4. Temporal Skew: Data distributions change over time
   - Concept drift, seasonal patterns
   - Simulates evolving real-world data

Author: Luminous Dynamics
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np


class SkewType(Enum):
    """Types of data heterogeneity in federated learning."""
    LABEL_SKEW = "label_skew"
    QUANTITY_SKEW = "quantity_skew"
    FEATURE_SKEW = "feature_skew"
    TEMPORAL_SKEW = "temporal_skew"
    COMBINED = "combined"


class AttackType(Enum):
    """Byzantine attack types for validation."""
    NONE = "none"
    SIGN_FLIP = "sign_flip"
    SCALED_GRADIENT = "scaled_grad"
    MODEL_REPLACEMENT = "model_replacement"
    GAUSSIAN_NOISE = "gaussian_noise"
    LABEL_FLIP = "label_flip"
    BACKDOOR = "backdoor"


class AggregatorType(Enum):
    """Aggregation algorithms to test."""
    FEDAVG = "fedavg"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    AEGIS = "aegis"
    AEGIS_GEN7 = "aegis_gen7"


@dataclass
class LabelSkewConfig:
    """Configuration for label skew (class imbalance across nodes).

    Label skew is the most common form of non-IID in federated learning.
    It occurs when different clients have different class distributions.

    Attributes:
        alpha: Dirichlet concentration parameter
               - alpha -> infinity: IID (uniform class distribution)
               - alpha = 1.0: Mild non-IID
               - alpha = 0.5: Moderate non-IID
               - alpha = 0.1: Severe non-IID (some nodes have only 1-2 classes)
        min_samples_per_class: Minimum samples per class per node (ensures coverage)
        enforce_coverage: If True, ensure every node has at least one sample per class
    """
    alpha: float = 1.0
    min_samples_per_class: int = 0
    enforce_coverage: bool = False

    def get_severity(self) -> str:
        """Return human-readable severity level."""
        if self.alpha >= 10.0:
            return "IID"
        elif self.alpha >= 1.0:
            return "Mild"
        elif self.alpha >= 0.5:
            return "Moderate"
        elif self.alpha >= 0.1:
            return "Severe"
        else:
            return "Extreme"


@dataclass
class QuantitySkewConfig:
    """Configuration for quantity skew (different dataset sizes per node).

    In real federated learning, some clients have much more data than others.
    This affects model quality and can make Byzantine detection harder.

    Attributes:
        distribution: How to distribute samples ("uniform", "exponential", "powerlaw")
        min_samples: Minimum samples per node
        max_samples: Maximum samples per node
        imbalance_ratio: Ratio of max to min samples (for exponential/powerlaw)
        large_node_fraction: Fraction of nodes that are "large" (have more data)
    """
    distribution: str = "uniform"
    min_samples: int = 50
    max_samples: int = 500
    imbalance_ratio: float = 10.0
    large_node_fraction: float = 0.2

    def get_sample_counts(self, n_clients: int, total_samples: int, seed: int = 42) -> List[int]:
        """Generate sample counts for each client based on distribution."""
        rng = np.random.default_rng(seed)

        if self.distribution == "uniform":
            base = total_samples // n_clients
            counts = [base] * n_clients
            # Distribute remainder
            remainder = total_samples - sum(counts)
            for i in range(remainder):
                counts[i % n_clients] += 1
            return counts

        elif self.distribution == "exponential":
            # Exponential distribution creates heavy tail
            rates = rng.exponential(1.0, n_clients)
            rates = rates / rates.sum()
            counts = (rates * total_samples).astype(int)
            counts = np.clip(counts, self.min_samples, self.max_samples)
            # Adjust to match total
            diff = total_samples - counts.sum()
            if diff > 0:
                idx = np.argmax(counts)
                counts[idx] += diff
            elif diff < 0:
                idx = np.argmax(counts)
                counts[idx] = max(counts[idx] + diff, self.min_samples)
            return counts.tolist()

        elif self.distribution == "powerlaw":
            # Power law: few nodes have lots of data, most have little
            n_large = int(self.large_node_fraction * n_clients)
            n_small = n_clients - n_large

            large_samples = int(total_samples * 0.8)  # 80% to large nodes
            small_samples = total_samples - large_samples

            large_counts = [large_samples // n_large] * n_large if n_large > 0 else []
            small_counts = [small_samples // n_small] * n_small if n_small > 0 else []

            counts = large_counts + small_counts
            rng.shuffle(counts)
            return counts

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class FeatureSkewConfig:
    """Configuration for feature skew (different feature distributions per node).

    Different nodes may have different data collection methods, sensors,
    or preprocessing pipelines, leading to feature-level heterogeneity.

    Attributes:
        noise_levels: Range of noise levels across nodes (min, max)
        feature_scaling: Whether nodes have different feature scales
        feature_subset_fraction: If <1.0, nodes only see subset of features
        covariate_shift_intensity: Intensity of covariate shift (0.0 = none)
    """
    noise_levels: Tuple[float, float] = (0.0, 0.5)
    feature_scaling: bool = False
    feature_subset_fraction: float = 1.0
    covariate_shift_intensity: float = 0.0

    def get_node_noise_level(self, node_idx: int, n_nodes: int) -> float:
        """Get noise level for a specific node."""
        min_noise, max_noise = self.noise_levels
        # Linear interpolation across nodes
        return min_noise + (max_noise - min_noise) * (node_idx / max(n_nodes - 1, 1))

    def get_node_scale(self, node_idx: int, n_nodes: int, seed: int = 42) -> float:
        """Get feature scaling factor for a specific node."""
        if not self.feature_scaling:
            return 1.0
        rng = np.random.default_rng(seed + node_idx)
        return rng.uniform(0.5, 2.0)


@dataclass
class TemporalSkewConfig:
    """Configuration for temporal skew (data distribution changes over time).

    In real systems, data distributions evolve:
    - Concept drift: The mapping from X to Y changes
    - Covariate shift: The distribution of X changes
    - Seasonal patterns: Periodic changes in distribution

    Attributes:
        drift_type: Type of temporal change ("gradual", "sudden", "periodic")
        drift_intensity: How much the distribution changes (0.0-1.0)
        drift_frequency: For periodic drift, number of periods per training run
        concept_drift: If True, the label mapping changes (harder problem)
    """
    drift_type: str = "gradual"
    drift_intensity: float = 0.2
    drift_frequency: int = 2
    concept_drift: bool = False

    def get_drift_factor(self, round_idx: int, total_rounds: int) -> float:
        """Get drift factor for a specific round."""
        progress = round_idx / max(total_rounds - 1, 1)

        if self.drift_type == "gradual":
            return self.drift_intensity * progress

        elif self.drift_type == "sudden":
            # Sudden shift at midpoint
            return self.drift_intensity if progress > 0.5 else 0.0

        elif self.drift_type == "periodic":
            # Sinusoidal drift
            return self.drift_intensity * np.abs(
                np.sin(2 * np.pi * self.drift_frequency * progress)
            )

        else:
            return 0.0


@dataclass
class NonIIDConfig:
    """Master configuration for non-IID experiments.

    Combines multiple types of data heterogeneity for comprehensive testing.

    Attributes:
        name: Human-readable name for this configuration
        description: Detailed description of what this tests

        # Data configuration
        n_clients: Number of federated learning clients
        n_samples_train: Total training samples
        n_samples_test: Test samples
        n_features: Feature dimensionality
        n_classes: Number of output classes
        dataset: Dataset name ("synthetic", "mnist", "emnist", "cifar10")

        # Non-IID configuration
        label_skew: Label skew configuration (or None for IID)
        quantity_skew: Quantity skew configuration (or None for balanced)
        feature_skew: Feature skew configuration (or None for homogeneous)
        temporal_skew: Temporal skew configuration (or None for stationary)

        # Byzantine configuration
        byzantine_fraction: Fraction of Byzantine nodes
        attack_type: Type of Byzantine attack
        attack_intensity: Attack strength parameter

        # Training configuration
        n_rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        aggregator: Aggregation algorithm

        # Experiment configuration
        seeds: Random seeds for statistical validation
        save_traces: Whether to save detailed per-round traces
    """
    name: str = "default"
    description: str = ""

    # Data configuration
    n_clients: int = 50
    n_samples_train: int = 6000
    n_samples_test: int = 1000
    n_features: int = 784  # MNIST/EMNIST
    n_classes: int = 10
    dataset: str = "emnist"

    # Non-IID configuration
    label_skew: Optional[LabelSkewConfig] = None
    quantity_skew: Optional[QuantitySkewConfig] = None
    feature_skew: Optional[FeatureSkewConfig] = None
    temporal_skew: Optional[TemporalSkewConfig] = None

    # Byzantine configuration
    byzantine_fraction: float = 0.3
    attack_type: AttackType = AttackType.SIGN_FLIP
    attack_intensity: float = 1.0

    # Training configuration
    n_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.05
    aggregator: AggregatorType = AggregatorType.AEGIS

    # Experiment configuration
    seeds: List[int] = field(default_factory=lambda: [42, 101, 202])
    save_traces: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.byzantine_fraction < 0 or self.byzantine_fraction > 0.5:
            raise ValueError("byzantine_fraction must be in [0, 0.5]")
        if self.n_clients < 3:
            raise ValueError("Need at least 3 clients for federated learning")
        if not self.description:
            self.description = self._generate_description()

    def _generate_description(self) -> str:
        """Generate automatic description from config."""
        parts = [f"{self.n_clients} clients, {self.n_rounds} rounds"]

        if self.label_skew:
            parts.append(f"label_skew(alpha={self.label_skew.alpha})")
        if self.quantity_skew:
            parts.append(f"quantity_skew({self.quantity_skew.distribution})")
        if self.feature_skew:
            parts.append(f"feature_skew(noise={self.feature_skew.noise_levels})")
        if self.temporal_skew:
            parts.append(f"temporal_skew({self.temporal_skew.drift_type})")

        parts.append(f"{self.byzantine_fraction*100:.0f}% {self.attack_type.value}")

        return ", ".join(parts)

    def get_effective_alpha(self) -> float:
        """Get effective Dirichlet alpha for label distribution."""
        if self.label_skew:
            return self.label_skew.alpha
        return 100.0  # Effectively IID

    def is_iid(self) -> bool:
        """Check if this configuration represents IID data."""
        has_label_skew = self.label_skew is not None and self.label_skew.alpha < 10.0
        has_quantity_skew = self.quantity_skew is not None
        has_feature_skew = (
            self.feature_skew is not None and
            (self.feature_skew.covariate_shift_intensity > 0 or
             self.feature_skew.noise_levels[1] > 0.1)
        )
        has_temporal_skew = (
            self.temporal_skew is not None and
            self.temporal_skew.drift_intensity > 0
        )

        return not (has_label_skew or has_quantity_skew or has_feature_skew or has_temporal_skew)

    def get_skew_types(self) -> List[SkewType]:
        """Get list of active skew types."""
        types = []
        if self.label_skew and self.label_skew.alpha < 10.0:
            types.append(SkewType.LABEL_SKEW)
        if self.quantity_skew:
            types.append(SkewType.QUANTITY_SKEW)
        if self.feature_skew and (
            self.feature_skew.covariate_shift_intensity > 0 or
            self.feature_skew.noise_levels[1] > 0.1
        ):
            types.append(SkewType.FEATURE_SKEW)
        if self.temporal_skew and self.temporal_skew.drift_intensity > 0:
            types.append(SkewType.TEMPORAL_SKEW)
        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "n_clients": self.n_clients,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "dataset": self.dataset,
            "label_skew": {
                "alpha": self.label_skew.alpha,
                "min_samples_per_class": self.label_skew.min_samples_per_class,
                "enforce_coverage": self.label_skew.enforce_coverage,
            } if self.label_skew else None,
            "quantity_skew": {
                "distribution": self.quantity_skew.distribution,
                "min_samples": self.quantity_skew.min_samples,
                "max_samples": self.quantity_skew.max_samples,
                "imbalance_ratio": self.quantity_skew.imbalance_ratio,
            } if self.quantity_skew else None,
            "feature_skew": {
                "noise_levels": self.feature_skew.noise_levels,
                "feature_scaling": self.feature_skew.feature_scaling,
                "covariate_shift_intensity": self.feature_skew.covariate_shift_intensity,
            } if self.feature_skew else None,
            "temporal_skew": {
                "drift_type": self.temporal_skew.drift_type,
                "drift_intensity": self.temporal_skew.drift_intensity,
                "drift_frequency": self.temporal_skew.drift_frequency,
            } if self.temporal_skew else None,
            "byzantine_fraction": self.byzantine_fraction,
            "attack_type": self.attack_type.value,
            "attack_intensity": self.attack_intensity,
            "n_rounds": self.n_rounds,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "aggregator": self.aggregator.value,
            "seeds": self.seeds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NonIIDConfig":
        """Create config from dictionary."""
        label_skew = None
        if data.get("label_skew"):
            label_skew = LabelSkewConfig(**data["label_skew"])

        quantity_skew = None
        if data.get("quantity_skew"):
            quantity_skew = QuantitySkewConfig(**data["quantity_skew"])

        feature_skew = None
        if data.get("feature_skew"):
            feature_skew = FeatureSkewConfig(**data["feature_skew"])

        temporal_skew = None
        if data.get("temporal_skew"):
            temporal_skew = TemporalSkewConfig(**data["temporal_skew"])

        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            n_clients=data.get("n_clients", 50),
            n_samples_train=data.get("n_samples_train", 6000),
            n_samples_test=data.get("n_samples_test", 1000),
            n_features=data.get("n_features", 784),
            n_classes=data.get("n_classes", 10),
            dataset=data.get("dataset", "emnist"),
            label_skew=label_skew,
            quantity_skew=quantity_skew,
            feature_skew=feature_skew,
            temporal_skew=temporal_skew,
            byzantine_fraction=data.get("byzantine_fraction", 0.3),
            attack_type=AttackType(data.get("attack_type", "sign_flip")),
            attack_intensity=data.get("attack_intensity", 1.0),
            n_rounds=data.get("n_rounds", 10),
            local_epochs=data.get("local_epochs", 5),
            learning_rate=data.get("learning_rate", 0.05),
            aggregator=AggregatorType(data.get("aggregator", "aegis")),
            seeds=data.get("seeds", [42, 101, 202]),
        )


# Convenience factory functions
def create_iid_baseline(
    n_clients: int = 50,
    byzantine_fraction: float = 0.3,
    attack_type: str = "sign_flip",
) -> NonIIDConfig:
    """Create IID baseline configuration for comparison."""
    return NonIIDConfig(
        name="iid_baseline",
        description="IID baseline with uniform class distribution",
        n_clients=n_clients,
        label_skew=LabelSkewConfig(alpha=100.0),  # Effectively IID
        byzantine_fraction=byzantine_fraction,
        attack_type=AttackType(attack_type),
    )


def create_label_skew_config(
    alpha: float,
    n_clients: int = 50,
    byzantine_fraction: float = 0.3,
    attack_type: str = "sign_flip",
) -> NonIIDConfig:
    """Create label skew configuration with specified alpha."""
    severity = LabelSkewConfig(alpha=alpha).get_severity()
    return NonIIDConfig(
        name=f"label_skew_alpha{alpha}",
        description=f"{severity} label skew (alpha={alpha})",
        n_clients=n_clients,
        label_skew=LabelSkewConfig(alpha=alpha),
        byzantine_fraction=byzantine_fraction,
        attack_type=AttackType(attack_type),
    )


def create_quantity_skew_config(
    distribution: str = "powerlaw",
    n_clients: int = 50,
    byzantine_fraction: float = 0.3,
) -> NonIIDConfig:
    """Create quantity skew configuration."""
    return NonIIDConfig(
        name=f"quantity_skew_{distribution}",
        description=f"Quantity skew with {distribution} distribution",
        n_clients=n_clients,
        quantity_skew=QuantitySkewConfig(distribution=distribution),
        byzantine_fraction=byzantine_fraction,
    )


def create_combined_skew_config(
    alpha: float = 0.5,
    distribution: str = "exponential",
    n_clients: int = 50,
    byzantine_fraction: float = 0.3,
) -> NonIIDConfig:
    """Create combined label + quantity skew configuration."""
    return NonIIDConfig(
        name=f"combined_alpha{alpha}_{distribution}",
        description=f"Combined label (alpha={alpha}) and quantity ({distribution}) skew",
        n_clients=n_clients,
        label_skew=LabelSkewConfig(alpha=alpha),
        quantity_skew=QuantitySkewConfig(distribution=distribution),
        byzantine_fraction=byzantine_fraction,
    )
