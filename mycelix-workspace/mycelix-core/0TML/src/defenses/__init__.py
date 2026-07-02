# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Defense Registry for Byzantine-Robust Federated Learning
=========================================================

Comprehensive registry of SOTA defense mechanisms for Gen-4 evaluation.

Tier 1 Defenses (Must-Have):
- FLTrust: Direction-based with server validation set
- RFA: Robust Federated Averaging (geometric median)
- Coordinate-wise Median: Simple coordinate-wise robust aggregation
- Trimmed Mean: Trim top/bottom β percentile per coordinate
- Krum / Multi-Krum / Bulyan: Distance-based selection
- CBF: Conformal Behavioral Filter (representation-based baseline)
- FedAvg: Vanilla baseline (should fail)

Tier 2 Defenses (Nice-to-Have):
- FoolsGold: Anti-Sybil via historical cosine similarity
- BOBA: Label-skew aware robust aggregation

Author: Luminous Dynamics
Date: November 8, 2025
Version: 1.0.0
"""

from typing import Dict, Type, Any
from .pogq_v4_enhanced import PoGQv41Enhanced, PoGQv41Config
from .adaptive_pogq import AdaptiveProofOfGoodQuality, AdaptivePoGQConfig
from .cbf import CBF, CBFConfig
from .rfa import RobustFederatedAveraging
from .coordinate_median import CoordinateMedian, CoordinateMedianSafe
from .trimmed_mean import TrimmedMean
from .krum import Krum, MultiKrum, Bulyan
from .fltrust import FLTrust
from .foolsgold import FoolsGold
from .boba import BOBA
from .fedavg import FedAvg

# Defense registry
DEFENSE_REGISTRY: Dict[str, Type] = {
    # Tier 1: Must-have baselines
    "pogq_v4.1": PoGQv41Enhanced,
    "adaptive_pogq": AdaptiveProofOfGoodQuality,  # NEW: Adaptive PoGQ for non-IID
    "fltrust": FLTrust,
    "rfa": RobustFederatedAveraging,
    "coord_median": CoordinateMedian,
    "coord_median_safe": CoordinateMedianSafe,  # Phase 5: Safe variant with guards
    "trimmed_mean": TrimmedMean,
    "krum": Krum,
    "multi_krum": MultiKrum,
    "bulyan": Bulyan,
    "cbf": CBF,

    # Tier 2: Nice-to-have
    "foolsgold": FoolsGold,
    "boba": BOBA,

    # Baseline (should fail)
    "fedavg": FedAvg,

    # Deprecated aliases (for backward compatibility)
    "fedguard_strict": CBF,  # Deprecated: use 'cbf' instead
}


def get_defense(name: str, **config) -> Any:
    """
    Factory function to instantiate a defense

    Args:
        name: Defense name from registry
        **config: Configuration parameters

    Returns:
        Instantiated defense object

    Example:
        >>> pogq = get_defense("pogq_v4.1", conformal_alpha=0.10)
        >>> adaptive_pogq = get_defense("adaptive_pogq", tau_min=0.1, confidence_level=2.0)
        >>> fltrust = get_defense("fltrust", server_lr=0.001)
    """
    if name not in DEFENSE_REGISTRY:
        available = ", ".join(DEFENSE_REGISTRY.keys())
        raise ValueError(f"Defense '{name}' not found. Available: {available}")

    defense_class = DEFENSE_REGISTRY[name]

    # Defenses with config dataclasses need config object construction
    if name == "pogq_v4.1":
        config_obj = PoGQv41Config(**config)
        return defense_class(config_obj)
    elif name == "adaptive_pogq":
        config_obj = AdaptivePoGQConfig(**config)
        return defense_class(config_obj)
    elif name in ("cbf", "fedguard_strict"):
        config_obj = CBFConfig(**config)
        return defense_class(config_obj)
    else:
        # Other defenses accept **kwargs directly
        return defense_class(**config)


def list_defenses() -> Dict[str, str]:
    """List all available defenses with descriptions"""
    descriptions = {
        "pogq_v4.1": "PoGQ-v4.1 Enhanced: Mondrian + Conformal + Adaptive Hybrid",
        "adaptive_pogq": "Adaptive PoGQ: Per-node adaptive thresholds for non-IID (85%→95% detection)",
        "fltrust": "FLTrust: Direction-based with server validation set",
        "rfa": "RFA: Robust Federated Averaging (geometric median via Weiszfeld)",
        "coord_median": "Coordinate-wise Median: Simple per-coordinate median",
        "coord_median_safe": "Coordinate-wise Median (Safe): With min-clients, norm-clamp, direction guards",
        "trimmed_mean": "Trimmed Mean: Trim top/bottom β per coordinate",
        "krum": "Krum: Select closest gradient to others",
        "multi_krum": "Multi-Krum: Select k closest gradients",
        "bulyan": "Bulyan: Krum + Trimmed Mean composition",
        "cbf": "CBF: Conformal Behavioral Filter (PCA + conformal thresholding)",
        "foolsgold": "FoolsGold: Anti-Sybil via historical cosine similarity",
        "boba": "BOBA: Label-skew aware robust aggregation",
        "fedavg": "FedAvg: Vanilla averaging (baseline, should fail)",
        "fedguard_strict": "[DEPRECATED] Use 'cbf' instead",
    }
    return descriptions


__all__ = [
    "DEFENSE_REGISTRY",
    "get_defense",
    "list_defenses",
    "PoGQv41Enhanced",
    "PoGQv41Config",
    "AdaptiveProofOfGoodQuality",
    "AdaptivePoGQConfig",
    "FLTrust",
    "RobustFederatedAveraging",
    "CoordinateMedian",
    "CoordinateMedianSafe",
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    "Bulyan",
    "CBF",
    "CBFConfig",
    "FoolsGold",
    "BOBA",
    "FedAvg",
]
