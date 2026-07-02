# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Configuration Module
============================

Provides unified configuration management for the Mycelix Byzantine-resistant
Federated Learning system.

Quick Start:
    from config import load_and_validate_config, MycelixConfig

    # Load and validate (fails fast on critical errors)
    config = load_and_validate_config()

    # Access parameters
    cos_min = config.byzantine_detection.label_skew_cos_min
    aggregation = config.federated_learning.aggregation

    # With dataset-specific parameters
    config = load_and_validate_config(dataset="cifar10")

CRITICAL: Always source .env.optimal before running:
    source .env.optimal
    python your_script.py

The LABEL_SKEW_COS_MIN parameter MUST be -0.5, NOT -0.3.
Using -0.3 causes 16x performance degradation (57-92% FP vs 3.55-7.1% FP).
"""

from .validator import (
    # Main loading functions
    load_and_validate_config,
    print_config_summary,
    get_dataset_optimal_params,

    # Configuration classes
    MycelixConfig,
    ByzantineDetectionConfig,
    FederatedLearningConfig,
    HolochainConfig,
    SecurityConfig,
    MonitoringConfig,

    # Exceptions
    ConfigValidationError,
    ConfigValidationWarning,

    # Constants
    CRITICAL_PARAMS,
    DATASET_OPTIMAL_PARAMS,
)

__all__ = [
    # Functions
    "load_and_validate_config",
    "print_config_summary",
    "get_dataset_optimal_params",

    # Classes
    "MycelixConfig",
    "ByzantineDetectionConfig",
    "FederatedLearningConfig",
    "HolochainConfig",
    "SecurityConfig",
    "MonitoringConfig",

    # Exceptions
    "ConfigValidationError",
    "ConfigValidationWarning",

    # Constants
    "CRITICAL_PARAMS",
    "DATASET_OPTIMAL_PARAMS",
]

__version__ = "1.0.0"
