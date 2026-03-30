# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Non-IID Data Distribution Validation Experiments for Byzantine Detection
=========================================================================

This package provides comprehensive validation of Byzantine detection mechanisms
under realistic non-IID (non-independent, identically distributed) data conditions.

Real-world federated learning faces heterogeneous data distributions where:
- Different nodes have different class distributions (label skew)
- Nodes have vastly different dataset sizes (quantity skew)
- Nodes have different feature distributions (feature skew)
- Data distributions change over time (temporal skew)

These experiments prove that Byzantine detection works in realistic conditions,
not just the idealized IID setting.

Modules:
    experiment_config: Configuration classes for non-IID scenarios
    data_partitioners: Data partitioning strategies (Dirichlet, pathological, etc.)
    scenarios: Predefined test scenarios (mild, moderate, severe non-IID)
    non_iid_experiment: Main experiment runner with comprehensive metrics

Usage:
    from experiments.non_iid_validation import run_all_scenarios
    results = run_all_scenarios()

Author: Luminous Dynamics
Date: January 2026
"""

from .experiment_config import (
    NonIIDConfig,
    LabelSkewConfig,
    QuantitySkewConfig,
    FeatureSkewConfig,
    TemporalSkewConfig,
)

from .data_partitioners import (
    DirichletPartitioner,
    PathologicalPartitioner,
    NaturalPartitioner,
    QuantitySkewPartitioner,
    FeatureSkewPartitioner,
    TemporalDriftPartitioner,
)

from .scenarios import (
    MILD_NON_IID,
    MODERATE_NON_IID,
    SEVERE_NON_IID,
    PATHOLOGICAL_NON_IID,
    QUANTITY_SKEW_SCENARIO,
    FEATURE_SKEW_SCENARIO,
    TEMPORAL_DRIFT_SCENARIO,
    get_all_scenarios,
)

from .non_iid_experiment import (
    NonIIDExperiment,
    ExperimentResults,
    run_single_experiment,
    run_all_scenarios,
    generate_validation_report,
)

__all__ = [
    # Config
    "NonIIDConfig",
    "LabelSkewConfig",
    "QuantitySkewConfig",
    "FeatureSkewConfig",
    "TemporalSkewConfig",
    # Partitioners
    "DirichletPartitioner",
    "PathologicalPartitioner",
    "NaturalPartitioner",
    "QuantitySkewPartitioner",
    "FeatureSkewPartitioner",
    "TemporalDriftPartitioner",
    # Scenarios
    "MILD_NON_IID",
    "MODERATE_NON_IID",
    "SEVERE_NON_IID",
    "PATHOLOGICAL_NON_IID",
    "QUANTITY_SKEW_SCENARIO",
    "FEATURE_SKEW_SCENARIO",
    "TEMPORAL_DRIFT_SCENARIO",
    "get_all_scenarios",
    # Experiments
    "NonIIDExperiment",
    "ExperimentResults",
    "run_single_experiment",
    "run_all_scenarios",
    "generate_validation_report",
]
