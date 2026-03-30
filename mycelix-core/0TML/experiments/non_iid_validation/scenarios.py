# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Predefined Non-IID Test Scenarios for Byzantine Detection Validation
=====================================================================

This module defines canonical test scenarios that span the spectrum of
non-IID data distributions, from IID baseline to extreme pathological cases.

Scenarios are organized by:
1. Label Skew Severity (IID -> Mild -> Moderate -> Severe -> Pathological)
2. Quantity Skew (Balanced -> Imbalanced -> Power Law)
3. Feature Skew (Homogeneous -> Heterogeneous)
4. Temporal Skew (Stationary -> Drifting)
5. Combined Scenarios (Real-world realistic combinations)

Each scenario is designed to test specific aspects of Byzantine detection:
- Can detection distinguish attack from legitimate heterogeneity?
- Does detection maintain high TPR while controlling FPR?
- Does the aggregated model maintain accuracy under non-IID + Byzantine?

Author: Luminous Dynamics
Date: January 2026
"""

from typing import List, Dict, Any
from .experiment_config import (
    NonIIDConfig,
    LabelSkewConfig,
    QuantitySkewConfig,
    FeatureSkewConfig,
    TemporalSkewConfig,
    AttackType,
    AggregatorType,
)


# =============================================================================
# LABEL SKEW SCENARIOS
# =============================================================================

# Baseline: IID data (control case)
IID_BASELINE = NonIIDConfig(
    name="iid_baseline",
    description="IID baseline with uniform class distribution across all clients",
    n_clients=50,
    n_samples_train=6000,
    n_samples_test=1000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=100.0),  # Effectively IID
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Mild non-IID (alpha=1.0)
MILD_NON_IID = NonIIDConfig(
    name="mild_non_iid",
    description="Mild label skew (alpha=1.0) - moderate class imbalance",
    n_clients=50,
    n_samples_train=6000,
    n_samples_test=1000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=1.0),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Moderate non-IID (alpha=0.5)
MODERATE_NON_IID = NonIIDConfig(
    name="moderate_non_iid",
    description="Moderate label skew (alpha=0.5) - significant class imbalance",
    n_clients=50,
    n_samples_train=6000,
    n_samples_test=1000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.5),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Severe non-IID (alpha=0.1)
SEVERE_NON_IID = NonIIDConfig(
    name="severe_non_iid",
    description="Severe label skew (alpha=0.1) - extreme class imbalance, clients dominated by 1-2 classes",
    n_clients=50,
    n_samples_train=6000,
    n_samples_test=1000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.1),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Pathological (2 classes per client)
PATHOLOGICAL_NON_IID = NonIIDConfig(
    name="pathological_non_iid",
    description="Pathological partitioning - each client has only 2 classes",
    n_clients=50,
    n_samples_train=6000,
    n_samples_test=1000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.05, enforce_coverage=False),  # Simulates k=2 classes
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# QUANTITY SKEW SCENARIOS
# =============================================================================

# Uniform quantity (baseline)
QUANTITY_UNIFORM = NonIIDConfig(
    name="quantity_uniform",
    description="Balanced data quantity across all clients",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    quantity_skew=QuantitySkewConfig(distribution="uniform"),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Exponential quantity skew
QUANTITY_SKEW_SCENARIO = NonIIDConfig(
    name="quantity_skew_exponential",
    description="Exponential quantity skew - some clients have 10x more data",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    quantity_skew=QuantitySkewConfig(
        distribution="exponential",
        min_samples=20,
        max_samples=500,
        imbalance_ratio=10.0,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Power law quantity (realistic)
QUANTITY_POWERLAW = NonIIDConfig(
    name="quantity_powerlaw",
    description="Power law distribution - 20% clients have 80% of data",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    quantity_skew=QuantitySkewConfig(
        distribution="powerlaw",
        large_node_fraction=0.2,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# FEATURE SKEW SCENARIOS
# =============================================================================

# Noise heterogeneity
FEATURE_SKEW_SCENARIO = NonIIDConfig(
    name="feature_skew_noise",
    description="Feature skew - different noise levels across clients",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    feature_skew=FeatureSkewConfig(
        noise_levels=(0.0, 0.5),
        feature_scaling=False,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Covariate shift
FEATURE_COVARIATE_SHIFT = NonIIDConfig(
    name="feature_covariate_shift",
    description="Covariate shift - clients have different feature distributions",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    feature_skew=FeatureSkewConfig(
        noise_levels=(0.1, 0.3),
        feature_scaling=True,
        covariate_shift_intensity=0.3,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# TEMPORAL SKEW SCENARIOS
# =============================================================================

# Gradual drift
TEMPORAL_DRIFT_SCENARIO = NonIIDConfig(
    name="temporal_gradual_drift",
    description="Gradual temporal drift - distribution changes slowly over rounds",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    temporal_skew=TemporalSkewConfig(
        drift_type="gradual",
        drift_intensity=0.3,
        concept_drift=False,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Sudden drift
TEMPORAL_SUDDEN_DRIFT = NonIIDConfig(
    name="temporal_sudden_drift",
    description="Sudden temporal drift - abrupt distribution change at midpoint",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    temporal_skew=TemporalSkewConfig(
        drift_type="sudden",
        drift_intensity=0.5,
        concept_drift=False,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Periodic drift
TEMPORAL_PERIODIC_DRIFT = NonIIDConfig(
    name="temporal_periodic_drift",
    description="Periodic temporal drift - oscillating distribution changes",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    temporal_skew=TemporalSkewConfig(
        drift_type="periodic",
        drift_intensity=0.3,
        drift_frequency=2,
        concept_drift=False,
    ),
    byzantine_fraction=0.3,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# COMBINED SCENARIOS (REALISTIC)
# =============================================================================

# Healthcare scenario: hospitals with different patient populations + data volumes
HEALTHCARE_SCENARIO = NonIIDConfig(
    name="healthcare_realistic",
    description="Healthcare: hospitals with different patient populations and data volumes",
    n_clients=20,
    n_samples_train=10000,
    n_samples_test=2000,
    dataset="emnist",  # Simulating medical imaging
    label_skew=LabelSkewConfig(alpha=0.5),  # Different disease prevalence
    quantity_skew=QuantitySkewConfig(distribution="powerlaw"),  # Larger hospitals
    byzantine_fraction=0.15,  # Lower Byzantine assumption for healthcare
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=15,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Mobile scenario: phones with different usage patterns
MOBILE_SCENARIO = NonIIDConfig(
    name="mobile_realistic",
    description="Mobile: phones with different user behaviors and data volumes",
    n_clients=100,
    n_samples_train=5000,
    n_samples_test=1000,
    dataset="emnist",  # Simulating keyboard/input data
    label_skew=LabelSkewConfig(alpha=0.3),  # Very heterogeneous usage
    quantity_skew=QuantitySkewConfig(distribution="exponential"),  # Varies by usage
    feature_skew=FeatureSkewConfig(noise_levels=(0.0, 0.2)),  # Device differences
    byzantine_fraction=0.2,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# IoT scenario: sensors with different calibrations + drift
IOT_SCENARIO = NonIIDConfig(
    name="iot_realistic",
    description="IoT: sensors with calibration drift and environmental variations",
    n_clients=50,
    n_samples_train=4000,
    n_samples_test=500,
    dataset="emnist",  # Simulating sensor readings
    feature_skew=FeatureSkewConfig(
        noise_levels=(0.1, 0.4),
        feature_scaling=True,
        covariate_shift_intensity=0.2,
    ),
    temporal_skew=TemporalSkewConfig(
        drift_type="gradual",
        drift_intensity=0.2,
    ),
    byzantine_fraction=0.25,  # Higher Byzantine for IoT (compromised devices)
    attack_type=AttackType.GAUSSIAN_NOISE,  # Simulates malfunctioning sensors
    n_rounds=12,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# HIGH BYZANTINE FRACTION SCENARIOS (STRESS TESTS)
# =============================================================================

# 40% Byzantine with moderate non-IID
HIGH_BYZ_MODERATE_NONIID = NonIIDConfig(
    name="high_byz_moderate_noniid",
    description="Stress test: 40% Byzantine + moderate non-IID (alpha=0.5)",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.5),
    byzantine_fraction=0.4,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# 50% Byzantine (BFT limit) with non-IID
BFT_LIMIT_NON_IID = NonIIDConfig(
    name="bft_limit_non_iid",
    description="BFT limit test: 50% Byzantine + severe non-IID (alpha=0.1)",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.1),
    byzantine_fraction=0.5,
    attack_type=AttackType.SIGN_FLIP,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS_GEN7,  # Need Gen7 for 50% BFT
    seeds=[42, 101, 202],
)


# =============================================================================
# ATTACK TYPE VARIATIONS
# =============================================================================

# Model replacement attack with non-IID
MODEL_REPLACEMENT_NON_IID = NonIIDConfig(
    name="model_replacement_non_iid",
    description="Model replacement attack under moderate non-IID",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.5),
    byzantine_fraction=0.3,
    attack_type=AttackType.MODEL_REPLACEMENT,
    attack_intensity=10.0,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)

# Scaled gradient attack with non-IID
SCALED_GRADIENT_NON_IID = NonIIDConfig(
    name="scaled_gradient_non_iid",
    description="Scaled gradient attack under moderate non-IID",
    n_clients=50,
    n_samples_train=6000,
    dataset="emnist",
    label_skew=LabelSkewConfig(alpha=0.5),
    byzantine_fraction=0.3,
    attack_type=AttackType.SCALED_GRADIENT,
    attack_intensity=5.0,
    n_rounds=10,
    aggregator=AggregatorType.AEGIS,
    seeds=[42, 101, 202],
)


# =============================================================================
# AGGREGATOR COMPARISON SCENARIOS
# =============================================================================

def create_aggregator_comparison_scenarios(
    base_alpha: float = 0.5,
    byzantine_fraction: float = 0.3,
) -> List[NonIIDConfig]:
    """Create scenarios comparing different aggregators on same non-IID data."""
    aggregators = [
        AggregatorType.KRUM,
        AggregatorType.TRIMMED_MEAN,
        AggregatorType.MEDIAN,
        AggregatorType.AEGIS,
        AggregatorType.AEGIS_GEN7,
    ]

    scenarios = []
    for agg in aggregators:
        scenario = NonIIDConfig(
            name=f"agg_comparison_{agg.value}_alpha{base_alpha}",
            description=f"{agg.value} aggregator with alpha={base_alpha}",
            n_clients=50,
            n_samples_train=6000,
            dataset="emnist",
            label_skew=LabelSkewConfig(alpha=base_alpha),
            byzantine_fraction=byzantine_fraction,
            attack_type=AttackType.SIGN_FLIP,
            n_rounds=10,
            aggregator=agg,
            seeds=[42, 101, 202],
        )
        scenarios.append(scenario)

    return scenarios


# =============================================================================
# ALPHA SWEEP SCENARIOS
# =============================================================================

def create_alpha_sweep_scenarios(
    alphas: List[float] = [100.0, 1.0, 0.5, 0.3, 0.1, 0.05],
    byzantine_fraction: float = 0.3,
) -> List[NonIIDConfig]:
    """Create scenarios sweeping across alpha values."""
    scenarios = []
    for alpha in alphas:
        severity = LabelSkewConfig(alpha=alpha).get_severity()
        scenario = NonIIDConfig(
            name=f"alpha_sweep_{alpha}",
            description=f"Alpha sweep: {severity} non-IID (alpha={alpha})",
            n_clients=50,
            n_samples_train=6000,
            dataset="emnist",
            label_skew=LabelSkewConfig(alpha=alpha),
            byzantine_fraction=byzantine_fraction,
            attack_type=AttackType.SIGN_FLIP,
            n_rounds=10,
            aggregator=AggregatorType.AEGIS,
            seeds=[42, 101, 202, 303, 404],  # More seeds for sweep
        )
        scenarios.append(scenario)

    return scenarios


# =============================================================================
# SCENARIO COLLECTIONS
# =============================================================================

# Core scenarios for quick validation
CORE_SCENARIOS = [
    IID_BASELINE,
    MILD_NON_IID,
    MODERATE_NON_IID,
    SEVERE_NON_IID,
]

# Extended scenarios for comprehensive validation
EXTENDED_SCENARIOS = CORE_SCENARIOS + [
    PATHOLOGICAL_NON_IID,
    QUANTITY_SKEW_SCENARIO,
    FEATURE_SKEW_SCENARIO,
    TEMPORAL_DRIFT_SCENARIO,
]

# Full test suite
FULL_SCENARIO_SUITE = EXTENDED_SCENARIOS + [
    QUANTITY_POWERLAW,
    FEATURE_COVARIATE_SHIFT,
    TEMPORAL_SUDDEN_DRIFT,
    TEMPORAL_PERIODIC_DRIFT,
    HEALTHCARE_SCENARIO,
    MOBILE_SCENARIO,
    IOT_SCENARIO,
    HIGH_BYZ_MODERATE_NONIID,
    BFT_LIMIT_NON_IID,
    MODEL_REPLACEMENT_NON_IID,
    SCALED_GRADIENT_NON_IID,
]


def get_all_scenarios() -> List[NonIIDConfig]:
    """Get all defined scenarios."""
    return FULL_SCENARIO_SUITE


def get_core_scenarios() -> List[NonIIDConfig]:
    """Get core scenarios for quick validation."""
    return CORE_SCENARIOS


def get_extended_scenarios() -> List[NonIIDConfig]:
    """Get extended scenarios for comprehensive validation."""
    return EXTENDED_SCENARIOS


def get_scenario_by_name(name: str) -> NonIIDConfig:
    """Get a specific scenario by name."""
    for scenario in get_all_scenarios():
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unknown scenario: {name}")


def get_scenarios_by_type(skew_type: str) -> List[NonIIDConfig]:
    """Get all scenarios of a specific type.

    Args:
        skew_type: One of "label", "quantity", "feature", "temporal", "combined"
    """
    result = []
    for scenario in get_all_scenarios():
        if skew_type == "label" and scenario.label_skew:
            result.append(scenario)
        elif skew_type == "quantity" and scenario.quantity_skew:
            result.append(scenario)
        elif skew_type == "feature" and scenario.feature_skew:
            result.append(scenario)
        elif skew_type == "temporal" and scenario.temporal_skew:
            result.append(scenario)
        elif skew_type == "combined":
            n_skews = sum([
                scenario.label_skew is not None,
                scenario.quantity_skew is not None,
                scenario.feature_skew is not None,
                scenario.temporal_skew is not None,
            ])
            if n_skews >= 2:
                result.append(scenario)
    return result


def print_scenario_summary():
    """Print summary of all available scenarios."""
    print("=" * 80)
    print("NON-IID VALIDATION SCENARIOS")
    print("=" * 80)

    categories = [
        ("Core (Quick)", CORE_SCENARIOS),
        ("Extended", EXTENDED_SCENARIOS),
        ("Full Suite", FULL_SCENARIO_SUITE),
    ]

    for cat_name, scenarios in categories:
        print(f"\n{cat_name} ({len(scenarios)} scenarios):")
        print("-" * 40)
        for s in scenarios[:5]:  # Show first 5
            skews = []
            if s.label_skew:
                skews.append(f"label(a={s.label_skew.alpha})")
            if s.quantity_skew:
                skews.append(f"qty({s.quantity_skew.distribution})")
            if s.feature_skew:
                skews.append("feature")
            if s.temporal_skew:
                skews.append(f"temporal({s.temporal_skew.drift_type})")
            skew_str = ", ".join(skews) if skews else "IID"
            print(f"  {s.name}: {skew_str}")
        if len(scenarios) > 5:
            print(f"  ... and {len(scenarios) - 5} more")


if __name__ == "__main__":
    print_scenario_summary()
