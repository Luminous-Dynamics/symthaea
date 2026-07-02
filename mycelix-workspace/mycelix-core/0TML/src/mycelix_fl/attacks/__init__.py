# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Attack Module

Byzantine attack implementations for testing and validation.
These attacks are for TESTING PURPOSES ONLY.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from mycelix_fl.attacks.advanced_attacks import (
    # Enums
    AttackType,
    # Config
    AttackConfig,
    AttackResult,
    # Base class
    ByzantineAttack,
    # Attack implementations
    GradientScalingAttack,
    SignFlipAttack,
    GaussianNoiseAttack,
    LittleIsEnoughAttack,
    IPMAttack,
    FreeRiderAttack,
    BackdoorAttack,
    AdaptiveAttack,
    CartelAttack,
    # Factory
    create_attack,
    # Orchestrator
    AttackOrchestrator,
    # Scenarios
    create_gradient_scaling_scenario,
    create_adaptive_attack_scenario,
    create_cartel_scenario,
    create_mixed_attack_scenario,
)

__all__ = [
    # Enums
    "AttackType",
    # Config
    "AttackConfig",
    "AttackResult",
    # Base class
    "ByzantineAttack",
    # Attack implementations
    "GradientScalingAttack",
    "SignFlipAttack",
    "GaussianNoiseAttack",
    "LittleIsEnoughAttack",
    "IPMAttack",
    "FreeRiderAttack",
    "BackdoorAttack",
    "AdaptiveAttack",
    "CartelAttack",
    # Factory
    "create_attack",
    # Orchestrator
    "AttackOrchestrator",
    # Scenarios
    "create_gradient_scaling_scenario",
    "create_adaptive_attack_scenario",
    "create_cartel_scenario",
    "create_mixed_attack_scenario",
]
