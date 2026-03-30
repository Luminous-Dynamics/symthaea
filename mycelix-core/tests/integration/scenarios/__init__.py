# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Scenario packages for Mycelix FL integration tests.

Provides domain-specific test scenarios including:
- Healthcare (HIPAA-compliant federated learning)
- Adversarial (coordinated attack scenarios)
"""

from .healthcare import (
    HIPAACompliantNode,
    HospitalNode,
    HealthcareNetworkConfig,
    HealthcareScenario,
)

from .adversarial import (
    AdversarialScenario,
    CoordinatedAttack,
    SybilAttack,
    AdaptiveAttack,
)

__all__ = [
    # Healthcare
    "HIPAACompliantNode",
    "HospitalNode",
    "HealthcareNetworkConfig",
    "HealthcareScenario",
    # Adversarial
    "AdversarialScenario",
    "CoordinatedAttack",
    "SybilAttack",
    "AdaptiveAttack",
]
