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
