# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Factor Decentralized Identity (MFDI) System for Mycelix
Phase 1 + Phase 2 MATL Integration + Week 5-6 DHT Integration + Week 7-8 Governance

Provides graduated identity verification with multi-factor authentication,
social recovery mechanisms, MATL trust scoring, and Holochain DHT integration.

Week 5-6 Addition:
- DHT_IdentityCoordinator: Bridges Python identity with Holochain DHT

Week 7-8 Addition:
- IdentityGovernanceExtensions: Governance-specific identity functions
- Capability-based access control
- Vote weight calculation
- Guardian authorization for emergency actions
"""

from .did_manager import DIDManager, MycelixDID
from .factors import (
    IdentityFactor,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    BiometricFactor,
    HardwareKeyFactor,
    FactorStatus,
    FactorCategory,
)
from .assurance import AssuranceLevel, calculate_assurance_level
from .recovery import RecoveryManager, RecoveryRequest, RecoveryScenario
from .verifiable_credentials import VCManager, VerifiableCredential, VCType
from .matl_integration import (
    IdentityMATLBridge,
    IdentityTrustSignal,
    EnhancedMATLScore,
    IdentityRiskLevel,
)
from .dht_coordinator import (
    DHT_IdentityCoordinator,
    IdentityCoordinatorConfig
)
from .governance_extensions import (
    IdentityGovernanceExtensions,
    Capability,
    CAPABILITY_REGISTRY,
    ASSURANCE_FACTORS,
    BASE_BUDGET,
)
from .gitcoin_passport import (
    GitcoinPassportClient,
    GitcoinPassportVerifier,
    PassportScore,
    Stamp,
    StampProvider,
    GovernanceVerificationResult,
    PassportError,
    RateLimitError,
    AuthenticationError,
    AddressNotFoundError,
    RetryExhaustedError,
    RetryConfig,
    PassportCache,
    CacheEntry,
    verify_humanity,
    get_passport_score,
)

__all__ = [
    # Core DID Management
    "DIDManager",
    "MycelixDID",
    # Identity Factors
    "IdentityFactor",
    "CryptoKeyFactor",
    "GitcoinPassportFactor",
    "SocialRecoveryFactor",
    "BiometricFactor",
    "HardwareKeyFactor",
    "FactorStatus",
    "FactorCategory",
    # Assurance Levels
    "AssuranceLevel",
    "calculate_assurance_level",
    # Recovery
    "RecoveryManager",
    "RecoveryRequest",
    "RecoveryScenario",
    # Verifiable Credentials
    "VCManager",
    "VerifiableCredential",
    "VCType",
    # MATL Integration
    "IdentityMATLBridge",
    "IdentityTrustSignal",
    "EnhancedMATLScore",
    "IdentityRiskLevel",
    # DHT Integration (Week 5-6)
    "DHT_IdentityCoordinator",
    "IdentityCoordinatorConfig",
    # Governance Integration (Week 7-8)
    "IdentityGovernanceExtensions",
    "Capability",
    "CAPABILITY_REGISTRY",
    "ASSURANCE_FACTORS",
    "BASE_BUDGET",
]

__version__ = "0.4.0-alpha"  # Bumped for governance integration
