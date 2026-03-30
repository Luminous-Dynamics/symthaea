# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Governance System
Week 7-8 Phase 3: Governance Coordinator + Phase 4: FL Integration

Provides complete governance functionality for Zero-TrustML including:
- Proposal lifecycle management
- Vote collection and tallying
- Capability-based access control
- Guardian authorization for emergency actions
- Integration with identity system and DHT storage
- FL coordinator integration (capability checks, emergency actions, rewards)
"""

from .coordinator import (
    GovernanceCoordinator,
    ProposalManager,
    VotingEngine,
    CapabilityEnforcer,
    GuardianAuthorizationManager,
)
from .models import (
    ProposalType,
    ProposalStatus,
    VoteChoice,
    ProposalData,
    VoteData,
    AuthorizationStatus,
)
from .fl_integration import (
    FLGovernanceIntegration,
    FLGovernanceConfig,
)

__all__ = [
    # Main Coordinator
    "GovernanceCoordinator",

    # Component Managers
    "ProposalManager",
    "VotingEngine",
    "CapabilityEnforcer",
    "GuardianAuthorizationManager",

    # FL Integration
    "FLGovernanceIntegration",
    "FLGovernanceConfig",

    # Data Models
    "ProposalType",
    "ProposalStatus",
    "VoteChoice",
    "ProposalData",
    "VoteData",
    "AuthorizationStatus",
]

__version__ = "0.2.0-alpha"  # Bumped for FL integration
