# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain integration for ZeroTrustML / Mycelix.

Provides Python clients for Holochain DHT operations:

Primary Clients:
- FLCoordinatorClient: Federated learning coordination (gradients, reputation, Byzantine detection)
- HolochainClient: General gradient storage and credits
- IdentityDHTClient: Identity DHT operations (DID, factors, reputation, guardians)

Architecture:
    Holochain Zomes (Rust) → On-chain validation, storage, reputation
    Python Clients → Orchestration, ML framework bridges
    fl-aggregator (Rust/PyO3) → High-performance aggregation

For production Byzantine detection, use:
1. FLCoordinatorClient to call the pogq_zome for on-chain PoGQ validation
2. fl-aggregator Rust library for aggregation algorithms

See docs/ARCHITECTURE.md for the full hybrid architecture.
"""

from .client import HolochainClient
from .fl_coordinator_client import (
    FLCoordinatorClient,
    create_fl_coordinator,
    GradientSubmission,
    ModelGradient,
    NodeReputation,
    ByzantineDetectionResult,
    CartelDetectionResult,
    RoundStats,
    ByzantineType,
)
from .identity_dht_client import (
    IdentityDHTClient,
    create_identity_client,
    DIDDocument,
    IdentityFactor,
    IdentitySignals,
    ReputationEntry,
    AggregatedReputation,
    GuardianRelationship,
    GuardianGraphMetrics
)
from . import bridges

__all__ = [
    # FL Coordinator (primary FL interface - calls Rust zomes)
    "FLCoordinatorClient",
    "create_fl_coordinator",
    "GradientSubmission",
    "ModelGradient",
    "NodeReputation",
    "ByzantineDetectionResult",
    "CartelDetectionResult",
    "RoundStats",
    "ByzantineType",
    # Legacy client
    "HolochainClient",
    # Identity DHT
    "IdentityDHTClient",
    "create_identity_client",
    "DIDDocument",
    "IdentityFactor",
    "IdentitySignals",
    "ReputationEntry",
    "AggregatedReputation",
    "GuardianRelationship",
    "GuardianGraphMetrics",
    # Bridges
    "bridges",
]
