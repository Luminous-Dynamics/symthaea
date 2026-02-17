#!/usr/bin/env python3
"""
ZeroTrustML Storage Backends

Modular storage backend system for Phase 10 federated learning.
Supports multiple storage backends: PostgreSQL, Holochain, Ethereum, Cosmos, LocalFile.
"""

from .storage_backend import (
    StorageBackend,
    BackendType,
    GradientRecord,
    CreditRecord,
    ByzantineEvent
)
from .postgresql_backend import PostgreSQLBackend
from .localfile_backend import LocalFileBackend
from .holochain_backend import HolochainBackend
from .holochain_event_listener import (
    HolochainEventListener,
    GradientSignal,
    ReputationSignal,
    RoundSignal,
    create_gradient_aggregator_integration,
)
from .ethereum_backend import EthereumBackend
from .cosmos_backend import CosmosBackend

__all__ = [
    "StorageBackend",
    "BackendType",
    "GradientRecord",
    "CreditRecord",
    "ByzantineEvent",
    "PostgreSQLBackend",
    "LocalFileBackend",
    "HolochainBackend",
    "HolochainEventListener",
    "GradientSignal",
    "ReputationSignal",
    "RoundSignal",
    "create_gradient_aggregator_integration",
    "EthereumBackend",
    "CosmosBackend",
]
