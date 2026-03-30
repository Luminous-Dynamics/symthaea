# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Holochain - Decentralized Federated Learning with Economic Incentives

This package provides a fully decentralized P2P federated learning system using:
- Holochain DHT for peer coordination and gradient storage
- ZeroTrustML Credits for economic incentives
- Byzantine-resistant aggregation algorithms
- Real PyTorch training with PoGQ quality scoring

Quick Start:
    >>> from zerotrustml import Node, NodeConfig
    >>> config = NodeConfig(node_id="hospital-1", data_path="/data/medical")
    >>> node = Node(config)
    >>> await node.start()

For deployment guide, see: https://zerotrustml.readthedocs.io
"""

__version__ = "1.0.0"
__author__ = "Luminous Dynamics"
__license__ = "MIT"

# Logging and error handling are always available (no torch dependency)
from zerotrustml.logging import (
    get_logger,
    configure_logging,
    correlation_id,
    correlation_context,
    async_correlation_context,
    with_correlation_id,
    timed,
    async_timed,
    log_operation,
    async_log_operation,
    StructuredJSONFormatter,
    LoggingConfig,
)

from zerotrustml.exceptions import (
    MycelixError,
    ErrorCode,
    ByzantineDetectionError,
    GradientPoisoningError,
    SybilAttackError,
    HolochainConnectionError,
    HolochainZomeError,
    ProofVerificationError,
    ZKPoCVerificationError,
    ConfigurationError,
    ConfigMissingError,
    IdentityError,
    IdentityNotFoundError,
    IdentityVerificationError,
    GovernanceError,
    GovernanceUnauthorizedError,
    CapabilityDeniedError,
    GuardianAuthorizationError,
    StorageError,
    AggregationError,
    wrap_exception,
)

from zerotrustml.error_handling import (
    retry_with_backoff,
    async_retry,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    ErrorAggregator,
    BatchOperationError,
    Fallback,
    RateLimiter,
)

# Core exports - make imports optional to avoid torch dependency
# This allows Phase 10 backends to be imported without torch
try:
    from zerotrustml.core import Node, NodeConfig, Trainer, TrainingConfig, SimpleNN, RealMLNode
    from zerotrustml.credits import CreditSystem, ReputationLevel
    from zerotrustml.aggregation import (
        FedAvg,
        Krum,
        TrimmedMean,
        ReputationWeighted,
        CoordinateMedian,
        HierarchicalAggregator,
        aggregate_gradients,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    # torch not available - Phase 10 components can still work
    _TORCH_AVAILABLE = False
    Node = None
    NodeConfig = None
    Trainer = None
    TrainingConfig = None
    SimpleNN = None
    RealMLNode = None
    CreditSystem = None
    ReputationLevel = None
    FedAvg = None
    Krum = None
    TrimmedMean = None
    ReputationWeighted = None
    CoordinateMedian = None
    HierarchicalAggregator = None
    aggregate_gradients = None

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",

    # Logging
    "get_logger",
    "configure_logging",
    "correlation_id",
    "correlation_context",
    "async_correlation_context",
    "with_correlation_id",
    "timed",
    "async_timed",
    "log_operation",
    "async_log_operation",
    "StructuredJSONFormatter",
    "LoggingConfig",

    # Exceptions
    "MycelixError",
    "ErrorCode",
    "ByzantineDetectionError",
    "GradientPoisoningError",
    "SybilAttackError",
    "HolochainConnectionError",
    "HolochainZomeError",
    "ProofVerificationError",
    "ZKPoCVerificationError",
    "ConfigurationError",
    "ConfigMissingError",
    "IdentityError",
    "IdentityNotFoundError",
    "IdentityVerificationError",
    "GovernanceError",
    "GovernanceUnauthorizedError",
    "CapabilityDeniedError",
    "GuardianAuthorizationError",
    "StorageError",
    "AggregationError",
    "wrap_exception",

    # Error Handling
    "retry_with_backoff",
    "async_retry",
    "RetryConfig",
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "ErrorAggregator",
    "BatchOperationError",
    "Fallback",
    "RateLimiter",

    # Core classes
    "Node",
    "NodeConfig",
    "Trainer",
    "TrainingConfig",
    "SimpleNN",
    "RealMLNode",

    # Credits
    "CreditSystem",
    "ReputationLevel",

    # Aggregation
    "FedAvg",
    "Krum",
    "TrimmedMean",
    "ReputationWeighted",
    "CoordinateMedian",
    "HierarchicalAggregator",
    "aggregate_gradients",
]
