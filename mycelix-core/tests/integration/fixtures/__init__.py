"""
Fixtures package for Mycelix FL integration tests.

Provides reusable components for network setup, node simulation, and metrics collection.
"""

from .network import (
    FLNetwork,
    NetworkConfig,
    FLAggregator,
    ZomeValidator,
    ReputationBridge,
    GradientStore,
)

from .nodes import (
    HonestNode,
    ByzantineNode,
    NodeBehavior,
    AttackType,
    NodeMetrics,
)

from .metrics import (
    MetricsCollector,
    TestReport,
    AggregationMetrics,
    DetectionMetrics,
    LatencyMetrics,
)

__all__ = [
    # Network
    "FLNetwork",
    "NetworkConfig",
    "FLAggregator",
    "ZomeValidator",
    "ReputationBridge",
    "GradientStore",
    # Nodes
    "HonestNode",
    "ByzantineNode",
    "NodeBehavior",
    "AttackType",
    "NodeMetrics",
    # Metrics
    "MetricsCollector",
    "TestReport",
    "AggregationMetrics",
    "DetectionMetrics",
    "LatencyMetrics",
]
