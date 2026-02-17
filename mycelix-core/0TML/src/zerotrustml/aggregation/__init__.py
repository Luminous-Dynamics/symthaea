"""
Aggregation algorithms for Byzantine-resistant federated learning.

This module provides multiple aggregation strategies:
- FedAvg: Standard weighted averaging (fast but vulnerable)
- Krum: Byzantine-resistant selection
- TrimmedMean: Remove outliers and average
- CoordinateMedian: Median per parameter
- HierarchicalAggregator: Tree-structured for O(n log n) scaling

For production deployments, use the Rust fl-aggregator library via PyO3 bindings:
    from fl_aggregator import Defense, Aggregator
"""

from zerotrustml.aggregation.algorithms import (
    FedAvg,
    Krum,
    TrimmedMean,
    ReputationWeighted,
    CoordinateMedian,
    HierarchicalAggregator,
    aggregate_gradients,
)

__all__ = [
    # Basic algorithms (reference implementations)
    # For production, use fl-aggregator Rust bindings
    "FedAvg",
    "Krum",
    "TrimmedMean",
    "ReputationWeighted",
    "CoordinateMedian",
    # Hierarchical
    "HierarchicalAggregator",
    "aggregate_gradients",
]
