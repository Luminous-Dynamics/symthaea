"""
Core federated learning components.
"""

# Make imports conditional to avoid torch dependency
# This allows Phase 10 coordinator to be imported without torch
try:
    from zerotrustml.core.node import Node, NodeConfig, GradientCheckpoint
except ImportError:
    Node = None
    NodeConfig = None
    GradientCheckpoint = None

try:
    from zerotrustml.core.training import (
        SimpleNN,
        RealMLNode,
        Trainer,
        TrainingConfig
    )
except ImportError:
    SimpleNN = None
    RealMLNode = None
    Trainer = None
    TrainingConfig = None

__all__ = [
    "Node",
    "NodeConfig",
    "GradientCheckpoint",
    "SimpleNN",
    "RealMLNode",
    "Trainer",
    "TrainingConfig"
]
