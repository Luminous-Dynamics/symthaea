"""
Mycelix FL Machine Learning Bridge

Compatibility layer for popular ML frameworks:
- PyTorch (full support)
- TensorFlow (full support)
- JAX (experimental)
- NumPy (always available)

Provides unified interface for:
- Gradient extraction
- Model state management
- Training callbacks
- Aggregation application
"""

from mycelix_fl.ml.bridge import (
    MLBridge,
    PyTorchBridge,
    TensorFlowBridge,
    NumPyBridge,
    create_bridge,
    detect_framework,
)

__all__ = [
    "MLBridge",
    "PyTorchBridge",
    "TensorFlowBridge",
    "NumPyBridge",
    "create_bridge",
    "detect_framework",
]
