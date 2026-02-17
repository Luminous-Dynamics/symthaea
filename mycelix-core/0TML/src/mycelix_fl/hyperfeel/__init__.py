"""
Mycelix FL HyperFeel Module

HyperFeel Protocol v2.0 implementation with:
- HV16 gradient compression (2000x compression)
- Real Φ measurement integration
- Causal structure encoding
- Temporal trajectory tracking
"""

from mycelix_fl.hyperfeel.encoder_v2 import (
    HyperFeelEncoderV2,
    HyperGradient,
    EncodingConfig,
)

__all__ = [
    "HyperFeelEncoderV2",
    "HyperGradient",
    "EncodingConfig",
]
