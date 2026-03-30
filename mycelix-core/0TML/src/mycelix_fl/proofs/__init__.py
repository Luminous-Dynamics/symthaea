# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Proofs Module

Zero-knowledge proof generation and verification for federated learning.

This module provides Python wrappers around the Rust zkSTARK proof system,
enabling gradient quality proofs that demonstrate honest computation
without revealing underlying training data.

Example:
    >>> from mycelix_fl.proofs import ProofOfGradientQuality, GradientProofConfig
    >>> import numpy as np
    >>>
    >>> # Generate gradient
    >>> gradients = np.random.randn(1000).astype(np.float32) * 0.1
    >>>
    >>> # Create proof
    >>> config = GradientProofConfig(max_norm=10.0, round_number=1)
    >>> proof = ProofOfGradientQuality.generate(gradients, config)
    >>>
    >>> # Verify proof
    >>> result = proof.verify()
    >>> assert result.valid

Author: Luminous Dynamics
Version: 1.0.0
"""

from mycelix_fl.proofs.zkproof import (
    # Core types
    ProofOfGradientQuality,
    GradientProofConfig,
    GradientStatistics,
    VerificationResult,
    # Batch operations
    BatchProofVerifier,
    # Constants
    PROOF_VERSION,
    MAX_GRADIENT_NORM,
    DEFAULT_SECURITY_LEVEL,
    # Utilities
    compute_gradient_commitment,
    serialize_proof,
    deserialize_proof,
    # Status checking
    has_rust_proofs,
    get_proof_backend_info,
)

__all__ = [
    # Core types
    "ProofOfGradientQuality",
    "GradientProofConfig",
    "GradientStatistics",
    "VerificationResult",
    # Batch operations
    "BatchProofVerifier",
    # Constants
    "PROOF_VERSION",
    "MAX_GRADIENT_NORM",
    "DEFAULT_SECURITY_LEVEL",
    # Utilities
    "compute_gradient_commitment",
    "serialize_proof",
    "deserialize_proof",
    # Status checking
    "has_rust_proofs",
    "get_proof_backend_info",
]
