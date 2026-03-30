# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Zero-Knowledge Proof Module

Provides ZK proof integration for the Federated Learning pipeline:
- RISC Zero zkSTARK proof verification for gradient validity
- Proof deserialization and validation
- Metrics tracking for ZK operations
- Configuration management

The ZK proofs verify:
1. Gradient contains no invalid values (NaN/Inf)
2. Gradient L2 norm is within bounds
3. Gradient is bound to a specific node and round
4. Gradient matches the claimed commitment (hash)

Usage:
    from zerotrustml.zk import (
        RISCZeroVerifier,
        ZKProofConfig,
        ZKVerificationResult,
        ZKVerificationError,
    )

    # Initialize verifier
    verifier = RISCZeroVerifier()

    # Verify a proof
    result = verifier.verify(proof_bytes)
    if result.is_valid:
        print(f"Proof verified for round {result.round_number}")
"""

from zerotrustml.zk.verifier import (
    RISCZeroVerifier,
    ZKProofConfig,
    ZKVerificationResult,
    ZKGradientProof,
    create_verifier,
    check_gen7_zkstark_available,
    preflight_check_production_environment,
    verify_proof_verification_is_real,
)
from zerotrustml.zk.exceptions import (
    ZKVerificationError,
    ZKProofDeserializationError,
    ZKProofTimeoutError,
    ZKProofInvalidError,
    ZKSecurityError,
)

__all__ = [
    # Main verifier
    "RISCZeroVerifier",
    # Configuration
    "ZKProofConfig",
    # Results
    "ZKVerificationResult",
    "ZKGradientProof",
    # Factory functions
    "create_verifier",
    "check_gen7_zkstark_available",
    # SEC-004: Pre-flight checks
    "preflight_check_production_environment",
    "verify_proof_verification_is_real",
    # Exceptions
    "ZKVerificationError",
    "ZKProofDeserializationError",
    "ZKProofTimeoutError",
    "ZKProofInvalidError",
    "ZKSecurityError",
]
