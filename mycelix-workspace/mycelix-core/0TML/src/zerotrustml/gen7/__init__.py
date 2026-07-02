# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Gen-7: HYPERION-FL - Proof-Carrying Gradients.

Revolutionary approach to Byzantine fault tolerance through cryptographic
verification instead of heuristic detection.

Core Innovation:
- Gen-5 tried: Detect Byzantine behavior (failed empirically)
- Gen-7 does: Prove honest behavior (cryptographically guaranteed)

Components:
- gradient_proof: zkSTARK circuits for gradient provenance
- staking: Economic hardening via stake-weighted participation
- proof_chain: Sequential composition for temporal auditability
"""

__version__ = "0.1.0"

from .gradient_proof import (
    GradientProofCircuit,
    GradientProof,
    prove_gradient,
    verify_gradient_proof,
)
from .staking import StakingCoordinator
from .proof_chain import ProofChain

__all__ = [
    "GradientProofCircuit",
    "GradientProof",
    "prove_gradient",
    "verify_gradient_proof",
    "StakingCoordinator",
    "ProofChain",
]
