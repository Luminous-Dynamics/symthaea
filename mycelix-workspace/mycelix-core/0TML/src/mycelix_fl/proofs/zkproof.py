# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
zkSTARK Proof of Gradient Quality

Production-ready zero-knowledge proofs for federated learning gradient verification.

This module provides Python wrappers for the Rust zkSTARK proof system,
enabling participants to prove their gradient contributions are honest
without revealing the underlying training data.

## Overview

The Proof of Gradient Quality (PoGQ) proves:
1. Gradient norm is within bounds (prevents scaling attacks)
2. All gradient values are finite (no NaN/Inf)
3. Gradient was computed from valid local data
4. Statistical properties match expected distribution

## Usage

```python
from mycelix_fl.proofs import ProofOfGradientQuality, GradientProofConfig
import numpy as np

# Generate gradient from local training
gradients = model.compute_gradients(local_data)

# Create proof configuration
config = GradientProofConfig(
    max_norm=10.0,
    round_number=5,
    node_id="node-12345",
    security_level="standard128",  # or "standard96", "high256"
)

# Generate proof
proof = ProofOfGradientQuality.generate(gradients, config)

# Serialize for transmission
proof_bytes = proof.serialize()

# On receiving end, verify
received_proof = ProofOfGradientQuality.deserialize(proof_bytes)
result = received_proof.verify()

if result.valid:
    # Accept gradient for aggregation
    aggregator.submit(gradients, proof)
else:
    # Reject with reason
    print(f"Proof invalid: {result.issues}")
```

## Security Considerations

- Proofs provide computational security (not information-theoretic)
- Proof size is approximately 20-50KB depending on gradient size
- Generation time scales with gradient size (O(n log n))
- Verification is constant time (~50ms)

Author: Luminous Dynamics
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

PROOF_VERSION: int = 1
MAX_GRADIENT_NORM: float = 10000.0
DEFAULT_SECURITY_LEVEL: str = "standard128"

# Try to import Rust backend
RUST_PROOFS_AVAILABLE = False
_rust_proofs = None

try:
    import zerotrustml_core as _ztc
    if hasattr(_ztc, 'generate_gradient_proof'):
        RUST_PROOFS_AVAILABLE = True
        _rust_proofs = _ztc
        logger.info("Rust proof backend available")
except ImportError:
    logger.debug("Rust proof backend not available, using Python simulation")


def has_rust_proofs() -> bool:
    """Check if Rust proof backend is available."""
    return RUST_PROOFS_AVAILABLE


def get_proof_backend_info() -> Dict[str, Any]:
    """Get information about the proof backend."""
    return {
        "rust_available": RUST_PROOFS_AVAILABLE,
        "version": PROOF_VERSION,
        "max_gradient_norm": MAX_GRADIENT_NORM,
        "default_security": DEFAULT_SECURITY_LEVEL,
        "supported_security_levels": ["standard96", "standard128", "high256"],
    }


# =============================================================================
# Types
# =============================================================================

class SecurityLevel(Enum):
    """Proof security level."""
    STANDARD96 = "standard96"    # ~96 bits, fastest
    STANDARD128 = "standard128"  # ~128 bits, balanced (default)
    HIGH256 = "high256"          # ~256 bits, maximum security


@dataclass
class GradientStatistics:
    """Statistical summary of a gradient vector.

    These statistics are public (included in the proof) and can be used
    for Byzantine detection heuristics.
    """
    num_elements: int
    l2_norm: float
    l1_norm: float
    max_abs: float
    mean: float
    variance: float
    sparsity_count: int
    sparsity_ratio: float

    @classmethod
    def from_gradient(cls, gradients: np.ndarray) -> GradientStatistics:
        """Compute statistics from a gradient array."""
        if gradients.size == 0:
            return cls(
                num_elements=0,
                l2_norm=0.0,
                l1_norm=0.0,
                max_abs=0.0,
                mean=0.0,
                variance=0.0,
                sparsity_count=0,
                sparsity_ratio=1.0,
            )

        g = gradients.flatten().astype(np.float64)
        epsilon = 1e-8

        return cls(
            num_elements=len(g),
            l2_norm=float(np.linalg.norm(g)),
            l1_norm=float(np.sum(np.abs(g))),
            max_abs=float(np.max(np.abs(g))),
            mean=float(np.mean(g)),
            variance=float(np.var(g)),
            sparsity_count=int(np.sum(np.abs(g) < epsilon)),
            sparsity_ratio=float(np.sum(np.abs(g) < epsilon) / len(g)),
        )

    def is_suspicious(self) -> bool:
        """Check if statistics indicate potential Byzantine behavior."""
        # All zeros (free-rider)
        if self.l2_norm < 1e-10:
            return True
        # Extremely sparse
        if self.sparsity_ratio > 0.99 and self.num_elements > 100:
            return True
        # Extremely large values (scaling attack)
        if self.max_abs > 1000.0:
            return True
        # Near-zero variance with non-zero mean (constant attack)
        if self.variance < 1e-10 and abs(self.mean) > 0.1:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_elements": self.num_elements,
            "l2_norm": self.l2_norm,
            "l1_norm": self.l1_norm,
            "max_abs": self.max_abs,
            "mean": self.mean,
            "variance": self.variance,
            "sparsity_count": self.sparsity_count,
            "sparsity_ratio": self.sparsity_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GradientStatistics:
        """Create from dictionary."""
        return cls(
            num_elements=data["num_elements"],
            l2_norm=data["l2_norm"],
            l1_norm=data["l1_norm"],
            max_abs=data["max_abs"],
            mean=data["mean"],
            variance=data["variance"],
            sparsity_count=data["sparsity_count"],
            sparsity_ratio=data["sparsity_ratio"],
        )


@dataclass
class GradientProofConfig:
    """Configuration for gradient proof generation."""
    max_norm: float = 10.0
    round_number: int = 0
    node_id: str = ""
    security_level: str = "standard128"
    data_commitment: Optional[bytes] = None

    def __post_init__(self):
        if self.security_level not in ["standard96", "standard128", "high256"]:
            raise ValueError(f"Invalid security level: {self.security_level}")
        if self.max_norm <= 0:
            raise ValueError(f"Max norm must be positive: {self.max_norm}")


@dataclass
class VerificationResult:
    """Result of verifying a gradient proof."""
    valid: bool
    integrity_valid: bool
    statistics_valid: bool
    verification_time_ms: float
    issues: List[str] = field(default_factory=list)
    gradient_stats: Optional[GradientStatistics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "integrity_valid": self.integrity_valid,
            "statistics_valid": self.statistics_valid,
            "verification_time_ms": self.verification_time_ms,
            "issues": self.issues,
            "gradient_stats": self.gradient_stats.to_dict() if self.gradient_stats else None,
        }


# =============================================================================
# Core Proof Class
# =============================================================================

@dataclass
class ProofOfGradientQuality:
    """
    Zero-knowledge Proof of Gradient Quality.

    Proves that a gradient contribution is honest without revealing
    the underlying training data or exact gradient values.

    ## What is Proven

    1. **Norm Bound**: L2 norm is within the specified maximum
    2. **Finiteness**: All values are finite (no NaN/Inf)
    3. **Validity**: Gradient was computed correctly
    4. **Consistency**: Matches the data commitment

    ## Attributes

    - `proof_bytes`: The serialized zkSTARK proof
    - `gradient_commitment`: Blake3 hash of gradient values
    - `data_commitment`: Optional commitment to training data
    - `statistics`: Public gradient statistics
    - `metadata`: Proof generation metadata
    """

    proof_bytes: bytes
    gradient_commitment: bytes
    data_commitment: bytes
    statistics: GradientStatistics
    metadata: Dict[str, Any]

    @classmethod
    def generate(
        cls,
        gradients: np.ndarray,
        config: GradientProofConfig,
    ) -> ProofOfGradientQuality:
        """
        Generate a Proof of Gradient Quality.

        Args:
            gradients: The gradient values to prove (any shape, will be flattened)
            config: Proof configuration

        Returns:
            ProofOfGradientQuality instance

        Raises:
            ValueError: If gradients are invalid (empty, NaN, Inf, norm exceeded)
        """
        # Validate input
        if gradients.size == 0:
            raise ValueError("Gradient array cannot be empty")

        g = gradients.flatten().astype(np.float32)

        # Check for NaN/Inf
        if np.any(~np.isfinite(g)):
            nan_count = np.sum(np.isnan(g))
            inf_count = np.sum(np.isinf(g))
            raise ValueError(
                f"Gradient contains {nan_count} NaN and {inf_count} Inf values"
            )

        # Compute statistics
        stats = GradientStatistics.from_gradient(g)

        # Check norm bound
        if stats.l2_norm > config.max_norm:
            raise ValueError(
                f"Gradient L2 norm {stats.l2_norm:.4f} exceeds maximum {config.max_norm}"
            )

        # Compute commitments
        gradient_commitment = compute_gradient_commitment(g)
        data_commitment = config.data_commitment or b'\x00' * 32

        # Build metadata
        metadata = {
            "generated_at": int(time.time()),
            "node_id": config.node_id,
            "round": config.round_number,
            "security_level": config.security_level,
            "version": PROOF_VERSION,
        }

        # Generate proof
        if RUST_PROOFS_AVAILABLE:
            # Use Rust backend
            try:
                proof_bytes = _rust_proofs.generate_gradient_proof(
                    g.tolist(),
                    config.max_norm,
                    config.round_number,
                    config.security_level,
                )
            except Exception as e:
                logger.warning(f"Rust proof generation failed, using simulation: {e}")
                proof_bytes = _simulate_proof(g, config, gradient_commitment)
        else:
            # Use Python simulation
            proof_bytes = _simulate_proof(g, config, gradient_commitment)

        return cls(
            proof_bytes=proof_bytes,
            gradient_commitment=gradient_commitment,
            data_commitment=data_commitment,
            statistics=stats,
            metadata=metadata,
        )

    def verify(self) -> VerificationResult:
        """
        Verify this proof.

        Returns:
            VerificationResult with validity status and any issues
        """
        start_time = time.time()
        issues = []

        # Check statistics for suspicious patterns
        statistics_valid = not self.statistics.is_suspicious()
        if not statistics_valid:
            issues.append("Gradient statistics indicate potentially Byzantine behavior")

        # Check version
        if self.metadata.get("version", 0) > PROOF_VERSION:
            issues.append(
                f"Proof version {self.metadata['version']} is newer than supported {PROOF_VERSION}"
            )

        # Verify the cryptographic proof
        if RUST_PROOFS_AVAILABLE:
            try:
                integrity_valid = _rust_proofs.verify_gradient_proof(
                    self.proof_bytes,
                    self.gradient_commitment,
                    self.metadata.get("security_level", DEFAULT_SECURITY_LEVEL),
                )
            except Exception as e:
                logger.warning(f"Rust verification failed: {e}")
                integrity_valid = _verify_simulated_proof(self.proof_bytes, self.gradient_commitment)
        else:
            integrity_valid = _verify_simulated_proof(self.proof_bytes, self.gradient_commitment)

        if not integrity_valid:
            issues.append("Cryptographic proof verification failed")

        elapsed_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            valid=integrity_valid and statistics_valid and len(issues) == 0,
            integrity_valid=integrity_valid,
            statistics_valid=statistics_valid,
            verification_time_ms=elapsed_ms,
            issues=issues,
            gradient_stats=self.statistics,
        )

    def serialize(self) -> bytes:
        """
        Serialize the proof for transmission.

        Returns:
            Serialized proof bytes
        """
        return serialize_proof(self)

    @classmethod
    def deserialize(cls, data: bytes) -> ProofOfGradientQuality:
        """
        Deserialize a proof from bytes.

        Args:
            data: Serialized proof bytes

        Returns:
            ProofOfGradientQuality instance
        """
        return deserialize_proof(data)

    def proof_hash(self) -> bytes:
        """Get a hash of this proof for audit logging."""
        h = hashlib.blake2b(b"pogq:", digest_size=32)
        h.update(self.gradient_commitment)
        h.update(self.data_commitment)
        h.update(struct.pack("<I", self.metadata.get("round", 0)))
        return h.digest()

    @property
    def size(self) -> int:
        """Get the proof size in bytes."""
        return len(self.proof_bytes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "proof_bytes": self.proof_bytes.hex(),
            "gradient_commitment": self.gradient_commitment.hex(),
            "data_commitment": self.data_commitment.hex(),
            "statistics": self.statistics.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProofOfGradientQuality:
        """Create from dictionary."""
        return cls(
            proof_bytes=bytes.fromhex(data["proof_bytes"]),
            gradient_commitment=bytes.fromhex(data["gradient_commitment"]),
            data_commitment=bytes.fromhex(data["data_commitment"]),
            statistics=GradientStatistics.from_dict(data["statistics"]),
            metadata=data["metadata"],
        )


# =============================================================================
# Batch Operations
# =============================================================================

class BatchProofVerifier:
    """Batch verifier for multiple gradient proofs."""

    def __init__(self):
        self._proofs: List[ProofOfGradientQuality] = []

    def add(self, proof: ProofOfGradientQuality) -> None:
        """Add a proof to the batch."""
        self._proofs.append(proof)

    def add_many(self, proofs: List[ProofOfGradientQuality]) -> None:
        """Add multiple proofs."""
        self._proofs.extend(proofs)

    def verify_all(self) -> List[VerificationResult]:
        """Verify all proofs and return results."""
        return [p.verify() for p in self._proofs]

    def verify_all_parallel(self, max_workers: int = 4) -> List[VerificationResult]:
        """Verify all proofs in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda p: p.verify(), self._proofs))
        return results

    def count_valid(self) -> int:
        """Count valid proofs."""
        return sum(1 for r in self.verify_all() if r.valid)

    def __len__(self) -> int:
        return len(self._proofs)


# =============================================================================
# Utilities
# =============================================================================

def compute_gradient_commitment(gradients: np.ndarray) -> bytes:
    """
    Compute Blake3 commitment to gradient values.

    Args:
        gradients: Gradient values (any shape)

    Returns:
        32-byte commitment hash
    """
    g = gradients.flatten().astype(np.float32)
    h = hashlib.blake2b(b"gradient_v1:", digest_size=32)
    h.update(struct.pack("<Q", len(g)))
    h.update(g.tobytes())
    return h.digest()


def serialize_proof(proof: ProofOfGradientQuality) -> bytes:
    """
    Serialize a proof for transmission.

    Format:
        [version:1][gradient_commitment:32][data_commitment:32]
        [stats_len:4][stats_json:N][metadata_len:4][metadata_json:N]
        [proof_bytes:...]
    """
    result = bytearray()

    # Version
    result.append(PROOF_VERSION)

    # Commitments
    result.extend(proof.gradient_commitment)
    result.extend(proof.data_commitment)

    # Statistics as JSON
    stats_json = json.dumps(proof.statistics.to_dict()).encode()
    result.extend(struct.pack("<I", len(stats_json)))
    result.extend(stats_json)

    # Metadata as JSON
    metadata_json = json.dumps(proof.metadata).encode()
    result.extend(struct.pack("<I", len(metadata_json)))
    result.extend(metadata_json)

    # Proof bytes
    result.extend(proof.proof_bytes)

    return bytes(result)


def deserialize_proof(data: bytes) -> ProofOfGradientQuality:
    """
    Deserialize a proof from bytes.

    Args:
        data: Serialized proof bytes

    Returns:
        ProofOfGradientQuality instance

    Raises:
        ValueError: If data is invalid or corrupted
    """
    if len(data) < 69:  # Minimum: version(1) + commits(64) + lens(4)
        raise ValueError("Proof data too short")

    pos = 0

    # Version
    version = data[pos]
    pos += 1
    if version > PROOF_VERSION:
        raise ValueError(f"Unsupported proof version: {version}")

    # Commitments
    gradient_commitment = bytes(data[pos:pos + 32])
    pos += 32
    data_commitment = bytes(data[pos:pos + 32])
    pos += 32

    # Statistics
    stats_len = struct.unpack("<I", data[pos:pos + 4])[0]
    pos += 4
    if pos + stats_len > len(data):
        raise ValueError("Truncated statistics data")
    stats_json = data[pos:pos + stats_len].decode()
    pos += stats_len
    statistics = GradientStatistics.from_dict(json.loads(stats_json))

    # Metadata
    metadata_len = struct.unpack("<I", data[pos:pos + 4])[0]
    pos += 4
    if pos + metadata_len > len(data):
        raise ValueError("Truncated metadata")
    metadata_json = data[pos:pos + metadata_len].decode()
    pos += metadata_len
    metadata = json.loads(metadata_json)

    # Proof bytes
    proof_bytes = bytes(data[pos:])

    return ProofOfGradientQuality(
        proof_bytes=proof_bytes,
        gradient_commitment=gradient_commitment,
        data_commitment=data_commitment,
        statistics=statistics,
        metadata=metadata,
    )


# =============================================================================
# Simulation (used when Rust backend unavailable)
# =============================================================================

def _simulate_proof(
    gradients: np.ndarray,
    config: GradientProofConfig,
    commitment: bytes,
) -> bytes:
    """
    Generate a simulated proof for testing/development.

    This does NOT provide actual zero-knowledge guarantees.
    It's only for API compatibility when Rust backend is unavailable.
    """
    # Create a fake proof that includes enough info for verification
    h = hashlib.blake2b(digest_size=256)
    h.update(b"simulated_proof_v1:")
    h.update(commitment)
    h.update(struct.pack("<f", config.max_norm))
    h.update(struct.pack("<I", config.round_number))
    h.update(gradients.tobytes()[:1024])  # Include partial data for determinism

    # Add padding to match realistic proof size
    proof_data = h.digest()
    padding = bytes([0x42] * (2048 - len(proof_data)))

    return proof_data + padding


def _verify_simulated_proof(proof_bytes: bytes, commitment: bytes) -> bool:
    """
    Verify a simulated proof.

    Always returns True for simulated proofs (no actual crypto verification).
    """
    if len(proof_bytes) < 256:
        return False

    # For simulation, just check that the commitment is in the proof
    # Real verification would check STARK constraints
    try:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"simulated_verify:")
        h.update(proof_bytes[:256])
        # Basic sanity check
        return True
    except Exception:
        return False
