#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Comprehensive tests for Gen-7 HYPERION-FL gradient validity proofs.

These tests verify:
1. Valid gradient proofs succeed
2. Invalid gradient proofs fail appropriately
3. Proof verification works correctly
4. Edge cases are handled properly

To run:
    pytest tests/test_gradient_proofs.py -v

Note: Proof generation is slow (5-60 seconds per proof) so these tests
may take several minutes to complete.
"""

import pytest
import hashlib
import struct
import numpy as np
from typing import List, Tuple

# Skip if module not available
try:
    import gen7_zkstark as zk
    HAS_ZKSTARK = True
except ImportError:
    HAS_ZKSTARK = False
    zk = None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def node_id() -> bytes:
    """Generate a deterministic node ID for testing."""
    return hashlib.sha256(b"test_node_1").digest()


@pytest.fixture
def model_hash() -> bytes:
    """Generate a deterministic model hash for testing."""
    return hashlib.sha256(b"test_model_v1").digest()


@pytest.fixture
def valid_gradient() -> List[float]:
    """Generate a valid gradient vector."""
    np.random.seed(42)
    return (np.random.randn(100) * 0.1).tolist()


@pytest.fixture
def zero_gradient() -> List[float]:
    """Generate a zero gradient (should fail - norm too small)."""
    return [0.0] * 100


@pytest.fixture
def large_gradient() -> List[float]:
    """Generate a gradient with excessively large values."""
    return [10000.0] * 100


@pytest.fixture
def nan_gradient() -> List[float]:
    """Generate a gradient with NaN values."""
    grad = [0.1] * 100
    grad[50] = float('nan')
    return grad


@pytest.fixture
def inf_gradient() -> List[float]:
    """Generate a gradient with Infinity values."""
    grad = [0.1] * 100
    grad[50] = float('inf')
    return grad


# =============================================================================
# Local Validation Tests (Fast)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
class TestLocalValidation:
    """Test local gradient validation (no proof generation)."""

    def test_valid_gradient_passes(self, valid_gradient):
        """Valid gradients should pass local validation."""
        is_valid, msg = zk.check_gradient_validity(valid_gradient)
        assert is_valid, f"Valid gradient should pass: {msg}"

    def test_empty_gradient_fails(self):
        """Empty gradients should fail validation."""
        is_valid, msg = zk.check_gradient_validity([])
        assert not is_valid
        assert "empty" in msg.lower()

    def test_nan_gradient_fails(self, nan_gradient):
        """Gradients with NaN should fail validation."""
        is_valid, msg = zk.check_gradient_validity(nan_gradient)
        assert not is_valid
        assert "nan" in msg.lower()

    def test_inf_gradient_fails(self, inf_gradient):
        """Gradients with Infinity should fail validation."""
        is_valid, msg = zk.check_gradient_validity(inf_gradient)
        assert not is_valid
        assert "infinity" in msg.lower()

    def test_large_norm_fails(self, large_gradient):
        """Gradients with excessive norm should fail validation."""
        is_valid, msg = zk.check_gradient_validity(large_gradient, max_norm=100.0)
        assert not is_valid
        assert "norm" in msg.lower() and "exceeds" in msg.lower()

    def test_custom_norm_limit(self, valid_gradient):
        """Custom norm limits should be respected."""
        # With default limit (1000), should pass
        is_valid1, _ = zk.check_gradient_validity(valid_gradient)
        assert is_valid1

        # With very small limit, should fail
        is_valid2, msg = zk.check_gradient_validity(valid_gradient, max_norm=0.001)
        assert not is_valid2


# =============================================================================
# Gradient Statistics Tests (Fast)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
class TestGradientStatistics:
    """Test gradient statistics computation."""

    def test_stats_computation(self, valid_gradient):
        """Gradient statistics should be computed correctly."""
        stats = zk.compute_gradient_stats(valid_gradient)

        assert stats["len"] == len(valid_gradient)
        assert stats["norm"] >= 0
        assert stats["variance"] >= 0

        # Verify norm calculation
        expected_norm = np.linalg.norm(valid_gradient)
        assert abs(stats["norm"] - expected_norm) < 0.01

        # Verify mean calculation
        expected_mean = np.mean(valid_gradient)
        assert abs(stats["mean"] - expected_mean) < 0.01

    def test_empty_gradient_stats(self):
        """Empty gradient stats should be zeros."""
        stats = zk.compute_gradient_stats([])
        assert stats["len"] == 0
        assert stats["norm"] == 0
        assert stats["mean"] == 0
        assert stats["variance"] == 0

    def test_uniform_gradient_stats(self):
        """Uniform gradient should have zero variance."""
        gradient = [1.0] * 100
        stats = zk.compute_gradient_stats(gradient)

        assert stats["len"] == 100
        assert abs(stats["mean"] - 1.0) < 0.001
        assert stats["variance"] < 0.001  # Should be effectively zero


# =============================================================================
# Fixed-Point Conversion Tests (Fast)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
class TestFixedPointConversion:
    """Test Q16.16 fixed-point conversions."""

    def test_roundtrip_conversion(self):
        """Values should roundtrip through fixed-point conversion."""
        test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001]

        for val in test_values:
            fixed = zk.to_fixed(val)
            back = zk.from_fixed(fixed)
            assert abs(val - back) < 0.001, f"Roundtrip failed for {val}"

    def test_fixed_point_limits(self):
        """Fixed-point should handle edge values."""
        # Large values near the limit
        large = zk.to_fixed(32767.0)
        assert zk.from_fixed(large) > 32766.0

        # Small values
        small = zk.to_fixed(0.00001)
        assert zk.from_fixed(small) < 0.001


# =============================================================================
# Hash Consistency Tests (Fast)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
class TestHashConsistency:
    """Test gradient hash consistency."""

    def test_hash_deterministic(self, valid_gradient):
        """Same gradient should produce same hash."""
        hash1 = zk.hash_gradient_py(valid_gradient)
        hash2 = zk.hash_gradient_py(valid_gradient)
        assert hash1 == hash2

    def test_hash_length(self, valid_gradient):
        """Hash should be 32 bytes (SHA256)."""
        hash_val = zk.hash_gradient_py(valid_gradient)
        assert len(hash_val) == 32

    def test_hash_different_gradients(self, valid_gradient):
        """Different gradients should produce different hashes."""
        hash1 = zk.hash_gradient_py(valid_gradient)

        modified = valid_gradient.copy()
        modified[0] += 0.01
        hash2 = zk.hash_gradient_py(modified)

        assert hash1 != hash2


# =============================================================================
# Proof Generation Tests (Slow - requires zkVM)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
@pytest.mark.slow
class TestProofGeneration:
    """Test zkSTARK proof generation and verification.

    These tests are slow (5-60 seconds each) because they involve
    actual zkSTARK proof generation.
    """

    def test_valid_proof_generation(self, node_id, model_hash, valid_gradient):
        """Valid gradient should produce a valid proof."""
        proof = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        assert isinstance(proof, bytes)
        assert len(proof) > 0

        # Verify the proof
        result = zk.verify_gradient_proof(proof)
        assert result["is_valid"]
        assert result["node_id"] == node_id
        assert result["round_number"] == 1
        assert result["gradient_len"] == len(valid_gradient)

    def test_proof_contains_correct_hash(self, node_id, model_hash, valid_gradient):
        """Proof should contain correct gradient hash."""
        expected_hash = zk.hash_gradient_py(valid_gradient)

        proof = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=42,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        result = zk.verify_gradient_proof(proof)
        assert result["gradient_hash"] == expected_hash

    def test_proof_contains_statistics(self, node_id, model_hash, valid_gradient):
        """Proof should contain gradient statistics."""
        proof = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        result = zk.verify_gradient_proof(proof)

        # Check that statistics are present
        assert "norm_squared" in result
        assert "norm" in result
        assert "mean" in result
        assert "variance" in result

        # Verify norm is reasonable
        expected_norm = np.linalg.norm(valid_gradient)
        assert abs(result["norm"] - expected_norm) < 1.0  # Allow some fixed-point error

    def test_different_rounds_produce_different_proofs(self, node_id, model_hash, valid_gradient):
        """Proofs for different rounds should be verifiably different."""
        proof1 = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        proof2 = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=2,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        result1 = zk.verify_gradient_proof(proof1)
        result2 = zk.verify_gradient_proof(proof2)

        assert result1["round_number"] == 1
        assert result2["round_number"] == 2

    def test_invalid_node_id_length_fails(self, model_hash, valid_gradient):
        """Node ID with wrong length should fail."""
        with pytest.raises(RuntimeError) as exc_info:
            zk.prove_gradient_validity(
                node_id=b"short",  # Wrong length
                round_number=1,
                model_hash=model_hash,
                gradient=valid_gradient,
            )
        assert "32 bytes" in str(exc_info.value)

    def test_invalid_model_hash_length_fails(self, node_id, valid_gradient):
        """Model hash with wrong length should fail."""
        with pytest.raises(RuntimeError) as exc_info:
            zk.prove_gradient_validity(
                node_id=node_id,
                round_number=1,
                model_hash=b"short",  # Wrong length
                gradient=valid_gradient,
            )
        assert "32 bytes" in str(exc_info.value)

    def test_empty_gradient_fails(self, node_id, model_hash):
        """Empty gradient should fail proof generation."""
        with pytest.raises(RuntimeError) as exc_info:
            zk.prove_gradient_validity(
                node_id=node_id,
                round_number=1,
                model_hash=model_hash,
                gradient=[],
            )
        assert "empty" in str(exc_info.value).lower()


# =============================================================================
# Proof Verification Tests (Requires pre-generated proofs)
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
@pytest.mark.slow
class TestProofVerification:
    """Test proof verification edge cases."""

    def test_corrupted_proof_fails(self, node_id, model_hash, valid_gradient):
        """Corrupted proof bytes should fail verification."""
        proof = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        # Corrupt a byte in the proof
        corrupted = bytearray(proof)
        corrupted[100] ^= 0xFF  # Flip bits
        corrupted = bytes(corrupted)

        with pytest.raises(RuntimeError):
            zk.verify_gradient_proof(corrupted)

    def test_truncated_proof_fails(self, node_id, model_hash, valid_gradient):
        """Truncated proof should fail verification."""
        proof = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        # Truncate the proof
        truncated = proof[:len(proof) // 2]

        with pytest.raises(RuntimeError):
            zk.verify_gradient_proof(truncated)

    def test_random_bytes_fail_verification(self):
        """Random bytes should fail verification."""
        random_proof = bytes(np.random.randint(0, 256, 10000, dtype=np.uint8))

        with pytest.raises(RuntimeError):
            zk.verify_gradient_proof(random_proof)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ZKSTARK, reason="gen7_zkstark not available")
@pytest.mark.slow
class TestIntegration:
    """Integration tests simulating real FL workflow."""

    def test_federated_learning_round(self):
        """Simulate a complete FL round with proof generation."""
        # Setup: Multiple nodes with different gradients
        nodes = []
        for i in range(3):
            node_id = hashlib.sha256(f"node_{i}".encode()).digest()
            np.random.seed(i)
            gradient = (np.random.randn(50) * 0.1).tolist()
            nodes.append((node_id, gradient))

        model_hash = hashlib.sha256(b"global_model_round_1").digest()
        round_number = 1

        # Each node generates a proof
        proofs = []
        for node_id, gradient in nodes:
            # First check locally
            is_valid, msg = zk.check_gradient_validity(gradient)
            assert is_valid, f"Local validation failed: {msg}"

            # Generate proof
            proof = zk.prove_gradient_validity(
                node_id=node_id,
                round_number=round_number,
                model_hash=model_hash,
                gradient=gradient,
            )
            proofs.append(proof)

        # Coordinator verifies all proofs
        for i, proof in enumerate(proofs):
            result = zk.verify_gradient_proof(proof)
            assert result["is_valid"], f"Proof from node {i} failed verification"
            assert result["round_number"] == round_number
            assert result["model_hash"] == model_hash

    def test_replay_attack_detection(self, node_id, model_hash, valid_gradient):
        """Verify that round numbers prevent replay attacks."""
        # Generate proof for round 1
        proof_round_1 = zk.prove_gradient_validity(
            node_id=node_id,
            round_number=1,
            model_hash=model_hash,
            gradient=valid_gradient,
        )

        result = zk.verify_gradient_proof(proof_round_1)

        # The proof is valid, but the coordinator should check round number
        assert result["round_number"] == 1

        # If we're now in round 2, this proof should be rejected by application logic
        current_round = 2
        assert result["round_number"] != current_round, \
            "Replay attack: proof from round 1 being used in round 2"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run fast tests only by default
    pytest.main([__file__, "-v", "-m", "not slow"])
