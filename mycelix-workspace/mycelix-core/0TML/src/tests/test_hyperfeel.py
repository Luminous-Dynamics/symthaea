# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for HyperFeel Encoder v2.0

Tests cover:
- EncodingConfig validation
- HyperGradient creation and serialization
- HyperFeelEncoderV2 encoding
- Compression ratio validation
- Similarity preservation
"""

import pytest
import numpy as np
from typing import Dict

from mycelix_fl.hyperfeel import (
    HyperFeelEncoderV2,
    HyperGradient,
    EncodingConfig,
)
from mycelix_fl.hyperfeel.encoder_v2 import (
    encode_gradient,
    compute_similarity_matrix,
    HV16_DIMENSION,
    HV16_BYTES,
)


# =============================================================================
# ENCODING CONFIG TESTS
# =============================================================================

class TestEncodingConfig:
    """Test EncodingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = EncodingConfig()
        assert config.dimension == HV16_DIMENSION
        assert config.quantize_bits == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = EncodingConfig(
            dimension=8192,
            quantize_bits=4,
        )
        assert config.dimension == 8192
        assert config.quantize_bits == 4

    def test_validation_dimension_minimum(self):
        """Test dimension minimum validation."""
        with pytest.raises(ValueError, match="dimension"):
            EncodingConfig(dimension=512)  # Too small

    def test_validation_dimension_power_of_two(self):
        """Test dimension must be power of 2."""
        with pytest.raises(ValueError, match="power of 2"):
            EncodingConfig(dimension=10000)  # Not power of 2

    def test_validation_quantize_bits(self):
        """Test quantize_bits validation."""
        with pytest.raises(ValueError, match="quantize_bits"):
            EncodingConfig(quantize_bits=0)
        with pytest.raises(ValueError, match="quantize_bits"):
            EncodingConfig(quantize_bits=17)

    def test_valid_dimensions(self):
        """Test valid dimension values."""
        for dim in [1024, 2048, 4096, 8192, 16384]:
            config = EncodingConfig(dimension=dim)
            assert config.dimension == dim


# =============================================================================
# HYPER GRADIENT TESTS
# =============================================================================

class TestHyperGradient:
    """Test HyperGradient dataclass."""

    def test_creation(self):
        """Test basic creation."""
        hg = HyperGradient(
            node_id="node_1",
            round_num=5,
            nonce=b"\x00" * 32,
            hypervector=b"\x00" * HV16_BYTES,
            phi_before=0.5,
            phi_after=0.6,
            epistemic_confidence=0.8,
            quality_score=0.9,
            pogq_proof_hash=b"\x00" * 32,
            timestamp=1234567890,  # int, not float
            original_size=100000,
            compression_ratio=50.0,
            metadata={},
        )

        assert hg.node_id == "node_1"
        assert hg.round_num == 5
        assert hg.phi_gain == pytest.approx(0.1)

    def test_phi_gain_property(self):
        """Test phi_gain computed property."""
        hg = HyperGradient(
            node_id="test",
            round_num=1,
            nonce=b"\x00" * 32,
            hypervector=b"\x00" * HV16_BYTES,
            phi_before=0.3,
            phi_after=0.5,
            epistemic_confidence=0.7,
            quality_score=0.8,
            pogq_proof_hash=b"\x00" * 32,
            timestamp=0,  # int, not float
            original_size=1000,
            compression_ratio=1.0,
            metadata={},
        )

        assert hg.phi_gain == pytest.approx(0.2)

    def test_is_beneficial_property(self):
        """Test is_beneficial computed property."""
        # Beneficial: positive phi gain + high confidence
        beneficial = HyperGradient(
            node_id="good",
            round_num=1,
            nonce=b"\x00" * 32,
            hypervector=b"\x00" * HV16_BYTES,
            phi_before=0.3,
            phi_after=0.5,
            epistemic_confidence=0.8,
            quality_score=0.9,
            pogq_proof_hash=b"\x00" * 32,
            timestamp=0,
            original_size=1000,
            compression_ratio=1.0,
            metadata={},
        )
        assert beneficial.is_beneficial

        # Not beneficial: negative phi gain
        harmful = HyperGradient(
            node_id="bad",
            round_num=1,
            nonce=b"\x00" * 32,
            hypervector=b"\x00" * HV16_BYTES,
            phi_before=0.5,
            phi_after=0.3,
            epistemic_confidence=0.8,
            quality_score=0.9,
            pogq_proof_hash=b"\x00" * 32,
            timestamp=0,
            original_size=1000,
            compression_ratio=1.0,
            metadata={},
        )
        assert not harmful.is_beneficial

    def test_to_dict(self):
        """Test serialization to dict."""
        hg = HyperGradient(
            node_id="serialize_test",
            round_num=3,
            nonce=b"\xab" * 32,
            hypervector=b"\xcd" * HV16_BYTES,
            phi_before=0.4,
            phi_after=0.5,
            epistemic_confidence=0.75,
            quality_score=0.85,
            pogq_proof_hash=b"\xef" * 32,
            timestamp=1000,
            original_size=50000,
            compression_ratio=25.0,
            metadata={"key": "value"},
        )

        d = hg.to_dict()

        assert d["node_id"] == "serialize_test"
        assert d["round_num"] == 3
        assert d["nonce"] == "ab" * 32
        assert d["phi_gain"] == pytest.approx(0.1)
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "node_id": "restore_test",
            "round_num": 7,
            "nonce": "00" * 32,
            "hypervector": "ff" * HV16_BYTES,
            "phi_before": 0.2,
            "phi_after": 0.4,
            "epistemic_confidence": 0.9,
            "quality_score": 0.95,
            "pogq_proof_hash": "11" * 32,
            "timestamp": 2000,
            "original_size": 80000,
            "compression_ratio": 40.0,
            "metadata": {"restored": True},
        }

        hg = HyperGradient.from_dict(d)

        assert hg.node_id == "restore_test"
        assert hg.round_num == 7
        assert hg.phi_gain == pytest.approx(0.2)
        assert hg.metadata["restored"] is True

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict preserves data."""
        original = HyperGradient(
            node_id="roundtrip",
            round_num=10,
            nonce=b"\x12" * 32,
            hypervector=b"\x34" * HV16_BYTES,
            phi_before=0.33,
            phi_after=0.66,
            epistemic_confidence=0.88,
            quality_score=0.77,
            pogq_proof_hash=b"\x56" * 32,
            timestamp=3000,
            original_size=200000,
            compression_ratio=100.0,
            metadata={"test": 123},
        )

        d = original.to_dict()
        restored = HyperGradient.from_dict(d)

        assert restored.node_id == original.node_id
        assert restored.round_num == original.round_num
        assert restored.phi_before == original.phi_before
        assert restored.phi_after == original.phi_after
        assert restored.nonce == original.nonce
        assert restored.hypervector == original.hypervector


# =============================================================================
# HYPERFEEL ENCODER TESTS
# =============================================================================

class TestHyperFeelEncoderV2:
    """Test HyperFeelEncoderV2 class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = HyperFeelEncoderV2()
        assert encoder.dimension == HV16_DIMENSION

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = EncodingConfig(dimension=8192)
        encoder = HyperFeelEncoderV2(config=config)
        assert encoder.dimension == 8192

    def test_encode_gradient_basic(self, small_gradient):
        """Test basic gradient encoding."""
        encoder = HyperFeelEncoderV2()

        hg = encoder.encode_gradient(
            gradient=small_gradient,
            round_num=1,
            node_id="test_node",
        )

        assert isinstance(hg, HyperGradient)
        assert hg.node_id == "test_node"
        assert hg.round_num == 1
        assert len(hg.hypervector) == HV16_BYTES
        assert len(hg.nonce) == 32

    def test_encode_gradient_compression(self, medium_gradient):
        """Test compression ratio."""
        encoder = HyperFeelEncoderV2()

        hg = encoder.encode_gradient(
            gradient=medium_gradient,
            round_num=1,
            node_id="compress_test",
        )

        # Original: 100K * 4 bytes = 400KB
        # Compressed: 2KB
        expected_ratio = len(medium_gradient) * 4 / HV16_BYTES
        assert hg.compression_ratio == pytest.approx(expected_ratio, rel=0.1)
        assert hg.compression_ratio > 100  # Should be > 100x compression

    def test_encode_different_sizes(self, rng):
        """Test encoding gradients of different sizes."""
        encoder = HyperFeelEncoderV2()

        sizes = [100, 1000, 10000, 100000]

        for size in sizes:
            gradient = rng.randn(size).astype(np.float32)
            hg = encoder.encode_gradient(
                gradient=gradient,
                round_num=1,
                node_id=f"size_{size}",
            )

            # All should produce same-size hypervector
            assert len(hg.hypervector) == HV16_BYTES
            # original_size is in bytes (size * 4 for float32)
            assert hg.original_size == size * 4

    def test_encode_preserves_similarity(self, rng):
        """Test that similar gradients have similar hypervectors."""
        encoder = HyperFeelEncoderV2()

        # Create base gradient
        base = rng.randn(10000).astype(np.float32)
        base /= np.linalg.norm(base)

        # Create similar gradient (small noise)
        similar = base + rng.randn(10000).astype(np.float32) * 0.05

        # Create different gradient
        different = rng.randn(10000).astype(np.float32)
        different /= np.linalg.norm(different)

        hg_base = encoder.encode_gradient(base, 1, "base")
        hg_similar = encoder.encode_gradient(similar, 1, "similar")
        hg_different = encoder.encode_gradient(different, 1, "different")

        # Compute similarities
        sim_base_similar = HyperFeelEncoderV2.cosine_similarity(
            hg_base.hypervector, hg_similar.hypervector
        )
        sim_base_different = HyperFeelEncoderV2.cosine_similarity(
            hg_base.hypervector, hg_different.hypervector
        )

        print(f"\n  Similarity (base, similar): {sim_base_similar:.3f}")
        print(f"  Similarity (base, different): {sim_base_different:.3f}")

        # Similar should be more similar than different
        assert sim_base_similar > sim_base_different

    def test_encode_deterministic(self, small_gradient):
        """Test that encoding is deterministic."""
        encoder = HyperFeelEncoderV2(config=EncodingConfig(projection_seed=42))

        hg1 = encoder.encode_gradient(small_gradient, 1, "test")

        # Create new encoder with same seed
        encoder2 = HyperFeelEncoderV2(config=EncodingConfig(projection_seed=42))
        hg2 = encoder2.encode_gradient(small_gradient, 1, "test")

        # Hypervectors should be identical (nonce is random)
        assert hg1.hypervector == hg2.hypervector

    def test_encode_phi_metrics(self, small_gradient):
        """Test that Φ metrics are computed."""
        encoder = HyperFeelEncoderV2()

        hg = encoder.encode_gradient(
            gradient=small_gradient,
            round_num=1,
            node_id="phi_test",
        )

        assert hg.phi_before >= 0
        assert hg.phi_after >= 0
        assert hg.epistemic_confidence >= 0

    def test_encode_quality_score(self, small_gradient):
        """Test that quality score is computed."""
        encoder = HyperFeelEncoderV2()

        hg = encoder.encode_gradient(
            gradient=small_gradient,
            round_num=1,
            node_id="quality_test",
        )

        # quality_score is L2 norm, not normalized to [0,1]
        assert hg.quality_score >= 0


# =============================================================================
# COSINE SIMILARITY TESTS
# =============================================================================

class TestCosineSimilarity:
    """Test hypervector cosine similarity."""

    def test_identical_hypervectors(self):
        """Test similarity of identical hypervectors."""
        hv = b"\xff" * HV16_BYTES
        sim = HyperFeelEncoderV2.cosine_similarity(hv, hv)
        assert sim == pytest.approx(1.0)

    def test_opposite_hypervectors(self):
        """Test similarity of opposite hypervectors."""
        hv1 = b"\xff" * HV16_BYTES  # All 1s
        hv2 = b"\x00" * HV16_BYTES  # All 0s
        sim = HyperFeelEncoderV2.cosine_similarity(hv1, hv2)
        # Should be -1 (opposite)
        assert sim == pytest.approx(-1.0)

    def test_random_hypervectors(self, rng):
        """Test similarity of random hypervectors."""
        hv1 = bytes(rng.randint(0, 256, HV16_BYTES, dtype=np.uint8))
        hv2 = bytes(rng.randint(0, 256, HV16_BYTES, dtype=np.uint8))

        sim = HyperFeelEncoderV2.cosine_similarity(hv1, hv2)

        # Random should be close to 0
        assert -0.5 <= sim <= 0.5

    def test_similarity_symmetry(self, rng):
        """Test that similarity is symmetric."""
        hv1 = bytes(rng.randint(0, 256, HV16_BYTES, dtype=np.uint8))
        hv2 = bytes(rng.randint(0, 256, HV16_BYTES, dtype=np.uint8))

        sim_12 = HyperFeelEncoderV2.cosine_similarity(hv1, hv2)
        sim_21 = HyperFeelEncoderV2.cosine_similarity(hv2, hv1)

        assert sim_12 == pytest.approx(sim_21)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_encode_gradient_function(self, small_gradient):
        """Test encode_gradient convenience function."""
        hg = encode_gradient(
            gradient=small_gradient,
            round_num=1,
            node_id="convenience_test",
        )

        assert isinstance(hg, HyperGradient)
        assert hg.node_id == "convenience_test"

    def test_compute_similarity_matrix(self, rng):
        """Test similarity matrix computation."""
        encoder = HyperFeelEncoderV2()

        gradients = [
            rng.randn(5000).astype(np.float32)
            for _ in range(5)
        ]

        hypergradients = [
            encoder.encode_gradient(g, 1, f"node_{i}")
            for i, g in enumerate(gradients)
        ]

        matrix = compute_similarity_matrix(hypergradients)

        assert matrix.shape == (5, 5)
        # Diagonal should be 1
        for i in range(5):
            assert matrix[i, i] == pytest.approx(1.0)
        # Symmetric
        assert np.allclose(matrix, matrix.T)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestHyperFeelPerformance:
    """Performance tests for HyperFeel encoder."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_encoding_speed(self, medium_gradient, timing_stats):
        """Benchmark encoding speed."""
        encoder = HyperFeelEncoderV2()

        import time

        for _ in range(50):
            start = time.perf_counter()
            encoder.encode_gradient(medium_gradient, 1, "bench")
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        print(f"\n  Encoding (100K params): {timing_stats.summary()}")

        # Should be fast enough for real-time FL
        assert timing_stats.mean < 50  # < 50ms average

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_encoding_scalability(self, rng):
        """Test how encoding scales with gradient size."""
        encoder = HyperFeelEncoderV2()

        import time

        sizes = [10_000, 100_000, 1_000_000]

        for size in sizes:
            gradient = rng.randn(size).astype(np.float32)

            start = time.perf_counter()
            encoder.encode_gradient(gradient, 1, "scale_test")
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  {size:,} params: {elapsed_ms:.2f}ms")


# =============================================================================
# EDGE CASES
# =============================================================================

class TestHyperFeelEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_gradient(self):
        """Test encoding zero gradient."""
        encoder = HyperFeelEncoderV2()

        gradient = np.zeros(1000, dtype=np.float32)
        hg = encoder.encode_gradient(gradient, 1, "zero")

        assert hg is not None
        assert len(hg.hypervector) == HV16_BYTES

    def test_very_small_gradient(self, rng):
        """Test encoding very small gradient values."""
        encoder = HyperFeelEncoderV2()

        gradient = rng.randn(1000).astype(np.float32) * 1e-10
        hg = encoder.encode_gradient(gradient, 1, "tiny")

        assert hg is not None

    def test_very_large_gradient(self, rng):
        """Test encoding very large gradient values."""
        encoder = HyperFeelEncoderV2()

        gradient = rng.randn(1000).astype(np.float32) * 1e6
        hg = encoder.encode_gradient(gradient, 1, "huge")

        assert hg is not None

    def test_single_element_gradient(self):
        """Test encoding single-element gradient."""
        encoder = HyperFeelEncoderV2()

        gradient = np.array([1.0], dtype=np.float32)
        hg = encoder.encode_gradient(gradient, 1, "single")

        assert hg is not None
        assert len(hg.hypervector) == HV16_BYTES

    def test_nan_handling(self, rng):
        """Test handling of NaN values."""
        encoder = HyperFeelEncoderV2()

        gradient = rng.randn(1000).astype(np.float32)
        gradient[500] = np.nan

        # Should handle gracefully (might replace or fail gracefully)
        try:
            hg = encoder.encode_gradient(gradient, 1, "nan")
            assert hg is not None
        except ValueError:
            pass  # Acceptable to reject NaN

    def test_inf_handling(self, rng):
        """Test handling of infinite values."""
        encoder = HyperFeelEncoderV2()

        gradient = rng.randn(1000).astype(np.float32)
        gradient[0] = np.inf
        gradient[1] = -np.inf

        # Should handle gracefully
        try:
            hg = encoder.encode_gradient(gradient, 1, "inf")
            assert hg is not None
        except ValueError:
            pass  # Acceptable to reject inf
