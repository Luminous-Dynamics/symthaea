# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Rust Bridge (zerotrustml-core integration)

Tests cover:
- Rust backend detection
- RustHypervectorEncoder
- RustUnifiedDetector
- RustPhiMeasurer
- RustShapleyComputer
- Python fallback behavior
"""

import pytest
import numpy as np
from typing import Dict

from mycelix_fl.rust_bridge import (
    RUST_AVAILABLE,
    RUST_IMPORT_ERROR,
    get_version,
    get_dimension,
    get_rust_status,
    get_encoder,
    get_detector,
    get_phi_measurer,
    get_shapley_computer,
    detect_byzantine,
    RustHypervectorEncoder,
    RustUnifiedDetector,
    RustPhiMeasurer,
    RustShapleyComputer,
)


# =============================================================================
# RUST AVAILABILITY TESTS
# =============================================================================

class TestRustAvailability:
    """Test Rust backend availability detection."""

    def test_rust_available_boolean(self):
        """Test RUST_AVAILABLE is a boolean."""
        assert isinstance(RUST_AVAILABLE, bool)

    def test_get_version(self):
        """Test version retrieval."""
        version = get_version()
        assert isinstance(version, str)

        if RUST_AVAILABLE:
            # Should be a version string like "0.1.0"
            assert version != "python-only"
        else:
            assert version == "python-only"

    def test_get_dimension(self):
        """Test dimension retrieval."""
        dim = get_dimension()
        assert isinstance(dim, int)
        assert dim >= 1024

        if RUST_AVAILABLE:
            assert dim == 16384  # Rust default
        else:
            assert dim == 8192  # Python default

    def test_get_rust_status(self):
        """Test detailed status retrieval."""
        status = get_rust_status()

        assert isinstance(status, dict)
        assert "available" in status
        assert "version" in status
        assert "hv_dimension" in status
        assert "import_error" in status
        assert "features" in status

        assert status["available"] == RUST_AVAILABLE

        if RUST_AVAILABLE:
            assert len(status["features"]) > 0
            assert "hypervector_encoder" in status["features"]
        else:
            assert len(status["features"]) == 0

    def test_import_error_message(self):
        """Test import error message when not available."""
        if not RUST_AVAILABLE:
            assert RUST_IMPORT_ERROR is not None
            assert isinstance(RUST_IMPORT_ERROR, str)
        else:
            assert RUST_IMPORT_ERROR is None


# =============================================================================
# RUST HYPERVECTOR ENCODER TESTS
# =============================================================================

class TestRustHypervectorEncoder:
    """Test RustHypervectorEncoder wrapper."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = get_encoder()
        assert isinstance(encoder, RustHypervectorEncoder)

    def test_initialization_custom_dimension(self):
        """Test initialization with custom dimension."""
        encoder = get_encoder(dimension=8192)
        assert encoder.dimension == 8192

    def test_is_rust_property(self):
        """Test is_rust property."""
        encoder = get_encoder()
        assert encoder.is_rust == RUST_AVAILABLE

    def test_encode_single(self, small_gradient):
        """Test encoding single gradient."""
        encoder = get_encoder()

        result = encoder.encode(small_gradient)
        assert result is not None

    def test_encode_batch(self, rng):
        """Test batch encoding."""
        encoder = get_encoder()

        gradients = [
            rng.randn(1000).astype(np.float32)
            for _ in range(5)
        ]

        results = encoder.encode_batch(gradients)
        assert len(results) == 5

    def test_encode_deterministic(self, small_gradient):
        """Test that encoding is deterministic."""
        encoder = get_encoder()

        result1 = encoder.encode(small_gradient)
        result2 = encoder.encode(small_gradient)

        # Results should be identical
        if hasattr(result1, 'hypervector'):
            assert result1.hypervector == result2.hypervector


# =============================================================================
# RUST UNIFIED DETECTOR TESTS
# =============================================================================

class TestRustUnifiedDetector:
    """Test RustUnifiedDetector wrapper."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = get_detector()
        assert isinstance(detector, RustUnifiedDetector)

    def test_is_rust_property(self):
        """Test is_rust property."""
        detector = get_detector()
        assert detector.is_rust == RUST_AVAILABLE

    def test_detect_honest(self, honest_gradients):
        """Test detection with all honest nodes."""
        detector = get_detector()

        result = detector.detect(honest_gradients, round_number=1)

        assert result is not None
        assert hasattr(result, "byzantine_nodes")
        assert hasattr(result, "aggregation_weights")
        assert hasattr(result, "confidence")

        # All honest should have no/few Byzantine
        assert len(result.byzantine_nodes) <= 1

    def test_detect_with_byzantine(self, byzantine_gradients):
        """Test detection with Byzantine nodes."""
        detector = get_detector()

        result = detector.detect(byzantine_gradients, round_number=1)

        assert result is not None
        # Should detect at least some Byzantine
        assert len(result.byzantine_nodes) > 0 or result.confidence > 0

    def test_aggregation_weights(self, honest_gradients):
        """Test aggregation weights are computed."""
        detector = get_detector()

        result = detector.detect(honest_gradients, round_number=1)

        assert len(result.aggregation_weights) > 0

        # Weights should be non-negative
        for weight in result.aggregation_weights.values():
            assert weight >= 0

    def test_confidence_score(self, honest_gradients):
        """Test confidence score range."""
        detector = get_detector()

        result = detector.detect(honest_gradients, round_number=1)

        assert 0 <= result.confidence <= 1


# =============================================================================
# RUST PHI MEASURER TESTS
# =============================================================================

class TestRustPhiMeasurer:
    """Test RustPhiMeasurer wrapper."""

    def test_initialization(self):
        """Test measurer initialization."""
        measurer = get_phi_measurer()
        assert isinstance(measurer, RustPhiMeasurer)

    def test_is_rust_property(self):
        """Test is_rust property."""
        measurer = get_phi_measurer()
        assert measurer.is_rust == RUST_AVAILABLE

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_measure_phi(self):
        """Test Φ measurement (Rust only)."""
        measurer = get_phi_measurer()
        encoder = get_encoder()

        # Encode some gradients
        gradients = [
            np.random.randn(1000).astype(np.float32)
            for _ in range(5)
        ]
        hypervectors = [encoder.encode(g) for g in gradients]

        result = measurer.measure_phi(hypervectors)

        assert hasattr(result, "phi_value")
        assert result.phi_value >= 0

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_detect_byzantine_phi(self):
        """Test Byzantine detection via Φ (Rust only)."""
        measurer = get_phi_measurer()
        encoder = get_encoder()

        gradients = [
            np.random.randn(1000).astype(np.float32)
            for _ in range(10)
        ]
        hypervectors = [encoder.encode(g) for g in gradients]

        byzantine_indices = measurer.detect_byzantine(hypervectors)

        assert isinstance(byzantine_indices, list)


# =============================================================================
# RUST SHAPLEY COMPUTER TESTS
# =============================================================================

class TestRustShapleyComputer:
    """Test RustShapleyComputer wrapper."""

    def test_initialization(self):
        """Test computer initialization."""
        computer = get_shapley_computer()
        assert isinstance(computer, RustShapleyComputer)

    def test_is_rust_property(self):
        """Test is_rust property."""
        computer = get_shapley_computer()
        assert computer.is_rust == RUST_AVAILABLE

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_compute_exact(self):
        """Test exact Shapley computation (Rust only)."""
        computer = get_shapley_computer()
        encoder = get_encoder()

        gradients = [
            np.random.randn(1000).astype(np.float32)
            for _ in range(5)
        ]
        hypervectors = [encoder.encode(g) for g in gradients]

        results = computer.compute_exact(hypervectors)

        assert len(results) == 5
        for r in results:
            assert hasattr(r, "shapley_value")
            assert hasattr(r, "normalized_contribution")

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_aggregation_weights(self):
        """Test aggregation weights from Shapley (Rust only)."""
        computer = get_shapley_computer()
        encoder = get_encoder()

        gradients = [
            np.random.randn(1000).astype(np.float32)
            for _ in range(5)
        ]
        hypervectors = [encoder.encode(g) for g in gradients]

        weights = computer.aggregation_weights(hypervectors)

        assert len(weights) == 5
        assert all(w >= 0 for w in weights)
        assert sum(weights) == pytest.approx(1.0, rel=0.1)


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestDetectByzantineUtility:
    """Test detect_byzantine utility function."""

    def test_basic_usage(self, honest_gradients):
        """Test basic usage of utility function."""
        byzantine, weights, confidence = detect_byzantine(
            honest_gradients,
            round_number=1,
        )

        assert isinstance(byzantine, set)
        assert isinstance(weights, dict)
        assert isinstance(confidence, float)

    def test_with_byzantine(self, byzantine_gradients):
        """Test with Byzantine gradients."""
        byzantine, weights, confidence = detect_byzantine(
            byzantine_gradients,
            round_number=1,
        )

        # Should detect at least some
        assert len(byzantine) > 0 or confidence > 0

    def test_use_adaptive_false(self, honest_gradients):
        """Test with adaptive disabled."""
        byzantine, weights, confidence = detect_byzantine(
            honest_gradients,
            round_number=1,
            use_adaptive=False,
        )

        assert isinstance(byzantine, set)


# =============================================================================
# PYTHON FALLBACK TESTS
# =============================================================================

class TestPythonFallback:
    """Test Python fallback behavior when Rust not available."""

    def test_encoder_fallback(self, small_gradient):
        """Test encoder falls back to Python."""
        encoder = get_encoder()

        # Should work regardless of Rust availability
        result = encoder.encode(small_gradient)
        assert result is not None

    def test_detector_fallback(self, honest_gradients):
        """Test detector falls back to Python."""
        detector = get_detector()

        # Should work regardless of Rust availability
        result = detector.detect(honest_gradients, round_number=1)
        assert result is not None

    def test_detect_byzantine_fallback(self, honest_gradients):
        """Test detect_byzantine falls back to Python."""
        # Should work regardless of Rust availability
        byzantine, weights, confidence = detect_byzantine(
            honest_gradients,
            round_number=1,
        )
        assert weights is not None


# =============================================================================
# PERFORMANCE COMPARISON TESTS
# =============================================================================

class TestRustPerformance:
    """Performance comparison tests."""

    @pytest.mark.slow
    @pytest.mark.rust
    @pytest.mark.benchmark
    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_rust_speedup_detection(self, byzantine_gradients, timing_stats):
        """Benchmark Rust detection vs Python."""
        rust_detector = get_detector()

        import time

        # Rust timing
        for _ in range(20):
            start = time.perf_counter()
            rust_detector.detect(byzantine_gradients, round_number=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        rust_mean = timing_stats.mean
        print(f"\n  Rust Detection: {timing_stats.summary()}")

        # Rust should be fast
        assert rust_mean < 100  # < 100ms

    @pytest.mark.slow
    @pytest.mark.rust
    @pytest.mark.benchmark
    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust backend")
    def test_rust_speedup_encoding(self, medium_gradient, timing_stats):
        """Benchmark Rust encoding."""
        encoder = get_encoder()

        import time

        for _ in range(100):
            start = time.perf_counter()
            encoder.encode(medium_gradient)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        print(f"\n  Rust Encoding (100K): {timing_stats.summary()}")

        # Should be very fast
        assert timing_stats.mean < 10  # < 10ms


# =============================================================================
# EDGE CASES
# =============================================================================

class TestRustBridgeEdgeCases:
    """Test edge cases for Rust bridge."""

    def test_empty_gradients(self):
        """Test with empty gradient dict."""
        detector = get_detector()

        result = detector.detect({}, round_number=1)
        assert result is not None
        assert len(result.byzantine_nodes) == 0

    def test_single_node(self, rng):
        """Test with single node."""
        detector = get_detector()

        gradients = {"only": rng.randn(1000).astype(np.float32)}

        result = detector.detect(gradients, round_number=1)
        assert result is not None

    def test_large_gradients(self, rng):
        """Test with large gradients."""
        encoder = get_encoder()

        large_gradient = rng.randn(1_000_000).astype(np.float32)

        result = encoder.encode(large_gradient)
        assert result is not None
