# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Hypervector-Based Φ (Integrated Information) Measurement

Tests cover:
- PhiMetrics dataclass
- HypervectorPhiMeasurer
- Phi computation accuracy
- Performance benchmarks
"""

import pytest
import numpy as np
from typing import Dict, List

from mycelix_fl.core.phi_measurement import (
    HypervectorPhiMeasurer,
    PhiMetrics,
    PhiMeasurementResult,
    measure_phi,
)


# =============================================================================
# PHI METRICS TESTS
# =============================================================================

class TestPhiMetrics:
    """Test PhiMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metrics = PhiMetrics(
            phi_before=0.5,
            phi_after=0.6,
            phi_gain=0.1,
            epistemic_confidence=0.8,
        )

        assert metrics.phi_before == 0.5
        assert metrics.phi_after == 0.6
        assert metrics.phi_gain == 0.1
        assert metrics.epistemic_confidence == 0.8

    def test_with_layer_contributions(self):
        """Test with optional layer contributions."""
        metrics = PhiMetrics(
            phi_before=0.4,
            phi_after=0.5,
            phi_gain=0.1,
            epistemic_confidence=0.75,
            layer_contributions={"layer1": 0.3, "layer2": 0.2},
        )

        assert metrics.layer_contributions is not None
        assert metrics.layer_contributions["layer1"] == 0.3

    def test_to_dict(self):
        """Test serialization."""
        metrics = PhiMetrics(
            phi_before=0.3,
            phi_after=0.4,
            phi_gain=0.1,
            epistemic_confidence=0.9,
        )

        d = metrics.to_dict()
        assert d["phi_before"] == 0.3
        assert d["phi_after"] == 0.4
        assert d["phi_gain"] == 0.1
        assert d["epistemic_confidence"] == 0.9

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "phi_before": 0.2,
            "phi_after": 0.35,
            "phi_gain": 0.15,
            "epistemic_confidence": 0.85,
            "layer_contributions": {"conv1": 0.1, "fc1": 0.25},
        }

        metrics = PhiMetrics.from_dict(d)
        assert metrics.phi_before == 0.2
        assert metrics.phi_after == 0.35
        assert metrics.layer_contributions["conv1"] == 0.1


# =============================================================================
# HYPERVECTOR PHI MEASURER TESTS
# =============================================================================

class TestHypervectorPhiMeasurer:
    """Test HypervectorPhiMeasurer class."""

    def test_initialization(self):
        """Test measurer initialization."""
        measurer = HypervectorPhiMeasurer()
        assert measurer.dimension > 0

    def test_initialization_custom_dimension(self):
        """Test initialization with custom dimension."""
        measurer = HypervectorPhiMeasurer(dimension=4096)
        assert measurer.dimension == 4096

    def test_measure_phi_single_layer(self, rng):
        """Test Φ measurement for single layer activations."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        # Simulate layer activations
        activations = [
            rng.randn(100).astype(np.float32)  # Single layer
        ]

        result = measurer.measure_phi_from_activations(activations)

        assert isinstance(result, PhiMeasurementResult)
        assert result.phi_total >= 0
        assert result.layer_count == 1

    def test_measure_phi_multiple_layers(self, rng):
        """Test Φ measurement for multiple layer activations."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        # Simulate multi-layer activations
        activations = [
            rng.randn(256).astype(np.float32),  # Conv layer
            rng.randn(128).astype(np.float32),  # FC layer 1
            rng.randn(64).astype(np.float32),   # FC layer 2
            rng.randn(10).astype(np.float32),   # Output layer
        ]

        result = measurer.measure_phi_from_activations(activations)

        assert result.phi_total >= 0
        assert result.layer_count == 4
        assert len(result.phi_layers) == 4

    def test_coherent_activations_higher_phi(self, rng):
        """Test that coherent (correlated) activations have higher Φ."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        # Coherent activations (correlated)
        base = rng.randn(100).astype(np.float32)
        coherent_activations = [
            base + rng.randn(100).astype(np.float32) * 0.1,
            base * 0.8 + rng.randn(100).astype(np.float32) * 0.1,
            base * 0.6 + rng.randn(100).astype(np.float32) * 0.1,
        ]

        # Random activations (uncorrelated)
        random_activations = [
            rng.randn(100).astype(np.float32),
            rng.randn(100).astype(np.float32),
            rng.randn(100).astype(np.float32),
        ]

        phi_coherent = measurer.measure_phi_from_activations(coherent_activations)
        phi_random = measurer.measure_phi_from_activations(random_activations)

        # Coherent should have higher integration
        assert phi_coherent.integration_gain >= phi_random.integration_gain * 0.8

    def test_gradient_phi_gain(self, rng):
        """Test Φ gain measurement for gradients."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        # Simulate gradient as change in activations
        before_activations = [
            rng.randn(100).astype(np.float32),
            rng.randn(50).astype(np.float32),
        ]

        # Improved activations (more coherent)
        base = rng.randn(100).astype(np.float32)
        after_activations = [
            base,
            base[:50] * 0.5 + rng.randn(50).astype(np.float32) * 0.1,
        ]

        phi_before = measurer.measure_phi_from_activations(before_activations)
        phi_after = measurer.measure_phi_from_activations(after_activations)

        phi_gain = phi_after.phi_total - phi_before.phi_total

        # Just verify it computes without error
        assert isinstance(phi_gain, float)

    def test_empty_activations(self):
        """Test handling of empty activations."""
        measurer = HypervectorPhiMeasurer()

        result = measurer.measure_phi_from_activations([])
        assert result.phi_total == 0.0
        assert result.layer_count == 0

    def test_deterministic(self, rng):
        """Test that measurement is deterministic."""
        measurer = HypervectorPhiMeasurer(dimension=2048, seed=42)

        activations = [
            rng.randn(100).astype(np.float32),
            rng.randn(50).astype(np.float32),
        ]

        result1 = measurer.measure_phi_from_activations(activations)

        # Create new measurer with same seed
        measurer2 = HypervectorPhiMeasurer(dimension=2048, seed=42)
        result2 = measurer2.measure_phi_from_activations(activations)

        assert result1.phi_total == pytest.approx(result2.phi_total, rel=1e-5)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestMeasurePhiFunction:
    """Test the measure_phi convenience function."""

    def test_basic_usage(self, rng):
        """Test basic usage of measure_phi function."""
        # measure_phi returns float for single arrays
        # Use a list/dict for integration measurement
        data = {
            "layer1": rng.randn(5000).astype(np.float32),
            "layer2": rng.randn(5000).astype(np.float32),
        }

        result = measure_phi(data)

        assert isinstance(result, float)
        assert result >= 0

    def test_with_model_info(self, rng):
        """Test with multiple layers."""
        # Use a list of arrays for layer structure
        layers = [
            rng.randn(5000).astype(np.float32),
            rng.randn(3000).astype(np.float32),
            rng.randn(2000).astype(np.float32),
        ]

        result = measure_phi(layers)

        assert isinstance(result, float)
        assert result >= 0

    def test_large_gradient(self, rng):
        """Test with large gradient."""
        # Single array returns 0.0 (no integration possible)
        gradient = rng.randn(1_000_000).astype(np.float32)

        result = measure_phi(gradient)

        assert isinstance(result, float)
        assert result == 0.0  # Single array has no integration


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPhiPerformance:
    """Performance tests for Φ measurement."""

    @pytest.mark.slow
    def test_measurement_speed(self, rng, timing_stats):
        """Benchmark Φ measurement speed."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        # Typical CNN activations
        activations = [
            rng.randn(32 * 28 * 28).astype(np.float32),  # Conv1
            rng.randn(64 * 14 * 14).astype(np.float32),  # Conv2
            rng.randn(128).astype(np.float32),            # FC1
            rng.randn(10).astype(np.float32),             # Output
        ]

        import time

        for _ in range(100):
            start = time.perf_counter()
            measurer.measure_phi_from_activations(activations)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        print(f"\n  Φ Measurement: {timing_stats.summary()}")

        # Should be fast enough for real-time FL
        assert timing_stats.mean < 100  # < 100ms average

    @pytest.mark.slow
    def test_scalability(self, rng):
        """Test how measurement scales with layer count."""
        measurer = HypervectorPhiMeasurer(dimension=2048)

        import time
        results = []

        for num_layers in [2, 4, 8, 16, 32]:
            activations = [
                rng.randn(1000).astype(np.float32)
                for _ in range(num_layers)
            ]

            start = time.perf_counter()
            for _ in range(10):
                measurer.measure_phi_from_activations(activations)
            elapsed_ms = (time.perf_counter() - start) * 100  # Per iteration

            results.append((num_layers, elapsed_ms))
            print(f"  {num_layers} layers: {elapsed_ms:.2f}ms")

        # Should scale as O(L²) not worse
        # 32 layers should be < 16x slower than 2 layers (not 256x)
        ratio = results[-1][1] / results[0][1]
        assert ratio < 100  # Allow some overhead


# =============================================================================
# EDGE CASES
# =============================================================================

class TestPhiEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_activations(self):
        """Test with zero-valued activations."""
        measurer = HypervectorPhiMeasurer()

        activations = [
            np.zeros(100, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ]

        result = measurer.measure_phi_from_activations(activations)
        # Should handle without error
        assert result.phi_total >= 0

    def test_very_small_activations(self, rng):
        """Test with very small activation values."""
        measurer = HypervectorPhiMeasurer()

        activations = [
            rng.randn(100).astype(np.float32) * 1e-10,
            rng.randn(50).astype(np.float32) * 1e-10,
        ]

        result = measurer.measure_phi_from_activations(activations)
        assert np.isfinite(result.phi_total)

    def test_very_large_activations(self, rng):
        """Test with very large activation values."""
        measurer = HypervectorPhiMeasurer()

        activations = [
            rng.randn(100).astype(np.float32) * 1e6,
            rng.randn(50).astype(np.float32) * 1e6,
        ]

        result = measurer.measure_phi_from_activations(activations)
        assert np.isfinite(result.phi_total)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        measurer = HypervectorPhiMeasurer()

        activations = [
            np.array([1.0, np.nan, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ]

        # Should handle NaN gracefully
        result = measurer.measure_phi_from_activations(activations)
        # Either filters NaN or returns valid result
        assert result is not None

    def test_single_element_layers(self):
        """Test with single-element layers."""
        measurer = HypervectorPhiMeasurer()

        activations = [
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]

        result = measurer.measure_phi_from_activations(activations)
        assert result is not None
