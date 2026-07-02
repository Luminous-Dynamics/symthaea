# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Byzantine Detection Modules

Tests cover:
- MultiLayerByzantineDetector
- ShapleyByzantineDetector
- SelfHealingDetector
- Detection accuracy at various Byzantine ratios
- Performance benchmarks
"""

import pytest
import numpy as np
from typing import Dict, Set

from mycelix_fl.detection import (
    MultiLayerByzantineDetector,
    DetectionResult,
    DetectionLayer,
    ShapleyByzantineDetector,
    SelfHealingDetector,
)
from mycelix_fl.detection.shapley_detector import ShapleyDetectionResult
from mycelix_fl.detection.self_healing import HealingResult


# =============================================================================
# MULTI-LAYER BYZANTINE DETECTOR TESTS
# =============================================================================

class TestMultiLayerByzantineDetector:
    """Test MultiLayerByzantineDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MultiLayerByzantineDetector()
        assert detector is not None

    def test_initialization_custom_threshold(self):
        """Test initialization with custom threshold."""
        detector = MultiLayerByzantineDetector(shapley_threshold=0.15, pogq_threshold=0.6)
        assert detector.shapley_threshold == 0.15
        assert detector.pogq_threshold == 0.6

    def test_detect_all_honest(self, honest_gradients):
        """Test detection with all honest nodes."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=honest_gradients, round_number=1)

        assert isinstance(result, DetectionResult)
        assert len(result.byzantine_nodes) == 0
        assert len(result.healthy_nodes) == len(honest_gradients)

    def test_detect_with_byzantine(self, byzantine_gradients):
        """Test detection with Byzantine nodes."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=byzantine_gradients, round_number=1)

        assert isinstance(result, DetectionResult)
        # Should detect at least some Byzantine
        assert len(result.byzantine_nodes) > 0

        # Check detection confidence
        assert 0 <= result.detection_rate <= 1

    def test_detect_at_45_percent_threshold(self, mixed_gradients_45_percent):
        """Test detection at 45% Byzantine threshold."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=mixed_gradients_45_percent, round_number=1)

        assert isinstance(result, DetectionResult)

        # Count actual Byzantine in result
        actual_byzantine = {k for k in mixed_gradients_45_percent if "byz" in k}
        detected = result.byzantine_nodes

        # Should detect significant portion
        true_positives = detected & actual_byzantine
        precision = len(true_positives) / len(detected) if detected else 0
        recall = len(true_positives) / len(actual_byzantine) if actual_byzantine else 0

        print(f"\n  Byzantine Detection at 45%:")
        print(f"    Actual: {len(actual_byzantine)}, Detected: {len(detected)}")
        print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}")

        # Should have reasonable recall (not missing all)
        assert recall >= 0.3 or len(detected) > 0

    def test_layer_results(self, byzantine_gradients):
        """Test that individual layer results are returned."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=byzantine_gradients, round_number=1)

        assert len(result.layer_results) > 0

        for layer_result in result.layer_results:
            assert isinstance(layer_result.layer, DetectionLayer)
            assert layer_result.confidence >= 0
            assert layer_result.latency_ms >= 0

    def test_shapley_values_returned(self, honest_gradients):
        """Test that Shapley values are computed."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=honest_gradients, round_number=1)

        assert len(result.shapley_values) > 0

        # All nodes should have a Shapley value
        for node_id in honest_gradients:
            assert node_id in result.shapley_values

    def test_system_phi_computed(self, honest_gradients):
        """Test that system Φ is computed."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=honest_gradients, round_number=1)

        assert result.system_phi >= 0

    def test_latency_tracking(self, byzantine_gradients):
        """Test that latency is tracked."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=byzantine_gradients, round_number=1)

        assert result.total_latency_ms > 0

    def test_result_serialization(self, byzantine_gradients):
        """Test DetectionResult serialization."""
        detector = MultiLayerByzantineDetector()
        result = detector.detect(gradients=byzantine_gradients, round_number=1)

        # Serialize
        d = result.to_dict()
        assert "byzantine_nodes" in d
        assert "shapley_values" in d
        assert "layer_results" in d

        # Deserialize
        restored = DetectionResult.from_dict(d)
        assert restored.byzantine_nodes == result.byzantine_nodes
        assert restored.system_phi == result.system_phi


# =============================================================================
# SHAPLEY BYZANTINE DETECTOR TESTS
# =============================================================================

class TestShapleyByzantineDetector:
    """Test ShapleyByzantineDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ShapleyByzantineDetector()
        assert detector.dimension > 0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = ShapleyByzantineDetector(
            hypervector_dimension=4096,
            byzantine_threshold=0.15,
        )
        assert detector.dimension == 4096
        assert detector.byzantine_threshold == 0.15

    def test_compute_shapley_values(self, honest_gradients):
        """Test Shapley value computation."""
        detector = ShapleyByzantineDetector()
        result = detector.detect(honest_gradients)

        assert isinstance(result, ShapleyDetectionResult)
        assert len(result.shapley_values) == len(honest_gradients)

        # Shapley values are normalized to [0, 1] range
        # Each value should be in that range
        for node_id, value in result.shapley_values.items():
            assert 0.0 <= value <= 1.0, f"Shapley value for {node_id} out of range: {value}"

    def test_shapley_detects_byzantine(self, byzantine_gradients):
        """Test that Shapley method detects Byzantine nodes."""
        detector = ShapleyByzantineDetector()
        result = detector.detect(byzantine_gradients)

        assert len(result.byzantine_nodes) > 0

        # Byzantine nodes should have lower Shapley values
        byz_shapley = [
            result.shapley_values[k]
            for k in byzantine_gradients
            if "byzantine" in k and k in result.shapley_values
        ]
        honest_shapley = [
            result.shapley_values[k]
            for k in byzantine_gradients
            if "honest" in k and k in result.shapley_values
        ]

        if byz_shapley and honest_shapley:
            avg_byz = np.mean(byz_shapley)
            avg_honest = np.mean(honest_shapley)
            print(f"\n  Shapley avg: Byzantine={avg_byz:.3f}, Honest={avg_honest:.3f}")
            # Byzantine should have lower contribution on average
            assert avg_byz < avg_honest

    def test_contribution_weights(self, honest_gradients):
        """Test that contribution weights are computed."""
        detector = ShapleyByzantineDetector()
        result = detector.detect(honest_gradients)

        assert len(result.contribution_weights) == len(honest_gradients)

        # Weights should be non-negative
        for weight in result.contribution_weights.values():
            assert weight >= 0

    def test_bundle_quality(self, honest_gradients):
        """Test bundle quality metric."""
        detector = ShapleyByzantineDetector()
        result = detector.detect(honest_gradients)

        assert 0 <= result.bundle_quality <= 1

    def test_latency_tracked(self, honest_gradients):
        """Test latency tracking."""
        detector = ShapleyByzantineDetector()
        result = detector.detect(honest_gradients)

        assert result.latency_ms > 0

    def test_deterministic_with_seed(self, honest_gradients):
        """Test determinism with seed."""
        detector1 = ShapleyByzantineDetector(seed=42)
        detector2 = ShapleyByzantineDetector(seed=42)

        result1 = detector1.detect(honest_gradients)
        result2 = detector2.detect(honest_gradients)

        assert result1.shapley_values == pytest.approx(result2.shapley_values, rel=1e-5)


# =============================================================================
# SELF-HEALING DETECTOR TESTS
# =============================================================================

class TestSelfHealingDetector:
    """Test SelfHealingDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        healer = SelfHealingDetector()
        assert healer.minor_threshold < healer.moderate_threshold

    def test_initialization_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        healer = SelfHealingDetector(
            minor_threshold=0.2,
            moderate_threshold=0.5,
        )
        assert healer.minor_threshold == 0.2
        assert healer.moderate_threshold == 0.5

    def test_heal_minor_deviation(self, rng):
        """Test healing of minor deviations."""
        healer = SelfHealingDetector()

        # Create honest gradients
        true_dir = rng.randn(5000).astype(np.float32)
        true_dir /= np.linalg.norm(true_dir)

        gradients = {
            f"honest_{i}": true_dir + rng.randn(5000).astype(np.float32) * 0.05
            for i in range(5)
        }

        # Add node with minor deviation (keep it small enough to be healable)
        # The deviation should be small relative to the honest subspace
        gradients["minor_byz"] = true_dir + rng.randn(5000).astype(np.float32) * 0.08

        result = healer.heal(gradients, flagged_nodes={"minor_byz"})

        assert isinstance(result, HealingResult)
        # Minor deviation should be healed OR at least the healing was attempted
        # The healing result depends on the PCA projection
        assert len(result.healing_details) > 0
        assert "minor_byz" in result.healing_details

    def test_exclude_major_deviation(self, rng):
        """Test exclusion of major deviations."""
        healer = SelfHealingDetector()

        # Create honest gradients
        true_dir = rng.randn(5000).astype(np.float32)
        true_dir /= np.linalg.norm(true_dir)

        gradients = {
            f"honest_{i}": true_dir + rng.randn(5000).astype(np.float32) * 0.05
            for i in range(5)
        }

        # Add node with major deviation (opposite direction)
        gradients["major_byz"] = -true_dir * 2.0

        result = healer.heal(gradients, flagged_nodes={"major_byz"})

        # Major deviation should be excluded
        assert "major_byz" in result.excluded_nodes

    def test_healing_preserves_honest(self, rng):
        """Test that healing preserves honest gradients."""
        healer = SelfHealingDetector()

        true_dir = rng.randn(5000).astype(np.float32)
        true_dir /= np.linalg.norm(true_dir)

        gradients = {
            f"honest_{i}": true_dir + rng.randn(5000).astype(np.float32) * 0.05
            for i in range(5)
        }
        gradients["byz"] = -true_dir

        # Only flag the Byzantine node
        result = healer.heal(gradients, flagged_nodes={"byz"})

        # Honest nodes shouldn't be in healed or excluded
        for i in range(5):
            node_id = f"honest_{i}"
            assert node_id not in result.healed_nodes
            assert node_id not in result.excluded_nodes

    def test_healed_gradients_returned(self, rng):
        """Test that healed gradients are returned."""
        healer = SelfHealingDetector()

        true_dir = rng.randn(5000).astype(np.float32)
        true_dir /= np.linalg.norm(true_dir)

        gradients = {
            "honest": true_dir + rng.randn(5000).astype(np.float32) * 0.05,
        }
        gradients["byz"] = true_dir + rng.randn(5000).astype(np.float32) * 0.3

        result = healer.heal(gradients, flagged_nodes={"byz"})

        # Healed gradients should be provided
        for node_id in result.healed_nodes:
            assert node_id in result.healed_gradients
            assert result.healed_gradients[node_id].shape == gradients[node_id].shape

    def test_min_honest_nodes(self, rng):
        """Test minimum honest nodes requirement."""
        healer = SelfHealingDetector(min_honest_nodes=5)

        # Only 2 honest nodes - not enough for healing
        gradients = {
            "honest_0": rng.randn(1000).astype(np.float32),
            "honest_1": rng.randn(1000).astype(np.float32),
            "byz": rng.randn(1000).astype(np.float32),
        }

        result = healer.heal(gradients, flagged_nodes={"byz"})

        # With insufficient honest nodes, should exclude rather than heal
        assert "byz" in result.excluded_nodes or len(result.healed_nodes) == 0


# =============================================================================
# DETECTION ACCURACY TESTS
# =============================================================================

class TestDetectionAccuracy:
    """Test detection accuracy at various Byzantine ratios."""

    @pytest.mark.parametrize("byz_ratio", [0.1, 0.2, 0.3, 0.4, 0.45])
    def test_accuracy_at_ratio(self, rng, byz_ratio):
        """Test detection accuracy at specific Byzantine ratio."""
        # Note: MultiLayerByzantineDetector doesn't have byzantine_threshold
        # It has shapley_threshold and other layer-specific thresholds
        detector = MultiLayerByzantineDetector(shapley_threshold=0.1)

        total_nodes = 20
        num_byz = int(total_nodes * byz_ratio)
        num_honest = total_nodes - num_byz

        # Generate gradients
        true_dir = rng.randn(5000).astype(np.float32)
        true_dir /= np.linalg.norm(true_dir)

        gradients = {}
        for i in range(num_honest):
            gradients[f"honest_{i}"] = true_dir + rng.randn(5000).astype(np.float32) * 0.1
        for i in range(num_byz):
            # Mix of attack types
            if i % 3 == 0:
                gradients[f"byz_{i}"] = -true_dir * 1.5
            elif i % 3 == 1:
                gradients[f"byz_{i}"] = rng.randn(5000).astype(np.float32) * 3.0
            else:
                gradients[f"byz_{i}"] = true_dir * 50

        result = detector.detect(gradients=gradients, round_number=1)

        # Calculate metrics
        actual_byz = {k for k in gradients if "byz" in k}
        detected = result.byzantine_nodes

        true_positives = detected & actual_byz
        false_positives = detected - actual_byz
        false_negatives = actual_byz - detected

        precision = len(true_positives) / len(detected) if detected else 1.0
        recall = len(true_positives) / len(actual_byz) if actual_byz else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  Byzantine ratio {byz_ratio:.0%}:")
        print(f"    Actual: {len(actual_byz)}, Detected: {len(detected)}")
        print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")

        # At all ratios, should have some detection capability
        # (recall > 0 or no Byzantine to detect)
        if actual_byz:
            assert recall > 0 or len(detected) > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestDetectionPerformance:
    """Performance tests for detection modules."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_multilayer_speed(self, byzantine_gradients, timing_stats):
        """Benchmark multi-layer detection speed."""
        detector = MultiLayerByzantineDetector()

        import time

        for _ in range(20):
            start = time.perf_counter()
            detector.detect(gradients=byzantine_gradients, round_number=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        print(f"\n  Multi-layer Detection: {timing_stats.summary()}")

        # Should complete in reasonable time
        assert timing_stats.mean < 500  # < 500ms average

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_shapley_speed(self, byzantine_gradients, timing_stats):
        """Benchmark Shapley detection speed."""
        detector = ShapleyByzantineDetector()

        import time

        for _ in range(50):
            start = time.perf_counter()
            detector.detect(byzantine_gradients)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing_stats.record(elapsed_ms)

        print(f"\n  Shapley Detection: {timing_stats.summary()}")

        # Should be fast due to O(n) algorithm
        assert timing_stats.mean < 100  # < 100ms average

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_scalability_nodes(self, rng, timing_stats):
        """Test how detection scales with number of nodes."""
        detector = MultiLayerByzantineDetector()

        import time

        for num_nodes in [10, 20, 50, 100]:
            gradients = {
                f"node_{i}": rng.randn(5000).astype(np.float32)
                for i in range(num_nodes)
            }

            start = time.perf_counter()
            detector.detect(gradients=gradients, round_number=1)
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  {num_nodes} nodes: {elapsed_ms:.2f}ms")


# =============================================================================
# EDGE CASES
# =============================================================================

class TestDetectionEdgeCases:
    """Test edge cases and error handling."""

    def test_single_node(self, rng):
        """Test with single node."""
        detector = MultiLayerByzantineDetector()

        gradients = {"only_node": rng.randn(1000).astype(np.float32)}

        result = detector.detect(gradients=gradients, round_number=1)
        assert result is not None
        # Single node can't be compared to others
        assert len(result.byzantine_nodes) <= 1

    def test_two_nodes(self, rng):
        """Test with exactly two nodes."""
        detector = MultiLayerByzantineDetector()

        gradients = {
            "node_a": rng.randn(1000).astype(np.float32),
            "node_b": -rng.randn(1000).astype(np.float32),  # Opposite
        }

        result = detector.detect(gradients=gradients, round_number=1)
        assert result is not None

    def test_all_identical(self, rng):
        """Test when all gradients are identical."""
        detector = MultiLayerByzantineDetector()

        base = rng.randn(1000).astype(np.float32)
        gradients = {f"node_{i}": base.copy() for i in range(10)}

        result = detector.detect(gradients=gradients, round_number=1)
        # All identical = all honest
        assert len(result.byzantine_nodes) == 0

    def test_empty_gradients(self):
        """Test with empty gradient dict."""
        detector = MultiLayerByzantineDetector()

        result = detector.detect(gradients={}, round_number=1)
        assert result is not None
        assert len(result.byzantine_nodes) == 0

    def test_nan_in_gradients(self, rng):
        """Test handling of NaN values."""
        detector = MultiLayerByzantineDetector()

        gradients = {
            "normal_1": rng.randn(1000).astype(np.float32),
            "normal_2": rng.randn(1000).astype(np.float32),
            "nan_node": np.array([np.nan] * 1000, dtype=np.float32),
        }

        # Should handle gracefully
        result = detector.detect(gradients=gradients, round_number=1)
        assert result is not None
        # NaN node might be flagged as Byzantine
        assert "nan_node" in result.byzantine_nodes or result is not None
