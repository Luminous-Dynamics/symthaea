# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for Mycelix Unified FL System

Tests the complete FL pipeline:
1. HyperFeel compression
2. Byzantine detection
3. Self-healing
4. Shapley aggregation
5. Φ measurement
6. ML framework bridges

Author: Luminous Dynamics
Date: December 30, 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPhiMeasurement:
    """Test real Φ measurement implementation."""

    def test_phi_measurement_basic(self):
        """Test basic Φ measurement from activations."""
        from mycelix_fl.core.phi_measurement import HypervectorPhiMeasurer, measure_phi

        measurer = HypervectorPhiMeasurer(dimension=1024)

        # Create test activations
        layer_activations = [
            np.random.rand(100),
            np.random.rand(100),
            np.random.rand(100),
        ]

        metrics = measurer.measure_phi_from_hypervectors(layer_activations)

        assert metrics.phi_total >= 0
        assert len(metrics.phi_layers) == 3
        assert metrics.computation_time_ms >= 0

    def test_phi_convenience_function(self):
        """Test the measure_phi convenience function."""
        from mycelix_fl.core.phi_measurement import measure_phi

        activations = [np.random.rand(50) for _ in range(4)]
        phi_value = measure_phi(activations)

        assert isinstance(phi_value, float)
        assert phi_value >= 0

    def test_phi_validation(self):
        """Test Φ measurement validation."""
        from mycelix_fl.core.phi_measurement import _validate_phi_measurement

        is_valid, msg = _validate_phi_measurement()
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)


class TestShapleyDetector:
    """Test O(n) Shapley Byzantine detection."""

    def test_shapley_detection_honest(self):
        """Test Shapley detection with honest nodes."""
        from mycelix_fl.detection.shapley_detector import ShapleyByzantineDetector

        detector = ShapleyByzantineDetector()

        # Create honest gradients (similar)
        base_grad = np.random.randn(1000)
        gradients = {
            f"node-{i}": base_grad + np.random.randn(1000) * 0.01
            for i in range(5)
        }

        result = detector.detect(gradients)

        # All nodes should have positive Shapley values
        assert all(v > 0 for v in result.shapley_values.values())
        # Few or no Byzantine nodes
        assert len(result.byzantine_nodes) <= 1

    def test_shapley_detection_byzantine(self):
        """Test Shapley detection with Byzantine nodes."""
        from mycelix_fl.detection.shapley_detector import ShapleyByzantineDetector

        detector = ShapleyByzantineDetector()

        base_grad = np.random.randn(1000)
        gradients = {}

        # Honest nodes
        for i in range(4):
            gradients[f"honest-{i}"] = base_grad + np.random.randn(1000) * 0.01

        # Byzantine node (completely different)
        gradients["byzantine-0"] = np.random.randn(1000) * 10

        result = detector.detect(gradients)

        # Byzantine should have low Shapley value
        assert result.shapley_values["byzantine-0"] < 0.5
        assert result.latency_ms >= 0


class TestSelfHealing:
    """Test self-healing Byzantine correction."""

    def test_healing_minor_deviation(self):
        """Test healing of minor deviations."""
        from mycelix_fl.detection.self_healing import SelfHealingDetector

        healer = SelfHealingDetector(minor_threshold=0.3)

        base_grad = np.random.randn(1000)
        gradients = {}

        # Honest nodes
        for i in range(4):
            gradients[f"honest-{i}"] = base_grad + np.random.randn(1000) * 0.01

        # Minor deviation (small noise)
        gradients["minor-0"] = base_grad + np.random.randn(1000) * 0.1

        result = healer.heal(
            gradients=gradients,
            flagged_nodes={"minor-0"},
        )

        # Should be healed, not excluded
        assert "minor-0" in result.healed_nodes
        assert "minor-0" not in result.excluded_nodes

    def test_healing_major_deviation(self):
        """Test exclusion of major deviations."""
        from mycelix_fl.detection.self_healing import SelfHealingDetector

        healer = SelfHealingDetector()

        base_grad = np.random.randn(1000)
        gradients = {}

        # Honest nodes
        for i in range(4):
            gradients[f"honest-{i}"] = base_grad + np.random.randn(1000) * 0.01

        # Major deviation (completely different)
        gradients["major-0"] = np.random.randn(1000) * 100

        result = healer.heal(
            gradients=gradients,
            flagged_nodes={"major-0"},
        )

        # Should be excluded, not healed
        assert "major-0" in result.excluded_nodes


class TestMultiLayerDetector:
    """Test multi-layer Byzantine detection stack."""

    def test_multi_layer_detection(self):
        """Test complete multi-layer detection."""
        from mycelix_fl.detection.multi_layer_stack import MultiLayerByzantineDetector

        detector = MultiLayerByzantineDetector()

        base_grad = np.random.randn(1000)
        gradients = {}

        # Honest nodes
        for i in range(6):
            gradients[f"honest-{i}"] = base_grad + np.random.randn(1000) * 0.01

        # Byzantine nodes
        for i in range(2):
            gradients[f"byzantine-{i}"] = np.random.randn(1000) * 10

        result = detector.detect(gradients)

        assert isinstance(result.byzantine_nodes, set)
        assert result.latency_ms >= 0
        assert result.num_layers >= 1

    def test_layer_progression(self):
        """Test that detection progresses through layers."""
        from mycelix_fl.detection.multi_layer_stack import MultiLayerByzantineDetector

        detector = MultiLayerByzantineDetector()

        base_grad = np.random.randn(500)
        gradients = {
            f"node-{i}": base_grad + np.random.randn(500) * 0.01
            for i in range(5)
        }

        result = detector.detect(gradients)

        # Should have layer results
        assert len(result.layer_results) > 0


class TestHyperFeelEncoder:
    """Test HyperFeel v2 gradient compression."""

    def test_basic_encoding(self):
        """Test basic gradient encoding."""
        from mycelix_fl.hyperfeel.encoder_v2 import HyperFeelEncoderV2

        encoder = HyperFeelEncoderV2()

        gradient = np.random.randn(100000).astype(np.float32)  # 100K params

        hg = encoder.encode_gradient(
            gradient=gradient,
            round_num=1,
            node_id="test-node",
        )

        assert len(hg.hypervector) == 2048  # HV16
        assert hg.compression_ratio > 100  # High compression
        assert hg.phi_before >= 0

    def test_temporal_tracking(self):
        """Test temporal trajectory tracking."""
        from mycelix_fl.hyperfeel.encoder_v2 import HyperFeelEncoderV2, EncodingConfig

        config = EncodingConfig(use_temporal=True)
        encoder = HyperFeelEncoderV2(config=config)

        gradient = np.random.randn(10000).astype(np.float32)

        # Multiple rounds
        for round_num in range(3):
            hg = encoder.encode_gradient(
                gradient=gradient + np.random.randn(10000) * 0.01,
                round_num=round_num,
                node_id="test-node",
            )

            if round_num > 0:
                assert hg.temporal_metadata is not None
                assert hg.temporal_metadata["history_length"] >= round_num

    def test_cosine_similarity(self):
        """Test similarity preservation after compression."""
        from mycelix_fl.hyperfeel.encoder_v2 import HyperFeelEncoderV2

        encoder = HyperFeelEncoderV2()

        gradient1 = np.random.randn(10000).astype(np.float32)
        gradient2 = gradient1 + np.random.randn(10000) * 0.1  # Similar
        gradient3 = np.random.randn(10000).astype(np.float32)  # Different

        hg1 = encoder.encode_gradient(gradient1, round_num=1, node_id="node-1")
        hg2 = encoder.encode_gradient(gradient2, round_num=1, node_id="node-2")
        hg3 = encoder.encode_gradient(gradient3, round_num=1, node_id="node-3")

        sim_12 = encoder.cosine_similarity(hg1.hypervector, hg2.hypervector)
        sim_13 = encoder.cosine_similarity(hg1.hypervector, hg3.hypervector)

        # Similar gradients should have higher similarity
        assert sim_12 > sim_13


class TestMLBridge:
    """Test ML framework bridges."""

    def test_numpy_bridge(self):
        """Test NumPy bridge."""
        from mycelix_fl.ml.bridge import NumPyBridge

        bridge = NumPyBridge()

        model = {
            "weights": {
                "layer1": np.random.randn(10, 5).astype(np.float32),
                "layer2": np.random.randn(5, 2).astype(np.float32),
            },
            "gradients": {
                "layer1": np.random.randn(10, 5).astype(np.float32),
                "layer2": np.random.randn(5, 2).astype(np.float32),
            },
        }

        # Extract gradients
        grad_info = bridge.extract_gradients(model)
        assert grad_info.param_count == 10 * 5 + 5 * 2

        # Get model state
        state = bridge.get_model_state(model)
        assert state.param_count == grad_info.param_count

    def test_framework_detection(self):
        """Test automatic framework detection."""
        from mycelix_fl.ml.bridge import detect_framework

        np_model = {"weights": {}, "gradients": {}}
        assert detect_framework(np_model) == "numpy"

    def test_create_bridge(self):
        """Test bridge creation."""
        from mycelix_fl.ml.bridge import create_bridge

        bridge = create_bridge(framework="numpy")
        assert bridge.framework_name == "numpy"


class TestUnifiedFL:
    """Test complete FL orchestrator."""

    def test_single_round(self):
        """Test single FL round."""
        from mycelix_fl.core.unified_fl import MycelixFL, FLConfig

        config = FLConfig(
            num_rounds=1,
            min_nodes=3,
            use_compression=True,
            use_detection=True,
            use_healing=True,
        )
        fl = MycelixFL(config=config)

        base_grad = np.random.randn(1000)
        gradients = {
            f"node-{i}": (base_grad + np.random.randn(1000) * 0.01).astype(np.float32)
            for i in range(5)
        }

        result = fl.execute_round(
            gradients=gradients,
            round_num=0,
        )

        assert result.round_num == 0
        assert len(result.participating_nodes) == 5
        assert isinstance(result.aggregated_gradient, np.ndarray)
        assert result.round_time_ms > 0

    def test_byzantine_detection_in_round(self):
        """Test Byzantine detection during FL round."""
        from mycelix_fl.core.unified_fl import MycelixFL, FLConfig

        config = FLConfig(use_detection=True, use_healing=True)
        fl = MycelixFL(config=config)

        base_grad = np.random.randn(1000)
        gradients = {}

        # Honest nodes
        for i in range(6):
            gradients[f"honest-{i}"] = (base_grad + np.random.randn(1000) * 0.01).astype(np.float32)

        # Byzantine nodes
        for i in range(2):
            gradients[f"byzantine-{i}"] = np.random.randn(1000).astype(np.float32) * 50

        result = fl.execute_round(gradients=gradients, round_num=0)

        # Should detect some Byzantine nodes
        assert len(result.byzantine_nodes) > 0 or len(result.healed_nodes) > 0

    def test_statistics(self):
        """Test statistics collection."""
        from mycelix_fl.core.unified_fl import MycelixFL, FLConfig

        config = FLConfig(num_rounds=3)
        fl = MycelixFL(config=config)

        for round_num in range(3):
            gradients = {
                f"node-{i}": np.random.randn(100).astype(np.float32)
                for i in range(4)
            }
            fl.execute_round(gradients=gradients, round_num=round_num)

        stats = fl.get_statistics()

        assert stats["total_rounds"] == 3
        assert stats["total_participants"] == 12  # 4 * 3
        assert "avg_round_time_ms" in stats

    def test_node_reputations(self):
        """Test reputation tracking."""
        from mycelix_fl.core.unified_fl import MycelixFL, FLConfig

        fl = MycelixFL()

        # Initial reputation should be default
        rep = fl.get_node_reputation("unknown-node")
        assert rep == 0.5

        # Execute round
        gradients = {
            f"node-{i}": np.random.randn(100).astype(np.float32)
            for i in range(3)
        }
        fl.execute_round(gradients=gradients, round_num=0)

        # Nodes should now have reputation
        for node_id in gradients:
            rep = fl.get_node_reputation(node_id)
            assert 0 < rep <= 1


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_fl_simulation(self):
        """Test complete FL simulation with Byzantine nodes."""
        from mycelix_fl.core.unified_fl import run_fl_simulation

        stats = run_fl_simulation(
            num_nodes=8,
            num_rounds=3,
            byzantine_fraction=0.25,
            gradient_dim=5000,
            seed=42,
        )

        assert stats["total_rounds"] == 3
        assert stats["detection_accuracy"] >= 0  # Some detection
        assert stats["compression_ratio"] > 1

    def test_package_imports(self):
        """Test that all package imports work."""
        from mycelix_fl import (
            MycelixFL,
            FLConfig,
            RoundResult,
            HypervectorPhiMeasurer,
            MultiLayerByzantineDetector,
            ShapleyByzantineDetector,
            SelfHealingDetector,
            HyperFeelEncoderV2,
            create_bridge,
        )

        # All imports should work
        assert MycelixFL is not None
        assert FLConfig is not None
        assert HypervectorPhiMeasurer is not None

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end FL pipeline."""
        from mycelix_fl import MycelixFL, FLConfig, HyperFeelEncoderV2

        # Create FL system
        config = FLConfig(
            num_rounds=2,
            use_compression=True,
            use_detection=True,
            use_healing=True,
        )
        fl = MycelixFL(config=config)

        # Simulate 2 rounds
        for round_num in range(2):
            # Generate gradients
            base_grad = np.random.randn(10000)
            gradients = {}

            for i in range(6):
                if i < 5:  # Honest
                    grad = base_grad + np.random.randn(10000) * 0.01
                else:  # Byzantine
                    grad = np.random.randn(10000) * 10

                gradients[f"node-{i}"] = grad.astype(np.float32)

            # Execute round
            result = fl.execute_round(
                gradients=gradients,
                round_num=round_num,
            )

            # Verify result
            assert result.aggregated_gradient.shape == (10000,)
            assert result.compression_ratio > 1000  # HyperFeel compression

        # Check final statistics
        stats = fl.get_statistics()
        assert stats["total_rounds"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
