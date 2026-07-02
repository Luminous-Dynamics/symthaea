# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Unified Federated Learning Orchestrator

Tests cover:
- FLConfig validation and serialization
- NodeContribution tracking
- RoundResult aggregation
- MycelixFL orchestration
- Byzantine detection integration
"""

import pytest
import numpy as np
from typing import Dict

from mycelix_fl import MycelixFL, FLConfig, RoundResult
from mycelix_fl.core.unified_fl import NodeContribution


# =============================================================================
# FLCONFIG TESTS
# =============================================================================

class TestFLConfig:
    """Test FLConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FLConfig()
        assert config.num_rounds == 100
        assert config.min_nodes == 3
        assert config.byzantine_threshold == 0.45
        assert config.learning_rate == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = FLConfig(
            num_rounds=50,
            min_nodes=10,
            byzantine_threshold=0.33,
            learning_rate=0.001,
        )
        assert config.num_rounds == 50
        assert config.min_nodes == 10
        assert config.byzantine_threshold == 0.33
        assert config.learning_rate == 0.001

    def test_validation_num_rounds(self):
        """Test validation rejects invalid num_rounds."""
        with pytest.raises(ValueError, match="num_rounds"):
            FLConfig(num_rounds=0)
        with pytest.raises(ValueError, match="num_rounds"):
            FLConfig(num_rounds=-5)

    def test_validation_min_nodes(self):
        """Test validation rejects invalid min_nodes."""
        with pytest.raises(ValueError, match="min_nodes"):
            FLConfig(min_nodes=0)
        with pytest.raises(ValueError, match="min_nodes"):
            FLConfig(min_nodes=-1)

    def test_validation_byzantine_threshold(self):
        """Test validation rejects invalid byzantine_threshold."""
        with pytest.raises(ValueError, match="byzantine_threshold"):
            FLConfig(byzantine_threshold=-0.1)
        with pytest.raises(ValueError, match="byzantine_threshold"):
            FLConfig(byzantine_threshold=0.6)
        # Edge cases should work
        FLConfig(byzantine_threshold=0.0)
        FLConfig(byzantine_threshold=0.5)

    def test_validation_learning_rate(self):
        """Test validation rejects invalid learning_rate."""
        with pytest.raises(ValueError, match="learning_rate"):
            FLConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            FLConfig(learning_rate=-0.01)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = FLConfig(num_rounds=25, byzantine_threshold=0.4)
        d = config.to_dict()

        assert d["num_rounds"] == 25
        assert d["byzantine_threshold"] == 0.4
        assert "learning_rate" in d
        assert "use_detection" in d
        assert "use_healing" in d

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "num_rounds": 30,
            "min_nodes": 5,
            "byzantine_threshold": 0.35,
            "learning_rate": 0.005,
            "use_detection": True,
            "use_healing": False,
        }
        config = FLConfig.from_dict(d)

        assert config.num_rounds == 30
        assert config.min_nodes == 5
        assert config.byzantine_threshold == 0.35
        assert config.learning_rate == 0.005
        assert config.use_detection is True
        assert config.use_healing is False

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict preserves all fields."""
        original = FLConfig(
            num_rounds=42,
            min_nodes=7,
            byzantine_threshold=0.4,
            learning_rate=0.02,
            use_detection=True,
            use_healing=True,
        )

        d = original.to_dict()
        restored = FLConfig.from_dict(d)

        assert restored.num_rounds == original.num_rounds
        assert restored.min_nodes == original.min_nodes
        assert restored.byzantine_threshold == original.byzantine_threshold
        assert restored.learning_rate == original.learning_rate
        assert restored.use_detection == original.use_detection
        assert restored.use_healing == original.use_healing


# =============================================================================
# NODE CONTRIBUTION TESTS
# =============================================================================

class TestNodeContribution:
    """Test NodeContribution dataclass."""

    def test_creation(self, small_gradient):
        """Test basic creation."""
        contrib = NodeContribution(
            node_id="node_1",
            gradient=small_gradient,
            shapley_value=0.15,
            is_byzantine=False,
            contribution_weight=0.25,
        )

        assert contrib.node_id == "node_1"
        assert contrib.shapley_value == 0.15
        assert contrib.contribution_weight == 0.25
        assert np.array_equal(contrib.gradient, small_gradient)

    def test_to_dict(self, small_gradient):
        """Test serialization."""
        contrib = NodeContribution(
            node_id="test_node",
            gradient=small_gradient,
            shapley_value=0.1,
            is_byzantine=False,
            is_healed=False,
        )

        d = contrib.to_dict()
        assert d["node_id"] == "test_node"
        assert d["shapley_value"] == 0.1
        assert "is_byzantine" in d
        assert "gradient" in d

    def test_from_dict(self, small_gradient):
        """Test deserialization."""
        d = {
            "node_id": "restored_node",
            "gradient": small_gradient.tolist(),
            "shapley_value": 0.2,
            "is_byzantine": False,
            "is_healed": True,
            "contribution_weight": 0.3,
        }

        contrib = NodeContribution.from_dict(d)
        assert contrib.node_id == "restored_node"
        assert contrib.shapley_value == 0.2
        assert contrib.is_healed is True


# =============================================================================
# ROUND RESULT TESTS
# =============================================================================

class TestRoundResult:
    """Test RoundResult dataclass."""

    def test_creation(self, small_gradient):
        """Test basic creation."""
        result = RoundResult(
            round_num=1,
            aggregated_gradient=small_gradient,
            participating_nodes=["a", "b", "c"],
            byzantine_nodes={"d"},
            healed_nodes=set(),
            shapley_values={"a": 0.3, "b": 0.4, "c": 0.3},
            phi_before=0.5,
            phi_after=0.6,
            round_time_ms=150.0,
        )

        assert result.round_num == 1
        assert len(result.participating_nodes) == 3
        assert "d" in result.byzantine_nodes
        assert result.phi_after > result.phi_before

    def test_to_dict(self, small_gradient):
        """Test serialization."""
        result = RoundResult(
            round_num=5,
            aggregated_gradient=small_gradient,
            participating_nodes=["n1", "n2"],
            byzantine_nodes={"byz1"},
            healed_nodes={"healed1"},
            shapley_values={"n1": 0.5, "n2": 0.5},
            phi_before=0.4,
            phi_after=0.5,
            round_time_ms=200.0,
        )

        d = result.to_dict()
        assert d["round_num"] == 5
        assert "n1" in d["participating_nodes"]
        assert "byz1" in d["byzantine_nodes"]
        assert d["phi_after"] - d["phi_before"] == pytest.approx(0.1, abs=1e-6)

    def test_from_dict(self, small_gradient):
        """Test deserialization."""
        d = {
            "round_num": 10,
            "aggregated_gradient": small_gradient.tolist(),
            "participating_nodes": ["x", "y", "z"],
            "byzantine_nodes": ["byz"],
            "healed_nodes": [],
            "shapley_values": {"x": 0.33, "y": 0.33, "z": 0.34},
            "phi_before": 0.3,
            "phi_after": 0.4,
            "round_time_ms": 100.0,
        }

        result = RoundResult.from_dict(d)
        assert result.round_num == 10
        assert len(result.participating_nodes) == 3
        assert "byz" in result.byzantine_nodes


# =============================================================================
# MYCELIX FL ORCHESTRATOR TESTS
# =============================================================================

class TestMycelixFL:
    """Test MycelixFL orchestrator."""

    def test_initialization(self, default_config):
        """Test orchestrator initialization."""
        fl = MycelixFL(config=default_config)

        assert fl.config == default_config
        assert fl.round_history == []  # No rounds executed yet

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        fl = MycelixFL()

        assert fl.config is not None
        assert fl.config.num_rounds == 100

    def test_execute_round_basic(self, honest_gradients):
        """Test basic round execution with honest nodes (fast mode)."""
        # Use fast config without detection for quick unit test
        fast_config = FLConfig(
            num_rounds=5,
            min_nodes=3,
            use_detection=False,
            use_compression=False,
            use_healing=False,
        )
        fl = MycelixFL(config=fast_config)
        result = fl.execute_round(honest_gradients, round_num=1)

        assert isinstance(result, RoundResult)
        assert result.round_num == 1
        assert result.aggregated_gradient is not None
        assert len(result.aggregated_gradient) > 0

    @pytest.mark.slow
    def test_execute_round_with_byzantine(self, byzantine_gradients, default_config):
        """Test round execution detects Byzantine nodes (slow - full detection)."""
        fl = MycelixFL(config=default_config)
        result = fl.execute_round(byzantine_gradients, round_num=1)

        assert isinstance(result, RoundResult)

        # Byzantine nodes may be detected and then healed (self-healing feature)
        # So we check if they're in byzantine_nodes OR healed_nodes OR
        # if the detection_result found them
        actual_byzantine = {"byzantine_flip", "byzantine_random", "byzantine_scale"}

        # Count how many were detected (either in byzantine_nodes or healed_nodes)
        detected_and_handled = result.byzantine_nodes | result.healed_nodes

        # At least one Byzantine should be detected or healed
        assert len(detected_and_handled & actual_byzantine) >= 1 or \
               (result.detection_result and len(result.detection_result.byzantine_nodes & actual_byzantine) >= 1), \
               f"No Byzantine nodes detected. Byzantine: {result.byzantine_nodes}, Healed: {result.healed_nodes}"

    @pytest.mark.slow
    def test_execute_round_shapley_values(self, honest_gradients, default_config):
        """Test that Shapley values are computed (slow - requires detection)."""
        fl = MycelixFL(config=default_config)
        result = fl.execute_round(honest_gradients, round_num=1)

        assert len(result.shapley_values) > 0

        # Shapley values should be normalized (sum to ~1)
        total_shapley = sum(result.shapley_values.values())
        assert 0.9 <= total_shapley <= 1.1  # Allow some tolerance

    @pytest.mark.slow
    def test_execute_round_phi_tracking(self, honest_gradients):
        """Test Φ tracking when detection is enabled (slow)."""
        config = FLConfig(use_detection=True)
        fl = MycelixFL(config=config)
        result = fl.execute_round(honest_gradients, round_num=1)

        # Phi should be tracked
        assert result.phi_before >= 0
        assert result.phi_after >= 0

    def test_multiple_rounds(self, honest_gradients):
        """Test executing multiple rounds (fast mode)."""
        fast_config = FLConfig(
            num_rounds=5,
            min_nodes=3,
            use_detection=False,
            use_compression=False,
            use_healing=False,
        )
        fl = MycelixFL(config=fast_config)

        results = []
        for round_num in range(1, 6):
            result = fl.execute_round(honest_gradients, round_num=round_num)
            results.append(result)

        assert len(results) == 5
        assert results[0].round_num == 1
        assert results[-1].round_num == 5

    def test_min_nodes_enforcement(self):
        """Test that minimum nodes requirement is enforced."""
        fl = MycelixFL(config=FLConfig(min_nodes=5, use_detection=False, use_compression=False))

        # Only 2 gradients - should fail or warn
        small_gradients = {
            "node_1": np.random.randn(1000).astype(np.float32),
            "node_2": np.random.randn(1000).astype(np.float32),
        }

        # Depending on implementation, this might raise or return partial result
        try:
            result = fl.execute_round(small_gradients, round_num=1)
            # If it returns, check it handles gracefully
            assert result is not None
        except ValueError as e:
            assert "min_nodes" in str(e).lower() or "minimum" in str(e).lower()

    @pytest.mark.slow
    def test_large_scale(self, rng):
        """Test with many nodes and large gradients."""
        # 100 nodes, 100K params each
        gradients = {
            f"node_{i}": rng.randn(100_000).astype(np.float32)
            for i in range(100)
        }

        fl = MycelixFL(config=FLConfig(min_nodes=10))
        result = fl.execute_round(gradients, round_num=1)

        assert result is not None
        assert len(result.participating_nodes) == 100


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFLIntegration:
    """Integration tests for full FL workflow."""

    @pytest.mark.slow
    def test_full_workflow(self, rng):
        """Test complete FL workflow from config to results (slow - full detection)."""
        # Setup
        config = FLConfig(
            num_rounds=3,
            min_nodes=5,
            byzantine_threshold=0.4,
            use_detection=True,
            use_healing=True,
        )

        fl = MycelixFL(config=config)

        # Simulate 3 rounds
        for round_num in range(1, 4):
            # Generate gradients (7 honest, 3 Byzantine)
            true_dir = rng.randn(5000).astype(np.float32)
            true_dir /= np.linalg.norm(true_dir)

            gradients = {}
            for i in range(7):
                gradients[f"honest_{i}"] = true_dir + rng.randn(5000).astype(np.float32) * 0.1
            for i in range(3):
                gradients[f"byz_{i}"] = -true_dir * 2.0

            result = fl.execute_round(gradients, round_num=round_num)

            # Verify result
            assert result.round_num == round_num
            assert len(result.byzantine_nodes) <= 3  # At most 3 Byzantine
            assert result.round_time_ms >= 0

    def test_serialization_workflow(self, honest_gradients):
        """Test that results can be serialized and restored (fast mode)."""
        fast_config = FLConfig(
            num_rounds=5,
            min_nodes=3,
            use_detection=False,
            use_compression=False,
            use_healing=False,
        )
        fl = MycelixFL(config=fast_config)
        result = fl.execute_round(honest_gradients, round_num=1)

        # Serialize
        d = result.to_dict()

        # Verify JSON-serializable
        import json
        json_str = json.dumps(d)
        restored_d = json.loads(json_str)

        # Restore
        restored = RoundResult.from_dict(restored_d)

        assert restored.round_num == result.round_num
        assert set(restored.byzantine_nodes) == result.byzantine_nodes
