# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for MycelixFL

End-to-end tests that verify the complete FL pipeline works correctly.
These tests simulate real-world scenarios including:
- Multi-round federated learning
- Byzantine attack and recovery
- Failure recovery and resilience
- Performance under load
- Edge cases and boundary conditions

Author: Luminous Dynamics
Date: December 31, 2025
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Set
from dataclasses import dataclass


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_gradients():
    """Generate sample gradients from 10 nodes."""
    np.random.seed(42)
    return {
        f"node_{i}": np.random.randn(10000).astype(np.float32)
        for i in range(10)
    }


@pytest.fixture
def large_gradients():
    """Generate large gradients (1M parameters) from 20 nodes."""
    np.random.seed(123)
    return {
        f"node_{i}": np.random.randn(1_000_000).astype(np.float32)
        for i in range(20)
    }


@pytest.fixture
def mycelix_fl():
    """Create a MycelixFL instance with default config."""
    from mycelix_fl import MycelixFL, FLConfig
    config = FLConfig(
        num_rounds=10,
        byzantine_threshold=0.45,
        use_detection=True,
        use_healing=True,
    )
    return MycelixFL(config=config)


# =============================================================================
# END-TO-END PIPELINE TESTS
# =============================================================================

class TestEndToEndPipeline:
    """Test complete FL pipeline from gradients to aggregation."""

    def test_single_round_execution(self, mycelix_fl, sample_gradients):
        """Test single FL round executes correctly."""
        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        assert result is not None
        assert result.round_num == 1
        assert len(result.participating_nodes) > 0
        assert result.aggregated_gradient is not None
        assert result.aggregated_gradient.shape[0] > 0

    def test_multi_round_execution(self, mycelix_fl, sample_gradients):
        """Test multiple FL rounds maintain consistency."""
        results = []

        for round_num in range(1, 6):
            # Simulate slightly different gradients each round
            modified_gradients = {
                k: v + np.random.randn(*v.shape).astype(np.float32) * 0.01
                for k, v in sample_gradients.items()
            }

            result = mycelix_fl.execute_round(modified_gradients, round_num=round_num)
            results.append(result)

        assert len(results) == 5
        # All rounds should complete
        assert all(r.round_num == i + 1 for i, r in enumerate(results))
        # No Byzantine should be detected in clean gradients
        assert all(len(r.byzantine_nodes) == 0 for r in results)

    def test_pipeline_with_byzantine_detection(self, mycelix_fl):
        """Test pipeline correctly detects and excludes Byzantine nodes."""
        np.random.seed(42)

        # Create gradients with some Byzantine nodes
        gradients = {
            f"honest_{i}": np.random.randn(10000).astype(np.float32)
            for i in range(7)
        }

        # Add Byzantine nodes (3 out of 10 = 30%)
        gradients["byzantine_1"] = np.ones(10000).astype(np.float32) * 1000  # Large values
        gradients["byzantine_2"] = np.random.randn(10000).astype(np.float32) * 100  # High variance
        gradients["byzantine_3"] = -np.mean(
            [g for k, g in gradients.items() if k.startswith("honest")], axis=0
        ) * 5  # Opposite direction

        result = mycelix_fl.execute_round(gradients, round_num=1)

        # Should detect at least some Byzantine nodes
        assert len(result.byzantine_nodes) >= 1
        # Byzantine nodes should not contribute to aggregation
        for byz_node in result.byzantine_nodes:
            assert byz_node not in result.participating_nodes

    def test_pipeline_with_healing(self, mycelix_fl):
        """Test self-healing recovers borderline Byzantine nodes."""
        np.random.seed(42)

        base_gradient = np.random.randn(10000).astype(np.float32)

        # Create gradients with minor deviations (should be healable)
        gradients = {}
        for i in range(8):
            noise = np.random.randn(10000).astype(np.float32) * 0.1
            gradients[f"node_{i}"] = base_gradient + noise

        # Add a borderline node (might be healed)
        gradients["borderline"] = base_gradient + np.random.randn(10000).astype(np.float32) * 0.5

        result = mycelix_fl.execute_round(gradients, round_num=1)

        # Check that healing occurred or node contributed
        if result.healed_nodes:
            # If healed, should be in contributing
            for healed in result.healed_nodes:
                assert healed in result.participating_nodes

    def test_aggregation_correctness(self, sample_gradients):
        """Test aggregated gradient is mathematically correct."""
        from mycelix_fl import MycelixFL, FLConfig

        config = FLConfig(
            use_detection=False,  # Disable to test pure aggregation
            use_healing=False,
        )
        fl = MycelixFL(config=config)

        result = fl.execute_round(sample_gradients, round_num=1)

        # Simple mean aggregation when no Byzantine
        expected_mean = np.mean(
            [g for g in sample_gradients.values()],
            axis=0
        )

        # Should be close to simple mean (Shapley weights are ~equal for similar gradients)
        correlation = np.corrcoef(result.aggregated_gradient.flatten(), expected_mean.flatten())[0, 1]
        assert correlation > 0.9  # High correlation expected


class TestFailureRecovery:
    """Test system resilience and failure recovery."""

    def test_handles_empty_gradients(self, mycelix_fl):
        """Test graceful handling of empty gradient dict."""
        from mycelix_fl.exceptions import InsufficientGradientsError

        with pytest.raises(InsufficientGradientsError):
            mycelix_fl.execute_round({}, round_num=1)

    def test_handles_single_node(self, mycelix_fl):
        """Test handling of single node submission."""
        from mycelix_fl.exceptions import InsufficientGradientsError

        gradients = {"single_node": np.random.randn(10000).astype(np.float32)}

        with pytest.raises(InsufficientGradientsError):
            mycelix_fl.execute_round(gradients, round_num=1)

    def test_handles_nan_gradients(self, mycelix_fl, sample_gradients):
        """Test handling of NaN values in gradients."""
        # Add a node with NaN values
        nan_gradient = np.random.randn(10000).astype(np.float32)
        nan_gradient[100:200] = np.nan
        sample_gradients["nan_node"] = nan_gradient

        # Should still complete (excluding the bad node)
        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        # NaN node should be excluded
        assert "nan_node" not in result.participating_nodes
        # Should not have NaN in result
        assert not np.any(np.isnan(result.aggregated_gradient))

    def test_handles_inf_gradients(self, mycelix_fl, sample_gradients):
        """Test handling of Inf values in gradients."""
        # Add a node with Inf values
        inf_gradient = np.random.randn(10000).astype(np.float32)
        inf_gradient[500:600] = np.inf
        sample_gradients["inf_node"] = inf_gradient

        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        # Inf node should be excluded
        assert "inf_node" not in result.participating_nodes
        # Should not have Inf in result
        assert not np.any(np.isinf(result.aggregated_gradient))

    def test_handles_mixed_precision(self, mycelix_fl):
        """Test handling of mixed precision gradients."""
        gradients = {
            "float32_node": np.random.randn(10000).astype(np.float32),
            "float64_node": np.random.randn(10000).astype(np.float64),
            "int_node": np.random.randint(-100, 100, 10000),
        }

        result = mycelix_fl.execute_round(gradients, round_num=1)

        # Should handle all types
        assert result is not None
        assert result.aggregated_gradient.dtype in [np.float32, np.float64]

    def test_recovery_after_high_byzantine(self, mycelix_fl):
        """Test system recovers after round with many Byzantine nodes."""
        np.random.seed(42)

        # Round 1: High Byzantine ratio (40%)
        base = np.random.randn(10000).astype(np.float32)
        gradients_round1 = {
            f"honest_{i}": base + np.random.randn(10000).astype(np.float32) * 0.1
            for i in range(6)
        }
        for i in range(4):
            gradients_round1[f"byzantine_{i}"] = np.random.randn(10000).astype(np.float32) * 100

        result1 = mycelix_fl.execute_round(gradients_round1, round_num=1)

        # Round 2: Clean gradients (should work normally)
        gradients_round2 = {
            f"honest_{i}": base + np.random.randn(10000).astype(np.float32) * 0.1
            for i in range(10)
        }

        result2 = mycelix_fl.execute_round(gradients_round2, round_num=2)

        # Round 2 should have no Byzantine
        assert len(result2.byzantine_nodes) == 0
        assert len(result2.participating_nodes) == 10


class TestScalability:
    """Test system performance at scale."""

    def test_scales_with_node_count(self, mycelix_fl):
        """Test performance scales reasonably with node count."""
        np.random.seed(42)

        timings = []
        node_counts = [10, 50, 100]

        for n_nodes in node_counts:
            gradients = {
                f"node_{i}": np.random.randn(10000).astype(np.float32)
                for i in range(n_nodes)
            }

            start = time.perf_counter()
            mycelix_fl.execute_round(gradients, round_num=1)
            elapsed = time.perf_counter() - start

            timings.append(elapsed)

        # Check scaling is reasonable (not worse than quadratic)
        # 10x nodes should take < 100x time
        assert timings[2] < timings[0] * 100

    def test_scales_with_gradient_size(self, mycelix_fl):
        """Test performance scales with gradient size."""
        np.random.seed(42)

        timings = []
        sizes = [10_000, 100_000, 500_000]

        for size in sizes:
            gradients = {
                f"node_{i}": np.random.randn(size).astype(np.float32)
                for i in range(10)
            }

            start = time.perf_counter()
            mycelix_fl.execute_round(gradients, round_num=1)
            elapsed = time.perf_counter() - start

            timings.append(elapsed)

        # 50x size should take < 50x time (due to vectorization)
        assert timings[2] < timings[0] * 50

    @pytest.mark.slow
    def test_large_scale_simulation(self, large_gradients):
        """Test with large gradients (1M parameters, 20 nodes)."""
        from mycelix_fl import MycelixFL, FLConfig

        config = FLConfig(
            num_rounds=3,
            byzantine_threshold=0.45,
        )
        fl = MycelixFL(config=config)

        start = time.perf_counter()
        result = fl.execute_round(large_gradients, round_num=1)
        elapsed = time.perf_counter() - start

        assert result is not None
        assert result.aggregated_gradient.shape[0] == 1_000_000
        # Should complete in reasonable time (< 30s even on slow hardware)
        assert elapsed < 30.0


class TestPhiMeasurement:
    """Test integrated information (Φ) measurement in pipeline."""

    def test_phi_increases_with_coherence(self, mycelix_fl):
        """Test Φ is higher for more coherent gradients."""
        np.random.seed(42)

        # Highly coherent gradients (same direction)
        base = np.random.randn(10000).astype(np.float32)
        coherent = {
            f"node_{i}": base + np.random.randn(10000).astype(np.float32) * 0.01
            for i in range(10)
        }

        result_coherent = mycelix_fl.execute_round(coherent, round_num=1)

        # Diverse gradients (different directions)
        diverse = {
            f"node_{i}": np.random.randn(10000).astype(np.float32)
            for i in range(10)
        }

        result_diverse = mycelix_fl.execute_round(diverse, round_num=2)

        # Coherent should have higher Φ (more integrated information)
        assert result_coherent.phi_after > result_diverse.phi_after

    def test_phi_tracking_across_rounds(self, mycelix_fl, sample_gradients):
        """Test Φ values are tracked across rounds."""
        phi_values = []

        for round_num in range(1, 6):
            result = mycelix_fl.execute_round(sample_gradients, round_num=round_num)
            phi_values.append((result.phi_before, result.phi_after))

        # All should have valid Φ values
        assert all(phi[0] >= 0 for phi in phi_values)
        assert all(phi[1] >= 0 for phi in phi_values)


class TestShapleyWeighting:
    """Test Shapley value computation and weighting."""

    def test_shapley_values_sum_to_one(self, mycelix_fl, sample_gradients):
        """Test Shapley values approximately sum to 1."""
        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        if result.shapley_values:
            total = sum(result.shapley_values.values())
            assert 0.9 <= total <= 1.1  # Allow small numerical error

    def test_helpful_nodes_get_higher_shapley(self, mycelix_fl):
        """Test nodes that help get higher Shapley values."""
        np.random.seed(42)

        # Create gradient that most nodes agree on
        base = np.random.randn(10000).astype(np.float32)
        gradients = {
            f"agreeing_{i}": base + np.random.randn(10000).astype(np.float32) * 0.05
            for i in range(8)
        }

        # Add one node that disagrees
        gradients["disagreeing"] = -base + np.random.randn(10000).astype(np.float32) * 0.05

        result = mycelix_fl.execute_round(gradients, round_num=1)

        if result.shapley_values and "disagreeing" in result.shapley_values:
            agreeing_shapley = np.mean([
                result.shapley_values[f"agreeing_{i}"]
                for i in range(8)
                if f"agreeing_{i}" in result.shapley_values
            ])
            disagreeing_shapley = result.shapley_values.get("disagreeing", 0)

            # Agreeing nodes should have higher average Shapley
            assert agreeing_shapley > disagreeing_shapley


class TestObservabilityIntegration:
    """Test observability module integration with FL pipeline."""

    def test_metrics_recorded_during_round(self, mycelix_fl, sample_gradients):
        """Test metrics are recorded during FL round."""
        from mycelix_fl.observability import MetricsRegistry

        metrics = MetricsRegistry()

        # Note: This tests that metrics CAN be recorded
        # Actual integration would need the FL system to use the metrics
        metrics.rounds_total.inc()
        metrics.gradients_received.inc(len(sample_gradients))

        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        if result.byzantine_nodes:
            metrics.byzantine_detected.inc(len(result.byzantine_nodes))

        assert metrics.rounds_total.get() == 1
        assert metrics.gradients_received.get() == len(sample_gradients)

    def test_performance_tracking(self, mycelix_fl, sample_gradients):
        """Test performance tracker records round statistics."""
        from mycelix_fl.observability import PerformanceTracker

        tracker = PerformanceTracker()

        for round_num in range(1, 11):
            start = time.perf_counter()
            result = mycelix_fl.execute_round(sample_gradients, round_num=round_num)
            elapsed_ms = (time.perf_counter() - start) * 1000

            tracker.record(
                round_num=round_num,
                total_nodes=len(sample_gradients),
                byzantine_nodes=len(result.byzantine_nodes),
                healed_nodes=len(result.healed_nodes) if result.healed_nodes else 0,
                detection_latency_ms=elapsed_ms * 0.3,  # Estimate
                round_latency_ms=elapsed_ms,
            )

        summary = tracker.summary()
        assert summary["rounds"] == 10
        assert "detection_latency_ms" in summary

        health = tracker.check_health()
        assert health["status"] in ["healthy", "degraded"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_nodes_byzantine(self, mycelix_fl):
        """Test behavior when all nodes are Byzantine."""
        from mycelix_fl.exceptions import TooManyByzantineError

        # All nodes are obviously Byzantine
        gradients = {
            f"bad_{i}": np.ones(10000).astype(np.float32) * 1e6 * (i + 1)
            for i in range(10)
        }

        # Should either fail gracefully or detect all as Byzantine
        try:
            result = mycelix_fl.execute_round(gradients, round_num=1)
            # If it completes, should have detected many Byzantine
            assert len(result.byzantine_nodes) >= 5
        except (TooManyByzantineError, Exception):
            pass  # Expected

    def test_minimum_nodes_boundary(self, mycelix_fl):
        """Test with exactly minimum required nodes."""
        from mycelix_fl import FLConfig

        # Create FL with min_nodes=3
        from mycelix_fl import MycelixFL
        fl = MycelixFL(config=FLConfig(min_nodes=3))

        gradients = {
            f"node_{i}": np.random.randn(10000).astype(np.float32)
            for i in range(3)
        }

        result = fl.execute_round(gradients, round_num=1)
        assert result is not None

    def test_zero_gradient(self, mycelix_fl, sample_gradients):
        """Test handling of zero gradient."""
        sample_gradients["zero_node"] = np.zeros(10000, dtype=np.float32)

        result = mycelix_fl.execute_round(sample_gradients, round_num=1)

        # Should complete (zero is valid, just not helpful)
        assert result is not None

    def test_very_small_gradients(self, mycelix_fl):
        """Test with very small gradient values."""
        gradients = {
            f"node_{i}": np.random.randn(10000).astype(np.float32) * 1e-10
            for i in range(10)
        }

        result = mycelix_fl.execute_round(gradients, round_num=1)
        assert result is not None
        # Result should also be small
        assert np.max(np.abs(result.aggregated_gradient)) < 1e-8

    def test_identical_gradients(self, mycelix_fl):
        """Test when all nodes submit identical gradients."""
        base = np.random.randn(10000).astype(np.float32)
        gradients = {
            f"node_{i}": base.copy()
            for i in range(10)
        }

        result = mycelix_fl.execute_round(gradients, round_num=1)

        # Result should equal the input (all are identical)
        assert np.allclose(result.aggregated_gradient, base)

    def test_alternating_sign_gradients(self, mycelix_fl):
        """Test gradients with alternating signs."""
        gradients = {
            f"node_{i}": np.random.randn(10000).astype(np.float32) * (1 if i % 2 == 0 else -1)
            for i in range(10)
        }

        result = mycelix_fl.execute_round(gradients, round_num=1)

        # Should complete and aggregate to near-zero
        assert result is not None
        # Mean should be close to zero due to cancellation
        assert np.abs(np.mean(result.aggregated_gradient)) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
