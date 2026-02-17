"""
Tests for v1.2.0 new modules (Privacy, Benchmarks, Distributed, CLI)

Author: Luminous Dynamics
Date: December 31, 2025
"""

import pytest
import numpy as np


class TestPrivacyModule:
    """Test differential privacy module."""

    def test_dp_config_validation(self):
        """Test DPConfig validates parameters."""
        from mycelix_fl import DPConfig

        # Valid config
        config = DPConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        assert config.epsilon == 1.0

        # Invalid epsilon
        with pytest.raises(ValueError):
            DPConfig(epsilon=-1.0)

        # Invalid delta
        with pytest.raises(ValueError):
            DPConfig(delta=2.0)

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        from mycelix_fl import clip_gradients

        gradient = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        clipped, original_norm = clip_gradients(gradient, max_norm=1.0)

        assert original_norm == 5.0
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_differential_privacy_privatize(self):
        """Test privatization adds noise."""
        from mycelix_fl import DifferentialPrivacy, DPConfig

        config = DPConfig(epsilon=1.0, max_grad_norm=1.0)
        dp = DifferentialPrivacy(config)

        gradient = np.ones(100, dtype=np.float32)
        privatized, metadata = dp.privatize_gradient(gradient)

        # Should be different due to noise
        assert not np.allclose(gradient, privatized)
        assert "noise_scale" in metadata
        assert "current_epsilon" in metadata

    def test_privacy_accountant(self):
        """Test privacy budget tracking."""
        from mycelix_fl import PrivacyAccountant

        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=1.0,
            delta=1e-5,
        )

        assert accountant.get_epsilon() == 0.0

        # Record some steps
        for _ in range(10):
            accountant.step()

        assert accountant.get_epsilon() > 0


class TestBenchmarkModule:
    """Test benchmark module."""

    def test_benchmark_result(self):
        """Test BenchmarkResult statistics."""
        from mycelix_fl import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            times_ms=[10.0, 12.0, 11.0, 13.0, 14.0],
        )

        assert result.mean_ms == 12.0
        assert result.min_ms == 10.0
        assert result.max_ms == 14.0
        assert 10.0 <= result.p50_ms <= 14.0

    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite creation."""
        from mycelix_fl import BenchmarkSuite

        suite = BenchmarkSuite(warmup_rounds=1, benchmark_rounds=2)
        assert suite.warmup_rounds == 1
        assert suite.benchmark_rounds == 2


class TestDistributedModule:
    """Test distributed/async module."""

    def test_async_fl_config(self):
        """Test AsyncFLConfig."""
        from mycelix_fl import AsyncFLConfig

        config = AsyncFLConfig(
            round_timeout_seconds=30.0,
            min_nodes_per_round=5,
        )
        assert config.round_timeout_seconds == 30.0
        assert config.min_nodes_per_round == 5

    def test_gradient_message(self):
        """Test GradientMessage creation."""
        from mycelix_fl import GradientMessage

        gradient = np.random.randn(100).astype(np.float32)
        msg = GradientMessage(
            node_id="node_1",
            gradient=gradient,
            round_num=1,
        )

        assert msg.node_id == "node_1"
        assert msg.round_num == 1
        assert msg.gradient.shape == (100,)

    def test_fl_coordinator_creation(self):
        """Test FLCoordinator creation."""
        from mycelix_fl import FLCoordinator, CoordinatorConfig

        config = CoordinatorConfig(num_rounds=10, min_nodes=3)
        coord = FLCoordinator(config)

        assert coord.config.num_rounds == 10
        assert len(coord.registered_nodes) == 0

    def test_coordinator_node_registration(self):
        """Test node registration."""
        from mycelix_fl import FLCoordinator

        coord = FLCoordinator()

        assert coord.register_node("node_1") is True
        assert coord.register_node("node_2") is True
        assert "node_1" in coord.registered_nodes
        assert "node_2" in coord.registered_nodes

        coord.unregister_node("node_1")
        assert "node_1" not in coord.registered_nodes


class TestCLIModule:
    """Test CLI module."""

    def test_cli_imports(self):
        """Test CLI imports."""
        from mycelix_fl import cli_main, cli_app
        assert callable(cli_main)
        assert callable(cli_app)

    def test_cli_help(self):
        """Test CLI --help doesn't crash."""
        from mycelix_fl.cli.main import app

        # Should not raise
        try:
            app(["--help"])
        except SystemExit as e:
            # argparse exits with 0 for --help
            assert e.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
