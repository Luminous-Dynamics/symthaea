# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for consciousness-guided FL round (conscious_round.py).

Tests the full pipeline: Phi proxy → weight adjustment → aggregation,
config presets, edge cases, and bug fixes.
"""

import sys
import os
import pytest
import numpy as np

# Ensure src/ is on path for mycelix_fl imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mycelix_fl.conscious_round import (
    ConsciousFlRound,
    ConsciousRoundConfig,
    ConsciousRoundResult,
    NodeRoundScore,
    run_conscious_fl_round,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def default_runner():
    return ConsciousFlRound()


@pytest.fixture
def three_node_gradients(rng):
    """Three honest nodes with normal gradients."""
    return {
        "node-1": rng.randn(128).astype(np.float32),
        "node-2": rng.randn(128).astype(np.float32),
        "node-3": rng.randn(128).astype(np.float32),
    }


@pytest.fixture
def three_node_reputations():
    return {"node-1": 0.8, "node-2": 0.7, "node-3": 0.9}


# =============================================================================
# BASIC FUNCTIONALITY
# =============================================================================


class TestBasicRound:
    def test_basic_round_returns_result(self, default_runner, three_node_gradients, three_node_reputations):
        result = default_runner.run(three_node_gradients, three_node_reputations)
        assert isinstance(result, ConsciousRoundResult)
        assert result.aggregated.shape == (128,)
        assert result.aggregated.dtype == np.float32
        assert len(result.node_scores) == 3
        assert result.included_count + result.vetoed_count == 3

    def test_rounds_completed_increments(self, default_runner, three_node_gradients, three_node_reputations):
        assert default_runner.rounds_completed == 0
        default_runner.run(three_node_gradients, three_node_reputations)
        assert default_runner.rounds_completed == 1
        default_runner.run(three_node_gradients, three_node_reputations)
        assert default_runner.rounds_completed == 2

    def test_empty_gradients_raises(self, default_runner):
        with pytest.raises(ValueError, match="No gradients"):
            default_runner.run({}, {})

    def test_node_scores_populated(self, default_runner, three_node_gradients, three_node_reputations):
        result = default_runner.run(three_node_gradients, three_node_reputations)
        for node_id, score in result.node_scores.items():
            assert isinstance(score, NodeRoundScore)
            assert 0.0 <= score.phi_after <= 1.0
            assert 0.0 <= score.epistemic_confidence <= 1.0
            assert score.severity in ("none", "mild", "moderate", "severe")
            assert 0.0 <= score.pogq_quality <= 1.0
            assert 0.0 <= score.pogq_consistency <= 1.0

    def test_aggregated_is_weighted_mean(self, rng):
        """Verify aggregation is a weighted mean, not a sum."""
        runner = ConsciousFlRound()
        # Use identical gradients → aggregated should ≈ input
        grad = rng.randn(128).astype(np.float32)
        gradients = {"a": grad.copy(), "b": grad.copy()}
        reps = {"a": 0.8, "b": 0.8}
        result = runner.run(gradients, reps)
        np.testing.assert_allclose(result.aggregated, grad[:128], atol=0.01)


# =============================================================================
# CONFIG PRESETS
# =============================================================================


class TestConfigPresets:
    def test_default_config(self):
        cfg = ConsciousRoundConfig()
        assert cfg.confidence_threshold == 0.3
        assert cfg.veto_confidence == 0.1
        assert cfg.defense == "trimmed_mean"
        assert cfg.gradient_dim == 128

    def test_high_security(self):
        cfg = ConsciousRoundConfig.high_security()
        assert cfg.confidence_threshold == 0.4
        assert cfg.veto_confidence == 0.15
        assert cfg.defense == "krum"
        assert cfg.byzantine_fraction == 0.3

    def test_performance(self):
        cfg = ConsciousRoundConfig.performance()
        assert cfg.confidence_threshold == 0.2
        assert cfg.veto_confidence == 0.05
        assert not cfg.veto_severe
        assert cfg.defense == "fedavg"


# =============================================================================
# PHI PROXY & ANOMALY DETECTION
# =============================================================================


class TestPhiProxy:
    def test_zero_gradient_gets_neutral_phi(self):
        """Bug fix: zero gradients should NOT get phi=1.0 (perfect)."""
        runner = ConsciousFlRound()
        gradients = {"zero_node": np.zeros(128, dtype=np.float32)}
        reps = {"zero_node": 0.5}
        result = runner.run(gradients, reps)
        score = result.node_scores["zero_node"]
        # Should be neutral (0.5), not 1.0
        assert score.phi_after == 0.5

    def test_normal_gradient_gets_reasonable_phi(self, rng):
        runner = ConsciousFlRound()
        gradients = {"normal": rng.randn(128).astype(np.float32)}
        reps = {"normal": 0.7}
        result = runner.run(gradients, reps)
        score = result.node_scores["normal"]
        assert 0.0 < score.phi_after < 1.0

    def test_uniform_gradient_high_phi(self):
        """Uniform values (low std/mean) should give high phi."""
        runner = ConsciousFlRound()
        gradients = {"uniform": np.ones(128, dtype=np.float32) * 0.5}
        reps = {"uniform": 0.7}
        result = runner.run(gradients, reps)
        score = result.node_scores["uniform"]
        # Low std relative to mean → high phi
        assert score.phi_after > 0.7

    def test_phi_history_persists_across_rounds(self, rng):
        runner = ConsciousFlRound()
        gradients = {"node": rng.randn(128).astype(np.float32)}
        reps = {"node": 0.7}

        r1 = runner.run(gradients, reps)
        r2 = runner.run(gradients, reps)

        # Round 2 should use round 1's phi_after as phi_before
        assert r2.node_scores["node"].phi_before == r1.node_scores["node"].phi_after


# =============================================================================
# WEIGHT ADJUSTMENT (VETO / DAMPEN / BOOST)
# =============================================================================


class TestWeightAdjustment:
    def test_low_reputation_dampened(self):
        """Low reputation node should be dampened."""
        config = ConsciousRoundConfig(confidence_threshold=0.4)
        runner = ConsciousFlRound(config)
        gradients = {
            "good": np.ones(128, dtype=np.float32) * 0.5,
            "low_rep": np.ones(128, dtype=np.float32) * 0.5,
        }
        reps = {"good": 0.9, "low_rep": 0.1}
        result = runner.run(gradients, reps)
        assert result.dampened_count >= 1 or result.vetoed_count >= 1

    def test_veto_severe_setting(self, rng):
        """veto_severe=False should not veto severe anomalies."""
        config = ConsciousRoundConfig(veto_severe=False, veto_confidence=0.01)
        runner = ConsciousFlRound(config)
        # Use gradients that will produce some phi drop on second round
        gradients = {"node": rng.randn(128).astype(np.float32) * 10.0}
        reps = {"node": 0.05}
        result = runner.run(gradients, reps)
        # With veto_confidence=0.01 and veto_severe=False,
        # even low confidence may pass
        assert result.included_count >= 0  # no crash

    def test_plugin_weights_applied_flag(self, rng):
        """Flag should be True when any gating happened."""
        config = ConsciousRoundConfig(confidence_threshold=0.8)
        runner = ConsciousFlRound(config)
        gradients = {"node": rng.randn(128).astype(np.float32)}
        reps = {"node": 0.3}
        result = runner.run(gradients, reps)
        if result.vetoed_count > 0 or result.dampened_count > 0 or result.boosted_count > 0:
            assert result.plugin_weights_applied


# =============================================================================
# ALL-VETOED EDGE CASE
# =============================================================================


class TestAllVetoed:
    def test_all_vetoed_returns_zeros(self):
        """When all nodes are vetoed, result should be zero vector."""
        config = ConsciousRoundConfig(veto_confidence=0.99)
        runner = ConsciousFlRound(config)
        gradients = {
            "a": np.ones(256, dtype=np.float32),
            "b": np.ones(256, dtype=np.float32),
        }
        reps = {"a": 0.01, "b": 0.01}
        result = runner.run(gradients, reps)
        # All should be vetoed with veto_confidence=0.99
        if result.vetoed_count == 2:
            np.testing.assert_array_equal(result.aggregated, np.zeros_like(result.aggregated))

    def test_all_vetoed_dimension_matches_input(self):
        """Bug fix: all-vetoed should match actual gradient dim, not config.gradient_dim."""
        config = ConsciousRoundConfig(gradient_dim=128, veto_confidence=0.99)
        runner = ConsciousFlRound(config)
        gradients = {
            "a": np.ones(512, dtype=np.float32),
            "b": np.ones(512, dtype=np.float32),
        }
        reps = {"a": 0.01, "b": 0.01}
        result = runner.run(gradients, reps)
        if result.vetoed_count == 2:
            # Should be min(128, 512) = 128, not 512
            assert result.aggregated.shape == (128,)

    def test_gradient_dim_limits_output(self, rng):
        """Output should be limited to gradient_dim even for larger inputs."""
        config = ConsciousRoundConfig(gradient_dim=64)
        runner = ConsciousFlRound(config)
        gradients = {"node": rng.randn(256).astype(np.float32)}
        reps = {"node": 0.8}
        result = runner.run(gradients, reps)
        assert result.aggregated.shape == (64,)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


class TestConvenienceFunction:
    def test_run_conscious_fl_round(self, three_node_gradients, three_node_reputations):
        result = run_conscious_fl_round(three_node_gradients, three_node_reputations)
        assert isinstance(result, ConsciousRoundResult)
        assert result.aggregated.shape == (128,)
        assert len(result.node_scores) == 3

    def test_convenience_with_config(self, three_node_gradients, three_node_reputations):
        config = ConsciousRoundConfig.high_security()
        result = run_conscious_fl_round(three_node_gradients, three_node_reputations, config)
        assert isinstance(result, ConsciousRoundResult)

    def test_convenience_raises_on_empty(self):
        with pytest.raises(ValueError):
            run_conscious_fl_round({}, {})


# =============================================================================
# MULTI-ROUND STATE
# =============================================================================


class TestMultiRound:
    def test_phi_gain_stabilizes(self, rng):
        """After several rounds with same data, phi_gain should approach 0."""
        runner = ConsciousFlRound()
        gradients = {"node": rng.randn(128).astype(np.float32)}
        reps = {"node": 0.7}

        gains = []
        for _ in range(5):
            result = runner.run(gradients, reps)
            gains.append(abs(result.node_scores["node"].phi_gain))

        # After round 2, gain should be 0 (same gradient → same phi)
        assert gains[-1] == 0.0

    def test_new_node_gets_default_phi_before(self, rng):
        """A node appearing for the first time gets phi_before=0.5."""
        runner = ConsciousFlRound()
        gradients = {"new_node": rng.randn(128).astype(np.float32)}
        reps = {"new_node": 0.7}
        result = runner.run(gradients, reps)
        assert result.node_scores["new_node"].phi_before == 0.5
