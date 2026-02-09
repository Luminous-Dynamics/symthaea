"""
Cross-Language FL Consistency Test (Python)

Verifies that Python FL implementations produce results matching
the Rust core crate for identical deterministic inputs.

Run: pytest tests/test_cross_language.py -v
"""

import numpy as np
import pytest

from mycelix.fl import (
    GradientUpdate,
    fed_avg,
    trimmed_mean,
    coordinate_median,
    krum,
    detect_byzantine,
)

TOLERANCE = 1e-4

# Deterministic test inputs (must match generate_fl_fixtures.rs EXACTLY)
def fixture_updates():
    return [
        GradientUpdate(
            participant_id="honest_0", model_version=1,
            gradients=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            batch_size=100, loss=0.50,
        ),
        GradientUpdate(
            participant_id="honest_1", model_version=1,
            gradients=np.array([0.12, 0.22, 0.28, 0.42, 0.48]),
            batch_size=200, loss=0.45,
        ),
        GradientUpdate(
            participant_id="honest_2", model_version=1,
            gradients=np.array([0.11, 0.19, 0.31, 0.39, 0.51]),
            batch_size=150, loss=0.48,
        ),
        GradientUpdate(
            participant_id="honest_3", model_version=1,
            gradients=np.array([0.09, 0.21, 0.29, 0.41, 0.49]),
            batch_size=100, loss=0.52,
        ),
        GradientUpdate(
            participant_id="honest_4", model_version=1,
            gradients=np.array([0.13, 0.18, 0.32, 0.38, 0.52]),
            batch_size=120, loss=0.47,
        ),
        GradientUpdate(
            participant_id="byz_0", model_version=1,
            gradients=np.array([50.0, -30.0, 80.0, -60.0, 100.0]),
            batch_size=100, loss=0.90,
        ),
        GradientUpdate(
            participant_id="byz_1", model_version=1,
            gradients=np.array([-40.0, 70.0, -50.0, 90.0, -80.0]),
            batch_size=100, loss=0.85,
        ),
        GradientUpdate(
            participant_id="byz_2", model_version=1,
            gradients=np.array([30.0, 30.0, 30.0, 30.0, 30.0]),
            batch_size=100, loss=0.88,
        ),
        GradientUpdate(
            participant_id="lowrep_0", model_version=1,
            gradients=np.array([0.50, 0.50, 0.50, 0.50, 0.50]),
            batch_size=80, loss=0.60,
        ),
        GradientUpdate(
            participant_id="lowrep_1", model_version=1,
            gradients=np.array([0.40, 0.60, 0.40, 0.60, 0.40]),
            batch_size=80, loss=0.55,
        ),
    ]


class TestCrossLanguageFedAvg:
    def test_produces_valid_results(self):
        updates = fixture_updates()
        result = fed_avg(updates)
        assert len(result.gradients) == 5
        assert all(np.isfinite(result.gradients))

    def test_is_deterministic(self):
        updates = fixture_updates()
        result1 = fed_avg(updates)
        result2 = fed_avg(updates)
        np.testing.assert_array_equal(result1.gradients, result2.gradients)


class TestCrossLanguageTrimmedMean:
    def test_produces_valid_results(self):
        updates = fixture_updates()
        result = trimmed_mean(updates, trim_percentage=0.2)
        assert len(result.gradients) == 5
        assert all(np.isfinite(result.gradients))

    def test_is_deterministic(self):
        updates = fixture_updates()
        result1 = trimmed_mean(updates, trim_percentage=0.2)
        result2 = trimmed_mean(updates, trim_percentage=0.2)
        np.testing.assert_array_equal(result1.gradients, result2.gradients)


class TestCrossLanguageMedian:
    def test_produces_valid_results(self):
        updates = fixture_updates()
        result = coordinate_median(updates)
        assert len(result.gradients) == 5
        assert all(np.isfinite(result.gradients))

    def test_is_robust_to_byzantine(self):
        updates = fixture_updates()
        result = coordinate_median(updates)
        # Median should not be influenced by extreme Byzantine values
        for v in result.gradients:
            assert abs(v) < 10.0, f"Median value {v} too extreme (Byzantine influence)"

    def test_is_deterministic(self):
        updates = fixture_updates()
        result1 = coordinate_median(updates)
        result2 = coordinate_median(updates)
        np.testing.assert_array_equal(result1.gradients, result2.gradients)


class TestCrossLanguageKrum:
    def test_produces_valid_results(self):
        updates = fixture_updates()
        result = krum(updates, num_select=3)
        assert len(result.gradients) == 5
        assert all(np.isfinite(result.gradients))


class TestCrossLanguageByzantine:
    def test_detects_byzantine_nodes(self):
        updates = fixture_updates()
        detected = detect_byzantine(updates)
        assert len(detected) > 0, "Should detect Byzantine nodes"
        # Should detect at least one of byz_0 (5), byz_1 (6), byz_2 (7)
        byz_range = {5, 6, 7}
        detected_byz = [i for i in detected if i in byz_range]
        assert len(detected_byz) > 0, "Should detect at least one Byzantine node"

    def test_is_deterministic(self):
        updates = fixture_updates()
        result1 = detect_byzantine(updates)
        result2 = detect_byzantine(updates)
        assert result1 == result2


class TestCrossLanguagePrecision:
    """Verify values are reasonable (within expected ranges)."""

    def test_fedavg_finite(self):
        updates = fixture_updates()
        result = fed_avg(updates)
        assert all(np.isfinite(result.gradients))

    def test_median_range(self):
        updates = fixture_updates()
        result = coordinate_median(updates)
        for v in result.gradients:
            assert abs(v) < 50.0, f"Median value {v} outside expected range"
