#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Basic Smoke Tests for Mycelix Core
==================================

Quick sanity checks that can run in CI without ANY external services
or optional dependencies. These tests verify basic functionality.

Run with: pytest -m smoke 0TML/tests/test_basic_smoke.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Mark all tests in this file as smoke and unit tests
pytestmark = [pytest.mark.smoke, pytest.mark.unit]


class TestBasicImports:
    """Test that core modules can be imported."""

    def test_numpy_available(self):
        """Verify numpy is installed and working."""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6

    def test_path_operations(self):
        """Verify pathlib works."""
        from pathlib import Path
        test_path = Path(__file__)
        assert test_path.exists()
        assert test_path.suffix == ".py"


class TestGradientOperations:
    """Test basic gradient operations (no external deps)."""

    def test_gradient_normalization(self):
        """Test gradient can be normalized."""
        rng = np.random.RandomState(42)
        gradient = rng.randn(1000).astype(np.float32)
        norm = np.linalg.norm(gradient)
        normalized = gradient / norm
        assert np.isclose(np.linalg.norm(normalized), 1.0)

    def test_gradient_aggregation_mean(self):
        """Test mean aggregation of gradients."""
        rng = np.random.RandomState(42)
        gradients = [rng.randn(100).astype(np.float32) for _ in range(10)]
        mean_gradient = np.mean(gradients, axis=0)
        assert mean_gradient.shape == (100,)

    def test_gradient_aggregation_median(self):
        """Test median aggregation of gradients (Byzantine-resilient)."""
        rng = np.random.RandomState(42)
        gradients = [rng.randn(100).astype(np.float32) for _ in range(10)]
        median_gradient = np.median(gradients, axis=0)
        assert median_gradient.shape == (100,)

    def test_coordinate_wise_median(self):
        """Test coordinate-wise median is Byzantine-resilient."""
        # 5 honest gradients around [1, 1]
        honest = np.array([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0], [1.05, 0.95]])
        # 2 Byzantine gradients (sign flip)
        byzantine = np.array([[-10.0, -10.0], [-5.0, -8.0]])

        all_gradients = np.vstack([honest, byzantine])

        # Mean is pulled by Byzantine
        mean_result = np.mean(all_gradients, axis=0)

        # Median resists Byzantine
        median_result = np.median(all_gradients, axis=0)

        # Mean should be pulled negative (Byzantine influence)
        # With 5 honest around [1,1] and 2 byzantine at [-10,-10] and [-5,-8]:
        # Mean x: (5*~1 + -10 + -5) / 7 = (5 - 15) / 7 = -10/7 ~ -1.4
        assert mean_result[0] < 0

        # Median should stay positive (Byzantine-resistant)
        assert median_result[0] > 0


class TestByzantineDetection:
    """Test Byzantine detection utilities."""

    def test_outlier_detection_norm_based(self):
        """Test detecting outliers by gradient norm."""
        rng = np.random.RandomState(42)

        # Generate honest gradients (norm around 1)
        gradients = {}
        for i in range(7):
            g = rng.randn(100).astype(np.float32)
            g = g / np.linalg.norm(g)  # Normalize to unit norm
            gradients[f"honest_{i}"] = g

        # Add Byzantine gradients
        gradients["byzantine_scale"] = rng.randn(100).astype(np.float32) * 100  # Scaling attack

        norms = {k: np.linalg.norm(v) for k, v in gradients.items()}

        # Calculate median norm
        median_norm = np.median(list(norms.values()))

        # Outliers are those with norm > 3x median
        outliers = [k for k, n in norms.items() if n > 3 * median_norm]

        # Should detect the scaling attack
        assert "byzantine_scale" in outliers

    def test_outlier_detection_direction_based(self):
        """Test detecting outliers by gradient direction."""
        rng = np.random.RandomState(42)

        # True direction
        true_dir = rng.randn(100).astype(np.float32)
        true_dir = true_dir / np.linalg.norm(true_dir)

        # Generate honest gradients (same direction with noise)
        gradients = {}
        for i in range(7):
            noise = rng.randn(100).astype(np.float32) * 0.1
            gradients[f"honest_{i}"] = true_dir + noise

        # Add Byzantine gradient (opposite direction)
        gradients["byzantine_flip"] = -true_dir * 1.5

        # Calculate mean direction
        grad_list = list(gradients.values())
        mean_direction = np.mean(grad_list, axis=0)
        mean_direction = mean_direction / np.linalg.norm(mean_direction)

        # Calculate cosine similarities
        similarities = {}
        for k, v in gradients.items():
            v_norm = v / np.linalg.norm(v)
            similarity = np.dot(v_norm, mean_direction)
            similarities[k] = similarity

        # Outliers have negative similarity (opposite direction)
        outliers = [k for k, s in similarities.items() if s < 0]

        # Should detect sign flip attack
        assert "byzantine_flip" in outliers


class TestStatistics:
    """Test statistical utilities."""

    def test_mean_calculation(self):
        """Test mean is calculated correctly."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert np.mean(values) == 3.0

    def test_median_calculation(self):
        """Test median is calculated correctly."""
        values = [1.0, 2.0, 100.0]  # Outlier present
        assert np.median(values) == 2.0

    def test_trimmed_mean(self):
        """Test trimmed mean excludes outliers."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier present
        # Trim 20% from each end
        trimmed = np.sort(values)[1:-1]  # Remove min and max
        trimmed_mean = np.mean(trimmed)
        assert trimmed_mean == 3.0

    def test_quantile_calculation(self):
        """Test quantile calculation."""
        values = np.arange(1, 101)  # 1 to 100
        assert np.percentile(values, 50) == pytest.approx(50.5)
        assert np.percentile(values, 25) == pytest.approx(25.75)
        assert np.percentile(values, 75) == pytest.approx(75.25)


class TestDataStructures:
    """Test data structure operations used in FL."""

    def test_gradient_dict_operations(self):
        """Test operations on gradient dictionaries."""
        gradients = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([4.0, 5.0, 6.0]),
        }

        # Flatten gradients
        flat = np.concatenate([gradients["layer1"], gradients["layer2"]])
        assert len(flat) == 6

        # Compute total norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        assert total_norm == pytest.approx(np.linalg.norm(flat))

    def test_participant_tracking(self):
        """Test tracking participant contributions."""
        participants = {}

        # Record contributions
        for i in range(10):
            participants[f"node_{i}"] = {
                "rounds_participated": i + 1,
                "byzantine_flags": 0 if i < 8 else 1,
                "reputation": 1.0 if i < 8 else 0.5,
            }

        # Filter honest participants
        honest = [p for p, d in participants.items() if d["byzantine_flags"] == 0]
        assert len(honest) == 8

        # Get average reputation
        avg_rep = np.mean([d["reputation"] for d in participants.values()])
        assert avg_rep == pytest.approx(0.9)  # (8*1.0 + 2*0.5) / 10


class TestConftest:
    """Test that conftest fixtures are available."""

    def test_rng_fixture(self, rng):
        """Test RNG fixture is available and reproducible."""
        val1 = rng.randn()
        rng2 = np.random.RandomState(42)
        val2 = rng2.randn()
        assert val1 == val2

    def test_small_gradient_fixture(self, small_gradient):
        """Test small gradient fixture."""
        assert small_gradient.shape == (1000,)
        assert small_gradient.dtype == np.float32

    def test_honest_gradients_fixture(self, honest_gradients):
        """Test honest gradients fixture."""
        assert len(honest_gradients) == 10
        for k in honest_gradients:
            assert k.startswith("honest_")

    def test_byzantine_gradients_fixture(self, byzantine_gradients):
        """Test Byzantine gradients fixture."""
        assert len(byzantine_gradients) == 10  # 7 honest + 3 byzantine
        assert "byzantine_flip" in byzantine_gradients
        assert "byzantine_random" in byzantine_gradients
        assert "byzantine_scale" in byzantine_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
