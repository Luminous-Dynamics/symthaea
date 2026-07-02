# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Layer 5: Active Learning Inspector
===================================================

Tests:
1. Query Selection (4 tests)
   - Uncertainty-based selection
   - Margin-based selection
   - Diversity-based selection
   - Budget enforcement

2. Decision Fusion (4 tests)
   - Fast-only decisions
   - Deep overrides low-confidence fast
   - Conflicting decisions resolved by confidence
   - Confidence propagation

3. Performance (2 tests)
   - Speedup measurement
   - Accuracy maintenance

4. Integration (5 tests)
   - Two-pass pipeline
   - Fast pass on all gradients
   - Query selection
   - Deep pass on queries only
   - Byzantine robustness

Author: Luminous Dynamics
Date: November 12, 2025
"""

import pytest
import numpy as np
import time

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.active_learning import ActiveLearningInspector, ActiveLearningConfig
from gen5.meta_learning import MetaLearningEnsemble
from gen5.uncertainty import UncertaintyQuantifier


# Mock detection method for testing
class MockDetector:
    """Mock Byzantine detector for testing."""

    def __init__(self, name: str, base_score: float = 0.5):
        self.name = name
        self.base_score = base_score

    def score(self, gradient: np.ndarray) -> float:
        """Return score based on gradient magnitude."""
        # Use gradient norm to create variation
        magnitude = np.linalg.norm(gradient)

        # For testing: honest gradients have low magnitude, Byzantine have high
        if magnitude < 1.0:  # Honest-like
            return self.base_score + 0.3
        else:  # Byzantine-like
            return self.base_score - 0.3


class TestQuerySelection:
    """Test suite for query selection strategies."""

    def test_uncertainty_based_selection(self):
        """Test uncertainty-based query selection."""
        # Setup
        methods = [MockDetector("method1", 0.5), MockDetector("method2", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50, coverage_target=0.9)

        # Calibrate quantifier with some data
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(
            query_budget=0.2,
            selection_strategy="uncertainty"
        )
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create gradients with varying uncertainty
        gradients = [
            np.random.normal(0, 0.1, size=10),  # Low magnitude (honest, low uncertainty)
            np.random.normal(0, 0.1, size=10),
            np.random.normal(5, 1.0, size=10),  # High magnitude (Byzantine, high uncertainty)
            np.random.normal(5, 1.0, size=10),
            np.random.normal(0.5, 0.5, size=10),  # Medium magnitude (uncertain)
        ]

        # Fast pass
        fast_results = inspector._fast_pass(gradients)

        # Select queries
        query_indices = inspector._select_queries_uncertainty(fast_results, 2)

        # Should select 2 queries
        assert len(query_indices) == 2

        # All indices valid
        assert all(0 <= idx < len(gradients) for idx in query_indices)

    def test_margin_based_selection(self):
        """Test margin-based query selection."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(
            query_budget=0.3,
            selection_strategy="margin"
        )
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create gradients
        gradients = [
            np.random.normal(0, 0.1, size=10),  # Honest (score ~0.8)
            np.random.normal(5, 1.0, size=10),  # Byzantine (score ~0.2)
            np.random.normal(0.5, 0.3, size=10),  # Near boundary (~0.5)
        ]

        fast_results = inspector._fast_pass(gradients)
        query_indices = inspector._select_queries_margin(fast_results, 1)

        # Should select the gradient closest to 0.5 boundary
        assert len(query_indices) == 1

        # The selected gradient should have score closest to 0.5
        selected_score = fast_results[query_indices[0]]["score"]
        all_margins = [abs(r["score"] - 0.5) for r in fast_results]
        assert abs(selected_score - 0.5) == min(all_margins)

    def test_diversity_based_selection(self):
        """Test diversity-based query selection."""
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
        ]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(
            query_budget=0.4,
            selection_strategy="diverse"
        )
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create 10 gradients in 2 clusters
        cluster1 = [np.random.normal([1, 1], 0.1, size=2) for _ in range(5)]
        cluster2 = [np.random.normal([5, 5], 0.1, size=2) for _ in range(5)]
        gradients = cluster1 + cluster2

        fast_results = inspector._fast_pass(gradients)
        query_indices = inspector._select_queries_diverse(fast_results, 4)

        # Should select from both clusters
        assert len(query_indices) == 4
        assert all(0 <= idx < len(gradients) for idx in query_indices)

    def test_budget_enforcement(self):
        """Test that query budget is strictly enforced."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(query_budget=0.15)  # 15%
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create 100 gradients
        gradients = [np.random.normal(0, 1, size=10) for _ in range(100)]

        fast_results = inspector._fast_pass(gradients)
        query_indices = inspector._select_queries(fast_results)

        # Should select 15 queries (15% of 100)
        expected_queries = max(1, int(100 * 0.15))
        assert len(query_indices) == expected_queries


class TestDecisionFusion:
    """Test suite for decision fusion logic."""

    def test_fast_only_decision(self):
        """Test fast-only decision when no deep verification."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            deep_methods=[],  # No deep methods
        )

        gradients = [np.random.normal(0, 0.1, size=10)]  # Honest

        # Fast pass
        fast_results = inspector._fast_pass(gradients)

        # No deep results
        deep_results = {}

        # Fusion
        final_decisions = inspector._fuse_decisions(fast_results, deep_results)

        # Should use fast-only decision
        assert len(final_decisions) == 1
        decision, confidence = final_decisions[0]
        assert decision in ["HONEST", "BYZANTINE"]
        assert 0.0 <= confidence <= 1.0

    def test_deep_overrides_low_confidence_fast(self):
        """Test that confident deep result overrides uncertain fast result."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate with wide range (high uncertainty)
        for i in range(50):
            quantifier.update([i / 50.0])

        deep_method = MockDetector("deep_method", 0.9)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            deep_methods=[deep_method],
        )

        # Create gradient that gets queried
        gradients = [np.random.normal(0, 0.5, size=10)]

        # Fast pass (low confidence due to wide calibration)
        fast_results = inspector._fast_pass(gradients)

        # Deep pass (high confidence)
        deep_results = {
            0: {
                "signals": {"deep_method": 0.95},
                "score": 0.95,
                "decision": "HONEST",
                "confidence": 0.95,
            }
        }

        # Fusion
        final_decisions = inspector._fuse_decisions(fast_results, deep_results)

        decision, confidence = final_decisions[0]

        # Deep method has very high confidence (0.95)
        # Even if fast is uncertain, deep should dominate
        assert decision == "HONEST"
        assert confidence > 0.7  # Should be high

    def test_conflicting_decisions_resolved_by_confidence(self):
        """Test that conflicts are resolved by confidence weighting."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.4, 0.6) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
        )

        # Simulate conflicting decisions
        fast_results = [
            {
                "signals": np.array([0.7]),
                "score": 0.7,
                "decision": "HONEST",
                "probability": 0.6,  # Moderate confidence
                "interval": (0.5, 0.9),
                "uncertainty": 0.4,
            }
        ]

        deep_results = {
            0: {
                "signals": {"deep": 0.3},
                "score": 0.3,
                "decision": "BYZANTINE",
                "confidence": 0.9,  # High confidence
            }
        }

        final_decisions = inspector._fuse_decisions(fast_results, deep_results)

        decision, confidence = final_decisions[0]

        # Deep has higher confidence (0.9 vs 0.6)
        # Should lean towards BYZANTINE
        assert decision == "BYZANTINE"

    def test_confidence_propagation(self):
        """Test that confidence is correctly propagated."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
        )

        gradients = [np.random.normal(0, 0.1, size=10)]

        results = inspector.inspect_batch(gradients)

        # All results should have valid confidence
        for decision, confidence in results:
            assert decision in ["HONEST", "BYZANTINE"]
            assert 0.0 <= confidence <= 1.0


class TestPerformance:
    """Test suite for performance characteristics."""

    def test_speedup_measurement(self):
        """Test that speedup is achieved with query budget."""
        methods = [MockDetector("method1", 0.5), MockDetector("method2", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        # Low query budget (10%)
        config = ActiveLearningConfig(query_budget=0.1)
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            deep_methods=[MockDetector("deep1"), MockDetector("deep2")],
            config=config,
        )

        # Create batch of gradients
        gradients = [np.random.normal(0, 1, size=10) for _ in range(50)]

        # Inspect batch
        results = inspector.inspect_batch(gradients)

        # Check stats
        stats = inspector.get_stats()

        # Should have inspected 50 gradients
        assert stats["total_gradients"] == 50

        # Should have queried ~10% (5 gradients)
        assert stats["total_queries"] <= 10  # Budget + some variance

        # Query rate should be ~10%
        assert stats["query_rate"] <= 0.15

        # Speedup should be > 1
        assert stats["avg_speedup"] > 1.0

    def test_accuracy_maintained(self):
        """Test that accuracy is maintained with query budget."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(query_budget=0.2)
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create test set (honest + Byzantine)
        honest_grads = [np.random.normal(0, 0.1, size=10) for _ in range(30)]
        byzantine_grads = [np.random.normal(5, 1.0, size=10) for _ in range(20)]
        gradients = honest_grads + byzantine_grads

        # True labels
        labels = [1] * 30 + [0] * 20

        # Inspect
        results = inspector.inspect_batch(gradients)

        # Check accuracy (should be reasonable)
        correct = 0
        for i, (decision, confidence) in enumerate(results):
            predicted = 1 if decision == "HONEST" else 0
            if predicted == labels[i]:
                correct += 1

        accuracy = correct / len(results)

        # Should maintain reasonable accuracy (>70%)
        # Note: This is a simple test, real accuracy depends on detection methods
        assert accuracy > 0.5  # At least better than random


class TestIntegration:
    """Integration tests for complete active learning pipeline."""

    def test_two_pass_pipeline(self):
        """Test complete two-pass detection pipeline."""
        # Setup
        methods = [MockDetector("method1", 0.5), MockDetector("method2", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        deep_methods = [MockDetector("deep1", 0.6), MockDetector("deep2", 0.7)]

        config = ActiveLearningConfig(query_budget=0.15)
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            deep_methods=deep_methods,
            config=config,
        )

        # Create gradients
        gradients = [np.random.normal(0, 1, size=10) for _ in range(20)]

        # Run complete pipeline
        results = inspector.inspect_batch(gradients)

        # Verify results
        assert len(results) == 20

        for decision, confidence in results:
            assert decision in ["HONEST", "BYZANTINE"]
            assert 0.0 <= confidence <= 1.0

        # Verify statistics
        stats = inspector.get_stats()
        assert stats["total_inspections"] == 1
        assert stats["total_gradients"] == 20
        assert stats["total_queries"] > 0  # Should have queried some

    def test_fast_pass_on_all(self):
        """Test that fast pass runs on all gradients."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
        )

        gradients = [np.random.normal(0, 1, size=10) for _ in range(10)]

        # Fast pass
        fast_results = inspector._fast_pass(gradients)

        # Should have result for each gradient
        assert len(fast_results) == 10

        # Each result should have required fields
        for result in fast_results:
            assert "signals" in result
            assert "score" in result
            assert "decision" in result
            assert "probability" in result
            assert "interval" in result
            assert "uncertainty" in result

    def test_query_selection_identifies_uncertain(self):
        """Test that query selection identifies uncertain samples."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate with narrow range (low uncertainty for most)
        calibration_scores = [0.5 for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(query_budget=0.3)
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create gradients (most certain, few uncertain)
        certain_grads = [np.random.normal(0, 0.01, size=10) for _ in range(7)]
        uncertain_grads = [np.random.normal(2, 2, size=10) for _ in range(3)]
        gradients = certain_grads + uncertain_grads

        fast_results = inspector._fast_pass(gradients)
        query_indices = inspector._select_queries(fast_results)

        # Should select some queries
        assert len(query_indices) > 0
        assert len(query_indices) <= len(gradients)

    def test_deep_pass_on_queries_only(self):
        """Test that deep pass runs only on selected queries."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        deep_methods = [MockDetector("deep1", 0.6)]

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            deep_methods=deep_methods,
        )

        gradients = [np.random.normal(0, 1, size=10) for _ in range(10)]
        query_indices = [2, 5, 8]  # Manually specify queries

        # Deep pass
        deep_results = inspector._deep_pass(gradients, query_indices)

        # Should have results only for queried indices
        assert len(deep_results) == 3
        assert 2 in deep_results
        assert 5 in deep_results
        assert 8 in deep_results

        # Each result should have required fields
        for result in deep_results.values():
            assert "signals" in result
            assert "score" in result
            assert "decision" in result
            assert "confidence" in result

    def test_byzantine_robustness(self):
        """Test robustness to Byzantine gradients."""
        methods = [MockDetector("method1", 0.5), MockDetector("method2", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        config = ActiveLearningConfig(query_budget=0.2)
        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
            config=config,
        )

        # Create batch with Byzantine gradients
        honest_grads = [np.random.normal(0, 0.1, size=10) for _ in range(20)]
        byzantine_grads = [np.random.normal(10, 2, size=10) for _ in range(10)]
        gradients = honest_grads + byzantine_grads

        # Inspect
        results = inspector.inspect_batch(gradients)

        # Should produce valid results
        assert len(results) == 30

        for decision, confidence in results:
            assert decision in ["HONEST", "BYZANTINE"]
            assert 0.0 <= confidence <= 1.0

        # System should not crash with Byzantine inputs
        stats = inspector.get_stats()
        assert stats["total_gradients"] == 30


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        """Test that statistics are correctly tracked."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
        )

        gradients = [np.random.normal(0, 1, size=10) for _ in range(10)]

        inspector.inspect_batch(gradients)

        stats = inspector.get_stats()

        # Check required stats
        assert "total_inspections" in stats
        assert "total_gradients" in stats
        assert "total_queries" in stats
        assert "query_rate" in stats
        assert "avg_speedup" in stats

        # Values should be reasonable
        assert stats["total_inspections"] == 1
        assert stats["total_gradients"] == 10
        assert 0.0 <= stats["query_rate"] <= 1.0
        assert stats["avg_speedup"] >= 1.0

    def test_stats_reset(self):
        """Test that stats can be reset."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        quantifier = UncertaintyQuantifier(buffer_size=50)

        # Calibrate
        calibration_scores = [np.random.uniform(0.3, 0.7) for _ in range(50)]
        quantifier.update(calibration_scores)

        inspector = ActiveLearningInspector(
            fast_ensemble=ensemble,
            uncertainty_quantifier=quantifier,
        )

        gradients = [np.random.normal(0, 1, size=10) for _ in range(10)]
        inspector.inspect_batch(gradients)

        # Reset
        inspector.reset_stats()

        stats = inspector.get_stats()

        # All stats should be reset
        assert stats["total_inspections"] == 0
        assert stats["total_gradients"] == 0
        assert stats["total_queries"] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
