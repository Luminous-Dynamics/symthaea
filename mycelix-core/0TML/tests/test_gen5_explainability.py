# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Causal Attribution Engine
=========================================

Tests:
1. Contribution computation (SHAP-inspired)
2. Contributor ranking
3. Byzantine decision explanations
4. Honest decision explanations
5. Batch explanations
6. Custom templates
7. Edge cases

Author: Luminous Dynamics
Date: November 11, 2025
"""

import pytest
import numpy as np

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.meta_learning import MetaLearningEnsemble, MetaLearningConfig
from gen5.explainability import CausalAttributionEngine


# Mock detection method for testing
class MockDetector:
    """Mock Byzantine detector for testing."""

    def __init__(self, name: str, base_score: float = 0.5):
        self.name = name
        self.base_score = base_score

    def score(self, gradient: np.ndarray) -> float:
        """Return deterministic score for testing."""
        return self.base_score


class TestCausalAttributionEngine:
    """Test suite for CausalAttributionEngine."""

    def test_initialization(self):
        """Test engine initializes correctly."""
        methods = [MockDetector("pogq", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        assert explainer.ensemble == ensemble
        assert explainer.min_contribution_threshold == 0.05
        assert len(explainer.method_templates) > 0

    def test_compute_contributions(self):
        """Test SHAP-inspired contribution computation."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.6),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Set non-uniform weights
        ensemble.weights = np.array([0.7, 0.3])  # pogq more important

        explainer = CausalAttributionEngine(ensemble)

        signals = {"pogq": 0.2, "fltrust": 0.9}

        contributions = explainer.compute_contributions(signals)

        # Get importances for verification
        importances = ensemble.get_method_importances()

        # Contributions should be signal × importance
        expected_pogq = signals["pogq"] * importances["pogq"]
        expected_fltrust = signals["fltrust"] * importances["fltrust"]

        assert abs(contributions["pogq"] - expected_pogq) < 1e-6
        assert abs(contributions["fltrust"] - expected_fltrust) < 1e-6

    def test_rank_contributors(self):
        """Test contributor ranking by absolute contribution."""
        methods = [MockDetector("method", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        contributions = {
            "pogq": 0.05,
            "fltrust": 0.35,
            "krum": 0.15,
        }

        ranked = explainer.rank_contributors(contributions)

        # Should be sorted by absolute value (descending)
        assert ranked[0] == ("fltrust", 0.35)
        assert ranked[1] == ("krum", 0.15)
        assert ranked[2] == ("pogq", 0.05)

    def test_get_top_k_reasons(self):
        """Test top-k reason extraction."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
            MockDetector("krum", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        contributions = {
            "pogq": 0.05,
            "fltrust": 0.35,
            "krum": 0.15,
        }

        signals = {
            "pogq": 0.15,  # Low (suspicious)
            "fltrust": 0.92,  # High (honest)
            "krum": 0.78,  # High (honest)
        }

        reasons = explainer.get_top_k_reasons(contributions, signals, k=2)

        # Should get top 2 contributors
        assert len(reasons) == 2

        # fltrust should be first (highest contribution)
        assert "fltrust" in reasons[0].lower() or "direction" in reasons[0].lower()

        # Each reason should have contribution percentage
        assert "%" in reasons[0]
        assert "%" in reasons[1]

    def test_explain_byzantine_decision(self):
        """Test explanation generation for Byzantine decision."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        # Byzantine signals (low scores)
        signals = {
            "pogq": 0.15,
            "fltrust": 0.25,
        }

        explanation = explainer.explain_decision(
            node_id=7,
            signals=signals,
            decision="BYZANTINE",
            score=0.10,  # Low score → high confidence Byzantine
        )

        # Check explanation format
        assert "Node 7" in explanation
        assert "BYZANTINE" in explanation
        assert "confidence" in explanation.lower()

        # Should mention at least one method
        assert "pogq" in explanation.lower() or "fltrust" in explanation.lower()

    def test_explain_honest_decision_all_high(self):
        """Test explanation for honest decision with all high signals."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        # Honest signals (high scores)
        signals = {
            "pogq": 0.88,
            "fltrust": 0.95,
        }

        explanation = explainer.explain_decision(
            node_id=3,
            signals=signals,
            decision="HONEST",
            score=0.92,
        )

        # Check explanation format
        assert "Node 3" in explanation
        assert "HONEST" in explanation
        assert "confidence" in explanation.lower()

        # Should mention "normal range" or similar
        assert "normal" in explanation.lower() or "within" in explanation.lower()

    def test_explain_honest_decision_mixed_signals(self):
        """Test explanation for honest decision with mixed signals."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
            MockDetector("krum", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Set weights to favor fltrust
        ensemble.weights = np.array([0.2, 0.6, 0.2])

        explainer = CausalAttributionEngine(ensemble)

        # Mixed signals: pogq low, fltrust high
        signals = {
            "pogq": 0.3,  # Low
            "fltrust": 0.9,  # High (dominant)
            "krum": 0.4,  # Low
        }

        explanation = explainer.explain_decision(
            node_id=5,
            signals=signals,
            decision="HONEST",
            score=0.75,
        )

        # Should mention "ensemble consensus" or similar
        assert "consensus" in explanation.lower() or "despite" in explanation.lower()

    def test_explain_batch(self):
        """Test batch explanation generation."""
        methods = [MockDetector("pogq", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        detections = [
            {
                "node_id": 0,
                "signals": {"pogq": 0.85},
                "decision": "HONEST",
                "score": 0.85,
            },
            {
                "node_id": 1,
                "signals": {"pogq": 0.15},
                "decision": "BYZANTINE",
                "score": 0.15,
            },
        ]

        explanations = explainer.explain_batch(detections)

        assert len(explanations) == 2
        assert "Node 0" in explanations[0]
        assert "Node 1" in explanations[1]
        assert "HONEST" in explanations[0]
        assert "BYZANTINE" in explanations[1]

    def test_add_custom_template(self):
        """Test adding custom explanation templates."""
        methods = [MockDetector("my_detector", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        # Add custom template
        explainer.add_custom_template(
            "my_detector",
            low_template="Custom low: {signal:.3f}",
            high_template="Custom high: {signal:.3f}",
        )

        # Verify template was added
        assert "my_detector" in explainer.method_templates
        assert explainer.method_templates["my_detector"]["low"] == "Custom low: {signal:.3f}"
        assert explainer.method_templates["my_detector"]["high"] == "Custom high: {signal:.3f}"

    def test_min_contribution_threshold(self):
        """Test that low contributions are filtered out."""
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
            MockDetector("method3", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Set weights so method3 has very low importance
        ensemble.weights = np.array([0.9, 0.09, 0.01])

        explainer = CausalAttributionEngine(
            ensemble, min_contribution_threshold=0.10
        )

        contributions = {
            "method1": 0.45,  # High
            "method2": 0.04,  # Low (will be filtered)
            "method3": 0.005,  # Very low (will be filtered)
        }

        signals = {
            "method1": 0.5,
            "method2": 0.5,
            "method3": 0.5,
        }

        reasons = explainer.get_top_k_reasons(contributions, signals, k=3)

        # Should only include method1 (contribution > 0.10)
        assert len(reasons) >= 1
        assert "method1" in reasons[0].lower()

    def test_repr(self):
        """Test string representation."""
        methods = [MockDetector("pogq", 0.5), MockDetector("fltrust", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        repr_str = repr(explainer)

        assert "CausalAttributionEngine" in repr_str
        assert "methods=" in repr_str

    def test_edge_case_empty_signals(self):
        """Test handling of empty signals dict."""
        methods = [MockDetector("pogq", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        # Empty signals
        contributions = explainer.compute_contributions({})

        assert len(contributions) == 0

    def test_edge_case_unknown_method_in_signals(self):
        """Test handling of unknown method in signals."""
        methods = [MockDetector("pogq", 0.5)]
        ensemble = MetaLearningEnsemble(methods)
        explainer = CausalAttributionEngine(ensemble)

        # Signal for unknown method
        signals = {"unknown_method": 0.7}

        contributions = explainer.compute_contributions(signals)

        # Should handle gracefully (importance=0 for unknown)
        assert "unknown_method" in contributions
        assert contributions["unknown_method"] == 0.0  # No importance

    def test_byzantine_explanation_no_low_signals(self):
        """Test Byzantine explanation when no signals are low."""
        methods = [MockDetector("pogq", 0.5), MockDetector("fltrust", 0.5)]

        ensemble = MetaLearningEnsemble(methods)

        # Set extreme weights to force Byzantine despite high signals
        ensemble.weights = np.array([1.0, -1.0])

        explainer = CausalAttributionEngine(ensemble)

        # All high signals but ensemble says Byzantine
        signals = {"pogq": 0.8, "fltrust": 0.9}

        explanation = explainer.explain_decision(
            node_id=10, signals=signals, decision="BYZANTINE", score=0.1
        )

        # Should mention "ensemble consensus"
        assert "consensus" in explanation.lower()


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_explanation_pipeline(self):
        """Test complete explanation pipeline with realistic data."""
        # Create ensemble with 3 methods
        methods = [
            MockDetector("pogq", 0.3),
            MockDetector("fltrust", 0.6),
            MockDetector("krum", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Train ensemble on synthetic data
        n_samples = 100
        signals_batch = []
        labels_batch = []

        np.random.seed(42)

        for i in range(n_samples):
            is_honest = (i % 2 == 0)

            if is_honest:
                signals = [0.85, 0.90, 0.80]
                label = 1.0
            else:
                signals = [0.20, 0.15, 0.25]
                label = 0.0

            signals_batch.append(signals)
            labels_batch.append(label)

        signals_batch = np.array(signals_batch)
        labels_batch = np.array(labels_batch)

        # Train
        for i in range(0, n_samples, 20):
            ensemble.update_weights(
                signals_batch[i:i+20], labels_batch[i:i+20]
            )

        # Create explainer
        explainer = CausalAttributionEngine(ensemble)

        # Test Byzantine explanation
        byzantine_signals = {"pogq": 0.18, "fltrust": 0.12, "krum": 0.22}

        byz_explanation = explainer.explain_decision(
            node_id=7,
            signals=byzantine_signals,
            decision="BYZANTINE",
            score=0.08,
        )

        # Verify Byzantine explanation
        assert "Node 7" in byz_explanation
        assert "BYZANTINE" in byz_explanation
        assert "confidence" in byz_explanation.lower()
        assert "pogq" in byz_explanation.lower() or "quality" in byz_explanation.lower()

        # Test Honest explanation
        honest_signals = {"pogq": 0.87, "fltrust": 0.93, "krum": 0.81}

        honest_explanation = explainer.explain_decision(
            node_id=3,
            signals=honest_signals,
            decision="HONEST",
            score=0.91,
        )

        # Verify Honest explanation
        assert "Node 3" in honest_explanation
        assert "HONEST" in honest_explanation
        assert "normal" in honest_explanation.lower() or "within" in honest_explanation.lower()

        print(f"\n🎯 Realistic Integration Test:")
        print(f"\nByzantine Explanation:\n{byz_explanation}")
        print(f"\nHonest Explanation:\n{honest_explanation}")
        print(f"\n✅ Explanations generated successfully!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
