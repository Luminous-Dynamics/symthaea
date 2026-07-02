# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Causal Attribution Engine - Gen 5 Layer 2
=========================================

SHAP-inspired explanations for Byzantine detection decisions.

Generates natural language explanations showing:
- Which detection methods contributed most to the decision
- Magnitude of each method's contribution
- Clear reasoning for why a node was flagged or accepted

Key Features:
- SHAP-inspired importance calculation
- Ranked contributor analysis
- Natural language template system
- Human-readable explanations for every decision

Algorithm:
----------
1. Compute contribution_i = signal_i × importance_i
2. Rank contributors by absolute contribution
3. Generate explanation from top contributors
4. Use method-specific templates for clarity

Author: Luminous Dynamics
Date: November 11, 2025
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .meta_learning import MetaLearningEnsemble


class CausalAttributionEngine:
    """
    Generate human-readable explanations for detection decisions.

    Uses SHAP-inspired causal attribution to identify which detection
    methods contributed most to a decision, then generates natural
    language explanations.

    Example:
        >>> from gen5 import MetaLearningEnsemble, CausalAttributionEngine
        >>>
        >>> ensemble = MetaLearningEnsemble(base_methods)
        >>> explainer = CausalAttributionEngine(ensemble)
        >>>
        >>> # After detection
        >>> signals = ensemble.compute_signals(gradient)
        >>> score = ensemble.compute_ensemble_score(signals)
        >>> decision = "HONEST" if score >= 0.5 else "BYZANTINE"
        >>>
        >>> # Generate explanation
        >>> explanation = explainer.explain_decision(
        ...     node_id=7,
        ...     signals=signals,
        ...     decision=decision,
        ...     score=score
        ... )
        >>> print(explanation)
        # "Node 7 flagged BYZANTINE (confidence=0.95):
        #  - Low gradient quality: PoGQ=0.15 (contributed 35%)
        #  - Direction mismatch: FLTrust=0.42 (contributed 28%)
        #  - Gradient outlier: Krum=0.51 (contributed 18%)"
    """

    def __init__(
        self,
        ensemble: MetaLearningEnsemble,
        min_contribution_threshold: float = 0.05,
    ):
        """
        Initialize causal attribution engine.

        Args:
            ensemble: MetaLearningEnsemble instance
            min_contribution_threshold: Minimum contribution to include
                                       in explanation (default: 5%)
        """
        self.ensemble = ensemble
        self.min_contribution_threshold = min_contribution_threshold

        # Method-specific explanation templates
        self.method_templates = {
            "pogq": {
                "low": "Low gradient quality: PoGQ={signal:.3f}",
                "high": "High gradient quality: PoGQ={signal:.3f}",
            },
            "pogq_v4.1": {
                "low": "Low gradient quality: PoGQ={signal:.3f}",
                "high": "High gradient quality: PoGQ={signal:.3f}",
            },
            "fltrust": {
                "low": "Direction mismatch: FLTrust={signal:.3f}",
                "high": "Direction alignment: FLTrust={signal:.3f}",
            },
            "krum": {
                "low": "Gradient outlier: Krum={signal:.3f}",
                "high": "Gradient similarity: Krum={signal:.3f}",
            },
            "cbf": {
                "low": "Anomalous behavior: CBF={signal:.3f}",
                "high": "Normal behavior: CBF={signal:.3f}",
            },
            "foolsgold": {
                "low": "Sybil pattern detected: FoolsGold={signal:.3f}",
                "high": "Unique gradient history: FoolsGold={signal:.3f}",
            },
            "reputation": {
                "low": "Low reputation: Score={signal:.3f}",
                "high": "High reputation: Score={signal:.3f}",
            },
            "temporal": {
                "low": "Unusual behavior pattern: Temporal={signal:.3f}",
                "high": "Consistent behavior: Temporal={signal:.3f}",
            },
            "blacklist": {
                "low": "Previously flagged node",
                "high": "Clean history",
            },
        }

    def compute_contributions(
        self, signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute SHAP-inspired marginal contribution of each method.

        Contribution = signal_value × method_importance

        This approximates the marginal effect of including method i
        in the ensemble decision.

        Args:
            signals: Dict of method_name → signal_value

        Returns:
            Dict of method_name → contribution value

        Example:
            >>> contributions = explainer.compute_contributions({
            ...     'pogq': 0.15,    # Low score (suspicious)
            ...     'fltrust': 0.92, # High score (honest)
            ...     'krum': 0.78
            ... })
            >>> # With importances: pogq=0.35, fltrust=0.40, krum=0.25
            >>> # Returns: {
            >>> #     'pogq': 0.15 × 0.35 = 0.0525
            >>> #     'fltrust': 0.92 × 0.40 = 0.368
            >>> #     'krum': 0.78 × 0.25 = 0.195
            >>> # }
        """
        importances = self.ensemble.get_method_importances()

        contributions = {}
        for method_name, signal_value in signals.items():
            importance = importances.get(method_name, 0.0)
            contributions[method_name] = signal_value * importance

        return contributions

    def rank_contributors(
        self, contributions: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Rank methods by absolute contribution magnitude.

        Args:
            contributions: Dict of method_name → contribution

        Returns:
            List of (method_name, contribution) sorted by |contribution|

        Example:
            >>> ranked = explainer.rank_contributors({
            ...     'pogq': 0.05,
            ...     'fltrust': 0.37,
            ...     'krum': 0.20
            ... })
            >>> # Returns: [
            >>> #     ('fltrust', 0.37),
            >>> #     ('krum', 0.20),
            >>> #     ('pogq', 0.05)
            >>> # ]
        """
        sorted_contributors = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return sorted_contributors

    def get_top_k_reasons(
        self,
        contributions: Dict[str, float],
        signals: Dict[str, float],
        k: int = 3,
    ) -> List[str]:
        """
        Get top-k contributing methods with explanations.

        Args:
            contributions: Method contributions
            signals: Original signal values
            k: Number of top contributors to return

        Returns:
            List of explanation strings

        Example:
            >>> reasons = explainer.get_top_k_reasons(
            ...     contributions={'pogq': 0.05, 'fltrust': 0.37, 'krum': 0.20},
            ...     signals={'pogq': 0.15, 'fltrust': 0.92, 'krum': 0.78},
            ...     k=2
            ... )
            >>> # Returns: [
            >>> #     "Direction alignment: FLTrust=0.920 (contributed 37%)",
            >>> #     "Gradient similarity: Krum=0.780 (contributed 20%)"
            >>> # ]
        """
        ranked = self.rank_contributors(contributions)
        top_k = ranked[:k]

        reasons = []
        for method_name, contribution in top_k:
            # Skip if contribution too small
            if abs(contribution) < self.min_contribution_threshold:
                continue

            signal_value = signals.get(method_name, 0.5)

            # Determine if signal is "low" or "high"
            signal_type = "low" if signal_value < 0.5 else "high"

            # Get template for this method
            if method_name in self.method_templates:
                template = self.method_templates[method_name].get(
                    signal_type, f"{method_name}={signal_value:.3f}"
                )
                reason = template.format(signal=signal_value)
            else:
                # Generic template
                reason = f"{method_name.capitalize()}={signal_value:.3f}"

            # Add contribution percentage
            contribution_pct = abs(contribution) * 100
            reason += f" (contributed {contribution_pct:.1f}%)"

            reasons.append(reason)

        return reasons

    def explain_decision(
        self,
        node_id: int,
        signals: Dict[str, float],
        decision: str,
        score: float,
    ) -> str:
        """
        Generate natural language explanation for detection decision.

        Args:
            node_id: Client node ID
            signals: All detection signals
            decision: Final decision ("HONEST" or "BYZANTINE")
            score: Ensemble confidence score

        Returns:
            Human-readable explanation string

        Examples:
            Byzantine decision:
            >>> explanation = explainer.explain_decision(
            ...     node_id=7,
            ...     signals={'pogq': 0.15, 'fltrust': 0.42, 'krum': 0.51},
            ...     decision="BYZANTINE",
            ...     score=0.05
            ... )
            >>> print(explanation)
            "Node 7 flagged BYZANTINE (confidence=0.95):
             - Low gradient quality: PoGQ=0.150 (contributed 25%)
             - Direction mismatch: FLTrust=0.420 (contributed 18%)
             - Gradient outlier: Krum=0.510 (contributed 12%)"

            Honest decision:
            >>> explanation = explainer.explain_decision(
            ...     node_id=3,
            ...     signals={'pogq': 0.88, 'fltrust': 0.95, 'krum': 0.82},
            ...     decision="HONEST",
            ...     score=0.92
            ... )
            >>> print(explanation)
            "Node 3 classified HONEST (confidence=0.92):
             All signals within normal range. PoGQ=0.88, FLTrust=0.95, Krum=0.82"
        """
        contributions = self.compute_contributions(signals)
        sorted_contributors = self.rank_contributors(contributions)

        if decision == "BYZANTINE":
            return self._explain_byzantine(
                node_id, signals, score, contributions, sorted_contributors
            )
        else:  # HONEST
            return self._explain_honest(
                node_id, signals, score, contributions, sorted_contributors
            )

    def _explain_byzantine(
        self,
        node_id: int,
        signals: Dict[str, float],
        score: float,
        contributions: Dict[str, float],
        sorted_contributors: List[Tuple[str, float]],
    ) -> str:
        """Generate explanation for Byzantine decision."""
        # Compute confidence (1 - score for Byzantine)
        confidence = 1.0 - score

        # Find methods with low signals (indicating Byzantine)
        low_signals = [
            (method, signal)
            for method, signal in signals.items()
            if signal < 0.5
        ]

        if not low_signals:
            # Edge case: Byzantine by ensemble consensus but no single low signal
            return (
                f"Node {node_id} flagged BYZANTINE (confidence={confidence:.3f}): "
                f"Ensemble consensus. Multiple weak signals combined to exceed "
                f"detection threshold."
            )

        # Build explanation
        explanation_lines = [
            f"Node {node_id} flagged BYZANTINE (confidence={confidence:.3f}):"
        ]

        # Get top 3 contributing reasons
        reasons = self.get_top_k_reasons(contributions, signals, k=3)

        for reason in reasons:
            explanation_lines.append(f" - {reason}")

        return "\n".join(explanation_lines)

    def _explain_honest(
        self,
        node_id: int,
        signals: Dict[str, float],
        score: float,
        contributions: Dict[str, float],
        sorted_contributors: List[Tuple[str, float]],
    ) -> str:
        """Generate explanation for Honest decision."""
        # Check if all signals are high
        all_high = all(signal > 0.7 for signal in signals.values())

        if all_high:
            # Simple case: all signals indicate honest
            top_3 = sorted_contributors[:3]
            signal_strs = [
                f"{method}={signals[method]:.2f}" for method, _ in top_3
            ]
            signal_summary = ", ".join(signal_strs)

            return (
                f"Node {node_id} classified HONEST (confidence={score:.3f}): "
                f"All signals within normal range. {signal_summary}"
            )
        else:
            # Some signals low but ensemble says honest
            top_2 = sorted_contributors[:2]
            top_contributors_str = ", ".join(
                [
                    f"{method} ({contributions[method]:.1%})"
                    for method, _ in top_2
                ]
            )

            return (
                f"Node {node_id} classified HONEST (confidence={score:.3f}): "
                f"Ensemble consensus despite some weak signals. "
                f"Top contributors: {top_contributors_str}"
            )

    def explain_batch(
        self, detections: List[Dict[str, any]]
    ) -> List[str]:
        """
        Generate explanations for batch of detections.

        Args:
            detections: List of detection results, each containing:
                       - node_id: int
                       - signals: Dict[str, float]
                       - decision: str
                       - score: float

        Returns:
            List of explanation strings (same order as input)

        Example:
            >>> detections = [
            ...     {'node_id': 0, 'signals': {...}, 'decision': 'HONEST', 'score': 0.92},
            ...     {'node_id': 1, 'signals': {...}, 'decision': 'BYZANTINE', 'score': 0.05},
            ... ]
            >>> explanations = explainer.explain_batch(detections)
            >>> for explanation in explanations:
            ...     print(explanation)
        """
        explanations = []

        for detection in detections:
            explanation = self.explain_decision(
                node_id=detection["node_id"],
                signals=detection["signals"],
                decision=detection["decision"],
                score=detection["score"],
            )
            explanations.append(explanation)

        return explanations

    def add_custom_template(
        self, method_name: str, low_template: str, high_template: str
    ) -> None:
        """
        Add custom explanation template for a detection method.

        Args:
            method_name: Name of detection method
            low_template: Template for low signal values
                         (use {signal} placeholder)
            high_template: Template for high signal values
                          (use {signal} placeholder)

        Example:
            >>> explainer.add_custom_template(
            ...     "my_detector",
            ...     low_template="Suspicious pattern: score={signal:.3f}",
            ...     high_template="Normal pattern: score={signal:.3f}"
            ... )
        """
        self.method_templates[method_name] = {
            "low": low_template,
            "high": high_template,
        }

    def __repr__(self) -> str:
        """String representation."""
        n_methods = len(self.ensemble.base_methods)
        n_templates = len(self.method_templates)

        return (
            f"CausalAttributionEngine(\n"
            f"  methods={n_methods},\n"
            f"  templates={n_templates},\n"
            f"  min_contribution={self.min_contribution_threshold:.1%}\n"
            f")"
        )
