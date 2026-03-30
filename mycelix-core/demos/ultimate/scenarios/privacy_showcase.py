# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Privacy Showcase Scenario - Differential Privacy Features Demonstration

Demonstrates Mycelix-Core's comprehensive privacy features:
- Differential privacy with configurable epsilon
- Gradient clipping and noise calibration
- Privacy budget accounting
- Utility vs privacy tradeoff visualization

Author: Luminous Dynamics
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


@dataclass
class PrivacyConfig:
    """Configuration for privacy settings."""
    epsilon: float
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0


@dataclass
class PrivacyRoundMetrics:
    """Metrics for a single round with privacy."""
    round_num: int
    epsilon_config: float
    epsilon_spent: float
    noise_scale: float
    gradients_clipped: int
    gradients_total: int
    clip_fraction: float
    avg_clip_ratio: float
    model_accuracy: float
    accuracy_drop: float  # vs no-privacy baseline
    utility_privacy_ratio: float


@dataclass
class EpsilonPhaseResult:
    """Result from testing a specific epsilon value."""
    epsilon: float
    rounds_completed: int
    total_epsilon_spent: float
    avg_noise_scale: float
    avg_clip_fraction: float
    final_accuracy: float
    accuracy_vs_baseline: float
    utility_score: float


@dataclass
class PrivacyShowcaseResult:
    """Final result of privacy showcase."""
    phases_completed: int
    epsilon_phases: List[EpsilonPhaseResult]
    optimal_epsilon: float
    optimal_utility_privacy: float
    privacy_budget_analysis: Dict[str, Any]
    verdict: str


class PrivacyShowcaseScenario:
    """
    Privacy Features Demonstration.

    Shows how Mycelix-Core maintains model utility while providing
    strong differential privacy guarantees:

    1. High epsilon (10.0) - minimal privacy, maximum utility
    2. Medium epsilon (5.0, 1.0) - balanced tradeoff
    3. Low epsilon (0.5, 0.1) - strong privacy, some utility loss
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize privacy showcase scenario."""
        self.config = config
        self.num_nodes = config.get("num_nodes", 50)
        self.gradient_dim = config.get("gradient_dim", 10000)
        self.seed = config.get("random_seed", 42)
        self.delta = config.get("delta", 1e-5)

        # Initialize RNG
        self.rng = np.random.default_rng(self.seed)

        # Epsilon values to test
        self.epsilon_values = config.get("epsilons", [10.0, 5.0, 1.0, 0.5, 0.1])
        self.rounds_per_epsilon = config.get("rounds_per_epsilon", 10)

        # State
        self.baseline_accuracy = 0.95  # Accuracy without privacy
        self.phase_results: List[EpsilonPhaseResult] = []
        self.current_epsilon_idx = 0

    def run(self, progress_callback=None) -> PrivacyShowcaseResult:
        """
        Run the privacy showcase demonstration.

        Args:
            progress_callback: Optional callback(epsilon_idx, round, metrics)

        Returns:
            PrivacyShowcaseResult with all metrics
        """
        for eps_idx, epsilon in enumerate(self.epsilon_values):
            self.current_epsilon_idx = eps_idx
            result = self._run_epsilon_phase(epsilon, progress_callback)
            self.phase_results.append(result)

            time.sleep(0.1)  # Brief pause between phases

        return self._compute_final_result()

    def _run_epsilon_phase(
        self, epsilon: float, progress_callback=None
    ) -> EpsilonPhaseResult:
        """Run privacy test for a specific epsilon value."""
        # Compute noise multiplier from epsilon
        noise_multiplier = self._compute_noise_multiplier(epsilon)

        # Track metrics
        total_epsilon_spent = 0.0
        total_noise_scale = 0.0
        total_clip_fraction = 0.0
        round_metrics: List[PrivacyRoundMetrics] = []

        # Simulated model accuracy (starts at baseline)
        model_accuracy = self.baseline_accuracy

        for round_num in range(self.rounds_per_epsilon):
            metrics = self._run_privacy_round(
                round_num, epsilon, noise_multiplier, model_accuracy
            )
            round_metrics.append(metrics)

            # Update totals
            total_epsilon_spent = metrics.epsilon_spent
            total_noise_scale += metrics.noise_scale
            total_clip_fraction += metrics.clip_fraction

            # Update accuracy (degrades slightly with privacy)
            model_accuracy = metrics.model_accuracy

            # Progress callback
            if progress_callback:
                progress_callback(self.current_epsilon_idx, round_num, metrics)

            time.sleep(0.02)

        # Compute phase result
        avg_noise = total_noise_scale / self.rounds_per_epsilon
        avg_clip = total_clip_fraction / self.rounds_per_epsilon
        accuracy_vs_baseline = model_accuracy / self.baseline_accuracy

        # Utility score: how much accuracy retained vs privacy provided
        # Higher epsilon = less privacy but more utility
        privacy_strength = 1 / (1 + epsilon)  # Normalized privacy strength
        utility_score = accuracy_vs_baseline * (1 + privacy_strength)

        return EpsilonPhaseResult(
            epsilon=epsilon,
            rounds_completed=self.rounds_per_epsilon,
            total_epsilon_spent=total_epsilon_spent,
            avg_noise_scale=avg_noise,
            avg_clip_fraction=avg_clip,
            final_accuracy=model_accuracy,
            accuracy_vs_baseline=accuracy_vs_baseline,
            utility_score=utility_score,
        )

    def _run_privacy_round(
        self,
        round_num: int,
        epsilon: float,
        noise_multiplier: float,
        current_accuracy: float,
    ) -> PrivacyRoundMetrics:
        """Execute a single round with differential privacy."""
        # Generate simulated gradients
        gradients = self._generate_gradients()

        # Apply gradient clipping
        clipped_gradients, clip_stats = self._clip_gradients(gradients)

        # Add calibrated noise
        noisy_gradients, noise_scale = self._add_noise(
            clipped_gradients, noise_multiplier
        )

        # Compute privacy spent (simplified RDP accounting)
        epsilon_spent = self._compute_epsilon_spent(
            noise_multiplier, round_num + 1
        )

        # Compute accuracy impact
        # More noise = more accuracy degradation
        noise_impact = noise_scale / (np.mean([np.linalg.norm(g) for g in gradients.values()]) + 1e-8)
        accuracy_drop = min(0.5, noise_impact * 0.1)
        new_accuracy = max(0.3, current_accuracy - accuracy_drop * 0.01)

        # Utility-privacy ratio
        utility_privacy = new_accuracy / (epsilon_spent + 1e-8)

        return PrivacyRoundMetrics(
            round_num=round_num,
            epsilon_config=epsilon,
            epsilon_spent=epsilon_spent,
            noise_scale=noise_scale,
            gradients_clipped=clip_stats["clipped_count"],
            gradients_total=len(gradients),
            clip_fraction=clip_stats["clip_fraction"],
            avg_clip_ratio=clip_stats["avg_ratio"],
            model_accuracy=new_accuracy,
            accuracy_drop=accuracy_drop,
            utility_privacy_ratio=utility_privacy,
        )

    def _generate_gradients(self) -> Dict[str, np.ndarray]:
        """Generate simulated gradients."""
        gradients = {}
        for i in range(self.num_nodes):
            # Generate gradient with varying magnitudes
            magnitude = self.rng.uniform(0.5, 2.0)
            gradient = self.rng.standard_normal(self.gradient_dim) * magnitude
            gradients[f"node_{i:03d}"] = gradient.astype(np.float32)
        return gradients

    def _clip_gradients(
        self, gradients: Dict[str, np.ndarray], max_norm: float = 1.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Clip gradients to maximum L2 norm."""
        clipped = {}
        clipped_count = 0
        clip_ratios = []

        for node_id, gradient in gradients.items():
            norm = np.linalg.norm(gradient)
            if norm > max_norm:
                clipped[node_id] = gradient * (max_norm / norm)
                clipped_count += 1
                clip_ratios.append(max_norm / norm)
            else:
                clipped[node_id] = gradient.copy()
                clip_ratios.append(1.0)

        stats = {
            "clipped_count": clipped_count,
            "clip_fraction": clipped_count / len(gradients),
            "avg_ratio": np.mean(clip_ratios),
        }

        return clipped, stats

    def _add_noise(
        self, gradients: Dict[str, np.ndarray], noise_multiplier: float
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Add Gaussian noise for differential privacy."""
        # Noise scale: sigma = noise_multiplier * sensitivity
        sensitivity = 1.0  # After clipping to max_norm=1.0
        sigma = noise_multiplier * sensitivity

        noisy = {}
        for node_id, gradient in gradients.items():
            noise = self.rng.normal(0, sigma, gradient.shape).astype(np.float32)
            noisy[node_id] = gradient + noise

        return noisy, sigma

    def _compute_noise_multiplier(self, epsilon: float) -> float:
        """Compute noise multiplier for target epsilon."""
        # Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        return math.sqrt(2 * math.log(1.25 / self.delta)) / epsilon

    def _compute_epsilon_spent(
        self, noise_multiplier: float, num_steps: int
    ) -> float:
        """Compute epsilon spent using simple composition."""
        # Advanced composition bound (simplified)
        sample_rate = 1.0  # All data used each round
        per_step_epsilon = sample_rate / noise_multiplier
        composed_epsilon = math.sqrt(2 * num_steps * math.log(1 / self.delta)) * per_step_epsilon
        return composed_epsilon

    def _compute_final_result(self) -> PrivacyShowcaseResult:
        """Compute final result with analysis."""
        if not self.phase_results:
            return PrivacyShowcaseResult(
                phases_completed=0,
                epsilon_phases=[],
                optimal_epsilon=1.0,
                optimal_utility_privacy=0.0,
                privacy_budget_analysis={},
                verdict="INCOMPLETE",
            )

        # Find optimal epsilon (best utility-privacy tradeoff)
        best_phase = max(self.phase_results, key=lambda p: p.utility_score)
        optimal_epsilon = best_phase.epsilon
        optimal_utility_privacy = best_phase.utility_score

        # Privacy budget analysis
        analysis = {
            "epsilon_values_tested": [p.epsilon for p in self.phase_results],
            "accuracy_by_epsilon": {
                p.epsilon: p.final_accuracy for p in self.phase_results
            },
            "utility_by_epsilon": {
                p.epsilon: p.utility_score for p in self.phase_results
            },
            "noise_by_epsilon": {
                p.epsilon: p.avg_noise_scale for p in self.phase_results
            },
            "recommendations": self._generate_recommendations(),
        }

        # Determine verdict
        high_eps_accuracy = next(
            (p.final_accuracy for p in self.phase_results if p.epsilon >= 5.0),
            0.0
        )
        low_eps_accuracy = next(
            (p.final_accuracy for p in self.phase_results if p.epsilon <= 0.5),
            0.0
        )

        if high_eps_accuracy > 0.90 and low_eps_accuracy > 0.70:
            verdict = "EXCELLENT - Strong Privacy with High Utility!"
        elif high_eps_accuracy > 0.85:
            verdict = "VERY GOOD - Good Privacy-Utility Tradeoff"
        else:
            verdict = "GOOD - Privacy Features Working"

        return PrivacyShowcaseResult(
            phases_completed=len(self.phase_results),
            epsilon_phases=self.phase_results,
            optimal_epsilon=optimal_epsilon,
            optimal_utility_privacy=optimal_utility_privacy,
            privacy_budget_analysis=analysis,
            verdict=verdict,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate privacy configuration recommendations."""
        recommendations = []

        # Based on results
        if self.phase_results:
            best_utility = max(p.utility_score for p in self.phase_results)
            best_phase = next(p for p in self.phase_results if p.utility_score == best_utility)

            recommendations.append(
                f"Recommended epsilon: {best_phase.epsilon} for optimal utility-privacy tradeoff"
            )

            if best_phase.epsilon > 1.0:
                recommendations.append(
                    "For stronger privacy, consider epsilon <= 1.0 with acceptable accuracy loss"
                )

            recommendations.append(
                "Use gradient clipping with max_norm=1.0 for bounded sensitivity"
            )
            recommendations.append(
                "Monitor privacy budget to avoid exceeding target epsilon"
            )

        return recommendations

    def get_current_status(self) -> Dict[str, Any]:
        """Get current status for display."""
        if self.current_epsilon_idx < len(self.epsilon_values):
            return {
                "epsilon_index": self.current_epsilon_idx,
                "current_epsilon": self.epsilon_values[self.current_epsilon_idx],
                "total_epsilons": len(self.epsilon_values),
                "phases_completed": len(self.phase_results),
            }
        return {}
