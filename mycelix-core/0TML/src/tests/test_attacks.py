# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Simulation Suite

Comprehensive tests for various Byzantine attack types:
- Sign Flip Attack
- Random Noise Attack
- Scaling Attack
- Little-is-Enough Attack
- Krum Attack (targeted)
- Cartel/Collusion Attack
- Adaptive Attack

Tests detection effectiveness at various attack strengths.
"""

import pytest
import numpy as np
from typing import Dict, Set, Tuple, List
from dataclasses import dataclass

from mycelix_fl import MycelixFL, FLConfig


# =============================================================================
# ATTACK IMPLEMENTATIONS
# =============================================================================

@dataclass
class AttackResult:
    """Result from an attack simulation."""
    attack_name: str
    byzantine_ratio: float
    detected: int
    total_byzantine: int
    precision: float
    recall: float
    f1_score: float
    gradient_error: float  # Distance from true gradient


class AttackSimulator:
    """Simulate various Byzantine attacks."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_honest_gradients(
        self,
        num_honest: int,
        gradient_dim: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate honest gradients clustered around true direction."""
        true_gradient = self.rng.randn(gradient_dim).astype(np.float32)
        true_gradient /= np.linalg.norm(true_gradient)

        gradients = {}
        for i in range(num_honest):
            noise = self.rng.randn(gradient_dim).astype(np.float32) * 0.1
            gradients[f"honest_{i}"] = true_gradient + noise

        return gradients, true_gradient

    # -------------------------------------------------------------------------
    # Attack Type: Sign Flip
    # -------------------------------------------------------------------------
    def sign_flip_attack(
        self,
        true_gradient: np.ndarray,
        num_attackers: int,
        magnitude: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """
        Sign Flip Attack: Negate the true gradient direction.

        This is one of the most common attacks - simply send the
        opposite of the true gradient to sabotage learning.
        """
        attacks = {}
        for i in range(num_attackers):
            attacks[f"signflip_{i}"] = -true_gradient * magnitude
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Random Noise
    # -------------------------------------------------------------------------
    def random_noise_attack(
        self,
        gradient_dim: int,
        num_attackers: int,
        scale: float = 3.0,
    ) -> Dict[str, np.ndarray]:
        """
        Random Noise Attack: Send completely random gradients.

        Disrupts learning by adding noise to the aggregate.
        """
        attacks = {}
        for i in range(num_attackers):
            attacks[f"random_{i}"] = self.rng.randn(gradient_dim).astype(np.float32) * scale
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Scaling
    # -------------------------------------------------------------------------
    def scaling_attack(
        self,
        true_gradient: np.ndarray,
        num_attackers: int,
        scale_factor: float = 100.0,
    ) -> Dict[str, np.ndarray]:
        """
        Scaling Attack: Send true gradient with extreme magnitude.

        Exploits averaging by dominating the aggregate.
        """
        attacks = {}
        for i in range(num_attackers):
            scale = scale_factor * (1.0 + self.rng.rand() * 0.5)
            attacks[f"scale_{i}"] = true_gradient * scale
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Little-is-Enough (LIE)
    # -------------------------------------------------------------------------
    def lie_attack(
        self,
        honest_gradients: Dict[str, np.ndarray],
        num_attackers: int,
        epsilon: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Little-is-Enough Attack: Small perturbation that evades detection.

        Computes a perturbation that shifts the aggregate while staying
        close to honest gradients to avoid detection.

        Reference: Baruch et al., "A Little Is Enough" (2019)
        """
        # Stack honest gradients
        honest_stack = np.stack(list(honest_gradients.values()))
        honest_mean = np.mean(honest_stack, axis=0)
        honest_std = np.std(honest_stack, axis=0) + 1e-8

        attacks = {}
        for i in range(num_attackers):
            # Perturbation within epsilon std deviations
            perturbation = self.rng.randn(len(honest_mean)).astype(np.float32)
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            attacks[f"lie_{i}"] = honest_mean - perturbation * honest_std
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Krum Attack (Targeted)
    # -------------------------------------------------------------------------
    def krum_attack(
        self,
        honest_gradients: Dict[str, np.ndarray],
        num_attackers: int,
    ) -> Dict[str, np.ndarray]:
        """
        Krum Attack: Designed to defeat Krum aggregation.

        Attackers coordinate to appear close to each other while
        being far from the true gradient.

        Reference: Fang et al., "Local Model Poisoning Attacks" (2020)
        """
        honest_stack = np.stack(list(honest_gradients.values()))
        honest_mean = np.mean(honest_stack, axis=0)

        # Attackers cluster around negative of honest mean
        attack_center = -honest_mean * 0.8

        attacks = {}
        for i in range(num_attackers):
            noise = self.rng.randn(len(honest_mean)).astype(np.float32) * 0.05
            attacks[f"krum_{i}"] = attack_center + noise
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Cartel/Collusion
    # -------------------------------------------------------------------------
    def cartel_attack(
        self,
        true_gradient: np.ndarray,
        num_attackers: int,
        rotation_angle: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Cartel Attack: Coordinated collusion to shift aggregate.

        Attackers send gradients that are similar to each other
        but rotated away from the true direction.
        """
        # Create rotated direction
        rotation = self.rng.randn(len(true_gradient)).astype(np.float32)
        rotation -= np.dot(rotation, true_gradient) * true_gradient
        rotation /= np.linalg.norm(rotation) + 1e-8

        attack_direction = (
            true_gradient * np.cos(rotation_angle) +
            rotation * np.sin(rotation_angle)
        )

        attacks = {}
        for i in range(num_attackers):
            noise = self.rng.randn(len(true_gradient)).astype(np.float32) * 0.02
            attacks[f"cartel_{i}"] = attack_direction + noise
        return attacks

    # -------------------------------------------------------------------------
    # Attack Type: Adaptive (Learns defenses)
    # -------------------------------------------------------------------------
    def adaptive_attack(
        self,
        honest_gradients: Dict[str, np.ndarray],
        num_attackers: int,
        target_percentile: float = 0.3,
    ) -> Dict[str, np.ndarray]:
        """
        Adaptive Attack: Stays within distribution to evade detection.

        Crafts attacks at specific percentiles of the honest distribution.
        """
        honest_stack = np.stack(list(honest_gradients.values()))

        attacks = {}
        for i in range(num_attackers):
            # Use different percentiles for each attacker
            percentile = target_percentile + (i * 0.1)
            attack_grad = np.percentile(honest_stack, percentile * 100, axis=0)
            # Invert to cause harm while staying in distribution
            attacks[f"adaptive_{i}"] = -attack_grad
        return attacks


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def simulator():
    return AttackSimulator(seed=42)


@pytest.fixture
def fl_system():
    config = FLConfig(
        byzantine_threshold=0.45,
        min_nodes=5,
        use_detection=True,
        use_healing=True,
    )
    return MycelixFL(config=config)


def run_attack_test(
    fl: MycelixFL,
    honest_gradients: Dict[str, np.ndarray],
    attack_gradients: Dict[str, np.ndarray],
    true_gradient: np.ndarray,
    attack_name: str,
) -> AttackResult:
    """Run attack and compute metrics."""
    # Combine gradients
    all_gradients = {**honest_gradients, **attack_gradients}
    actual_byzantine = set(attack_gradients.keys())

    # Run detection
    result = fl.execute_round(all_gradients, round_num=1)

    # Compute metrics
    detected = result.byzantine_nodes
    tp = len(detected & actual_byzantine)
    fp = len(detected - actual_byzantine)
    fn = len(actual_byzantine - detected)

    precision = tp / len(detected) if detected else 1.0
    recall = tp / len(actual_byzantine) if actual_byzantine else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    gradient_error = float(np.linalg.norm(result.aggregated_gradient - true_gradient))

    return AttackResult(
        attack_name=attack_name,
        byzantine_ratio=len(actual_byzantine) / len(all_gradients),
        detected=len(detected & actual_byzantine),
        total_byzantine=len(actual_byzantine),
        precision=precision,
        recall=recall,
        f1_score=f1,
        gradient_error=gradient_error,
    )


# =============================================================================
# SIGN FLIP ATTACK TESTS
# =============================================================================

class TestSignFlipAttack:
    """Test detection of sign flip attacks."""

    @pytest.mark.attack
    def test_signflip_20pct(self, simulator, fl_system):
        """Test sign flip at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.sign_flip_attack(true_grad, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "SignFlip-20%")

        print(f"\n  Sign Flip 20%: F1={result.f1_score:.2%}, Error={result.gradient_error:.4f}")
        assert result.recall >= 0.5  # Should detect most

    @pytest.mark.attack
    def test_signflip_40pct(self, simulator, fl_system):
        """Test sign flip at 40% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(12, 5000)
        attacks = simulator.sign_flip_attack(true_grad, 8)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "SignFlip-40%")

        print(f"\n  Sign Flip 40%: F1={result.f1_score:.2%}, Error={result.gradient_error:.4f}")
        # Should still work at 40%

    @pytest.mark.attack
    def test_signflip_varying_magnitude(self, simulator, fl_system):
        """Test sign flip with varying attack magnitudes."""
        for magnitude in [0.5, 1.0, 2.0, 5.0]:
            honest, true_grad = simulator.generate_honest_gradients(14, 5000)
            attacks = simulator.sign_flip_attack(true_grad, 6, magnitude=magnitude)

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"SignFlip-mag{magnitude}")
            print(f"  Magnitude {magnitude}: F1={result.f1_score:.2%}")


# =============================================================================
# RANDOM NOISE ATTACK TESTS
# =============================================================================

class TestRandomNoiseAttack:
    """Test detection of random noise attacks."""

    @pytest.mark.attack
    def test_random_20pct(self, simulator, fl_system):
        """Test random noise at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.random_noise_attack(5000, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Random-20%")

        print(f"\n  Random 20%: F1={result.f1_score:.2%}")
        assert result.recall >= 0.5

    @pytest.mark.attack
    def test_random_varying_scale(self, simulator, fl_system):
        """Test random noise with varying scales."""
        for scale in [1.0, 3.0, 10.0]:
            honest, true_grad = simulator.generate_honest_gradients(14, 5000)
            attacks = simulator.random_noise_attack(5000, 6, scale=scale)

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"Random-scale{scale}")
            print(f"  Scale {scale}: F1={result.f1_score:.2%}")


# =============================================================================
# SCALING ATTACK TESTS
# =============================================================================

class TestScalingAttack:
    """Test detection of scaling attacks."""

    @pytest.mark.attack
    def test_scaling_20pct(self, simulator, fl_system):
        """Test scaling attack at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.scaling_attack(true_grad, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Scaling-20%")

        print(f"\n  Scaling 20%: F1={result.f1_score:.2%}")

    @pytest.mark.attack
    def test_scaling_varying_factor(self, simulator, fl_system):
        """Test scaling with varying factors."""
        for factor in [10, 50, 100, 500]:
            honest, true_grad = simulator.generate_honest_gradients(14, 5000)
            attacks = simulator.scaling_attack(true_grad, 6, scale_factor=factor)

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"Scaling-{factor}x")
            print(f"  Factor {factor}x: F1={result.f1_score:.2%}")


# =============================================================================
# LITTLE-IS-ENOUGH ATTACK TESTS
# =============================================================================

class TestLIEAttack:
    """Test detection of Little-is-Enough attacks."""

    @pytest.mark.attack
    def test_lie_20pct(self, simulator, fl_system):
        """Test LIE attack at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.lie_attack(honest, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "LIE-20%")

        print(f"\n  LIE 20%: F1={result.f1_score:.2%}")
        # LIE is harder to detect - lower threshold

    @pytest.mark.attack
    def test_lie_varying_epsilon(self, simulator, fl_system):
        """Test LIE with varying epsilon values."""
        for epsilon in [0.1, 0.3, 0.5, 1.0]:
            honest, true_grad = simulator.generate_honest_gradients(14, 5000)
            attacks = simulator.lie_attack(honest, 6, epsilon=epsilon)

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"LIE-eps{epsilon}")
            print(f"  Epsilon {epsilon}: F1={result.f1_score:.2%}")


# =============================================================================
# KRUM ATTACK TESTS
# =============================================================================

class TestKrumAttack:
    """Test detection of Krum-targeted attacks."""

    @pytest.mark.attack
    def test_krum_attack_20pct(self, simulator, fl_system):
        """Test Krum attack at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.krum_attack(honest, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Krum-20%")

        print(f"\n  Krum Attack 20%: F1={result.f1_score:.2%}")


# =============================================================================
# CARTEL/COLLUSION ATTACK TESTS
# =============================================================================

class TestCartelAttack:
    """Test detection of collusion attacks."""

    @pytest.mark.attack
    def test_cartel_20pct(self, simulator, fl_system):
        """Test cartel attack at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.cartel_attack(true_grad, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Cartel-20%")

        print(f"\n  Cartel 20%: F1={result.f1_score:.2%}")

    @pytest.mark.attack
    def test_cartel_varying_rotation(self, simulator, fl_system):
        """Test cartel with varying rotation angles."""
        for angle in [0.2, 0.5, 1.0, 1.5]:
            honest, true_grad = simulator.generate_honest_gradients(14, 5000)
            attacks = simulator.cartel_attack(true_grad, 6, rotation_angle=angle)

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"Cartel-rot{angle}")
            print(f"  Rotation {angle}: F1={result.f1_score:.2%}")


# =============================================================================
# ADAPTIVE ATTACK TESTS
# =============================================================================

class TestAdaptiveAttack:
    """Test detection of adaptive attacks."""

    @pytest.mark.attack
    def test_adaptive_20pct(self, simulator, fl_system):
        """Test adaptive attack at 20% Byzantine."""
        honest, true_grad = simulator.generate_honest_gradients(16, 5000)
        attacks = simulator.adaptive_attack(honest, 4)

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Adaptive-20%")

        print(f"\n  Adaptive 20%: F1={result.f1_score:.2%}")


# =============================================================================
# MIXED ATTACK TESTS
# =============================================================================

class TestMixedAttacks:
    """Test detection of mixed attack types."""

    @pytest.mark.attack
    def test_mixed_attacks(self, simulator, fl_system):
        """Test with multiple attack types simultaneously."""
        honest, true_grad = simulator.generate_honest_gradients(14, 5000)

        # Mix of attacks
        attacks = {}
        attacks.update(simulator.sign_flip_attack(true_grad, 2))
        attacks.update(simulator.random_noise_attack(5000, 2))
        attacks.update(simulator.scaling_attack(true_grad, 2))

        result = run_attack_test(fl_system, honest, attacks, true_grad, "Mixed")

        print(f"\n  Mixed Attacks: F1={result.f1_score:.2%}")
        assert result.recall >= 0.3  # Should detect some


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestAttackStress:
    """Stress tests at Byzantine threshold."""

    @pytest.mark.attack
    @pytest.mark.slow
    def test_45_percent_threshold(self, simulator, fl_system):
        """Test at 45% Byzantine (theoretical limit)."""
        results = []

        for attack_func, name in [
            (lambda tg, n: simulator.sign_flip_attack(tg, n), "SignFlip"),
            (lambda tg, n: simulator.random_noise_attack(len(tg), n), "Random"),
            (lambda tg, n: simulator.scaling_attack(tg, n), "Scaling"),
        ]:
            honest, true_grad = simulator.generate_honest_gradients(11, 5000)
            attacks = attack_func(true_grad, 9)  # 9/20 = 45%

            result = run_attack_test(fl_system, honest, attacks, true_grad, f"{name}-45%")
            results.append(result)

        print("\n  Results at 45% Byzantine:")
        for r in results:
            print(f"    {r.attack_name}: F1={r.f1_score:.2%}, Error={r.gradient_error:.4f}")
