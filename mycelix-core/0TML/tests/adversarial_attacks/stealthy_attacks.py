#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stealthy Byzantine Attacks - Testing PoGQ Against Sophisticated Evasion

These attacks are designed to evade detection by:
1. Adding noise to mask malicious intent
2. Gradually escalating attack intensity
3. Mimicking statistical properties of honest gradients

Goal: Find the REAL detection rate against attacks we didn't design
"""

import numpy as np
from typing import List, Dict, Tuple


class StealthyAttacks:
    """Collection of sophisticated Byzantine attacks"""

    @staticmethod
    def noise_masked_poisoning(honest_gradient: np.ndarray,
                                poison_intensity: float = 0.3,
                                noise_level: float = 0.4) -> np.ndarray:
        """
        Attack 1: Add malicious component masked by Gaussian noise

        Strategy:
        - Poison direction: opposite of honest gradient (hurts model)
        - Mask with significant noise to look statistically normal
        - Subtle enough that PoGQ might not detect

        Expected PoGQ score: 0.65-0.80 (borderline, might pass!)
        """
        # Malicious direction (opposite of what helps model)
        poison_direction = -honest_gradient

        # Add significant Gaussian noise
        noise = np.random.normal(0, noise_level, honest_gradient.shape)

        # Blend: mostly honest, some poison, lots of noise
        stealthy_gradient = (
            (1 - poison_intensity - 0.1) * honest_gradient +
            poison_intensity * poison_direction +
            0.1 * noise
        )

        return stealthy_gradient

    @staticmethod
    def slow_degradation(honest_gradient: np.ndarray,
                         round_num: int,
                         reputation_building_rounds: int = 30,
                         max_attack_intensity: float = 0.8) -> np.ndarray:
        """
        Attack 2: Behave honestly, then gradually poison

        Strategy:
        - Build reputation for first N rounds (fully honest)
        - Slowly increase attack intensity to avoid detection
        - By the time attack is strong, model is already damaged

        Expected detection: Rounds 1-30 (0%), Rounds 31-40 (maybe 50%), 41+ (70%)
        """
        if round_num < reputation_building_rounds:
            # Phase 1: Behave honestly to build reputation
            return honest_gradient

        # Phase 2: Slowly increase attack
        rounds_since_honest = round_num - reputation_building_rounds
        attack_intensity = min(
            max_attack_intensity,
            rounds_since_honest / 20.0  # Increase 5% per round
        )

        # Gradually degrade gradient quality
        degraded_gradient = honest_gradient * (1 - attack_intensity)

        return degraded_gradient

    @staticmethod
    def targeted_neuron_attack(honest_gradient: np.ndarray,
                                target_percentage: float = 0.05) -> np.ndarray:
        """
        Attack 3: Modify only small percentage of gradient (backdoor-like)

        Strategy:
        - Most of gradient is honest (95%)
        - Small portion (5%) is heavily poisoned
        - Statistical properties look mostly normal
        - Can inject backdoor without obvious anomaly

        Expected PoGQ score: 0.80-0.90 (likely passes!)
        """
        poisoned_gradient = honest_gradient.copy()

        # Select random subset of neurons to poison
        num_params = honest_gradient.size
        num_poisoned = int(num_params * target_percentage)

        # Random indices to poison
        poison_indices = np.random.choice(num_params, num_poisoned, replace=False)

        # Flatten for indexing
        flat_gradient = poisoned_gradient.flatten()

        # Strongly poison selected neurons
        flat_gradient[poison_indices] *= -10.0  # Reverse these specific weights

        # Reshape back
        poisoned_gradient = flat_gradient.reshape(honest_gradient.shape)

        return poisoned_gradient

    @staticmethod
    def statistical_mimicry(honest_gradients: List[np.ndarray],
                             target_mean_shift: float = 0.3) -> np.ndarray:
        """
        Attack 4: Generate gradient with statistical properties matching honest nodes

        Strategy:
        - Calculate mean/std of honest gradients
        - Generate malicious gradient with similar statistics
        - PoGQ uses statistical tests, so match the statistics

        Expected PoGQ score: 0.75-0.85 (might pass!)
        """
        # Calculate statistics of honest gradients
        honest_stack = np.stack(honest_gradients)
        mean_grad = np.mean(honest_stack, axis=0)
        std_grad = np.std(honest_stack, axis=0)

        # Generate malicious gradient with similar distribution
        # but shifted mean (to hurt model)
        malicious_gradient = np.random.normal(
            loc=mean_grad - target_mean_shift * std_grad,  # Shifted mean
            scale=std_grad,  # Same std as honest
            size=mean_grad.shape
        )

        return malicious_gradient

    @staticmethod
    def adaptive_noise_injection(honest_gradient: np.ndarray,
                                  previous_detections: List[bool],
                                  base_noise: float = 0.2) -> np.ndarray:
        """
        Attack 5: Learn from detection history and adapt

        Strategy:
        - If recently detected, reduce attack intensity
        - If not detected, increase attack intensity
        - Simple RL-like adaptation

        Expected detection: Decreases over time as attacker learns
        """
        # Calculate recent detection rate
        if len(previous_detections) > 0:
            recent_detection_rate = np.mean(previous_detections[-5:])
        else:
            recent_detection_rate = 0.5  # Assume 50% initially

        # If detection rate is high, be more subtle
        if recent_detection_rate > 0.7:
            attack_intensity = 0.1  # Very subtle
            noise_level = base_noise * 1.5  # More noise to mask
        elif recent_detection_rate > 0.4:
            attack_intensity = 0.3  # Moderate
            noise_level = base_noise
        else:
            attack_intensity = 0.5  # Aggressive (not getting caught)
            noise_level = base_noise * 0.5  # Less noise needed

        # Apply adaptive attack
        poison = -honest_gradient * attack_intensity
        noise = np.random.normal(0, noise_level, honest_gradient.shape)

        adaptive_gradient = honest_gradient + poison + noise

        return adaptive_gradient


def test_stealthy_attacks():
    """Quick test of stealthy attacks"""
    print("🎯 Testing Stealthy Byzantine Attacks")
    print("=" * 60)

    # Generate fake honest gradient for testing
    test_gradient = np.random.randn(100)

    attacks = StealthyAttacks()

    print("\n1. Noise-Masked Poisoning:")
    poisoned = attacks.noise_masked_poisoning(test_gradient)
    print(f"   Original norm: {np.linalg.norm(test_gradient):.3f}")
    print(f"   Poisoned norm: {np.linalg.norm(poisoned):.3f}")
    print(f"   Similarity: {np.dot(test_gradient, poisoned) / (np.linalg.norm(test_gradient) * np.linalg.norm(poisoned)):.3f}")

    print("\n2. Slow Degradation:")
    for round_num in [1, 20, 30, 35, 40, 50]:
        degraded = attacks.slow_degradation(test_gradient, round_num)
        similarity = np.dot(test_gradient, degraded) / (np.linalg.norm(test_gradient) * np.linalg.norm(degraded))
        print(f"   Round {round_num:2d}: similarity = {similarity:.3f}")

    print("\n3. Targeted Neuron Attack:")
    targeted = attacks.targeted_neuron_attack(test_gradient)
    diff = np.abs(test_gradient - targeted)
    print(f"   Modified neurons: {(diff > 1.0).sum()}/{len(test_gradient)}")
    print(f"   Max modification: {diff.max():.3f}")

    print("\n4. Statistical Mimicry:")
    honest_grads = [np.random.randn(100) for _ in range(5)]
    mimicked = attacks.statistical_mimicry(honest_grads, test_gradient)
    print(f"   Honest mean: {np.mean([g.mean() for g in honest_grads]):.3f}")
    print(f"   Mimicked mean: {mimicked.mean():.3f}")

    print("\n5. Adaptive Noise Injection:")
    detections = [True, True, False, True]  # Recently detected
    adaptive = attacks.adaptive_noise_injection(test_gradient, detections)
    print(f"   Detection rate: {np.mean(detections):.1%}")
    print(f"   Attack intensity: Subtle (due to high detection)")

    print("\n✅ All stealthy attacks implemented and tested")


if __name__ == "__main__":
    test_stealthy_attacks()
