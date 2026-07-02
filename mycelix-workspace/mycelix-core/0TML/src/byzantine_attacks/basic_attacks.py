# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Basic Byzantine Attacks - Simple Gradient Manipulation

These are the simplest forms of Byzantine attacks:
- Random noise
- Sign flip (gradient negation)
- Scaling (multiply by large constant)

Detectability: Very High
Sophistication: Low
Real-World: Accidental corruption, simple malicious attacks
"""

import numpy as np


class RandomNoiseAttack:
    """
    Random Noise Attack - Gaussian Noise Gradients

    Behavior: Submit pure random noise instead of gradient

    Detection:
    - Very easy to detect (similarity signal)
    - Magnitude often wrong (magnitude signal)
    - PoGQ: Terrible quality score

    Real-World:
    - Network corruption
    - Hardware failure
    - Simple denial-of-service
    """

    def __init__(self, noise_std: float = 1.0):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise (default: 1.0)
        """
        self.noise_std = noise_std

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """Generate pure random noise"""
        return np.random.normal(0, self.noise_std, honest_gradient.shape)

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Random Noise (σ={self.noise_std})"

    @property
    def sophistication(self) -> str:
        return "low"


class SignFlipAttack:
    """
    Sign Flip Attack - Negate Gradient

    Behavior: Submit -1 * honest_gradient

    Detection:
    - Very easy to detect (similarity signal: cosine ≈ -1)
    - PoGQ: Very poor quality score

    Real-World:
    - Classic Byzantine attack from BFT literature
    - Intentional model sabotage
    - Gradient ascent instead of descent
    """

    def __init__(self, flip_intensity: float = 1.0):
        """
        Args:
            flip_intensity: Scaling factor for flipped gradient (default: 1.0)
        """
        self.flip_intensity = flip_intensity

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """Negate the honest gradient"""
        return -honest_gradient * self.flip_intensity

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Sign Flip (intensity={self.flip_intensity})"

    @property
    def sophistication(self) -> str:
        return "low"


class ScalingAttack:
    """
    Scaling Attack - Multiply Gradient by Large Constant

    Behavior: Submit honest_gradient * large_scale

    Detection:
    - Easy to detect (magnitude signal: norm >> average)
    - Similarity may be good (direction preserved)
    - PoGQ: May have good quality if direction is right

    Real-World:
    - Amplification attack
    - Try to dominate aggregation
    - Gradient explosion
    """

    def __init__(self, scale_factor: float = 100.0):
        """
        Args:
            scale_factor: Scaling multiplier (default: 100.0)
        """
        self.scale_factor = scale_factor

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """Scale the honest gradient"""
        return honest_gradient * self.scale_factor

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Scaling (×{self.scale_factor})"

    @property
    def sophistication(self) -> str:
        return "low"


# Example usage
if __name__ == "__main__":
    print("Testing Basic Byzantine Attacks...")

    # Create honest gradient
    honest_grad = np.random.randn(100)
    honest_norm = np.linalg.norm(honest_grad)

    attacks = [
        RandomNoiseAttack(noise_std=1.0),
        SignFlipAttack(flip_intensity=1.0),
        ScalingAttack(scale_factor=100.0),
    ]

    for attack in attacks:
        print(f"\n{attack.name}:")

        # Generate attack gradient
        attack_grad = attack.generate(honest_grad, round_num=1)

        # Metrics
        attack_norm = np.linalg.norm(attack_grad)
        similarity = np.dot(attack_grad, honest_grad) / (attack_norm * honest_norm)

        print(f"  Honest norm: {honest_norm:.3f}")
        print(f"  Attack norm: {attack_norm:.3f}")
        print(f"  Similarity: {similarity:.3f}")

        # Detection signals
        if abs(similarity) < 0.5:
            print(f"  ✅ Similarity signal: DETECTED (|cos| < 0.5)")
        else:
            print(f"  ❌ Similarity signal: missed")

        if attack_norm > 3 * honest_norm:
            print(f"  ✅ Magnitude signal: DETECTED (norm > 3σ)")
        else:
            print(f"  ❌ Magnitude signal: missed")

    print("\n✅ Basic attacks test complete!")
