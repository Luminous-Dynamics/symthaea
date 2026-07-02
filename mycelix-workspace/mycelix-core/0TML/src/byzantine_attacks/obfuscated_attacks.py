# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Obfuscated Byzantine Attacks - Medium Sophistication

These attacks attempt to hide malicious intent through obfuscation:
- Noise-masked poisoning
- Targeted neuron attacks (backdoors)
- Adaptive stealth

Detectability: Medium
Sophistication: Medium-High
Real-World: Sophisticated attackers, backdoor injection
"""

import numpy as np


class NoiseMaskedAttack:
    """
    Noise-Masked Poisoning - Hide Malicious Intent

    Behavior:
    - Mix honest gradient + poison + noise
    - Noise makes poison less obvious
    - Harder to detect than pure sign flip

    Detection:
    - Moderate difficulty (ensemble detector needed)
    - Similarity signal: somewhat low (mixed directions)
    - Temporal signal: may show inconsistency
    - PoGQ: worse quality than honest

    Real-World:
    - Sophisticated attacker hiding malicious intent
    - Trying to evade simple detection
    """

    def __init__(
        self,
        honest_weight: float = 0.5,
        poison_weight: float = 0.3,
        noise_weight: float = 0.2,
        noise_std: float = 0.4
    ):
        """
        Args:
            honest_weight: Weight for honest component (default: 0.5)
            poison_weight: Weight for poison component (default: 0.3)
            noise_weight: Weight for noise component (default: 0.2)
            noise_std: Std dev of noise (default: 0.4)
        """
        self.honest_weight = honest_weight
        self.poison_weight = poison_weight
        self.noise_weight = noise_weight
        self.noise_std = noise_std

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """Generate noise-masked poisoned gradient"""
        poison = -honest_gradient  # Flip direction
        noise = np.random.normal(0, self.noise_std, honest_gradient.shape)

        return (
            self.honest_weight * honest_gradient +
            self.poison_weight * poison +
            self.noise_weight * noise
        )

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Noise-Masked ({self.honest_weight}/{self.poison_weight}/{self.noise_weight})"

    @property
    def sophistication(self) -> str:
        return "medium"


class TargetedNeuronAttack:
    """
    Targeted Neuron Attack - Backdoor Injection

    Behavior:
    - Keep most of gradient honest (95%)
    - Modify small subset of weights (5%)
    - Create backdoor trigger

    Detection:
    - Hard to detect (mostly honest gradient)
    - Similarity signal: very high (95% honest)
    - Magnitude signal: may be normal
    - PoGQ: quality depends on which neurons modified

    Real-World:
    - Backdoor injection
    - Trigger-based attacks
    - Model trojan
    """

    def __init__(
        self,
        modify_percentage: float = 0.05,
        modification_scale: float = -10.0
    ):
        """
        Args:
            modify_percentage: Fraction of gradient to modify (default: 0.05 = 5%)
            modification_scale: Scale factor for modified weights (default: -10.0)
        """
        self.modify_percentage = modify_percentage
        self.modification_scale = modification_scale

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """Generate backdoor gradient (modify 5% of weights)"""
        attack_grad = honest_gradient.copy()

        # Select random subset to modify
        num_to_modify = int(len(attack_grad) * self.modify_percentage)
        if num_to_modify > 0:
            indices = np.random.choice(len(attack_grad), num_to_modify, replace=False)
            attack_grad[indices] *= self.modification_scale

        return attack_grad

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Targeted Neuron ({self.modify_percentage*100:.0f}%, ×{self.modification_scale})"

    @property
    def sophistication(self) -> str:
        return "medium_high"


class AdaptiveStealthAttack:
    """
    Adaptive Stealth Attack - Learn to Evade Detection

    Behavior:
    - Small perturbations to stay below detection thresholds
    - Adapt based on what gets through
    - Try to learn detection boundaries

    Detection:
    - Hard to detect (designed to evade thresholds)
    - Similarity signal: may be high (subtle changes)
    - Temporal signal: may show consistency if well-tuned
    - PoGQ: slightly worse quality

    Real-World:
    - Sophisticated adversary with ML knowledge
    - Adaptive attack that learns system defenses
    - Evasion optimization
    """

    def __init__(
        self,
        perturbation_std: float = 0.1,
        gradient_scale: float = 0.7,
        adaptation_rate: float = 0.1
    ):
        """
        Args:
            perturbation_std: Std dev of perturbation noise (default: 0.1)
            gradient_scale: Scale honest gradient (default: 0.7)
            adaptation_rate: How fast to adapt (default: 0.1)
        """
        self.perturbation_std = perturbation_std
        self.gradient_scale = gradient_scale
        self.adaptation_rate = adaptation_rate

        # State: Track what worked
        self.successful_perturbations = []

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        was_detected: bool = False,  # Feedback signal
        **kwargs
    ) -> np.ndarray:
        """
        Generate stealthy attack gradient

        Args:
            honest_gradient: Honest gradient
            round_num: Current round
            was_detected: Feedback - was previous gradient detected?
        """
        # Adapt based on feedback
        if was_detected and len(self.successful_perturbations) > 0:
            # Last perturbation was detected, try something different
            self.perturbation_std *= 0.9  # Reduce perturbation
            self.gradient_scale = min(1.0, self.gradient_scale + 0.1)  # More honest

        # Generate adaptive perturbation
        perturbation = np.random.normal(0, self.perturbation_std, honest_gradient.shape)

        attack_grad = honest_gradient * self.gradient_scale + perturbation

        # Track successful perturbations
        if not was_detected:
            self.successful_perturbations.append(perturbation)

        return attack_grad

    def reset(self) -> None:
        """Reset adaptive state"""
        self.successful_perturbations = []

    @property
    def name(self) -> str:
        return f"Adaptive Stealth (σ={self.perturbation_std:.2f}, scale={self.gradient_scale:.2f})"

    @property
    def sophistication(self) -> str:
        return "high"


# Example usage
if __name__ == "__main__":
    print("Testing Obfuscated Byzantine Attacks...")

    # Create honest gradient
    honest_grad = np.random.randn(1000)
    honest_norm = np.linalg.norm(honest_grad)

    attacks = [
        NoiseMaskedAttack(),
        TargetedNeuronAttack(modify_percentage=0.05),
        AdaptiveStealthAttack(),
    ]

    for attack in attacks:
        print(f"\n{attack.name}:")

        # Generate attack gradient
        attack_grad = attack.generate(honest_grad, round_num=1)

        # Metrics
        attack_norm = np.linalg.norm(attack_grad)
        similarity = np.dot(attack_grad, honest_grad) / (attack_norm * honest_norm)
        norm_ratio = attack_norm / honest_norm

        print(f"  Similarity: {similarity:.3f}")
        print(f"  Norm ratio: {norm_ratio:.3f}")

        # Detection likelihood
        if similarity < 0.7:
            print(f"  Similarity detector: Likely detected")
        else:
            print(f"  Similarity detector: May evade")

        if norm_ratio > 2.0 or norm_ratio < 0.5:
            print(f"  Magnitude detector: Likely detected")
        else:
            print(f"  Magnitude detector: May evade")

    print("\n✅ Obfuscated attacks test complete!")
