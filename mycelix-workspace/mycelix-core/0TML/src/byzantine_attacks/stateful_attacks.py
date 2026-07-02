# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stateful Byzantine Attacks - The Most Sophisticated Adversaries

These attacks maintain state across rounds and adapt their behavior over time.
They are the hardest to detect and represent real-world APT (Advanced Persistent Threat) scenarios.

Key Characteristics:
- Maintain internal state across training rounds
- Adapt behavior based on feedback
- May appear honest for extended periods
- Require temporal consistency detection to catch
"""

import numpy as np
from typing import Optional, List, Dict


class SleeperAgentAttack:
    """
    Sleeper Agent Attack - The Classic Delayed Adversary

    Behavior:
    - Rounds 1-N: Behaves honestly to build reputation
    - Rounds N+1+: Switches to Byzantine behavior (attacks the model)

    This tests the temporal consistency detector's ability to detect
    behavioral changes over time.

    Real-World Analogy:
    - Insider threat that waits to be trusted before attacking
    - Compromised node that activates after infiltration period
    - Long-term APT that builds cover before striking

    Detection Strategy:
    - Temporal Consistency Signal: Should detect sudden behavior change
    - Reputation System: Byzantine behavior should tank reputation quickly
    - PoGQ (Mode 1): Should detect poor gradient quality regardless of history
    """

    def __init__(
        self,
        activation_round: int = 5,
        honest_period_noise: float = 0.0,
        byzantine_attack_type: str = "sign_flip",
        byzantine_intensity: float = 1.0
    ):
        """
        Initialize Sleeper Agent attack

        Args:
            activation_round: Round number when attack activates (default: 5)
            honest_period_noise: Small noise during honest phase (default: 0.0)
            byzantine_attack_type: Type of attack after activation
                - "sign_flip": Negate gradient (most obvious)
                - "noise_masked": Poison masked by noise (subtle)
                - "scaling": Scale gradient (magnitude attack)
            byzantine_intensity: Strength of Byzantine attack (0-1, default: 1.0)
        """
        self.activation_round = activation_round
        self.honest_period_noise = honest_period_noise
        self.byzantine_attack_type = byzantine_attack_type
        self.byzantine_intensity = byzantine_intensity

        # State tracking
        self.current_round = 0
        self.is_activated = False
        self.honest_gradients_history: List[np.ndarray] = []

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate gradient based on current phase (honest vs. Byzantine)

        Args:
            honest_gradient: Reference honest gradient
            round_num: Current round number

        Returns:
            Gradient to submit (honest or Byzantine depending on phase)
        """
        self.current_round = round_num

        # Check activation
        if round_num >= self.activation_round and not self.is_activated:
            self.is_activated = True
            print(f"      ⚠️  SleeperAgent activated at round {round_num}!")

        # Phase 1: Honest behavior (build reputation)
        if not self.is_activated:
            gradient = self._honest_phase(honest_gradient)
            self.honest_gradients_history.append(honest_gradient.copy())
            return gradient

        # Phase 2: Byzantine behavior (attack!)
        return self._byzantine_phase(honest_gradient)

    def _honest_phase(self, honest_gradient: np.ndarray) -> np.ndarray:
        """
        Honest phase: Submit honest gradient (with optional small noise)

        The small noise simulates natural variation and makes the transition
        less obvious.
        """
        if self.honest_period_noise > 0:
            noise = np.random.normal(0, self.honest_period_noise, honest_gradient.shape)
            return honest_gradient + noise
        return honest_gradient.copy()

    def _byzantine_phase(self, honest_gradient: np.ndarray) -> np.ndarray:
        """
        Byzantine phase: Submit malicious gradient

        Strategy depends on configured attack type
        """
        if self.byzantine_attack_type == "sign_flip":
            # Classic: Negate gradient (most obvious, easiest to detect)
            return -honest_gradient * self.byzantine_intensity

        elif self.byzantine_attack_type == "noise_masked":
            # Subtle: Poison masked by noise
            poison = -honest_gradient
            noise = np.random.normal(0, 0.4, honest_gradient.shape)
            return (
                0.5 * honest_gradient +
                0.3 * poison * self.byzantine_intensity +
                0.2 * noise
            )

        elif self.byzantine_attack_type == "scaling":
            # Magnitude attack: Scale gradient
            scale_factor = 100.0 * self.byzantine_intensity
            return honest_gradient * scale_factor

        elif self.byzantine_attack_type == "adaptive_stealth":
            # Adaptive: Try to evade detection
            if len(self.honest_gradients_history) > 0:
                # Use historical honest gradients as reference
                ref_gradient = self.honest_gradients_history[-1]
                perturbation = np.random.normal(0, 0.1, honest_gradient.shape)
                return ref_gradient * 0.7 + perturbation
            else:
                # Fallback to subtle scaling
                return honest_gradient * 0.5

        else:
            raise ValueError(f"Unknown Byzantine attack type: {self.byzantine_attack_type}")

    def reset(self) -> None:
        """Reset attack state (for multi-run testing)"""
        self.current_round = 0
        self.is_activated = False
        self.honest_gradients_history = []

    @property
    def name(self) -> str:
        return f"Sleeper Agent (activate round {self.activation_round}, {self.byzantine_attack_type})"

    @property
    def sophistication(self) -> str:
        return "very_high"

    def get_detection_metrics(self) -> Dict[str, any]:
        """
        Get metrics for analyzing detection

        Returns:
            Dict with:
            - activation_round: When attack activated
            - rounds_honest: Number of honest rounds
            - rounds_byzantine: Number of Byzantine rounds
            - is_currently_byzantine: Current state
        """
        return {
            "activation_round": self.activation_round,
            "rounds_honest": min(self.current_round, self.activation_round),
            "rounds_byzantine": max(0, self.current_round - self.activation_round),
            "is_currently_byzantine": self.is_activated,
        }


class ModelPoisoningAttack:
    """
    Model Poisoning Attack - Persistent Backdoor Injection

    Behavior:
    - Inject subtle backdoor into model
    - Backdoor survives aggregation
    - Activates on specific trigger pattern

    Status: ⏳ Placeholder (to be implemented)
    """

    def __init__(self, trigger_pattern: Optional[np.ndarray] = None):
        self.trigger_pattern = trigger_pattern

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        # Placeholder implementation
        return honest_gradient * 0.9  # Subtle scaling

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Model Poisoning (placeholder)"

    @property
    def sophistication(self) -> str:
        return "very_high"


class ReputationManipulationAttack:
    """
    Reputation Manipulation Attack - Social Engineering

    Behavior:
    - Try to boost Byzantine node reputations
    - Try to tank honest node reputations
    - Requires coordination with other Byzantine nodes

    Status: ⏳ Placeholder (to be implemented)
    """

    def __init__(self, target_nodes: Optional[List[int]] = None):
        self.target_nodes = target_nodes or []

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        # Placeholder implementation
        return honest_gradient * 0.95  # Subtle manipulation

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Reputation Manipulation (placeholder)"

    @property
    def sophistication(self) -> str:
        return "very_high"


# Example usage and testing
if __name__ == "__main__":
    print("Testing SleeperAgentAttack...")

    # Create attack that activates at round 5
    attack = SleeperAgentAttack(
        activation_round=5,
        byzantine_attack_type="sign_flip"
    )

    # Simulate honest gradient
    honest_grad = np.random.randn(100)

    # Simulate 10 training rounds
    for round_num in range(1, 11):
        grad = attack.generate(honest_grad, round_num)

        # Check gradient similarity
        similarity = np.dot(grad, honest_grad) / (
            np.linalg.norm(grad) * np.linalg.norm(honest_grad)
        )

        metrics = attack.get_detection_metrics()

        print(f"Round {round_num}:")
        print(f"  State: {'BYZANTINE' if metrics['is_currently_byzantine'] else 'HONEST'}")
        print(f"  Similarity to honest: {similarity:.3f}")
        print(f"  Gradient norm: {np.linalg.norm(grad):.3f}")

    print("\n✅ SleeperAgentAttack test complete!")
    print(f"   Detection window: Rounds {attack.activation_round}-{attack.current_round}")
    print(f"   Temporal signal should detect sudden change at round {attack.activation_round}")
