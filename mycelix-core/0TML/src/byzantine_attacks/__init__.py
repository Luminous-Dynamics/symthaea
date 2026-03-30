# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Types - Comprehensive Taxonomy

This module provides implementations of all Byzantine attack types for testing
Byzantine-robust federated learning systems.

Attack Categories:
1. Basic Attacks: Random noise, sign flip, scaling
2. Obfuscated Attacks: Noise-masked, targeted neuron, adaptive stealth
3. Coordinated Attacks: Collusion, sybil
4. Stateful Attacks: Sleeper agent, reputation manipulation, adaptive learning

Usage:
    from byzantine_attacks import AttackType, create_attack

    # Create a specific attack
    attack = create_attack(AttackType.SLEEPER_AGENT, activation_round=5)

    # Generate malicious gradient
    byzantine_gradient = attack.generate(honest_gradient, round_num=3)
"""

from enum import Enum, auto
from typing import Protocol, Optional
import numpy as np


class AttackType(Enum):
    """
    Comprehensive taxonomy of Byzantine attack types

    Ordered by sophistication (low → very high)
    """
    # Category 1: Basic Gradient Manipulation
    RANDOM_NOISE = auto()       # Gaussian noise
    SIGN_FLIP = auto()          # Negate gradient
    SCALING = auto()            # Multiply by large constant

    # Category 2: Obfuscated Attacks
    NOISE_MASKED = auto()       # Poison masked by noise
    TARGETED_NEURON = auto()    # Backdoor (modify 5% of weights)
    ADAPTIVE_STEALTH = auto()   # Learn to evade thresholds

    # Category 3: Coordinated Attacks
    COORDINATED_COLLUSION = auto()  # Multiple Byzantine collaborate
    SYBIL = auto()                  # Single adversary, multiple identities

    # Category 4: Stateful/Adaptive Attacks
    SLEEPER_AGENT = auto()          # Honest → Byzantine after N rounds
    MODEL_POISONING = auto()        # Persistent backdoor injection
    REPUTATION_MANIPULATION = auto()  # Boost/tank reputation scores


class ByzantineAttack(Protocol):
    """
    Protocol for all Byzantine attack implementations

    All attacks must implement this interface for compatibility with testing framework
    """

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate malicious gradient for this round

        Args:
            honest_gradient: The honest gradient (used as reference)
            round_num: Current training round number
            **kwargs: Attack-specific parameters

        Returns:
            Byzantine gradient to submit
        """
        ...

    def reset(self) -> None:
        """Reset attack state (for stateful attacks)"""
        ...

    @property
    def name(self) -> str:
        """Human-readable attack name"""
        ...

    @property
    def sophistication(self) -> str:
        """Attack sophistication level (low, medium, high, very_high)"""
        ...


# Import all attack implementations
from .basic_attacks import (
    RandomNoiseAttack,
    SignFlipAttack,
    ScalingAttack,
)

from .obfuscated_attacks import (
    NoiseMaskedAttack,
    TargetedNeuronAttack,
    AdaptiveStealthAttack,
)

from .coordinated_attacks import (
    CoordinatedCollusionAttack,
    SybilAttack,
)

from .stateful_attacks import (
    SleeperAgentAttack,
    ModelPoisoningAttack,
    ReputationManipulationAttack,
)


def create_attack(attack_type: AttackType, **kwargs) -> ByzantineAttack:
    """
    Factory function to create attack instance

    Args:
        attack_type: Type of attack to create
        **kwargs: Attack-specific configuration

    Returns:
        Configured attack instance

    Example:
        >>> attack = create_attack(AttackType.SLEEPER_AGENT, activation_round=5)
        >>> byzantine_grad = attack.generate(honest_grad, round_num=3)
    """
    attack_map = {
        AttackType.RANDOM_NOISE: RandomNoiseAttack,
        AttackType.SIGN_FLIP: SignFlipAttack,
        AttackType.SCALING: ScalingAttack,
        AttackType.NOISE_MASKED: NoiseMaskedAttack,
        AttackType.TARGETED_NEURON: TargetedNeuronAttack,
        AttackType.ADAPTIVE_STEALTH: AdaptiveStealthAttack,
        AttackType.COORDINATED_COLLUSION: CoordinatedCollusionAttack,
        AttackType.SYBIL: SybilAttack,
        AttackType.SLEEPER_AGENT: SleeperAgentAttack,
        AttackType.MODEL_POISONING: ModelPoisoningAttack,
        AttackType.REPUTATION_MANIPULATION: ReputationManipulationAttack,
    }

    attack_class = attack_map.get(attack_type)
    if attack_class is None:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return attack_class(**kwargs)


__all__ = [
    # Enum
    "AttackType",
    # Protocol
    "ByzantineAttack",
    # Factory
    "create_attack",
    # Basic attacks
    "RandomNoiseAttack",
    "SignFlipAttack",
    "ScalingAttack",
    # Obfuscated attacks
    "NoiseMaskedAttack",
    "TargetedNeuronAttack",
    "AdaptiveStealthAttack",
    # Coordinated attacks
    "CoordinatedCollusionAttack",
    "SybilAttack",
    # Stateful attacks
    "SleeperAgentAttack",
    "ModelPoisoningAttack",
    "ReputationManipulationAttack",
]
