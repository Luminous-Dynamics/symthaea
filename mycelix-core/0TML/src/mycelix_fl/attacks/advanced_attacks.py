# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Advanced Byzantine Attack Implementations

Sophisticated attack strategies for testing Byzantine resilience:
- Adaptive attacks that learn from detection
- Coordinated cartel attacks
- Backdoor attacks with triggers
- Model poisoning attacks
- Free-rider attacks

These attacks are for TESTING PURPOSES ONLY to validate detection systems.

Author: Luminous Dynamics
Date: December 31, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of Byzantine attacks."""
    GRADIENT_SCALING = "gradient_scaling"
    SIGN_FLIP = "sign_flip"
    GAUSSIAN_NOISE = "gaussian_noise"
    LABEL_FLIP = "label_flip"
    BACKDOOR = "backdoor"
    FREE_RIDER = "free_rider"
    ADAPTIVE = "adaptive"
    CARTEL = "cartel"
    LITTLE_IS_ENOUGH = "little_is_enough"
    INNER_PRODUCT_MANIPULATION = "ipm"


@dataclass
class AttackConfig:
    """Configuration for Byzantine attacks."""
    attack_type: AttackType
    intensity: float = 1.0  # Attack strength multiplier
    stealth: float = 0.5  # How hidden the attack is (0=obvious, 1=very stealthy)
    target_accuracy_drop: float = 0.3  # Target model accuracy reduction
    adaptation_rate: float = 0.1  # How fast adaptive attacks learn
    cartel_size: int = 3  # Number of coordinated attackers
    backdoor_trigger: Optional[np.ndarray] = None
    random_seed: Optional[int] = None


@dataclass
class AttackResult:
    """Result of applying an attack."""
    original_gradient: np.ndarray
    attacked_gradient: np.ndarray
    attack_type: AttackType
    detected: bool = False
    detection_confidence: float = 0.0
    stealth_score: float = 0.0  # How stealthy the attack was


# =============================================================================
# BASE ATTACK CLASS
# =============================================================================

class ByzantineAttack:
    """Base class for Byzantine attacks."""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._detection_history: List[bool] = []

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Apply attack to gradient. Override in subclasses."""
        raise NotImplementedError

    def record_detection(self, detected: bool) -> None:
        """Record whether attack was detected (for adaptive attacks)."""
        self._detection_history.append(detected)

    @property
    def detection_rate(self) -> float:
        """Get historical detection rate."""
        if not self._detection_history:
            return 0.0
        return sum(self._detection_history) / len(self._detection_history)


# =============================================================================
# GRADIENT SCALING ATTACK
# =============================================================================

class GradientScalingAttack(ByzantineAttack):
    """
    Scale gradients to overwhelm aggregation.

    Strategy: Multiply gradients by large factor to dominate the average.
    Stealth: Lower scaling is harder to detect.
    """

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Scale based on intensity and stealth
        # High stealth = smaller scale to avoid detection
        base_scale = 10.0 * self.config.intensity
        stealth_factor = 1.0 - self.config.stealth * 0.9  # At max stealth, only 10% of base
        scale = base_scale * stealth_factor

        return gradient * scale


# =============================================================================
# SIGN FLIP ATTACK
# =============================================================================

class SignFlipAttack(ByzantineAttack):
    """
    Flip gradient signs to reverse learning direction.

    Strategy: Negate gradients so model learns wrong direction.
    Stealth: Partial flipping is harder to detect.
    """

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Determine flip probability based on stealth
        # High stealth = flip fewer elements
        flip_prob = 1.0 - self.config.stealth * 0.7

        if flip_prob >= 0.99:
            # Full flip
            return -gradient * self.config.intensity
        else:
            # Partial flip
            mask = self.rng.random(gradient.shape) < flip_prob
            result = gradient.copy()
            result[mask] = -result[mask] * self.config.intensity
            return result


# =============================================================================
# GAUSSIAN NOISE ATTACK
# =============================================================================

class GaussianNoiseAttack(ByzantineAttack):
    """
    Add Gaussian noise to corrupt gradients.

    Strategy: Add random noise to prevent convergence.
    Stealth: Lower noise magnitude is harder to detect.
    """

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Noise std based on gradient magnitude, intensity, and stealth
        grad_std = np.std(gradient) + 1e-8
        noise_std = grad_std * self.config.intensity * (1.0 - self.config.stealth * 0.8)

        noise = self.rng.normal(0, noise_std, gradient.shape).astype(gradient.dtype)
        return gradient + noise


# =============================================================================
# LITTLE IS ENOUGH ATTACK
# =============================================================================

class LittleIsEnoughAttack(ByzantineAttack):
    """
    Subtle attack that stays just under detection threshold.

    Strategy: Small perturbations that accumulate over time.
    Based on: "A Little Is Enough" (Baruch et al., NeurIPS 2019)
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.perturbation_history: List[np.ndarray] = []

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Get honest gradients from context if available
        honest_mean = context.get("honest_mean") if context else None
        honest_std = context.get("honest_std") if context else None

        if honest_mean is None:
            # Estimate from gradient itself
            honest_mean = gradient
            honest_std = np.abs(gradient) * 0.1 + 1e-8

        # Stay within detection threshold (typically 2-3 std)
        threshold_multiplier = 1.5 + self.config.stealth  # 1.5 to 2.5 std
        max_perturbation = honest_std * threshold_multiplier

        # Create perturbation in opposite direction of mean
        perturbation = -np.sign(honest_mean) * max_perturbation * self.config.intensity

        # Apply stealth: blend with original
        blend = 1.0 - self.config.stealth * 0.5
        result = gradient + perturbation * blend

        self.perturbation_history.append(perturbation)
        return result


# =============================================================================
# INNER PRODUCT MANIPULATION (IPM) ATTACK
# =============================================================================

class IPMAttack(ByzantineAttack):
    """
    Inner Product Manipulation attack.

    Strategy: Craft gradient to have negative inner product with true gradient.
    Based on: "Manipulating the Byzantine" (Xie et al., ICML 2020)
    """

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Get estimated true gradient direction
        true_direction = context.get("true_gradient") if context else gradient

        # Compute component to remove
        inner_product = np.dot(gradient.flatten(), true_direction.flatten())
        norm_sq = np.dot(true_direction.flatten(), true_direction.flatten()) + 1e-8

        # Project out true direction and flip
        projection = (inner_product / norm_sq) * true_direction
        orthogonal = gradient - projection

        # Attack: flip the projection component
        attack_strength = self.config.intensity * (1.0 - self.config.stealth * 0.5)
        attacked = orthogonal - projection * attack_strength

        return attacked.astype(gradient.dtype)


# =============================================================================
# FREE-RIDER ATTACK
# =============================================================================

class FreeRiderAttack(ByzantineAttack):
    """
    Free-rider attack: contribute nothing while getting model updates.

    Strategy: Submit zero or near-zero gradients to avoid computation.
    """

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Scale down dramatically or return zeros
        if self.config.stealth > 0.8:
            # Very stealthy: small random noise to avoid exact-zero detection
            noise_level = 1e-6 * (1.0 - self.config.stealth)
            return self.rng.normal(0, noise_level, gradient.shape).astype(gradient.dtype)
        else:
            # Less stealthy: scaled down gradient
            scale = 0.01 * (1.0 - self.config.intensity)
            return gradient * scale


# =============================================================================
# BACKDOOR ATTACK
# =============================================================================

class BackdoorAttack(ByzantineAttack):
    """
    Backdoor attack: inject trigger pattern for targeted misclassification.

    Strategy: Craft gradients that embed a backdoor trigger.
    When trigger is present in input, model outputs attacker's chosen label.
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        # Initialize trigger pattern if not provided
        if config.backdoor_trigger is None:
            # Default: small square pattern
            self.trigger = self.rng.random(100).astype(np.float32)
        else:
            self.trigger = config.backdoor_trigger

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Embed backdoor pattern in gradient
        # Scale trigger to match gradient magnitude
        grad_scale = np.std(gradient) + 1e-8
        trigger_scaled = self.trigger * grad_scale * self.config.intensity

        # Tile or truncate trigger to match gradient size
        if len(trigger_scaled) < len(gradient):
            # Repeat trigger
            repeats = len(gradient) // len(trigger_scaled) + 1
            trigger_full = np.tile(trigger_scaled, repeats)[:len(gradient)]
        else:
            trigger_full = trigger_scaled[:len(gradient)]

        # Blend based on stealth
        blend = self.config.intensity * (1.0 - self.config.stealth * 0.7)
        result = gradient * (1 - blend) + trigger_full.astype(gradient.dtype) * blend

        return result


# =============================================================================
# ADAPTIVE ATTACK
# =============================================================================

class AdaptiveAttack(ByzantineAttack):
    """
    Adaptive attack that learns to evade detection.

    Strategy: Adjust attack parameters based on detection feedback.
    Uses simple online learning to find optimal attack strength.
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.current_intensity = config.intensity
        self.current_stealth = config.stealth
        self._success_count = 0
        self._total_count = 0

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        # Adapt based on detection history
        if len(self._detection_history) > 5:
            recent_detection_rate = sum(self._detection_history[-5:]) / 5

            if recent_detection_rate > 0.5:
                # Being detected too often: increase stealth
                self.current_stealth = min(0.95, self.current_stealth + self.config.adaptation_rate)
                self.current_intensity = max(0.1, self.current_intensity - self.config.adaptation_rate * 0.5)
            elif recent_detection_rate < 0.2:
                # Not being detected: can increase intensity
                self.current_intensity = min(2.0, self.current_intensity + self.config.adaptation_rate * 0.3)

        # Apply a blend of attacks based on learned parameters
        # Combine scaling and noise for flexibility
        scale = 1.0 + (self.current_intensity - 1.0) * (1.0 - self.current_stealth)
        noise_std = np.std(gradient) * self.current_intensity * (1.0 - self.current_stealth) * 0.5

        noise = self.rng.normal(0, noise_std, gradient.shape).astype(gradient.dtype)
        result = gradient * scale + noise

        self._total_count += 1
        return result

    def record_detection(self, detected: bool) -> None:
        super().record_detection(detected)
        if not detected:
            self._success_count += 1


# =============================================================================
# CARTEL (COORDINATED) ATTACK
# =============================================================================

class CartelAttack(ByzantineAttack):
    """
    Coordinated attack by multiple colluding nodes.

    Strategy: Attackers coordinate to submit similar malicious gradients,
    making them appear legitimate through consensus.
    """

    def __init__(self, config: AttackConfig, cartel_id: int = 0):
        super().__init__(config)
        self.cartel_id = cartel_id
        self.cartel_size = config.cartel_size
        self._shared_direction: Optional[np.ndarray] = None

    def set_shared_direction(self, direction: np.ndarray) -> None:
        """Set the shared attack direction for the cartel."""
        self._shared_direction = direction / (np.linalg.norm(direction) + 1e-8)

    def attack(self, gradient: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        if self._shared_direction is None:
            # Initialize shared direction (negative of gradient)
            self._shared_direction = -gradient / (np.linalg.norm(gradient) + 1e-8)

        # All cartel members push in same direction
        grad_magnitude = np.linalg.norm(gradient) + 1e-8

        # Shared attack gradient
        attack_direction = self._shared_direction[:len(gradient)] if len(self._shared_direction) >= len(gradient) else np.tile(self._shared_direction, len(gradient) // len(self._shared_direction) + 1)[:len(gradient)]

        base_attack = attack_direction * grad_magnitude * self.config.intensity

        # Add small per-node variation to avoid exact-match detection
        variation = self.rng.normal(0, grad_magnitude * 0.05 * (1.0 - self.config.stealth), gradient.shape)

        # Blend with original gradient based on stealth
        blend = 1.0 - self.config.stealth * 0.6
        result = gradient * (1 - blend) + (base_attack + variation).astype(gradient.dtype) * blend

        return result


# =============================================================================
# ATTACK FACTORY
# =============================================================================

def create_attack(config: AttackConfig, **kwargs) -> ByzantineAttack:
    """
    Factory function to create attack instances.

    Args:
        config: Attack configuration
        **kwargs: Additional arguments for specific attacks

    Returns:
        ByzantineAttack instance
    """
    attack_classes = {
        AttackType.GRADIENT_SCALING: GradientScalingAttack,
        AttackType.SIGN_FLIP: SignFlipAttack,
        AttackType.GAUSSIAN_NOISE: GaussianNoiseAttack,
        AttackType.LITTLE_IS_ENOUGH: LittleIsEnoughAttack,
        AttackType.INNER_PRODUCT_MANIPULATION: IPMAttack,
        AttackType.FREE_RIDER: FreeRiderAttack,
        AttackType.BACKDOOR: BackdoorAttack,
        AttackType.ADAPTIVE: AdaptiveAttack,
        AttackType.CARTEL: CartelAttack,
    }

    attack_class = attack_classes.get(config.attack_type)
    if attack_class is None:
        raise ValueError(f"Unknown attack type: {config.attack_type}")

    return attack_class(config, **kwargs)


# =============================================================================
# ATTACK ORCHESTRATOR
# =============================================================================

class AttackOrchestrator:
    """
    Orchestrate multiple Byzantine attacks for testing.

    Manages attack execution, detection tracking, and reporting.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)
        self.attacks: Dict[str, ByzantineAttack] = {}
        self.attack_history: List[Dict] = []

    def register_attacker(self, node_id: str, attack: ByzantineAttack) -> None:
        """Register a Byzantine attacker."""
        self.attacks[node_id] = attack

    def apply_attacks(
        self,
        gradients: Dict[str, np.ndarray],
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Set[str]]:
        """
        Apply attacks to gradients.

        Args:
            gradients: Dict of node_id -> gradient
            context: Optional context for attacks

        Returns:
            Tuple of (modified gradients, set of attacker node_ids)
        """
        result = {}
        attackers = set()

        for node_id, gradient in gradients.items():
            if node_id in self.attacks:
                # Apply attack
                attack = self.attacks[node_id]
                attacked_gradient = attack.attack(gradient, context)
                result[node_id] = attacked_gradient
                attackers.add(node_id)

                self.attack_history.append({
                    "node_id": node_id,
                    "attack_type": attack.config.attack_type.value,
                    "intensity": attack.config.intensity,
                    "stealth": attack.config.stealth,
                })
            else:
                result[node_id] = gradient

        return result, attackers

    def record_detections(self, detected_nodes: Set[str]) -> None:
        """Record which attackers were detected."""
        for node_id, attack in self.attacks.items():
            attack.record_detection(node_id in detected_nodes)

    def get_statistics(self) -> Dict:
        """Get attack statistics."""
        if not self.attacks:
            return {"attackers": 0}

        detection_rates = {
            node_id: attack.detection_rate
            for node_id, attack in self.attacks.items()
        }

        return {
            "attackers": len(self.attacks),
            "total_attacks": len(self.attack_history),
            "detection_rates": detection_rates,
            "mean_detection_rate": np.mean(list(detection_rates.values())) if detection_rates else 0.0,
            "attack_types": list(set(a.config.attack_type.value for a in self.attacks.values())),
        }


# =============================================================================
# PRESET ATTACK SCENARIOS
# =============================================================================

def create_gradient_scaling_scenario(
    num_attackers: int = 3,
    intensity: float = 10.0,
    stealth: float = 0.0,
    seed: Optional[int] = None,
) -> List[Tuple[str, ByzantineAttack]]:
    """Create gradient scaling attack scenario."""
    attacks = []
    for i in range(num_attackers):
        config = AttackConfig(
            attack_type=AttackType.GRADIENT_SCALING,
            intensity=intensity,
            stealth=stealth,
            random_seed=seed + i if seed else None,
        )
        attacks.append((f"attacker_{i}", GradientScalingAttack(config)))
    return attacks


def create_adaptive_attack_scenario(
    num_attackers: int = 3,
    initial_intensity: float = 1.0,
    adaptation_rate: float = 0.1,
    seed: Optional[int] = None,
) -> List[Tuple[str, ByzantineAttack]]:
    """Create adaptive attack scenario."""
    attacks = []
    for i in range(num_attackers):
        config = AttackConfig(
            attack_type=AttackType.ADAPTIVE,
            intensity=initial_intensity,
            stealth=0.5,
            adaptation_rate=adaptation_rate,
            random_seed=seed + i if seed else None,
        )
        attacks.append((f"adaptive_{i}", AdaptiveAttack(config)))
    return attacks


def create_cartel_scenario(
    cartel_size: int = 5,
    intensity: float = 2.0,
    stealth: float = 0.7,
    seed: Optional[int] = None,
) -> List[Tuple[str, ByzantineAttack]]:
    """Create coordinated cartel attack scenario."""
    attacks = []

    # Create shared direction for cartel
    rng = np.random.default_rng(seed)
    shared_direction = rng.standard_normal(10000).astype(np.float32)

    for i in range(cartel_size):
        config = AttackConfig(
            attack_type=AttackType.CARTEL,
            intensity=intensity,
            stealth=stealth,
            cartel_size=cartel_size,
            random_seed=seed + i if seed else None,
        )
        attack = CartelAttack(config, cartel_id=i)
        attack.set_shared_direction(shared_direction)
        attacks.append((f"cartel_{i}", attack))

    return attacks


def create_mixed_attack_scenario(
    total_attackers: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[str, ByzantineAttack]]:
    """Create scenario with mix of attack types."""
    rng = np.random.default_rng(seed)
    attacks = []

    attack_types = [
        AttackType.GRADIENT_SCALING,
        AttackType.SIGN_FLIP,
        AttackType.GAUSSIAN_NOISE,
        AttackType.LITTLE_IS_ENOUGH,
        AttackType.ADAPTIVE,
    ]

    for i in range(total_attackers):
        attack_type = attack_types[i % len(attack_types)]
        config = AttackConfig(
            attack_type=attack_type,
            intensity=rng.uniform(0.5, 2.0),
            stealth=rng.uniform(0.3, 0.8),
            random_seed=seed + i if seed else None,
        )
        attack = create_attack(config)
        attacks.append((f"mixed_{i}_{attack_type.value}", attack))

    return attacks
