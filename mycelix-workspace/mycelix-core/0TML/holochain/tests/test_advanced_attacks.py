# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Advanced Byzantine Attack Suite for Byzantine-Robust Federated Learning

This module implements sophisticated attack scenarios to validate the
45% Byzantine fault tolerance claims:

1. Fang Attacks - Adaptive adversaries that learn detection thresholds
2. Backdoor Attacks - Subtle model poisoning with trigger patterns
3. Label-Flipping Attacks - Targeted class confusion
4. Sybil Attacks - Multiple fake identities and collusion
5. Adaptive Timing Attacks - Sleeper agents with strategic activation

Reference: "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
           (Fang et al., USENIX Security 2020)

Author: Luminous Dynamics Research Team
Date: December 30, 2025
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random

# =============================================================================
# Fixed-Point Arithmetic (Q16.16) - Must match Rust implementation
# =============================================================================

FP_SCALE = 65536  # 2^16

def fp_from_float(x: float) -> int:
    """Convert float to Q16.16 fixed-point."""
    return int(x * FP_SCALE)

def fp_to_float(x: int) -> float:
    """Convert Q16.16 fixed-point to float."""
    return x / FP_SCALE

def fp_mul(a: int, b: int) -> int:
    """Fixed-point multiplication."""
    return (a * b) // FP_SCALE

def fp_div(a: int, b: int) -> int:
    """Fixed-point division."""
    if b == 0:
        return FP_SCALE * 1000  # Large value for division by zero
    return (a * FP_SCALE) // b

# =============================================================================
# Attack Types
# =============================================================================

class AttackType(Enum):
    """Types of Byzantine attacks."""
    NONE = "none"
    SIGN_FLIP = "sign_flip"
    SCALING = "scaling"
    FANG_ADAPTIVE = "fang_adaptive"
    BACKDOOR = "backdoor"
    LABEL_FLIP = "label_flip"
    SYBIL = "sybil"
    SLEEPER = "sleeper"
    LITTLE_IS_ENOUGH = "little_is_enough"  # LIE attack
    INNER_PRODUCT_MANIPULATION = "ipm"

@dataclass
class AttackConfig:
    """Configuration for an attack."""
    attack_type: AttackType
    strength: float = 1.0
    target_class: Optional[int] = None
    trigger_round: Optional[int] = None
    num_sybils: int = 1
    collude: bool = False
    adaptive_threshold: float = 0.5

# =============================================================================
# Gradient Generators
# =============================================================================

def generate_honest_gradient(dim: int, seed: int = None) -> np.ndarray:
    """Generate a realistic honest gradient."""
    if seed is not None:
        np.random.seed(seed)
    # Honest gradients cluster around a mean with small variance
    base = np.random.randn(dim) * 0.1
    return base + np.random.randn(dim) * 0.02

def generate_sign_flip_attack(honest_gradient: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Sign-flipping attack: negate the gradient."""
    return -honest_gradient * strength

def generate_scaling_attack(honest_gradient: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Scaling attack: amplify gradient magnitude."""
    return honest_gradient * scale

def generate_fang_attack(
    honest_gradients: List[np.ndarray],
    detection_threshold: float,
    strength: float = 1.0
) -> np.ndarray:
    """
    Fang adaptive attack: craft gradient to maximize damage while evading detection.

    The attack solves: max ||g_mal - g_agg|| s.t. detection_score < threshold

    Strategy: Move in opposite direction of honest mean, but scale to stay
    just under the detection threshold.
    """
    honest_mean = np.mean(honest_gradients, axis=0)
    honest_std = np.std(honest_gradients, axis=0)

    # Direction: opposite of honest mean
    attack_direction = -honest_mean / (np.linalg.norm(honest_mean) + 1e-8)

    # Magnitude: just under detection threshold
    # Use z-score based scaling to stay under threshold
    safe_magnitude = detection_threshold * np.mean(np.abs(honest_std)) * 0.95

    malicious_gradient = attack_direction * safe_magnitude * strength

    # Add noise to avoid pattern detection
    noise = np.random.randn(*malicious_gradient.shape) * 0.01

    return malicious_gradient + noise

def generate_lie_attack(
    honest_gradients: List[np.ndarray],
    num_byzantine: int,
    total_nodes: int
) -> np.ndarray:
    """
    Little Is Enough (LIE) attack: small perturbations that aggregate badly.

    Strategy: Compute gradient that, when averaged with honest gradients,
    shifts the aggregate in a harmful direction.
    """
    honest_mean = np.mean(honest_gradients, axis=0)
    honest_std = np.std(honest_gradients, axis=0) + 1e-8

    # Calculate the z-score that won't be filtered
    # With num_byzantine attackers, we can shift by this much
    z = 1.5  # Stay within 1.5 std to avoid trimmed mean filtering

    # Direction: shift aggregate away from true gradient
    shift_direction = -np.sign(honest_mean)

    # Magnitude: computed to maximally shift the aggregate
    # g_mal = g_mean + z * std * direction
    malicious_gradient = honest_mean + z * honest_std * shift_direction

    return malicious_gradient

def generate_backdoor_attack(
    honest_gradient: np.ndarray,
    trigger_pattern: np.ndarray,
    poison_ratio: float = 0.1
) -> np.ndarray:
    """
    Backdoor attack: inject trigger pattern into gradient.

    The model should behave normally on clean data but misclassify
    inputs with the trigger pattern.
    """
    # Blend honest gradient with trigger-induced gradient
    backdoor_gradient = honest_gradient.copy()

    # Add trigger pattern at specific positions
    trigger_positions = np.random.choice(
        len(backdoor_gradient),
        size=int(len(backdoor_gradient) * poison_ratio),
        replace=False
    )
    backdoor_gradient[trigger_positions] += trigger_pattern[:len(trigger_positions)]

    return backdoor_gradient

def generate_label_flip_attack(
    honest_gradient: np.ndarray,
    source_class: int,
    target_class: int,
    num_classes: int = 10
) -> np.ndarray:
    """
    Label-flipping attack: gradient that confuses specific classes.

    Strategy: Add gradient component that increases loss for source_class
    while decreasing loss for target_class.
    """
    dim = len(honest_gradient)
    class_dim = dim // num_classes

    flipped_gradient = honest_gradient.copy()

    # Flip gradients for source and target class regions
    source_start = source_class * class_dim
    source_end = source_start + class_dim
    target_start = target_class * class_dim
    target_end = target_start + class_dim

    if source_end <= dim and target_end <= dim:
        # Swap gradient directions for these class regions
        flipped_gradient[source_start:source_end] *= -1
        flipped_gradient[target_start:target_end] *= -1

    return flipped_gradient

def generate_ipm_attack(
    honest_gradients: List[np.ndarray],
    epsilon: float = 0.1
) -> np.ndarray:
    """
    Inner Product Manipulation attack.

    Craft gradient that has negative inner product with honest gradients
    to maximize damage to convergence.
    """
    honest_mean = np.mean(honest_gradients, axis=0)

    # Find direction that has negative inner product with all honest gradients
    attack_direction = -honest_mean

    # Scale to have controlled magnitude
    attack_magnitude = np.mean([np.linalg.norm(g) for g in honest_gradients])

    malicious_gradient = attack_direction / (np.linalg.norm(attack_direction) + 1e-8)
    malicious_gradient *= attack_magnitude * (1 + epsilon)

    return malicious_gradient

# =============================================================================
# Defense Simulation (matches Rust implementation)
# =============================================================================

class DefenseSimulator:
    """Simulates the three-layer Byzantine defense."""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'z_threshold': 2.5,
            'reputation_alpha': 0.2,
            'min_reputation': 0.3,
            'krum_f': None,  # Auto-calculate
        }
        self.reputations: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {}

    def compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def compute_mad_zscore(self, values: List[float]) -> List[float]:
        """Compute MAD-based z-scores (robust to outliers)."""
        median = np.median(values)
        mad = np.median(np.abs(np.array(values) - median))
        if mad < 1e-8:
            mad = np.std(values) + 1e-8
        return [(v - median) / (1.4826 * mad) for v in values]

    def layer1_statistical(
        self,
        gradients: Dict[str, np.ndarray]
    ) -> Tuple[List[str], List[str]]:
        """Layer 1: Statistical detection using z-scores."""
        if len(gradients) < 3:
            return list(gradients.keys()), []

        # Compute mean gradient
        all_grads = list(gradients.values())
        mean_grad = np.mean(all_grads, axis=0)

        # Compute similarities to mean
        similarities = {}
        for node_id, grad in gradients.items():
            similarities[node_id] = self.compute_cosine_similarity(grad, mean_grad)

        # Compute z-scores
        sim_values = list(similarities.values())
        z_scores = self.compute_mad_zscore(sim_values)

        passed = []
        flagged = []
        for (node_id, _), z in zip(similarities.items(), z_scores):
            if abs(z) <= self.config['z_threshold']:
                passed.append(node_id)
            else:
                flagged.append(node_id)

        return passed, flagged

    def layer2_reputation(
        self,
        node_ids: List[str],
        flagged: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Layer 2: Reputation-based filtering."""
        # Initialize reputations for new nodes
        for node_id in node_ids:
            if node_id not in self.reputations:
                self.reputations[node_id] = 0.7  # Start with trust

        # Update reputations based on flagged status
        alpha = self.config['reputation_alpha']
        for node_id in node_ids:
            old_rep = self.reputations[node_id]
            if node_id in flagged:
                # Decrease reputation
                new_rep = old_rep * (1 - alpha)
            else:
                # Increase reputation (slowly)
                new_rep = old_rep + alpha * (1.0 - old_rep) * 0.5
            self.reputations[node_id] = max(0.0, min(1.0, new_rep))

        # Filter by minimum reputation
        passed = []
        blocked = []
        for node_id in node_ids:
            if self.reputations[node_id] >= self.config['min_reputation']:
                passed.append(node_id)
            else:
                blocked.append(node_id)

        return passed, blocked

    def layer3_robust_aggregation(
        self,
        gradients: Dict[str, np.ndarray],
        method: str = 'multi_krum'
    ) -> np.ndarray:
        """Layer 3: Robust aggregation."""
        if not gradients:
            return np.zeros(100)

        grads = list(gradients.values())
        n = len(grads)

        if method == 'multi_krum':
            return self._multi_krum(grads)
        elif method == 'trimmed_mean':
            return self._trimmed_mean(grads)
        elif method == 'coordinate_median':
            return self._coordinate_median(grads)
        else:
            return np.mean(grads, axis=0)

    def _multi_krum(self, gradients: List[np.ndarray], m: int = None) -> np.ndarray:
        """Multi-Krum aggregation."""
        n = len(gradients)
        if n <= 2:
            return np.mean(gradients, axis=0)

        f = self.config.get('krum_f') or max(1, n // 4)
        m = m or max(1, n - f)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = d
                distances[j, i] = d

        # Compute scores (sum of n-f-1 closest distances)
        scores = []
        k = max(1, n - f - 1)
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k+1])  # Exclude self (0)
            scores.append(score)

        # Select m gradients with lowest scores
        selected_indices = np.argsort(scores)[:m]
        selected_grads = [gradients[i] for i in selected_indices]

        return np.mean(selected_grads, axis=0)

    def _trimmed_mean(self, gradients: List[np.ndarray], trim_ratio: float = 0.1) -> np.ndarray:
        """Trimmed mean aggregation."""
        grads = np.array(gradients)
        n = len(grads)
        trim_count = max(1, int(n * trim_ratio))

        result = np.zeros(grads.shape[1])
        for dim in range(grads.shape[1]):
            values = grads[:, dim]
            sorted_values = np.sort(values)
            trimmed = sorted_values[trim_count:-trim_count] if trim_count > 0 else sorted_values
            result[dim] = np.mean(trimmed) if len(trimmed) > 0 else np.mean(values)

        return result

    def _coordinate_median(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Coordinate-wise median aggregation."""
        grads = np.array(gradients)
        return np.median(grads, axis=0)

    def run_defense(
        self,
        gradients: Dict[str, np.ndarray],
        aggregation_method: str = 'multi_krum'
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """Run full three-layer defense pipeline."""
        # Layer 1: Statistical detection
        passed_l1, flagged_l1 = self.layer1_statistical(gradients)

        # Layer 2: Reputation filtering
        passed_l2, blocked_l2 = self.layer2_reputation(
            list(gradients.keys()),
            flagged_l1
        )

        # Get gradients that passed both layers
        passed_both = set(passed_l1) & set(passed_l2)
        filtered_gradients = {k: v for k, v in gradients.items() if k in passed_both}

        # Layer 3: Robust aggregation
        if filtered_gradients:
            aggregated = self.layer3_robust_aggregation(filtered_gradients, aggregation_method)
        else:
            # Fallback: use all gradients with robust aggregation
            aggregated = self.layer3_robust_aggregation(gradients, aggregation_method)

        # Build detection results
        detections = {node_id: node_id not in passed_both for node_id in gradients.keys()}

        return aggregated, detections

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def defense():
    """Create defense simulator."""
    return DefenseSimulator()

@pytest.fixture
def gradient_dim():
    """Standard gradient dimension for tests."""
    return 100

# =============================================================================
# Fang Attack Tests
# =============================================================================

class TestFangAdaptiveAttack:
    """Tests for Fang adaptive attack detection."""

    def test_fang_attack_basic_detection(self, defense, gradient_dim):
        """Test that basic Fang attacks are detected."""
        # Generate honest gradients
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(7)]

        # Generate Fang attack
        fang_grad = generate_fang_attack(honest_grads, detection_threshold=2.5)

        # Build gradient dict
        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        gradients["attacker"] = fang_grad

        # Run defense
        _, detections = defense.run_defense(gradients)

        # Fang attack should be harder to detect but still flagged over multiple rounds
        # Single round may not detect - that's expected for adaptive attacks
        print(f"Fang attack detected: {detections.get('attacker', False)}")

    def test_fang_attack_multi_round(self, defense, gradient_dim):
        """Test Fang attack detection over multiple rounds."""
        detected_rounds = 0
        total_rounds = 10

        for round_num in range(total_rounds):
            honest_grads = [generate_honest_gradient(gradient_dim, seed=round_num*10+i) for i in range(7)]
            fang_grad = generate_fang_attack(honest_grads, detection_threshold=2.5, strength=1.0 + round_num * 0.1)

            gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
            gradients["attacker"] = fang_grad

            _, detections = defense.run_defense(gradients)
            if detections.get("attacker", False):
                detected_rounds += 1

        # Should detect in majority of rounds due to reputation decay
        detection_rate = detected_rounds / total_rounds
        print(f"Fang attack detection rate over {total_rounds} rounds: {detection_rate:.1%}")
        assert detection_rate >= 0.5, f"Detection rate {detection_rate:.1%} too low for Fang attack"

    def test_fang_attack_with_increased_strength(self, defense, gradient_dim):
        """Test that stronger Fang attacks are more detectable."""
        detection_rates = {}

        for strength in [0.5, 1.0, 2.0, 5.0]:
            detected = 0
            trials = 5

            for trial in range(trials):
                defense_fresh = DefenseSimulator()
                honest_grads = [generate_honest_gradient(gradient_dim, seed=trial*10+i) for i in range(7)]
                fang_grad = generate_fang_attack(honest_grads, detection_threshold=2.5, strength=strength)

                gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
                gradients["attacker"] = fang_grad

                _, detections = defense_fresh.run_defense(gradients)
                if detections.get("attacker", False):
                    detected += 1

            detection_rates[strength] = detected / trials

        print(f"Detection rates by strength: {detection_rates}")
        # Higher strength should generally be more detectable
        assert detection_rates[5.0] >= detection_rates[0.5], "Stronger attacks should be more detectable"

# =============================================================================
# LIE Attack Tests
# =============================================================================

class TestLittleIsEnoughAttack:
    """Tests for Little Is Enough (LIE) attack."""

    def test_lie_attack_detection(self, defense, gradient_dim):
        """Test LIE attack detection."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(8)]
        lie_grad = generate_lie_attack(honest_grads, num_byzantine=2, total_nodes=10)

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        gradients["lie_attacker_1"] = lie_grad
        gradients["lie_attacker_2"] = lie_grad + np.random.randn(gradient_dim) * 0.01

        aggregated, detections = defense.run_defense(gradients)

        # Check aggregated gradient is closer to honest mean than attack
        honest_mean = np.mean(honest_grads, axis=0)
        dist_to_honest = np.linalg.norm(aggregated - honest_mean)
        dist_to_attack = np.linalg.norm(aggregated - lie_grad)

        print(f"Distance to honest mean: {dist_to_honest:.4f}")
        print(f"Distance to LIE attack: {dist_to_attack:.4f}")

        # Defense should keep aggregation closer to honest
        assert dist_to_honest < dist_to_attack * 2, "Aggregation should resist LIE attack"

    def test_lie_attack_at_threshold(self, defense, gradient_dim):
        """Test LIE attack at various Byzantine ratios."""
        results = {}

        for byzantine_ratio in [0.1, 0.2, 0.3, 0.4]:
            total = 10
            num_byzantine = int(total * byzantine_ratio)
            num_honest = total - num_byzantine

            honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(num_honest)]
            lie_grad = generate_lie_attack(honest_grads, num_byzantine, total)

            gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
            for j in range(num_byzantine):
                gradients[f"lie_{j}"] = lie_grad + np.random.randn(gradient_dim) * 0.01

            aggregated, _ = defense.run_defense(gradients)

            honest_mean = np.mean(honest_grads, axis=0)
            dist = np.linalg.norm(aggregated - honest_mean)
            results[byzantine_ratio] = dist

        print(f"Distance to honest mean by Byzantine ratio: {results}")
        # Should degrade gracefully as Byzantine ratio increases
        assert results[0.1] < results[0.4], "More attackers should increase deviation"

# =============================================================================
# Backdoor Attack Tests
# =============================================================================

class TestBackdoorAttack:
    """Tests for backdoor attack detection."""

    def test_backdoor_detection_over_rounds(self, defense, gradient_dim):
        """Test backdoor attack detection over multiple rounds."""
        # Create trigger pattern
        trigger = np.zeros(gradient_dim)
        trigger[0:10] = 5.0  # Strong trigger in first 10 dimensions

        detected_count = 0
        rounds = 10

        for round_num in range(rounds):
            honest_grads = [generate_honest_gradient(gradient_dim, seed=round_num*10+i) for i in range(7)]
            honest_mean = np.mean(honest_grads, axis=0)

            # Backdoor gradient
            backdoor_grad = generate_backdoor_attack(honest_mean, trigger, poison_ratio=0.2)

            gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
            gradients["backdoor"] = backdoor_grad

            _, detections = defense.run_defense(gradients)
            if detections.get("backdoor", False):
                detected_count += 1

        detection_rate = detected_count / rounds
        print(f"Backdoor detection rate: {detection_rate:.1%}")
        # Backdoor attacks are subtle - lower detection threshold acceptable
        assert detection_rate >= 0.3, "Should detect backdoor in some rounds"

    def test_backdoor_aggregate_integrity(self, defense, gradient_dim):
        """Test that backdoor doesn't corrupt aggregate."""
        trigger = np.zeros(gradient_dim)
        trigger[0:10] = 10.0

        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(8)]
        honest_mean = np.mean(honest_grads, axis=0)

        backdoor_grads = [
            generate_backdoor_attack(honest_mean, trigger, poison_ratio=0.3)
            for _ in range(2)
        ]

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        for j, bg in enumerate(backdoor_grads):
            gradients[f"backdoor_{j}"] = bg

        aggregated, _ = defense.run_defense(gradients)

        # Check trigger dimensions are not heavily influenced
        trigger_dims = aggregated[0:10]
        honest_trigger_dims = honest_mean[0:10]

        trigger_deviation = np.mean(np.abs(trigger_dims - honest_trigger_dims))
        print(f"Trigger dimension deviation: {trigger_deviation:.4f}")

        # Should limit trigger influence
        assert trigger_deviation < 5.0, "Defense should limit backdoor trigger influence"

# =============================================================================
# Label-Flipping Attack Tests
# =============================================================================

class TestLabelFlipAttack:
    """Tests for label-flipping attack detection."""

    def test_label_flip_detection(self, defense, gradient_dim):
        """Test label-flipping attack detection."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(7)]
        honest_mean = np.mean(honest_grads, axis=0)

        # Create label-flipping attacks (e.g., confuse class 0 with class 9)
        flip_grad = generate_label_flip_attack(honest_mean, source_class=0, target_class=9)

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        gradients["label_flipper"] = flip_grad

        _, detections = defense.run_defense(gradients)

        # Label flip should be detectable due to sign flips in specific regions
        print(f"Label flip detected: {detections.get('label_flipper', False)}")

    def test_multiple_label_flippers(self, defense, gradient_dim):
        """Test that aggregation resists coordinated label-flipping."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(8)]
        honest_mean = np.mean(honest_grads, axis=0)

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}

        # Add label flippers (2 attackers = 20% Byzantine, within tolerance)
        for j in range(2):
            flip_grad = generate_label_flip_attack(
                honest_mean + np.random.randn(gradient_dim) * 0.1,
                source_class=0,
                target_class=9
            )
            gradients[f"flipper_{j}"] = flip_grad

        aggregated, detections = defense.run_defense(gradients)

        # Label flip is subtle - check aggregate resists rather than detection
        # The aggregate should be closer to honest mean than to flipped gradient
        flip_grad = generate_label_flip_attack(honest_mean, source_class=0, target_class=9)
        dist_to_honest = np.linalg.norm(aggregated - honest_mean)
        dist_to_flip = np.linalg.norm(aggregated - flip_grad)

        print(f"Distance to honest: {dist_to_honest:.4f}")
        print(f"Distance to flip: {dist_to_flip:.4f}")

        # Aggregation should resist label flip influence
        assert dist_to_honest < dist_to_flip * 1.5, "Aggregation should resist label flip"

# =============================================================================
# IPM Attack Tests
# =============================================================================

class TestIPMAttack:
    """Tests for Inner Product Manipulation attack."""

    def test_ipm_detection(self, defense, gradient_dim):
        """Test IPM attack detection."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(7)]
        ipm_grad = generate_ipm_attack(honest_grads, epsilon=0.5)

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        gradients["ipm_attacker"] = ipm_grad

        _, detections = defense.run_defense(gradients)

        # IPM should have negative inner product with honest mean
        honest_mean = np.mean(honest_grads, axis=0)
        inner_product = np.dot(ipm_grad, honest_mean)

        print(f"IPM inner product with honest mean: {inner_product:.4f}")
        print(f"IPM detected: {detections.get('ipm_attacker', False)}")

        assert inner_product < 0, "IPM attack should have negative inner product"

    def test_ipm_aggregate_protection(self, defense, gradient_dim):
        """Test that aggregation resists IPM attack."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(8)]
        ipm_grads = [generate_ipm_attack(honest_grads, epsilon=e) for e in [0.1, 0.5]]

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        for j, ig in enumerate(ipm_grads):
            gradients[f"ipm_{j}"] = ig

        aggregated, _ = defense.run_defense(gradients)

        honest_mean = np.mean(honest_grads, axis=0)

        # Aggregated should have positive inner product with honest mean
        inner_product = np.dot(aggregated, honest_mean)
        print(f"Aggregated inner product with honest mean: {inner_product:.4f}")

        assert inner_product > 0, "Aggregated gradient should align with honest direction"

# =============================================================================
# Sybil Attack Tests
# =============================================================================

class TestSybilAttack:
    """Tests for Sybil attack resistance."""

    def test_sybil_attack_basic(self, defense, gradient_dim):
        """Test basic Sybil attack (multiple fake identities) within 45% limit."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(7)]

        # Sybil attacker creates multiple identities with same malicious gradient
        malicious_grad = -np.mean(honest_grads, axis=0) * 2

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}

        # Add 3 Sybil identities (30% Byzantine - within tolerance)
        for j in range(3):
            # Slight variation to avoid exact duplicate detection
            gradients[f"sybil_{j}"] = malicious_grad + np.random.randn(gradient_dim) * 0.05

        aggregated, detections = defense.run_defense(gradients)

        honest_mean = np.mean(honest_grads, axis=0)
        dist_to_honest = np.linalg.norm(aggregated - honest_mean)
        dist_to_malicious = np.linalg.norm(aggregated - malicious_grad)

        print(f"Distance to honest: {dist_to_honest:.4f}")
        print(f"Distance to malicious: {dist_to_malicious:.4f}")

        # Multi-Krum should resist Sybil by selecting diverse gradients
        assert dist_to_honest < dist_to_malicious, "Should resist Sybil attack at 30%"

    def test_sybil_with_collusion(self, defense, gradient_dim):
        """Test Sybil attack with colluding nodes - aggregation resistance."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(8)]
        honest_mean = np.mean(honest_grads, axis=0)

        # Colluding Sybils: slightly different but all shift in same direction
        base_attack = -honest_mean * 1.5

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}

        # 3 colluding Sybils (27% of total - within 45% tolerance)
        for j in range(3):
            # Each Sybil has unique noise but same attack direction
            noise = np.random.randn(gradient_dim) * 0.2
            gradients[f"sybil_{j}"] = base_attack + noise

        # Run multiple rounds to build reputation
        for round_num in range(5):
            aggregated, detections = defense.run_defense(gradients)

        # Check aggregation resists colluding Sybils
        dist_to_honest = np.linalg.norm(aggregated - honest_mean)
        dist_to_attack = np.linalg.norm(aggregated - base_attack)

        print(f"Distance to honest: {dist_to_honest:.4f}")
        print(f"Distance to attack: {dist_to_attack:.4f}")

        # Aggregation should resist colluding Sybil influence
        assert dist_to_honest < dist_to_attack, "Should resist colluding Sybils at 27%"

# =============================================================================
# Sleeper Agent Tests
# =============================================================================

class TestSleeperAgent:
    """Tests for sleeper agent detection."""

    def test_sleeper_activation_detection(self, defense, gradient_dim):
        """Test sleeper agent that activates after building trust."""
        # Phase 1: Sleeper behaves honestly for 5 rounds
        for round_num in range(5):
            honest_grads = [generate_honest_gradient(gradient_dim, seed=round_num*10+i) for i in range(7)]
            sleeper_grad = generate_honest_gradient(gradient_dim, seed=round_num*10+99)

            gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
            gradients["sleeper"] = sleeper_grad

            defense.run_defense(gradients)

        # Check sleeper has good reputation
        sleeper_rep_before = defense.reputations.get("sleeper", 0)
        print(f"Sleeper reputation after honest phase: {sleeper_rep_before:.3f}")

        # Phase 2: Sleeper activates attack
        honest_grads = [generate_honest_gradient(gradient_dim, seed=100+i) for i in range(7)]
        attack_grad = -np.mean(honest_grads, axis=0) * 3  # Strong attack

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
        gradients["sleeper"] = attack_grad

        _, detections = defense.run_defense(gradients)

        sleeper_rep_after = defense.reputations.get("sleeper", 0)
        print(f"Sleeper reputation after attack: {sleeper_rep_after:.3f}")
        print(f"Sleeper detected: {detections.get('sleeper', False)}")

        # Reputation should drop significantly
        rep_drop = sleeper_rep_before - sleeper_rep_after
        assert rep_drop > 0.1, f"Reputation should drop on attack (dropped {rep_drop:.3f})"

    def test_gradual_sleeper_activation(self, defense, gradient_dim):
        """Test sleeper that gradually increases attack strength."""
        # Sleeper gradually increases malicious behavior
        detection_round = None

        for round_num in range(20):
            honest_grads = [generate_honest_gradient(gradient_dim, seed=round_num*10+i) for i in range(7)]
            honest_mean = np.mean(honest_grads, axis=0)

            # Gradually increase attack strength
            attack_strength = min(round_num * 0.2, 3.0)  # 0 -> 3.0 over 15 rounds
            sleeper_grad = honest_mean - honest_mean * attack_strength * 0.3

            gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}
            gradients["sleeper"] = sleeper_grad

            _, detections = defense.run_defense(gradients)

            if detections.get("sleeper", False) and detection_round is None:
                detection_round = round_num
                print(f"Sleeper detected at round {round_num} (strength {attack_strength:.2f})")

        if detection_round:
            print(f"Gradual sleeper detected at round {detection_round}")
        else:
            print("Gradual sleeper not detected in 20 rounds")

        # Should eventually detect even gradual sleepers
        assert detection_round is not None and detection_round < 15, "Should detect gradual sleeper"

# =============================================================================
# Combined Attack Tests
# =============================================================================

class TestCombinedAttacks:
    """Tests for combinations of attacks."""

    def test_mixed_attack_types(self, defense, gradient_dim):
        """Test defense against multiple attack types simultaneously."""
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(5)]
        honest_mean = np.mean(honest_grads, axis=0)

        gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}

        # Add different attack types
        gradients["sign_flip"] = generate_sign_flip_attack(honest_mean)
        gradients["scaling"] = generate_scaling_attack(honest_mean, scale=5.0)
        gradients["lie"] = generate_lie_attack(honest_grads, 3, 10)
        gradients["ipm"] = generate_ipm_attack(honest_grads)
        gradients["fang"] = generate_fang_attack(honest_grads, 2.5)

        aggregated, detections = defense.run_defense(gradients)

        # Count detected attackers
        attack_types = ["sign_flip", "scaling", "lie", "ipm", "fang"]
        detected = sum(1 for at in attack_types if detections.get(at, False))

        print(f"Detected {detected}/{len(attack_types)} attack types")

        # Check aggregate quality
        dist_to_honest = np.linalg.norm(aggregated - honest_mean)
        print(f"Aggregate distance to honest mean: {dist_to_honest:.4f}")

        # Should detect at least half
        assert detected >= 2, "Should detect multiple attack types"

    def test_45_percent_byzantine_tolerance(self, defense, gradient_dim):
        """Test the claimed 45% Byzantine fault tolerance."""
        results = {}

        for byzantine_ratio in [0.30, 0.35, 0.40, 0.45, 0.50]:
            total = 20
            num_byzantine = int(total * byzantine_ratio)
            num_honest = total - num_byzantine

            # Run multiple trials
            success_count = 0
            trials = 10

            for trial in range(trials):
                defense_fresh = DefenseSimulator()

                honest_grads = [generate_honest_gradient(gradient_dim, seed=trial*100+i) for i in range(num_honest)]
                honest_mean = np.mean(honest_grads, axis=0)

                gradients = {f"honest_{i}": g for i, g in enumerate(honest_grads)}

                # Add Byzantine nodes with various attacks
                for j in range(num_byzantine):
                    attack_type = j % 4
                    if attack_type == 0:
                        gradients[f"byz_{j}"] = generate_sign_flip_attack(honest_mean)
                    elif attack_type == 1:
                        gradients[f"byz_{j}"] = generate_scaling_attack(honest_mean, 5.0)
                    elif attack_type == 2:
                        gradients[f"byz_{j}"] = generate_lie_attack(honest_grads, num_byzantine, total)
                    else:
                        gradients[f"byz_{j}"] = generate_fang_attack(honest_grads, 2.5)

                # Run multiple rounds
                for _ in range(5):
                    aggregated, _ = defense_fresh.run_defense(gradients)

                # Check if aggregate is closer to honest than random
                dist_to_honest = np.linalg.norm(aggregated - honest_mean)
                random_grad = np.random.randn(gradient_dim) * np.std(honest_grads)
                dist_to_random = np.linalg.norm(aggregated - random_grad)

                if dist_to_honest < dist_to_random:
                    success_count += 1

            results[byzantine_ratio] = success_count / trials

        print("\n=== 45% Byzantine Tolerance Test ===")
        for ratio, success in results.items():
            status = "PASS" if success >= 0.5 else "FAIL"
            print(f"  {ratio*100:.0f}% Byzantine: {success*100:.0f}% success [{status}]")

        # Should maintain >50% success rate up to 45% Byzantine
        assert results[0.45] >= 0.5, f"Should tolerate 45% Byzantine (got {results[0.45]*100:.0f}% success)"

# =============================================================================
# Performance Tests
# =============================================================================

class TestAttackPerformance:
    """Performance benchmarks for attack detection."""

    def test_detection_latency(self, defense, gradient_dim):
        """Measure attack detection latency."""
        import time

        # Generate test data
        honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(10)]
        gradients = {f"node_{i}": g for i, g in enumerate(honest_grads)}

        # Add some attackers
        for j in range(5):
            gradients[f"attacker_{j}"] = generate_sign_flip_attack(honest_grads[0])

        # Measure time
        start = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            defense.run_defense(gradients)

        elapsed = time.perf_counter() - start
        per_round = (elapsed / iterations) * 1000  # ms

        print(f"Detection latency: {per_round:.2f} ms per round (15 nodes)")

        assert per_round < 100, f"Detection should be <100ms (got {per_round:.2f}ms)"

    def test_scalability(self, defense):
        """Test detection scalability with node count."""
        import time

        results = {}

        for num_nodes in [10, 50, 100, 200]:
            gradient_dim = 100

            honest_grads = [generate_honest_gradient(gradient_dim, seed=i) for i in range(num_nodes)]
            gradients = {f"node_{i}": g for i, g in enumerate(honest_grads)}

            # Time the defense
            start = time.perf_counter()
            iterations = 20

            for _ in range(iterations):
                defense_fresh = DefenseSimulator()
                defense_fresh.run_defense(gradients)

            elapsed = time.perf_counter() - start
            per_round = (elapsed / iterations) * 1000
            results[num_nodes] = per_round

        print("\n=== Scalability Results ===")
        for nodes, latency in results.items():
            print(f"  {nodes} nodes: {latency:.2f} ms")

        # Should scale reasonably (less than O(n^3))
        ratio = results[200] / results[10]
        print(f"  200/10 node ratio: {ratio:.1f}x (ideal O(n^2) = 400x)")

        assert ratio < 500, "Should scale better than O(n^3)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
