# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Advanced Byzantine Attacks

Validates attack implementations and their detectability.

Author: Luminous Dynamics
Date: December 31, 2025
"""

import pytest
import numpy as np
from typing import Dict


class TestGradientScalingAttack:
    """Test gradient scaling attack."""

    def test_scales_gradient(self):
        """Test attack scales gradients."""
        from mycelix_fl.attacks import GradientScalingAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.GRADIENT_SCALING,
            intensity=1.0,
            stealth=0.0,
            random_seed=42,
        )
        attack = GradientScalingAttack(config)

        gradient = np.ones(1000, dtype=np.float32)
        attacked = attack.attack(gradient)

        # Should be scaled up significantly
        assert np.mean(attacked) > np.mean(gradient) * 5

    def test_stealth_reduces_scale(self):
        """Test high stealth reduces scaling."""
        from mycelix_fl.attacks import GradientScalingAttack, AttackConfig, AttackType

        gradient = np.ones(1000, dtype=np.float32)

        # Low stealth
        config_low = AttackConfig(attack_type=AttackType.GRADIENT_SCALING, stealth=0.0)
        attack_low = GradientScalingAttack(config_low)
        attacked_low = attack_low.attack(gradient)

        # High stealth
        config_high = AttackConfig(attack_type=AttackType.GRADIENT_SCALING, stealth=0.9)
        attack_high = GradientScalingAttack(config_high)
        attacked_high = attack_high.attack(gradient)

        # High stealth should have lower scale
        assert np.mean(np.abs(attacked_high)) < np.mean(np.abs(attacked_low))


class TestSignFlipAttack:
    """Test sign flip attack."""

    def test_flips_signs(self):
        """Test attack flips gradient signs."""
        from mycelix_fl.attacks import SignFlipAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.SIGN_FLIP,
            intensity=1.0,
            stealth=0.0,
            random_seed=42,
        )
        attack = SignFlipAttack(config)

        gradient = np.ones(1000, dtype=np.float32)
        attacked = attack.attack(gradient)

        # Full flip at zero stealth
        assert np.mean(attacked) < 0

    def test_partial_flip_with_stealth(self):
        """Test partial flip with stealth."""
        from mycelix_fl.attacks import SignFlipAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.SIGN_FLIP,
            stealth=0.8,  # High stealth = partial flip
            random_seed=42,
        )
        attack = SignFlipAttack(config)

        gradient = np.ones(1000, dtype=np.float32)
        attacked = attack.attack(gradient)

        # Some positive, some negative
        assert np.sum(attacked > 0) > 0
        assert np.sum(attacked < 0) > 0


class TestGaussianNoiseAttack:
    """Test Gaussian noise attack."""

    def test_adds_noise(self):
        """Test attack adds noise."""
        from mycelix_fl.attacks import GaussianNoiseAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.GAUSSIAN_NOISE,
            intensity=1.0,
            stealth=0.0,
            random_seed=42,
        )
        attack = GaussianNoiseAttack(config)

        gradient = np.zeros(1000, dtype=np.float32)
        attacked = attack.attack(gradient)

        # Should have non-zero variance
        assert np.std(attacked) > 0

    def test_noise_scales_with_intensity(self):
        """Test noise magnitude scales with intensity."""
        from mycelix_fl.attacks import GaussianNoiseAttack, AttackConfig, AttackType

        gradient = np.ones(10000, dtype=np.float32)

        config_low = AttackConfig(attack_type=AttackType.GAUSSIAN_NOISE, intensity=0.1, random_seed=42)
        config_high = AttackConfig(attack_type=AttackType.GAUSSIAN_NOISE, intensity=2.0, random_seed=42)

        attack_low = GaussianNoiseAttack(config_low)
        attack_high = GaussianNoiseAttack(config_high)

        attacked_low = attack_low.attack(gradient)
        attacked_high = attack_high.attack(gradient)

        # Higher intensity = more noise
        assert np.std(attacked_high - gradient) > np.std(attacked_low - gradient)


class TestLittleIsEnoughAttack:
    """Test Little Is Enough attack."""

    def test_subtle_perturbation(self):
        """Test attack creates subtle perturbation."""
        from mycelix_fl.attacks import LittleIsEnoughAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.LITTLE_IS_ENOUGH,
            intensity=1.0,
            stealth=0.5,
            random_seed=42,
        )
        attack = LittleIsEnoughAttack(config)

        gradient = np.random.randn(1000).astype(np.float32)
        attacked = attack.attack(gradient)

        # Should be different but not extremely
        diff = np.linalg.norm(attacked - gradient) / np.linalg.norm(gradient)
        assert 0.1 < diff < 10.0  # Modified but not extreme


class TestIPMAttack:
    """Test Inner Product Manipulation attack."""

    def test_negative_inner_product(self):
        """Test attack creates negative inner product with true gradient."""
        from mycelix_fl.attacks import IPMAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.INNER_PRODUCT_MANIPULATION,
            intensity=2.0,
            stealth=0.0,
            random_seed=42,
        )
        attack = IPMAttack(config)

        gradient = np.random.randn(1000).astype(np.float32)
        attacked = attack.attack(gradient, context={"true_gradient": gradient})

        # Inner product should be negative or small
        inner_product = np.dot(attacked, gradient)
        original_inner_product = np.dot(gradient, gradient)
        assert inner_product < original_inner_product * 0.5


class TestFreeRiderAttack:
    """Test free-rider attack."""

    def test_near_zero_gradient(self):
        """Test attack produces near-zero gradient."""
        from mycelix_fl.attacks import FreeRiderAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.FREE_RIDER,
            intensity=1.0,
            stealth=0.9,
            random_seed=42,
        )
        attack = FreeRiderAttack(config)

        gradient = np.random.randn(1000).astype(np.float32)
        attacked = attack.attack(gradient)

        # Should be very small
        assert np.max(np.abs(attacked)) < 1e-4


class TestBackdoorAttack:
    """Test backdoor attack."""

    def test_embeds_pattern(self):
        """Test attack embeds trigger pattern."""
        from mycelix_fl.attacks import BackdoorAttack, AttackConfig, AttackType

        trigger = np.array([1.0, -1.0, 1.0, -1.0] * 25, dtype=np.float32)
        config = AttackConfig(
            attack_type=AttackType.BACKDOOR,
            intensity=1.0,
            stealth=0.0,
            backdoor_trigger=trigger,
            random_seed=42,
        )
        attack = BackdoorAttack(config)

        # Use non-zero gradient so trigger scales properly
        gradient = np.random.randn(1000).astype(np.float32)
        attacked = attack.attack(gradient)

        # Should be different from original
        assert not np.allclose(attacked, gradient)


class TestAdaptiveAttack:
    """Test adaptive attack."""

    def test_adapts_to_detection(self):
        """Test attack adapts based on detection feedback."""
        from mycelix_fl.attacks import AdaptiveAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.ADAPTIVE,
            intensity=1.0,
            stealth=0.5,
            adaptation_rate=0.2,
            random_seed=42,
        )
        attack = AdaptiveAttack(config)

        gradient = np.random.randn(1000).astype(np.float32)

        # Initial attack
        initial_intensity = attack.current_intensity
        initial_stealth = attack.current_stealth

        # Simulate being detected multiple times
        for _ in range(10):
            attack.attack(gradient)
            attack.record_detection(True)  # Detected

        # Should have increased stealth and decreased intensity
        assert attack.current_stealth > initial_stealth
        assert attack.current_intensity < initial_intensity

    def test_increases_intensity_when_undetected(self):
        """Test attack increases intensity when not detected."""
        from mycelix_fl.attacks import AdaptiveAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.ADAPTIVE,
            intensity=0.5,
            stealth=0.5,
            adaptation_rate=0.2,
            random_seed=42,
        )
        attack = AdaptiveAttack(config)

        gradient = np.random.randn(1000).astype(np.float32)
        initial_intensity = attack.current_intensity

        # Simulate not being detected
        for _ in range(10):
            attack.attack(gradient)
            attack.record_detection(False)  # Not detected

        # Should have increased intensity
        assert attack.current_intensity > initial_intensity


class TestCartelAttack:
    """Test coordinated cartel attack."""

    def test_similar_attack_vectors(self):
        """Test cartel members produce similar attack vectors."""
        from mycelix_fl.attacks import CartelAttack, AttackConfig, AttackType

        config = AttackConfig(
            attack_type=AttackType.CARTEL,
            intensity=1.0,
            stealth=0.5,
            cartel_size=3,
            random_seed=42,
        )

        # Create cartel
        shared_direction = np.random.randn(1000).astype(np.float32)
        attacks = []
        for i in range(3):
            attack = CartelAttack(config, cartel_id=i)
            attack.set_shared_direction(shared_direction)
            attacks.append(attack)

        gradient = np.random.randn(1000).astype(np.float32)

        # Get attacks from all cartel members
        attacked_gradients = [a.attack(gradient) for a in attacks]

        # Should be similar (high correlation)
        for i in range(len(attacked_gradients) - 1):
            corr = np.corrcoef(attacked_gradients[i], attacked_gradients[i + 1])[0, 1]
            assert corr > 0.8  # High correlation


class TestAttackFactory:
    """Test attack factory function."""

    def test_creates_all_attack_types(self):
        """Test factory creates all attack types."""
        from mycelix_fl.attacks import create_attack, AttackConfig, AttackType

        for attack_type in AttackType:
            if attack_type == AttackType.LABEL_FLIP:
                continue  # Not implemented yet

            config = AttackConfig(attack_type=attack_type, random_seed=42)
            attack = create_attack(config)

            gradient = np.random.randn(1000).astype(np.float32)
            attacked = attack.attack(gradient)

            assert attacked is not None
            assert attacked.shape == gradient.shape


class TestAttackOrchestrator:
    """Test attack orchestrator."""

    def test_applies_attacks(self):
        """Test orchestrator applies attacks to gradients."""
        from mycelix_fl.attacks import (
            AttackOrchestrator,
            GradientScalingAttack,
            AttackConfig,
            AttackType,
        )

        orchestrator = AttackOrchestrator(random_seed=42)

        # Register attackers
        config = AttackConfig(attack_type=AttackType.GRADIENT_SCALING, intensity=10.0)
        orchestrator.register_attacker("malicious_1", GradientScalingAttack(config))
        orchestrator.register_attacker("malicious_2", GradientScalingAttack(config))

        # Create gradients
        gradients = {
            "honest_1": np.ones(1000, dtype=np.float32),
            "honest_2": np.ones(1000, dtype=np.float32),
            "malicious_1": np.ones(1000, dtype=np.float32),
            "malicious_2": np.ones(1000, dtype=np.float32),
        }

        # Apply attacks
        modified, attackers = orchestrator.apply_attacks(gradients)

        assert attackers == {"malicious_1", "malicious_2"}
        assert np.mean(modified["malicious_1"]) > np.mean(modified["honest_1"]) * 5
        assert np.allclose(modified["honest_1"], gradients["honest_1"])

    def test_tracks_detections(self):
        """Test orchestrator tracks detection statistics."""
        from mycelix_fl.attacks import (
            AttackOrchestrator,
            AdaptiveAttack,
            AttackConfig,
            AttackType,
        )

        orchestrator = AttackOrchestrator(random_seed=42)

        config = AttackConfig(attack_type=AttackType.ADAPTIVE)
        orchestrator.register_attacker("adaptive_1", AdaptiveAttack(config))

        # Simulate detections
        orchestrator.record_detections({"adaptive_1"})
        orchestrator.record_detections(set())  # Not detected
        orchestrator.record_detections({"adaptive_1"})

        stats = orchestrator.get_statistics()
        assert stats["attackers"] == 1
        assert 0.5 < stats["detection_rates"]["adaptive_1"] < 0.8


class TestAttackScenarios:
    """Test preset attack scenarios."""

    def test_gradient_scaling_scenario(self):
        """Test gradient scaling scenario creation."""
        from mycelix_fl.attacks import create_gradient_scaling_scenario

        attacks = create_gradient_scaling_scenario(
            num_attackers=5,
            intensity=20.0,
            stealth=0.0,
            seed=42,
        )

        assert len(attacks) == 5
        for node_id, attack in attacks:
            assert "attacker" in node_id
            assert attack.config.intensity == 20.0

    def test_adaptive_scenario(self):
        """Test adaptive attack scenario creation."""
        from mycelix_fl.attacks import create_adaptive_attack_scenario

        attacks = create_adaptive_attack_scenario(
            num_attackers=3,
            initial_intensity=1.0,
            adaptation_rate=0.15,
            seed=42,
        )

        assert len(attacks) == 3
        for node_id, attack in attacks:
            assert "adaptive" in node_id
            assert attack.config.adaptation_rate == 0.15

    def test_cartel_scenario(self):
        """Test cartel attack scenario creation."""
        from mycelix_fl.attacks import create_cartel_scenario

        attacks = create_cartel_scenario(
            cartel_size=5,
            intensity=2.0,
            stealth=0.7,
            seed=42,
        )

        assert len(attacks) == 5

        # All should have same shared direction
        gradient = np.random.randn(10000).astype(np.float32)
        attacked = [a.attack(gradient) for _, a in attacks]

        # Should be correlated (allow for stealth variation)
        for i in range(len(attacked) - 1):
            corr = np.corrcoef(attacked[i].flatten(), attacked[i + 1].flatten())[0, 1]
            assert corr > 0.5  # Lowered threshold to account for stealth variation

    def test_mixed_scenario(self):
        """Test mixed attack scenario creation."""
        from mycelix_fl.attacks import create_mixed_attack_scenario

        attacks = create_mixed_attack_scenario(total_attackers=10, seed=42)

        assert len(attacks) == 10

        # Should have variety of attack types
        attack_types = set(a.config.attack_type.value for _, a in attacks)
        assert len(attack_types) >= 4  # At least 4 different types


class TestDetectionResistance:
    """Test how attacks perform against detection."""

    def test_stealthy_attacks_harder_to_detect(self):
        """Test stealthy attacks are more similar to honest gradients."""
        from mycelix_fl.attacks import GradientScalingAttack, AttackConfig, AttackType

        gradient = np.random.randn(10000).astype(np.float32)

        # Obvious attack
        config_obvious = AttackConfig(
            attack_type=AttackType.GRADIENT_SCALING,
            intensity=10.0,
            stealth=0.0,
        )
        attack_obvious = GradientScalingAttack(config_obvious)
        attacked_obvious = attack_obvious.attack(gradient)

        # Stealthy attack
        config_stealth = AttackConfig(
            attack_type=AttackType.GRADIENT_SCALING,
            intensity=10.0,
            stealth=0.9,
        )
        attack_stealth = GradientScalingAttack(config_stealth)
        attacked_stealth = attack_stealth.attack(gradient)

        # Stealthy should be more similar to original
        dist_obvious = np.linalg.norm(attacked_obvious - gradient)
        dist_stealth = np.linalg.norm(attacked_stealth - gradient)

        assert dist_stealth < dist_obvious


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
