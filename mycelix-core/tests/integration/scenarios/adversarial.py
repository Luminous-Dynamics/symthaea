# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Adversarial Scenarios: Coordinated Attack Testing

Tests the system against sophisticated coordinated attacks:
- Sybil attacks (multiple fake identities)
- Collusion attacks (coordinated Byzantine behavior)
- Adaptive attacks (learning and adapting to defenses)
- Model poisoning campaigns
"""

import pytest
import numpy as np
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from enum import Enum

from ..fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    FLAggregator,
    ByzantineDetector,
    ReputationBridge,
)
from ..fixtures.nodes import ByzantineNode, HonestNode, AttackType
from ..fixtures.metrics import MetricsCollector, TestReport


# ============================================================================
# Attack Definitions
# ============================================================================

class AttackPhase(Enum):
    """Phases of an attack campaign."""
    RECONNAISSANCE = "reconnaissance"  # Learn system behavior
    INFILTRATION = "infiltration"  # Blend in with honest nodes
    EXECUTION = "execution"  # Execute attack
    EVASION = "evasion"  # Evade detection


@dataclass
class AttackConfig:
    """Configuration for an attack scenario."""
    attack_type: str
    n_attackers: int
    coordination_level: float = 1.0  # 0-1, how coordinated attackers are
    stealth_mode: bool = False
    target_round: Optional[int] = None  # When to execute (None = always)
    reconnaissance_rounds: int = 5  # Rounds to observe before attacking


# ============================================================================
# Coordinated Attack
# ============================================================================

class CoordinatedAttack:
    """
    Simulates a coordinated attack by multiple Byzantine nodes.

    Attackers:
    1. Observe honest gradient patterns
    2. Coordinate their attack strategy
    3. Execute synchronized attack
    4. Adapt to detection attempts
    """

    def __init__(
        self,
        n_attackers: int,
        gradient_dimension: int,
        target_direction: Optional[np.ndarray] = None,
        coordination_level: float = 1.0,
    ):
        self.n_attackers = n_attackers
        self.gradient_dimension = gradient_dimension
        self.coordination_level = coordination_level

        # Target direction for the attack
        if target_direction is None:
            self.target_direction = np.random.randn(gradient_dimension)
            self.target_direction /= np.linalg.norm(self.target_direction)
        else:
            self.target_direction = target_direction / np.linalg.norm(target_direction)

        # Shared state among attackers
        self._observed_gradients: List[np.ndarray] = []
        self._honest_mean: Optional[np.ndarray] = None
        self._honest_std: Optional[np.ndarray] = None
        self._phase = AttackPhase.RECONNAISSANCE
        self._round = 0

        # Create attacker nodes
        self.attackers: List[ByzantineNode] = []
        for i in range(n_attackers):
            attacker = ByzantineNode(
                node_id=f"coordinated_attacker_{i}",
                gradient_dimension=gradient_dimension,
                attack_type="empire",
                attack_params={
                    "n_byzantine": n_attackers,
                    "n_total": 20,  # Will be updated
                    "target_shift": self.target_direction,
                },
                stealth_mode=True,
            )
            self.attackers.append(attacker)

    def observe(self, gradients: List[np.ndarray]):
        """Observe honest gradients for reconnaissance."""
        self._observed_gradients.extend(gradients)

        # Keep recent history
        if len(self._observed_gradients) > 100:
            self._observed_gradients = self._observed_gradients[-100:]

        # Update statistics
        if len(self._observed_gradients) >= 5:
            self._honest_mean = np.mean(self._observed_gradients, axis=0)
            self._honest_std = np.std(self._observed_gradients, axis=0)

            # Share with all attackers
            for attacker in self.attackers:
                for g in gradients:
                    attacker.observe_gradient(g)

    def get_attack_gradients(
        self,
        round_id: int,
        n_total_nodes: int,
    ) -> List[np.ndarray]:
        """Generate coordinated attack gradients."""
        self._round = round_id

        # Update attack parameters
        for attacker in self.attackers:
            attacker.attack_params["n_total"] = n_total_nodes
            attacker.attack_params["n_byzantine"] = self.n_attackers

        # Determine phase
        if len(self._observed_gradients) < 10:
            self._phase = AttackPhase.RECONNAISSANCE
        elif self._phase == AttackPhase.RECONNAISSANCE:
            self._phase = AttackPhase.INFILTRATION
        else:
            self._phase = AttackPhase.EXECUTION

        gradients = []
        for i, attacker in enumerate(self.attackers):
            if self._phase == AttackPhase.RECONNAISSANCE:
                # Behave honestly during reconnaissance
                gradient = self._generate_honest_like_gradient()
            elif self._phase == AttackPhase.INFILTRATION:
                # Slightly poisoned but mostly honest
                gradient = self._generate_infiltration_gradient(i)
            else:
                # Full attack
                gradient = self._generate_attack_gradient(i)

            gradients.append(gradient)

        return gradients

    def _generate_honest_like_gradient(self) -> np.ndarray:
        """Generate gradient that mimics honest behavior."""
        if self._honest_mean is not None:
            noise = np.random.randn(self.gradient_dimension)
            noise *= np.mean(self._honest_std) if self._honest_std is not None else 0.1
            return self._honest_mean + noise
        else:
            return np.random.randn(self.gradient_dimension) * 0.1

    def _generate_infiltration_gradient(self, attacker_idx: int) -> np.ndarray:
        """Generate slightly poisoned gradient during infiltration."""
        honest = self._generate_honest_like_gradient()

        # Small poison toward target
        poison_strength = 0.1 * self.coordination_level
        poison = self.target_direction * np.linalg.norm(honest) * poison_strength

        return honest + poison

    def _generate_attack_gradient(self, attacker_idx: int) -> np.ndarray:
        """Generate full attack gradient."""
        if self._honest_mean is None:
            return self.target_direction

        # Compute gradient to shift aggregate toward target
        n_honest = 20 - self.n_attackers  # Estimated
        n_total = 20

        # Attack gradient = (target * n_total - honest_mean * n_honest) / n_attackers
        target_aggregate = self.target_direction * np.linalg.norm(self._honest_mean)
        attack_gradient = (
            target_aggregate * n_total - self._honest_mean * n_honest
        ) / self.n_attackers

        # Add some variation between attackers (imperfect coordination)
        variation = (1 - self.coordination_level) * np.random.randn(self.gradient_dimension)
        attack_gradient += variation * np.linalg.norm(attack_gradient) * 0.1

        return attack_gradient.astype(np.float32)


# ============================================================================
# Sybil Attack
# ============================================================================

class SybilAttack:
    """
    Simulates Sybil attack with multiple fake identities.

    Attackers create multiple fake node identities to:
    1. Gain disproportionate influence
    2. Outvote honest nodes
    3. Bypass Byzantine thresholds
    """

    def __init__(
        self,
        n_real_attackers: int,
        sybil_factor: int,  # Each attacker creates this many identities
        gradient_dimension: int,
        attack_type: str = "sign_flip",
    ):
        self.n_real_attackers = n_real_attackers
        self.sybil_factor = sybil_factor
        self.gradient_dimension = gradient_dimension
        self.attack_type = attack_type

        # Total sybil identities
        self.n_sybil_nodes = n_real_attackers * sybil_factor

        # Create sybil nodes
        self.sybil_nodes: List[ByzantineNode] = []
        for real_id in range(n_real_attackers):
            for sybil_id in range(sybil_factor):
                node = ByzantineNode(
                    node_id=f"sybil_{real_id}_{sybil_id}",
                    gradient_dimension=gradient_dimension,
                    attack_type=attack_type,
                )
                self.sybil_nodes.append(node)

        self._observed_gradients: List[np.ndarray] = []

    def observe(self, gradients: List[np.ndarray]):
        """Observe honest gradients."""
        self._observed_gradients.extend(gradients)
        for node in self.sybil_nodes:
            for g in gradients:
                node.observe_gradient(g)

    async def get_sybil_gradients(self, round_id: int) -> List[Tuple[str, np.ndarray]]:
        """Get gradients from all sybil nodes."""
        results = []
        for node in self.sybil_nodes:
            gradient = await node.compute_gradient(round_id)
            results.append((node.node_id, gradient))
        return results

    def get_effective_byzantine_ratio(self, n_honest: int) -> float:
        """Calculate effective Byzantine ratio with sybil nodes."""
        total = n_honest + self.n_sybil_nodes
        return self.n_sybil_nodes / total


# ============================================================================
# Adaptive Attack
# ============================================================================

class AdaptiveAttack:
    """
    Attack that learns and adapts to defensive mechanisms.

    The attacker:
    1. Probes the detection system
    2. Identifies detection thresholds
    3. Crafts gradients just under detection threshold
    4. Adapts when detected
    """

    def __init__(
        self,
        n_attackers: int,
        gradient_dimension: int,
        learning_rate: float = 0.1,
    ):
        self.n_attackers = n_attackers
        self.gradient_dimension = gradient_dimension
        self.learning_rate = learning_rate

        # Detection history for learning
        self._detection_history: List[Dict[str, Any]] = []
        self._attack_magnitude = 1.0  # Current attack strength
        self._detected_rounds = 0
        self._evasion_rounds = 0

        # Estimated detection threshold
        self._estimated_threshold = 3.0  # Z-score threshold guess

        # Attackers
        self.attackers: List[ByzantineNode] = []
        for i in range(n_attackers):
            self.attackers.append(
                ByzantineNode(
                    node_id=f"adaptive_attacker_{i}",
                    gradient_dimension=gradient_dimension,
                    attack_type="little",
                    attack_params={"epsilon": self._attack_magnitude},
                )
            )

        self._observed_gradients: List[np.ndarray] = []
        self._honest_stats: Optional[Dict[str, np.ndarray]] = None

    def observe(self, gradients: List[np.ndarray]):
        """Observe honest gradient statistics."""
        self._observed_gradients.extend(gradients)

        if len(self._observed_gradients) >= 5:
            self._honest_stats = {
                "mean": np.mean(self._observed_gradients, axis=0),
                "std": np.std(self._observed_gradients, axis=0),
                "norms": [np.linalg.norm(g) for g in self._observed_gradients],
            }

        for attacker in self.attackers:
            for g in gradients:
                attacker.observe_gradient(g)

    def report_detection(self, detected_ids: Set[str]):
        """Report which attackers were detected."""
        my_detected = sum(1 for a in self.attackers if a.node_id in detected_ids)

        self._detection_history.append({
            "round": len(self._detection_history),
            "detected": my_detected,
            "total_attackers": self.n_attackers,
            "attack_magnitude": self._attack_magnitude,
        })

        if my_detected > 0:
            self._detected_rounds += 1
            self._adapt_to_detection()
        else:
            self._evasion_rounds += 1
            self._increase_attack()

    def _adapt_to_detection(self):
        """Reduce attack magnitude after detection."""
        self._attack_magnitude *= (1 - self.learning_rate)
        self._attack_magnitude = max(0.1, self._attack_magnitude)

        # Update attacker parameters
        for attacker in self.attackers:
            attacker.attack_params["epsilon"] = self._attack_magnitude

    def _increase_attack(self):
        """Increase attack magnitude if not detected."""
        self._attack_magnitude *= (1 + self.learning_rate * 0.5)
        self._attack_magnitude = min(2.0, self._attack_magnitude)

        for attacker in self.attackers:
            attacker.attack_params["epsilon"] = self._attack_magnitude

    async def get_attack_gradients(self, round_id: int) -> List[np.ndarray]:
        """Generate adaptive attack gradients."""
        if self._honest_stats is None:
            # Don't attack without reconnaissance
            return [np.zeros(self.gradient_dimension) for _ in self.attackers]

        gradients = []
        for attacker in self.attackers:
            # Generate gradient that stays under estimated threshold
            base_gradient = await attacker.compute_gradient(round_id)

            # Clip to stay under detection threshold
            honest_mean = self._honest_stats["mean"]
            honest_std = self._honest_stats["std"]

            z_limit = self._estimated_threshold * 0.9  # Stay 10% under

            clipped = np.clip(
                base_gradient,
                honest_mean - z_limit * (honest_std + 1e-10),
                honest_mean + z_limit * (honest_std + 1e-10),
            )

            gradients.append(clipped.astype(np.float32))

        return gradients

    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get statistics about attack effectiveness."""
        total_rounds = self._detected_rounds + self._evasion_rounds

        return {
            "total_rounds": total_rounds,
            "detected_rounds": self._detected_rounds,
            "evasion_rounds": self._evasion_rounds,
            "evasion_rate": self._evasion_rounds / total_rounds if total_rounds > 0 else 0,
            "current_magnitude": self._attack_magnitude,
            "detection_history": self._detection_history,
        }


# ============================================================================
# Adversarial Scenario
# ============================================================================

class AdversarialScenario:
    """
    Complete adversarial test scenario.

    Orchestrates attacks against an FL network and measures resilience.
    """

    def __init__(
        self,
        n_honest_nodes: int = 15,
        attack_config: Optional[AttackConfig] = None,
        gradient_dimension: int = 2000,
    ):
        self.n_honest_nodes = n_honest_nodes
        self.attack_config = attack_config or AttackConfig(
            attack_type="coordinated",
            n_attackers=5,
            coordination_level=1.0,
        )
        self.gradient_dimension = gradient_dimension

        # Create network components
        self.honest_nodes: List[HonestNode] = []
        self.gradient_store = GradientStore()
        self.aggregator = FLAggregator(
            strategy="bulyan",
            byzantine_threshold=0.45,
        )
        self.detector = ByzantineDetector(z_threshold=2.5)
        self.reputation = ReputationBridge()

        # Attack instance
        self.attack: Optional[Any] = None

        self._round = 0
        self._results: List[Dict[str, Any]] = []

    async def setup(self):
        """Initialize the scenario."""
        # Create honest nodes
        for i in range(self.n_honest_nodes):
            node = HonestNode(
                node_id=f"honest_{i}",
                gradient_dimension=self.gradient_dimension,
            )
            self.honest_nodes.append(node)

        # Create attack
        if self.attack_config.attack_type == "coordinated":
            self.attack = CoordinatedAttack(
                n_attackers=self.attack_config.n_attackers,
                gradient_dimension=self.gradient_dimension,
                coordination_level=self.attack_config.coordination_level,
            )
        elif self.attack_config.attack_type == "sybil":
            self.attack = SybilAttack(
                n_real_attackers=self.attack_config.n_attackers,
                sybil_factor=3,
                gradient_dimension=self.gradient_dimension,
            )
        elif self.attack_config.attack_type == "adaptive":
            self.attack = AdaptiveAttack(
                n_attackers=self.attack_config.n_attackers,
                gradient_dimension=self.gradient_dimension,
            )

    async def run_round(self) -> Dict[str, Any]:
        """Run a single round with attack."""
        self._round += 1

        # Collect honest gradients
        honest_gradients = []
        honest_ids = []
        for node in self.honest_nodes:
            gradient = await node.compute_gradient(self._round)
            honest_gradients.append(gradient)
            honest_ids.append(node.node_id)

        # Let attack observe
        if self.attack:
            self.attack.observe(honest_gradients)

        # Get attack gradients
        attack_gradients = []
        attack_ids = []

        if self.attack:
            if isinstance(self.attack, CoordinatedAttack):
                total_nodes = self.n_honest_nodes + self.attack.n_attackers
                attack_gradients = self.attack.get_attack_gradients(
                    self._round, total_nodes
                )
                attack_ids = [a.node_id for a in self.attack.attackers]
            elif isinstance(self.attack, SybilAttack):
                sybil_results = await self.attack.get_sybil_gradients(self._round)
                attack_ids = [r[0] for r in sybil_results]
                attack_gradients = [r[1] for r in sybil_results]
            elif isinstance(self.attack, AdaptiveAttack):
                attack_gradients = await self.attack.get_attack_gradients(self._round)
                attack_ids = [a.node_id for a in self.attack.attackers]

        # Combine all gradients
        all_gradients = honest_gradients + attack_gradients
        all_ids = honest_ids + attack_ids

        # Detect Byzantine
        detected, confidence = await self.detector.detect(all_gradients, all_ids)

        # Report detection to adaptive attack
        if isinstance(self.attack, AdaptiveAttack):
            self.attack.report_detection(detected)

        # Filter for aggregation
        filtered_gradients = []
        filtered_ids = []
        for i, node_id in enumerate(all_ids):
            if node_id not in detected:
                filtered_gradients.append(all_gradients[i])
                filtered_ids.append(node_id)

        # Aggregate
        if len(filtered_gradients) < 3:
            result = {
                "round_id": self._round,
                "success": False,
                "error": "Too few gradients after filtering",
            }
            self._results.append(result)
            return result

        aggregated, metadata = await self.aggregator.aggregate(filtered_gradients)

        # Compute quality against honest mean
        honest_mean = np.mean(honest_gradients, axis=0)
        quality = np.dot(aggregated, honest_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
        )

        # Detection accuracy
        true_byzantines = set(attack_ids)
        tp = len(detected & true_byzantines)
        fp = len(detected - true_byzantines)
        fn = len(true_byzantines - detected)

        result = {
            "round_id": self._round,
            "success": True,
            "aggregated_gradient": aggregated,
            "quality": float(quality),
            "n_honest": len(honest_gradients),
            "n_byzantine": len(attack_gradients),
            "detected": list(detected),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

        self._results.append(result)
        return result

    async def run_rounds(self, n: int) -> List[Dict[str, Any]]:
        """Run multiple rounds."""
        results = []
        for _ in range(n):
            result = await self.run_round()
            results.append(result)
        return results

    def get_attack_effectiveness(self) -> Dict[str, Any]:
        """Analyze attack effectiveness."""
        successful_rounds = [r for r in self._results if r["success"]]

        if not successful_rounds:
            return {"error": "No successful rounds"}

        qualities = [r["quality"] for r in successful_rounds]
        tp_total = sum(r["true_positives"] for r in successful_rounds)
        fp_total = sum(r["false_positives"] for r in successful_rounds)
        fn_total = sum(r["false_negatives"] for r in successful_rounds)

        return {
            "total_rounds": len(self._results),
            "successful_rounds": len(successful_rounds),
            "avg_quality": float(np.mean(qualities)),
            "min_quality": float(np.min(qualities)),
            "quality_degradation": 1.0 - float(np.mean(qualities)),
            "detection_true_positives": tp_total,
            "detection_false_positives": fp_total,
            "detection_false_negatives": fn_total,
            "precision": tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0,
            "recall": tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0,
        }


# ============================================================================
# Adversarial Integration Tests
# ============================================================================

class TestAdversarialScenarios:
    """Test adversarial attack scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_coordinated_attack_scenario(self, metrics_collector):
        """Test coordinated attack with 5 attackers."""
        config = AttackConfig(
            attack_type="coordinated",
            n_attackers=5,
            coordination_level=1.0,
        )

        scenario = AdversarialScenario(
            n_honest_nodes=15,
            attack_config=config,
            gradient_dimension=1000,
        )
        await scenario.setup()

        # Run scenario
        results = await scenario.run_rounds(20)

        effectiveness = scenario.get_attack_effectiveness()

        # System should maintain quality above threshold
        assert effectiveness["avg_quality"] > 0.5, (
            f"Coordinated attack succeeded: quality={effectiveness['avg_quality']}"
        )

        metrics_collector.record_custom("coordinated_attack", effectiveness)

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_sybil_attack_scenario(self, metrics_collector):
        """Test Sybil attack with multiple fake identities."""
        config = AttackConfig(
            attack_type="sybil",
            n_attackers=3,  # 3 real attackers
        )

        scenario = AdversarialScenario(
            n_honest_nodes=15,
            attack_config=config,
            gradient_dimension=1000,
        )
        await scenario.setup()

        # Run scenario
        results = await scenario.run_rounds(20)

        effectiveness = scenario.get_attack_effectiveness()

        # Detection should catch sybil patterns
        assert effectiveness["recall"] > 0.3, "Failed to detect Sybil attack"

        metrics_collector.record_custom("sybil_attack", effectiveness)

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_adaptive_attack_scenario(self, metrics_collector):
        """Test adaptive attack that learns defenses."""
        config = AttackConfig(
            attack_type="adaptive",
            n_attackers=4,
        )

        scenario = AdversarialScenario(
            n_honest_nodes=16,
            attack_config=config,
            gradient_dimension=1000,
        )
        await scenario.setup()

        # Run longer to let attack adapt
        results = await scenario.run_rounds(30)

        effectiveness = scenario.get_attack_effectiveness()

        # Get attack statistics
        attack_stats = scenario.attack.get_attack_statistics()

        # Even adaptive attack should have limited success
        assert effectiveness["avg_quality"] > 0.6, (
            f"Adaptive attack too effective: quality={effectiveness['avg_quality']}"
        )

        metrics_collector.record_custom(
            "adaptive_attack",
            {
                "effectiveness": effectiveness,
                "attack_stats": attack_stats,
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.slow
    async def test_multi_phase_attack(self, metrics_collector, test_report):
        """Test multi-phase coordinated attack."""
        test_report.add_tag("adversarial")
        test_report.add_tag("multi_phase")

        config = AttackConfig(
            attack_type="coordinated",
            n_attackers=6,
            coordination_level=0.9,
            reconnaissance_rounds=10,
        )

        scenario = AdversarialScenario(
            n_honest_nodes=14,
            attack_config=config,
            gradient_dimension=1000,
        )
        await scenario.setup()

        # Phase 1: Reconnaissance (attack observes)
        phase1_results = await scenario.run_rounds(10)

        # Phase 2: Infiltration (subtle attack)
        phase2_results = await scenario.run_rounds(10)

        # Phase 3: Full attack
        phase3_results = await scenario.run_rounds(10)

        # Analyze phases
        phase_qualities = {
            "reconnaissance": np.mean([r["quality"] for r in phase1_results if r["success"]]),
            "infiltration": np.mean([r["quality"] for r in phase2_results if r["success"]]),
            "attack": np.mean([r["quality"] for r in phase3_results if r["success"]]),
        }

        effectiveness = scenario.get_attack_effectiveness()

        # Quality should remain reasonable even in attack phase
        assert phase_qualities["attack"] > 0.5, (
            f"Multi-phase attack succeeded: attack_quality={phase_qualities['attack']}"
        )

        metrics_collector.record_custom(
            "multi_phase_attack",
            {
                "phase_qualities": phase_qualities,
                "effectiveness": effectiveness,
            }
        )

        test_report.passed = True
