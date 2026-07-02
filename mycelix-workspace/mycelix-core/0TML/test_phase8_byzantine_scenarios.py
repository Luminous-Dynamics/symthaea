#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 8 Task 2: Real-World Malicious Scenario Validation

Tests sophisticated Byzantine attack scenarios:
1. Adaptive Attacks (changing behavior over time)
2. Coordinated Attacks (multiple nodes working together)
3. Stealthy Attacks (gradual model poisoning)
4. Sybil Attacks (multiple fake identities)
5. Targeted Attacks (specific objective poisoning)
"""

import sys
import asyncio
import numpy as np
import time
from typing import List, Dict, Callable
from dataclasses import dataclass
from datetime import datetime
import json

sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')

from modular_architecture import ZeroTrustMLFactory


@dataclass
class AttackScenario:
    """Configuration for a specific attack scenario"""
    name: str
    description: str
    num_attackers: int
    attack_behavior: Callable
    stealth_level: str  # "obvious", "moderate", "stealthy"


class AdaptiveAttacker:
    """
    Byzantine node that adapts its behavior based on detection

    Starts with obvious attacks, learns when detected, becomes stealthier
    """
    def __init__(self, node_id: int, gradient_size: int):
        self.node_id = node_id
        self.gradient_size = gradient_size
        self.detected_count = 0
        self.attack_strength = 10.0  # Start strong
        self.rounds_since_detection = 0

    def generate_gradient(self, round_num: int) -> np.ndarray:
        """Generate adaptive attack gradient"""
        # Adapt based on detection history
        if self.detected_count > 0:
            # Reduce attack strength after detection
            self.attack_strength *= 0.7
            self.rounds_since_detection += 1

        # Lay low for several rounds after detection
        if self.rounds_since_detection < 5:
            # Look honest temporarily
            return np.random.randn(self.gradient_size) * 0.1

        # Resume attack with adapted strength
        return np.random.randn(self.gradient_size) * self.attack_strength

    def notify_detected(self):
        """Called when this node is detected as Byzantine"""
        self.detected_count += 1
        self.rounds_since_detection = 0


class CoordinatedAttackers:
    """
    Multiple Byzantine nodes coordinating their attacks

    They communicate and synchronize attacks for maximum impact
    """
    def __init__(self, node_ids: List[int], gradient_size: int):
        self.node_ids = node_ids
        self.gradient_size = gradient_size
        self.coordination_phase = 0
        self.attack_pattern = None

    def generate_coordinated_gradients(self, round_num: int) -> Dict[int, np.ndarray]:
        """Generate coordinated attack gradients for all nodes"""
        gradients = {}

        # Synchronize attack pattern every 5 rounds
        if round_num % 5 == 0:
            self.attack_pattern = np.random.randn(self.gradient_size) * 5.0

        # All attackers use same base pattern with small variations
        for node_id in self.node_ids:
            variation = np.random.randn(self.gradient_size) * 0.5
            gradients[node_id] = self.attack_pattern + variation

        return gradients


class StealthyAttacker:
    """
    Byzantine node that poisons the model gradually

    Mixes honest and malicious contributions to avoid detection
    """
    def __init__(self, node_id: int, gradient_size: int):
        self.node_id = node_id
        self.gradient_size = gradient_size
        self.poison_rate = 0.1  # Start with 10% poisoning

    def generate_gradient(self, round_num: int) -> np.ndarray:
        """Generate stealthy gradient - mostly honest with subtle poisoning"""
        honest_gradient = np.random.randn(self.gradient_size) * 0.1
        poison = np.random.randn(self.gradient_size) * 2.0

        # Mix honest and poisoned (90% honest, 10% poison)
        gradient = (1 - self.poison_rate) * honest_gradient + self.poison_rate * poison

        # Gradually increase poison rate
        self.poison_rate = min(0.3, self.poison_rate * 1.02)

        return gradient


class TargetedAttacker:
    """
    Byzantine node targeting specific model weights

    Tries to push specific features in a particular direction
    """
    def __init__(self, node_id: int, gradient_size: int, target_indices: List[int]):
        self.node_id = node_id
        self.gradient_size = gradient_size
        self.target_indices = target_indices

    def generate_gradient(self, round_num: int) -> np.ndarray:
        """Generate targeted attack gradient"""
        gradient = np.random.randn(self.gradient_size) * 0.1  # Look honest

        # But poison specific target indices
        for idx in self.target_indices:
            gradient[idx] = 10.0  # Large value in target direction

        return gradient


class ByzantineScenarioTester:
    """Orchestrates testing of different Byzantine attack scenarios"""

    def __init__(self, num_honest: int = 40, gradient_size: int = 1000):
        self.num_honest = num_honest
        self.gradient_size = gradient_size
        self.honest_nodes = []

    async def setup_honest_network(self):
        """Create honest node network"""
        print(f"🏗️  Setting up {self.num_honest} honest nodes...")
        self.honest_nodes = []
        for i in range(self.num_honest):
            node = ZeroTrustMLFactory.for_research(i)
            self.honest_nodes.append(node)
        print(f"✅ Honest network ready")

    def generate_honest_gradient(self) -> np.ndarray:
        """Generate honest gradient"""
        return np.random.randn(self.gradient_size) * 0.1

    async def test_scenario(
        self,
        scenario: AttackScenario,
        num_rounds: int = 10
    ) -> Dict:
        """Test a specific attack scenario"""
        print(f"\n{'='*70}")
        print(f"🎯 Testing: {scenario.name}")
        print(f"{'='*70}")
        print(f"Description: {scenario.description}")
        print(f"Attackers: {scenario.num_attackers}")
        print(f"Stealth: {scenario.stealth_level}")
        print(f"Rounds: {num_rounds}")

        detected_per_round = []
        start_time = time.time()

        for round_num in range(num_rounds):
            # Generate all gradients
            all_gradients = []

            # Honest gradients
            for i in range(self.num_honest):
                gradient = self.generate_honest_gradient()
                all_gradients.append((i, gradient, False))  # (node_id, gradient, is_byzantine)

            # Attack gradients
            attack_gradients = scenario.attack_behavior(round_num)
            for node_id, gradient in attack_gradients.items():
                all_gradients.append((node_id, gradient, True))

            # Honest nodes validate all gradients
            detected_this_round = set()

            for honest_node in self.honest_nodes[:10]:  # Use subset for speed
                for node_id, gradient, is_byzantine in all_gradients:
                    is_valid = await honest_node.validate_gradient(
                        gradient,
                        node_id,
                        round_num
                    )

                    if not is_valid and is_byzantine:
                        detected_this_round.add(node_id)

            detected_per_round.append(len(detected_this_round))

            if round_num % 3 == 0:
                detection_rate = len(detected_this_round) / scenario.num_attackers * 100
                print(f"   Round {round_num + 1}: {detection_rate:.1f}% detected")

        duration = time.time() - start_time

        # Calculate results
        total_detected = sum(detected_per_round)
        total_possible = scenario.num_attackers * num_rounds
        overall_detection_rate = total_detected / total_possible

        avg_detected = np.mean(detected_per_round)
        final_detected = detected_per_round[-1]

        result = {
            'scenario': scenario.name,
            'stealth_level': scenario.stealth_level,
            'num_attackers': scenario.num_attackers,
            'num_rounds': num_rounds,
            'total_detected': int(total_detected),
            'total_possible': int(total_possible),
            'overall_detection_rate': float(overall_detection_rate),
            'avg_detected_per_round': float(avg_detected),
            'final_round_detected': int(final_detected),
            'duration_seconds': float(duration)
        }

        # Print results
        print(f"\n📊 Results:")
        print(f"   Overall detection: {overall_detection_rate:.1%}")
        print(f"   Avg per round: {avg_detected:.1f} / {scenario.num_attackers}")
        print(f"   Final round: {final_detected} / {scenario.num_attackers}")
        print(f"   Duration: {duration:.2f}s")

        # Assess result
        if scenario.stealth_level == "obvious":
            success = overall_detection_rate >= 0.9
        elif scenario.stealth_level == "moderate":
            success = overall_detection_rate >= 0.7
        else:  # stealthy
            success = overall_detection_rate >= 0.5

        print(f"   {'✅ PASSED' if success else '⚠️ REVIEW NEEDED'}")
        result['success'] = bool(success)

        return result


async def run_all_byzantine_scenarios():
    """Run comprehensive Byzantine attack scenario validation"""
    print("\n" + "="*70)
    print("🎭 PHASE 8 TASK 2: REAL-WORLD BYZANTINE SCENARIOS")
    print("="*70)

    tester = ByzantineScenarioTester(num_honest=40, gradient_size=1000)
    await tester.setup_honest_network()

    # Define attack scenarios

    # Scenario 1: Adaptive Attackers
    adaptive_attackers = [AdaptiveAttacker(i + 1000, 1000) for i in range(5)]
    def adaptive_attack_behavior(round_num):
        return {attacker.node_id: attacker.generate_gradient(round_num)
                for attacker in adaptive_attackers}

    scenario1 = AttackScenario(
        name="Adaptive Attack",
        description="5 attackers that adapt behavior when detected",
        num_attackers=5,
        attack_behavior=adaptive_attack_behavior,
        stealth_level="moderate"
    )

    # Scenario 2: Coordinated Attack
    coordinated = CoordinatedAttackers([2000 + i for i in range(8)], 1000)
    def coordinated_attack_behavior(round_num):
        return coordinated.generate_coordinated_gradients(round_num)

    scenario2 = AttackScenario(
        name="Coordinated Attack",
        description="8 attackers coordinating their malicious gradients",
        num_attackers=8,
        attack_behavior=coordinated_attack_behavior,
        stealth_level="moderate"
    )

    # Scenario 3: Stealthy Attack
    stealthy_attackers = [StealthyAttacker(3000 + i, 1000) for i in range(3)]
    def stealthy_attack_behavior(round_num):
        return {attacker.node_id: attacker.generate_gradient(round_num)
                for attacker in stealthy_attackers}

    scenario3 = AttackScenario(
        name="Stealthy Attack",
        description="3 attackers with gradual model poisoning",
        num_attackers=3,
        attack_behavior=stealthy_attack_behavior,
        stealth_level="stealthy"
    )

    # Scenario 4: Targeted Attack
    target_indices = [50, 100, 200, 500, 900]  # Target specific features
    targeted_attackers = [TargetedAttacker(4000 + i, 1000, target_indices) for i in range(4)]
    def targeted_attack_behavior(round_num):
        return {attacker.node_id: attacker.generate_gradient(round_num)
                for attacker in targeted_attackers}

    scenario4 = AttackScenario(
        name="Targeted Attack",
        description="4 attackers targeting specific model features",
        num_attackers=4,
        attack_behavior=targeted_attack_behavior,
        stealth_level="moderate"
    )

    # Scenario 5: Obvious Attack (baseline)
    def obvious_attack_behavior(round_num):
        return {5000 + i: np.random.randn(1000) * 100 for i in range(10)}

    scenario5 = AttackScenario(
        name="Obvious Attack (Baseline)",
        description="10 attackers with large random noise",
        num_attackers=10,
        attack_behavior=obvious_attack_behavior,
        stealth_level="obvious"
    )

    # Run all scenarios
    scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5]
    results = []

    for scenario in scenarios:
        result = await tester.test_scenario(scenario, num_rounds=15)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("📈 COMPREHENSIVE BYZANTINE SCENARIO SUMMARY")
    print("="*70)

    print(f"\n{'Scenario':<30} {'Stealth':<12} {'Detection':<12} {'Result':<10}")
    print("-" * 70)

    all_passed = True
    for result in results:
        status = "✅ PASS" if result['success'] else "⚠️  REVIEW"
        print(f"{result['scenario']:<30} {result['stealth_level']:<12} "
              f"{result['overall_detection_rate']:>6.1%}     {status}")
        all_passed = all_passed and result['success']

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL BYZANTINE SCENARIOS: ✅ SYSTEM RESILIENT")
        print("   ZeroTrustML successfully detects diverse real-world attacks!")
    else:
        print("⚠️  SOME SCENARIOS NEED REVIEW")
        print("   Review scenarios marked above for improvements")
    print("="*70)

    return {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 8 Task 2 - Byzantine Scenario Validation',
        'num_scenarios': len(scenarios),
        'all_passed': bool(all_passed),
        'scenarios': results
    }


async def main():
    """Main entry point"""
    start_time = time.time()

    results = await run_all_byzantine_scenarios()

    results['test_duration_seconds'] = time.time() - start_time

    # Save results
    output_file = '/srv/luminous-dynamics/Mycelix-Core/0TML/phase8_byzantine_scenarios_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")
    print(f"⏱️  Total duration: {results['test_duration_seconds']:.1f}s")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())

    # Exit code based on pass/fail
    exit(0 if results['all_passed'] else 1)
