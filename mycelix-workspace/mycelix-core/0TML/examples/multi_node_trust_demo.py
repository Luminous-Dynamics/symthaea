#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Node Trust Demonstration with Holochain Credits

Demonstrates:
- Trust-based node selection using credit balances
- Byzantine node isolation through reputation
- Weighted aggregation based on trust scores
- Trust score evolution over time
- Practical decision-making with trust

This example shows how Holochain credits create a decentralized
reputation system that naturally isolates untrusted nodes.
"""

import asyncio
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import Holochain bridge
try:
    from holochain_credits_bridge import HolochainBridge
    HOLOCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️  Holochain bridge not available, using mock mode")
    HOLOCHAIN_AVAILABLE = False


@dataclass
class NodeReport:
    """A report submitted by a node"""
    node_id: int
    value: float
    is_honest: bool


class TrustedNode:
    """A node in the network with trust-based reputation"""

    def __init__(self, node_id: int, is_honest: bool = True, bridge=None):
        self.node_id = node_id
        self.is_honest = is_honest
        self.bridge = bridge
        self.credits = 0
        self.reports_submitted = 0
        self.reports_accepted = 0

    def submit_report(self, true_value: float) -> float:
        """Submit a report (honest or malicious)"""
        self.reports_submitted += 1

        if self.is_honest:
            # Honest node: report with small random noise
            return true_value + random.gauss(0, 0.1)
        else:
            # Byzantine node: return garbage
            return random.uniform(-100, 100)

    def update_credits(self, amount: int):
        """Update credit balance"""
        self.credits += amount
        if amount > 0:
            self.reports_accepted += 1

    @property
    def trust_score(self) -> float:
        """Calculate trust score (0-1) based on credits"""
        # Credits range from 0 to ~1000 over time
        # Normalize to 0-1 range
        return min(1.0, self.credits / 1000.0)

    @property
    def success_rate(self) -> float:
        """Calculate acceptance rate"""
        if self.reports_submitted == 0:
            return 0.0
        return self.reports_accepted / self.reports_submitted

    def __repr__(self):
        node_type = "⚠️  BYZANTINE" if not self.is_honest else "✅ HONEST"
        return (f"Node {self.node_id:2d} ({node_type}) - "
                f"Credits: {self.credits:4d}, "
                f"Trust: {self.trust_score:.2f}, "
                f"Success: {self.success_rate:.1%}")


class TrustBasedNetwork:
    """Network that makes decisions based on node trust"""

    def __init__(self, conductor_url: str = "ws://localhost:8888"):
        self.nodes: List[TrustedNode] = []
        self.round_num = 0

        # Initialize Holochain bridge
        if HOLOCHAIN_AVAILABLE:
            self.bridge = HolochainBridge(conductor_url, enabled=True)
            print(f"✅ Holochain bridge initialized")
        else:
            self.bridge = None
            print("⚠️  Running in mock mode")

    def register_node(self, node: TrustedNode):
        """Register a node in the network"""
        node.bridge = self.bridge
        self.nodes.append(node)
        print(f"📝 Registered {node}")

    def select_trusted_nodes(self, min_trust: float = 0.3, count: int = 5) -> List[TrustedNode]:
        """Select most trusted nodes above threshold"""
        # Sort by trust score
        trusted = [n for n in self.nodes if n.trust_score >= min_trust]
        trusted.sort(key=lambda n: n.trust_score, reverse=True)
        return trusted[:count]

    def validate_report(self, value: float, expected: float, tolerance: float = 1.0) -> bool:
        """Validate if a report is close to expected value"""
        return abs(value - expected) <= tolerance

    def weighted_aggregate(self, reports: List[Tuple[TrustedNode, float]]) -> float:
        """Aggregate reports weighted by trust scores"""
        if not reports:
            return 0.0

        total_weight = sum(node.trust_score for node, _ in reports)
        if total_weight == 0:
            # No trust yet - equal weights
            return sum(value for _, value in reports) / len(reports)

        weighted_sum = sum(node.trust_score * value for node, value in reports)
        return weighted_sum / total_weight

    async def consensus_round(self, true_value: float):
        """
        Run one consensus round where nodes report a value
        and are rewarded based on accuracy
        """
        self.round_num += 1

        print(f"\n{'='*70}")
        print(f"  Round {self.round_num}: Consensus on value {true_value:.2f}")
        print(f"{'='*70}\n")

        # Collect reports from all nodes
        reports = []
        for node in self.nodes:
            value = node.submit_report(true_value)
            reports.append((node, value))

        # Validate each report and issue credits
        for node, value in reports:
            is_valid = self.validate_report(value, true_value, tolerance=1.0)

            if is_valid:
                # Calculate quality score (inverse of error)
                error = abs(value - true_value)
                quality = max(0.0, 1.0 - error)
                credits_earned = int(quality * 100)

                print(f"✅ {node.node_id:2d}: Report={value:7.2f}, "
                      f"Error={error:.2f}, Credits=+{credits_earned}")

                # Issue credits via Holochain
                if self.bridge:
                    try:
                        action_hash = self.bridge.issue_credits(
                            node_id=node.node_id,
                            event_type="accurate_report",
                            amount=credits_earned,
                            pogq_score=quality
                        )
                    except Exception as e:
                        print(f"   ⚠️  Credit issuance failed: {e}")

                node.update_credits(credits_earned)
            else:
                print(f"❌ {node.node_id:2d}: Report={value:7.2f}, "
                      f"REJECTED (too far from truth)")

        # Calculate consensus using trust-weighted aggregation
        consensus = self.weighted_aggregate(reports)
        consensus_error = abs(consensus - true_value)

        print(f"\n📊 Consensus Result:")
        print(f"   True Value:      {true_value:.2f}")
        print(f"   Consensus:       {consensus:.2f}")
        print(f"   Error:           {consensus_error:.2f}")

        # Show top trusted nodes
        trusted = self.select_trusted_nodes(min_trust=0.3, count=5)
        if trusted:
            print(f"\n🏆 Top {len(trusted)} Trusted Nodes (trust ≥ 0.3):")
            for i, node in enumerate(trusted, 1):
                print(f"   {i}. {node}")

    def print_network_summary(self):
        """Print summary of network trust distribution"""
        print(f"\n{'='*70}")
        print(f"  Network Summary (After {self.round_num} Rounds)")
        print(f"{'='*70}\n")

        # Sort by trust
        sorted_nodes = sorted(self.nodes, key=lambda n: n.trust_score, reverse=True)

        print("Node Rankings by Trust Score:\n")
        for rank, node in enumerate(sorted_nodes, 1):
            node_type = "⚠️  BYZANTINE" if not node.is_honest else "✅ HONEST"
            print(f"{rank:2d}. Node {node.node_id:2d} ({node_type})")
            print(f"    Trust Score:  {node.trust_score:.3f}")
            print(f"    Credits:      {node.credits:4d}")
            print(f"    Success Rate: {node.success_rate:5.1%}")
            print(f"    Reports:      {node.reports_accepted}/{node.reports_submitted}")
            print()

        # Statistics
        honest_nodes = [n for n in self.nodes if n.is_honest]
        byzantine_nodes = [n for n in self.nodes if not n.is_honest]

        avg_honest_trust = sum(n.trust_score for n in honest_nodes) / len(honest_nodes)
        avg_byzantine_trust = (sum(n.trust_score for n in byzantine_nodes) / len(byzantine_nodes)
                              if byzantine_nodes else 0)

        print(f"Network Statistics:")
        print(f"  Honest nodes:     {len(honest_nodes)} (avg trust: {avg_honest_trust:.3f})")
        print(f"  Byzantine nodes:  {len(byzantine_nodes)} (avg trust: {avg_byzantine_trust:.3f})")
        print(f"  Trust separation: {avg_honest_trust - avg_byzantine_trust:.3f}")

        # Check if Byzantine nodes are effectively isolated
        isolated = sum(1 for n in byzantine_nodes if n.trust_score < 0.1)
        print(f"  Isolated (trust < 0.1): {isolated}/{len(byzantine_nodes)} Byzantine nodes")


async def main():
    """Main demonstration"""
    print("=" * 70)
    print("  Multi-Node Trust Demonstration")
    print("  Decentralized Reputation with Holochain Credits")
    print("=" * 70)

    # Create network
    network = TrustBasedNetwork()

    # Register 10 nodes: 8 honest, 2 Byzantine
    print("\n📝 Registering nodes...\n")

    for i in range(1, 9):
        node = TrustedNode(node_id=i, is_honest=True, bridge=network.bridge)
        network.register_node(node)

    for i in range(9, 11):
        node = TrustedNode(node_id=i, is_honest=False, bridge=network.bridge)
        network.register_node(node)

    print(f"\n✅ Registered 8 honest + 2 Byzantine nodes")

    # Run consensus rounds
    num_rounds = 10
    print(f"\n🚀 Running {num_rounds} consensus rounds...\n")

    for round_num in range(1, num_rounds + 1):
        # True value changes each round
        true_value = 50.0 + round_num * 5.0
        await network.consensus_round(true_value)
        await asyncio.sleep(0.1)

    # Print final summary
    network.print_network_summary()

    print("\n" + "=" * 70)
    print("  ✅ Demonstration Complete!")
    print("=" * 70)
    print("\n🔑 Key Observations:")
    print("  1. Honest nodes accumulate trust (high credits)")
    print("  2. Byzantine nodes remain untrusted (low/zero credits)")
    print("  3. Trust-weighted consensus ignores Byzantine nodes")
    print("  4. Reputation emerges naturally from contribution quality")
    print("  5. No central authority needed - DHT stores all credits")
    print("\nThis demonstrates how Holochain credits create a")
    print("self-organizing reputation system that isolates bad actors!")


if __name__ == "__main__":
    asyncio.run(main())
