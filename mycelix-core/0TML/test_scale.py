#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Scale Testing Framework for ZeroTrustML
Tests Byzantine resistance with 50+ nodes
"""

import sys
import asyncio
import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp

sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - using simulated gradients")

from modular_architecture import ZeroTrustMLFactory, UseCase, MemoryStorage


@dataclass
class ScaleTestConfig:
    """Configuration for scale testing"""
    num_honest: int
    num_byzantine: int
    num_rounds: int
    gradient_size: int
    use_real_ml: bool = False
    storage_backend: str = "memory"  # memory, postgresql, holochain
    parallel_workers: int = 4


@dataclass
class ScaleTestResults:
    """Results from scale test"""
    total_nodes: int
    honest_nodes: int
    byzantine_nodes: int
    num_rounds: int
    avg_detection_rate: float
    avg_round_time: float
    total_time: float
    memory_usage_mb: float
    gradients_processed: int
    throughput: float  # gradients/sec


class ScaleTestNode:
    """Individual node for scale testing"""

    def __init__(
        self,
        node_id: int,
        byzantine: bool,
        gradient_size: int,
        use_real_ml: bool = False
    ):
        self.node_id = node_id
        self.byzantine = byzantine
        self.gradient_size = gradient_size
        self.use_real_ml = use_real_ml

        # Create ZeroTrustML node
        self.trust_node = ZeroTrustMLFactory.for_research(node_id)

    def compute_gradient(self) -> np.ndarray:
        """Compute gradient (honest or Byzantine)"""
        if self.byzantine:
            # Byzantine attack strategies
            attack_type = self.node_id % 5
            if attack_type == 0:
                return np.random.randn(self.gradient_size) * 100  # Large noise
            elif attack_type == 1:
                return np.zeros(self.gradient_size)  # All zeros
            elif attack_type == 2:
                return np.ones(self.gradient_size) * 5.0  # Constant
            elif attack_type == 3:
                grad = np.random.randn(self.gradient_size)
                return -grad * 10  # Sign flip
            else:
                return np.random.randn(self.gradient_size) * (np.random.rand(self.gradient_size) < 0.01)  # Sparse
        else:
            # Honest gradient
            return np.random.randn(self.gradient_size) * 0.1

    async def validate_peer_gradients(
        self,
        peer_gradients: List[tuple],
        round_num: int
    ) -> Dict:
        """Validate gradients from peers"""
        detected_byzantine = set()

        for peer_id, gradient in peer_gradients:
            is_valid = await self.trust_node.validate_gradient(
                gradient,
                peer_id,
                round_num
            )

            if not is_valid:
                detected_byzantine.add(peer_id)

        return {
            'detected': list(detected_byzantine),
            'total_validated': len(peer_gradients)
        }


class ScaleTester:
    """Scale testing orchestrator"""

    def __init__(self, config: ScaleTestConfig):
        self.config = config
        self.nodes: List[ScaleTestNode] = []
        self.byzantine_ids: List[int] = []

    def setup_network(self):
        """Create network of nodes"""
        print(f"\n🏗️  Setting up network...")
        print(f"   Honest nodes: {self.config.num_honest}")
        print(f"   Byzantine nodes: {self.config.num_byzantine}")
        print(f"   Total: {self.config.num_honest + self.config.num_byzantine}")

        # Create honest nodes
        for i in range(self.config.num_honest):
            node = ScaleTestNode(
                node_id=i,
                byzantine=False,
                gradient_size=self.config.gradient_size,
                use_real_ml=self.config.use_real_ml
            )
            self.nodes.append(node)

        # Create Byzantine nodes
        for i in range(self.config.num_byzantine):
            node_id = self.config.num_honest + i
            node = ScaleTestNode(
                node_id=node_id,
                byzantine=True,
                gradient_size=self.config.gradient_size,
                use_real_ml=self.config.use_real_ml
            )
            self.nodes.append(node)
            self.byzantine_ids.append(node_id)

        print(f"✅ Network ready with {len(self.nodes)} nodes")
        print(f"🚨 Byzantine nodes: {self.byzantine_ids}")

    async def run_federated_round(self, round_num: int) -> Dict:
        """Run one round of federated learning"""
        round_start = time.time()

        # All nodes compute gradients
        gradients = []
        for node in self.nodes:
            gradient = node.compute_gradient()
            gradients.append((node.node_id, gradient))

        # Honest nodes validate received gradients
        detection_results = []

        # Use asyncio.gather for parallel validation
        validation_tasks = [
            node.validate_peer_gradients(gradients, round_num)
            for node in self.nodes
            if not node.byzantine
        ]

        validation_results = await asyncio.gather(*validation_tasks)
        detection_results.extend(validation_results)

        # Calculate detection rate
        detected_byzantine = set()
        for result in detection_results:
            detected_byzantine.update(result['detected'])

        detection_rate = len(detected_byzantine) / len(self.byzantine_ids) if self.byzantine_ids else 1.0
        round_time = time.time() - round_start

        return {
            'round_num': round_num,
            'detection_rate': detection_rate,
            'detected_count': len(detected_byzantine),
            'total_byzantine': len(self.byzantine_ids),
            'round_time': round_time,
            'gradients_processed': len(gradients)
        }

    async def run_test(self) -> ScaleTestResults:
        """Run complete scale test"""
        print("\n" + "="*60)
        print("🚀 STARTING SCALE TEST")
        print("="*60)

        self.setup_network()

        print(f"\n📊 Running {self.config.num_rounds} training rounds...")
        test_start = time.time()

        round_results = []
        for round_num in range(self.config.num_rounds):
            result = await self.run_federated_round(round_num)
            round_results.append(result)

            if round_num % 5 == 0:
                print(f"   Round {round_num + 1}/{self.config.num_rounds}: "
                      f"{result['detection_rate']:.1%} detection, "
                      f"{result['round_time']:.2f}s")

        total_time = time.time() - test_start

        # Calculate statistics
        avg_detection = np.mean([r['detection_rate'] for r in round_results])
        avg_round_time = np.mean([r['round_time'] for r in round_results])
        total_gradients = sum(r['gradients_processed'] for r in round_results)
        throughput = total_gradients / total_time

        # Estimate memory usage (rough)
        gradient_mb = (self.config.gradient_size * 8) / (1024 * 1024)  # float64 = 8 bytes
        memory_mb = gradient_mb * len(self.nodes) * 2  # Current + history

        results = ScaleTestResults(
            total_nodes=len(self.nodes),
            honest_nodes=self.config.num_honest,
            byzantine_nodes=self.config.num_byzantine,
            num_rounds=self.config.num_rounds,
            avg_detection_rate=avg_detection,
            avg_round_time=avg_round_time,
            total_time=total_time,
            memory_usage_mb=memory_mb,
            gradients_processed=total_gradients,
            throughput=throughput
        )

        self.print_results(results)
        return results

    def print_results(self, results: ScaleTestResults):
        """Print test results"""
        print("\n" + "="*60)
        print("📊 SCALE TEST RESULTS")
        print("="*60)

        print(f"\n🏗️  Network Size:")
        print(f"   Total nodes: {results.total_nodes}")
        print(f"   Honest: {results.honest_nodes}")
        print(f"   Byzantine: {results.byzantine_nodes}")

        print(f"\n🎯 Byzantine Detection:")
        print(f"   Average detection rate: {results.avg_detection_rate:.1%}")
        print(f"   Target: 90%")
        if results.avg_detection_rate >= 0.9:
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED")

        print(f"\n⚡ Performance:")
        print(f"   Total time: {results.total_time:.2f}s")
        print(f"   Average round time: {results.avg_round_time:.3f}s")
        print(f"   Throughput: {results.throughput:.1f} gradients/sec")

        print(f"\n💾 Resource Usage:")
        print(f"   Memory (estimated): {results.memory_usage_mb:.1f} MB")
        print(f"   Gradients processed: {results.gradients_processed}")

        print("\n" + "="*60)


# ============================================================
# Predefined Test Scenarios
# ============================================================

async def test_small_scale():
    """Test with 10 nodes (baseline)"""
    config = ScaleTestConfig(
        num_honest=7,
        num_byzantine=3,
        num_rounds=10,
        gradient_size=1000
    )
    tester = ScaleTester(config)
    return await tester.run_test()


async def test_medium_scale():
    """Test with 25 nodes"""
    config = ScaleTestConfig(
        num_honest=20,
        num_byzantine=5,
        num_rounds=10,
        gradient_size=1000
    )
    tester = ScaleTester(config)
    return await tester.run_test()


async def test_large_scale():
    """Test with 50 nodes"""
    config = ScaleTestConfig(
        num_honest=40,
        num_byzantine=10,
        num_rounds=10,
        gradient_size=1000
    )
    tester = ScaleTester(config)
    return await tester.run_test()


async def test_very_large_scale():
    """Test with 100 nodes"""
    config = ScaleTestConfig(
        num_honest=80,
        num_byzantine=20,
        num_rounds=10,
        gradient_size=1000
    )
    tester = ScaleTester(config)
    return await tester.run_test()


async def test_extreme_scale():
    """Test with 200 nodes"""
    config = ScaleTestConfig(
        num_honest=160,
        num_byzantine=40,
        num_rounds=5,  # Fewer rounds for extreme scale
        gradient_size=1000
    )
    tester = ScaleTester(config)
    return await tester.run_test()


async def run_all_scale_tests():
    """Run complete scale test suite"""
    print("\n" + "="*60)
    print("🏆 COMPLETE SCALE TEST SUITE")
    print("="*60)

    tests = [
        ("Small (10 nodes)", test_small_scale),
        ("Medium (25 nodes)", test_medium_scale),
        ("Large (50 nodes)", test_large_scale),
        ("Very Large (100 nodes)", test_very_large_scale),
        ("Extreme (200 nodes)", test_extreme_scale),
    ]

    all_results = []

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        try:
            result = await test_fn()
            all_results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("📈 SCALE TEST SUMMARY")
    print("="*60)

    print(f"\n{'Test':<25} {'Nodes':<10} {'Detection':<15} {'Time':<10} {'Throughput':<15}")
    print("-" * 75)

    for name, result in all_results:
        print(f"{name:<25} "
              f"{result.total_nodes:<10} "
              f"{result.avg_detection_rate:>6.1%}{'  ✅' if result.avg_detection_rate >= 0.9 else '  ❌':<9} "
              f"{result.total_time:>6.1f}s   "
              f"{result.throughput:>8.1f} grad/s")

    # Performance analysis
    print("\n💡 Key Insights:")

    if all_results:
        # Check if detection rate stays high
        detection_rates = [r.avg_detection_rate for _, r in all_results]
        if all(dr >= 0.9 for dr in detection_rates):
            print("   ✅ Byzantine detection maintains >90% across all scales")
        else:
            print("   ⚠️  Byzantine detection degrades at scale")

        # Check if throughput scales
        throughputs = [r.throughput for _, r in all_results]
        if throughputs[-1] > throughputs[0] * 0.5:
            print("   ✅ Throughput scales reasonably well")
        else:
            print("   ⚠️  Throughput degrades significantly at scale")

        # Check memory usage
        max_memory = max(r.memory_usage_mb for _, r in all_results)
        if max_memory < 500:
            print(f"   ✅ Memory usage stays low ({max_memory:.0f} MB max)")
        else:
            print(f"   ⚠️  High memory usage ({max_memory:.0f} MB max)")


def main():
    """Main entry point"""
    print("\n🏗️  ZeroTrustML Scale Testing Framework")
    print("="*60)

    print("\nAvailable tests:")
    print("  1. Small scale (10 nodes)")
    print("  2. Medium scale (25 nodes)")
    print("  3. Large scale (50 nodes)")
    print("  4. Very large scale (100 nodes)")
    print("  5. Extreme scale (200 nodes)")
    print("  6. Run all tests")
    print("  7. Custom configuration")

    choice = input("\nEnter choice (1-7): ").strip()

    if choice == '1':
        asyncio.run(test_small_scale())
    elif choice == '2':
        asyncio.run(test_medium_scale())
    elif choice == '3':
        asyncio.run(test_large_scale())
    elif choice == '4':
        asyncio.run(test_very_large_scale())
    elif choice == '5':
        asyncio.run(test_extreme_scale())
    elif choice == '6':
        asyncio.run(run_all_scale_tests())
    elif choice == '7':
        # Custom configuration
        num_honest = int(input("Number of honest nodes: "))
        num_byzantine = int(input("Number of Byzantine nodes: "))
        num_rounds = int(input("Number of rounds: "))

        config = ScaleTestConfig(
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            num_rounds=num_rounds,
            gradient_size=1000
        )

        tester = ScaleTester(config)
        asyncio.run(tester.run_test())
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()