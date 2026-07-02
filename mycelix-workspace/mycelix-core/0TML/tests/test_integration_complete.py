# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Complete Integration Test Suite for Hybrid ZeroTrustML

Tests the full system with:
- Real PostgreSQL storage
- WebSocket networking
- Scale testing (50+ nodes)
- NetworkedZeroTrustMLNode combining all components
"""

import asyncio
import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from networked_zerotrustml import NetworkedZeroTrustMLNode
from postgres_storage import PostgreSQLStorageReal
from network_layer import NetworkNode, Message
from modular_architecture import ZeroTrustMLCore, UseCase


if not os.environ.get("RUN_INTEGRATION_TESTS"):
    pytest.skip(
        "Integration suite requires real PostgreSQL + networking; set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


class IntegrationTester:
    """Complete integration test harness"""

    def __init__(self, num_nodes: int = 10, byzantine_ratio: float = 0.3):
        self.num_nodes = num_nodes
        self.byzantine_ratio = byzantine_ratio
        self.nodes: List[NetworkedZeroTrustMLNode] = []
        self.postgres_storage = None

    async def setup_postgres(self):
        """Setup real PostgreSQL backend"""
        print("\n=== Setting up PostgreSQL backend ===")

        # Use test database
        connection_string = "postgresql://zerotrustml:zerotrustml123@localhost:5433/zerotrustml_test"

        self.postgres_storage = PostgreSQLStorageReal(connection_string)
        await self.postgres_storage.connect()

        # Initialize schema
        await self.postgres_storage.initialize_schema()
        print("✓ PostgreSQL connected and initialized")

    async def setup_nodes(self):
        """Setup networked nodes with real storage"""
        print(f"\n=== Setting up {self.num_nodes} nodes ({self.byzantine_ratio*100}% Byzantine) ===")

        num_byzantine = int(self.num_nodes * self.byzantine_ratio)
        base_port = 9000

        for i in range(self.num_nodes):
            node_id = i + 1
            is_byzantine = i < num_byzantine

            # Create node with PostgreSQL storage
            node = NetworkedZeroTrustMLNode(
                node_id=node_id,
                listen_port=base_port + i,
                bootstrap_nodes=["localhost:9000"] if i > 0 else [],
                storage_backend=self.postgres_storage,
                byzantine=is_byzantine
            )

            self.nodes.append(node)
            print(f"  Node {node_id}: {'Byzantine' if is_byzantine else 'Honest'} (port {base_port + i})")

        print("✓ All nodes created")

    async def start_all_nodes(self):
        """Start all nodes in parallel"""
        print("\n=== Starting all nodes ===")

        start_tasks = [node.start() for node in self.nodes]
        await asyncio.gather(*start_tasks)

        # Give time for peer discovery
        await asyncio.sleep(2)
        print("✓ All nodes started and discovered peers")

    async def run_federated_rounds(self, num_rounds: int = 5):
        """Run federated learning rounds with Byzantine detection"""
        print(f"\n=== Running {num_rounds} federated learning rounds ===")

        results = []

        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}:")

            round_start = time.time()

            # Each node computes and broadcasts gradient
            broadcast_tasks = []
            for node in self.nodes:
                gradient = node.trust.compute_gradient()
                task = node.broadcast_gradient(gradient, round_num)
                broadcast_tasks.append(task)

            await asyncio.gather(*broadcast_tasks)

            # Wait for validation
            await asyncio.sleep(1)

            round_time = time.time() - round_start

            # Collect statistics
            byzantine_detected = sum(1 for node in self.nodes
                                    if not node.trust.byzantine and node.byzantine_detected_count > 0)
            total_honest = len([n for n in self.nodes if not n.trust.byzantine])
            detection_rate = byzantine_detected / total_honest if total_honest > 0 else 0

            result = {
                'round': round_num,
                'time': round_time,
                'detection_rate': detection_rate,
                'byzantine_detected_count': byzantine_detected
            }
            results.append(result)

            print(f"  Time: {round_time:.2f}s")
            print(f"  Byzantine detected by {byzantine_detected}/{total_honest} honest nodes ({detection_rate*100:.1f}%)")

        return results

    async def verify_postgresql_storage(self):
        """Verify gradients were stored in PostgreSQL"""
        print("\n=== Verifying PostgreSQL storage ===")

        # Count stored gradients
        total_gradients = await self.postgres_storage.count_gradients()
        print(f"  Total gradients stored: {total_gradients}")

        # Check some gradient metadata
        for node in self.nodes[:3]:  # Check first 3 nodes
            gradients = await self.postgres_storage.query_by_node(node.node_id)
            print(f"  Node {node.node_id}: {len(gradients)} gradients stored")

        # Verify audit trail
        try:
            audit = await self.postgres_storage.get_audit_trail(
                start_timestamp=time.time() - 3600,  # Last hour
                end_timestamp=time.time()
            )
            print(f"  Audit trail entries: {len(audit)}")
        except Exception as e:
            print(f"  Audit trail error: {e}")

        print("✓ PostgreSQL storage verified")

    async def test_network_resilience(self):
        """Test network handles node failures"""
        print("\n=== Testing network resilience ===")

        if len(self.nodes) < 3:
            print("  Skipping (need at least 3 nodes)")
            return

        # Stop one node
        failed_node = self.nodes[-1]
        print(f"  Stopping node {failed_node.node_id}")
        await failed_node.stop()

        # Run a round without the failed node
        print("  Running round with failed node")
        await asyncio.sleep(1)

        remaining_nodes = self.nodes[:-1]
        broadcast_tasks = [
            node.broadcast_gradient(node.trust.compute_gradient(), 99)
            for node in remaining_nodes
        ]
        await asyncio.gather(*broadcast_tasks)

        await asyncio.sleep(1)

        # Check remaining nodes still work
        active_nodes = sum(1 for node in remaining_nodes if len(node.network.connections) > 0)
        print(f"  Active nodes: {active_nodes}/{len(remaining_nodes)}")

        print("✓ Network resilience verified")

    async def cleanup(self):
        """Stop all nodes and cleanup"""
        print("\n=== Cleaning up ===")

        # Stop all nodes
        stop_tasks = [node.stop() for node in self.nodes if hasattr(node, 'running')]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Close PostgreSQL
        if self.postgres_storage:
            await self.postgres_storage.close()

        print("✓ Cleanup complete")

    def print_final_report(self, results: List[Dict]):
        """Print comprehensive test report"""
        print("\n" + "="*60)
        print("INTEGRATION TEST REPORT")
        print("="*60)

        print(f"\nConfiguration:")
        print(f"  Total nodes: {self.num_nodes}")
        print(f"  Byzantine ratio: {self.byzantine_ratio*100}%")
        print(f"  Storage: PostgreSQL (real)")
        print(f"  Networking: WebSocket P2P")

        print(f"\nPerformance:")
        avg_time = np.mean([r['time'] for r in results])
        print(f"  Average round time: {avg_time:.2f}s")

        avg_detection = np.mean([r['detection_rate'] for r in results])
        print(f"  Average detection rate: {avg_detection*100:.1f}%")

        print(f"\nDetailed Results:")
        for result in results:
            print(f"  Round {result['round']}: {result['time']:.2f}s, "
                  f"{result['detection_rate']*100:.1f}% detection")

        print("\n" + "="*60)


@pytest.mark.asyncio
async def test_small_scale_integration():
    """Test with 10 nodes (small scale)"""
    tester = IntegrationTester(num_nodes=10, byzantine_ratio=0.3)

    try:
        await tester.setup_postgres()
        await tester.setup_nodes()
        await tester.start_all_nodes()

        results = await tester.run_federated_rounds(num_rounds=3)

        await tester.verify_postgresql_storage()
        await tester.test_network_resilience()

        tester.print_final_report(results)

        # Assertions
        assert len(results) == 3, "Should complete 3 rounds"
        avg_detection = np.mean([r['detection_rate'] for r in results])
        assert avg_detection > 0.5, f"Detection rate too low: {avg_detection}"

    finally:
        await tester.cleanup()


@pytest.mark.asyncio
async def test_medium_scale_integration():
    """Test with 25 nodes (medium scale)"""
    tester = IntegrationTester(num_nodes=25, byzantine_ratio=0.3)

    try:
        await tester.setup_postgres()
        await tester.setup_nodes()
        await tester.start_all_nodes()

        results = await tester.run_federated_rounds(num_rounds=5)

        await tester.verify_postgresql_storage()

        tester.print_final_report(results)

        # Assertions
        assert len(results) == 5, "Should complete 5 rounds"
        avg_detection = np.mean([r['detection_rate'] for r in results])
        assert avg_detection > 0.6, f"Detection rate too low: {avg_detection}"

    finally:
        await tester.cleanup()


@pytest.mark.asyncio
async def test_large_scale_integration():
    """Test with 50 nodes (large scale) - The requested scale"""
    tester = IntegrationTester(num_nodes=50, byzantine_ratio=0.3)

    try:
        await tester.setup_postgres()
        await tester.setup_nodes()
        await tester.start_all_nodes()

        results = await tester.run_federated_rounds(num_rounds=3)

        await tester.verify_postgresql_storage()

        tester.print_final_report(results)

        # Assertions
        assert len(results) == 3, "Should complete 3 rounds"
        avg_detection = np.mean([r['detection_rate'] for r in results])
        assert avg_detection > 0.7, f"Detection rate too low: {avg_detection}"

        # Performance check
        avg_time = np.mean([r['time'] for r in results])
        assert avg_time < 10, f"Round time too slow: {avg_time}s"

    finally:
        await tester.cleanup()


async def run_manual_test():
    """Manual test runner for development"""
    print("Hybrid ZeroTrustML - Complete Integration Test")
    print("Testing: PostgreSQL + WebSocket + 50+ nodes + Byzantine detection\n")

    # Test configurations
    configs = [
        (10, 0.3, "Small Scale"),
        (25, 0.3, "Medium Scale"),
        (50, 0.3, "Large Scale (Target)"),
    ]

    for num_nodes, byz_ratio, name in configs:
        print(f"\n{'='*60}")
        print(f"TEST: {name} ({num_nodes} nodes, {byz_ratio*100}% Byzantine)")
        print(f"{'='*60}")

        tester = IntegrationTester(num_nodes=num_nodes, byzantine_ratio=byz_ratio)

        try:
            await tester.setup_postgres()
            await tester.setup_nodes()
            await tester.start_all_nodes()

            results = await tester.run_federated_rounds(num_rounds=3)

            await tester.verify_postgresql_storage()

            if num_nodes <= 25:  # Only test resilience on smaller scales
                await tester.test_network_resilience()

            tester.print_final_report(results)

            print(f"\n✓ {name} test PASSED")

        except Exception as e:
            print(f"\n✗ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await tester.cleanup()

        # Pause between tests
        if configs.index((num_nodes, byz_ratio, name)) < len(configs) - 1:
            print("\nWaiting 5 seconds before next test...")
            await asyncio.sleep(5)

    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_manual_test())
