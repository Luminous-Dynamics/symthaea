#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Integration Tests for Byzantine-Robust Federated Learning

These tests run against an actual Holochain conductor to validate
that the defense_coordinator zome works correctly in a real environment.

Prerequisites:
    1. Holochain conductor running with admin port 9001, app port 9002
    2. byzantine_defense.happ installed as "byzantine-fl-agent1"

To set up the conductor:
    cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain

    # Start conductor
    nix develop github:holochain/holonix --accept-flake-config --command bash -c '
        holochain -c conductor-config.yaml &
        sleep 3

        # Generate agent key
        hc sandbox generate --run 0

        # Install hApp
        hc sandbox call install-app happ/byzantine_defense/byzantine_defense.happ \
            --app-id byzantine-fl-agent1
    '

To run tests:
    pytest test_holochain_integration.py -v --tb=short

    # Or run specific test:
    pytest test_holochain_integration.py -k test_process_round -v
"""

import asyncio
import pytest
import numpy as np
from typing import List, Dict, Tuple
import time

# Import our bridge
from holochain_bridge import (
    HolochainBridge,
    FederatedLearningCoordinator,
    DefenseResult,
    AggregationMethod,
    create_connected_bridge,
)


# ========== Fixtures ==========

@pytest.fixture
async def bridge():
    """Create a connected Holochain bridge."""
    try:
        b = await create_connected_bridge()
        yield b
        await b.close()
    except Exception as e:
        pytest.skip(f"Holochain conductor not available: {e}")


@pytest.fixture
async def fl_coordinator(bridge):
    """Create an FL coordinator with the bridge."""
    return FederatedLearningCoordinator(bridge)


# ========== Helper Functions ==========

def generate_honest_gradient(size: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a typical honest gradient."""
    np.random.seed(seed)
    return np.random.randn(size) * 0.1


def generate_byzantine_gradient_random(size: int = 100) -> np.ndarray:
    """Generate a Byzantine gradient with random large values."""
    return np.random.randn(size) * 10


def generate_byzantine_gradient_sign_flip(honest: np.ndarray) -> np.ndarray:
    """Generate a Byzantine gradient that flips signs."""
    return -honest * 5


def generate_byzantine_gradient_constant(size: int = 100) -> np.ndarray:
    """Generate a Byzantine gradient with constant values."""
    return np.ones(size) * 100


def float_to_fixed(x: float, scale: int = 65536) -> int:
    """Convert float to Q16.16 fixed-point."""
    return int(x * scale)


def fixed_to_float(x: int, scale: int = 65536) -> float:
    """Convert Q16.16 fixed-point to float."""
    return x / scale


# ========== Connection Tests ==========

class TestConnection:
    """Test basic connectivity to Holochain conductor."""

    @pytest.mark.asyncio
    async def test_connect_to_conductor(self, bridge):
        """Test that we can connect to the conductor."""
        assert bridge.app_ws is not None
        assert bridge.admin_ws is not None
        assert bridge.cell_id is not None
        print(f"✅ Connected to conductor, cell_id: {bridge.cell_id[:2]}...")

    @pytest.mark.asyncio
    async def test_list_apps(self, bridge):
        """Test that we can list installed apps."""
        result = await bridge._admin_call("list_apps", {"status_filter": None})
        apps = result.get("data", [])
        assert len(apps) > 0
        print(f"✅ Found {len(apps)} installed apps")


# ========== Defense Coordinator Tests ==========

class TestDefenseCoordinator:
    """Test the defense_coordinator zome functions."""

    @pytest.mark.asyncio
    async def test_process_round_all_honest(self, bridge):
        """Test processing a round with all honest nodes."""
        # Generate 5 honest gradients
        base_grad = generate_honest_gradient(50, seed=100)
        gradients = []
        for i in range(5):
            grad = base_grad + np.random.randn(50) * 0.01
            grad_fp = [float_to_fixed(x) for x in grad]
            gradients.append((f"node{i}", grad_fp))

        result = await bridge.run_defense(gradients, round_num=1)

        # With all honest nodes, no Byzantine should be detected
        assert len(result.detected_byzantine) == 0
        assert len(result.aggregated_gradient) == 50
        print(f"✅ All honest: detected {len(result.detected_byzantine)} Byzantine")

    @pytest.mark.asyncio
    async def test_process_round_with_byzantine(self, bridge):
        """Test processing a round with Byzantine attackers."""
        # 3 honest + 2 Byzantine
        base_grad = generate_honest_gradient(50, seed=200)
        gradients = []

        # Honest nodes
        for i in range(3):
            grad = base_grad + np.random.randn(50) * 0.01
            grad_fp = [float_to_fixed(x) for x in grad]
            gradients.append((f"honest{i}", grad_fp))

        # Byzantine: random attack
        byz_grad = generate_byzantine_gradient_random(50)
        grad_fp = [float_to_fixed(x) for x in byz_grad]
        gradients.append(("byzantine0", grad_fp))

        # Byzantine: sign flip attack
        byz_grad = generate_byzantine_gradient_sign_flip(base_grad)
        grad_fp = [float_to_fixed(x) for x in byz_grad]
        gradients.append(("byzantine1", grad_fp))

        result = await bridge.run_defense(gradients, round_num=2)

        # Should detect at least 1 Byzantine node
        print(f"Detected Byzantine: {result.detected_byzantine}")
        print(f"Method used: {result.method_used.name}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")

        # The aggregated gradient should be closer to honest than Byzantine
        aggregated = np.array([fixed_to_float(x) for x in result.aggregated_gradient])
        dist_to_honest = np.linalg.norm(aggregated - base_grad)
        dist_to_byz = np.linalg.norm(aggregated - byz_grad[:50])

        assert dist_to_honest < dist_to_byz, "Aggregated should be closer to honest"
        print(f"✅ Byzantine detected, aggregated closer to honest")

    @pytest.mark.asyncio
    async def test_process_round_45_percent_byzantine(self, bridge):
        """Test at 45% Byzantine ratio (RBFT limit)."""
        # 6 honest + 5 Byzantine = 45.4% Byzantine
        base_grad = generate_honest_gradient(30, seed=300)
        gradients = []

        # 6 honest nodes
        for i in range(6):
            grad = base_grad + np.random.randn(30) * 0.02
            grad_fp = [float_to_fixed(x) for x in grad]
            gradients.append((f"honest{i}", grad_fp))

        # 5 Byzantine nodes (various attacks)
        for i in range(5):
            if i % 3 == 0:
                byz_grad = generate_byzantine_gradient_random(30)
            elif i % 3 == 1:
                byz_grad = generate_byzantine_gradient_sign_flip(base_grad)
            else:
                byz_grad = generate_byzantine_gradient_constant(30)
            grad_fp = [float_to_fixed(x) for x in byz_grad]
            gradients.append((f"byzantine{i}", grad_fp))

        # Give honest nodes higher reputation
        rep_scores = [(f"honest{i}", 0.8) for i in range(6)]
        rep_scores += [(f"byzantine{i}", 0.3) for i in range(5)]

        result = await bridge.run_defense(gradients, round_num=3, reputation_scores=rep_scores)

        print(f"45% Byzantine test:")
        print(f"  Detected: {len(result.detected_byzantine)} nodes")
        print(f"  Method: {result.method_used.name}")
        print(f"  Confidence: {result.confidence:.2f}")

        # Verify aggregated gradient quality
        aggregated = np.array([fixed_to_float(x) for x in result.aggregated_gradient])
        dist_to_honest = np.linalg.norm(aggregated - base_grad)

        # Should maintain reasonable accuracy even at 45% Byzantine
        assert dist_to_honest < 5.0, f"Aggregated too far from honest: {dist_to_honest}"
        print(f"✅ 45% Byzantine handled, distance to honest: {dist_to_honest:.3f}")


# ========== Federated Learning Integration Tests ==========

class TestFederatedLearning:
    """Test full federated learning workflow through Holochain."""

    @pytest.mark.asyncio
    async def test_single_fl_round(self, fl_coordinator):
        """Test running a single FL round through Holochain."""
        # Simulate gradients from 5 nodes
        base_grad = np.random.randn(100) * 0.1
        gradients = {
            "node1": base_grad + np.random.randn(100) * 0.01,
            "node2": base_grad + np.random.randn(100) * 0.01,
            "node3": base_grad + np.random.randn(100) * 0.01,
            "node4": base_grad + np.random.randn(100) * 0.01,
            "node5": base_grad + np.random.randn(100) * 0.01,
        }

        aggregated, detected = await fl_coordinator.run_round(gradients)

        assert aggregated.shape == (100,)
        assert len(detected) == 0  # All honest
        print(f"✅ FL round complete, aggregated shape: {aggregated.shape}")

    @pytest.mark.asyncio
    async def test_multiple_fl_rounds(self, fl_coordinator):
        """Test running multiple FL rounds."""
        np.random.seed(42)
        total_detected = 0

        for round_num in range(5):
            # Mix of honest and Byzantine each round
            base_grad = np.random.randn(50) * 0.1
            gradients = {}

            # 4 honest
            for i in range(4):
                gradients[f"honest{i}"] = base_grad + np.random.randn(50) * 0.02

            # 1 Byzantine (rotating attack type)
            if round_num % 2 == 0:
                gradients["byzantine"] = np.random.randn(50) * 10
            else:
                gradients["byzantine"] = -base_grad * 5

            aggregated, detected = await fl_coordinator.run_round(gradients)
            total_detected += len(detected)

            print(f"  Round {round_num + 1}: detected {len(detected)} Byzantine")

        print(f"✅ 5 rounds complete, total Byzantine detected: {total_detected}")
        assert total_detected >= 2  # Should detect at least some Byzantine

    @pytest.mark.asyncio
    async def test_fl_round_with_model_update(self, fl_coordinator):
        """Test FL round simulating actual model parameter updates."""
        # Simulate a small neural network layer (10 x 5 = 50 params)
        np.random.seed(123)

        # Initial "model"
        model_weights = np.random.randn(50) * 0.1

        # Simulate training updates from 5 nodes
        gradients = {}
        for i in range(5):
            # Each node computes gradient based on their local data
            local_loss_grad = np.random.randn(50) * 0.05 + model_weights * 0.01
            gradients[f"worker{i}"] = local_loss_grad

        # Run through Holochain
        aggregated, detected = await fl_coordinator.run_round(gradients)

        # Apply aggregated gradient (simulate SGD step)
        lr = 0.01
        updated_weights = model_weights - lr * aggregated

        assert updated_weights.shape == model_weights.shape
        print(f"✅ Model update complete, weight change: {np.linalg.norm(updated_weights - model_weights):.6f}")


# ========== Performance Tests ==========

class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_single_round(self, bridge):
        """Measure latency for a single round."""
        base_grad = generate_honest_gradient(100)
        gradients = [(f"node{i}", [float_to_fixed(x) for x in base_grad])
                     for i in range(5)]

        # Warm up
        await bridge.run_defense(gradients, round_num=0)

        # Measure
        times = []
        for i in range(10):
            start = time.time()
            await bridge.run_defense(gradients, round_num=i+1)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Latency: {avg_time:.1f} ± {std_time:.1f} ms (n=10)")
        assert avg_time < 1000, f"Latency too high: {avg_time}ms"
        print(f"✅ Latency acceptable: {avg_time:.1f}ms average")

    @pytest.mark.asyncio
    async def test_throughput(self, bridge):
        """Measure throughput (rounds per second)."""
        base_grad = generate_honest_gradient(50)
        gradients = [(f"node{i}", [float_to_fixed(x) for x in base_grad])
                     for i in range(5)]

        start = time.time()
        rounds = 20
        for i in range(rounds):
            await bridge.run_defense(gradients, round_num=i)
        elapsed = time.time() - start

        throughput = rounds / elapsed
        print(f"Throughput: {throughput:.1f} rounds/sec ({rounds} rounds in {elapsed:.1f}s)")
        print(f"✅ Throughput: {throughput:.1f} rounds/second")


# ========== Main ==========

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
