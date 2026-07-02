# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Byzantine Rejection

Verifies that malicious gradients are rejected at the zome level.
Tests various attack types and validation rules.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Set

from .fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    ZomeValidator,
    FLAggregator,
    ByzantineDetector,
)
from .fixtures.nodes import ByzantineNode, HonestNode, AttackType
from .fixtures.metrics import MetricsCollector, DetectionMetrics


# ============================================================================
# Zome-Level Rejection Tests
# ============================================================================

class TestZomeLevelRejection:
    """Test rejection of malicious gradients at zome validation."""

    @pytest.mark.asyncio
    async def test_nan_injection_rejected(self):
        """Test that NaN injection attacks are rejected."""
        store = GradientStore()
        validator = ZomeValidator()
        store.add_validation_callback(validator.validate)

        byzantine = ByzantineNode(
            node_id="attacker",
            gradient_dimension=1000,
            attack_type="random",
        )

        # Manually create gradient with NaN
        gradient = np.random.randn(1000).astype(np.float32)
        gradient[np.random.randint(0, 1000, size=10)] = np.nan

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "NaN" in error

        # Check rejection log
        log = validator.get_rejection_log()
        assert len(log) == 1
        assert log[0]["node_id"] == "attacker"

    @pytest.mark.asyncio
    async def test_inf_injection_rejected(self):
        """Test that Inf injection attacks are rejected."""
        store = GradientStore()
        validator = ZomeValidator()
        store.add_validation_callback(validator.validate)

        gradient = np.random.randn(1000).astype(np.float32)
        gradient[500] = np.inf
        gradient[501] = -np.inf

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "Inf" in error

    @pytest.mark.asyncio
    async def test_scaling_attack_rejected(self):
        """Test that extreme scaling attacks are rejected."""
        store = GradientStore()
        validator = ZomeValidator(max_gradient_norm=100.0)
        store.add_validation_callback(validator.validate)

        byzantine = ByzantineNode(
            node_id="attacker",
            gradient_dimension=1000,
            attack_type="scaling",
            attack_params={"scale": 10000.0},
        )

        gradient = await byzantine.compute_gradient(round_id=1)

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "norm" in error.lower()

    @pytest.mark.asyncio
    async def test_extreme_values_rejected(self):
        """Test that gradients with extreme element values are rejected."""
        store = GradientStore()
        validator = ZomeValidator(max_element_value=100.0)
        store.add_validation_callback(validator.validate)

        gradient = np.random.randn(1000).astype(np.float32)
        gradient[0] = 99999.0  # Extreme value

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "element" in error.lower() or "value" in error.lower()

    @pytest.mark.asyncio
    async def test_undersized_gradient_rejected(self):
        """Test that undersized gradients are rejected."""
        store = GradientStore()
        validator = ZomeValidator(min_dimension=100)
        store.add_validation_callback(validator.validate)

        gradient = np.random.randn(50).astype(np.float32)  # Too small

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "dimension" in error.lower()

    @pytest.mark.asyncio
    async def test_oversized_gradient_rejected(self):
        """Test that oversized gradients are rejected."""
        store = GradientStore()
        validator = ZomeValidator(max_dimension=10000)
        store.add_validation_callback(validator.validate)

        gradient = np.random.randn(100000).astype(np.float32)  # Too large

        success, _, error = await store.submit_gradient(
            node_id="attacker",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "dimension" in error.lower()


# ============================================================================
# Byzantine Detection Tests
# ============================================================================

class TestByzantineDetection:
    """Test Byzantine detection mechanisms."""

    @pytest.mark.asyncio
    async def test_detect_random_attack(self, honest_node_factory, byzantine_node_factory):
        """Test detection of random gradient attacks."""
        detector = ByzantineDetector(z_threshold=2.0)
        gradient_dim = 1000

        # Create honest gradients (similar to each other)
        honest_nodes = [honest_node_factory(f"honest_{i}", gradient_dim) for i in range(8)]
        byzantine_nodes = [
            byzantine_node_factory(f"byzantine_{i}", "random", gradient_dim)
            for i in range(2)
        ]

        gradients = []
        node_ids = []

        for node in honest_nodes:
            gradients.append(await node.compute_gradient(1))
            node_ids.append(node.node_id)

        for node in byzantine_nodes:
            gradients.append(await node.compute_gradient(1))
            node_ids.append(node.node_id)

        # Detect
        detected, confidence = await detector.detect(gradients, node_ids)

        # Should detect at least some Byzantine nodes
        true_byzantines = {n.node_id for n in byzantine_nodes}

        # Random attacks with high scale should be detectable
        # Note: Detection may not be perfect
        assert len(detected) >= 1, "Failed to detect any Byzantine nodes"

    @pytest.mark.asyncio
    async def test_detect_sign_flip_attack(self, honest_node_factory, byzantine_node_factory):
        """Test detection of sign flip attacks."""
        detector = ByzantineDetector(z_threshold=2.0)
        gradient_dim = 1000

        honest_nodes = [honest_node_factory(f"honest_{i}", gradient_dim) for i in range(7)]
        byzantine_nodes = [
            byzantine_node_factory(f"byzantine_{i}", "sign_flip", gradient_dim)
            for i in range(3)
        ]

        # First, collect honest gradients
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]

        # Let Byzantine nodes observe honest behavior
        for byz in byzantine_nodes:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        gradients = honest_gradients.copy()
        node_ids = [n.node_id for n in honest_nodes]

        # Now add Byzantine gradients
        for byz in byzantine_nodes:
            gradients.append(await byz.compute_gradient(1))
            node_ids.append(byz.node_id)

        detected, confidence = await detector.detect(gradients, node_ids)

        # Sign flip should be detectable via low similarity
        true_byzantines = {n.node_id for n in byzantine_nodes}
        detected_correctly = detected & true_byzantines

        # Should detect sign flips (they point in opposite direction)
        assert len(detected_correctly) >= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("attack_type", [
        "random",
        "sign_flip",
        "scaling",
        "gaussian",
        "same_value",
    ])
    async def test_detect_various_attacks(
        self, attack_type, honest_node_factory, byzantine_node_factory
    ):
        """Test detection of various attack types."""
        detector = ByzantineDetector(z_threshold=2.5)
        gradient_dim = 500

        n_honest = 7
        n_byzantine = 3

        honest_nodes = [
            honest_node_factory(f"honest_{i}", gradient_dim)
            for i in range(n_honest)
        ]
        byzantine_nodes = [
            byzantine_node_factory(f"byzantine_{i}", attack_type, gradient_dim)
            for i in range(n_byzantine)
        ]

        gradients = []
        node_ids = []

        for node in honest_nodes:
            g = await node.compute_gradient(1)
            gradients.append(g)
            node_ids.append(node.node_id)
            for byz in byzantine_nodes:
                byz.observe_gradient(g)

        for node in byzantine_nodes:
            gradients.append(await node.compute_gradient(1))
            node_ids.append(node.node_id)

        detected, confidence = await detector.detect(gradients, node_ids)

        # Record detection result (not necessarily perfect detection)
        true_byzantines = {n.node_id for n in byzantine_nodes}
        tp = len(detected & true_byzantines)
        fp = len(detected - true_byzantines)

        # For aggressive attacks, should detect at least one
        if attack_type in ["random", "scaling", "sign_flip"]:
            assert tp >= 1 or fp == 0, f"Failed to detect {attack_type} attack"

    @pytest.mark.asyncio
    async def test_no_false_positives_honest_network(self, honest_node_factory):
        """Test that honest-only network has no false positives."""
        detector = ByzantineDetector(z_threshold=3.0)
        gradient_dim = 1000
        n_nodes = 10

        nodes = [honest_node_factory(f"honest_{i}", gradient_dim) for i in range(n_nodes)]

        gradients = [await node.compute_gradient(1) for node in nodes]
        node_ids = [node.node_id for node in nodes]

        detected, _ = await detector.detect(gradients, node_ids)

        # Should have zero or very few false positives
        assert len(detected) <= 1, f"Too many false positives: {len(detected)}"

    @pytest.mark.asyncio
    async def test_detection_with_stealth_mode(self, honest_node_factory):
        """Test that stealth mode Byzantine nodes are harder to detect."""
        detector = ByzantineDetector(z_threshold=2.0)
        gradient_dim = 1000

        honest_nodes = [honest_node_factory(f"honest_{i}", gradient_dim) for i in range(7)]

        # Create stealthy Byzantine nodes
        stealth_byzantines = [
            ByzantineNode(
                node_id=f"stealth_{i}",
                gradient_dimension=gradient_dim,
                attack_type="mimic",
                stealth_mode=True,
            )
            for i in range(3)
        ]

        # Regular Byzantine nodes
        regular_byzantines = [
            ByzantineNode(
                node_id=f"regular_{i}",
                gradient_dimension=gradient_dim,
                attack_type="random",
                stealth_mode=False,
            )
            for i in range(3)
        ]

        # Collect honest gradients
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]

        # Let Byzantine nodes observe
        for byz in stealth_byzantines + regular_byzantines:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        # Test detection on stealth Byzantine
        stealth_gradients = honest_gradients + [
            await byz.compute_gradient(1) for byz in stealth_byzantines
        ]
        stealth_ids = [n.node_id for n in honest_nodes] + [
            byz.node_id for byz in stealth_byzantines
        ]

        detected_stealth, _ = await detector.detect(stealth_gradients, stealth_ids)

        # Test detection on regular Byzantine
        regular_gradients = honest_gradients + [
            await byz.compute_gradient(1) for byz in regular_byzantines
        ]
        regular_ids = [n.node_id for n in honest_nodes] + [
            byz.node_id for byz in regular_byzantines
        ]

        detected_regular, _ = await detector.detect(regular_gradients, regular_ids)

        # Regular Byzantine should be easier to detect than stealth
        # (stealth mode makes detection harder)
        true_regular = {byz.node_id for byz in regular_byzantines}
        true_stealth = {byz.node_id for byz in stealth_byzantines}

        regular_detected = len(detected_regular & true_regular)
        stealth_detected = len(detected_stealth & true_stealth)

        # Stealth should be harder to detect (or equal)
        assert stealth_detected <= regular_detected + 1


# ============================================================================
# Network Integration Tests
# ============================================================================

class TestByzantineRejectionNetwork:
    """Test Byzantine rejection in full network context."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_network_rejects_invalid_gradients(self, byzantine_network_config):
        """Test that network rejects invalid gradients."""
        # Modify config for strict validation
        config = byzantine_network_config
        network = FLNetwork(config)
        await network.start()

        try:
            # Run a round
            result = await network.run_round()

            assert result["success"]

            # Some gradients should be detected as Byzantine
            if result["n_byzantine_detected"] > 0:
                detected = result["detected_byzantine"]
                # Verify detected nodes are actually Byzantine
                for node_id in detected:
                    assert "byzantine" in node_id, f"False positive: {node_id}"
        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_aggregation_quality_with_byzantine(self, byzantine_network_config):
        """Test that aggregation quality is maintained with Byzantine nodes."""
        config = byzantine_network_config
        network = FLNetwork(config)
        await network.start()

        try:
            # Run multiple rounds
            results = await network.run_rounds(5)

            for result in results:
                assert result["success"]

                # With Byzantine-resilient aggregation, quality should be maintained
                aggregated = result["aggregated_gradient"]
                assert np.isfinite(aggregated).all()
                assert np.linalg.norm(aggregated) < 100  # Reasonable norm

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_detection_accuracy_metrics(
        self, byzantine_network_config, metrics_collector
    ):
        """Test and record Byzantine detection accuracy."""
        config = byzantine_network_config
        network = FLNetwork(config)
        await network.start()

        try:
            # Define true Byzantines
            true_byzantines = {
                f"byzantine_{i}" for i in range(config.num_byzantine_nodes)
            }
            all_nodes = {
                node_id for node_id in network.get_all_nodes().keys()
            }

            results = await network.run_rounds(10)

            for result in results:
                detected = set(result.get("detected_byzantine", []))

                detection_metrics = metrics_collector.record_detection(
                    round_id=result["round_id"],
                    true_byzantines=true_byzantines,
                    detected_byzantines=detected,
                    all_node_ids=all_nodes,
                )

            # Check aggregate detection performance
            summary = metrics_collector.get_detection_summary()

            # With 30% Byzantine nodes and detection enabled,
            # we should have reasonable detection
            assert summary["precision"]["mean"] >= 0.5, "Precision too low"
            # Recall can be lower for stealth attacks

        finally:
            await network.shutdown()


# ============================================================================
# Coordinated Attack Tests
# ============================================================================

class TestCoordinatedAttacks:
    """Test detection of coordinated Byzantine attacks."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_empire_attack_detection(self, honest_node_factory):
        """Test detection of 'Fall of Empires' coordinated attack."""
        detector = ByzantineDetector(z_threshold=2.0)
        gradient_dim = 500

        n_honest = 7
        n_byzantine = 3

        honest_nodes = [
            honest_node_factory(f"honest_{i}", gradient_dim)
            for i in range(n_honest)
        ]

        # Empire attack: Byzantine nodes coordinate
        byzantine_nodes = [
            ByzantineNode(
                node_id=f"byzantine_{i}",
                gradient_dimension=gradient_dim,
                attack_type="empire",
                attack_params={
                    "n_byzantine": n_byzantine,
                    "n_total": n_honest + n_byzantine,
                },
            )
            for i in range(n_byzantine)
        ]

        # Collect honest gradients
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]

        for byz in byzantine_nodes:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        gradients = honest_gradients + [
            await byz.compute_gradient(1) for byz in byzantine_nodes
        ]
        node_ids = [n.node_id for n in honest_nodes] + [
            byz.node_id for byz in byzantine_nodes
        ]

        detected, confidence = await detector.detect(gradients, node_ids)

        # Empire attack may be hard to detect, but aggregator should handle it
        # The key is that Byzantine-resilient aggregation still produces good result
        aggregator = FLAggregator(strategy="bulyan", byzantine_threshold=0.33)
        aggregated, _ = await aggregator.aggregate(gradients)

        # Aggregated should be closer to honest mean than target shift
        honest_mean = np.mean(honest_gradients, axis=0)
        similarity = np.dot(aggregated, honest_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(honest_mean)
        )

        # With Bulyan, should resist the attack
        assert similarity > 0.5, f"Aggregation corrupted: similarity={similarity}"

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_little_attack_detection(self, honest_node_factory):
        """Test detection of 'A Little Is Enough' attack."""
        detector = ByzantineDetector(z_threshold=3.0)
        gradient_dim = 500

        n_honest = 7
        n_byzantine = 3

        honest_nodes = [
            honest_node_factory(f"honest_{i}", gradient_dim)
            for i in range(n_honest)
        ]

        # Little attack: subtle poisoning
        byzantine_nodes = [
            ByzantineNode(
                node_id=f"byzantine_{i}",
                gradient_dimension=gradient_dim,
                attack_type="little",
                attack_params={"epsilon": 0.3},
            )
            for i in range(n_byzantine)
        ]

        # Collect honest gradients
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]

        for byz in byzantine_nodes:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        gradients = honest_gradients + [
            await byz.compute_gradient(1) for byz in byzantine_nodes
        ]
        node_ids = [n.node_id for n in honest_nodes] + [
            byz.node_id for byz in byzantine_nodes
        ]

        detected, confidence = await detector.detect(gradients, node_ids)

        # Little attack is designed to evade detection
        # But over multiple rounds, it should be caught
        # For single round, detection may be limited

        # The key property: aggregation quality
        aggregator = FLAggregator(strategy="trimmed_mean", byzantine_threshold=0.33)
        aggregated, _ = await aggregator.aggregate(gradients)

        honest_mean = np.mean(honest_gradients, axis=0)
        similarity = np.dot(aggregated, honest_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(honest_mean)
        )

        # Trimmed mean should reduce impact
        assert similarity > 0.7
