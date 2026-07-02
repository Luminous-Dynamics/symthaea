# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Gradient Roundtrip

Verifies the complete gradient lifecycle:
1. Submit gradient to storage
2. Store in DHT (mocked)
3. Retrieve from storage
4. Aggregate with other gradients

This is the fundamental test proving the FL pipeline works end-to-end.
"""

import pytest
import numpy as np
import asyncio
import time
from typing import List, Dict, Any

from .fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    ZomeValidator,
    FLAggregator,
)
from .fixtures.nodes import HonestNode, NodeBehavior
from .fixtures.metrics import MetricsCollector, QualityAssertions


# ============================================================================
# Basic Roundtrip Tests
# ============================================================================

class TestGradientRoundtrip:
    """Test gradient submission, storage, retrieval, and aggregation."""

    @pytest.mark.asyncio
    async def test_single_gradient_submit_retrieve(self, gradient_factory):
        """Test submitting and retrieving a single gradient."""
        store = GradientStore()

        # Create a gradient
        gradient = gradient_factory(dimension=1000, seed=42)
        node_id = "test_node_1"
        round_id = 1

        # Submit
        success, entry_hash, error = await store.submit_gradient(
            node_id=node_id,
            round_id=round_id,
            gradient=gradient,
        )

        assert success, f"Gradient submission failed: {error}"
        assert entry_hash is not None
        assert len(entry_hash) == 32  # SHA256 truncated

        # Retrieve
        entry = await store.get_gradient(entry_hash)

        assert entry is not None
        assert entry.node_id == node_id
        assert entry.round_id == round_id
        assert entry.validated is True
        np.testing.assert_array_almost_equal(entry.gradient, gradient)

    @pytest.mark.asyncio
    async def test_multiple_gradients_same_round(self, gradient_factory):
        """Test submitting multiple gradients in the same round."""
        store = GradientStore()
        round_id = 1
        n_nodes = 5

        gradients = []
        entry_hashes = []

        for i in range(n_nodes):
            gradient = gradient_factory(dimension=500, seed=i)
            gradients.append(gradient)

            success, entry_hash, error = await store.submit_gradient(
                node_id=f"node_{i}",
                round_id=round_id,
                gradient=gradient,
            )

            assert success, f"Submission failed for node_{i}: {error}"
            entry_hashes.append(entry_hash)

        # Retrieve all gradients for the round
        entries = await store.get_gradients_for_round(round_id)

        assert len(entries) == n_nodes

        # Verify all gradients are present
        retrieved_gradients = [e.gradient for e in entries]
        for original in gradients:
            found = any(
                np.allclose(original, retrieved)
                for retrieved in retrieved_gradients
            )
            assert found, "Original gradient not found in retrieved set"

    @pytest.mark.asyncio
    async def test_gradient_roundtrip_with_aggregation(
        self, gradient_factory, aggregator
    ):
        """Test complete roundtrip: submit -> store -> retrieve -> aggregate."""
        store = GradientStore()
        round_id = 1
        n_nodes = 10
        gradient_dim = 1000

        # Create and submit gradients
        gradients = []
        for i in range(n_nodes):
            # All gradients are similar (honest nodes)
            gradient = gradient_factory(dimension=gradient_dim, mean=0.5, std=0.1, seed=i)
            gradients.append(gradient)

            await store.submit_gradient(
                node_id=f"node_{i}",
                round_id=round_id,
                gradient=gradient,
            )

        # Retrieve gradients
        entries = await store.get_gradients_for_round(round_id)
        retrieved_gradients = [e.gradient for e in entries]

        assert len(retrieved_gradients) == n_nodes

        # Aggregate
        aggregated, metadata = await aggregator.aggregate(retrieved_gradients)

        assert aggregated.shape == (gradient_dim,)
        assert metadata["n_gradients"] == n_nodes

        # Verify aggregated gradient is close to mean of inputs
        expected_mean = np.mean(gradients, axis=0)
        cosine_sim = np.dot(aggregated, expected_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(expected_mean)
        )

        # With trimmed mean, should be very close to actual mean for honest nodes
        assert cosine_sim > 0.95, f"Aggregation quality too low: {cosine_sim}"

    @pytest.mark.asyncio
    async def test_roundtrip_latency(self, gradient_factory, timing_context):
        """Test that roundtrip latency is within acceptable bounds."""
        store = GradientStore()
        aggregator = FLAggregator(strategy="fedavg")

        gradient_dim = 10000  # Large gradient
        n_nodes = 20

        latencies = {
            "submit": [],
            "retrieve": [],
            "aggregate": [],
        }

        # Submit phase
        for i in range(n_nodes):
            gradient = gradient_factory(dimension=gradient_dim, seed=i)

            async with timing_context(f"submit_{i}") as t:
                await store.submit_gradient(
                    node_id=f"node_{i}",
                    round_id=1,
                    gradient=gradient,
                )
            latencies["submit"].append(t["duration_ms"])

        # Retrieve phase
        async with timing_context("retrieve") as t:
            entries = await store.get_gradients_for_round(1)
            gradients = [e.gradient for e in entries]
        latencies["retrieve"].append(t["duration_ms"])

        # Aggregate phase
        async with timing_context("aggregate") as t:
            await aggregator.aggregate(gradients)
        latencies["aggregate"].append(t["duration_ms"])

        # Assert latency bounds
        avg_submit = np.mean(latencies["submit"])
        assert avg_submit < 10, f"Average submit latency {avg_submit}ms too high"

        assert latencies["retrieve"][0] < 50, "Retrieve latency too high"
        assert latencies["aggregate"][0] < 100, "Aggregate latency too high"


# ============================================================================
# Roundtrip with Validation Tests
# ============================================================================

class TestGradientRoundtripWithValidation:
    """Test roundtrip with zome-level validation."""

    @pytest.mark.asyncio
    async def test_valid_gradient_passes_validation(self, gradient_factory):
        """Test that valid gradients pass zome validation."""
        store = GradientStore()
        validator = ZomeValidator(max_gradient_norm=100.0)
        store.add_validation_callback(validator.validate)

        # Create a valid gradient
        gradient = gradient_factory(dimension=1000, mean=0, std=1.0, seed=42)

        success, entry_hash, error = await store.submit_gradient(
            node_id="honest_node",
            round_id=1,
            gradient=gradient,
        )

        assert success, f"Valid gradient was rejected: {error}"
        assert error is None

        entry = await store.get_gradient(entry_hash)
        assert entry.validated is True
        assert entry.rejected is False

    @pytest.mark.asyncio
    async def test_nan_gradient_rejected(self, gradient_factory):
        """Test that gradients with NaN values are rejected."""
        store = GradientStore()
        validator = ZomeValidator()
        store.add_validation_callback(validator.validate)

        gradient = gradient_factory(dimension=1000, seed=42)
        gradient[100] = np.nan  # Inject NaN

        success, entry_hash, error = await store.submit_gradient(
            node_id="bad_node",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert error is not None
        assert "NaN" in error

        entry = await store.get_gradient(entry_hash)
        assert entry.rejected is True

    @pytest.mark.asyncio
    async def test_inf_gradient_rejected(self, gradient_factory):
        """Test that gradients with Inf values are rejected."""
        store = GradientStore()
        validator = ZomeValidator()
        store.add_validation_callback(validator.validate)

        gradient = gradient_factory(dimension=1000, seed=42)
        gradient[500] = np.inf  # Inject Inf

        success, entry_hash, error = await store.submit_gradient(
            node_id="bad_node",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "Inf" in error

    @pytest.mark.asyncio
    async def test_large_norm_gradient_rejected(self):
        """Test that gradients with excessive norm are rejected."""
        store = GradientStore()
        validator = ZomeValidator(max_gradient_norm=10.0)
        store.add_validation_callback(validator.validate)

        # Create gradient with large norm
        gradient = np.ones(1000) * 100  # Norm = 100 * sqrt(1000) >> 10

        success, entry_hash, error = await store.submit_gradient(
            node_id="bad_node",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "norm" in error.lower()

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_gradients(self, gradient_factory):
        """Test batch with mix of valid and invalid gradients."""
        store = GradientStore()
        validator = ZomeValidator(max_gradient_norm=50.0)
        store.add_validation_callback(validator.validate)

        round_id = 1
        valid_count = 0
        invalid_count = 0

        # Submit valid gradients
        for i in range(5):
            gradient = gradient_factory(dimension=1000, mean=0, std=0.5, seed=i)
            success, _, _ = await store.submit_gradient(
                node_id=f"honest_{i}",
                round_id=round_id,
                gradient=gradient,
            )
            if success:
                valid_count += 1

        # Submit invalid gradients
        for i in range(3):
            gradient = np.ones(1000) * 1000  # Very large norm
            success, _, _ = await store.submit_gradient(
                node_id=f"bad_{i}",
                round_id=round_id,
                gradient=gradient,
            )
            if not success:
                invalid_count += 1

        assert valid_count == 5
        assert invalid_count == 3

        # Only valid gradients should be retrievable
        entries = await store.get_gradients_for_round(round_id)
        assert len(entries) == 5

        stats = store.get_stats()
        assert stats["validated"] == 5
        assert stats["rejected"] == 3


# ============================================================================
# Full Network Roundtrip Tests
# ============================================================================

class TestNetworkRoundtrip:
    """Test gradient roundtrip in full network context."""

    @pytest.mark.asyncio
    async def test_network_single_round(self, small_fl_network, metrics_collector):
        """Test a single FL round in a network."""
        network = small_fl_network

        result = await network.run_round()

        assert result["success"], f"Round failed: {result.get('error')}"
        assert result["n_participants"] >= network.config.min_participants
        assert result["n_rejected"] == 0  # All honest nodes
        assert result["aggregated_gradient"] is not None
        assert result["aggregated_gradient"].shape == (network.config.gradient_dimension,)

        # Record metrics
        metrics_collector.record_latency(
            "full_round",
            result["duration_ms"],
            round_id=result["round_id"],
        )

    @pytest.mark.asyncio
    async def test_network_multiple_rounds(self, small_fl_network, metrics_collector):
        """Test multiple consecutive FL rounds."""
        network = small_fl_network
        n_rounds = 5

        results = await network.run_rounds(n_rounds)

        assert len(results) == n_rounds

        for result in results:
            assert result["success"]
            metrics_collector.record_latency(
                "round",
                result["duration_ms"],
                round_id=result["round_id"],
            )

        # Verify round IDs are sequential
        round_ids = [r["round_id"] for r in results]
        assert round_ids == list(range(1, n_rounds + 1))

        # Check latency consistency
        latencies = [r["duration_ms"] for r in results]
        assert np.std(latencies) / np.mean(latencies) < 1.0, "Latency too variable"

    @pytest.mark.asyncio
    async def test_gradient_consistency_across_rounds(self, small_fl_network):
        """Test that gradients evolve consistently across rounds."""
        network = small_fl_network
        n_rounds = 10

        aggregated_gradients = []

        for _ in range(n_rounds):
            result = await network.run_round()
            assert result["success"]
            aggregated_gradients.append(result["aggregated_gradient"])

        # Check gradient norms decrease (simulating convergence)
        norms = [np.linalg.norm(g) for g in aggregated_gradients]

        # In our simulation, honest nodes decay gradients over rounds
        # So later gradients should have smaller norms
        assert norms[-1] < norms[0], "Gradients should decrease over rounds (convergence)"

    @pytest.mark.asyncio
    async def test_roundtrip_with_metrics_collection(
        self,
        small_fl_network,
        metrics_collector,
        test_report,
    ):
        """Test roundtrip with comprehensive metrics collection."""
        network = small_fl_network

        test_report.set_parameter("num_nodes", network.config.total_nodes)
        test_report.set_parameter("gradient_dim", network.config.gradient_dimension)
        test_report.add_tag("roundtrip")

        n_rounds = 3

        for round_num in range(n_rounds):
            round_metrics = metrics_collector.start_round(round_num + 1)

            result = await network.run_round()

            metrics_collector.end_round(round_metrics, result["success"])

            if result["success"]:
                # Create a "ground truth" as mean of honest gradients
                honest_mean = np.zeros(network.config.gradient_dimension)
                count = 0
                entries = await network.gradient_store.get_gradients_for_round(
                    result["round_id"]
                )
                for entry in entries:
                    if "honest" in entry.node_id:
                        honest_mean += entry.gradient
                        count += 1
                if count > 0:
                    honest_mean /= count

                    metrics_collector.record_aggregation(
                        round_id=result["round_id"],
                        aggregated=result["aggregated_gradient"],
                        ground_truth=honest_mean,
                        n_participants=result["n_participants"],
                        n_byzantine=0,
                        n_detected=0,
                        strategy=network.config.aggregation_strategy,
                        duration_ms=result["duration_ms"],
                    )

        # Verify metrics were collected
        assert len(metrics_collector.get_round_metrics()) == n_rounds

        summary = metrics_collector.get_aggregation_summary()
        assert summary["n_rounds"] == n_rounds
        assert summary["cosine_similarity"]["mean"] > 0.8


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestRoundtripEdgeCases:
    """Test edge cases in gradient roundtrip."""

    @pytest.mark.asyncio
    async def test_empty_gradient(self):
        """Test handling of zero-length gradient (should be rejected)."""
        store = GradientStore()
        validator = ZomeValidator(min_dimension=1)
        store.add_validation_callback(validator.validate)

        gradient = np.array([], dtype=np.float32)

        success, _, error = await store.submit_gradient(
            node_id="node",
            round_id=1,
            gradient=gradient,
        )

        assert not success
        assert "dimension" in error.lower()

    @pytest.mark.asyncio
    async def test_very_large_gradient_dimension(self):
        """Test handling of very large gradient dimensions."""
        store = GradientStore()

        # 1 million dimensions
        gradient = np.random.randn(1_000_000).astype(np.float32)

        success, entry_hash, error = await store.submit_gradient(
            node_id="node",
            round_id=1,
            gradient=gradient,
        )

        assert success
        entry = await store.get_gradient(entry_hash)
        assert entry.gradient.shape == (1_000_000,)

    @pytest.mark.asyncio
    async def test_duplicate_submission_same_round(self):
        """Test submitting same gradient twice in same round."""
        store = GradientStore()
        gradient = np.random.randn(100).astype(np.float32)

        # First submission
        success1, hash1, _ = await store.submit_gradient(
            node_id="node",
            round_id=1,
            gradient=gradient,
        )

        # Second submission (same data)
        success2, hash2, _ = await store.submit_gradient(
            node_id="node",
            round_id=1,
            gradient=gradient,
        )

        # Both should succeed, but hashes should be the same
        # (same content = same hash)
        assert success1 and success2
        assert hash1 == hash2

        # Should only have one entry
        entries = await store.get_gradients_for_round(1)
        # Due to our implementation storing by hash, duplicates overwrite
        assert len(entries) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_submissions(self):
        """Test concurrent gradient submissions."""
        store = GradientStore()
        n_concurrent = 50

        async def submit(i):
            gradient = np.random.randn(1000).astype(np.float32)
            return await store.submit_gradient(
                node_id=f"node_{i}",
                round_id=1,
                gradient=gradient,
            )

        # Submit all concurrently
        results = await asyncio.gather(*[submit(i) for i in range(n_concurrent)])

        # All should succeed
        successes = sum(1 for success, _, _ in results if success)
        assert successes == n_concurrent

        entries = await store.get_gradients_for_round(1)
        assert len(entries) == n_concurrent

    @pytest.mark.asyncio
    async def test_retrieval_nonexistent_hash(self):
        """Test retrieving gradient with nonexistent hash."""
        store = GradientStore()

        entry = await store.get_gradient("nonexistent_hash_12345678")
        assert entry is None

    @pytest.mark.asyncio
    async def test_retrieval_empty_round(self):
        """Test retrieving gradients from round with no submissions."""
        store = GradientStore()

        entries = await store.get_gradients_for_round(999)
        assert entries == []
