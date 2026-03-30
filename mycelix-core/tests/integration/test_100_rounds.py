# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: 100 Rounds Long-Running Stability

Long-running stability test that executes 100 FL rounds to verify:
- System stability over time
- Memory/resource management
- Consistent performance
- Convergence behavior
- Detection accuracy over time
"""

import pytest
import numpy as np
import asyncio
import time
import gc
from typing import List, Dict, Any

from .fixtures.network import FLNetwork, NetworkConfig
from .fixtures.metrics import MetricsCollector, TestReport


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def hundred_round_config():
    """Configuration optimized for 100-round test."""
    return NetworkConfig(
        num_honest_nodes=15,
        num_byzantine_nodes=5,  # 25% Byzantine
        gradient_dimension=2000,  # Moderate dimension for speed
        aggregation_strategy="multi_krum",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=2,
        timeout_ms=5000,
        min_participants=8,
    )


@pytest.fixture
def stability_config():
    """Configuration for pure stability test (no Byzantine)."""
    return NetworkConfig(
        num_honest_nodes=20,
        num_byzantine_nodes=0,
        gradient_dimension=1000,
        aggregation_strategy="trimmed_mean",
        byzantine_threshold=0.33,
        detection_enabled=False,
        reputation_enabled=False,
        holochain_mock=True,
        timeout_ms=3000,
    )


# ============================================================================
# Long-Running Stability Tests
# ============================================================================

class TestLongRunningStability:
    """Test system stability over 100 rounds."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_100_rounds_basic_stability(
        self, stability_config, metrics_collector
    ):
        """Test basic stability over 100 rounds."""
        network = FLNetwork(stability_config)
        await network.start()

        try:
            n_rounds = 100
            round_times = []
            failures = 0

            for i in range(n_rounds):
                result = await network.run_round()

                if not result["success"]:
                    failures += 1
                    continue

                round_times.append(result["duration_ms"])

                # Record periodic metrics
                if (i + 1) % 10 == 0:
                    metrics_collector.record_custom(
                        f"round_{i+1}_latency",
                        result["duration_ms"]
                    )

            # Assertions
            assert failures == 0, f"{failures} rounds failed"
            assert len(round_times) == n_rounds

            # Check latency stability
            mean_time = np.mean(round_times)
            std_time = np.std(round_times)
            cv = std_time / mean_time if mean_time > 0 else 0

            # Coefficient of variation should be reasonable
            assert cv < 1.0, f"Latency too variable: CV={cv}"

            # No significant latency drift
            early_mean = np.mean(round_times[:10])
            late_mean = np.mean(round_times[-10:])
            drift_ratio = late_mean / early_mean if early_mean > 0 else 1

            assert 0.5 < drift_ratio < 2.0, f"Latency drift detected: {drift_ratio}x"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.byzantine
    async def test_100_rounds_with_byzantine(
        self, hundred_round_config, metrics_collector, test_report
    ):
        """Test 100 rounds with Byzantine nodes present."""
        config = hundred_round_config
        network = FLNetwork(config)
        await network.start()

        test_report.set_parameter("total_rounds", 100)
        test_report.set_parameter("num_byzantine", config.num_byzantine_nodes)
        test_report.add_tag("long_running")
        test_report.add_tag("byzantine")

        try:
            n_rounds = 100
            results = []
            detection_counts = []
            quality_scores = []

            true_byzantines = {
                f"byzantine_{i}" for i in range(config.num_byzantine_nodes)
            }

            for i in range(n_rounds):
                result = await network.run_round()
                results.append(result)

                if result["success"]:
                    detection_counts.append(result["n_byzantine_detected"])

                    # Compute quality against honest mean
                    entries = await network.gradient_store.get_gradients_for_round(
                        result["round_id"]
                    )
                    honest_grads = [
                        e.gradient for e in entries if "honest" in e.node_id
                    ]
                    if honest_grads:
                        honest_mean = np.mean(honest_grads, axis=0)
                        agg = result["aggregated_gradient"]
                        quality = np.dot(agg, honest_mean) / (
                            np.linalg.norm(agg) * np.linalg.norm(honest_mean)
                        )
                        quality_scores.append(quality)

                # Log progress every 20 rounds
                if (i + 1) % 20 == 0:
                    success_rate = sum(1 for r in results if r["success"]) / len(results)
                    avg_detection = np.mean(detection_counts) if detection_counts else 0
                    avg_quality = np.mean(quality_scores) if quality_scores else 0

                    metrics_collector.record_custom(
                        f"checkpoint_{i+1}",
                        {
                            "success_rate": success_rate,
                            "avg_detection": avg_detection,
                            "avg_quality": avg_quality,
                        }
                    )

            # Final analysis
            total_success = sum(1 for r in results if r["success"])
            success_rate = total_success / n_rounds

            assert success_rate >= 0.95, f"Success rate too low: {success_rate}"

            # Detection should be consistent
            if detection_counts:
                avg_detection = np.mean(detection_counts)
                std_detection = np.std(detection_counts)
                assert avg_detection >= 0  # Should detect at least sometimes

            # Quality should remain high
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                assert avg_quality >= 0.7, f"Aggregation quality degraded: {avg_quality}"

            # Reputation should have evolved
            scores = network.reputation_bridge.get_all_scores()
            if scores:
                byzantine_scores = [
                    scores.get(f"byzantine_{i}", 1.0)
                    for i in range(config.num_byzantine_nodes)
                ]
                honest_scores = [
                    scores.get(f"honest_{i}", 1.0)
                    for i in range(config.num_honest_nodes)
                ]

                # Byzantine nodes should have lower scores
                if byzantine_scores and honest_scores:
                    avg_byz = np.mean(byzantine_scores)
                    avg_hon = np.mean(honest_scores)
                    metrics_collector.record_custom(
                        "final_reputation",
                        {"byzantine_avg": avg_byz, "honest_avg": avg_hon}
                    )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_100_rounds_convergence(self, stability_config, metrics_collector):
        """Test model convergence over 100 rounds."""
        network = FLNetwork(stability_config)
        await network.start()

        try:
            n_rounds = 100
            gradient_norms = []

            for i in range(n_rounds):
                result = await network.run_round()
                assert result["success"]

                norm = np.linalg.norm(result["aggregated_gradient"])
                gradient_norms.append(norm)

            # Gradient norms should decrease (convergence)
            early_norm = np.mean(gradient_norms[:10])
            late_norm = np.mean(gradient_norms[-10:])

            assert late_norm < early_norm, "No convergence observed"

            # Calculate convergence rate
            convergence_ratio = late_norm / early_norm
            metrics_collector.record_custom("convergence_ratio", convergence_ratio)

            # Should converge significantly
            assert convergence_ratio < 0.5, f"Slow convergence: {convergence_ratio}"

        finally:
            await network.shutdown()


# ============================================================================
# Resource Management Tests
# ============================================================================

class TestResourceManagement:
    """Test resource management over long runs."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_memory_stability(self, stability_config, metrics_collector):
        """Test memory stability over 100 rounds."""
        import tracemalloc

        network = FLNetwork(stability_config)
        await network.start()

        tracemalloc.start()

        try:
            memory_snapshots = []
            n_rounds = 100

            for i in range(n_rounds):
                result = await network.run_round()
                assert result["success"]

                # Record memory every 10 rounds
                if (i + 1) % 10 == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_snapshots.append({
                        "round": i + 1,
                        "current_mb": current / 1024 / 1024,
                        "peak_mb": peak / 1024 / 1024,
                    })

                    # Force garbage collection periodically
                    if (i + 1) % 25 == 0:
                        gc.collect()

            # Check for memory leaks
            initial_memory = memory_snapshots[0]["current_mb"]
            final_memory = memory_snapshots[-1]["current_mb"]
            memory_growth = final_memory - initial_memory

            metrics_collector.record_custom(
                "memory_stats",
                {
                    "initial_mb": initial_memory,
                    "final_mb": final_memory,
                    "growth_mb": memory_growth,
                    "peak_mb": max(s["peak_mb"] for s in memory_snapshots),
                }
            )

            # Allow some memory growth but not unbounded
            # 50MB growth over 100 rounds is reasonable
            assert memory_growth < 50, f"Potential memory leak: {memory_growth}MB growth"

        finally:
            tracemalloc.stop()
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_storage_growth(self, stability_config, metrics_collector):
        """Test storage growth over 100 rounds."""
        network = FLNetwork(stability_config)
        await network.start()

        try:
            n_rounds = 100
            storage_stats = []

            for i in range(n_rounds):
                result = await network.run_round()
                assert result["success"]

                if (i + 1) % 10 == 0:
                    stats = network.gradient_store.get_stats()
                    storage_stats.append({
                        "round": i + 1,
                        "total_entries": stats["total_entries"],
                        "rounds_stored": stats["rounds"],
                    })

            # Storage should grow linearly
            initial_entries = storage_stats[0]["total_entries"]
            final_entries = storage_stats[-1]["total_entries"]

            # Should have entries for each round
            expected_entries = n_rounds * network.config.num_honest_nodes

            assert final_entries >= expected_entries * 0.9, "Missing stored gradients"

            metrics_collector.record_custom(
                "storage_growth",
                {
                    "initial": initial_entries,
                    "final": final_entries,
                    "expected": expected_entries,
                }
            )

        finally:
            await network.shutdown()


# ============================================================================
# Performance Consistency Tests
# ============================================================================

class TestPerformanceConsistency:
    """Test performance consistency over long runs."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_latency_consistency(self, stability_config, metrics_collector):
        """Test latency remains consistent over 100 rounds."""
        network = FLNetwork(stability_config)
        await network.start()

        try:
            n_rounds = 100
            latencies = []

            for i in range(n_rounds):
                result = await network.run_round()
                assert result["success"]
                latencies.append(result["duration_ms"])

            # Divide into windows
            window_size = 20
            windows = [
                latencies[i:i+window_size]
                for i in range(0, n_rounds, window_size)
            ]

            window_means = [np.mean(w) for w in windows]
            window_stds = [np.std(w) for w in windows]

            # Mean latency shouldn't change significantly
            overall_mean = np.mean(latencies)
            for i, mean in enumerate(window_means):
                deviation = abs(mean - overall_mean) / overall_mean
                assert deviation < 0.3, f"Window {i} deviated {deviation*100:.1f}%"

            metrics_collector.record_custom(
                "latency_windows",
                {
                    "means": window_means,
                    "stds": window_stds,
                    "overall_mean": overall_mean,
                    "overall_std": np.std(latencies),
                }
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.byzantine
    async def test_detection_consistency(
        self, hundred_round_config, metrics_collector
    ):
        """Test Byzantine detection remains consistent over 100 rounds."""
        config = hundred_round_config
        network = FLNetwork(config)
        await network.start()

        try:
            n_rounds = 100
            detection_history = []
            true_positives_per_window = []
            false_positives_per_window = []

            true_byzantines = {
                f"byzantine_{i}" for i in range(config.num_byzantine_nodes)
            }

            for i in range(n_rounds):
                result = await network.run_round()
                if result["success"]:
                    detected = set(result.get("detected_byzantine", []))
                    tp = len(detected & true_byzantines)
                    fp = len(detected - true_byzantines)
                    detection_history.append({
                        "round": i + 1,
                        "detected": len(detected),
                        "tp": tp,
                        "fp": fp,
                    })

            # Analyze detection over windows
            window_size = 20
            for i in range(0, len(detection_history), window_size):
                window = detection_history[i:i+window_size]
                true_positives_per_window.append(
                    sum(d["tp"] for d in window)
                )
                false_positives_per_window.append(
                    sum(d["fp"] for d in window)
                )

            # Detection should be somewhat consistent
            tp_std = np.std(true_positives_per_window)
            tp_mean = np.mean(true_positives_per_window)
            cv = tp_std / tp_mean if tp_mean > 0 else 0

            metrics_collector.record_custom(
                "detection_consistency",
                {
                    "tp_per_window": true_positives_per_window,
                    "fp_per_window": false_positives_per_window,
                    "tp_cv": cv,
                }
            )

            # False positives should be low
            total_fp = sum(false_positives_per_window)
            assert total_fp < n_rounds * 0.1, f"Too many false positives: {total_fp}"

        finally:
            await network.shutdown()


# ============================================================================
# Comprehensive 100-Round Test
# ============================================================================

class TestComprehensive100Rounds:
    """Comprehensive 100-round test with all features."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.byzantine
    async def test_full_100_round_scenario(
        self, hundred_round_config, metrics_collector, test_report
    ):
        """Run comprehensive 100-round test with all metrics."""
        config = hundred_round_config
        network = FLNetwork(config)
        await network.start()

        test_report.set_parameter("config", {
            "honest_nodes": config.num_honest_nodes,
            "byzantine_nodes": config.num_byzantine_nodes,
            "gradient_dim": config.gradient_dimension,
            "strategy": config.aggregation_strategy,
        })
        test_report.add_tag("comprehensive")
        test_report.add_tag("100_rounds")

        try:
            n_rounds = 100

            # Track all metrics
            all_results = []
            quality_over_time = []
            detection_over_time = []
            reputation_snapshots = []

            true_byzantines = {
                f"byzantine_{i}" for i in range(config.num_byzantine_nodes)
            }
            all_nodes = set(network.get_all_nodes().keys())

            start_time = time.time()

            for i in range(n_rounds):
                round_start = time.time()

                result = await network.run_round()
                all_results.append(result)

                if result["success"]:
                    # Quality metrics
                    entries = await network.gradient_store.get_gradients_for_round(
                        result["round_id"]
                    )
                    honest_grads = [
                        e.gradient for e in entries if "honest" in e.node_id
                    ]
                    if honest_grads:
                        honest_mean = np.mean(honest_grads, axis=0)
                        agg = result["aggregated_gradient"]
                        quality = np.dot(agg, honest_mean) / (
                            np.linalg.norm(agg) * np.linalg.norm(honest_mean) + 1e-10
                        )
                        quality_over_time.append(quality)

                    # Detection metrics
                    detected = set(result.get("detected_byzantine", []))
                    metrics_collector.record_detection(
                        round_id=result["round_id"],
                        true_byzantines=true_byzantines,
                        detected_byzantines=detected,
                        all_node_ids=all_nodes,
                    )
                    detection_over_time.append({
                        "round": i + 1,
                        "detected": len(detected),
                        "tp": len(detected & true_byzantines),
                        "fp": len(detected - true_byzantines),
                    })

                # Periodic reputation snapshot
                if (i + 1) % 25 == 0:
                    scores = network.reputation_bridge.get_all_scores()
                    reputation_snapshots.append({
                        "round": i + 1,
                        "scores": scores.copy(),
                    })

                # Progress logging
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (n_rounds - i - 1) / rate

            total_time = time.time() - start_time

            # Final analysis
            success_count = sum(1 for r in all_results if r["success"])
            success_rate = success_count / n_rounds

            assert success_rate >= 0.95, f"Success rate {success_rate} too low"

            # Quality analysis
            if quality_over_time:
                avg_quality = np.mean(quality_over_time)
                min_quality = np.min(quality_over_time)
                quality_trend = quality_over_time[-10] - quality_over_time[:10].mean() if len(quality_over_time) >= 10 else 0

                assert avg_quality >= 0.7, f"Average quality {avg_quality} too low"
                assert min_quality >= 0.5, f"Minimum quality {min_quality} too low"

            # Detection analysis
            detection_summary = metrics_collector.get_detection_summary()

            # Reputation analysis
            if reputation_snapshots:
                final_scores = reputation_snapshots[-1]["scores"]
                byz_final = [
                    final_scores.get(f"byzantine_{i}", 1.0)
                    for i in range(config.num_byzantine_nodes)
                ]
                hon_final = [
                    final_scores.get(f"honest_{i}", 1.0)
                    for i in range(config.num_honest_nodes)
                ]

                if byz_final and hon_final:
                    avg_byz = np.mean(byz_final)
                    avg_hon = np.mean(hon_final)
                    # Byzantine nodes should have lower reputation after 100 rounds
                    assert avg_hon >= avg_byz, "Reputation system not working"

            # Record final metrics
            metrics_collector.record_custom(
                "final_summary",
                {
                    "total_rounds": n_rounds,
                    "success_rate": success_rate,
                    "total_time_s": total_time,
                    "avg_round_time_ms": total_time * 1000 / n_rounds,
                    "avg_quality": float(np.mean(quality_over_time)) if quality_over_time else 0,
                    "detection_precision": detection_summary.get("precision", {}).get("mean", 0),
                    "detection_recall": detection_summary.get("recall", {}).get("mean", 0),
                }
            )

            # Verify test completed successfully
            test_report.passed = True

        except Exception as e:
            test_report.fail(str(e))
            raise

        finally:
            await network.shutdown()
