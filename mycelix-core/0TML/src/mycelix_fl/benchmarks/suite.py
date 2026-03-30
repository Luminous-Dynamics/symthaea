# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Benchmark Suite

Comprehensive performance benchmarking tools.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    name: str
    times_ms: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms))

    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms))

    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms))

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.times_ms, 50))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.times_ms, 99))

    def summary(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_ms:.2f}ms ± {self.std_ms:.2f}ms "
            f"(min={self.min_ms:.2f}, max={self.max_ms:.2f}, p95={self.p95_ms:.2f})"
        )


class BenchmarkSuite:
    """Benchmark suite for MycelixFL operations."""

    def __init__(
        self,
        warmup_rounds: int = 3,
        benchmark_rounds: int = 10,
        seed: int = 42,
    ):
        self.warmup_rounds = warmup_rounds
        self.benchmark_rounds = benchmark_rounds
        self.seed = seed
        self.results: List[BenchmarkResult] = []

    def _run_timed(
        self,
        func: Callable[[], Any],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run a function multiple times and measure execution time."""
        # Warmup
        for _ in range(self.warmup_rounds):
            func()
            gc.collect()

        # Benchmark
        times = []
        for _ in range(self.benchmark_rounds):
            gc.collect()
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        result = BenchmarkResult(
            name=name,
            times_ms=times,
            metadata=metadata or {},
        )
        self.results.append(result)
        return result

    def benchmark_fl_round(
        self,
        num_nodes: int = 10,
        gradient_size: int = 10000,
    ) -> BenchmarkResult:
        """Benchmark a single FL round."""
        from mycelix_fl import MycelixFL, FLConfig

        np.random.seed(self.seed)
        config = FLConfig(use_detection=True, use_healing=True)
        fl = MycelixFL(config=config)

        gradients = {
            f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_nodes)
        }

        round_num = [0]

        def run():
            round_num[0] += 1
            fl.execute_round(gradients, round_num=round_num[0])

        return self._run_timed(
            run,
            f"fl_round_{num_nodes}nodes_{gradient_size}params",
            {"num_nodes": num_nodes, "gradient_size": gradient_size},
        )

    def benchmark_hyperfeel_encode(
        self,
        gradient_size: int = 10000,
    ) -> BenchmarkResult:
        """Benchmark HyperFeel encoding."""
        from mycelix_fl import HyperFeelEncoderV2

        np.random.seed(self.seed)
        encoder = HyperFeelEncoderV2()
        gradient = np.random.randn(gradient_size).astype(np.float32)

        def run():
            encoder.encode(gradient)

        return self._run_timed(
            run,
            f"hyperfeel_encode_{gradient_size}params",
            {"gradient_size": gradient_size},
        )

    def benchmark_byzantine_detection(
        self,
        num_nodes: int = 10,
        gradient_size: int = 10000,
        byzantine_ratio: float = 0.3,
    ) -> BenchmarkResult:
        """Benchmark Byzantine detection."""
        from mycelix_fl import MultiLayerByzantineDetector

        np.random.seed(self.seed)
        detector = MultiLayerByzantineDetector()

        num_byzantine = int(num_nodes * byzantine_ratio)
        gradients = {}
        for i in range(num_nodes - num_byzantine):
            gradients[f"honest_{i}"] = np.random.randn(gradient_size).astype(np.float32)
        for i in range(num_byzantine):
            gradients[f"byzantine_{i}"] = np.random.randn(gradient_size).astype(np.float32) * 100

        def run():
            detector.detect(gradients)

        return self._run_timed(
            run,
            f"byzantine_detection_{num_nodes}nodes_{int(byzantine_ratio * 100)}pct",
            {"num_nodes": num_nodes, "byzantine_ratio": byzantine_ratio},
        )

    def benchmark_shapley_computation(
        self,
        num_nodes: int = 10,
        gradient_size: int = 10000,
    ) -> BenchmarkResult:
        """Benchmark Shapley value computation."""
        from mycelix_fl import ShapleyByzantineDetector

        np.random.seed(self.seed)
        detector = ShapleyByzantineDetector()

        gradients = {
            f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_nodes)
        }

        def run():
            detector.compute_shapley_values(gradients)

        return self._run_timed(
            run,
            f"shapley_computation_{num_nodes}nodes",
            {"num_nodes": num_nodes, "gradient_size": gradient_size},
        )

    def print_results(self) -> None:
        """Print all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        for result in self.results:
            print(f"\n{result}")
        print("\n" + "=" * 60)


def run_standard_benchmarks(
    warmup: int = 3,
    rounds: int = 10,
) -> List[BenchmarkResult]:
    """Run standard benchmark suite."""
    suite = BenchmarkSuite(warmup_rounds=warmup, benchmark_rounds=rounds)

    print("Running standard benchmarks...")

    # FL round benchmarks
    for nodes in [5, 10, 20]:
        result = suite.benchmark_fl_round(num_nodes=nodes, gradient_size=10000)
        print(f"  {result}")

    # HyperFeel benchmarks
    for size in [1000, 10000, 100000]:
        result = suite.benchmark_hyperfeel_encode(gradient_size=size)
        print(f"  {result}")

    # Byzantine detection
    for ratio in [0.1, 0.3, 0.45]:
        result = suite.benchmark_byzantine_detection(byzantine_ratio=ratio)
        print(f"  {result}")

    suite.print_results()
    return suite.results


def run_scalability_test(
    max_nodes: int = 100,
    step: int = 10,
    gradient_size: int = 10000,
) -> List[Tuple[int, float]]:
    """Test scalability with increasing node counts."""
    from mycelix_fl import MycelixFL, FLConfig

    print(f"Running scalability test (max {max_nodes} nodes)...")

    config = FLConfig(use_detection=True, use_healing=True)
    fl = MycelixFL(config=config)

    results = []
    for num_nodes in range(step, max_nodes + 1, step):
        np.random.seed(42)
        gradients = {
            f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_nodes)
        }

        # Warmup
        fl.execute_round(gradients, round_num=1)

        # Measure
        times = []
        for i in range(5):
            start = time.perf_counter()
            fl.execute_round(gradients, round_num=i + 2)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        results.append((num_nodes, avg_time))
        print(f"  {num_nodes} nodes: {avg_time:.2f}ms")

    return results


def run_byzantine_detection_benchmark(
    num_nodes: int = 20,
    gradient_size: int = 10000,
) -> Dict[str, Dict[str, float]]:
    """Benchmark Byzantine detection at various adversarial ratios."""
    from mycelix_fl import MycelixFL, FLConfig
    from mycelix_fl.attacks import (
        AttackOrchestrator,
        create_mixed_attack_scenario,
    )

    print("Running Byzantine detection benchmark...")

    results = {}
    for byzantine_ratio in [0.1, 0.2, 0.3, 0.4, 0.45]:
        num_attackers = int(num_nodes * byzantine_ratio)
        num_honest = num_nodes - num_attackers

        np.random.seed(42)
        attacks = create_mixed_attack_scenario(
            total_attackers=num_attackers,
            seed=42,
        )

        orchestrator = AttackOrchestrator(random_seed=42)
        for node_id, attack in attacks:
            orchestrator.register_attacker(node_id, attack)

        gradients = {
            f"honest_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_honest)
        }
        for node_id, _ in attacks:
            gradients[node_id] = np.random.randn(gradient_size).astype(np.float32)

        modified_grads, attacker_ids = orchestrator.apply_attacks(gradients)

        config = FLConfig(byzantine_threshold=0.45, use_detection=True)
        fl = MycelixFL(config=config)

        result = fl.execute_round(modified_grads, round_num=1)

        detected = result.byzantine_nodes
        true_positives = len(detected & attacker_ids)
        false_positives = len(detected - attacker_ids)
        false_negatives = len(attacker_ids - detected)

        precision = true_positives / len(detected) if detected else 0
        recall = true_positives / len(attacker_ids) if attacker_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[f"{int(byzantine_ratio * 100)}%"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        print(f"  {int(byzantine_ratio * 100)}% Byzantine: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

    return results


def compare_backends() -> Dict[str, Dict[str, float]]:
    """Compare Python vs Rust backend performance."""
    from mycelix_fl import has_rust_backend

    print("Comparing backends...")
    print(f"Rust backend available: {has_rust_backend()}")

    suite = BenchmarkSuite(warmup_rounds=2, benchmark_rounds=5)

    results = {
        "current_backend": "rust" if has_rust_backend() else "python",
        "benchmarks": {},
    }

    for name, func in [
        ("fl_round_10", lambda: suite.benchmark_fl_round(10, 10000)),
        ("fl_round_50", lambda: suite.benchmark_fl_round(50, 10000)),
        ("hyperfeel_10k", lambda: suite.benchmark_hyperfeel_encode(10000)),
        ("hyperfeel_100k", lambda: suite.benchmark_hyperfeel_encode(100000)),
    ]:
        result = func()
        results["benchmarks"][name] = result.summary()
        print(f"  {result}")

    return results
