#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Throughput Benchmark Suite
===========================

Measures throughput for PoGQ operations:
1. Transactions per second for reputation updates
2. Concurrent node capacity

Author: Luminous Dynamics
Date: January 8, 2026
"""

import hashlib
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@dataclass
class ThroughputResult:
    """Result of a throughput measurement."""
    concurrent_clients: int
    total_operations: int
    duration_seconds: float
    latencies: List[float] = field(default_factory=list)

    @property
    def tps_mean(self) -> float:
        """Transactions per second."""
        return self.total_operations / max(self.duration_seconds, 1e-6)

    @property
    def tps_max(self) -> float:
        """Peak TPS (based on minimum latency)."""
        if not self.latencies:
            return 0.0
        min_latency = min(self.latencies)
        return self.concurrent_clients / max(min_latency, 1e-6)

    @property
    def latency_p50_ms(self) -> float:
        """Median latency in ms."""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 50) * 1000

    @property
    def latency_p99_ms(self) -> float:
        """99th percentile latency in ms."""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 99) * 1000

    def to_dict(self) -> Dict:
        return {
            "concurrent_clients": self.concurrent_clients,
            "total_operations": self.total_operations,
            "duration_seconds": self.duration_seconds,
            "tps_mean": self.tps_mean,
            "tps_max": self.tps_max,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


class ReputationSystem:
    """
    Simulated reputation update system for throughput testing.

    Models the reputation update operations that would occur
    in a PoGQ-based federated learning system.
    """

    def __init__(self):
        self.reputations: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.update_count = 0

    def update_reputation(
        self,
        node_id: str,
        quality_score: float,
        is_byzantine: bool,
    ) -> float:
        """
        Update reputation for a node.

        Args:
            node_id: Node identifier
            quality_score: Quality score from gradient evaluation
            is_byzantine: Whether node was detected as Byzantine

        Returns:
            New reputation score
        """
        with self.lock:
            current = self.reputations.get(node_id, 1.0)

            if is_byzantine:
                # Exponential decay for Byzantine behavior
                new_rep = current * 0.5
            else:
                # Gradual recovery based on quality
                delta = quality_score * 0.1
                new_rep = min(1.0, current * 0.9 + delta)

            self.reputations[node_id] = new_rep
            self.update_count += 1

            return new_rep

    def get_reputation(self, node_id: str) -> float:
        """Get current reputation for a node."""
        with self.lock:
            return self.reputations.get(node_id, 1.0)

    def compute_proof(self, node_id: str, reputation: float) -> str:
        """Compute a proof of reputation update."""
        data = f"{node_id}:{reputation:.6f}:{time.time()}".encode()
        return hashlib.sha3_256(data).hexdigest()


class GradientSubmissionSystem:
    """
    Simulated gradient submission system for throughput testing.
    """

    def __init__(self, gradient_dim: int = 1000):
        self.gradient_dim = gradient_dim
        self.submissions: queue.Queue = queue.Queue()
        self.processed_count = 0
        self.lock = threading.Lock()

    def submit_gradient(self, node_id: str, gradient: np.ndarray) -> Dict:
        """
        Submit a gradient for processing.

        Returns:
            Submission receipt
        """
        # Compute gradient hash
        grad_hash = hashlib.sha256(gradient.tobytes()).hexdigest()

        # Simulate processing
        submission = {
            "node_id": node_id,
            "hash": grad_hash,
            "timestamp": time.time(),
            "size": gradient.nbytes,
        }

        self.submissions.put(submission)

        with self.lock:
            self.processed_count += 1

        return submission

    def generate_random_gradient(self) -> np.ndarray:
        """Generate a random gradient for testing."""
        return np.random.randn(self.gradient_dim).astype(np.float32)


class ThroughputBenchmark:
    """
    Throughput benchmark suite for PoGQ operations.

    Measures:
    1. Reputation update TPS
    2. Gradient submission TPS
    3. Concurrent node capacity
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
    ):
        """
        Initialize benchmark.

        Args:
            seed: Random seed
            quick_mode: Run shorter tests
        """
        self.seed = seed
        self.quick_mode = quick_mode

        np.random.seed(seed)

        # Benchmark configuration
        if quick_mode:
            self.test_duration = 2.0  # seconds
            self.concurrent_levels = [1, 5, 10, 20, 50]
            self.gradient_dim = 1000
        else:
            self.test_duration = 5.0  # seconds
            self.concurrent_levels = [1, 5, 10, 20, 50, 100, 200]
            self.gradient_dim = 10000

    def benchmark_reputation_updates(self) -> List[Dict]:
        """
        Benchmark reputation update throughput.

        Returns:
            List of results for each concurrency level
        """
        results = []

        for num_clients in self.concurrent_levels:
            print(f"    Testing {num_clients} concurrent clients...")

            rep_system = ReputationSystem()
            latencies = []
            operations = 0
            errors = 0

            start_time = time.time()
            end_time = start_time + self.test_duration

            def worker(client_id: int):
                nonlocal operations, errors
                local_latencies = []

                while time.time() < end_time:
                    node_id = f"node_{client_id}_{np.random.randint(0, 1000)}"
                    quality = np.random.uniform(0.5, 1.0)
                    is_byz = np.random.random() < 0.1

                    try:
                        op_start = time.perf_counter()
                        rep_system.update_reputation(node_id, quality, is_byz)
                        op_end = time.perf_counter()

                        local_latencies.append(op_end - op_start)
                        operations += 1
                    except Exception:
                        errors += 1

                return local_latencies

            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(worker, i) for i in range(num_clients)]

                for future in as_completed(futures):
                    latencies.extend(future.result())

            actual_duration = time.time() - start_time

            result = ThroughputResult(
                concurrent_clients=num_clients,
                total_operations=operations,
                duration_seconds=actual_duration,
                latencies=latencies,
            )

            results.append(result.to_dict())

        return results

    def benchmark_gradient_submissions(self) -> List[Dict]:
        """
        Benchmark gradient submission throughput.

        Returns:
            List of results for each concurrency level
        """
        results = []

        for num_clients in self.concurrent_levels:
            print(f"    Testing {num_clients} concurrent gradient submissions...")

            grad_system = GradientSubmissionSystem(self.gradient_dim)
            latencies = []
            operations = 0

            start_time = time.time()
            end_time = start_time + self.test_duration

            def worker(client_id: int):
                nonlocal operations
                local_latencies = []

                while time.time() < end_time:
                    node_id = f"node_{client_id}"
                    gradient = grad_system.generate_random_gradient()

                    op_start = time.perf_counter()
                    grad_system.submit_gradient(node_id, gradient)
                    op_end = time.perf_counter()

                    local_latencies.append(op_end - op_start)
                    operations += 1

                return local_latencies

            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(worker, i) for i in range(num_clients)]

                for future in as_completed(futures):
                    latencies.extend(future.result())

            actual_duration = time.time() - start_time

            result = ThroughputResult(
                concurrent_clients=num_clients,
                total_operations=operations,
                duration_seconds=actual_duration,
                latencies=latencies,
            )

            results.append(result.to_dict())

        return results

    def benchmark_mixed_workload(self) -> List[Dict]:
        """
        Benchmark mixed workload (reputation + gradients).

        Simulates realistic FL round with both operations.
        """
        results = []

        for num_clients in self.concurrent_levels:
            print(f"    Testing {num_clients} clients with mixed workload...")

            rep_system = ReputationSystem()
            grad_system = GradientSubmissionSystem(self.gradient_dim)
            latencies = []
            operations = 0

            start_time = time.time()
            end_time = start_time + self.test_duration

            def worker(client_id: int):
                nonlocal operations
                local_latencies = []

                while time.time() < end_time:
                    node_id = f"node_{client_id}"

                    op_start = time.perf_counter()

                    # Submit gradient
                    gradient = grad_system.generate_random_gradient()
                    grad_system.submit_gradient(node_id, gradient)

                    # Update reputation
                    quality = np.random.uniform(0.5, 1.0)
                    is_byz = np.random.random() < 0.1
                    rep_system.update_reputation(node_id, quality, is_byz)

                    op_end = time.perf_counter()

                    local_latencies.append(op_end - op_start)
                    operations += 2  # Count both operations

                return local_latencies

            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(worker, i) for i in range(num_clients)]

                for future in as_completed(futures):
                    latencies.extend(future.result())

            actual_duration = time.time() - start_time

            result = ThroughputResult(
                concurrent_clients=num_clients,
                total_operations=operations,
                duration_seconds=actual_duration,
                latencies=latencies,
            )

            results.append(result.to_dict())

        return results

    def benchmark_burst_load(self) -> Dict:
        """
        Benchmark burst load handling.

        Tests how system handles sudden spikes in traffic.
        """
        print("    Testing burst load handling...")

        rep_system = ReputationSystem()
        results = {
            "normal_tps": 0,
            "burst_tps": 0,
            "recovery_time_ms": 0,
        }

        # Normal load
        normal_ops = 0
        normal_start = time.time()
        normal_duration = 1.0

        while time.time() - normal_start < normal_duration:
            node_id = f"node_{np.random.randint(0, 100)}"
            rep_system.update_reputation(node_id, 0.8, False)
            normal_ops += 1

        results["normal_tps"] = normal_ops / normal_duration

        # Burst load (10x normal rate attempt)
        burst_ops = 0
        burst_latencies = []
        burst_start = time.time()
        burst_duration = 0.5

        def burst_worker(worker_id: int):
            ops = 0
            lats = []
            while time.time() - burst_start < burst_duration:
                op_start = time.perf_counter()
                node_id = f"burst_{worker_id}_{np.random.randint(0, 1000)}"
                rep_system.update_reputation(node_id, 0.8, False)
                lats.append(time.perf_counter() - op_start)
                ops += 1
            return ops, lats

        num_burst_workers = 20
        with ThreadPoolExecutor(max_workers=num_burst_workers) as executor:
            futures = [executor.submit(burst_worker, i) for i in range(num_burst_workers)]

            for future in as_completed(futures):
                ops, lats = future.result()
                burst_ops += ops
                burst_latencies.extend(lats)

        actual_burst_duration = time.time() - burst_start
        results["burst_tps"] = burst_ops / actual_burst_duration

        # Recovery (measure when latency returns to normal)
        if burst_latencies:
            normal_latency = 1.0 / results["normal_tps"] if results["normal_tps"] > 0 else 0.001
            burst_p99 = np.percentile(burst_latencies, 99)
            results["recovery_time_ms"] = (burst_p99 - normal_latency) * 1000

        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Run all throughput benchmarks.

        Returns:
            Complete benchmark results
        """
        results = {}

        print("  Running reputation update throughput benchmark...")
        results["reputation_updates"] = self.benchmark_reputation_updates()

        print("  Running gradient submission throughput benchmark...")
        results["gradient_submissions"] = self.benchmark_gradient_submissions()

        print("  Running mixed workload throughput benchmark...")
        results["mixed_workload"] = self.benchmark_mixed_workload()

        print("  Running burst load benchmark...")
        results["burst_load"] = self.benchmark_burst_load()

        return results


def main():
    """Run throughput benchmark standalone."""
    print("Throughput Benchmark Suite")
    print("=" * 50)

    benchmark = ThroughputBenchmark(
        seed=42,
        quick_mode=True,
    )

    results = benchmark.run_all()

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)

    print("\nReputation Update TPS:")
    for result in results["reputation_updates"]:
        print(f"  {result['concurrent_clients']} clients: "
              f"{result['tps_mean']:.0f} TPS, "
              f"p50={result['latency_p50_ms']:.2f}ms")

    print("\nBurst Load:")
    burst = results["burst_load"]
    print(f"  Normal TPS: {burst['normal_tps']:.0f}")
    print(f"  Burst TPS: {burst['burst_tps']:.0f}")
    print(f"  Recovery time: {burst['recovery_time_ms']:.2f}ms")


if __name__ == "__main__":
    main()
