#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Scalability Benchmark Suite
============================

Tests PoGQ scalability across node counts:
- 10, 50, 100, 500, 1000 simulated nodes

Measures:
1. Detection accuracy at scale
2. Memory usage per node
3. Network bandwidth requirements
4. Latency scaling

Author: Luminous Dynamics
Date: January 8, 2026
"""

import gc
import hashlib
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@dataclass
class ScalabilityResult:
    """Result from a scalability test."""
    num_nodes: int
    method: str
    detection_rate: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    memory_mb: float
    bandwidth_kb: float
    gradient_dim: int


class MemoryProfiler:
    """Simple memory profiling utility."""

    def __init__(self):
        self.snapshots = []
        self._started = False

    def start(self):
        """Start memory tracking."""
        gc.collect()
        tracemalloc.start()
        self._started = True

    def snapshot(self) -> float:
        """Take memory snapshot and return current usage in MB."""
        if not self._started:
            return 0.0

        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)  # Convert to MB

    def stop(self) -> Tuple[float, float]:
        """Stop tracking and return (current_mb, peak_mb)."""
        if not self._started:
            return 0.0, 0.0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._started = False

        return current / (1024 * 1024), peak / (1024 * 1024)


class BandwidthEstimator:
    """Estimate network bandwidth requirements."""

    @staticmethod
    def estimate_gradient_size(gradient: np.ndarray) -> int:
        """Estimate serialized gradient size in bytes."""
        # Assume float32 serialization with minimal overhead
        return gradient.nbytes + 64  # 64 bytes for metadata

    @staticmethod
    def estimate_proof_size() -> int:
        """Estimate proof size in bytes."""
        # SHA3-256 hash: 32 bytes
        # Metadata (timestamp, nonce, etc.): 64 bytes
        # Signature placeholder: 64 bytes
        return 160

    @staticmethod
    def estimate_round_bandwidth(
        num_nodes: int,
        gradient_dim: int,
        include_proofs: bool = True,
    ) -> int:
        """
        Estimate total bandwidth for one FL round.

        Args:
            num_nodes: Number of participating nodes
            gradient_dim: Dimension of gradients
            include_proofs: Whether to include proof overhead

        Returns:
            Estimated bytes per round
        """
        gradient_size = gradient_dim * 4 + 64  # float32 + metadata

        # Each node sends: gradient + proof
        per_node = gradient_size
        if include_proofs:
            per_node += BandwidthEstimator.estimate_proof_size()

        # Aggregated gradient sent back to all nodes
        aggregated_broadcast = gradient_size * num_nodes

        total = (per_node * num_nodes) + aggregated_broadcast

        return total


class ScalablePoGQ:
    """
    Scalable PoGQ implementation for benchmarking.

    Optimized for large node counts with:
    - Batch processing
    - Early rejection
    - Memory-efficient aggregation
    """

    def __init__(
        self,
        quality_threshold: float = 0.35,
        batch_size: int = 100,
    ):
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size

    def detect_and_aggregate(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Set[str]]:
        """
        Detect Byzantine nodes and aggregate honest gradients.

        Optimized for scalability with batch processing.
        """
        node_ids = list(gradients.keys())
        all_grads = [gradients[nid] for nid in node_ids]
        n = len(all_grads)

        if n == 0:
            raise ValueError("No gradients provided")

        # Compute reference gradient (sample for large N)
        if n > 100:
            sample_indices = np.random.choice(n, 100, replace=False)
            sample_grads = [all_grads[i] for i in sample_indices]
            reference = np.mean(sample_grads, axis=0)
        else:
            reference = np.mean(all_grads, axis=0)

        ref_norm = np.linalg.norm(reference) + 1e-8

        # Compute quality scores in batches
        detected = set()
        valid_indices = []
        quality_scores = []

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)

            for i in range(batch_start, batch_end):
                grad = all_grads[i]
                grad_norm = np.linalg.norm(grad) + 1e-8

                # Cosine similarity
                cosine = np.dot(grad, reference) / (grad_norm * ref_norm)
                quality = (cosine + 1) / 2

                # Magnitude check
                mag_ratio = grad_norm / ref_norm
                if mag_ratio > 3 or mag_ratio < 0.1:
                    quality *= 0.5

                if quality < self.quality_threshold:
                    detected.add(node_ids[i])
                else:
                    valid_indices.append(i)
                    quality_scores.append(quality)

        # Aggregate valid gradients
        if len(valid_indices) > 0:
            weights = np.array(quality_scores)
            weights /= weights.sum()

            aggregated = np.zeros_like(all_grads[0])
            for w, idx in zip(weights, valid_indices):
                aggregated += w * all_grads[idx]
        else:
            aggregated = np.median(all_grads, axis=0)

        return aggregated, detected


class ScalabilityBenchmark:
    """
    Scalability benchmark suite for PoGQ.

    Tests with 10, 50, 100, 500, 1000 nodes measuring:
    - Detection accuracy
    - Memory usage
    - Network bandwidth
    - Latency
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
            quick_mode: Run smaller tests
        """
        self.seed = seed
        self.quick_mode = quick_mode

        np.random.seed(seed)

        # Benchmark configuration
        if quick_mode:
            self.node_counts = [10, 50, 100, 200]
            self.gradient_dim = 5000
            self.num_trials = 3
            self.byzantine_ratio = 0.20
        else:
            self.node_counts = [10, 50, 100, 500, 1000]
            self.gradient_dim = 10000
            self.num_trials = 5
            self.byzantine_ratio = 0.20

    def generate_gradients(
        self,
        num_nodes: int,
        byzantine_ratio: float,
    ) -> Tuple[Dict[str, np.ndarray], Set[str], np.ndarray]:
        """Generate gradients with Byzantine nodes."""
        num_byzantine = int(num_nodes * byzantine_ratio)
        num_honest = num_nodes - num_byzantine

        # True gradient direction
        true_gradient = np.random.randn(self.gradient_dim).astype(np.float32)
        true_gradient /= (np.linalg.norm(true_gradient) + 1e-8)

        gradients = {}
        actual_byzantine = set()

        # Honest gradients
        for i in range(num_honest):
            node_id = f"h_{i}"
            noise = np.random.randn(self.gradient_dim).astype(np.float32) * 0.1
            gradients[node_id] = true_gradient + noise

        # Byzantine gradients (mixed attacks)
        for i in range(num_byzantine):
            node_id = f"b_{i}"
            actual_byzantine.add(node_id)

            attack_type = i % 3
            if attack_type == 0:
                # Sign flip
                gradients[node_id] = -true_gradient * np.random.uniform(0.5, 1.5)
            elif attack_type == 1:
                # Scaling
                gradients[node_id] = true_gradient * np.random.uniform(10, 50)
            else:
                # Random noise
                gradients[node_id] = np.random.randn(self.gradient_dim).astype(np.float32)

        return gradients, actual_byzantine, true_gradient

    def benchmark_node_scaling(self) -> List[Dict]:
        """
        Benchmark how detection scales with node count.

        Returns:
            List of results for each node count
        """
        results = []

        for num_nodes in self.node_counts:
            print(f"    Testing {num_nodes} nodes...")

            trial_results = {
                "detection_rates": [],
                "precisions": [],
                "recalls": [],
                "f1_scores": [],
                "latencies": [],
                "memories": [],
            }

            for trial in range(self.num_trials):
                # Generate gradients
                gradients, actual_byz, true_grad = self.generate_gradients(
                    num_nodes, self.byzantine_ratio
                )

                # Memory profiling
                profiler = MemoryProfiler()
                profiler.start()

                # Run detection
                pogq = ScalablePoGQ()
                start_time = time.perf_counter()
                _, detected = pogq.detect_and_aggregate(gradients)
                latency = time.perf_counter() - start_time

                # Get memory usage
                current_mb, peak_mb = profiler.stop()

                # Compute detection metrics
                honest = set(gradients.keys()) - actual_byz
                tp = len(detected & actual_byz)
                fp = len(detected & honest)
                fn = len(actual_byz - detected)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                detection_rate = recall

                trial_results["detection_rates"].append(detection_rate)
                trial_results["precisions"].append(precision)
                trial_results["recalls"].append(recall)
                trial_results["f1_scores"].append(f1)
                trial_results["latencies"].append(latency * 1000)  # ms
                trial_results["memories"].append(peak_mb)

            # Estimate bandwidth
            bandwidth_bytes = BandwidthEstimator.estimate_round_bandwidth(
                num_nodes, self.gradient_dim, include_proofs=True
            )
            bandwidth_kb = bandwidth_bytes / 1024

            result = ScalabilityResult(
                num_nodes=num_nodes,
                method="pogq",
                detection_rate=float(np.mean(trial_results["detection_rates"])),
                precision=float(np.mean(trial_results["precisions"])),
                recall=float(np.mean(trial_results["recalls"])),
                f1_score=float(np.mean(trial_results["f1_scores"])),
                latency_ms=float(np.mean(trial_results["latencies"])),
                memory_mb=float(np.mean(trial_results["memories"])),
                bandwidth_kb=bandwidth_kb,
                gradient_dim=self.gradient_dim,
            )

            results.append({
                "num_nodes": result.num_nodes,
                "method": result.method,
                "detection_rate": result.detection_rate,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "latency_ms": result.latency_ms,
                "memory_mb": result.memory_mb,
                "bandwidth_kb": result.bandwidth_kb,
                "gradient_dim": result.gradient_dim,
            })

        return results

    def benchmark_gradient_dimension_scaling(self) -> List[Dict]:
        """
        Benchmark how performance scales with gradient dimension.
        """
        results = []

        dimensions = [1000, 5000, 10000, 50000]
        if self.quick_mode:
            dimensions = [1000, 5000, 10000]

        num_nodes = 50

        for dim in dimensions:
            print(f"    Testing gradient dimension {dim}...")

            self.gradient_dim = dim

            trial_results = {
                "latencies": [],
                "memories": [],
            }

            for trial in range(self.num_trials):
                gradients, actual_byz, true_grad = self.generate_gradients(
                    num_nodes, self.byzantine_ratio
                )

                profiler = MemoryProfiler()
                profiler.start()

                pogq = ScalablePoGQ()
                start_time = time.perf_counter()
                pogq.detect_and_aggregate(gradients)
                latency = time.perf_counter() - start_time

                current_mb, peak_mb = profiler.stop()

                trial_results["latencies"].append(latency * 1000)
                trial_results["memories"].append(peak_mb)

            bandwidth_bytes = BandwidthEstimator.estimate_round_bandwidth(
                num_nodes, dim, include_proofs=True
            )

            results.append({
                "gradient_dim": dim,
                "num_nodes": num_nodes,
                "latency_ms": float(np.mean(trial_results["latencies"])),
                "memory_mb": float(np.mean(trial_results["memories"])),
                "bandwidth_kb": bandwidth_bytes / 1024,
            })

        return results

    def benchmark_byzantine_ratio_scaling(self) -> List[Dict]:
        """
        Benchmark detection accuracy at different Byzantine ratios.
        """
        results = []

        byzantine_ratios = [0.10, 0.20, 0.33, 0.40, 0.45]
        num_nodes = 100

        for byz_ratio in byzantine_ratios:
            print(f"    Testing {byz_ratio:.0%} Byzantine ratio...")

            trial_results = {
                "detection_rates": [],
                "f1_scores": [],
            }

            for trial in range(self.num_trials):
                gradients, actual_byz, true_grad = self.generate_gradients(
                    num_nodes, byz_ratio
                )

                pogq = ScalablePoGQ()
                _, detected = pogq.detect_and_aggregate(gradients)

                honest = set(gradients.keys()) - actual_byz
                tp = len(detected & actual_byz)
                fp = len(detected & honest)
                fn = len(actual_byz - detected)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                trial_results["detection_rates"].append(recall)
                trial_results["f1_scores"].append(f1)

            results.append({
                "byzantine_ratio": byz_ratio,
                "num_nodes": num_nodes,
                "detection_rate": float(np.mean(trial_results["detection_rates"])),
                "f1_score": float(np.mean(trial_results["f1_scores"])),
            })

        return results

    def benchmark_memory_efficiency(self) -> Dict:
        """
        Detailed memory efficiency analysis.
        """
        results = {
            "per_node_memory": [],
            "peak_memory": [],
            "memory_scaling": "linear",  # or "sublinear", "superlinear"
        }

        for num_nodes in self.node_counts[:4]:  # Limit for memory safety
            gradients, _, _ = self.generate_gradients(num_nodes, self.byzantine_ratio)

            gc.collect()
            profiler = MemoryProfiler()
            profiler.start()

            pogq = ScalablePoGQ()
            pogq.detect_and_aggregate(gradients)

            current_mb, peak_mb = profiler.stop()

            per_node = peak_mb / num_nodes

            results["per_node_memory"].append({
                "num_nodes": num_nodes,
                "total_mb": peak_mb,
                "per_node_kb": per_node * 1024,
            })
            results["peak_memory"].append(peak_mb)

        # Determine scaling behavior
        if len(results["peak_memory"]) >= 3:
            mems = np.array(results["peak_memory"])
            nodes = np.array(self.node_counts[:len(mems)])

            # Linear regression on log-log scale
            log_nodes = np.log(nodes)
            log_mems = np.log(mems + 1e-6)

            if len(log_nodes) > 1:
                slope = np.polyfit(log_nodes, log_mems, 1)[0]

                if slope < 0.9:
                    results["memory_scaling"] = "sublinear"
                elif slope > 1.1:
                    results["memory_scaling"] = "superlinear"
                else:
                    results["memory_scaling"] = "linear"

        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Run all scalability benchmarks.

        Returns:
            Complete benchmark results
        """
        results = {}

        print("  Running node scaling benchmark...")
        results["node_scaling"] = self.benchmark_node_scaling()

        print("  Running gradient dimension scaling benchmark...")
        results["dimension_scaling"] = self.benchmark_gradient_dimension_scaling()

        print("  Running Byzantine ratio scaling benchmark...")
        results["byzantine_scaling"] = self.benchmark_byzantine_ratio_scaling()

        print("  Running memory efficiency benchmark...")
        results["memory_efficiency"] = self.benchmark_memory_efficiency()

        return results


def main():
    """Run scalability benchmark standalone."""
    print("Scalability Benchmark Suite")
    print("=" * 50)

    benchmark = ScalabilityBenchmark(
        seed=42,
        quick_mode=True,
    )

    results = benchmark.run_all()

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)

    print("\nNode Scaling:")
    for result in results["node_scaling"]:
        print(f"  {result['num_nodes']} nodes: "
              f"detection={result['detection_rate']:.1%}, "
              f"latency={result['latency_ms']:.1f}ms, "
              f"memory={result['memory_mb']:.1f}MB")

    print(f"\nMemory Scaling: {results['memory_efficiency']['memory_scaling']}")


if __name__ == "__main__":
    main()
