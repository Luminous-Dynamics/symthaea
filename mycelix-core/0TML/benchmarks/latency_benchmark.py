#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Latency Benchmark Suite
========================

Measures latency for key PoGQ operations:
1. Aggregation latency (target: <1ms median)
2. End-to-end round latency
3. Proof generation time (RISC0 vs Winterfell)

Author: Luminous Dynamics
Date: January 8, 2026
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    samples: List[float]

    @property
    def mean_ms(self) -> float:
        return np.mean(self.samples) * 1000

    @property
    def std_ms(self) -> float:
        return np.std(self.samples) * 1000

    @property
    def median_ms(self) -> float:
        return np.median(self.samples) * 1000

    @property
    def p95_ms(self) -> float:
        return np.percentile(self.samples, 95) * 1000

    @property
    def p99_ms(self) -> float:
        return np.percentile(self.samples, 99) * 1000

    @property
    def min_ms(self) -> float:
        return np.min(self.samples) * 1000

    @property
    def max_ms(self) -> float:
        return np.max(self.samples) * 1000

    def to_dict(self) -> Dict:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "num_samples": len(self.samples),
        }


class AggregationMethods:
    """Aggregation method implementations for latency testing."""

    @staticmethod
    def pogq_aggregate(
        gradients: Dict[str, np.ndarray],
        quality_threshold: float = 0.35,
    ) -> np.ndarray:
        """PoGQ quality-weighted aggregation."""
        all_grads = list(gradients.values())
        reference = np.mean(all_grads, axis=0)
        reference_norm = np.linalg.norm(reference) + 1e-8

        valid_gradients = []
        weights = []

        for grad in all_grads:
            grad_norm = np.linalg.norm(grad) + 1e-8
            cosine = np.dot(grad, reference) / (grad_norm * reference_norm)
            quality = (cosine + 1) / 2

            if quality >= quality_threshold:
                valid_gradients.append(grad)
                weights.append(quality)

        if len(valid_gradients) > 0:
            weights = np.array(weights)
            weights /= weights.sum()
            aggregated = np.zeros_like(valid_gradients[0])
            for w, g in zip(weights, valid_gradients):
                aggregated += w * g
        else:
            aggregated = np.median(all_grads, axis=0)

        return aggregated

    @staticmethod
    def fltrust_aggregate(
        gradients: Dict[str, np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """FLTrust trust-weighted aggregation."""
        all_grads = list(gradients.values())

        if server_gradient is None:
            server_gradient = np.mean(all_grads, axis=0)

        server_norm = np.linalg.norm(server_gradient) + 1e-8

        trust_scores = []
        for grad in all_grads:
            grad_norm = np.linalg.norm(grad) + 1e-8
            cosine = np.dot(grad, server_gradient) / (grad_norm * server_norm)
            trust_scores.append(max(0.0, cosine))

        trust_scores = np.array(trust_scores)
        if trust_scores.sum() > 1e-8:
            trust_scores /= trust_scores.sum()
        else:
            trust_scores = np.ones(len(all_grads)) / len(all_grads)

        aggregated = np.zeros_like(all_grads[0])
        for w, g in zip(trust_scores, all_grads):
            aggregated += w * g

        return aggregated

    @staticmethod
    def krum_aggregate(
        gradients: Dict[str, np.ndarray],
        f: int = 2,
    ) -> np.ndarray:
        """Krum gradient selection."""
        all_grads = list(gradients.values())
        n = len(all_grads)

        if n <= 1:
            return all_grads[0] if all_grads else np.zeros(1)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((all_grads[i] - all_grads[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        m = max(1, n - f - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores.append(np.sum(sorted_dists[1:m+1]))

        selected_idx = int(np.argmin(scores))
        return all_grads[selected_idx]

    @staticmethod
    def fedavg_aggregate(gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple FedAvg averaging."""
        all_grads = list(gradients.values())
        return np.mean(all_grads, axis=0)


class ProofGenerators:
    """Proof generation implementations for latency testing."""

    @staticmethod
    def sha3_proof(gradient: np.ndarray, nonce: str = "") -> str:
        """SHA3-256 hash-based proof (baseline)."""
        grad_bytes = gradient.astype(np.float32).tobytes()
        combined = grad_bytes + nonce.encode()
        return hashlib.sha3_256(combined).hexdigest()

    @staticmethod
    def risc0_proof(gradient: np.ndarray, difficulty: int = 4) -> Dict:
        """
        Simulated RISC0 ZK proof generation.

        In production, this would invoke the RISC0 zkVM.
        We simulate realistic latency based on gradient size.
        """
        # Simulate ZK computation overhead
        # RISC0 proof generation is typically 10-100x slower than SHA3
        gradient_size = gradient.size

        # Simulate proof-of-work for complexity
        grad_bytes = gradient.astype(np.float32).tobytes()
        target = "0" * difficulty

        counter = 0
        nonce = f"{np.random.randint(0, 1000000)}"

        while counter < 1000:  # Limited iterations for simulation
            attempt = grad_bytes + f"{nonce}{counter}".encode()
            hash_result = hashlib.sha3_256(attempt).hexdigest()

            if hash_result.startswith(target):
                return {
                    "proof_hash": hash_result,
                    "counter": counter,
                    "nonce": nonce,
                    "type": "risc0_simulated",
                }
            counter += 1

        # Fallback
        return {
            "proof_hash": hashlib.sha3_256(grad_bytes + nonce.encode()).hexdigest(),
            "counter": counter,
            "nonce": nonce,
            "type": "risc0_simulated",
        }

    @staticmethod
    def winterfell_proof(gradient: np.ndarray, trace_length: int = 64) -> Dict:
        """
        Simulated Winterfell STARK proof generation.

        Winterfell uses AIR (Algebraic Intermediate Representation)
        for STARK proofs. We simulate realistic computation.
        """
        grad_bytes = gradient.astype(np.float32).tobytes()

        # Simulate trace computation
        trace = []
        state = int.from_bytes(hashlib.sha256(grad_bytes[:32]).digest()[:8], 'big')

        for _ in range(trace_length):
            state = (state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
            trace.append(state)

        # Generate constraint polynomial commitment
        commitment = hashlib.sha3_256(
            grad_bytes + b"".join(t.to_bytes(8, 'big') for t in trace)
        ).hexdigest()

        return {
            "commitment": commitment,
            "trace_length": trace_length,
            "type": "winterfell_simulated",
        }


class LatencyBenchmark:
    """
    Latency benchmark suite for PoGQ operations.

    Measures:
    1. Aggregation latency across methods
    2. End-to-end FL round latency
    3. Proof generation time for different backends
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
        gradient_dim: int = 10000,
    ):
        """
        Initialize benchmark.

        Args:
            seed: Random seed
            quick_mode: Run fewer iterations
            gradient_dim: Gradient dimension
        """
        self.seed = seed
        self.quick_mode = quick_mode
        self.gradient_dim = gradient_dim

        np.random.seed(seed)

        # Benchmark configuration
        if quick_mode:
            self.num_iterations = 100
            self.num_nodes = 20
            self.warmup_iterations = 10
        else:
            self.num_iterations = 1000
            self.num_nodes = 50
            self.warmup_iterations = 50

    def generate_gradients(self, num_nodes: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate random gradients for benchmarking."""
        n = num_nodes or self.num_nodes
        gradients = {}

        for i in range(n):
            gradients[f"node_{i}"] = np.random.randn(self.gradient_dim).astype(np.float32)

        return gradients

    def benchmark_aggregation(self) -> Dict[str, Any]:
        """
        Benchmark aggregation latency for all methods.

        Returns:
            Dict mapping method_name -> latency stats
        """
        results = {}

        methods = [
            ("pogq", lambda g: AggregationMethods.pogq_aggregate(g)),
            ("fltrust", lambda g: AggregationMethods.fltrust_aggregate(g)),
            ("krum", lambda g: AggregationMethods.krum_aggregate(g)),
            ("fedavg", lambda g: AggregationMethods.fedavg_aggregate(g)),
        ]

        for method_name, method_func in methods:
            print(f"    Benchmarking {method_name} aggregation...")

            # Warmup
            for _ in range(self.warmup_iterations):
                gradients = self.generate_gradients()
                method_func(gradients)

            # Benchmark
            latencies = []
            for _ in range(self.num_iterations):
                gradients = self.generate_gradients()

                start = time.perf_counter()
                method_func(gradients)
                elapsed = time.perf_counter() - start

                latencies.append(elapsed)

            stats = LatencyStats(latencies)
            results[method_name] = stats.to_dict()

        return results

    def benchmark_e2e_round(self) -> Dict[str, Any]:
        """
        Benchmark end-to-end FL round latency.

        Includes:
        - Gradient collection
        - Detection
        - Aggregation
        - Result distribution (simulated)
        """
        results = {}

        def simulate_round(method_func):
            """Simulate a complete FL round."""
            # 1. Generate gradients (simulates collection)
            gradients = self.generate_gradients()

            # 2. Detection + Aggregation
            aggregated = method_func(gradients)

            # 3. Simulate result distribution (hash computation)
            result_hash = hashlib.sha256(aggregated.tobytes()).hexdigest()

            return aggregated, result_hash

        methods = [
            ("pogq", lambda g: AggregationMethods.pogq_aggregate(g)),
            ("fltrust", lambda g: AggregationMethods.fltrust_aggregate(g)),
            ("krum", lambda g: AggregationMethods.krum_aggregate(g)),
            ("fedavg", lambda g: AggregationMethods.fedavg_aggregate(g)),
        ]

        for method_name, method_func in methods:
            print(f"    Benchmarking {method_name} E2E round...")

            # Warmup
            for _ in range(self.warmup_iterations):
                simulate_round(method_func)

            # Benchmark
            latencies = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                simulate_round(method_func)
                elapsed = time.perf_counter() - start

                latencies.append(elapsed)

            stats = LatencyStats(latencies)
            results[method_name] = stats.to_dict()

        return results

    def benchmark_proof_generation(self) -> Dict[str, Any]:
        """
        Benchmark proof generation latency.

        Compares:
        - SHA3-only (baseline)
        - RISC0 (simulated)
        - Winterfell (simulated)
        """
        results = {}

        # Generate sample gradient
        sample_gradient = np.random.randn(self.gradient_dim).astype(np.float32)

        proof_methods = [
            ("sha3_only", lambda g: ProofGenerators.sha3_proof(g)),
            ("risc0", lambda g: ProofGenerators.risc0_proof(g, difficulty=3)),
            ("winterfell", lambda g: ProofGenerators.winterfell_proof(g, trace_length=64)),
        ]

        for method_name, proof_func in proof_methods:
            print(f"    Benchmarking {method_name} proof generation...")

            # Use fewer iterations for ZK proofs (they're slower)
            num_iter = self.num_iterations // 10 if method_name != "sha3_only" else self.num_iterations

            # Warmup
            for _ in range(min(10, num_iter // 10)):
                proof_func(sample_gradient)

            # Benchmark
            latencies = []
            for _ in range(num_iter):
                # Generate fresh gradient each time
                gradient = np.random.randn(self.gradient_dim).astype(np.float32)

                start = time.perf_counter()
                proof_func(gradient)
                elapsed = time.perf_counter() - start

                latencies.append(elapsed)

            stats = LatencyStats(latencies)
            results[method_name] = stats.to_dict()

        return results

    def benchmark_scalability_latency(self) -> Dict[str, Any]:
        """
        Benchmark how latency scales with number of nodes.
        """
        results = {}

        node_counts = [10, 20, 50, 100]
        if not self.quick_mode:
            node_counts.extend([200, 500])

        for num_nodes in node_counts:
            print(f"    Benchmarking latency with {num_nodes} nodes...")

            latencies = []
            num_iter = self.num_iterations // 10

            for _ in range(num_iter):
                gradients = self.generate_gradients(num_nodes)

                start = time.perf_counter()
                AggregationMethods.pogq_aggregate(gradients)
                elapsed = time.perf_counter() - start

                latencies.append(elapsed)

            stats = LatencyStats(latencies)
            results[str(num_nodes)] = stats.to_dict()

        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Run all latency benchmarks.

        Returns:
            Complete benchmark results
        """
        results = {}

        print("  Running aggregation latency benchmark...")
        results["aggregation"] = self.benchmark_aggregation()

        print("  Running E2E round latency benchmark...")
        results["e2e_round"] = self.benchmark_e2e_round()

        print("  Running proof generation benchmark...")
        results["proof_generation"] = self.benchmark_proof_generation()

        print("  Running scalability latency benchmark...")
        results["scalability_latency"] = self.benchmark_scalability_latency()

        return results


def main():
    """Run latency benchmark standalone."""
    print("Latency Benchmark Suite")
    print("=" * 50)

    benchmark = LatencyBenchmark(
        seed=42,
        quick_mode=True,
    )

    results = benchmark.run_all()

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)

    print("\nAggregation Latency:")
    for method, stats in results["aggregation"].items():
        print(f"  {method}: median={stats['median_ms']:.3f}ms, p99={stats['p99_ms']:.3f}ms")

    print("\nProof Generation:")
    for method, stats in results["proof_generation"].items():
        print(f"  {method}: median={stats['median_ms']:.3f}ms, p99={stats['p99_ms']:.3f}ms")


if __name__ == "__main__":
    main()
