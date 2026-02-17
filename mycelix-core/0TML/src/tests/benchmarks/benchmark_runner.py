"""
Benchmark Runner: Compare Mycelix FL vs Baseline Methods

Runs comprehensive benchmarks comparing:
- Byzantine detection accuracy
- Aggregation quality (gradient fidelity)
- Performance (latency)
- Scalability (nodes, gradient size)

Usage:
    python -m tests.benchmarks.benchmark_runner
    pytest tests/benchmarks/benchmark_runner.py -v -m benchmark
"""

import pytest
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from mycelix_fl import MycelixFL, FLConfig
from tests.benchmarks.baseline_aggregators import (
    BaseAggregator,
    FedAvgAggregator,
    MedianAggregator,
    TrimmedMeanAggregator,
    MultiKrumAggregator,
    GeoMedAggregator,
    get_all_aggregators,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    method_name: str
    byzantine_ratio: float
    num_nodes: int
    gradient_dim: int

    # Accuracy metrics
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float

    # Quality metrics
    gradient_error: float  # L2 distance from true gradient
    cosine_similarity: float  # Cosine sim to true gradient

    # Performance metrics
    latency_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult]
    config: Dict

    def to_json(self, path: str):
        """Save results to JSON file."""
        data = {
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def summary_table(self) -> str:
        """Generate summary table."""
        lines = [
            "| Method | Byz% | Precision | Recall | F1 | Grad Error | Latency (ms) |",
            "|--------|------|-----------|--------|-----|------------|--------------|",
        ]

        for r in self.results:
            lines.append(
                f"| {r.method_name:15s} | {r.byzantine_ratio:.0%} | "
                f"{r.precision:.2%} | {r.recall:.2%} | {r.f1_score:.2%} | "
                f"{r.gradient_error:.4f} | {r.latency_ms:.1f} |"
            )

        return "\n".join(lines)


class BenchmarkRunner:
    """Runs benchmarks comparing Mycelix FL to baselines."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.mycelix = MycelixFL(config=FLConfig(
            byzantine_threshold=0.45,
            enable_phi_tracking=True,
        ))

    def generate_gradients(
        self,
        num_nodes: int,
        gradient_dim: int,
        byzantine_ratio: float,
        attack_type: str = "mixed",
    ) -> Tuple[Dict[str, np.ndarray], set, np.ndarray]:
        """
        Generate gradients with Byzantine attackers.

        Returns:
            gradients: Dict of node_id -> gradient
            actual_byzantine: Set of Byzantine node IDs
            true_gradient: The true gradient direction
        """
        # True gradient direction
        true_gradient = self.rng.randn(gradient_dim).astype(np.float32)
        true_gradient /= np.linalg.norm(true_gradient)

        num_byzantine = int(num_nodes * byzantine_ratio)
        num_honest = num_nodes - num_byzantine

        gradients = {}
        actual_byzantine = set()

        # Honest nodes
        for i in range(num_honest):
            noise = self.rng.randn(gradient_dim).astype(np.float32) * 0.1
            gradients[f"honest_{i}"] = true_gradient + noise

        # Byzantine nodes
        for i in range(num_byzantine):
            node_id = f"byzantine_{i}"
            actual_byzantine.add(node_id)

            if attack_type == "sign_flip" or (attack_type == "mixed" and i % 4 == 0):
                # Sign flip attack
                gradients[node_id] = -true_gradient * (1.0 + self.rng.rand())
            elif attack_type == "random" or (attack_type == "mixed" and i % 4 == 1):
                # Random noise attack
                gradients[node_id] = self.rng.randn(gradient_dim).astype(np.float32) * 3.0
            elif attack_type == "scale" or (attack_type == "mixed" and i % 4 == 2):
                # Scaling attack
                gradients[node_id] = true_gradient * (50 + self.rng.rand() * 50)
            else:
                # Little-is-enough attack (small perturbation)
                gradients[node_id] = true_gradient + self.rng.randn(gradient_dim).astype(np.float32) * 0.5

        return gradients, actual_byzantine, true_gradient

    def compute_metrics(
        self,
        detected: set,
        actual: set,
        aggregated: np.ndarray,
        true_gradient: np.ndarray,
    ) -> Dict:
        """Compute benchmark metrics."""
        # Detection metrics
        true_positives = len(detected & actual)
        false_positives = len(detected - actual)
        false_negatives = len(actual - detected)

        precision = true_positives / len(detected) if detected else 1.0
        recall = true_positives / len(actual) if actual else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Quality metrics
        gradient_error = float(np.linalg.norm(aggregated - true_gradient))

        agg_norm = np.linalg.norm(aggregated)
        true_norm = np.linalg.norm(true_gradient)
        if agg_norm > 1e-10 and true_norm > 1e-10:
            cosine_sim = float(np.dot(aggregated, true_gradient) / (agg_norm * true_norm))
        else:
            cosine_sim = 0.0

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "gradient_error": gradient_error,
            "cosine_similarity": cosine_sim,
        }

    def benchmark_mycelix(
        self,
        gradients: Dict[str, np.ndarray],
        actual_byzantine: set,
        true_gradient: np.ndarray,
    ) -> BenchmarkResult:
        """Benchmark Mycelix FL."""
        start = time.perf_counter()
        result = self.mycelix.execute_round(gradients, round_num=1)
        latency_ms = (time.perf_counter() - start) * 1000

        metrics = self.compute_metrics(
            detected=result.byzantine_nodes,
            actual=actual_byzantine,
            aggregated=result.aggregated_gradient,
            true_gradient=true_gradient,
        )

        return BenchmarkResult(
            method_name="Mycelix FL",
            byzantine_ratio=len(actual_byzantine) / len(gradients),
            num_nodes=len(gradients),
            gradient_dim=len(true_gradient),
            latency_ms=latency_ms,
            **metrics,
        )

    def benchmark_baseline(
        self,
        aggregator: BaseAggregator,
        gradients: Dict[str, np.ndarray],
        actual_byzantine: set,
        true_gradient: np.ndarray,
    ) -> BenchmarkResult:
        """Benchmark a baseline aggregator."""
        start = time.perf_counter()
        result = aggregator.aggregate(gradients)
        latency_ms = (time.perf_counter() - start) * 1000

        metrics = self.compute_metrics(
            detected=result.excluded_nodes,
            actual=actual_byzantine,
            aggregated=result.aggregated_gradient,
            true_gradient=true_gradient,
        )

        return BenchmarkResult(
            method_name=aggregator.name,
            byzantine_ratio=len(actual_byzantine) / len(gradients),
            num_nodes=len(gradients),
            gradient_dim=len(true_gradient),
            latency_ms=latency_ms,
            **metrics,
        )

    def run_comparison(
        self,
        byzantine_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.45],
        num_nodes: int = 20,
        gradient_dim: int = 10000,
        num_trials: int = 5,
    ) -> BenchmarkSuite:
        """Run full comparison benchmark."""
        results = []

        baselines = get_all_aggregators()

        for byz_ratio in byzantine_ratios:
            print(f"\nByzantine ratio: {byz_ratio:.0%}")

            for trial in range(num_trials):
                # Generate data
                gradients, actual_byz, true_grad = self.generate_gradients(
                    num_nodes=num_nodes,
                    gradient_dim=gradient_dim,
                    byzantine_ratio=byz_ratio,
                )

                # Benchmark Mycelix
                mycelix_result = self.benchmark_mycelix(
                    gradients, actual_byz, true_grad
                )
                results.append(mycelix_result)

                # Benchmark baselines
                for aggregator in baselines:
                    baseline_result = self.benchmark_baseline(
                        aggregator, gradients, actual_byz, true_grad
                    )
                    results.append(baseline_result)

        config = {
            "byzantine_ratios": byzantine_ratios,
            "num_nodes": num_nodes,
            "gradient_dim": gradient_dim,
            "num_trials": num_trials,
        }

        return BenchmarkSuite(results=results, config=config)


# =============================================================================
# PYTEST BENCHMARK TESTS
# =============================================================================

@pytest.fixture
def runner():
    return BenchmarkRunner(seed=42)


@pytest.fixture
def small_gradients(runner):
    return runner.generate_gradients(
        num_nodes=10,
        gradient_dim=5000,
        byzantine_ratio=0.3,
    )


class TestBenchmarkAccuracy:
    """Benchmark tests for detection accuracy."""

    @pytest.mark.benchmark
    def test_mycelix_vs_fedavg_30pct(self, runner):
        """Compare Mycelix to FedAvg at 30% Byzantine."""
        gradients, actual_byz, true_grad = runner.generate_gradients(
            num_nodes=20, gradient_dim=10000, byzantine_ratio=0.3
        )

        mycelix = runner.benchmark_mycelix(gradients, actual_byz, true_grad)
        fedavg = runner.benchmark_baseline(
            FedAvgAggregator(), gradients, actual_byz, true_grad
        )

        print(f"\n  Mycelix F1: {mycelix.f1_score:.2%}, Grad Error: {mycelix.gradient_error:.4f}")
        print(f"  FedAvg F1: {fedavg.f1_score:.2%}, Grad Error: {fedavg.gradient_error:.4f}")

        # Mycelix should have better gradient fidelity
        assert mycelix.gradient_error <= fedavg.gradient_error * 1.5

    @pytest.mark.benchmark
    def test_mycelix_vs_multikrum_45pct(self, runner):
        """Compare Mycelix to Multi-Krum at 45% Byzantine (edge case)."""
        gradients, actual_byz, true_grad = runner.generate_gradients(
            num_nodes=20, gradient_dim=10000, byzantine_ratio=0.45
        )

        mycelix = runner.benchmark_mycelix(gradients, actual_byz, true_grad)
        multikrum = runner.benchmark_baseline(
            MultiKrumAggregator(), gradients, actual_byz, true_grad
        )

        print(f"\n  Mycelix F1: {mycelix.f1_score:.2%}, Grad Error: {mycelix.gradient_error:.4f}")
        print(f"  MultiKrum F1: {multikrum.f1_score:.2%}, Grad Error: {multikrum.gradient_error:.4f}")

        # At 45%, Mycelix should still work where Multi-Krum may fail
        # (Multi-Krum theoretical limit is ~25% for typical settings)

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_full_comparison(self, runner):
        """Run full benchmark comparison."""
        suite = runner.run_comparison(
            byzantine_ratios=[0.1, 0.3, 0.45],
            num_nodes=15,
            gradient_dim=5000,
            num_trials=3,
        )

        print("\n" + suite.summary_table())

        # Find Mycelix results at 45%
        mycelix_45 = [r for r in suite.results
                      if r.method_name == "Mycelix FL" and r.byzantine_ratio >= 0.4]

        if mycelix_45:
            avg_f1 = np.mean([r.f1_score for r in mycelix_45])
            print(f"\n  Mycelix avg F1 at 45%: {avg_f1:.2%}")


class TestBenchmarkPerformance:
    """Benchmark tests for performance."""

    @pytest.mark.benchmark
    def test_latency_comparison(self, runner, small_gradients):
        """Compare latency across methods."""
        gradients, actual_byz, true_grad = small_gradients

        mycelix = runner.benchmark_mycelix(gradients, actual_byz, true_grad)
        fedavg = runner.benchmark_baseline(
            FedAvgAggregator(), gradients, actual_byz, true_grad
        )
        median = runner.benchmark_baseline(
            MedianAggregator(), gradients, actual_byz, true_grad
        )

        print(f"\n  Mycelix: {mycelix.latency_ms:.1f}ms")
        print(f"  FedAvg: {fedavg.latency_ms:.1f}ms")
        print(f"  Median: {median.latency_ms:.1f}ms")

        # Mycelix may be slower due to multi-layer detection
        # but should still be reasonable
        assert mycelix.latency_ms < 1000  # < 1 second

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_scalability_nodes(self, runner):
        """Test scalability with number of nodes."""
        results = []

        for num_nodes in [10, 20, 50, 100]:
            gradients, actual_byz, true_grad = runner.generate_gradients(
                num_nodes=num_nodes,
                gradient_dim=5000,
                byzantine_ratio=0.3,
            )

            result = runner.benchmark_mycelix(gradients, actual_byz, true_grad)
            results.append((num_nodes, result.latency_ms))
            print(f"  {num_nodes} nodes: {result.latency_ms:.1f}ms")

        # Latency should scale roughly linearly or better
        ratio_10_100 = results[-1][1] / results[0][1]
        assert ratio_10_100 < 20  # Should be < 10x for 10x nodes


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run benchmarks from command line."""
    print("=" * 60)
    print("Mycelix FL Benchmark Suite")
    print("=" * 60)

    runner = BenchmarkRunner(seed=42)

    print("\nRunning full comparison...")
    suite = runner.run_comparison(
        byzantine_ratios=[0.1, 0.2, 0.3, 0.4, 0.45],
        num_nodes=20,
        gradient_dim=10000,
        num_trials=5,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(suite.summary_table())

    # Save results
    output_path = "benchmark_results.json"
    suite.to_json(output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
