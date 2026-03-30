# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Scalability Benchmark Scenario

Tests FL aggregation scalability with increasing node counts:
- 10, 50, 100, 500 nodes
- Measures how latency scales with node count
- Identifies scaling bottlenecks

Outputs scaling coefficient (ideally O(n) or O(n log n)).

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "0TML" / "src"))

logger = logging.getLogger(__name__)


@dataclass
class ScalabilityConfig:
    """Configuration for scalability benchmark."""
    node_counts: List[int] = field(
        default_factory=lambda: [10, 50, 100, 500]
    )
    gradient_size: int = 10000
    num_rounds: int = 20
    warmup_rounds: int = 5
    seed: int = 42
    test_components: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_counts": self.node_counts,
            "gradient_size": self.gradient_size,
            "num_rounds": self.num_rounds,
            "warmup_rounds": self.warmup_rounds,
            "seed": self.seed,
            "test_components": self.test_components,
        }


@dataclass
class ScalabilityResult:
    """Result for a single node count test."""
    num_nodes: int
    gradient_size: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_nodes_per_sec: float
    memory_mb: Optional[float] = None
    component_latencies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "num_nodes": self.num_nodes,
            "gradient_size": self.gradient_size,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_nodes_per_sec": self.throughput_nodes_per_sec,
        }
        if self.memory_mb:
            result["memory_mb"] = self.memory_mb
        if self.component_latencies:
            result["component_latencies"] = self.component_latencies
        return result


@dataclass
class ScalabilityAnalysis:
    """Analysis of scalability results."""
    results: List[ScalabilityResult]
    scaling_coefficient: float  # How latency scales with n
    complexity_estimate: str  # "O(n)", "O(n log n)", "O(n^2)"
    bottleneck_component: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "scaling_coefficient": self.scaling_coefficient,
            "complexity_estimate": self.complexity_estimate,
            "bottleneck_component": self.bottleneck_component,
        }


class ScalabilityScenario:
    """
    Scalability benchmark scenario.

    Tests how aggregation latency scales with node count.

    Example:
        >>> scenario = ScalabilityScenario()
        >>> analysis = scenario.run()
        >>> print(f"Complexity: {analysis.complexity_estimate}")
        >>> for r in analysis.results:
        ...     print(f"  {r.num_nodes} nodes: {r.mean_latency_ms:.2f}ms")
    """

    def __init__(self, config: Optional[ScalabilityConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or ScalabilityConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Try importing mycelix_fl
        try:
            from mycelix_fl import MycelixFL, FLConfig
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.available = False

    def generate_gradients(self, num_nodes: int) -> Dict[str, np.ndarray]:
        """Generate test gradients."""
        base_gradient = self.rng.randn(self.config.gradient_size).astype(np.float32)
        gradients = {}

        for i in range(num_nodes):
            noise = self.rng.randn(self.config.gradient_size).astype(np.float32) * 0.1
            gradients[f"node_{i}"] = base_gradient + noise

        return gradients

    def run_single_count(self, num_nodes: int) -> ScalabilityResult:
        """Run benchmark for a single node count."""
        logger.info(f"Testing with {num_nodes} nodes...")

        # Reset RNG for reproducibility
        self.rng = np.random.RandomState(self.config.seed + num_nodes)

        gradients = self.generate_gradients(num_nodes)

        if not self.available:
            return self._run_fallback(gradients, num_nodes)

        # Set up FL system
        fl_config = self.FLConfig(
            use_detection=True,
            use_healing=True,
        )
        fl = self.MycelixFL(config=fl_config)

        # Warmup
        for i in range(self.config.warmup_rounds):
            fl.execute_round(gradients, round_num=i + 1)
            gc.collect()

        # Benchmark
        latencies = []

        for i in range(self.config.num_rounds):
            gc.collect()
            start = time.perf_counter()
            _ = fl.execute_round(gradients, round_num=self.config.warmup_rounds + i + 1)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        arr = np.array(latencies)
        mean_latency = float(np.mean(arr))
        throughput = num_nodes / (mean_latency / 1000) if mean_latency > 0 else 0

        return ScalabilityResult(
            num_nodes=num_nodes,
            gradient_size=self.config.gradient_size,
            mean_latency_ms=mean_latency,
            std_latency_ms=float(np.std(arr)),
            min_latency_ms=float(np.min(arr)),
            max_latency_ms=float(np.max(arr)),
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            throughput_nodes_per_sec=throughput,
        )

    def _run_fallback(
        self,
        gradients: Dict[str, np.ndarray],
        num_nodes: int,
    ) -> ScalabilityResult:
        """Fallback benchmark without mycelix_fl."""
        latencies = []

        # Warmup
        for _ in range(self.config.warmup_rounds):
            _ = np.mean(list(gradients.values()), axis=0)
            gc.collect()

        # Benchmark
        for _ in range(self.config.num_rounds):
            gc.collect()
            start = time.perf_counter()

            # Simple aggregation with outlier detection
            grads = list(gradients.values())
            norms = [np.linalg.norm(g) for g in grads]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)

            # Filter outliers
            valid_indices = [
                i for i, n in enumerate(norms)
                if abs(n - mean_norm) <= 2 * std_norm
            ]
            valid_grads = [grads[i] for i in valid_indices]

            # Aggregate
            aggregated = np.mean(valid_grads, axis=0) if valid_grads else np.zeros_like(grads[0])

            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        arr = np.array(latencies)
        mean_latency = float(np.mean(arr))
        throughput = num_nodes / (mean_latency / 1000) if mean_latency > 0 else 0

        return ScalabilityResult(
            num_nodes=num_nodes,
            gradient_size=self.config.gradient_size,
            mean_latency_ms=mean_latency,
            std_latency_ms=float(np.std(arr)),
            min_latency_ms=float(np.min(arr)),
            max_latency_ms=float(np.max(arr)),
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            throughput_nodes_per_sec=throughput,
        )

    def analyze_scaling(self, results: List[ScalabilityResult]) -> Tuple[float, str]:
        """
        Analyze scaling behavior from results.

        Returns:
            (scaling_coefficient, complexity_estimate)
        """
        if len(results) < 2:
            return 1.0, "O(n)"

        # Extract data points
        ns = np.array([r.num_nodes for r in results])
        latencies = np.array([r.mean_latency_ms for r in results])

        # Fit different models
        # O(n): latency = a * n + b
        # O(n log n): latency = a * n * log(n) + b
        # O(n^2): latency = a * n^2 + b

        # Normalize for numerical stability
        ns_norm = ns / ns.max()
        latencies_norm = latencies / latencies.max()

        # Fit O(n)
        A_linear = np.column_stack([ns_norm, np.ones(len(ns))])
        coef_linear, residuals_linear, _, _ = np.linalg.lstsq(A_linear, latencies_norm, rcond=None)
        if len(residuals_linear) > 0:
            error_linear = float(residuals_linear[0])
        else:
            error_linear = float(np.sum((A_linear @ coef_linear - latencies_norm) ** 2))

        # Fit O(n log n)
        ns_logn = ns_norm * np.log(ns + 1)
        A_nlogn = np.column_stack([ns_logn, np.ones(len(ns))])
        coef_nlogn, residuals_nlogn, _, _ = np.linalg.lstsq(A_nlogn, latencies_norm, rcond=None)
        if len(residuals_nlogn) > 0:
            error_nlogn = float(residuals_nlogn[0])
        else:
            error_nlogn = float(np.sum((A_nlogn @ coef_nlogn - latencies_norm) ** 2))

        # Fit O(n^2)
        ns_sq = ns_norm ** 2
        A_quad = np.column_stack([ns_sq, np.ones(len(ns))])
        coef_quad, residuals_quad, _, _ = np.linalg.lstsq(A_quad, latencies_norm, rcond=None)
        if len(residuals_quad) > 0:
            error_quad = float(residuals_quad[0])
        else:
            error_quad = float(np.sum((A_quad @ coef_quad - latencies_norm) ** 2))

        # Choose best fit
        errors = {
            "O(n)": error_linear,
            "O(n log n)": error_nlogn,
            "O(n^2)": error_quad,
        }
        best_fit = min(errors, key=errors.get)

        # Compute scaling coefficient (ratio of latency increase to node increase)
        if len(results) >= 2:
            n_ratio = results[-1].num_nodes / results[0].num_nodes
            latency_ratio = results[-1].mean_latency_ms / max(results[0].mean_latency_ms, 0.001)
            scaling_coef = np.log(latency_ratio) / np.log(n_ratio)
        else:
            scaling_coef = 1.0

        return float(scaling_coef), best_fit

    def run(self) -> ScalabilityAnalysis:
        """Run scalability benchmark for all configured node counts."""
        logger.info(f"Running scalability benchmark for {self.config.node_counts}")

        results = []
        for num_nodes in self.config.node_counts:
            result = self.run_single_count(num_nodes)
            results.append(result)

            logger.info(
                f"  {num_nodes} nodes: "
                f"{result.mean_latency_ms:.2f}ms mean, "
                f"{result.throughput_nodes_per_sec:.0f} nodes/sec"
            )

        # Analyze scaling
        scaling_coef, complexity = self.analyze_scaling(results)
        logger.info(f"Scaling analysis: {complexity} (coefficient: {scaling_coef:.2f})")

        return ScalabilityAnalysis(
            results=results,
            scaling_coefficient=scaling_coef,
            complexity_estimate=complexity,
        )

    def to_benchmark_output(self, analysis: ScalabilityAnalysis) -> Dict[str, Any]:
        """Convert results to standard benchmark output format."""
        import socket
        import platform

        return {
            "benchmark": "scalability",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            "parameters": self.config.to_dict(),
            "results": {
                "by_node_count": {
                    str(r.num_nodes): r.to_dict()
                    for r in analysis.results
                },
                "analysis": {
                    "scaling_coefficient": analysis.scaling_coefficient,
                    "complexity_estimate": analysis.complexity_estimate,
                    "bottleneck_component": analysis.bottleneck_component,
                },
            },
            "metadata": {
                "mycelix_available": self.available,
                "node_counts_tested": len(analysis.results),
            },
        }


def run_scalability_benchmark(
    output_dir: Optional[Path] = None,
    config: Optional[ScalabilityConfig] = None,
) -> Dict[str, Any]:
    """
    Run complete scalability benchmark.

    Args:
        output_dir: Directory for output files
        config: Custom configuration

    Returns:
        Benchmark results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Scalability Benchmark")
    logger.info("=" * 60)

    scenario = ScalabilityScenario(config=config)
    analysis = scenario.run()
    output = scenario.to_benchmark_output(analysis)

    if output_dir:
        output_path = Path(output_dir) / "scalability.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scalability Benchmark")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500],
        help="Node counts to test",
    )
    parser.add_argument("--num-rounds", type=int, default=20, help="Rounds per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = ScalabilityConfig(
        node_counts=args.node_counts,
        num_rounds=args.num_rounds,
        seed=args.seed,
    )

    output = run_scalability_benchmark(
        output_dir=args.output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Complexity: {output['results']['analysis']['complexity_estimate']}")
    print(f"  Scaling coefficient: {output['results']['analysis']['scaling_coefficient']:.2f}")
    print()
    for count, data in output["results"]["by_node_count"].items():
        print(
            f"  {count} nodes: "
            f"{data['mean_latency_ms']:.2f}ms mean, "
            f"{data['throughput_nodes_per_sec']:.0f} nodes/sec"
        )
