# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Latency Distribution Benchmark Scenario

Measures detailed latency distributions for FL aggregation:
- p50 (median) latency
- p95 latency (important for SLA)
- p99 latency (tail latency)
- Min/Max latency
- Standard deviation

Collects enough samples for statistically significant percentile estimation.

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
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "0TML" / "src"))

logger = logging.getLogger(__name__)


@dataclass
class LatencyDistributionConfig:
    """Configuration for latency distribution benchmark."""
    num_nodes: int = 10
    gradient_size: int = 10000
    num_samples: int = 1000  # Need many samples for accurate percentiles
    warmup_samples: int = 50
    seed: int = 42
    breakdown_components: bool = True  # Measure component-level latencies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "gradient_size": self.gradient_size,
            "num_samples": self.num_samples,
            "warmup_samples": self.warmup_samples,
            "seed": self.seed,
            "breakdown_components": self.breakdown_components,
        }


@dataclass
class LatencyStats:
    """Statistical summary of latency distribution."""
    samples: List[float]
    count: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        """Compute stats from samples."""
        if not samples:
            return cls(samples=[])

        arr = np.array(samples)
        return cls(
            samples=samples,
            count=len(samples),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p75_ms=float(np.percentile(arr, 75)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            p999_ms=float(np.percentile(arr, 99.9)),
        )

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p75_ms": self.p75_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "p999_ms": self.p999_ms,
        }
        if include_samples:
            result["samples"] = self.samples
        return result


@dataclass
class LatencyBreakdown:
    """Component-level latency breakdown."""
    total: LatencyStats
    detection: Optional[LatencyStats] = None
    aggregation: Optional[LatencyStats] = None
    encoding: Optional[LatencyStats] = None
    shapley: Optional[LatencyStats] = None

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {"total": self.total.to_dict(include_samples)}
        if self.detection:
            result["detection"] = self.detection.to_dict(include_samples)
        if self.aggregation:
            result["aggregation"] = self.aggregation.to_dict(include_samples)
        if self.encoding:
            result["encoding"] = self.encoding.to_dict(include_samples)
        if self.shapley:
            result["shapley"] = self.shapley.to_dict(include_samples)
        return result


class LatencyDistributionScenario:
    """
    Latency distribution benchmark scenario.

    Measures aggregation latency with high precision and collects
    enough samples for accurate percentile estimation.

    Example:
        >>> scenario = LatencyDistributionScenario()
        >>> breakdown = scenario.run()
        >>> print(f"p50: {breakdown.total.p50_ms:.2f}ms")
        >>> print(f"p99: {breakdown.total.p99_ms:.2f}ms")
    """

    def __init__(self, config: Optional[LatencyDistributionConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or LatencyDistributionConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Try importing mycelix_fl
        try:
            from mycelix_fl import (
                MycelixFL, FLConfig,
                MultiLayerByzantineDetector,
                HyperFeelEncoderV2,
            )
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.MultiLayerByzantineDetector = MultiLayerByzantineDetector
            self.HyperFeelEncoderV2 = HyperFeelEncoderV2
            self.available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.available = False

    def generate_gradients(self) -> Dict[str, np.ndarray]:
        """Generate test gradients."""
        base_gradient = self.rng.randn(self.config.gradient_size).astype(np.float32)
        gradients = {}

        for i in range(self.config.num_nodes):
            noise = self.rng.randn(self.config.gradient_size).astype(np.float32) * 0.1
            gradients[f"node_{i}"] = base_gradient + noise

        return gradients

    def run(self) -> LatencyBreakdown:
        """Run latency distribution benchmark."""
        logger.info(f"Running latency distribution benchmark ({self.config.num_samples} samples)")

        gradients = self.generate_gradients()

        if not self.available:
            # Fallback: measure simple aggregation
            return self._run_fallback(gradients)

        # Set up FL system
        fl_config = self.FLConfig(
            use_detection=True,
            use_healing=True,
        )
        fl = self.MycelixFL(config=fl_config)

        # Warmup
        logger.info(f"Warming up ({self.config.warmup_samples} samples)...")
        for i in range(self.config.warmup_samples):
            fl.execute_round(gradients, round_num=i + 1)
            gc.collect()

        # Collect samples
        logger.info(f"Collecting {self.config.num_samples} samples...")
        total_times = []
        detection_times = []
        aggregation_times = []

        for i in range(self.config.num_samples):
            gc.collect()

            # Total time
            start = time.perf_counter()
            result = fl.execute_round(gradients, round_num=self.config.warmup_samples + i + 1)
            total_elapsed = (time.perf_counter() - start) * 1000
            total_times.append(total_elapsed)

            # Extract component times if available
            if hasattr(result, 'detection_result') and result.detection_result:
                detection_times.append(result.detection_result.total_latency_ms)
                # Estimate aggregation as total - detection
                aggregation_times.append(max(0, total_elapsed - result.detection_result.total_latency_ms))
            elif hasattr(result, 'aggregation_time_ms'):
                aggregation_times.append(result.aggregation_time_ms)

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{self.config.num_samples}")

        # Compute stats
        total_stats = LatencyStats.from_samples(total_times)
        detection_stats = LatencyStats.from_samples(detection_times) if detection_times else None
        aggregation_stats = LatencyStats.from_samples(aggregation_times) if aggregation_times else None

        logger.info(
            f"Latency: mean={total_stats.mean_ms:.2f}ms, "
            f"p50={total_stats.p50_ms:.2f}ms, "
            f"p95={total_stats.p95_ms:.2f}ms, "
            f"p99={total_stats.p99_ms:.2f}ms"
        )

        return LatencyBreakdown(
            total=total_stats,
            detection=detection_stats,
            aggregation=aggregation_stats,
        )

    def _run_fallback(self, gradients: Dict[str, np.ndarray]) -> LatencyBreakdown:
        """Fallback benchmark without mycelix_fl."""
        logger.info("Running fallback latency benchmark (numpy only)")

        times = []

        # Warmup
        for _ in range(self.config.warmup_samples):
            _ = np.mean(list(gradients.values()), axis=0)
            gc.collect()

        # Benchmark
        for i in range(self.config.num_samples):
            gc.collect()
            start = time.perf_counter()

            # Simple FedAvg aggregation
            grads = list(gradients.values())
            aggregated = np.mean(grads, axis=0)

            # Simple outlier detection (magnitude-based)
            norms = [np.linalg.norm(g) for g in grads]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            outliers = [i for i, n in enumerate(norms) if abs(n - mean_norm) > 2 * std_norm]

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{self.config.num_samples}")

        return LatencyBreakdown(
            total=LatencyStats.from_samples(times),
        )

    def run_component_breakdown(self) -> LatencyBreakdown:
        """Run detailed component-level latency breakdown."""
        if not self.available:
            return self._run_fallback(self.generate_gradients())

        logger.info("Running component-level latency breakdown...")
        gradients = self.generate_gradients()

        # Initialize components
        detector = self.MultiLayerByzantineDetector()
        encoder = self.HyperFeelEncoderV2()

        # Warmup
        for _ in range(self.config.warmup_samples):
            detector.detect(gradients)
            gc.collect()

        # Collect samples for each component
        total_times = []
        detection_times = []
        encoding_times = []
        aggregation_times = []

        for i in range(self.config.num_samples):
            gc.collect()

            # Detection
            start = time.perf_counter()
            detection_result = detector.detect(gradients)
            detection_time = (time.perf_counter() - start) * 1000
            detection_times.append(detection_time)

            # Encoding
            sample_gradient = gradients[list(gradients.keys())[0]]
            start = time.perf_counter()
            _ = encoder.encode_gradient(sample_gradient, round_num=i, node_id="bench")
            encoding_time = (time.perf_counter() - start) * 1000
            encoding_times.append(encoding_time)

            # Aggregation (simple median)
            healthy_ids = detection_result.healthy_nodes
            healthy_grads = [gradients[nid] for nid in healthy_ids if nid in gradients]

            start = time.perf_counter()
            if healthy_grads:
                aggregated = np.median(healthy_grads, axis=0)
            aggregation_time = (time.perf_counter() - start) * 1000
            aggregation_times.append(aggregation_time)

            # Total
            total_times.append(detection_time + encoding_time + aggregation_time)

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{self.config.num_samples}")

        return LatencyBreakdown(
            total=LatencyStats.from_samples(total_times),
            detection=LatencyStats.from_samples(detection_times),
            aggregation=LatencyStats.from_samples(aggregation_times),
            encoding=LatencyStats.from_samples(encoding_times),
        )

    def to_benchmark_output(
        self,
        breakdown: LatencyBreakdown,
        include_samples: bool = False,
    ) -> Dict[str, Any]:
        """Convert results to standard benchmark output format."""
        import socket
        import platform

        return {
            "benchmark": "latency_distribution",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            "parameters": self.config.to_dict(),
            "results": breakdown.to_dict(include_samples),
            "metadata": {
                "mycelix_available": self.available,
                "samples_collected": breakdown.total.count,
            },
        }


def run_latency_benchmark(
    output_dir: Optional[Path] = None,
    config: Optional[LatencyDistributionConfig] = None,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Run complete latency distribution benchmark.

    Args:
        output_dir: Directory for output files
        config: Custom configuration
        detailed: If True, run component breakdown

    Returns:
        Benchmark results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Latency Distribution Benchmark")
    logger.info("=" * 60)

    scenario = LatencyDistributionScenario(config=config)

    if detailed:
        breakdown = scenario.run_component_breakdown()
    else:
        breakdown = scenario.run()

    output = scenario.to_benchmark_output(breakdown, include_samples=False)

    if output_dir:
        output_path = Path(output_dir) / "latency_distribution.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latency Distribution Benchmark")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--num-nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--detailed", action="store_true", help="Component breakdown")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = LatencyDistributionConfig(
        num_nodes=args.num_nodes,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    output = run_latency_benchmark(
        output_dir=args.output_dir,
        config=config,
        detailed=args.detailed,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  p50:  {output['results']['total']['p50_ms']:.2f}ms")
    print(f"  p95:  {output['results']['total']['p95_ms']:.2f}ms")
    print(f"  p99:  {output['results']['total']['p99_ms']:.2f}ms")
    print(f"  p999: {output['results']['total']['p999_ms']:.2f}ms")
