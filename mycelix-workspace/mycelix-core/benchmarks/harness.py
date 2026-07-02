# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Benchmark Harness for Mycelix Federated Learning

Provides infrastructure for running reproducible benchmarks with:
- Configurable FL network setup
- Byzantine behavior injection
- Comprehensive metric collection
- Publication-ready report generation

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Try importing Rust backend
RUST_BACKEND_AVAILABLE = False
try:
    import fl_aggregator
    RUST_BACKEND_AVAILABLE = True
except ImportError:
    pass

# Try importing mycelix_fl
MYCELIX_FL_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "0TML" / "src"))
    import mycelix_fl
    from mycelix_fl import (
        MycelixFL, FLConfig, RoundResult,
        MultiLayerByzantineDetector, DetectionResult,
        HyperFeelEncoderV2, HyperGradient,
        AttackOrchestrator, create_attack, AttackType,
    )
    MYCELIX_FL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"mycelix_fl not available: {e}")


class Backend(Enum):
    """Available backends for benchmarking."""
    RUST = "rust"
    PYTHON = "python"
    AUTO = "auto"


@dataclass
class EnvironmentInfo:
    """System environment information for reproducibility."""
    hostname: str
    platform: str
    platform_version: str
    python_version: str
    numpy_version: str
    cpu_count: int
    rust_backend_available: bool
    mycelix_fl_version: str
    timestamp: str

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Capture current environment info."""
        return cls(
            hostname=socket.gethostname(),
            platform=platform.system(),
            platform_version=platform.version(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            cpu_count=os.cpu_count() or 1,
            rust_backend_available=RUST_BACKEND_AVAILABLE,
            mycelix_fl_version=mycelix_fl.__version__ if MYCELIX_FL_AVAILABLE else "N/A",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    description: str = ""
    num_nodes: int = 10
    gradient_size: int = 10000
    num_rounds: int = 5
    warmup_rounds: int = 2
    byzantine_ratio: float = 0.0
    attack_types: List[str] = field(default_factory=list)
    backend: Backend = Backend.AUTO
    seed: int = 42
    collect_latency_percentiles: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["backend"] = self.backend.value
        return result


@dataclass
class TimingMetrics:
    """Timing metrics from benchmark runs."""
    samples_ms: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples_ms)) if self.samples_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.samples_ms)) if self.samples_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.samples_ms)) if self.samples_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.samples_ms)) if self.samples_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 50)) if self.samples_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 95)) if self.samples_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 99)) if self.samples_ms else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "samples_ms": self.samples_ms,
        }


@dataclass
class ByzantineMetrics:
    """Byzantine detection metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
        }


@dataclass
class CompressionMetrics:
    """HyperFeel compression metrics."""
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    encoding_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    cosine_similarity: float = 0.0  # Quality: similarity between original and reconstructed
    mse: float = 0.0  # Mean squared error for reconstruction quality

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete result from a benchmark run."""
    benchmark: str
    timestamp: str
    environment: EnvironmentInfo
    parameters: BenchmarkConfig
    timing: TimingMetrics
    byzantine: Optional[ByzantineMetrics] = None
    compression: Optional[CompressionMetrics] = None
    extra_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "benchmark": self.benchmark,
            "timestamp": self.timestamp,
            "environment": self.environment.to_dict(),
            "parameters": self.parameters.to_dict(),
            "results": {
                "timing": self.timing.to_dict(),
            },
            "metadata": self.metadata,
        }
        if self.byzantine:
            result["results"]["byzantine"] = self.byzantine.to_dict()
        if self.compression:
            result["results"]["compression"] = self.compression.to_dict()
        if self.extra_results:
            result["results"]["extra"] = self.extra_results
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved benchmark result to {filepath}")


class BenchmarkHarness:
    """
    Main benchmark harness for Mycelix FL.

    Sets up FL network with configurable nodes, injects Byzantine behavior,
    collects metrics, and generates publication-ready reports.

    Example:
        >>> harness = BenchmarkHarness(seed=42)
        >>> config = BenchmarkConfig(
        ...     name="byzantine_30pct",
        ...     num_nodes=20,
        ...     byzantine_ratio=0.3,
        ... )
        >>> result = harness.run_benchmark(config)
        >>> result.save("results/byzantine_30pct.json")
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        backend: Backend = Backend.AUTO,
    ):
        """
        Initialize benchmark harness.

        Args:
            seed: Random seed for reproducibility
            output_dir: Directory for output files
            backend: Backend to use (Rust, Python, or Auto)
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.backend = backend

        # Determine actual backend
        if backend == Backend.AUTO:
            self.active_backend = Backend.RUST if RUST_BACKEND_AVAILABLE else Backend.PYTHON
        else:
            self.active_backend = backend

        # Environment info
        self.environment = EnvironmentInfo.capture()

        # Cached components
        self._fl_system: Optional[MycelixFL] = None
        self._detector: Optional[MultiLayerByzantineDetector] = None
        self._encoder: Optional[HyperFeelEncoderV2] = None

        logger.info(f"BenchmarkHarness initialized (backend={self.active_backend.value}, seed={seed})")

    def _get_fl_system(self, config: Optional[FLConfig] = None) -> MycelixFL:
        """Get or create FL system."""
        if not MYCELIX_FL_AVAILABLE:
            raise RuntimeError("mycelix_fl not available")
        if config or self._fl_system is None:
            fl_config = config or FLConfig(
                use_detection=True,
                use_healing=True,
            )
            self._fl_system = MycelixFL(config=fl_config)
        return self._fl_system

    def _get_detector(self) -> MultiLayerByzantineDetector:
        """Get or create Byzantine detector."""
        if not MYCELIX_FL_AVAILABLE:
            raise RuntimeError("mycelix_fl not available")
        if self._detector is None:
            self._detector = MultiLayerByzantineDetector(
                enable_zkstark=False,  # Not available in benchmarks
                enable_pogq=True,
                enable_shapley=True,
                enable_hypervector=True,
                enable_self_healing=True,
            )
        return self._detector

    def _get_encoder(self) -> HyperFeelEncoderV2:
        """Get or create HyperFeel encoder."""
        if not MYCELIX_FL_AVAILABLE:
            raise RuntimeError("mycelix_fl not available")
        if self._encoder is None:
            self._encoder = HyperFeelEncoderV2()
        return self._encoder

    def generate_gradients(
        self,
        num_nodes: int,
        gradient_size: int,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic honest gradients.

        Args:
            num_nodes: Number of nodes
            gradient_size: Size of each gradient
            seed: Optional random seed

        Returns:
            Dict of node_id -> gradient array
        """
        rng = np.random.RandomState(seed or self.seed)

        # Generate a base gradient
        base_gradient = rng.randn(gradient_size).astype(np.float32)

        # Generate node gradients with small variations
        gradients = {}
        for i in range(num_nodes):
            noise = rng.randn(gradient_size).astype(np.float32) * 0.1
            gradients[f"node_{i}"] = base_gradient + noise

        return gradients

    def inject_byzantine(
        self,
        gradients: Dict[str, np.ndarray],
        byzantine_ratio: float,
        attack_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Set[str]]:
        """
        Inject Byzantine behavior into gradients.

        Args:
            gradients: Original gradients
            byzantine_ratio: Fraction of nodes to make Byzantine
            attack_types: Types of attacks to use
            seed: Optional random seed

        Returns:
            (modified_gradients, byzantine_node_ids)
        """
        if not MYCELIX_FL_AVAILABLE:
            # Fallback: simple random noise attack
            return self._inject_byzantine_fallback(gradients, byzantine_ratio, seed)

        rng = np.random.RandomState(seed or self.seed)
        node_ids = list(gradients.keys())
        num_byzantine = int(len(node_ids) * byzantine_ratio)

        if num_byzantine == 0:
            return gradients, set()

        # Select Byzantine nodes
        byzantine_ids = set(rng.choice(node_ids, num_byzantine, replace=False))

        # Create attack orchestrator
        orchestrator = AttackOrchestrator(random_seed=seed or self.seed)

        # Available attack types
        available_attacks = attack_types or [
            "gradient_scaling", "sign_flip", "gaussian_noise", "label_flip"
        ]

        # Register attackers
        for node_id in byzantine_ids:
            attack_type_str = rng.choice(available_attacks)
            attack = create_attack(AttackType(attack_type_str))
            orchestrator.register_attacker(node_id, attack)

        # Apply attacks
        modified, _ = orchestrator.apply_attacks(gradients)

        return modified, byzantine_ids

    def _inject_byzantine_fallback(
        self,
        gradients: Dict[str, np.ndarray],
        byzantine_ratio: float,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Set[str]]:
        """Fallback Byzantine injection without mycelix_fl."""
        rng = np.random.RandomState(seed or self.seed)
        node_ids = list(gradients.keys())
        num_byzantine = int(len(node_ids) * byzantine_ratio)

        if num_byzantine == 0:
            return gradients, set()

        byzantine_ids = set(rng.choice(node_ids, num_byzantine, replace=False))
        modified = dict(gradients)

        for node_id in byzantine_ids:
            grad = gradients[node_id]
            attack_type = rng.choice(["scale", "flip", "noise"])

            if attack_type == "scale":
                modified[node_id] = grad * rng.uniform(5, 100)
            elif attack_type == "flip":
                modified[node_id] = -grad
            else:
                modified[node_id] = grad + rng.randn(*grad.shape) * 10

        return modified, byzantine_ids

    def run_timed(
        self,
        func: Callable[[], Any],
        warmup_rounds: int = 2,
        num_rounds: int = 5,
    ) -> TimingMetrics:
        """
        Run a function multiple times and collect timing metrics.

        Args:
            func: Function to benchmark
            warmup_rounds: Number of warmup rounds
            num_rounds: Number of measurement rounds

        Returns:
            TimingMetrics with all samples
        """
        # Warmup
        for _ in range(warmup_rounds):
            func()
            gc.collect()

        # Measure
        metrics = TimingMetrics()
        for _ in range(num_rounds):
            gc.collect()
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            metrics.samples_ms.append(elapsed)

        return metrics

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a complete benchmark with the given configuration.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Running benchmark: {config.name}")
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Generate gradients
        gradients = self.generate_gradients(
            num_nodes=config.num_nodes,
            gradient_size=config.gradient_size,
            seed=config.seed,
        )

        # Inject Byzantine if needed
        modified_gradients, byzantine_ids = self.inject_byzantine(
            gradients=gradients,
            byzantine_ratio=config.byzantine_ratio,
            attack_types=config.attack_types,
            seed=config.seed,
        )

        # Get FL system
        fl_system = self._get_fl_system()

        # Run timed benchmark
        round_counter = [0]

        def run_round():
            round_counter[0] += 1
            return fl_system.execute_round(modified_gradients, round_num=round_counter[0])

        timing = self.run_timed(
            func=run_round,
            warmup_rounds=config.warmup_rounds,
            num_rounds=config.num_rounds,
        )

        # Collect Byzantine metrics if applicable
        byzantine_metrics = None
        if config.byzantine_ratio > 0:
            result = fl_system.execute_round(modified_gradients, round_num=round_counter[0] + 1)
            detected = result.byzantine_nodes

            tp = len(detected & byzantine_ids)
            fp = len(detected - byzantine_ids)
            fn = len(byzantine_ids - detected)
            tn = config.num_nodes - tp - fp - fn

            byzantine_metrics = ByzantineMetrics(
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
            )

        return BenchmarkResult(
            benchmark=config.name,
            timestamp=timestamp,
            environment=self.environment,
            parameters=config,
            timing=timing,
            byzantine=byzantine_metrics,
            metadata={
                "backend_used": self.active_backend.value,
                "total_gradients": len(gradients),
                "byzantine_count": len(byzantine_ids) if byzantine_ids else 0,
            },
        )

    def run_latency_benchmark(
        self,
        num_nodes: int = 10,
        gradient_size: int = 10000,
        num_samples: int = 100,
    ) -> BenchmarkResult:
        """
        Run latency distribution benchmark.

        Collects many samples for accurate percentile estimation.
        """
        config = BenchmarkConfig(
            name="latency_distribution",
            description="Measure p50, p95, p99 aggregation latency",
            num_nodes=num_nodes,
            gradient_size=gradient_size,
            num_rounds=num_samples,
            warmup_rounds=5,
            seed=self.seed,
        )

        return self.run_benchmark(config)

    def run_scalability_benchmark(
        self,
        node_counts: List[int],
        gradient_size: int = 10000,
        num_rounds: int = 10,
    ) -> List[BenchmarkResult]:
        """
        Run scalability benchmark with varying node counts.

        Args:
            node_counts: List of node counts to test
            gradient_size: Size of gradients
            num_rounds: Rounds per configuration

        Returns:
            List of BenchmarkResult for each node count
        """
        results = []

        for num_nodes in node_counts:
            config = BenchmarkConfig(
                name=f"scalability_{num_nodes}nodes",
                description=f"Scalability test with {num_nodes} nodes",
                num_nodes=num_nodes,
                gradient_size=gradient_size,
                num_rounds=num_rounds,
                warmup_rounds=3,
                seed=self.seed,
            )
            result = self.run_benchmark(config)
            results.append(result)

            logger.info(
                f"Scalability {num_nodes} nodes: "
                f"{result.timing.mean_ms:.2f}ms (p99: {result.timing.p99_ms:.2f}ms)"
            )

        return results

    def run_byzantine_benchmark(
        self,
        byzantine_ratios: List[float],
        num_nodes: int = 20,
        gradient_size: int = 10000,
        num_rounds: int = 20,
    ) -> List[BenchmarkResult]:
        """
        Run Byzantine detection benchmark at various adversarial ratios.

        Args:
            byzantine_ratios: List of Byzantine ratios to test
            num_nodes: Number of nodes
            gradient_size: Size of gradients
            num_rounds: Rounds per configuration

        Returns:
            List of BenchmarkResult for each ratio
        """
        results = []

        for ratio in byzantine_ratios:
            config = BenchmarkConfig(
                name=f"byzantine_{int(ratio * 100)}pct",
                description=f"Byzantine detection at {ratio:.0%} adversarial ratio",
                num_nodes=num_nodes,
                gradient_size=gradient_size,
                num_rounds=num_rounds,
                warmup_rounds=3,
                byzantine_ratio=ratio,
                attack_types=["gradient_scaling", "sign_flip", "gaussian_noise"],
                seed=self.seed,
            )
            result = self.run_benchmark(config)
            results.append(result)

            if result.byzantine:
                logger.info(
                    f"Byzantine {ratio:.0%}: "
                    f"P={result.byzantine.precision:.3f} "
                    f"R={result.byzantine.recall:.3f} "
                    f"F1={result.byzantine.f1_score:.3f}"
                )

        return results

    def run_compression_benchmark(
        self,
        gradient_sizes: List[int],
        num_samples: int = 50,
    ) -> List[BenchmarkResult]:
        """
        Run HyperFeel compression benchmark.

        Args:
            gradient_sizes: List of gradient sizes to test
            num_samples: Number of samples per configuration

        Returns:
            List of BenchmarkResult for each size
        """
        results = []
        encoder = self._get_encoder()

        for size in gradient_sizes:
            self.rng = np.random.RandomState(self.seed)

            config = BenchmarkConfig(
                name=f"compression_{size}params",
                description=f"HyperFeel compression for {size} parameters",
                gradient_size=size,
                num_rounds=num_samples,
                warmup_rounds=5,
                seed=self.seed,
            )

            # Generate test gradient
            gradient = self.rng.randn(size).astype(np.float32)
            original_bytes = gradient.nbytes

            # Measure encoding
            encoding_times = []
            decoding_times = []
            similarities = []

            for i in range(config.warmup_rounds + num_samples):
                # Encode
                start = time.perf_counter()
                hg = encoder.encode_gradient(gradient, round_num=i, node_id="bench")
                encode_time = (time.perf_counter() - start) * 1000

                # Decode
                start = time.perf_counter()
                decoded = encoder.decode_hypervector(hg.hypervector, len(gradient))
                decode_time = (time.perf_counter() - start) * 1000

                # Quality: cosine similarity
                sim = np.dot(gradient, decoded) / (
                    np.linalg.norm(gradient) * np.linalg.norm(decoded) + 1e-10
                )

                if i >= config.warmup_rounds:
                    encoding_times.append(encode_time)
                    decoding_times.append(decode_time)
                    similarities.append(float(sim))

            compression_metrics = CompressionMetrics(
                original_bytes=original_bytes,
                compressed_bytes=len(hg.hypervector),
                compression_ratio=original_bytes / len(hg.hypervector),
                encoding_time_ms=float(np.mean(encoding_times)),
                decoding_time_ms=float(np.mean(decoding_times)),
                cosine_similarity=float(np.mean(similarities)),
            )

            timing = TimingMetrics(samples_ms=encoding_times)

            result = BenchmarkResult(
                benchmark=config.name,
                timestamp=datetime.utcnow().isoformat() + "Z",
                environment=self.environment,
                parameters=config,
                timing=timing,
                compression=compression_metrics,
                extra_results={
                    "decoding_times_ms": decoding_times,
                    "similarities": similarities,
                },
                metadata={
                    "encoder_version": "v2.0",
                    "hypervector_bytes": len(hg.hypervector),
                },
            )
            results.append(result)

            logger.info(
                f"Compression {size}: "
                f"{compression_metrics.compression_ratio:.0f}x ratio, "
                f"{compression_metrics.cosine_similarity:.4f} similarity"
            )

        return results

    def generate_summary_report(
        self,
        results: List[BenchmarkResult],
        output_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generate summary report from benchmark results.

        Args:
            results: List of benchmark results
            output_file: Optional file to save report

        Returns:
            Summary report dictionary
        """
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "environment": self.environment.to_dict(),
            "num_benchmarks": len(results),
            "benchmarks": {},
        }

        for result in results:
            report["benchmarks"][result.benchmark] = {
                "timing": {
                    "mean_ms": result.timing.mean_ms,
                    "p50_ms": result.timing.p50_ms,
                    "p95_ms": result.timing.p95_ms,
                    "p99_ms": result.timing.p99_ms,
                },
            }
            if result.byzantine:
                report["benchmarks"][result.benchmark]["byzantine"] = {
                    "precision": result.byzantine.precision,
                    "recall": result.byzantine.recall,
                    "f1_score": result.byzantine.f1_score,
                }
            if result.compression:
                report["benchmarks"][result.benchmark]["compression"] = {
                    "ratio": result.compression.compression_ratio,
                    "quality": result.compression.cosine_similarity,
                }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved summary report to {output_path}")

        return report


def create_harness(
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> BenchmarkHarness:
    """
    Factory function to create a benchmark harness.

    Args:
        seed: Random seed for reproducibility
        output_dir: Output directory for results

    Returns:
        Configured BenchmarkHarness
    """
    return BenchmarkHarness(seed=seed, output_dir=output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Mycelix Benchmark Harness")
    print("=" * 60)

    harness = create_harness(seed=42)
    print(f"Backend: {harness.active_backend.value}")
    print(f"Environment: {harness.environment.hostname}")
    print(f"mycelix_fl available: {MYCELIX_FL_AVAILABLE}")
    print(f"Rust backend available: {RUST_BACKEND_AVAILABLE}")

    if MYCELIX_FL_AVAILABLE:
        print("\nRunning quick test...")
        config = BenchmarkConfig(
            name="quick_test",
            num_nodes=5,
            gradient_size=1000,
            num_rounds=3,
        )
        result = harness.run_benchmark(config)
        print(f"Result: {result.timing.mean_ms:.2f}ms mean latency")
