# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
HyperFeel Compression Benchmark Scenario

Measures HyperFeel gradient compression performance:
- Compression ratios (target: 2000x)
- Encoding/decoding latency
- Reconstruction quality (cosine similarity)
- Scalability with gradient size

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

# HyperFeel constants
HV16_BYTES = 2048  # Target compressed size


@dataclass
class CompressionConfig:
    """Configuration for compression benchmark."""
    gradient_sizes: List[int] = field(
        default_factory=lambda: [
            1_000,       # 1K params (4KB)
            10_000,      # 10K params (40KB)
            100_000,     # 100K params (400KB)
            1_000_000,   # 1M params (4MB)
            10_000_000,  # 10M params (40MB)
        ]
    )
    num_samples: int = 50
    warmup_samples: int = 10
    seed: int = 42
    test_reconstruction: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gradient_sizes": self.gradient_sizes,
            "num_samples": self.num_samples,
            "warmup_samples": self.warmup_samples,
            "seed": self.seed,
            "test_reconstruction": self.test_reconstruction,
        }


@dataclass
class CompressionResult:
    """Result for a single gradient size test."""
    gradient_size: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mean_encode_ms: float
    std_encode_ms: float
    p95_encode_ms: float
    mean_decode_ms: float
    std_decode_ms: float
    p95_decode_ms: float
    mean_cosine_similarity: float
    std_cosine_similarity: float
    mean_mse: float
    throughput_mb_per_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gradient_size": self.gradient_size,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "encoding": {
                "mean_ms": self.mean_encode_ms,
                "std_ms": self.std_encode_ms,
                "p95_ms": self.p95_encode_ms,
            },
            "decoding": {
                "mean_ms": self.mean_decode_ms,
                "std_ms": self.std_decode_ms,
                "p95_ms": self.p95_decode_ms,
            },
            "quality": {
                "mean_cosine_similarity": self.mean_cosine_similarity,
                "std_cosine_similarity": self.std_cosine_similarity,
                "mean_mse": self.mean_mse,
            },
            "throughput_mb_per_sec": self.throughput_mb_per_sec,
        }


class CompressionScenario:
    """
    HyperFeel compression benchmark scenario.

    Tests compression performance across various gradient sizes.

    Example:
        >>> scenario = CompressionScenario()
        >>> results = scenario.run()
        >>> for r in results:
        ...     print(f"{r.gradient_size}: {r.compression_ratio:.0f}x, sim={r.mean_cosine_similarity:.4f}")
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or CompressionConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Try importing mycelix_fl
        try:
            from mycelix_fl import HyperFeelEncoderV2, EncodingConfig
            self.HyperFeelEncoderV2 = HyperFeelEncoderV2
            self.EncodingConfig = EncodingConfig
            self.available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl HyperFeel not available: {e}")
            self.available = False

    def generate_gradient(self, size: int) -> np.ndarray:
        """Generate a realistic gradient."""
        # Use a mixture of Gaussian distributions to simulate
        # realistic gradient statistics (some layers have larger gradients)
        gradient = np.zeros(size, dtype=np.float32)

        # Divide into "layers" with different scales
        num_layers = max(1, size // 1000)
        layer_size = size // num_layers

        for i in range(num_layers):
            start = i * layer_size
            end = min(start + layer_size, size)
            scale = self.rng.exponential(0.1)  # Varying scales
            gradient[start:end] = self.rng.randn(end - start).astype(np.float32) * scale

        return gradient

    def run_single_size(self, gradient_size: int) -> CompressionResult:
        """Run benchmark for a single gradient size."""
        logger.info(f"Testing gradient size: {gradient_size:,} ({gradient_size * 4 / 1e6:.2f} MB)")

        # Reset RNG for reproducibility
        self.rng = np.random.RandomState(self.config.seed + gradient_size)

        gradient = self.generate_gradient(gradient_size)
        original_bytes = gradient.nbytes

        if not self.available:
            return self._run_fallback(gradient, gradient_size)

        encoder = self.HyperFeelEncoderV2()

        # Warmup
        for i in range(self.config.warmup_samples):
            hg = encoder.encode_gradient(gradient, round_num=i, node_id="bench")
            _ = encoder.decode_hypervector(hg.hypervector, len(gradient))
            gc.collect()

        # Benchmark
        encode_times = []
        decode_times = []
        similarities = []
        mses = []

        for i in range(self.config.num_samples):
            gc.collect()

            # Encode
            start = time.perf_counter()
            hg = encoder.encode_gradient(gradient, round_num=self.config.warmup_samples + i, node_id="bench")
            encode_time = (time.perf_counter() - start) * 1000
            encode_times.append(encode_time)

            # Decode
            start = time.perf_counter()
            decoded = encoder.decode_hypervector(hg.hypervector, len(gradient))
            decode_time = (time.perf_counter() - start) * 1000
            decode_times.append(decode_time)

            # Quality metrics
            if self.config.test_reconstruction:
                # Cosine similarity
                dot = np.dot(gradient, decoded)
                norm_orig = np.linalg.norm(gradient)
                norm_dec = np.linalg.norm(decoded)
                if norm_orig > 1e-10 and norm_dec > 1e-10:
                    sim = dot / (norm_orig * norm_dec)
                else:
                    sim = 0.0
                similarities.append(float(sim))

                # MSE (normalized)
                mse = np.mean((gradient / (norm_orig + 1e-10) - decoded / (norm_dec + 1e-10)) ** 2)
                mses.append(float(mse))

        compressed_bytes = len(hg.hypervector)
        compression_ratio = original_bytes / compressed_bytes

        # Throughput: MB of original data processed per second
        mean_encode_ms = float(np.mean(encode_times))
        throughput = (original_bytes / 1e6) / (mean_encode_ms / 1000) if mean_encode_ms > 0 else 0

        return CompressionResult(
            gradient_size=gradient_size,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            mean_encode_ms=mean_encode_ms,
            std_encode_ms=float(np.std(encode_times)),
            p95_encode_ms=float(np.percentile(encode_times, 95)),
            mean_decode_ms=float(np.mean(decode_times)),
            std_decode_ms=float(np.std(decode_times)),
            p95_decode_ms=float(np.percentile(decode_times, 95)),
            mean_cosine_similarity=float(np.mean(similarities)) if similarities else 0.0,
            std_cosine_similarity=float(np.std(similarities)) if similarities else 0.0,
            mean_mse=float(np.mean(mses)) if mses else 0.0,
            throughput_mb_per_sec=throughput,
        )

    def _run_fallback(
        self,
        gradient: np.ndarray,
        gradient_size: int,
    ) -> CompressionResult:
        """Fallback compression using random projection."""
        original_bytes = gradient.nbytes
        output_dim = HV16_BYTES * 8  # Binary hypervector bits

        # Create random projection matrix
        proj = self.rng.randn(len(gradient), output_dim // 8).astype(np.float32)
        proj /= np.sqrt(output_dim // 8)

        encode_times = []
        decode_times = []
        similarities = []
        mses = []

        for i in range(self.config.num_samples):
            gc.collect()

            # Encode: project and binarize
            start = time.perf_counter()
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                normalized = gradient / grad_norm
            else:
                normalized = gradient
            projected = normalized @ proj
            binary = (projected > 0).astype(np.uint8)
            compressed = np.packbits(binary).tobytes()
            encode_time = (time.perf_counter() - start) * 1000
            encode_times.append(encode_time)

            # Decode: unpack and project back
            start = time.perf_counter()
            unpacked = np.unpackbits(np.frombuffer(compressed, dtype=np.uint8))
            unpacked_float = unpacked[:proj.shape[1]].astype(np.float32) * 2 - 1
            decoded = unpacked_float @ proj.T
            decoded_norm = np.linalg.norm(decoded)
            if decoded_norm > 1e-10:
                decoded = decoded / decoded_norm
            decode_time = (time.perf_counter() - start) * 1000
            decode_times.append(decode_time)

            # Quality
            if self.config.test_reconstruction:
                dot = np.dot(normalized, decoded)
                sim = float(np.clip(dot, -1, 1))
                similarities.append(sim)

                mse = np.mean((normalized - decoded) ** 2)
                mses.append(float(mse))

        compressed_bytes = HV16_BYTES
        compression_ratio = original_bytes / compressed_bytes
        mean_encode_ms = float(np.mean(encode_times))
        throughput = (original_bytes / 1e6) / (mean_encode_ms / 1000) if mean_encode_ms > 0 else 0

        return CompressionResult(
            gradient_size=gradient_size,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            mean_encode_ms=mean_encode_ms,
            std_encode_ms=float(np.std(encode_times)),
            p95_encode_ms=float(np.percentile(encode_times, 95)),
            mean_decode_ms=float(np.mean(decode_times)),
            std_decode_ms=float(np.std(decode_times)),
            p95_decode_ms=float(np.percentile(decode_times, 95)),
            mean_cosine_similarity=float(np.mean(similarities)) if similarities else 0.0,
            std_cosine_similarity=float(np.std(similarities)) if similarities else 0.0,
            mean_mse=float(np.mean(mses)) if mses else 0.0,
            throughput_mb_per_sec=throughput,
        )

    def run(self) -> List[CompressionResult]:
        """Run compression benchmark for all configured sizes."""
        logger.info(f"Running compression benchmark for {len(self.config.gradient_sizes)} sizes")

        results = []
        for size in self.config.gradient_sizes:
            result = self.run_single_size(size)
            results.append(result)

            logger.info(
                f"  {size:>10,}: "
                f"{result.compression_ratio:>6.0f}x ratio, "
                f"sim={result.mean_cosine_similarity:.4f}, "
                f"{result.throughput_mb_per_sec:.1f} MB/s"
            )

        return results

    def to_benchmark_output(self, results: List[CompressionResult]) -> Dict[str, Any]:
        """Convert results to standard benchmark output format."""
        import socket
        import platform

        # Compute summary statistics
        compression_ratios = [r.compression_ratio for r in results]
        similarities = [r.mean_cosine_similarity for r in results]
        throughputs = [r.throughput_mb_per_sec for r in results]

        return {
            "benchmark": "compression",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            "parameters": self.config.to_dict(),
            "results": {
                "by_size": {
                    str(r.gradient_size): r.to_dict()
                    for r in results
                },
                "summary": {
                    "min_compression_ratio": min(compression_ratios),
                    "max_compression_ratio": max(compression_ratios),
                    "mean_compression_ratio": float(np.mean(compression_ratios)),
                    "min_similarity": min(similarities) if similarities else 0.0,
                    "max_similarity": max(similarities) if similarities else 0.0,
                    "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
                    "mean_throughput_mb_per_sec": float(np.mean(throughputs)),
                    "target_bytes": HV16_BYTES,
                },
            },
            "metadata": {
                "mycelix_available": self.available,
                "sizes_tested": len(results),
            },
        }


def run_compression_benchmark(
    output_dir: Optional[Path] = None,
    config: Optional[CompressionConfig] = None,
) -> Dict[str, Any]:
    """
    Run complete compression benchmark.

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
    logger.info("HyperFeel Compression Benchmark")
    logger.info("=" * 60)

    scenario = CompressionScenario(config=config)
    results = scenario.run()
    output = scenario.to_benchmark_output(results)

    if output_dir:
        output_path = Path(output_dir) / "compression.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HyperFeel Compression Benchmark")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 10000, 100000, 1000000],
        help="Gradient sizes to test",
    )
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = CompressionConfig(
        gradient_sizes=args.sizes,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    output = run_compression_benchmark(
        output_dir=args.output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean compression ratio: {output['results']['summary']['mean_compression_ratio']:.0f}x")
    print(f"  Mean quality (similarity): {output['results']['summary']['mean_similarity']:.4f}")
    print(f"  Mean throughput: {output['results']['summary']['mean_throughput_mb_per_sec']:.1f} MB/s")
    print()
    for size, data in output["results"]["by_size"].items():
        print(
            f"  {int(size):>10,} params: "
            f"{data['compression_ratio']:>6.0f}x, "
            f"sim={data['quality']['mean_cosine_similarity']:.4f}"
        )
