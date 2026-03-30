#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Benchmark Suite - Main Entry Point

Runs all benchmarks and generates publication-ready results.

Usage:
    python benchmarks/run_benchmarks.py --all
    python benchmarks/run_benchmarks.py --byzantine
    python benchmarks/run_benchmarks.py --latency --detailed
    python benchmarks/run_benchmarks.py --scalability --node-counts 10 50 100
    python benchmarks/run_benchmarks.py --compression

Output:
    benchmarks/results/{timestamp}/
        byzantine_detection.json
        latency_distribution.json
        scalability.json
        compression.json
        summary.json

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure parent paths are available
BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "src"))

# Import scenarios
from scenarios.byzantine_detection import (
    ByzantineDetectionScenario,
    ByzantineDetectionConfig,
    run_byzantine_benchmark,
)
from scenarios.latency_distribution import (
    LatencyDistributionScenario,
    LatencyDistributionConfig,
    run_latency_benchmark,
)
from scenarios.scalability import (
    ScalabilityScenario,
    ScalabilityConfig,
    run_scalability_benchmark,
)
from scenarios.compression import (
    CompressionScenario,
    CompressionConfig,
    run_compression_benchmark,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    import numpy as np

    env_info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
    }

    # Check for GPU
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env_info["torch_available"] = False

    # Check for mycelix_fl
    try:
        import mycelix_fl
        env_info["mycelix_fl_version"] = mycelix_fl.__version__
        env_info["rust_backend"] = mycelix_fl.has_rust_backend()
    except ImportError:
        env_info["mycelix_fl_available"] = False

    return env_info


def create_output_dir(base_dir: Optional[Path] = None) -> Path:
    """Create timestamped output directory."""
    if base_dir is None:
        base_dir = BENCHMARKS_DIR / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    return output_dir


def run_all_benchmarks(
    output_dir: Path,
    seed: int = 42,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run all benchmark scenarios.

    Args:
        output_dir: Directory for output files
        seed: Random seed for reproducibility
        quick: If True, use reduced sample sizes for faster execution

    Returns:
        Summary of all benchmark results
    """
    logger.info("=" * 70)
    logger.info("MYCELIX BENCHMARK SUITE")
    logger.info("=" * 70)

    results = {}
    start_time = datetime.now()

    # Collect environment info
    env_info = get_environment_info()
    env_path = output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)
    logger.info(f"Environment info saved to {env_path}")

    # 1. Byzantine Detection Benchmark
    logger.info("\n" + "=" * 70)
    logger.info("1. BYZANTINE DETECTION BENCHMARK")
    logger.info("=" * 70)

    byzantine_config = ByzantineDetectionConfig(
        byzantine_ratios=[0.10, 0.20, 0.30, 0.40, 0.45],
        num_nodes=20 if not quick else 10,
        num_rounds=20 if not quick else 5,
        seed=seed,
    )
    byzantine_results = run_byzantine_benchmark(
        output_dir=output_dir,
        config=byzantine_config,
    )
    results["byzantine_detection"] = byzantine_results

    # 2. Latency Distribution Benchmark
    logger.info("\n" + "=" * 70)
    logger.info("2. LATENCY DISTRIBUTION BENCHMARK")
    logger.info("=" * 70)

    latency_config = LatencyDistributionConfig(
        num_nodes=10,
        gradient_size=10000,
        num_samples=1000 if not quick else 100,
        seed=seed,
    )
    latency_results = run_latency_benchmark(
        output_dir=output_dir,
        config=latency_config,
        detailed=True,
    )
    results["latency_distribution"] = latency_results

    # 3. Scalability Benchmark
    logger.info("\n" + "=" * 70)
    logger.info("3. SCALABILITY BENCHMARK")
    logger.info("=" * 70)

    scalability_config = ScalabilityConfig(
        node_counts=[10, 50, 100, 500] if not quick else [10, 50, 100],
        num_rounds=20 if not quick else 5,
        seed=seed,
    )
    scalability_results = run_scalability_benchmark(
        output_dir=output_dir,
        config=scalability_config,
    )
    results["scalability"] = scalability_results

    # 4. Compression Benchmark
    logger.info("\n" + "=" * 70)
    logger.info("4. COMPRESSION BENCHMARK")
    logger.info("=" * 70)

    compression_config = CompressionConfig(
        gradient_sizes=[1000, 10000, 100000, 1000000] if not quick else [1000, 10000, 100000],
        num_samples=50 if not quick else 10,
        seed=seed,
    )
    compression_results = run_compression_benchmark(
        output_dir=output_dir,
        config=compression_config,
    )
    results["compression"] = compression_results

    # Generate summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary = {
        "benchmark_suite": "mycelix",
        "version": "1.0.0",
        "timestamp": start_time.isoformat() + "Z",
        "duration_seconds": duration,
        "environment": env_info,
        "benchmarks_run": list(results.keys()),
        "results_summary": {
            "byzantine_detection": {
                "ratios_tested": byzantine_config.byzantine_ratios,
                "mean_f1": results["byzantine_detection"]["results"]["summary"]["mean_f1"],
            },
            "latency": {
                "p50_ms": results["latency_distribution"]["results"]["total"]["p50_ms"],
                "p95_ms": results["latency_distribution"]["results"]["total"]["p95_ms"],
                "p99_ms": results["latency_distribution"]["results"]["total"]["p99_ms"],
            },
            "scalability": {
                "complexity": results["scalability"]["results"]["analysis"]["complexity_estimate"],
                "scaling_coefficient": results["scalability"]["results"]["analysis"]["scaling_coefficient"],
            },
            "compression": {
                "mean_ratio": results["compression"]["results"]["summary"]["mean_compression_ratio"],
                "mean_quality": results["compression"]["results"]["summary"]["mean_similarity"],
            },
        },
        "output_directory": str(output_dir),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUITE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {output_dir}")
    logger.info("\nKey Results:")
    logger.info(f"  Byzantine Detection:")
    logger.info(f"    Mean F1 Score: {summary['results_summary']['byzantine_detection']['mean_f1']:.3f}")
    logger.info(f"  Latency:")
    logger.info(f"    p50: {summary['results_summary']['latency']['p50_ms']:.2f}ms")
    logger.info(f"    p99: {summary['results_summary']['latency']['p99_ms']:.2f}ms")
    logger.info(f"  Scalability:")
    logger.info(f"    Complexity: {summary['results_summary']['scalability']['complexity']}")
    logger.info(f"  Compression:")
    logger.info(f"    Mean Ratio: {summary['results_summary']['compression']['mean_ratio']:.0f}x")
    logger.info(f"    Mean Quality: {summary['results_summary']['compression']['mean_quality']:.4f}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mycelix Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python benchmarks/run_benchmarks.py --all

    # Run only Byzantine detection benchmark
    python benchmarks/run_benchmarks.py --byzantine

    # Run latency benchmark with detailed component breakdown
    python benchmarks/run_benchmarks.py --latency --detailed

    # Run scalability with custom node counts
    python benchmarks/run_benchmarks.py --scalability --node-counts 10 50 100 200

    # Run compression benchmark
    python benchmarks/run_benchmarks.py --compression

    # Quick run (reduced samples for faster execution)
    python benchmarks/run_benchmarks.py --all --quick

    # Custom output directory
    python benchmarks/run_benchmarks.py --all --output-dir /path/to/results
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--byzantine", "-b",
        action="store_true",
        help="Run Byzantine detection benchmark",
    )
    parser.add_argument(
        "--latency", "-l",
        action="store_true",
        help="Run latency distribution benchmark",
    )
    parser.add_argument(
        "--scalability", "-s",
        action="store_true",
        help="Run scalability benchmark",
    )
    parser.add_argument(
        "--compression", "-c",
        action="store_true",
        help="Run compression benchmark",
    )

    # Configuration options
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: benchmarks/results/{timestamp})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick run with reduced samples",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Detailed component breakdown (for latency benchmark)",
    )

    # Byzantine-specific options
    parser.add_argument(
        "--byzantine-ratios",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.45],
        help="Byzantine ratios to test",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of nodes for Byzantine benchmark",
    )

    # Scalability-specific options
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500],
        help="Node counts for scalability benchmark",
    )

    # Compression-specific options
    parser.add_argument(
        "--gradient-sizes",
        type=int,
        nargs="+",
        default=[1000, 10000, 100000, 1000000],
        help="Gradient sizes for compression benchmark",
    )

    # General options
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=20,
        help="Number of rounds per configuration",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for latency benchmark",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output_dir)

    # Determine which benchmarks to run
    run_byzantine = args.byzantine or args.all
    run_latency = args.latency or args.all
    run_scalability = args.scalability or args.all
    run_compression = args.compression or args.all

    # If no specific benchmark selected, show help
    if not any([run_byzantine, run_latency, run_scalability, run_compression]):
        parser.print_help()
        return

    # Run all if --all specified
    if args.all:
        run_all_benchmarks(
            output_dir=output_dir,
            seed=args.seed,
            quick=args.quick,
        )
        return

    # Run individual benchmarks
    results = {}

    if run_byzantine:
        logger.info("Running Byzantine Detection Benchmark...")
        config = ByzantineDetectionConfig(
            byzantine_ratios=args.byzantine_ratios,
            num_nodes=args.num_nodes,
            num_rounds=args.num_rounds if not args.quick else 5,
            seed=args.seed,
        )
        results["byzantine"] = run_byzantine_benchmark(
            output_dir=output_dir,
            config=config,
        )

    if run_latency:
        logger.info("Running Latency Distribution Benchmark...")
        config = LatencyDistributionConfig(
            num_nodes=args.num_nodes if args.num_nodes != 20 else 10,
            num_samples=args.num_samples if not args.quick else 100,
            seed=args.seed,
        )
        results["latency"] = run_latency_benchmark(
            output_dir=output_dir,
            config=config,
            detailed=args.detailed,
        )

    if run_scalability:
        logger.info("Running Scalability Benchmark...")
        config = ScalabilityConfig(
            node_counts=args.node_counts,
            num_rounds=args.num_rounds if not args.quick else 5,
            seed=args.seed,
        )
        results["scalability"] = run_scalability_benchmark(
            output_dir=output_dir,
            config=config,
        )

    if run_compression:
        logger.info("Running Compression Benchmark...")
        config = CompressionConfig(
            gradient_sizes=args.gradient_sizes,
            num_samples=50 if not args.quick else 10,
            seed=args.seed,
        )
        results["compression"] = run_compression_benchmark(
            output_dir=output_dir,
            config=config,
        )

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARKS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")

    for benchmark_name, result in results.items():
        logger.info(f"\n{benchmark_name}:")
        if "summary" in result.get("results", {}):
            for key, value in result["results"]["summary"].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
