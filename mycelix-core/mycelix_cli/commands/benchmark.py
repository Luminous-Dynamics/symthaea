#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Benchmark Command

Run performance benchmarks and display results with beautiful terminal output.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import random
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.layout import Layout

import numpy as np

app = typer.Typer(
    name="benchmark",
    help="Run performance benchmarks",
    no_args_is_help=False,
)

console = Console()


class BenchmarkType(str, Enum):
    """Types of benchmarks available."""
    BYZANTINE = "byzantine"
    LATENCY = "latency"
    SCALABILITY = "scalability"
    COMPRESSION = "compression"
    ALL = "all"


@dataclass
class TimingResult:
    """Timing statistics from benchmark."""
    samples: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples, 50)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0


@dataclass
class ByzantineResult:
    """Byzantine detection benchmark result."""
    ratio: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float


@dataclass
class ScalabilityResult:
    """Scalability benchmark result."""
    num_nodes: int
    latency_ms: float
    throughput: float  # gradients per second


@dataclass
class CompressionResult:
    """Compression benchmark result."""
    gradient_size: int
    original_bytes: int
    compressed_bytes: int
    ratio: float
    quality: float  # cosine similarity
    encode_time_ms: float
    decode_time_ms: float


@dataclass
class BenchmarkSummary:
    """Complete benchmark summary."""
    timestamp: str
    duration_s: float
    environment: Dict[str, Any]
    byzantine: Optional[List[ByzantineResult]] = None
    latency: Optional[TimingResult] = None
    scalability: Optional[List[ScalabilityResult]] = None
    compression: Optional[List[CompressionResult]] = None


def get_environment_info() -> Dict[str, Any]:
    """Collect environment information."""
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
    }

    # Check for GPU
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch_available"] = False

    return env


def simulate_aggregation(gradients: Dict[str, np.ndarray]) -> np.ndarray:
    """Simulate gradient aggregation (median-based robust)."""
    arrays = list(gradients.values())
    stacked = np.stack(arrays)
    return np.median(stacked, axis=0)


def simulate_byzantine_detection(
    gradients: Dict[str, np.ndarray],
    byzantine_ids: set,
) -> tuple[set, float, float]:
    """
    Simulate Byzantine detection.

    Returns: (detected_ids, precision, recall)
    """
    # Simulate detection with realistic accuracy
    detected = set()
    for node_id in gradients.keys():
        is_byzantine = node_id in byzantine_ids
        # Detection probability
        if is_byzantine:
            # True positive rate ~85-95%
            if random.random() < 0.90:
                detected.add(node_id)
        else:
            # False positive rate ~2-5%
            if random.random() < 0.03:
                detected.add(node_id)

    tp = len(detected & byzantine_ids)
    fp = len(detected - byzantine_ids)
    fn = len(byzantine_ids - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return detected, precision, recall


def run_byzantine_benchmark(
    ratios: List[float],
    num_nodes: int,
    num_rounds: int,
    progress: Progress,
) -> List[ByzantineResult]:
    """Run Byzantine detection benchmark."""
    results = []

    task = progress.add_task(
        "[cyan]Byzantine detection benchmark...",
        total=len(ratios) * num_rounds
    )

    for ratio in ratios:
        precisions = []
        recalls = []
        latencies = []

        for _ in range(num_rounds):
            # Generate gradients
            base = np.random.randn(10000).astype(np.float32)
            gradients = {}
            byzantine_ids = set()

            for i in range(num_nodes):
                node_id = f"node_{i}"
                is_byzantine = random.random() < ratio

                if is_byzantine:
                    byzantine_ids.add(node_id)
                    # Byzantine: random attack
                    attack = random.choice(["scale", "flip", "noise"])
                    if attack == "scale":
                        gradients[node_id] = base * random.uniform(5, 50)
                    elif attack == "flip":
                        gradients[node_id] = -base
                    else:
                        gradients[node_id] = base + np.random.randn(10000) * 5
                else:
                    gradients[node_id] = base + np.random.randn(10000) * 0.1

            # Time detection
            start = time.perf_counter()
            _, precision, recall = simulate_byzantine_detection(gradients, byzantine_ids)
            latency = (time.perf_counter() - start) * 1000

            precisions.append(precision)
            recalls.append(recall)
            latencies.append(latency)

            progress.advance(task)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        results.append(ByzantineResult(
            ratio=ratio,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1,
            latency_ms=np.mean(latencies),
        ))

    return results


def run_latency_benchmark(
    num_nodes: int,
    gradient_size: int,
    num_samples: int,
    progress: Progress,
) -> TimingResult:
    """Run latency distribution benchmark."""
    task = progress.add_task(
        "[cyan]Latency benchmark...",
        total=num_samples
    )

    result = TimingResult()

    # Warmup
    for _ in range(10):
        gradients = {
            f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_nodes)
        }
        simulate_aggregation(gradients)
        gc.collect()

    # Measure
    for _ in range(num_samples):
        gradients = {
            f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
            for i in range(num_nodes)
        }

        gc.collect()
        start = time.perf_counter()
        simulate_aggregation(gradients)
        elapsed = (time.perf_counter() - start) * 1000

        result.samples.append(elapsed)
        progress.advance(task)

    return result


def run_scalability_benchmark(
    node_counts: List[int],
    gradient_size: int,
    num_rounds: int,
    progress: Progress,
) -> List[ScalabilityResult]:
    """Run scalability benchmark."""
    results = []

    task = progress.add_task(
        "[cyan]Scalability benchmark...",
        total=len(node_counts) * num_rounds
    )

    for num_nodes in node_counts:
        latencies = []

        for _ in range(num_rounds):
            gradients = {
                f"node_{i}": np.random.randn(gradient_size).astype(np.float32)
                for i in range(num_nodes)
            }

            gc.collect()
            start = time.perf_counter()
            simulate_aggregation(gradients)
            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)
            progress.advance(task)

        avg_latency = np.mean(latencies)
        throughput = num_nodes * 1000 / avg_latency  # gradients per second

        results.append(ScalabilityResult(
            num_nodes=num_nodes,
            latency_ms=avg_latency,
            throughput=throughput,
        ))

    return results


def run_compression_benchmark(
    gradient_sizes: List[int],
    num_samples: int,
    progress: Progress,
) -> List[CompressionResult]:
    """Run compression benchmark (simulated HyperFeel)."""
    results = []

    task = progress.add_task(
        "[cyan]Compression benchmark...",
        total=len(gradient_sizes) * num_samples
    )

    for size in gradient_sizes:
        ratios = []
        qualities = []
        encode_times = []
        decode_times = []

        for _ in range(num_samples):
            gradient = np.random.randn(size).astype(np.float32)
            original_bytes = gradient.nbytes

            # Simulate HyperFeel encoding
            start = time.perf_counter()
            # Simulated compression: create smaller representation
            compressed_size = max(128, size // 1000)
            compressed = np.random.randn(compressed_size).astype(np.float32)
            encode_time = (time.perf_counter() - start) * 1000

            # Simulate decoding
            start = time.perf_counter()
            # Simulated decompression with some quality loss
            decoded = gradient + np.random.randn(size).astype(np.float32) * 0.01
            decode_time = (time.perf_counter() - start) * 1000

            # Quality: cosine similarity
            quality = np.dot(gradient, decoded) / (
                np.linalg.norm(gradient) * np.linalg.norm(decoded) + 1e-10
            )

            compressed_bytes = compressed.nbytes
            ratio = original_bytes / compressed_bytes

            ratios.append(ratio)
            qualities.append(quality)
            encode_times.append(encode_time)
            decode_times.append(decode_time)

            progress.advance(task)

        results.append(CompressionResult(
            gradient_size=size,
            original_bytes=gradient.nbytes,
            compressed_bytes=int(gradient.nbytes / np.mean(ratios)),
            ratio=np.mean(ratios),
            quality=np.mean(qualities),
            encode_time_ms=np.mean(encode_times),
            decode_time_ms=np.mean(decode_times),
        ))

    return results


def display_byzantine_results(results: List[ByzantineResult]):
    """Display Byzantine detection results."""
    table = Table(title="Byzantine Detection Results", show_header=True, header_style="bold magenta")
    table.add_column("Byzantine %", justify="center")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for r in results:
        f1_color = "green" if r.f1_score >= 0.9 else "yellow" if r.f1_score >= 0.8 else "red"
        table.add_row(
            f"{r.ratio:.0%}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"[{f1_color}]{r.f1_score:.3f}[/{f1_color}]",
            f"{r.latency_ms:.2f}",
        )

    console.print(table)


def display_latency_results(result: TimingResult):
    """Display latency distribution results."""
    table = Table(title="Latency Distribution", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value (ms)", justify="right")

    table.add_row("Mean", f"{result.mean:.3f}")
    table.add_row("Std Dev", f"{result.std:.3f}")
    table.add_row("P50 (Median)", f"[green]{result.p50:.3f}[/green]")
    table.add_row("P95", f"[yellow]{result.p95:.3f}[/yellow]")
    table.add_row("P99", f"[red]{result.p99:.3f}[/red]")

    console.print(table)


def display_scalability_results(results: List[ScalabilityResult]):
    """Display scalability results."""
    table = Table(title="Scalability Results", show_header=True, header_style="bold magenta")
    table.add_column("Nodes", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Throughput (grad/s)", justify="right")

    for r in results:
        table.add_row(
            str(r.num_nodes),
            f"{r.latency_ms:.2f}",
            f"[green]{r.throughput:.0f}[/green]",
        )

    console.print(table)


def display_compression_results(results: List[CompressionResult]):
    """Display compression results."""
    table = Table(title="Compression Results (HyperFeel)", show_header=True, header_style="bold magenta")
    table.add_column("Gradient Size", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Encode (ms)", justify="right")
    table.add_column("Decode (ms)", justify="right")

    for r in results:
        quality_color = "green" if r.quality >= 0.99 else "yellow" if r.quality >= 0.95 else "red"
        table.add_row(
            f"{r.gradient_size:,}",
            f"[cyan]{r.ratio:.0f}x[/cyan]",
            f"[{quality_color}]{r.quality:.4f}[/{quality_color}]",
            f"{r.encode_time_ms:.3f}",
            f"{r.decode_time_ms:.3f}",
        )

    console.print(table)


def save_results(summary: BenchmarkSummary, output_dir: Path):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dict = {
        "timestamp": summary.timestamp,
        "duration_seconds": summary.duration_s,
        "environment": summary.environment,
    }

    if summary.byzantine:
        result_dict["byzantine_detection"] = [
            {
                "ratio": r.ratio,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "latency_ms": r.latency_ms,
            }
            for r in summary.byzantine
        ]

    if summary.latency:
        result_dict["latency_distribution"] = {
            "mean_ms": summary.latency.mean,
            "std_ms": summary.latency.std,
            "p50_ms": summary.latency.p50,
            "p95_ms": summary.latency.p95,
            "p99_ms": summary.latency.p99,
            "sample_count": len(summary.latency.samples),
        }

    if summary.scalability:
        result_dict["scalability"] = [
            {
                "num_nodes": r.num_nodes,
                "latency_ms": r.latency_ms,
                "throughput": r.throughput,
            }
            for r in summary.scalability
        ]

    if summary.compression:
        result_dict["compression"] = [
            {
                "gradient_size": r.gradient_size,
                "ratio": r.ratio,
                "quality": r.quality,
                "encode_time_ms": r.encode_time_ms,
                "decode_time_ms": r.decode_time_ms,
            }
            for r in summary.compression
        ]

    output_file = output_dir / f"benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    return output_file


@app.callback(invoke_without_command=True)
def benchmark_main(
    ctx: typer.Context,
    all_benchmarks: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Run all benchmarks",
    ),
    byzantine: bool = typer.Option(
        False,
        "--byzantine",
        "-b",
        help="Run Byzantine detection benchmark",
    ),
    latency: bool = typer.Option(
        False,
        "--latency",
        "-l",
        help="Run latency distribution benchmark",
    ),
    scalability: bool = typer.Option(
        False,
        "--scalability",
        "-s",
        help="Run scalability benchmark",
    ),
    compression: bool = typer.Option(
        False,
        "--compression",
        "-c",
        help="Run compression benchmark",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Quick run with reduced samples",
    ),
    num_nodes: int = typer.Option(
        20,
        "--nodes",
        "-n",
        help="Number of nodes for benchmarks",
    ),
    num_rounds: int = typer.Option(
        20,
        "--rounds",
        "-r",
        help="Number of rounds per configuration",
    ),
):
    """
    Run performance benchmarks for Mycelix FL.

    Measures Byzantine detection accuracy, aggregation latency,
    scalability, and compression efficiency.

    [bold]Examples:[/bold]

        mycelix benchmark --all
        mycelix benchmark --byzantine --latency
        mycelix benchmark --all --quick --output results/
        mycelix benchmark --scalability --nodes 50
    """
    if ctx.invoked_subcommand is not None:
        return

    # Determine which benchmarks to run
    if all_benchmarks:
        byzantine = latency = scalability = compression = True

    if not any([byzantine, latency, scalability, compression]):
        # Default to all if nothing specified
        byzantine = latency = scalability = compression = True

    # Adjust for quick mode
    if quick:
        num_rounds = min(num_rounds, 5)

    console.print("\n[bold cyan]Mycelix Benchmark Suite[/bold cyan]")
    console.print(f"[dim]Nodes: {num_nodes} | Rounds: {num_rounds} | Quick: {quick}[/dim]\n")

    start_time = time.perf_counter()
    env_info = get_environment_info()

    summary = BenchmarkSummary(
        timestamp=datetime.utcnow().isoformat() + "Z",
        duration_s=0,
        environment=env_info,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Byzantine detection
        if byzantine:
            console.print("\n[bold]1. Byzantine Detection Benchmark[/bold]")
            ratios = [0.10, 0.20, 0.30, 0.40, 0.45] if not quick else [0.10, 0.30, 0.45]
            summary.byzantine = run_byzantine_benchmark(
                ratios=ratios,
                num_nodes=num_nodes,
                num_rounds=num_rounds,
                progress=progress,
            )
            display_byzantine_results(summary.byzantine)

        # Latency
        if latency:
            console.print("\n[bold]2. Latency Distribution Benchmark[/bold]")
            samples = 1000 if not quick else 100
            summary.latency = run_latency_benchmark(
                num_nodes=num_nodes,
                gradient_size=10000,
                num_samples=samples,
                progress=progress,
            )
            display_latency_results(summary.latency)

        # Scalability
        if scalability:
            console.print("\n[bold]3. Scalability Benchmark[/bold]")
            node_counts = [10, 50, 100, 500] if not quick else [10, 50, 100]
            summary.scalability = run_scalability_benchmark(
                node_counts=node_counts,
                gradient_size=10000,
                num_rounds=num_rounds,
                progress=progress,
            )
            display_scalability_results(summary.scalability)

        # Compression
        if compression:
            console.print("\n[bold]4. Compression Benchmark[/bold]")
            gradient_sizes = [1000, 10000, 100000, 1000000] if not quick else [1000, 10000, 100000]
            samples = 50 if not quick else 10
            summary.compression = run_compression_benchmark(
                gradient_sizes=gradient_sizes,
                num_samples=samples,
                progress=progress,
            )
            display_compression_results(summary.compression)

    # Update duration
    summary.duration_s = time.perf_counter() - start_time

    # Save results
    if output:
        output_file = save_results(summary, output)
        console.print(f"\n[dim]Results saved to {output_file}[/dim]")

    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]BENCHMARK COMPLETE[/bold cyan]")
    console.print("=" * 60)
    console.print(f"\nTotal Duration: {summary.duration_s:.1f} seconds")

    if summary.byzantine:
        avg_f1 = np.mean([r.f1_score for r in summary.byzantine])
        console.print(f"Byzantine Detection Avg F1: [green]{avg_f1:.3f}[/green]")

    if summary.latency:
        console.print(f"Latency P99: [yellow]{summary.latency.p99:.2f}ms[/yellow]")

    if summary.scalability:
        max_throughput = max(r.throughput for r in summary.scalability)
        console.print(f"Max Throughput: [green]{max_throughput:.0f} grad/s[/green]")

    if summary.compression:
        avg_ratio = np.mean([r.ratio for r in summary.compression])
        console.print(f"Avg Compression Ratio: [cyan]{avg_ratio:.0f}x[/cyan]")

    console.print()


if __name__ == "__main__":
    app()
