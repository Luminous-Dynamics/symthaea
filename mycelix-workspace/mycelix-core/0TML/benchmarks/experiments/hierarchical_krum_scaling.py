#!/usr/bin/env python3
"""
Experiment: Krum vs Hierarchical Krum Scaling

Purpose:
    Compare wall-clock aggregation time for standard Krum vs hierarchical
    Krum as the number of participating clients grows.

Usage:
    python benchmarks/experiments/hierarchical_krum_scaling.py

Notes:
    - This benchmark uses synthetic gradients (Gaussian noise) and uniform
      reputations. It is intended to measure aggregation overhead only,
      not model quality.
    - Run inside the 0TML Nix shell so that numpy and the zerotrustml
      package are available.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is on the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from zerotrustml.aggregation import aggregate_gradients  # type: ignore


def benchmark_once(
    n_clients: int,
    dim: int = 1024,
    num_byzantine: int | None = None,
    branching_factor: int = 10,
) -> tuple[float, float]:
    """
    Run a single timing comparison for a given number of clients.

    Returns:
        (t_krum_ms, t_hierarchical_ms)
    """
    if num_byzantine is None:
        num_byzantine = max(0, n_clients // 3)

    gradients = [np.random.randn(dim).astype(np.float32) for _ in range(n_clients)]
    reputations = [1.0] * n_clients

    # Standard Krum
    t0 = time.perf_counter()
    _ = aggregate_gradients(
        gradients,
        reputations,
        algorithm="krum",
        num_byzantine=num_byzantine,
    )
    t_krum_ms = (time.perf_counter() - t0) * 1000.0

    # Hierarchical Krum
    t0 = time.perf_counter()
    _ = aggregate_gradients(
        gradients,
        reputations,
        algorithm="hierarchical_krum",
        branching_factor=branching_factor,
        num_byzantine=num_byzantine,
    )
    t_hierarchical_ms = (time.perf_counter() - t0) * 1000.0

    return t_krum_ms, t_hierarchical_ms


def main() -> None:
    client_counts = [20, 50, 100, 200, 500, 1000]
    dim = 2048
    branching_factor = 10
    repetitions = 5

    print("Hierarchical Krum Scaling Benchmark")
    print("===================================")
    print(f"Gradient dimension: {dim}")
    print(f"Branching factor:   {branching_factor}")
    print(f"Repetitions:        {repetitions}")
    print()
    print(f"{'N':>6}  {'Krum (ms)':>12}  {'Hier-Krum (ms)':>16}  {'Speedup':>8}")

    for n in client_counts:
        krum_times = []
        hier_times = []
        for _ in range(repetitions):
            t_krum, t_hier = benchmark_once(
                n_clients=n,
                dim=dim,
                branching_factor=branching_factor,
            )
            krum_times.append(t_krum)
            hier_times.append(t_hier)

        mean_krum = float(np.mean(krum_times))
        mean_hier = float(np.mean(hier_times))
        speedup = mean_krum / mean_hier if mean_hier > 0 else float("inf")

        print(f"{n:6d}  {mean_krum:12.3f}  {mean_hier:16.3f}  {speedup:8.2f}x")


if __name__ == "__main__":
    main()

