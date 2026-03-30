# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Benchmark Suite

Publication-ready benchmarking for Mycelix Federated Learning.

Scenarios:
    - Byzantine Detection: Tests detection rates at 10-45% adversarial ratios
    - Latency Distribution: Measures p50, p95, p99 aggregation latency
    - Scalability: Tests with 10-500 nodes
    - Compression: Measures HyperFeel compression ratios and quality

Usage:
    python benchmarks/run_benchmarks.py --all
    python benchmarks/run_benchmarks.py --byzantine
    python benchmarks/run_benchmarks.py --latency --detailed
    python benchmarks/run_benchmarks.py --scalability
    python benchmarks/run_benchmarks.py --compression

Author: Luminous Dynamics
Date: January 2026
"""

from .harness import (
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkResult,
    TimingMetrics,
    ByzantineMetrics,
    CompressionMetrics,
    Backend,
    create_harness,
)

__all__ = [
    "BenchmarkHarness",
    "BenchmarkConfig",
    "BenchmarkResult",
    "TimingMetrics",
    "ByzantineMetrics",
    "CompressionMetrics",
    "Backend",
    "create_harness",
]

__version__ = "1.0.0"
