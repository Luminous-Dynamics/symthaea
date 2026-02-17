"""
MycelixFL Benchmark Suite

Author: Luminous Dynamics
Date: December 31, 2025
"""

from mycelix_fl.benchmarks.suite import (
    BenchmarkSuite,
    BenchmarkResult,
    run_standard_benchmarks,
    run_scalability_test,
    run_byzantine_detection_benchmark,
    compare_backends,
)

__all__ = [
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_standard_benchmarks",
    "run_scalability_test",
    "run_byzantine_detection_benchmark",
    "compare_backends",
]
