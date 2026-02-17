"""
Benchmark Scenarios for Mycelix FL

Collection of benchmark scenarios for publication-ready results:
- Byzantine detection rates
- Latency distributions
- Scalability testing
- HyperFeel compression

Author: Luminous Dynamics
Date: January 2026
"""

from .byzantine_detection import ByzantineDetectionScenario, run_byzantine_benchmark
from .latency_distribution import LatencyDistributionScenario, run_latency_benchmark
from .scalability import ScalabilityScenario, run_scalability_benchmark
from .compression import CompressionScenario, run_compression_benchmark

__all__ = [
    "ByzantineDetectionScenario",
    "run_byzantine_benchmark",
    "LatencyDistributionScenario",
    "run_latency_benchmark",
    "ScalabilityScenario",
    "run_scalability_benchmark",
    "CompressionScenario",
    "run_compression_benchmark",
]
