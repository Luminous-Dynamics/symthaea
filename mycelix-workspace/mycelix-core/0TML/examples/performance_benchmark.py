#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Performance Benchmarking for Identity DHT System

Measures performance characteristics:
- Identity creation throughput
- Query latency (cached vs uncached)
- Guardian relationship operations
- Reputation synchronization
- Recovery authorization computation
- Concurrent operation scaling

Week 5-6 Phase 7: End-to-End Testing
"""

import asyncio
import time
import statistics
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import identity modules
try:
    from zerotrustml.identity import (
        DHT_IdentityCoordinator,
        IdentityCoordinatorConfig,
        AgentType
    )
    IDENTITY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Identity modules not available: {e}")
    IDENTITY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result from a benchmark test"""
    test_name: str
    operation_count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_sec: float
    success_rate: float


class PerformanceBenchmark:
    """Performance benchmarking suite for Identity DHT system"""

    def __init__(self, dht_enabled: bool = False):
        """Initialize benchmark suite"""
        self.dht_enabled = dht_enabled
        self.results: List[BenchmarkResult] = []

        config = IdentityCoordinatorConfig(
            dht_enabled=dht_enabled,
            auto_sync_to_dht=False,  # Disable auto-sync for controlled testing
            cache_dht_queries=True,
            cache_ttl=600,
            enable_local_fallback=True
        )

        self.coordinator = DHT_IdentityCoordinator(config=config)

    async def _measure_operation(self, operation, *args, **kwargs) -> Tuple[float, bool, any]:
        """Measure single operation execution time"""
        start = time.perf_counter()
        success = True
        result = None

        try:
            result = await operation(*args, **kwargs)
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            success = False

        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, success, result

    async def _measure_batch(self, operation_name: str, operations: List) -> BenchmarkResult:
        """Measure batch of operations and compute statistics"""
        timings = []
        successes = 0

        total_start = time.perf_counter()

        for op, args, kwargs in operations:
            elapsed_ms, success, _ = await self._measure_operation(op, *args, **kwargs)
            timings.append(elapsed_ms)
            if success:
                successes += 1

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000

        # Compute statistics
        result = BenchmarkResult(
            test_name=operation_name,
            operation_count=len(operations),
            total_time_ms=total_elapsed_ms,
            avg_time_ms=statistics.mean(timings),
            min_time_ms=min(timings),
            max_time_ms=max(timings),
            median_time_ms=statistics.median(timings),
            p95_time_ms=self._percentile(timings, 95),
            p99_time_ms=self._percentile(timings, 99),
            throughput_ops_per_sec=(len(operations) / total_elapsed_ms) * 1000 if total_elapsed_ms > 0 else 0,
            success_rate=successes / len(operations) if operations else 0
        )

        self.results.append(result)
        return result

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    # ========================================
    # Benchmark Tests
    # ========================================

    async def benchmark_identity_creation(self, count: int = 100) -> BenchmarkResult:
        """Benchmark identity creation throughput"""
        logger.info(f"Benchmarking identity creation ({count} operations)...")

        operations = [
            (
                self.coordinator.create_identity,
                (),
                {
                    "participant_id": f"bench_create_{i:04d}",
                    "agent_type": AgentType.HUMAN_MEMBER
                }
            )
            for i in range(count)
        ]

        result = await self._measure_batch("Identity Creation", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_identity_query_uncached(self, count: int = 100) -> BenchmarkResult:
        """Benchmark uncached identity queries"""
        logger.info(f"Benchmarking uncached identity queries ({count} operations)...")

        # Create identities first
        for i in range(count):
            await self.coordinator.create_identity(
                participant_id=f"bench_query_uncached_{i:04d}",
                agent_type=AgentType.HUMAN_MEMBER
            )

        operations = [
            (
                self.coordinator.get_identity,
                (),
                {
                    "participant_id": f"bench_query_uncached_{i:04d}",
                    "use_cache": False
                }
            )
            for i in range(count)
        ]

        result = await self._measure_batch("Identity Query (Uncached)", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_identity_query_cached(self, count: int = 100) -> BenchmarkResult:
        """Benchmark cached identity queries"""
        logger.info(f"Benchmarking cached identity queries ({count} operations)...")

        # Create identities and prime cache
        for i in range(count):
            await self.coordinator.create_identity(
                participant_id=f"bench_query_cached_{i:04d}",
                agent_type=AgentType.HUMAN_MEMBER
            )
            # Prime cache
            await self.coordinator.get_identity(
                participant_id=f"bench_query_cached_{i:04d}",
                use_cache=True
            )

        operations = [
            (
                self.coordinator.get_identity,
                (),
                {
                    "participant_id": f"bench_query_cached_{i:04d}",
                    "use_cache": True
                }
            )
            for i in range(count)
        ]

        result = await self._measure_batch("Identity Query (Cached)", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_guardian_addition(self, count: int = 100) -> BenchmarkResult:
        """Benchmark guardian relationship creation"""
        logger.info(f"Benchmarking guardian addition ({count} operations)...")

        # Create subject and guardians
        await self.coordinator.create_identity("bench_subject", AgentType.HUMAN_MEMBER)

        for i in range(count):
            await self.coordinator.create_identity(
                participant_id=f"bench_guardian_{i:04d}",
                agent_type=AgentType.HUMAN_MEMBER
            )

        # Note: Cannot exceed total weight of 1.0, so we use small weights
        operations = [
            (
                self.coordinator.add_guardian,
                (),
                {
                    "subject_participant_id": "bench_subject",
                    "guardian_participant_id": f"bench_guardian_{i:04d}",
                    "relationship_type": "RECOVERY",
                    "weight": 0.001  # Very small weight
                }
            )
            for i in range(min(count, 100))  # Limited by weight constraint
        ]

        result = await self._measure_batch("Guardian Addition", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_recovery_authorization(self, count: int = 100) -> BenchmarkResult:
        """Benchmark recovery authorization computation"""
        logger.info(f"Benchmarking recovery authorization ({count} operations)...")

        # Create subject with 5 guardians
        await self.coordinator.create_identity("bench_recovery_subject", AgentType.HUMAN_MEMBER)

        guardians = []
        for i in range(5):
            guardian_id = f"bench_recovery_guardian_{i}"
            await self.coordinator.create_identity(guardian_id, AgentType.HUMAN_MEMBER)
            await self.coordinator.add_guardian(
                "bench_recovery_subject",
                guardian_id,
                "RECOVERY",
                0.2  # Each guardian has 20% weight
            )
            guardians.append(guardian_id)

        operations = [
            (
                self.coordinator.authorize_recovery,
                (),
                {
                    "subject_participant_id": "bench_recovery_subject",
                    "approving_guardian_ids": guardians[:3],  # 3 guardians = 60%
                    "required_threshold": 0.6
                }
            )
            for _ in range(count)
        ]

        result = await self._measure_batch("Recovery Authorization", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_fl_verification(self, count: int = 100) -> BenchmarkResult:
        """Benchmark FL identity verification"""
        logger.info(f"Benchmarking FL identity verification ({count} operations)...")

        # Create identities
        for i in range(count):
            await self.coordinator.create_identity(
                participant_id=f"bench_fl_{i:04d}",
                agent_type=AgentType.AI_AGENT
            )

        operations = [
            (
                self.coordinator.verify_identity_for_fl,
                (),
                {
                    "participant_id": f"bench_fl_{i:04d}",
                    "required_assurance": "E0"
                }
            )
            for i in range(count)
        ]

        result = await self._measure_batch("FL Verification", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_reputation_sync(self, count: int = 100) -> BenchmarkResult:
        """Benchmark reputation synchronization"""
        logger.info(f"Benchmarking reputation sync ({count} operations)...")

        # Create identities
        for i in range(count):
            await self.coordinator.create_identity(
                participant_id=f"bench_rep_{i:04d}",
                agent_type=AgentType.HUMAN_MEMBER
            )

        operations = [
            (
                self.coordinator.sync_reputation_to_dht,
                (),
                {
                    "participant_id": f"bench_rep_{i:04d}",
                    "reputation_score": 0.75 + (i % 25) / 100,
                    "score_type": "trust",
                    "metadata": {"round": i}
                }
            )
            for i in range(count)
        ]

        result = await self._measure_batch("Reputation Sync", operations)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    async def benchmark_concurrent_operations(self, concurrency: int = 10, operations_per_task: int = 10) -> BenchmarkResult:
        """Benchmark concurrent operation execution"""
        logger.info(f"Benchmarking concurrent operations ({concurrency} tasks × {operations_per_task} ops)...")

        async def worker_task(worker_id: int) -> List[float]:
            """Worker that performs multiple operations"""
            timings = []

            for i in range(operations_per_task):
                start = time.perf_counter()

                await self.coordinator.create_identity(
                    participant_id=f"bench_concurrent_{worker_id:02d}_{i:02d}",
                    agent_type=AgentType.HUMAN_MEMBER
                )

                elapsed_ms = (time.perf_counter() - start) * 1000
                timings.append(elapsed_ms)

            return timings

        # Execute workers concurrently
        total_start = time.perf_counter()

        tasks = [worker_task(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000

        # Flatten all timings
        all_timings = [t for worker_timings in results for t in worker_timings]

        result = BenchmarkResult(
            test_name="Concurrent Operations",
            operation_count=len(all_timings),
            total_time_ms=total_elapsed_ms,
            avg_time_ms=statistics.mean(all_timings),
            min_time_ms=min(all_timings),
            max_time_ms=max(all_timings),
            median_time_ms=statistics.median(all_timings),
            p95_time_ms=self._percentile(all_timings, 95),
            p99_time_ms=self._percentile(all_timings, 99),
            throughput_ops_per_sec=(len(all_timings) / total_elapsed_ms) * 1000,
            success_rate=1.0
        )

        self.results.append(result)

        logger.info(
            f"  Avg: {result.avg_time_ms:.2f}ms | "
            f"P95: {result.p95_time_ms:.2f}ms | "
            f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec"
        )

        return result

    # ========================================
    # Reporting
    # ========================================

    def print_summary_report(self):
        """Print human-readable summary report"""
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE BENCHMARK SUMMARY")
        logger.info("=" * 80)

        if not self.results:
            logger.info("No benchmark results available")
            return

        logger.info(f"\n{'Test Name':<30} {'Operations':<12} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Throughput (ops/s)':<20}")
        logger.info("-" * 80)

        for result in self.results:
            logger.info(
                f"{result.test_name:<30} "
                f"{result.operation_count:<12} "
                f"{result.avg_time_ms:<10.2f} "
                f"{result.p95_time_ms:<10.2f} "
                f"{result.throughput_ops_per_sec:<20.1f}"
            )

        logger.info("=" * 80)

        # Cache speedup analysis
        uncached_results = [r for r in self.results if "Uncached" in r.test_name]
        cached_results = [r for r in self.results if "Cached" in r.test_name]

        if uncached_results and cached_results:
            speedup = uncached_results[0].avg_time_ms / cached_results[0].avg_time_ms
            logger.info(f"\nCache Speedup: {speedup:.1f}x")

        logger.info("")

    def export_json_report(self, filepath: str = "benchmark_results.json"):
        """Export results to JSON file"""
        data = {
            "benchmark_timestamp": time.time(),
            "dht_enabled": self.dht_enabled,
            "results": [asdict(result) for result in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results exported to: {filepath}")


async def run_full_benchmark_suite():
    """Run complete benchmark suite"""
    if not IDENTITY_AVAILABLE:
        logger.error("Identity modules not available - cannot run benchmark")
        return

    logger.info("=" * 80)
    logger.info("IDENTITY DHT PERFORMANCE BENCHMARK SUITE")
    logger.info("=" * 80)
    logger.info("")

    benchmark = PerformanceBenchmark(dht_enabled=False)

    # Run all benchmarks
    await benchmark.benchmark_identity_creation(count=100)
    await benchmark.benchmark_identity_query_uncached(count=100)
    await benchmark.benchmark_identity_query_cached(count=100)
    await benchmark.benchmark_guardian_addition(count=50)
    await benchmark.benchmark_recovery_authorization(count=100)
    await benchmark.benchmark_fl_verification(count=100)
    await benchmark.benchmark_reputation_sync(count=100)
    await benchmark.benchmark_concurrent_operations(concurrency=10, operations_per_task=10)

    # Print summary
    benchmark.print_summary_report()

    # Export results
    benchmark.export_json_report("identity_dht_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(run_full_benchmark_suite())
