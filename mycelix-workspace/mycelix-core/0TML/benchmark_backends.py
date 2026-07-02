#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Backend Performance Benchmarking
Comprehensive performance testing across all storage backends

Measures:
- Connection time
- Write latency (single)
- Read latency (single)
- Batch write (100 gradients)
- Credit operations
- Byzantine event logging
- Statistics retrieval
- Memory usage
- Storage efficiency

Results saved to: benchmark_results.json
"""

import asyncio
import sys
import time
import json
import hashlib
import tracemalloc
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, '/srv/luminous-dynamics/Mycelix-Core/0TML/src')

from zerotrustml.backends import (
    StorageBackend,
    BackendType,
    PostgreSQLBackend,
    LocalFileBackend,
    HolochainBackend,
    EthereumBackend,
    CosmosBackend
)


# ============================================
# BENCHMARK RESULT STRUCTURES
# ============================================

@dataclass
class BenchmarkResult:
    """Result for a single benchmark"""
    backend_type: str
    operation: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    success_rate: float
    memory_mb: Optional[float] = None


@dataclass
class BackendBenchmarkSuite:
    """Complete benchmark suite for one backend"""
    backend_type: str
    connection_time_ms: float
    single_write_ms: float
    single_read_ms: float
    batch_write_100_ms: float
    batch_write_avg_ms: float
    credit_issue_ms: float
    credit_balance_ms: float
    byzantine_log_ms: float
    stats_retrieval_ms: float
    memory_usage_mb: float
    storage_size_bytes: int
    total_time_ms: float


# ============================================
# TEST DATA GENERATION
# ============================================

def create_test_gradient(gradient_id: str, node_id: str, round_num: int) -> Dict[str, Any]:
    """Create test gradient data"""
    gradient_data = [0.1, 0.2, 0.3, 0.4, 0.5]
    gradient_hash = hashlib.sha256(str(gradient_data).encode()).hexdigest()

    return {
        "id": gradient_id,
        "node_id": node_id,
        "round_num": round_num,
        "gradient": gradient_data,
        "gradient_hash": gradient_hash,
        "pogq_score": 0.95,
        "zkpoc_verified": True,
        "validation_passed": True,
        "reputation_score": 0.85,
        "timestamp": time.time()
    }


# ============================================
# BENCHMARK UTILITIES
# ============================================

async def measure_async(operation, iterations: int = 1) -> Dict[str, float]:
    """
    Measure async operation performance

    Returns:
        Dict with total_ms, avg_ms, min_ms, max_ms, std_dev_ms
    """
    times = []

    for _ in range(iterations):
        start = time.time()
        try:
            await operation()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        except Exception as e:
            print(f"   ⚠️  Operation failed: {e}")
            times.append(float('inf'))  # Mark as failure

    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]

    if not valid_times:
        return {
            "total_ms": 0,
            "avg_ms": 0,
            "min_ms": 0,
            "max_ms": 0,
            "std_dev_ms": 0,
            "success_rate": 0.0
        }

    total_ms = sum(valid_times)
    avg_ms = total_ms / len(valid_times)
    min_ms = min(valid_times)
    max_ms = max(valid_times)

    # Standard deviation
    if len(valid_times) > 1:
        variance = sum((t - avg_ms) ** 2 for t in valid_times) / len(valid_times)
        std_dev_ms = variance ** 0.5
    else:
        std_dev_ms = 0.0

    success_rate = len(valid_times) / iterations

    return {
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_dev_ms": std_dev_ms,
        "success_rate": success_rate
    }


# ============================================
# BACKEND BENCHMARKING
# ============================================

async def benchmark_backend(backend: StorageBackend) -> BackendBenchmarkSuite:
    """
    Run complete benchmark suite on a backend

    Args:
        backend: Connected storage backend

    Returns:
        BackendBenchmarkSuite with all results
    """
    backend_name = backend.backend_type.value
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARKING: {backend_name.upper()}")
    print(f"{'='*60}\n")

    suite_start = time.time()

    # Start memory tracking
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[0]

    # 1. Connection time (already done, estimate from health check)
    print("1️⃣  Connection Time")
    start = time.time()
    health = await backend.health_check()
    connection_time_ms = (time.time() - start) * 1000
    print(f"   ✅ {connection_time_ms:.2f}ms\n")

    # 2. Single write
    print("2️⃣  Single Write (1 gradient)")
    gradient = create_test_gradient("bench-single", "node-1", 1)
    result = await measure_async(lambda: backend.store_gradient(gradient), iterations=1)
    single_write_ms = result["avg_ms"]
    print(f"   ✅ {single_write_ms:.2f}ms\n")

    # 3. Single read
    print("3️⃣  Single Read (1 gradient)")
    result = await measure_async(lambda: backend.get_gradient("bench-single"), iterations=1)
    single_read_ms = result["avg_ms"]
    print(f"   ✅ {single_read_ms:.2f}ms\n")

    # 4. Batch write (100 gradients)
    print("4️⃣  Batch Write (100 gradients)")
    async def batch_write():
        tasks = []
        for i in range(100):
            gradient = create_test_gradient(f"bench-batch-{i}", f"node-{i % 10}", i % 5)
            tasks.append(backend.store_gradient(gradient))
        await asyncio.gather(*tasks)

    result = await measure_async(batch_write, iterations=1)
    batch_write_100_ms = result["avg_ms"]
    batch_write_avg_ms = batch_write_100_ms / 100 if batch_write_100_ms > 0 else 0
    print(f"   ✅ Total: {batch_write_100_ms:.2f}ms")
    print(f"   📊 Average per gradient: {batch_write_avg_ms:.2f}ms\n")

    # 5. Credit issuance
    print("5️⃣  Credit Issuance")
    result = await measure_async(
        lambda: backend.issue_credit("bench-node", 100, "benchmark_test"),
        iterations=1
    )
    credit_issue_ms = result["avg_ms"]
    print(f"   ✅ {credit_issue_ms:.2f}ms\n")

    # 6. Credit balance query
    print("6️⃣  Credit Balance Query")
    result = await measure_async(lambda: backend.get_credit_balance("bench-node"), iterations=1)
    credit_balance_ms = result["avg_ms"]
    print(f"   ✅ {credit_balance_ms:.2f}ms\n")

    # 7. Byzantine event logging
    print("7️⃣  Byzantine Event Logging")
    event = {
        "node_id": "malicious-bench",
        "round_num": 1,
        "detection_method": "benchmark",
        "severity": "low",
        "details": {"test": True}
    }
    result = await measure_async(lambda: backend.log_byzantine_event(event), iterations=1)
    byzantine_log_ms = result["avg_ms"]
    print(f"   ✅ {byzantine_log_ms:.2f}ms\n")

    # 8. Statistics retrieval
    print("8️⃣  Statistics Retrieval")
    result = await measure_async(backend.get_stats, iterations=1)
    stats_retrieval_ms = result["avg_ms"]
    stats = await backend.get_stats()
    storage_size_bytes = stats.get("storage_size_bytes", 0)
    print(f"   ✅ {stats_retrieval_ms:.2f}ms")
    print(f"   📦 Storage: {storage_size_bytes} bytes\n")

    # Memory usage
    memory_current, memory_peak = tracemalloc.get_traced_memory()
    memory_usage_mb = (memory_peak - memory_start) / (1024 * 1024)
    tracemalloc.stop()

    print(f"💾 Memory Usage: {memory_usage_mb:.2f} MB\n")

    suite_end = time.time()
    total_time_ms = (suite_end - suite_start) * 1000

    print(f"⏱️  Total Benchmark Time: {total_time_ms:.2f}ms\n")

    return BackendBenchmarkSuite(
        backend_type=backend_name,
        connection_time_ms=connection_time_ms,
        single_write_ms=single_write_ms,
        single_read_ms=single_read_ms,
        batch_write_100_ms=batch_write_100_ms,
        batch_write_avg_ms=batch_write_avg_ms,
        credit_issue_ms=credit_issue_ms,
        credit_balance_ms=credit_balance_ms,
        byzantine_log_ms=byzantine_log_ms,
        stats_retrieval_ms=stats_retrieval_ms,
        memory_usage_mb=memory_usage_mb,
        storage_size_bytes=storage_size_bytes,
        total_time_ms=total_time_ms
    )


# ============================================
# COMPARISON & REPORTING
# ============================================

def print_comparison_table(results: List[BackendBenchmarkSuite]):
    """Print formatted comparison table"""
    print("\n" + "="*100)
    print("📊 PERFORMANCE COMPARISON TABLE")
    print("="*100 + "\n")

    # Header
    backends = [r.backend_type for r in results]
    print(f"{'Operation':<30} " + " ".join(f"{b:>15}" for b in backends))
    print("-" * 100)

    # Rows
    operations = [
        ("Connection", "connection_time_ms"),
        ("Single Write", "single_write_ms"),
        ("Single Read", "single_read_ms"),
        ("Batch Write (100)", "batch_write_100_ms"),
        ("Batch Avg (per item)", "batch_write_avg_ms"),
        ("Credit Issue", "credit_issue_ms"),
        ("Credit Balance", "credit_balance_ms"),
        ("Byzantine Log", "byzantine_log_ms"),
        ("Stats Retrieval", "stats_retrieval_ms"),
        ("Memory Usage (MB)", "memory_usage_mb"),
        ("Storage (bytes)", "storage_size_bytes"),
    ]

    for op_name, op_field in operations:
        values = [getattr(r, op_field) for r in results]

        # Format values
        if "mb" in op_field.lower():
            formatted = [f"{v:.2f} MB" for v in values]
        elif "bytes" in op_field.lower():
            formatted = [f"{v:,}" for v in values]
        else:
            formatted = [f"{v:.2f}ms" for v in values]

        print(f"{op_name:<30} " + " ".join(f"{v:>15}" for v in formatted))

    print("-" * 100)

    # Total time
    totals = [r.total_time_ms for r in results]
    formatted_totals = [f"{t:.2f}ms" for t in totals]
    print(f"{'TOTAL BENCHMARK TIME':<30} " + " ".join(f"{v:>15}" for v in formatted_totals))

    print("\n" + "="*100 + "\n")


def print_winner_analysis(results: List[BackendBenchmarkSuite]):
    """Analyze and print winners for each metric"""
    print("🏆 PERFORMANCE WINNERS\n")

    metrics = [
        ("Fastest Connection", "connection_time_ms", False),
        ("Fastest Single Write", "single_write_ms", False),
        ("Fastest Single Read", "single_read_ms", False),
        ("Fastest Batch Write", "batch_write_100_ms", False),
        ("Fastest Credit Issue", "credit_issue_ms", False),
        ("Fastest Stats Query", "stats_retrieval_ms", False),
        ("Lowest Memory", "memory_usage_mb", False),
        ("Most Space Efficient", "storage_size_bytes", False),
    ]

    for metric_name, field, higher_better in metrics:
        values = [(r.backend_type, getattr(r, field)) for r in results]

        if higher_better:
            winner = max(values, key=lambda x: x[1])
        else:
            winner = min(values, key=lambda x: x[1])

        backend, value = winner

        # Format value
        if "mb" in field.lower():
            formatted_value = f"{value:.2f} MB"
        elif "bytes" in field.lower():
            formatted_value = f"{value:,} bytes"
        else:
            formatted_value = f"{value:.2f}ms"

        print(f"   {metric_name:<30} → {backend:>12} ({formatted_value})")

    print()


def save_results(results: List[BackendBenchmarkSuite], filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON"""
    data = {
        "timestamp": time.time(),
        "results": [asdict(r) for r in results]
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Results saved to: {filename}\n")


# ============================================
# MAIN BENCHMARK
# ============================================

async def main():
    """Run complete benchmark suite"""
    print("\n" + "🎯" * 50)
    print("   ZeroTrustML BACKEND PERFORMANCE BENCHMARK")
    print("🎯" * 50 + "\n")

    # Setup backends (only available ones)
    backends = []

    # LocalFile (always available)
    print("Setting up LocalFile backend...")
    try:
        localfile = LocalFileBackend(data_dir="/tmp/zerotrustml_benchmark")
        await localfile.connect()
        backends.append(("LocalFile", localfile))
        print("✅ LocalFile ready\n")
    except Exception as e:
        print(f"❌ LocalFile failed: {e}\n")

    # PostgreSQL (if available)
    print("Setting up PostgreSQL backend...")
    try:
        postgresql = PostgreSQLBackend(
            host="localhost",
            port=5432,
            database="zerotrustml_benchmark",
            user="zerotrustml",
            password="test123"
        )
        await postgresql.connect()
        backends.append(("PostgreSQL", postgresql))
        print("✅ PostgreSQL ready\n")
    except Exception as e:
        print(f"⚠️  PostgreSQL not available: {e}\n")

    # Holochain (if available)
    print("Setting up Holochain backend...")
    try:
        holochain = HolochainBackend()
        await holochain.connect()
        backends.append(("Holochain", holochain))
        print("✅ Holochain ready\n")
    except Exception as e:
        print(f"⚠️  Holochain not available: {e}\n")

    if not backends:
        print("❌ No backends available for benchmarking!")
        return

    print(f"🎯 Benchmarking {len(backends)} backend(s)\n")

    # Run benchmarks
    results = []
    for name, backend in backends:
        try:
            result = await benchmark_backend(backend)
            results.append(result)
        except Exception as e:
            print(f"❌ Benchmark failed for {name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if results:
        print_comparison_table(results)
        print_winner_analysis(results)
        save_results(results)

    # Cleanup
    print("🧹 Cleaning up...")
    for name, backend in backends:
        await backend.disconnect()
        print(f"✅ Disconnected: {name}")

    print("\n" + "🎉" * 50)
    print("   BENCHMARK COMPLETE!")
    print("🎉" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
