#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 8 Scale Test: Production Readiness Validation
- 100+ nodes
- 1000+ transactions
- Byzantine detection validation
- Performance metrics collection
"""

import sys
import asyncio
import time
from datetime import datetime
import json

sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')

from test_scale import ScaleTestConfig, ScaleTester, ScaleTestResults


async def phase8_scale_test() -> dict:
    """
    Phase 8 Production Readiness Test
    - 100 nodes (80 honest, 20 Byzantine)
    - 15 rounds = 1500 transactions
    - Full Byzantine attack diversity
    """
    print("\n" + "="*70)
    print("🚀 PHASE 8: PRODUCTION READINESS SCALE TEST")
    print("="*70)
    print("\nTarget Requirements:")
    print("  ✓ 100+ nodes")
    print("  ✓ 1000+ transactions")
    print("  ✓ Byzantine detection validation")
    print("  ✓ Performance metrics")
    print("\n" + "="*70)

    config = ScaleTestConfig(
        num_honest=80,
        num_byzantine=20,
        num_rounds=15,  # 100 nodes × 15 rounds = 1500 transactions
        gradient_size=1000,
        use_real_ml=False,  # Use simulated for consistency
        storage_backend="memory",
        parallel_workers=8
    )

    tester = ScaleTester(config)
    result = await tester.run_test()

    # Additional analysis for Phase 8
    print("\n" + "="*70)
    print("📋 PHASE 8 REQUIREMENTS CHECK")
    print("="*70)

    total_transactions = result.gradients_processed
    nodes_check = result.total_nodes >= 100
    transactions_check = total_transactions >= 1000
    detection_check = result.avg_detection_rate >= 0.90
    performance_check = result.avg_round_time < 5.0  # 5s per round acceptable

    print(f"\n✓ Nodes: {result.total_nodes} {'✅ PASS' if nodes_check else '❌ FAIL'}")
    print(f"✓ Transactions: {total_transactions} {'✅ PASS' if transactions_check else '❌ FAIL'}")
    print(f"✓ Byzantine Detection: {result.avg_detection_rate:.1%} {'✅ PASS' if detection_check else '❌ FAIL'}")
    print(f"✓ Performance: {result.avg_round_time:.2f}s/round {'✅ PASS' if performance_check else '❌ FAIL'}")

    all_passed = nodes_check and transactions_check and detection_check and performance_check

    print("\n" + "="*70)
    if all_passed:
        print("🎉 PHASE 8 SCALE TEST: ✅ PASSED")
        print("   ZeroTrustML Credits DNA is production-ready!")
    else:
        print("⚠️  PHASE 8 SCALE TEST: ❌ NEEDS WORK")
        print("   Review failed requirements above")
    print("="*70)

    # Return structured results (convert numpy types to Python native)
    return {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 8 - Production Readiness',
        'requirements': {
            'nodes': {'target': 100, 'actual': int(result.total_nodes), 'passed': bool(nodes_check)},
            'transactions': {'target': 1000, 'actual': int(total_transactions), 'passed': bool(transactions_check)},
            'detection_rate': {'target': 0.90, 'actual': float(result.avg_detection_rate), 'passed': bool(detection_check)},
            'performance': {'target': 5.0, 'actual': float(result.avg_round_time), 'passed': bool(performance_check)}
        },
        'overall_pass': bool(all_passed),
        'detailed_results': {
            'total_nodes': int(result.total_nodes),
            'honest_nodes': int(result.honest_nodes),
            'byzantine_nodes': int(result.byzantine_nodes),
            'num_rounds': int(result.num_rounds),
            'avg_detection_rate': float(result.avg_detection_rate),
            'avg_round_time': float(result.avg_round_time),
            'total_time': float(result.total_time),
            'memory_usage_mb': float(result.memory_usage_mb),
            'gradients_processed': int(result.gradients_processed),
            'throughput': float(result.throughput)
        }
    }


async def main():
    """Run Phase 8 scale test and save results"""
    test_start = time.time()

    # Run the test
    results = await phase8_scale_test()

    test_duration = time.time() - test_start
    results['test_duration_seconds'] = test_duration

    # Save results to JSON
    output_file = '/srv/luminous-dynamics/Mycelix-Core/0TML/phase8_scale_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")
    print(f"⏱️  Total test duration: {test_duration:.1f}s")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())

    # Exit code based on pass/fail
    exit(0 if results['overall_pass'] else 1)
