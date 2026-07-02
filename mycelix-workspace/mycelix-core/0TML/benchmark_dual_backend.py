#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick benchmark: RISC Zero vs Winterfell

Tests the dual backend abstraction with a simple PoGQ scenario.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests/integration"))

from zkbackend_abstraction import DualBackendTester

def main():
    print("=" * 70)
    print("  DUAL BACKEND PERFORMANCE BENCHMARK")
    print("  RISC Zero (zkVM) vs Winterfell (STARK)")
    print("=" * 70)
    print()

    # Test scenario: 10 FL rounds, k=3 violations triggers quarantine
    public_inputs = {
        "n": 10,  # 10 rounds
        "k": 3,   # 3 violations trigger quarantine
        "m": 2,   # 2 clears to release
        "w": 2,   # 2-round warm-up
        "threshold": 65536,  # Q16.16 fixed-point (1.0)
        "scale": 65536
    }

    # Witness: quality scores showing 4 violations (should quarantine)
    witness = [
        100000,  # Round 0: 1.52 (good) - warm-up
        80000,   # Round 1: 1.22 (good) - warm-up
        50000,   # Round 2: 0.76 (bad) - violation 1
        40000,   # Round 3: 0.61 (bad) - violation 2
        45000,   # Round 4: 0.69 (bad) - violation 3 → QUARANTINE
        55000,   # Round 5: 0.84 (bad) - violation 4 (in quarantine)
        70000,   # Round 6: 1.07 (good)
        75000,   # Round 7: 1.14 (good)
        80000,   # Round 8: 1.22 (good)
        85000,   # Round 9: 1.30 (good)
    ]

    output_dir = Path(__file__).parent / "benchmark_outputs"
    output_dir.mkdir(exist_ok=True)

    tester = DualBackendTester()

    if not tester.risc_zero or not tester.risc_zero.is_available():
        print("❌ RISC Zero backend not available")
        return 1

    if not tester.winterfell or not tester.winterfell.is_available():
        print("❌ Winterfell backend not available")
        return 1

    print("Running benchmark...")
    print()

    results = tester.compare_backends(public_inputs, witness, output_dir)

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if results["risc_zero"]:
        r = results["risc_zero"]
        print(f"\nRISC Zero:")
        print(f"  Proving time: {r.proving_time_ms:.0f}ms ({r.proving_time_ms/1000:.1f}s)")
        print(f"  Proof size: {r.proof_size_bytes:,} bytes ({r.proof_size_bytes/1024:.0f} KB)")
        print(f"  Quarantine decision: {r.quarantine_decision}")

    if results["winterfell"]:
        w = results["winterfell"]
        print(f"\nWinterfell:")
        print(f"  Proving time: {w.proving_time_ms:.0f}ms ({w.proving_time_ms/1000:.1f}s)")
        print(f"  Proof size: {w.proof_size_bytes:,} bytes ({w.proof_size_bytes/1024:.0f} KB)")
        print(f"  Quarantine decision: {w.quarantine_decision}")

    if results["speedup"]:
        print(f"\n{'='*70}")
        print(f"  🚀 WINTERFELL SPEEDUP: {results['speedup']:.2f}x FASTER")
        print(f"{'='*70}")
        print(f"\nDecisions match: {'✅ YES' if results['decisions_match'] else '❌ NO'}")

    print()

    if results["decisions_match"]:
        print("✅ SUCCESS: Both backends agree on quarantine decision!")
        print()
        print("🎉 Dual backend architecture is production-ready!")
        return 0
    else:
        print("❌ FAILURE: Backends disagree on quarantine decision!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
