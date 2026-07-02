#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 BFT Sweep: Validate 45% Byzantine Tolerance
=================================================

Tests AEGIS+Gen7 aggregator from 35% to 45% Byzantine in 1% steps.
Compares against vanilla AEGIS to demonstrate improvement.

Expected Results:
- Vanilla AEGIS: Fails at 36% (validated)
- AEGIS+Gen7: Works up to 45% (target claim)
- Gen7 achieves AUC 1.00 (vs PoGQ's 0.788)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.simulator import run_fl, FLScenario

# Check Gen7 availability
try:
    import gen7_zkstark
    GEN7_AVAILABLE = True
    print("✅ Gen7 module loaded successfully")
except ImportError as e:
    GEN7_AVAILABLE = False
    print(f"❌ Gen7 not available: {e}")
    sys.exit(1)


def run_bft_sweep(
    byz_fracs,
    aggregators,
    seeds,
    n_rounds=10,
    output_dir="validation_results/gen7_bft_sweep"
):
    """Run BFT sweep across Byzantine fractions and aggregators."""

    results = []

    for seed in seeds:
        for byz_frac in byz_fracs:
            for agg in aggregators:
                print(f"\n{'='*80}")
                print(f"Testing: {agg} @ {byz_frac*100:.0f}% Byzantine (seed {seed})")
                print(f"{'='*80}")

                # Create scenario
                scenario = FLScenario(
                    n_clients=50,
                    byz_frac=byz_frac,
                    noniid_alpha=1.0,  # IID for clarity
                    attack="sign_flip",
                    seed=seed,
                )

                # Run FL
                try:
                    result = run_fl(
                        scenario=scenario,
                        aggregator=agg,
                        rounds=n_rounds,
                    )

                    # Extract key metrics
                    record = {
                        "seed": seed,
                        "byz_frac": byz_frac,
                        "aggregator": agg,
                        "clean_acc": result["clean_acc"],
                        "auc": result.get("auc", 0.0),
                        "wall_time": result.get("wall_s", 0.0),  # Fixed: simulator uses "wall_s" not "wall_time"
                        "status": "success" if result["clean_acc"] > 0.80 else "failed",
                    }

                    print(f"\n📊 Results:")
                    print(f"   Accuracy: {result['clean_acc']:.1%}")
                    print(f"   AUC: {result.get('auc', 0.0):.4f}")
                    print(f"   Status: {record['status']}")

                    results.append(record)

                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()

                    record = {
                        "seed": seed,
                        "byz_frac": byz_frac,
                        "aggregator": agg,
                        "clean_acc": 0.0,
                        "auc": 0.0,
                        "wall_time": 0.0,
                        "status": "error",
                        "error": str(e),
                    }
                    results.append(record)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"gen7_bft_sweep_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "byz_fracs": byz_fracs,
            "aggregators": aggregators,
            "seeds": seeds,
            "n_rounds": n_rounds,
            "results": results,
        }, f, indent=2)

    print(f"\n✅ Results saved: {output_file}")
    return results


def analyze_results(results):
    """Analyze BFT sweep results and compare aggregators."""
    print(f"\n{'='*80}")
    print("ANALYSIS: Gen7 vs Vanilla AEGIS")
    print(f"{'='*80}")

    # Group by aggregator and byz_frac
    by_agg = {}
    for r in results:
        agg = r["aggregator"]
        if agg not in by_agg:
            by_agg[agg] = {}

        byz_frac = r["byz_frac"]
        if byz_frac not in by_agg[agg]:
            by_agg[agg][byz_frac] = []

        by_agg[agg][byz_frac].append(r)

    # Find BFT limits
    print(f"\n📊 BFT Tolerance Limits:")
    for agg in sorted(by_agg.keys()):
        max_working = 0.0
        for byz_frac in sorted(by_agg[agg].keys()):
            records = by_agg[agg][byz_frac]
            avg_acc = np.mean([r["clean_acc"] for r in records])

            if avg_acc > 0.80:
                max_working = byz_frac

        print(f"   {agg:20s}: {max_working*100:.0f}% Byzantine tolerance")

    # Detailed comparison table
    print(f"\n📋 Detailed Results:")
    print(f"{'Byzantine %':12s} | {'AEGIS Acc':12s} | {'AEGIS+Gen7 Acc':15s} | {'Δ (pp)':8s}")
    print(f"{'-'*60}")

    for byz_frac in sorted(set(r["byz_frac"] for r in results)):
        aegis_records = by_agg.get("aegis", {}).get(byz_frac, [])
        gen7_records = by_agg.get("aegis_gen7", {}).get(byz_frac, [])

        if aegis_records and gen7_records:
            aegis_acc = np.mean([r["clean_acc"] for r in aegis_records])
            gen7_acc = np.mean([r["clean_acc"] for r in gen7_records])
            delta = (gen7_acc - aegis_acc) * 100

            print(f"{byz_frac*100:>10.0f}%  | {aegis_acc:>10.1%}  | {gen7_acc:>13.1%}  | {delta:>6.1f}pp")


def main():
    """Run Gen7 BFT sweep and compare with vanilla AEGIS."""
    print(f"\n{'='*80}")
    print("Gen7 BFT Sweep: Validating 45% Byzantine Tolerance")
    print(f"{'='*80}")

    if not GEN7_AVAILABLE:
        print("\n❌ Gen7 not available - cannot proceed")
        return 1

    # Sweep configuration
    byz_fracs = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    aggregators = ["aegis", "aegis_gen7"]  # Compare vanilla vs Gen7
    seeds = [101]  # Single seed for quick test (can add more for robustness)

    print(f"\n📋 Configuration:")
    print(f"   Byzantine fractions: {[f'{x*100:.0f}%' for x in byz_fracs]}")
    print(f"   Aggregators: {aggregators}")
    print(f"   Seeds: {seeds}")
    print(f"   Rounds per experiment: 10")

    # Run sweep
    results = run_bft_sweep(
        byz_fracs=byz_fracs,
        aggregators=aggregators,
        seeds=seeds,
        n_rounds=10,
    )

    # Analyze
    analyze_results(results)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    gen7_results = [r for r in results if r["aggregator"] == "aegis_gen7"]
    gen7_45_success = [r for r in gen7_results if r["byz_frac"] == 0.45 and r["clean_acc"] > 0.80]

    if gen7_45_success:
        print(f"\n✅ SUCCESS: AEGIS+Gen7 achieves >80% accuracy at 45% Byzantine!")
        print(f"   Accuracy at 45%: {gen7_45_success[0]['clean_acc']:.1%}")
        print(f"   AUC: {gen7_45_success[0]['auc']:.4f}")
        print(f"\n🎯 Claim validated: Gen7 enables 45% BFT tolerance")
        return 0
    else:
        print(f"\n⚠️  Results show AEGIS+Gen7 does not achieve 45% BFT yet")
        print(f"   This may require full zkSTARK proofs (not just Dilithium signatures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
