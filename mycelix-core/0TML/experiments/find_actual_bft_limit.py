#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Find the ACTUAL Byzantine fault tolerance limit empirically.

Stop chasing theoretical 45%. Find what we can VALIDATE.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_stubs import run_e1_byzantine_sweep_emnist


def find_bft_limit():
    """Sweep Byzantine ratios to find actual AEGIS advantage point."""

    print("\n" + "=" * 80)
    print("🎯 Finding ACTUAL Byzantine Fault Tolerance Limit")
    print("=" * 80)
    print("\nHonest empirical validation - claim what we prove, not what we hope.\n")

    # Test Byzantine ratios from 10% to 40% in 5% steps
    ratios = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    results = []

    for byz_frac in ratios:
        print("=" * 80)
        print(f"Testing {int(byz_frac*100)}% Byzantine ({int(byz_frac*50)}/50 clients)")
        print("=" * 80)

        config = {
            "byz_frac": byz_frac,
            "seed": 101,
            "rounds": 25,
            "epochs": 5,
            "n_clients": 50,
        }

        print(f"Running FL simulation...")
        result = run_e1_byzantine_sweep_emnist(config)
        results.append(result)

        aegis_acc = result['robust_acc_aegis']
        median_acc = result['robust_acc_median']
        status = result['status']

        print(f"\nResults:")
        print(f"  AEGIS:  {aegis_acc*100:.1f}%")
        print(f"  Median: {median_acc*100:.1f}%")
        print(f"  Status: {status}")
        print()

        # Check if we found the limit
        if status == "aegis_wins":
            print(f"✅ AEGIS ADVANTAGE at {int(byz_frac*100)}%!")
        elif status == "both_fail":
            print(f"❌ Both methods fail at {int(byz_frac*100)}%")
            print(f"   (AEGIS limit likely between {int((byz_frac-0.05)*100)}% and {int(byz_frac*100)}%)")
            break

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Byzantine Fault Tolerance Limit")
    print("=" * 80)

    # Find highest Byzantine ratio where AEGIS wins
    aegis_wins_at = [r for r in results if r['status'] == 'aegis_wins']

    if aegis_wins_at:
        max_bft = max(r['byz_frac'] for r in aegis_wins_at)
        print(f"\n✅ AEGIS VALIDATED at: {int(max_bft*100)}% Byzantine")
        print(f"   Highest empirically validated BFT level")

        # Get the result
        max_result = [r for r in results if r['byz_frac'] == max_bft][0]
        print(f"\n   AEGIS:  {max_result['robust_acc_aegis']*100:.1f}%")
        print(f"   Median: {max_result['robust_acc_median']*100:.1f}%")
        print(f"   Delta:  {max_result['robust_acc_delta_pp']:+.1f}pp")

        # Recommendation
        if max_bft >= 0.35:
            print(f"\n🎉 EXCEEDS 33% classical limit!")
            print(f"   Paper claim: 'AEGIS achieves {int(max_bft*100)}% BFT, exceeding classical 33% barrier'")
        elif max_bft >= 0.33:
            print(f"\n✅ Matches classical limit with practical advantages")
            print(f"   Paper claim: 'AEGIS achieves 33% BFT with enhanced non-IID robustness'")
        else:
            print(f"\n📊 Validated BFT: {int(max_bft*100)}%")
            print(f"   Paper claim: 'AEGIS provides {int(max_bft*100)}% Byzantine tolerance in practical settings'")
    else:
        print("\n❌ AEGIS did not achieve advantage at any tested ratio")
        print("   Need to debug AEGIS implementation or adjust experimental setup")

    # Save results
    output_dir = Path("validation_results/E1_bft_limit")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "bft_sweep_results.json", "w") as f:
        json.dump({
            "ratios_tested": ratios,
            "results": results,
            "max_validated_bft": max_bft if aegis_wins_at else None,
        }, f, indent=2)

    print(f"\nResults saved: {output_dir / 'bft_sweep_results.json'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    find_bft_limit()
