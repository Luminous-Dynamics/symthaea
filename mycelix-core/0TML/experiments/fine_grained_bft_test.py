#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Fine-grained BFT testing between 35% and 40% in 1% increments.

Based on initial sweep:
- 35% Byzantine: Both work (AEGIS 87.3%, Median 84.7%)
- 40% Byzantine: Both fail (AEGIS 24.0%, Median 1.3%)

Now test 36%, 37%, 38%, 39% to find exact boundary.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_stubs import run_e1_byzantine_sweep_emnist


def fine_grained_bft_test():
    """Test Byzantine ratios from 35% to 40% in 1% steps."""

    print("\n" + "=" * 80)
    print("🔬 Fine-Grained BFT Boundary Testing (35-40% in 1% steps)")
    print("=" * 80)
    print("\nFinding exact Byzantine fault tolerance boundary.\n")

    # Test 35-40% in 1% increments
    ratios = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40]

    results = []
    exact_limit = None

    for byz_frac in ratios:
        n_byzantine = int(byz_frac * 50)

        print("=" * 80)
        print(f"Testing {int(byz_frac*100)}% Byzantine ({n_byzantine}/50 clients)")
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

        # Define failure as <80% accuracy (well below honest baseline ~87%)
        aegis_works = aegis_acc >= 0.80
        median_works = median_acc >= 0.80

        print(f"  AEGIS works: {aegis_works}")
        print(f"  Median works: {median_works}")
        print()

        # Check for failure point
        if not aegis_works:
            print(f"❌ AEGIS FAILS at {int(byz_frac*100)}%")
            exact_limit = byz_frac - 0.01  # Previous ratio was the limit
            print(f"✅ Exact limit: {int(exact_limit*100)}% Byzantine tolerance")
            break
        elif aegis_works:
            print(f"✅ AEGIS STILL WORKS at {int(byz_frac*100)}%")

    # Analysis
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS: Exact Byzantine Fault Tolerance Limit")
    print("=" * 80)

    if exact_limit:
        print(f"\n✅ EXACT BFT LIMIT FOUND: {int(exact_limit*100)}%")
        print(f"\n   AEGIS maintains >80% accuracy up to {int(exact_limit*100)}% Byzantine clients")
        print(f"   Exceeds classical 33% BFT limit by {int(exact_limit*100) - 33}pp")

        # Get the last working result
        last_working = [r for r in results if r['robust_acc_aegis'] >= 0.80][-1]
        print(f"\n   At {int(last_working['byz_frac']*100)}% Byzantine:")
        print(f"   AEGIS:  {last_working['robust_acc_aegis']*100:.1f}%")
        print(f"   Median: {last_working['robust_acc_median']*100:.1f}%")
        print(f"   Delta:  {last_working['robust_acc_delta_pp']:+.1f}pp")
    else:
        # Check if AEGIS still works at 40%
        if results[-1]['robust_acc_aegis'] >= 0.80:
            print(f"\n✅ AEGIS WORKS UP TO 40% (highest tested)")
            print(f"\n   Exceeds classical 33% BFT limit by 7pp")
        else:
            print(f"\n⚠️  Need to test beyond 40% to find upper limit")

    # Save results
    output_dir = Path("validation_results/E1_bft_fine_grained")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fine_grained_results.json", "w") as f:
        json.dump({
            "ratios_tested": ratios,
            "results": results,
            "exact_limit": exact_limit if exact_limit else (0.40 if results[-1]['robust_acc_aegis'] >= 0.80 else None),
        }, f, indent=2)

    print(f"\nResults saved: {output_dir / 'fine_grained_results.json'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    fine_grained_bft_test()
