#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Quick test: E1 with EMNIST to validate baseline accuracy fix.

This script tests whether switching from synthetic to EMNIST dataset
resolves the baseline accuracy issue (67% cap → expected 85%+).

Phase 1: Test 0% Byzantine (baseline accuracy)
Phase 2: Test 45% Byzantine (if baseline ≥85%)
"""

import sys
import json
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_stubs import run_e1_byzantine_sweep_emnist


def test_baseline():
    """Test baseline accuracy with 0% Byzantine on EMNIST."""
    print("=" * 80)
    print("PHASE 1: Testing Baseline Accuracy (0% Byzantine on EMNIST)")
    print("=" * 80)
    print()
    print("Expected: ≥85% accuracy (both AEGIS and Median)")
    print("Previous (synthetic): 67% accuracy ceiling")
    print()

    config = {
        "byz_frac": 0.0,  # NO Byzantine clients
        "seed": 101,
        "rounds": 25,
        "epochs": 5,
        "n_clients": 50,
    }

    print("Running FL simulation (this will take ~2-3 minutes)...")
    print()

    result = run_e1_byzantine_sweep_emnist(config)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS (0% Byzantine):")
    print("=" * 80)
    print(f"  AEGIS Accuracy:  {result['robust_acc_aegis']*100:.1f}%")
    print(f"  Median Accuracy: {result['robust_acc_median']*100:.1f}%")
    print(f"  Delta:           {result['robust_acc_delta_pp']:+.1f}pp")
    print(f"  Convergence:     AEGIS={result['convergence_round_aegis']}, Median={result['convergence_round_median']}")
    print("=" * 80)
    print()

    # Assess baseline
    aegis_acc = result['robust_acc_aegis']
    median_acc = result['robust_acc_median']

    if aegis_acc >= 0.85 and median_acc >= 0.85:
        print("✅ SUCCESS: Baseline accuracy ≥85% achieved!")
        print("   EMNIST has FIXED the baseline accuracy issue.")
        print()
        return True, result
    elif aegis_acc >= 0.75 and median_acc >= 0.75:
        print("⚠️  MARGINAL: Baseline 75-85% (better than 67%, but not ideal)")
        print("   May need hyperparameter tuning.")
        print()
        return True, result  # Still proceed to 45% test
    else:
        print("❌ FAILED: Baseline accuracy <75%")
        print(f"   AEGIS: {aegis_acc*100:.1f}%, Median: {median_acc*100:.1f}%")
        print("   Deeper issue with model/training - investigate before proceeding.")
        print()
        return False, result


def test_45pct_byzantine(baseline_result):
    """Test 45% Byzantine tolerance."""
    print("=" * 80)
    print("PHASE 2: Testing 45% Byzantine Tolerance")
    print("=" * 80)
    print()
    print("Expected: AEGIS ≥70%, Median <70%")
    print("Previous (synthetic): Both failed (AEGIS 51.1%, Median 46.1%)")
    print()

    config = {
        "byz_frac": 0.45,  # 45% Byzantine clients
        "seed": 101,
        "rounds": 25,
        "epochs": 5,
        "n_clients": 50,
    }

    print("Running FL simulation (this will take ~2-3 minutes)...")
    print()

    result = run_e1_byzantine_sweep_emnist(config)

    print("\n" + "=" * 80)
    print("45% BYZANTINE RESULTS:")
    print("=" * 80)
    print(f"  AEGIS Accuracy:  {result['robust_acc_aegis']*100:.1f}% (target: ≥70%)")
    print(f"  Median Accuracy: {result['robust_acc_median']*100:.1f}% (target: <70%)")
    print(f"  Delta:           {result['robust_acc_delta_pp']:+.1f}pp")
    print(f"  Status:          {result['status']}")
    print(f"  AUC:             {result['auc']:.3f}")
    print(f"  Convergence:     AEGIS={result['convergence_round_aegis']}, Median={result['convergence_round_median']}")
    print("=" * 80)
    print()

    # Assess 45% BFT
    aegis_acc = result['robust_acc_aegis']
    median_acc = result['robust_acc_median']
    status = result['status']

    if status == "aegis_wins":
        print("🎉 SUCCESS: 45% BFT VALIDATED!")
        print(f"   AEGIS achieves {aegis_acc*100:.1f}% at 45% Byzantine")
        print(f"   Median fails at {median_acc*100:.1f}%")
        print("   Paper claim is EMPIRICALLY SUPPORTED!")
        print()
        return True, result
    elif status == "both_work":
        print("⚠️  MARGINAL: Both methods work at 45%")
        print(f"   AEGIS {aegis_acc*100:.1f}%, Median {median_acc*100:.1f}% (both ≥70%)")
        print("   May need to test higher Byzantine ratios to find Median failure point.")
        print()
        return True, result  # Still good, just need to find the limit
    else:
        print("❌ INSUFFICIENT: 45% BFT not achieved")
        print(f"   AEGIS: {aegis_acc*100:.1f}% (target ≥70%)")
        print(f"   Status: {status}")
        print("   Need to either:")
        print("   - Lower BFT claim to empirically supported level (e.g., 35% or 40%)")
        print("   - Tune AEGIS parameters for better performance")
        print()
        return False, result


def main():
    print("\n" + "=" * 80)
    print("🔬 E1 EMNIST Baseline Test - Critical Validation Fix")
    print("=" * 80)
    print()
    print("Testing if EMNIST solves the synthetic dataset baseline accuracy issue.")
    print("This determines whether the paper can proceed to submission.")
    print()

    # Phase 1: Baseline accuracy
    baseline_ok, baseline_result = test_baseline()

    if not baseline_ok:
        print("\n❌ CRITICAL: Baseline accuracy test FAILED")
        print("Cannot proceed to 45% Byzantine test.")
        print("Deeper investigation required.")
        sys.exit(1)

    # Save baseline result
    output_dir = Path("validation_results/E1_emnist_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "e1_baseline_0pct.json", "w") as f:
        json.dump(baseline_result, f, indent=2)
    print(f"Baseline result saved: {output_dir / 'e1_baseline_0pct.json'}")
    print()

    # Phase 2: 45% Byzantine test
    bft_ok, bft_result = test_45pct_byzantine(baseline_result)

    # Save 45% result
    with open(output_dir / "e1_test_45pct.json", "w") as f:
        json.dump(bft_result, f, indent=2)
    print(f"45% BFT result saved: {output_dir / 'e1_test_45pct.json'}")
    print()

    # Final assessment
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)

    if baseline_ok and bft_ok:
        print("✅ ✅ EMNIST VALIDATION SUCCESSFUL!")
        print()
        print("Next steps:")
        print("1. Run full E1 sweep (10% → 50% Byzantine) on EMNIST")
        print("2. Re-run E2, E3, E5 on EMNIST")
        print("3. Update GEN5_VALIDATION_REPORT with real results")
        print("4. Draft paper Section 4 with empirical evidence")
        print()
        print("Estimated timeline to paper ready: 3-5 days")
    elif baseline_ok:
        print("⚠️  PARTIAL SUCCESS: Baseline fixed, but 45% BFT not achieved")
        print()
        print("Options:")
        print("1. Test lower Byzantine ratios (35%, 40%) to find actual limit")
        print("2. Adjust paper claim to empirically supported level")
        print("3. Tune AEGIS hyperparameters for better performance")
    else:
        print("❌ CRITICAL FAILURE: Baseline accuracy issue persists")
        print()
        print("Deeper investigation required:")
        print("1. Check model architecture (may need MLP instead of logistic regression)")
        print("2. Verify EMNIST data loading and partitioning")
        print("3. Inspect training dynamics (loss curves, gradient norms)")

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
