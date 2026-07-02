#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
E1 Byzantine Sweep Runner - Validates 45% BFT Claim

Usage:
    python experiments/run_e1_sweep.py --mode quick   # Test 45% only (1 hour)
    python experiments/run_e1_sweep.py --mode full    # Full sweep (8 hours)
    python experiments/run_e1_sweep.py --byz-frac 0.45 --seed 101  # Single config
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_stubs import run_e1_byzantine_sweep


def run_quick_test(output_dir: Path):
    """Quick validation: Test 45% BFT only (critical claim)."""
    print("🎯 Quick Test: Validating 45% BFT Claim")
    print("="*80)

    config = {"byz_frac": 0.45, "seed": 101}
    print(f"\nRunning: byz_frac={config['byz_frac']} (45% Byzantine)")
    print("Estimated time: ~60 minutes")
    print()

    result = run_e1_byzantine_sweep(config)

    # Save result
    result_file = output_dir / f"e1_byz045_seed101.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print interpretation
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  AEGIS Robust Acc:  {result['robust_acc_aegis']:.1%}")
    print(f"  Median Robust Acc: {result['robust_acc_median']:.1%}")
    print(f"  Delta:             {result['robust_acc_delta_pp']:+.1f}pp")
    print(f"  Status:            {result['status']}")
    print(f"  AUC:               {result['auc']:.3f}")
    print()

    # Interpret for paper
    if result['status'] == 'aegis_wins':
        print("✅ SUCCESS: AEGIS achieves 45% BFT!")
        print(f"   AEGIS maintains {result['robust_acc_aegis']:.1%} robust accuracy")
        print(f"   while Median fails at {result['robust_acc_median']:.1%}")
        print()
        print("📝 Paper claim VALIDATED: '45% Byzantine tolerance exceeding classical 33% barrier'")
    elif result['status'] == 'both_work':
        print("⚠️  MARGINAL: Both AEGIS and Median work at 45%")
        print("   Need to test higher Byzantine ratios (50%+) to find AEGIS limit")
    else:
        print("❌ UNEXPECTED: AEGIS did not achieve 45% BFT")
        print("   Need to investigate implementation or tune thresholds")

    print("\n" + "="*80)
    print(f"Result saved: {result_file}")
    return result


def run_full_sweep(output_dir: Path, seed: int = 101):
    """Full Byzantine sweep: 10% to 50% in 7 steps."""
    print("🚀 Full Byzantine Sweep: 10% → 50%")
    print("="*80)
    print("Estimated time: 8-10 hours")
    print()

    configs = [
        {"byz_frac": 0.10, "seed": seed},  # Sanity check
        {"byz_frac": 0.20, "seed": seed},  # Current baseline
        {"byz_frac": 0.30, "seed": seed},  # Classical limit approach
        {"byz_frac": 0.33, "seed": seed},  # Exactly classical limit
        {"byz_frac": 0.35, "seed": seed},  # Median should fail here
        {"byz_frac": 0.40, "seed": seed},  # AEGIS should work
        {"byz_frac": 0.45, "seed": seed},  # KEY: AEGIS wins
        {"byz_frac": 0.50, "seed": seed},  # Both fail (sanity)
    ]

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running byz_frac={config['byz_frac']:.0%}")
        print(f"Estimated remaining: {(len(configs) - i + 1) * 60} minutes")

        result = run_e1_byzantine_sweep(config)
        results.append(result)

        # Save intermediate result
        result_file = output_dir / f"e1_byz{int(config['byz_frac']*100):02d}_seed{seed}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Quick status
        status_icon = {
            "both_work": "✅",
            "aegis_wins": "🎯",
            "both_fail": "❌",
            "median_wins": "⚠️"
        }.get(result['status'], "?")

        print(f"  {status_icon} AEGIS: {result['robust_acc_aegis']:.1%} | "
              f"Median: {result['robust_acc_median']:.1%} | "
              f"Δ: {result['robust_acc_delta_pp']:+.1f}pp")

    # Save complete results
    summary_file = output_dir / f"e1_sweep_summary_seed{seed}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "num_configs": len(configs),
            "results": results
        }, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE:")
    print()
    print("| Byz% | AEGIS Acc | Median Acc | Delta | Status |")
    print("|------|-----------|------------|-------|--------|")
    for r in results:
        print(f"| {r['byz_frac']*100:4.0f} | "
              f"{r['robust_acc_aegis']:8.1%} | "
              f"{r['robust_acc_median']:9.1%} | "
              f"{r['robust_acc_delta_pp']:+5.1f} | "
              f"{r['status']:10s} |")

    # Find critical transition points
    print("\n" + "="*80)
    print("KEY FINDINGS:")

    # Find where Median fails (robust_acc < 70%)
    median_fail_idx = next((i for i, r in enumerate(results) if r['robust_acc_median'] < 0.70), None)
    if median_fail_idx is not None:
        median_fail_byz = results[median_fail_idx]['byz_frac']
        print(f"  Median fails at:     {median_fail_byz:.0%} Byzantine")

    # Find where AEGIS fails
    aegis_fail_idx = next((i for i, r in enumerate(results) if r['robust_acc_aegis'] < 0.70), None)
    if aegis_fail_idx is not None:
        aegis_fail_byz = results[aegis_fail_idx]['byz_frac']
        print(f"  AEGIS fails at:      {aegis_fail_byz:.0%} Byzantine")
        print(f"  BFT improvement:     {(aegis_fail_byz - median_fail_byz)*100:+.0f}pp")
    else:
        print(f"  AEGIS fails at:      >50% Byzantine")
        print(f"  BFT improvement:     ≥{(0.50 - median_fail_byz)*100:.0f}pp")

    # Check 45% specifically
    idx_45 = next((i for i, r in enumerate(results) if r['byz_frac'] == 0.45), None)
    if idx_45 is not None:
        r45 = results[idx_45]
        if r45['status'] == 'aegis_wins':
            print(f"\n  ✅ 45% BFT VALIDATED!")
            print(f"     AEGIS: {r45['robust_acc_aegis']:.1%} | Median: {r45['robust_acc_median']:.1%}")
        else:
            print(f"\n  ⚠️  45% BFT MARGINAL")
            print(f"     Status: {r45['status']}")

    print("\n" + "="*80)
    print(f"Complete results saved: {summary_file}")

    return results


def run_single_config(byz_frac: float, seed: int, output_dir: Path):
    """Run single Byzantine ratio configuration."""
    print(f"Running E1 with byz_frac={byz_frac:.0%}, seed={seed}")

    config = {"byz_frac": byz_frac, "seed": seed}
    result = run_e1_byzantine_sweep(config)

    # Save result
    result_file = output_dir / f"e1_byz{int(byz_frac*100):02d}_seed{seed}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print result
    print()
    print(f"AEGIS:  {result['robust_acc_aegis']:.1%}")
    print(f"Median: {result['robust_acc_median']:.1%}")
    print(f"Delta:  {result['robust_acc_delta_pp']:+.1f}pp")
    print(f"Status: {result['status']}")
    print(f"\nSaved: {result_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="E1 Byzantine Sweep - Validate 45% BFT Claim"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "single"],
        default="quick",
        help="Test mode: quick (45%% only), full (10-50%% sweep), single (custom)"
    )
    parser.add_argument(
        "--byz-frac",
        type=float,
        help="Byzantine fraction for single mode (e.g., 0.45)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Random seed (default: 101)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results/E1_byzantine_sweep"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print()
    print("🔬 E1 Byzantine Sweep Runner")
    print("="*80)
    print(f"Mode:   {args.mode}")
    print(f"Seed:   {args.seed}")
    print(f"Output: {args.output}")
    print("="*80)
    print()

    # Run appropriate mode
    if args.mode == "quick":
        result = run_quick_test(args.output)
    elif args.mode == "full":
        results = run_full_sweep(args.output, args.seed)
    elif args.mode == "single":
        if args.byz_frac is None:
            print("ERROR: --byz-frac required for single mode")
            sys.exit(1)
        result = run_single_config(args.byz_frac, args.seed, args.output)

    print()
    print("✅ E1 Byzantine Sweep Complete!")
    print()


if __name__ == "__main__":
    main()
