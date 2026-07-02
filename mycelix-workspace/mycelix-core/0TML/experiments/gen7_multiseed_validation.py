#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 Multi-Seed Validation - Statistical Significance Testing
==============================================================

Purpose: Validate Gen7's 45% BFT achievement across multiple random seeds
         to ensure reproducibility and statistical significance.

Expected: ~87% accuracy (±1-2%) across all seeds at 45% Byzantine
If validated: Publication-ready claim
If fails: Identify seed-dependent behavior and investigate

Author: Luminous Dynamics
Date: November 14, 2025
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from experiments.simulator import run_fl, FLScenario
from experiments.datasets.common import make_emnist

# Multi-seed configuration
SEEDS = [101, 202, 303, 404, 505]  # 5 seeds for statistical significance
BYZ_FRACS = [0.35, 0.40, 0.45, 0.50]  # Key Byzantine ratios
AGGREGATORS = ["aegis", "aegis_gen7"]

# Dataset configuration (consistent across all runs)
DATASET_CONFIG = {
    "n_clients": 50,
    "noniid_alpha": 1.0,
    "n_train": 6000,
    "n_test": 1000,
    "seed": 42,  # Dataset seed (separate from FL seed)
}

print("=" * 80)
print("🔬 Gen7 Multi-Seed Validation - Statistical Significance Testing")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Seeds: {SEEDS}")
print(f"   Byzantine ratios: {[f'{int(f*100)}%' for f in BYZ_FRACS]}")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Dataset: EMNIST ({DATASET_CONFIG['n_train']} train, {DATASET_CONFIG['n_test']} test)")
print(f"   Clients: {DATASET_CONFIG['n_clients']}, α={DATASET_CONFIG['noniid_alpha']}")
print()

# Load EMNIST dataset once (reuse for all experiments)
print("📊 Loading EMNIST dataset (this will be reused for all experiments)...")
fl_data = make_emnist(**DATASET_CONFIG)
print(f"✅ Dataset loaded: {fl_data.n_features} features, {fl_data.n_classes} classes")
print(f"   Clients: {len(fl_data.train_splits)}, Test samples: {len(fl_data.test_clean[1])}")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(BYZ_FRACS) * len(AGGREGATORS)
current_experiment = 0

# Run experiments
for seed in SEEDS:
    for byz_frac in BYZ_FRACS:
        for agg in AGGREGATORS:
            current_experiment += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {current_experiment}/{experiment_count}: {agg} @ {byz_frac*100:.0f}% Byzantine (seed {seed})")
            print(f"{'=' * 80}")

            scenario = FLScenario(
                n_clients=50,
                byz_frac=byz_frac,
                noniid_alpha=1.0,
                attack="sign_flip",
                seed=seed,
            )

            try:
                result = run_fl(
                    scenario=scenario,
                    dataset=fl_data,
                    aggregator=agg,
                    rounds=10,
                )

                record = {
                    "seed": seed,
                    "byz_frac": byz_frac,
                    "aggregator": agg,
                    "clean_acc": result["clean_acc"],
                    "auc": result.get("auc", 0.0),
                    "wall_time": result.get("wall_s", 0.0),
                    "status": "success" if result["clean_acc"] > 0.80 else "failed",
                }

                print(f"\n📊 Results:")
                print(f"   Accuracy: {result['clean_acc']:.1%}")
                print(f"   AUC: {result.get('auc', 0.0):.4f}")
                print(f"   Wall Time: {result.get('wall_s', 0.0):.2f}s")
                print(f"   Status: {record['status']}")

                all_results.append(record)

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
                all_results.append(record)

# Statistical analysis
print(f"\n{'=' * 80}")
print("📈 STATISTICAL ANALYSIS: Multi-Seed Results")
print(f"{'=' * 80}")

# Group by aggregator and Byzantine fraction
for agg in AGGREGATORS:
    print(f"\n## {agg.upper()} Results")
    print()
    print(f"{'Byzantine %':>12} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Min Acc':>10} | {'Max Acc':>10} | {'Success Rate':>15}")
    print("-" * 80)

    for byz_frac in BYZ_FRACS:
        # Filter results for this aggregator and Byzantine fraction
        subset = [r for r in all_results if r["aggregator"] == agg and r["byz_frac"] == byz_frac]

        if not subset:
            continue

        accuracies = [r["clean_acc"] for r in subset]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        success_count = sum(1 for r in subset if r["status"] == "success")
        success_rate = success_count / len(subset)

        print(f"{byz_frac*100:>11.0f}%  | {mean_acc:>9.1%}  | {std_acc:>9.1%}  | {min_acc:>9.1%}  | {max_acc:>9.1%}  | {success_rate:>7.0%} ({success_count}/{len(subset)})")

# Comparison: AEGIS vs Gen7 at 45%
print(f"\n{'=' * 80}")
print("🎯 KEY RESULT: Gen7 vs AEGIS at 45% Byzantine")
print(f"{'=' * 80}")

aegis_45 = [r for r in all_results if r["aggregator"] == "aegis" and r["byz_frac"] == 0.45]
gen7_45 = [r for r in all_results if r["aggregator"] == "aegis_gen7" and r["byz_frac"] == 0.45]

if aegis_45 and gen7_45:
    aegis_accs = [r["clean_acc"] for r in aegis_45]
    gen7_accs = [r["clean_acc"] for r in gen7_45]

    aegis_mean = np.mean(aegis_accs)
    aegis_std = np.std(aegis_accs)
    gen7_mean = np.mean(gen7_accs)
    gen7_std = np.std(gen7_accs)

    improvement = gen7_mean - aegis_mean

    print(f"\nAEGIS (vanilla):    {aegis_mean:.1%} ± {aegis_std:.1%}")
    print(f"AEGIS+Gen7:         {gen7_mean:.1%} ± {gen7_std:.1%}")
    print(f"Improvement:        {improvement:+.1%} ({improvement*100:+.1f} percentage points)")
    print()

    # Statistical significance (t-test)
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(gen7_accs, aegis_accs)
        print(f"Statistical significance:")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"   ✅ SIGNIFICANT (p < 0.05) - Gen7 improvement is statistically valid!")
        else:
            print(f"   ⚠️  Not significant (p ≥ 0.05) - More seeds may be needed")
    except ImportError:
        print("   (scipy not available for t-test)")

    # Reproducibility assessment
    gen7_success_rate = sum(1 for r in gen7_45 if r["status"] == "success") / len(gen7_45)
    print()
    print(f"Reproducibility:")
    print(f"   Gen7 success rate: {gen7_success_rate:.0%} ({sum(1 for r in gen7_45 if r['status'] == 'success')}/{len(gen7_45)} seeds)")

    if gen7_success_rate == 1.0:
        print(f"   ✅ PERFECT - 45% BFT achieved across ALL seeds!")
    elif gen7_success_rate >= 0.8:
        print(f"   ✅ GOOD - 45% BFT achieved in majority of seeds")
    else:
        print(f"   ⚠️  INCONSISTENT - Further investigation needed")

# Save results
output_dir = Path("validation_results/gen7_multiseed")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"multiseed_validation_{timestamp}.json"

output_data = {
    "experiment": "Gen7 Multi-Seed Validation",
    "date": datetime.now().isoformat(),
    "seeds": SEEDS,
    "byz_fracs": BYZ_FRACS,
    "aggregators": AGGREGATORS,
    "dataset_config": DATASET_CONFIG,
    "results": all_results,
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved: {output_file}")

# Final verdict
print(f"\n{'=' * 80}")
print("🏆 FINAL VERDICT")
print(f"{'=' * 80}")

if gen7_45:
    gen7_mean = np.mean([r["clean_acc"] for r in gen7_45])
    gen7_std = np.std([r["clean_acc"] for r in gen7_45])
    gen7_min = np.min([r["clean_acc"] for r in gen7_45])

    if gen7_min > 0.80 and gen7_mean > 0.85:
        print()
        print("🎉 GEN7 45% BFT VALIDATED ACROSS MULTIPLE SEEDS!")
        print()
        print("Publication-Ready Claim:")
        print(f'   "Gen7 achieves 45% Byzantine fault tolerance with {gen7_mean:.1%} ± {gen7_std:.1%}')
        print(f'    accuracy across {len(gen7_45)} independent random seeds (EMNIST dataset)."')
        print()
        print("This exceeds the classical 33% BFT limit by 12 percentage points through")
        print("cryptographic gradient verification using Dilithium5 post-quantum signatures.")
        print()
    else:
        print()
        print("⚠️  45% BFT not consistently achieved across all seeds")
        print(f"   Mean accuracy: {gen7_mean:.1%} ± {gen7_std:.1%}")
        print(f"   Minimum accuracy: {gen7_min:.1%}")
        print()
        print("Further investigation needed to identify seed-dependent behavior.")
        print()

print(f"{'=' * 80}")
print("Validation complete!")
print(f"{'=' * 80}")
