#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 CIFAR-10 Dataset Generalization Test
==========================================

Purpose: Validate Gen7's 50% BFT on CIFAR-10 (color images) vs EMNIST (grayscale)
         to demonstrate dataset generalization beyond just grayscale digits.

Test Matrix:
- Dataset: CIFAR-10 (32×32×3 RGB images, 3072 features)
- Byzantine ratios: [35%, 40%, 45%, 50%]
- Seeds: [101, 202, 303] (3 seeds for statistical validation)
- Aggregators: ["aegis", "aegis_gen7"]
- Attack: sign_flip (validated attack type)
- Data distribution: α=1.0 (IID, matches EMNIST baseline)

Expected: Gen7 should maintain 85-87% accuracy at 50% BFT on CIFAR-10,
          demonstrating dataset-agnostic defense (grayscale → color).

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
from experiments.datasets.common import make_cifar10

# Test configuration
SEEDS = [101, 202, 303]  # 3 seeds for validation
BYZ_RATIOS = [0.35, 0.40, 0.45, 0.50]  # Test 35-50% Byzantine
AGGREGATORS = ["aegis", "aegis_gen7"]
ATTACK = "sign_flip"  # Validated attack type

# CIFAR-10 configuration
DATASET_CONFIG = {
    "n_clients": 50,
    "noniid_alpha": 1.0,  # IID (matches EMNIST baseline)
    "n_train": 2500,      # CIFAR-10 default
    "n_test": 500,
    "seed": 42,
}

print("=" * 80)
print("🎨 Gen7 CIFAR-10 Dataset Generalization Test")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Dataset: CIFAR-10 (32×32×3 RGB, 3072 features)")
print(f"   Seeds: {SEEDS}")
print(f"   Byzantine ratios: {[f'{r*100:.0f}%' for r in BYZ_RATIOS]}")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Attack: {ATTACK}")
print(f"   Data distribution: α={DATASET_CONFIG['noniid_alpha']} (IID)")
print()

# Load CIFAR-10 dataset
print("📊 Loading CIFAR-10 dataset...")
fl_data = make_cifar10(**DATASET_CONFIG)
print(f"✅ Dataset loaded: {fl_data.n_features} features, {fl_data.n_classes} classes")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(BYZ_RATIOS) * len(AGGREGATORS)
current_experiment = 0

# Run experiments
for seed in SEEDS:
    for byz_frac in BYZ_RATIOS:
        for agg in AGGREGATORS:
            current_experiment += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {current_experiment}/{experiment_count}: {agg} @ {byz_frac*100:.0f}% Byzantine (seed {seed})")
            print(f"{'=' * 80}")

            scenario = FLScenario(
                n_clients=50,
                byz_frac=byz_frac,
                noniid_alpha=1.0,
                attack=ATTACK,
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
                    "attack_type": ATTACK,
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
                    "attack_type": ATTACK,
                    "aggregator": agg,
                    "clean_acc": 0.0,
                    "auc": 0.0,
                    "wall_time": 0.0,
                    "status": "error",
                    "error": str(e),
                }
                all_results.append(record)

# Analysis by Byzantine ratio
print(f"\n{'=' * 80}")
print("📈 CIFAR-10 RESULTS BY BYZANTINE RATIO")
print(f"{'=' * 80}")

for byz_frac in BYZ_RATIOS:
    print(f"\n## Byzantine Ratio: {byz_frac*100:.0f}%")
    print()
    print(f"{'Aggregator':>15} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Mean AUC':>10} | {'Success Rate':>15}")
    print("-" * 80)

    for agg in AGGREGATORS:
        # Filter results for this Byzantine ratio and aggregator
        subset = [r for r in all_results if r["byz_frac"] == byz_frac and r["aggregator"] == agg]

        if not subset:
            continue

        accuracies = [r["clean_acc"] for r in subset]
        aucs = [r["auc"] for r in subset]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_auc = np.mean(aucs)
        success_count = sum(1 for r in subset if r["status"] == "success")
        success_rate = success_count / len(subset)

        print(f"{agg:>15} | {mean_acc:>9.1%}  | {std_acc:>9.1%}  | {mean_auc:>9.4f}  | {success_rate:>7.0%} ({success_count}/{len(subset)})")

# Gen7 performance comparison
print(f"\n{'=' * 80}")
print("🎯 GEN7 CIFAR-10 vs EMNIST COMPARISON")
print(f"{'=' * 80}")
print()
print(f"{'Byzantine %':>12} | {'CIFAR-10 Acc':>14} | {'EMNIST Acc':>12} | {'Difference':>12}")
print("-" * 80)

# EMNIST baseline results (from multi-seed validation)
emnist_baseline = {
    0.35: 0.864,  # 86.4% ± 0.6%
    0.40: 0.867,  # 86.7% ± 0.2%
    0.45: 0.868,  # 86.8% ± 0.4%
    0.50: 0.869,  # 86.9% ± 0.2%
}

for byz_frac in BYZ_RATIOS:
    gen7_subset = [r for r in all_results if r["byz_frac"] == byz_frac and r["aggregator"] == "aegis_gen7"]

    if not gen7_subset:
        continue

    cifar10_acc = np.mean([r["clean_acc"] for r in gen7_subset])
    emnist_acc = emnist_baseline.get(byz_frac, 0.0)
    diff = cifar10_acc - emnist_acc

    diff_str = f"{diff:+.1%}" if diff != 0 else "0.0%"
    print(f"{byz_frac*100:>11.0f}%  | {cifar10_acc:>13.1%}  | {emnist_acc:>11.1%}  | {diff_str:>11}")

# Save results
output_dir = Path("validation_results/gen7_cifar10_sweep")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"cifar10_sweep_{timestamp}.json"

output_data = {
    "experiment": "Gen7 CIFAR-10 Dataset Generalization Test",
    "date": datetime.now().isoformat(),
    "seeds": SEEDS,
    "byz_ratios": BYZ_RATIOS,
    "attack": ATTACK,
    "aggregators": AGGREGATORS,
    "dataset_config": DATASET_CONFIG,
    "emnist_baseline": emnist_baseline,
    "results": all_results,
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved: {output_file}")

# Final verdict
print(f"\n{'=' * 80}")
print("🏆 FINAL VERDICT: Dataset Generalization Assessment")
print(f"{'=' * 80}")

gen7_50pct = [r for r in all_results if r["aggregator"] == "aegis_gen7" and r["byz_frac"] == 0.50]
if gen7_50pct:
    all_success = all(r["status"] == "success" for r in gen7_50pct)
    mean_acc = np.mean([r["clean_acc"] for r in gen7_50pct])
    mean_auc = np.mean([r["auc"] for r in gen7_50pct])
    std_acc = np.std([r["clean_acc"] for r in gen7_50pct])

    print()
    if all_success:
        print("🎉 GEN7 ACHIEVES 50% BFT ON CIFAR-10!")
        print()
        print(f"   CIFAR-10 @ 50% BFT: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"   EMNIST @ 50% BFT:   {emnist_baseline[0.50]:.1%} ± 0.2%")
        print(f"   Mean AUC: {mean_auc:.4f}")
        print()
        if abs(mean_acc - emnist_baseline[0.50]) < 0.02:
            print("✅ DATASET-AGNOSTIC DEFENSE VALIDATED!")
            print("   Gen7 maintains consistent performance across grayscale and color images.")
        else:
            print("⚠️  Performance gap detected between CIFAR-10 and EMNIST")
            print(f"   Difference: {(mean_acc - emnist_baseline[0.50]):+.1%}")
    else:
        print(f"⚠️  Gen7 partially succeeds on CIFAR-10 @ 50% BFT")
        print(f"   Mean accuracy: {mean_acc:.1%}")
        print(f"   Success rate: {sum(1 for r in gen7_50pct if r['status'] == 'success')}/{len(gen7_50pct)}")

print(f"\n{'=' * 80}")
print("CIFAR-10 sweep complete!")
print(f"{'=' * 80}")
