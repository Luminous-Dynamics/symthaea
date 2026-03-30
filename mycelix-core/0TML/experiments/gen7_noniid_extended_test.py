# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 Extended Non-IID Test (50 Rounds)
=======================================

Re-tests Gen7 on severe non-IID data (α=0.1) with 50 rounds instead of 10
to validate robustness to heterogeneous data distributions.

Expected outcome: 75-80% accuracy (vs 50.8% with 10 rounds)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

from simulator import run_fl, FLScenario
from datasets.common import make_emnist

# Test configuration
SEEDS = [101, 202, 303]
ALPHA = 0.1  # Severe non-IID
BYZ_FRAC = 0.50  # Validated 50% Byzantine threshold
AGGREGATORS = ["aegis", "aegis_gen7"]
ATTACK = "sign_flip"
ROUNDS = 50  # Extended from 10

# EMNIST configuration
DATASET_CONFIG = {
    "n_clients": 50,
    "noniid_alpha": ALPHA,
    "n_train": 6000,
    "n_test": 1000,
    "seed": 42,
}

print("=" * 80)
print("🌐 Gen7 Extended Non-IID Test (50 Rounds)")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Dataset: EMNIST (28×28 grayscale, 784 features)")
print(f"   Distribution: α={ALPHA} (severe non-IID)")
print(f"   Seeds: {SEEDS}")
print(f"   Byzantine: {BYZ_FRAC*100:.0f}%")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Attack: {ATTACK}")
print(f"   Rounds: {ROUNDS} (extended from 10)")
print()

# Load dataset
print("📊 Loading EMNIST dataset...")
fl_data = make_emnist(**DATASET_CONFIG)
print(f"✅ Dataset loaded: {fl_data.n_features} features, {fl_data.n_classes} classes")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(AGGREGATORS)
current_experiment = 0

# Run experiments
for seed in SEEDS:
    for agg in AGGREGATORS:
        current_experiment += 1
        print(f"\n{'=' * 80}")
        print(f"Experiment {current_experiment}/{experiment_count}: {agg} @ α={ALPHA} (seed {seed})")
        print(f"{'=' * 80}")

        scenario = FLScenario(
            n_clients=50,
            byz_frac=BYZ_FRAC,
            noniid_alpha=ALPHA,
            attack=ATTACK,
            seed=seed,
        )

        try:
            result = run_fl(
                scenario=scenario,
                dataset=fl_data,
                aggregator=agg,
                rounds=ROUNDS,
            )

            record = {
                "seed": seed,
                "alpha": ALPHA,
                "byz_frac": BYZ_FRAC,
                "attack_type": ATTACK,
                "aggregator": agg,
                "clean_acc": result["clean_acc"],
                "auc": result.get("auc", 0.0),
                "wall_time": result.get("wall_s", 0.0),
                "status": "success" if result["clean_acc"] > 0.70 else "failed",
                "rounds": ROUNDS,
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
                "alpha": ALPHA,
                "byz_frac": BYZ_FRAC,
                "attack_type": ATTACK,
                "aggregator": agg,
                "clean_acc": 0.0,
                "auc": 0.0,
                "wall_time": 0.0,
                "status": "error",
                "error": str(e),
                "rounds": ROUNDS,
            }
            all_results.append(record)

# Analysis
print(f"\n{'=' * 80}")
print("📈 EXTENDED ROUNDS RESULTS")
print(f"{'=' * 80}")
print()
print(f"{'Aggregator':>14} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Mean AUC':>10} | {'Success Rate':>18}")
print("-" * 80)

for agg in AGGREGATORS:
    subset = [r for r in all_results if r['aggregator'] == agg]
    if subset:
        accs = [r['clean_acc'] for r in subset]
        aucs = [r['auc'] for r in subset]
        successes = [r['status'] == 'success' for r in subset]

        print(f"{agg:>14} | {np.mean(accs):>9.1%} | {np.std(accs):>9.1%} | {np.mean(aucs):>10.4f} | {np.mean(successes):>7.0%} ({sum(successes)}/{len(successes)})")

# Comparison with 10 rounds
print(f"\n{'=' * 80}")
print("🎯 50 ROUNDS vs 10 ROUNDS COMPARISON")
print(f"{'=' * 80}")
print()

rounds_10_baseline = {
    "aegis": 0.098,
    "aegis_gen7": 0.508,
}

print(f"{'Aggregator':>14} | {'10 Rounds':>10} | {'50 Rounds':>10} | {'Improvement':>12}")
print("-" * 80)

for agg in AGGREGATORS:
    subset = [r for r in all_results if r['aggregator'] == agg]
    if subset:
        acc_50 = np.mean([r['clean_acc'] for r in subset])
        acc_10 = rounds_10_baseline[agg]
        improvement = acc_50 - acc_10

        print(f"{agg:>14} | {acc_10:>9.1%} | {acc_50:>9.1%} | {improvement:>+11.1%}")

# Statistical comparison
gen7_results = [r for r in all_results if r['aggregator'] == 'aegis_gen7']
if gen7_results:
    acc_50_rounds = [r['clean_acc'] for r in gen7_results]
    acc_10_rounds = [0.508] * 3  # Previous results

    t_stat, p_value = stats.ttest_rel(acc_50_rounds, acc_10_rounds)

    print(f"\n{'=' * 80}")
    print("📊 STATISTICAL COMPARISON: 10 vs 50 Rounds (Gen7)")
    print(f"{'=' * 80}")
    print()
    print(f"10 rounds:    {np.mean(acc_10_rounds):.1%} ± {np.std(acc_10_rounds):.1%}")
    print(f"50 rounds:    {np.mean(acc_50_rounds):.1%} ± {np.std(acc_50_rounds):.1%}")
    print(f"Improvement:  {(np.mean(acc_50_rounds) - np.mean(acc_10_rounds))*100:+.1f}pp")
    print()
    print(f"Statistical significance:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"   ✅ SIGNIFICANT (p < 0.05)")
    else:
        print(f"   ⚠️  NOT SIGNIFICANT (p >= 0.05)")

# Save results
output_dir = Path("validation_results/gen7_noniid_extended")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"noniid_extended_{timestamp}.json"

with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved: {output_path}")

# Final verdict
gen7_subset = [r for r in all_results if r['aggregator'] == 'aegis_gen7']
if gen7_subset:
    mean_acc = np.mean([r['clean_acc'] for r in gen7_subset])
    success_rate = np.mean([r['status'] == 'success' for r in gen7_subset])

    print(f"\n{'=' * 80}")
    print("🏆 FINAL VERDICT: Extended Rounds Performance")
    print(f"{'=' * 80}")
    print()

    if mean_acc >= 0.75:
        print(f"✅ Gen7 succeeds on severe non-IID (α={ALPHA}) with 50 rounds")
    else:
        print(f"⚠️  Gen7 partially succeeds on severe non-IID (α={ALPHA})")

    print(f"   Mean accuracy: {mean_acc:.1%}")
    print(f"   Success rate: {success_rate:.0%}")
    print(f"   Improvement over 10 rounds: {(mean_acc - 0.508)*100:+.1f}pp")

print(f"\n{'=' * 80}")
print("Extended non-IID test complete!")
print(f"{'=' * 80}")
