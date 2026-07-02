#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 Non-IID Data Distribution Test
====================================

Purpose: Validate Gen7's 50% BFT on realistic non-IID data distribution (α=0.1)
         vs previous IID baseline (α=1.0) to demonstrate robustness to heterogeneous
         client data distributions.

Test Matrix:
- Dataset: EMNIST (28×28 grayscale, 784 features)
- Byzantine ratio: 50% (validated threshold)
- Seeds: [101, 202, 303] (3 seeds for statistical validation)
- Aggregators: ["aegis", "aegis_gen7"]
- Attack: sign_flip (validated attack type)
- Data distributions:
  - α=1.0 (IID baseline - uniform client data)
  - α=0.5 (moderately non-IID)
  - α=0.1 (severely non-IID - realistic federated setting)

Expected: Gen7 should maintain 85-87% accuracy at 50% BFT even with α=0.1,
          demonstrating robustness to heterogeneous data distributions typical
          in real-world federated learning.

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

# Test configuration
SEEDS = [101, 202, 303]  # 3 seeds for validation
ALPHAS = [1.0, 0.5, 0.1]  # IID → moderately non-IID → severely non-IID
BYZ_FRAC = 0.50  # Test at validated 50% Byzantine threshold
AGGREGATORS = ["aegis", "aegis_gen7"]
ATTACK = "sign_flip"  # Validated attack type

# EMNIST configuration
DATASET_CONFIG_BASE = {
    "n_clients": 50,
    "n_train": 6000,
    "n_test": 1000,
    "seed": 42,
}

print("=" * 80)
print("📊 Gen7 Non-IID Data Distribution Test")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Dataset: EMNIST (28×28 grayscale, 784 features)")
print(f"   Seeds: {SEEDS}")
print(f"   Data distributions: α={ALPHAS} (1.0=IID, 0.1=severely non-IID)")
print(f"   Byzantine ratio: {BYZ_FRAC*100:.0f}%")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Attack: {ATTACK}")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(ALPHAS) * len(AGGREGATORS)
current_experiment = 0

# Run experiments for each alpha value
for alpha in ALPHAS:
    print(f"\n{'=' * 80}")
    print(f"📊 Loading EMNIST dataset with α={alpha:.1f} ({'IID' if alpha == 1.0 else 'non-IID'})...")
    print(f"{'=' * 80}")

    # Load dataset with specific alpha
    fl_data = make_emnist(noniid_alpha=alpha, **DATASET_CONFIG_BASE)
    print(f"✅ Dataset loaded: {fl_data.n_features} features, {fl_data.n_classes} classes")
    print()

    for seed in SEEDS:
        for agg in AGGREGATORS:
            current_experiment += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {current_experiment}/{experiment_count}: {agg} @ α={alpha:.1f} (seed {seed})")
            print(f"{'=' * 80}")

            scenario = FLScenario(
                n_clients=50,
                byz_frac=BYZ_FRAC,
                noniid_alpha=alpha,  # Note: this is just metadata, actual distribution from make_emnist
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
                    "alpha": alpha,
                    "byz_frac": BYZ_FRAC,
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
                    "alpha": alpha,
                    "byz_frac": BYZ_FRAC,
                    "attack_type": ATTACK,
                    "aggregator": agg,
                    "clean_acc": 0.0,
                    "auc": 0.0,
                    "wall_time": 0.0,
                    "status": "error",
                    "error": str(e),
                }
                all_results.append(record)

# Analysis by data distribution
print(f"\n{'=' * 80}")
print("📈 RESULTS BY DATA DISTRIBUTION (α)")
print(f"{'=' * 80}")

for alpha in ALPHAS:
    print(f"\n## Data Distribution: α={alpha:.1f} ({'IID' if alpha == 1.0 else 'non-IID'})")
    print()
    print(f"{'Aggregator':>15} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Mean AUC':>10} | {'Success Rate':>15}")
    print("-" * 80)

    for agg in AGGREGATORS:
        # Filter results for this alpha and aggregator
        subset = [r for r in all_results if r["alpha"] == alpha and r["aggregator"] == agg]

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

# Gen7 performance across distributions
print(f"\n{'=' * 80}")
print("🎯 GEN7 ROBUSTNESS TO DATA HETEROGENEITY")
print(f"{'=' * 80}")
print()
print(f"{'Alpha (α)':>12} | {'Distribution':>18} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Success':>10}")
print("-" * 80)

distribution_labels = {
    1.0: "IID (uniform)",
    0.5: "Moderate non-IID",
    0.1: "Severe non-IID"
}

for alpha in ALPHAS:
    gen7_subset = [r for r in all_results if r["alpha"] == alpha and r["aggregator"] == "aegis_gen7"]

    if not gen7_subset:
        continue

    mean_acc = np.mean([r["clean_acc"] for r in gen7_subset])
    std_acc = np.std([r["clean_acc"] for r in gen7_subset])
    success_count = sum(1 for r in gen7_subset if r["status"] == "success")
    success_rate = success_count / len(gen7_subset)

    success_str = f"{success_rate:.0%} ({success_count}/{len(gen7_subset)})"
    print(f"{alpha:>11.1f}  | {distribution_labels[alpha]:>18} | {mean_acc:>9.1%}  | {std_acc:>9.1%}  | {success_str:>10}")

# Statistical comparison
print(f"\n{'=' * 80}")
print("📊 STATISTICAL COMPARISON: IID (α=1.0) vs Non-IID (α=0.1)")
print(f"{'=' * 80}")

gen7_iid = [r for r in all_results if r["alpha"] == 1.0 and r["aggregator"] == "aegis_gen7"]
gen7_noniid = [r for r in all_results if r["alpha"] == 0.1 and r["aggregator"] == "aegis_gen7"]

if gen7_iid and gen7_noniid:
    iid_accs = [r["clean_acc"] for r in gen7_iid]
    noniid_accs = [r["clean_acc"] for r in gen7_noniid]

    iid_mean = np.mean(iid_accs)
    noniid_mean = np.mean(noniid_accs)
    iid_std = np.std(iid_accs)
    noniid_std = np.std(noniid_accs)

    diff = noniid_mean - iid_mean

    print()
    print(f"IID (α=1.0):        {iid_mean:.1%} ± {iid_std:.1%}")
    print(f"Non-IID (α=0.1):    {noniid_mean:.1%} ± {noniid_std:.1%}")
    print(f"Difference:         {diff:+.1%}")
    print()

    if abs(diff) < 0.03:  # Less than 3% difference
        print("✅ ROBUST TO DATA HETEROGENEITY!")
        print("   Gen7 maintains consistent performance across IID and non-IID distributions.")
    elif diff < 0:
        print("⚠️  Performance degradation on non-IID data detected")
        print(f"   Impact: {diff:.1%} accuracy drop")
    else:
        print("🎉 Performance IMPROVEMENT on non-IID data!")
        print(f"   Gain: {diff:+.1%}")

    # t-test for statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(iid_accs, noniid_accs)
    print()
    print(f"Statistical significance:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"   ⚠️  SIGNIFICANT (p < 0.05) - Distributions differ statistically")
    else:
        print(f"   ✅ NOT SIGNIFICANT (p ≥ 0.05) - No statistical difference")

# Save results
output_dir = Path("validation_results/gen7_noniid")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"noniid_test_{timestamp}.json"

output_data = {
    "experiment": "Gen7 Non-IID Data Distribution Test",
    "date": datetime.now().isoformat(),
    "seeds": SEEDS,
    "alphas": ALPHAS,
    "byz_frac": BYZ_FRAC,
    "attack": ATTACK,
    "aggregators": AGGREGATORS,
    "dataset_config": DATASET_CONFIG_BASE,
    "results": all_results,
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved: {output_file}")

# Final verdict
print(f"\n{'=' * 80}")
print("🏆 FINAL VERDICT: Non-IID Robustness Assessment")
print(f"{'=' * 80}")

gen7_severe = [r for r in all_results if r["alpha"] == 0.1 and r["aggregator"] == "aegis_gen7"]
if gen7_severe:
    all_success = all(r["status"] == "success" for r in gen7_severe)
    mean_acc = np.mean([r["clean_acc"] for r in gen7_severe])
    std_acc = np.std([r["clean_acc"] for r in gen7_severe])
    mean_auc = np.mean([r["auc"] for r in gen7_severe])

    print()
    if all_success:
        print("🎉 GEN7 ACHIEVES 50% BFT ON SEVERELY NON-IID DATA (α=0.1)!")
        print()
        print(f"   Non-IID (α=0.1): {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"   IID (α=1.0):     {iid_mean:.1%} ± {iid_std:.1%}")
        print(f"   Mean AUC: {mean_auc:.4f}")
        print()
        print("✅ REALISTIC FEDERATED LEARNING VALIDATED!")
        print("   Gen7 robust to heterogeneous client data distributions.")
    else:
        print(f"⚠️  Gen7 partially succeeds on non-IID data (α=0.1)")
        print(f"   Mean accuracy: {mean_acc:.1%}")
        print(f"   Success rate: {sum(1 for r in gen7_severe if r['status'] == 'success')}/{len(gen7_severe)}")

print(f"\n{'=' * 80}")
print("Non-IID test complete!")
print(f"{'=' * 80}")
