#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 Comprehensive Byzantine Attack Suite
==========================================

Purpose: Validate Gen7's cryptographic verification against all 7 Byzantine attack types
         to ensure robustness beyond just sign-flip attacks.

Attack Types Tested:
1. Sign-flip (already validated) ✅
2. Gaussian noise (random noise)
3. Targeted poison (backdoor/amplified gradients)
4. Model replacement (completely random weights)
5. Adaptive attack (evades distance-based detection)
6. Sybil attack (coordinated malicious clients)
7. Label-flip (train on corrupted labels) - requires retraining

Expected: Gen7 should detect attacks 1-6 with AUC ~1.00 due to cryptographic verification
          Attack 7 (label-flip) may be harder since it produces valid gradients from corrupted data

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
BYZ_FRAC = 0.45  # Test at validated 45% Byzantine threshold
ATTACK_TYPES = [
    "sign_flip",          # ✅ Already validated
    "gaussian_noise",     # Random noise
    "model_replacement",  # Completely random weights
    "scaled_grad",        # Gradient scaling (targeted poison)
    # "adaptive" and "sybil" require peer gradient info - test separately
]

AGGREGATORS = ["aegis", "aegis_gen7"]

# Dataset configuration
DATASET_CONFIG = {
    "n_clients": 50,
    "noniid_alpha": 1.0,
    "n_train": 6000,
    "n_test": 1000,
    "seed": 42,
}

print("=" * 80)
print("🛡️  Gen7 Comprehensive Byzantine Attack Suite")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Seeds: {SEEDS}")
print(f"   Byzantine ratio: {BYZ_FRAC*100:.0f}%")
print(f"   Attack types: {ATTACK_TYPES}")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Dataset: EMNIST ({DATASET_CONFIG['n_train']} train, {DATASET_CONFIG['n_test']} test)")
print()

# Load EMNIST dataset
print("📊 Loading EMNIST dataset...")
fl_data = make_emnist(**DATASET_CONFIG)
print(f"✅ Dataset loaded: {fl_data.n_features} features, {fl_data.n_classes} classes")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(ATTACK_TYPES) * len(AGGREGATORS)
current_experiment = 0

# Run experiments
for seed in SEEDS:
    for attack in ATTACK_TYPES:
        for agg in AGGREGATORS:
            current_experiment += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {current_experiment}/{experiment_count}: {agg} vs {attack} (seed {seed})")
            print(f"{'=' * 80}")

            scenario = FLScenario(
                n_clients=50,
                byz_frac=BYZ_FRAC,
                noniid_alpha=1.0,
                attack=attack,
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
                    "attack_type": attack,
                    "byz_frac": BYZ_FRAC,
                    "aggregator": agg,
                    "clean_acc": result["clean_acc"],
                    "auc": result.get("auc", 0.0),
                    "wall_time": result.get("wall_s", 0.0),
                    "status": "success" if result["clean_acc"] > 0.80 else "failed",
                }

                print(f"\n📊 Results:")
                print(f"   Attack: {attack}")
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
                    "attack_type": attack,
                    "byz_frac": BYZ_FRAC,
                    "aggregator": agg,
                    "clean_acc": 0.0,
                    "auc": 0.0,
                    "wall_time": 0.0,
                    "status": "error",
                    "error": str(e),
                }
                all_results.append(record)

# Analysis by attack type
print(f"\n{'=' * 80}")
print("📈 ATTACK-SPECIFIC ANALYSIS")
print(f"{'=' * 80}")

for attack in ATTACK_TYPES:
    print(f"\n## Attack: {attack.upper()}")
    print()
    print(f"{'Aggregator':>15} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Mean AUC':>10} | {'Success Rate':>15}")
    print("-" * 80)

    for agg in AGGREGATORS:
        # Filter results for this attack and aggregator
        subset = [r for r in all_results if r["attack_type"] == attack and r["aggregator"] == agg]

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

# Cross-attack comparison
print(f"\n{'=' * 80}")
print("🎯 CROSS-ATTACK ROBUSTNESS: Gen7 Performance")
print(f"{'=' * 80}")
print()
print(f"{'Attack Type':>20} | {'Mean Acc':>10} | {'Mean AUC':>10} | {'Success Rate':>15}")
print("-" * 80)

for attack in ATTACK_TYPES:
    gen7_subset = [r for r in all_results if r["attack_type"] == attack and r["aggregator"] == "aegis_gen7"]

    if not gen7_subset:
        continue

    accuracies = [r["clean_acc"] for r in gen7_subset]
    aucs = [r["auc"] for r in gen7_subset]
    mean_acc = np.mean(accuracies)
    mean_auc = np.mean(aucs)
    success_count = sum(1 for r in gen7_subset if r["status"] == "success")
    success_rate = success_count / len(gen7_subset)

    status_emoji = "✅" if success_rate == 1.0 else "⚠️"
    print(f"{attack:>20} | {mean_acc:>9.1%}  | {mean_auc:>9.4f}  | {status_emoji} {success_rate:>6.0%} ({success_count}/{len(gen7_subset)})")

# Detection quality analysis
print(f"\n{'=' * 80}")
print("🔍 CRYPTOGRAPHIC DETECTION QUALITY")
print(f"{'=' * 80}")

gen7_results = [r for r in all_results if r["aggregator"] == "aegis_gen7"]
if gen7_results:
    perfect_detection = sum(1 for r in gen7_results if r["auc"] >= 0.99)
    good_detection = sum(1 for r in gen7_results if 0.9 <= r["auc"] < 0.99)
    poor_detection = sum(1 for r in gen7_results if r["auc"] < 0.9)
    total = len(gen7_results)

    print()
    print(f"Gen7 Detection Rate Distribution:")
    print(f"   Perfect (AUC ≥ 0.99): {perfect_detection}/{total} ({perfect_detection/total:.1%})")
    print(f"   Good (AUC 0.9-0.99):  {good_detection}/{total} ({good_detection/total:.1%})")
    print(f"   Poor (AUC < 0.9):     {poor_detection}/{total} ({poor_detection/total:.1%})")
    print()

    if perfect_detection == total:
        print("✅ PERFECT - Gen7 achieves AUC ≥ 0.99 against ALL attack types!")
    elif perfect_detection + good_detection == total:
        print("✅ EXCELLENT - Gen7 achieves AUC ≥ 0.9 against all attack types")
    else:
        print("⚠️  MIXED - Some attacks evade Gen7 detection")

# Save results
output_dir = Path("validation_results/gen7_attack_suite")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"comprehensive_attack_suite_{timestamp}.json"

output_data = {
    "experiment": "Gen7 Comprehensive Byzantine Attack Suite",
    "date": datetime.now().isoformat(),
    "seeds": SEEDS,
    "byz_frac": BYZ_FRAC,
    "attack_types": ATTACK_TYPES,
    "aggregators": AGGREGATORS,
    "dataset_config": DATASET_CONFIG,
    "results": all_results,
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved: {output_file}")

# Final verdict
print(f"\n{'=' * 80}")
print("🏆 FINAL VERDICT: Gen7 Robustness Assessment")
print(f"{'=' * 80}")

gen7_results = [r for r in all_results if r["aggregator"] == "aegis_gen7"]
if gen7_results:
    all_success = all(r["status"] == "success" for r in gen7_results)
    mean_acc = np.mean([r["clean_acc"] for r in gen7_results])
    mean_auc = np.mean([r["auc"] for r in gen7_results])

    attacks_tested = len(set(r["attack_type"] for r in gen7_results))
    attacks_defended = len(set(r["attack_type"] for r in gen7_results if r["status"] == "success"))

    print()
    if all_success:
        print("🎉 GEN7 SUCCESSFULLY DEFENDS AGAINST ALL TESTED ATTACK TYPES!")
        print()
        print(f"   Attacks tested: {attacks_tested}")
        print(f"   Attacks defended: {attacks_defended}/{attacks_tested} (100%)")
        print(f"   Mean accuracy: {mean_acc:.1%}")
        print(f"   Mean AUC: {mean_auc:.4f}")
        print()
        print("Cryptographic gradient verification provides robust defense against diverse Byzantine attacks.")
    else:
        print(f"⚠️  Gen7 successfully defends {attacks_defended}/{attacks_tested} attack types")
        print(f"   Mean accuracy: {mean_acc:.1%}")
        print(f"   Mean AUC: {mean_auc:.4f}")
        print()
        print("Further investigation needed for attacks that evade cryptographic verification.")

print(f"\n{'=' * 80}")
print("Attack suite validation complete!")
print(f"{'=' * 80}")
