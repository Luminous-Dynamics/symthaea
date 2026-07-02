#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mini-Validation Suite: Quick PoGQ Smoke Test

Runs 5 critical experiments (~2.5 GPU hours) to validate that PoGQ beats Multi-Krum
BEFORE committing to the full Stage 1 (100+ GPU hours).

Decision Gate:
- ✅ PoGQ wins clearly (>3% improvement) → Run full Stage 1
- ⚠️ Marginal (1-2% improvement) → Debug and tune
- ❌ PoGQ loses → Stop and analyze why

Usage:
    python run_mini_validation.py

Estimated time: ~2.5 hours (5 experiments × 10 rounds × ~30 min)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
from datetime import datetime
from runner import ExperimentRunner

# 5 critical experiments covering:
# 1. Hardest single attack (adaptive)
# 2. Hardest coordinated attack (sybil)
# 3. Control scenario (IID data)
# 4. Moderate difficulty (moderate non-IID)
# 5. Best case (no attack baseline)

EXPERIMENTS = [
    {
        'name': 'extreme_adaptive',
        'description': 'Extreme non-IID + Adaptive attack (hardest single)',
        'alpha': 0.1,
        'attack': 'adaptive',
        'num_byzantine': 3
    },
    {
        'name': 'extreme_sybil',
        'description': 'Extreme non-IID + Sybil attack (hardest coordinated)',
        'alpha': 0.1,
        'attack': 'sybil',
        'num_byzantine': 3
    },
    {
        'name': 'iid_adaptive',
        'description': 'IID + Adaptive attack (control scenario)',
        'alpha': 100,
        'attack': 'adaptive',
        'num_byzantine': 3
    },
    {
        'name': 'moderate_gaussian',
        'description': 'Moderate non-IID + Gaussian noise (moderate difficulty)',
        'alpha': 1.0,
        'attack': 'gaussian_noise',
        'num_byzantine': 3
    },
    {
        'name': 'extreme_noattack',
        'description': 'Extreme non-IID + No attack (baseline)',
        'alpha': 0.1,
        'attack': None,
        'num_byzantine': 0
    }
]


def create_experiment_config(experiment: dict) -> dict:
    """Create experiment configuration by modifying base mini_validation.yaml"""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'mini_validation.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"mini_validation_{experiment['name']}"

    # Set heterogeneity level
    config['data_split']['alpha'] = experiment['alpha']

    # Set Byzantine configuration
    config['byzantine']['num_byzantine'] = experiment['num_byzantine']
    config['federated']['num_byzantine'] = experiment['num_byzantine']

    if experiment['attack'] is None:
        # No attack scenario
        config['byzantine']['attack_types'] = []
    else:
        # Set attack type
        config['byzantine']['attack_types'] = [experiment['attack']]

    # Add experiment metadata
    config['mini_validation'] = {
        'scenario': experiment['name'],
        'description': experiment['description'],
        'alpha': experiment['alpha'],
        'attack': experiment['attack']
    }

    return config


def main():
    """Run mini-validation suite and generate decision."""
    print("\n" + "=" * 70)
    print("🔬 MINI-VALIDATION SUITE: PoGQ vs Multi-Krum")
    print("=" * 70)
    print("\nPurpose: Validate PoGQ beats Multi-Krum on hardest cases")
    print("         BEFORE committing to 100+ GPU hours")
    print("\nStrategy: Test 5 critical scenarios covering:")
    print("  1. Hardest single attack (adaptive)")
    print("  2. Hardest coordinated attack (sybil)")
    print("  3. Control scenario (IID data)")
    print("  4. Moderate difficulty (moderate non-IID)")
    print("  5. Best case (no attack baseline)")

    total_experiments = len(EXPERIMENTS)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 0.5} hours")

    print("\n" + "=" * 70)
    print("EXPERIMENTS:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']:20s} - {exp['description']}")
    print("=" * 70)

    results = []
    start_time = time.time()

    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{total_experiments}: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Alpha: {experiment['alpha']} | Attack: {experiment['attack']} | Byzantine: {experiment['num_byzantine']}")
        print(f"{'=' * 70}")

        # Create config
        config = create_experiment_config(experiment)

        # Save config to temp file
        temp_config_path = f"/tmp/mini_validation_{experiment['name']}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        try:
            runner = ExperimentRunner(temp_config_path)
            runner.run()

            # Extract results for both baselines
            multikrum_results = runner.results['baselines']['multikrum']
            pogq_results = runner.results['baselines']['pogq']

            multikrum_acc = multikrum_results['test_accuracy'][-1]
            pogq_acc = pogq_results['test_accuracy'][-1]
            improvement = pogq_acc - multikrum_acc

            results.append({
                'scenario': experiment['name'],
                'description': experiment['description'],
                'alpha': experiment['alpha'],
                'attack': experiment['attack'],
                'status': 'success',
                'multikrum_accuracy': multikrum_acc,
                'pogq_accuracy': pogq_acc,
                'improvement': improvement
            })

            print(f"\n✅ Results:")
            print(f"   Multi-Krum: {multikrum_acc * 100:.2f}%")
            print(f"   PoGQ:       {pogq_acc * 100:.2f}%")
            print(f"   Improvement: {improvement * 100:+.2f} percentage points")

            if improvement > 0.03:  # >3 percentage points
                print(f"   🌟 EXCELLENT: PoGQ wins by {improvement * 100:.2f} points!")
            elif improvement > 0:
                print(f"   ⚠️  MARGINAL: Only {improvement * 100:.2f} point improvement")
            else:
                print(f"   ❌ CONCERN: PoGQ LOSES by {abs(improvement) * 100:.2f} points")

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scenario': experiment['name'],
                'description': experiment['description'],
                'alpha': experiment['alpha'],
                'attack': experiment['attack'],
                'status': 'failed',
                'error': str(e)
            })

    # Summary and Decision
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏁 MINI-VALIDATION COMPLETE!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # Calculate decision
    successful_results = [r for r in results if r['status'] == 'success']

    if successful_results:
        avg_improvement = sum(r['improvement'] for r in successful_results) / len(successful_results)
        min_improvement = min(r['improvement'] for r in successful_results)
        max_improvement = max(r['improvement'] for r in successful_results)

        print("\n" + "=" * 70)
        print("📊 RESULTS SUMMARY:")
        print("=" * 70)

        for result in successful_results:
            print(f"\n  {result['scenario']}:")
            print(f"    Multi-Krum: {result['multikrum_accuracy'] * 100:.2f}%")
            print(f"    PoGQ:       {result['pogq_accuracy'] * 100:.2f}%")
            print(f"    Improvement: {result['improvement'] * 100:+.2f} percentage points")

        print("\n" + "=" * 70)
        print("📈 AGGREGATE METRICS:")
        print("=" * 70)
        print(f"  Average improvement: {avg_improvement * 100:+.2f} percentage points")
        print(f"  Min improvement:     {min_improvement * 100:+.2f} percentage points")
        print(f"  Max improvement:     {max_improvement * 100:+.2f} percentage points")

        # Decision Gate
        print("\n" + "=" * 70)
        print("🎯 DECISION GATE:")
        print("=" * 70)

        if avg_improvement > 0.03 and min_improvement > 0.01:  # >3% and >1%
            print("✅ RECOMMENDATION: PROCEED TO FULL STAGE 1")
            print("   Reasoning:")
            print(f"   - Average improvement: {avg_improvement * 100:.2f} points (target: >3 points)")
            print(f"   - All scenarios show positive improvement")
            print(f"   - PoGQ demonstrates clear superiority")
            print("\n   Next steps:")
            print("   1. Run full Stage 1 Sets A, B, D (43 experiments)")
            print("   2. Expect ~100 GPU hours")
            print("   3. Proceed with confidence!")
            decision = "PROCEED"

        elif avg_improvement > 0.01:  # >1%
            print("⚠️  RECOMMENDATION: DEBUG AND TUNE")
            print("   Reasoning:")
            print(f"   - Average improvement: {avg_improvement * 100:.2f} points (marginal)")
            print(f"   - PoGQ shows promise but not dominant")
            print("\n   Next steps:")
            print("   1. Analyze why improvement is marginal")
            print("   2. Tune hyperparameters (quality_threshold, reputation_decay)")
            print("   3. Re-run mini-validation with optimized settings")
            print("   4. Only proceed to Stage 1 if improvement increases")
            decision = "TUNE"

        else:
            print("❌ RECOMMENDATION: STOP AND ANALYZE")
            print("   Reasoning:")
            print(f"   - Average improvement: {avg_improvement * 100:.2f} points (negative or minimal)")
            print(f"   - PoGQ does NOT beat Multi-Krum")
            print("\n   Next steps:")
            print("   1. Analyze failure modes - why doesn't PoGQ work?")
            print("   2. Check implementation correctness")
            print("   3. Review quality scoring function")
            print("   4. Consider alternative approaches")
            print("   5. DO NOT proceed to Stage 1 until PoGQ wins")
            decision = "STOP"

        # Save summary
        summary_path = project_root / 'experiments' / 'results' / f"mini_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            yaml.dump({
                'suite': 'mini_validation',
                'timestamp': datetime.now().isoformat(),
                'total_experiments': total_experiments,
                'elapsed_hours': elapsed_time / 3600,
                'avg_improvement': float(avg_improvement),
                'min_improvement': float(min_improvement),
                'max_improvement': float(max_improvement),
                'decision': decision,
                'results': results
            }, f)

        print(f"\n✅ Summary saved: {summary_path}")

    else:
        print("\n❌ No successful experiments - cannot make decision")
        print("   Recommendation: Fix experimental setup and re-run")


if __name__ == "__main__":
    main()
