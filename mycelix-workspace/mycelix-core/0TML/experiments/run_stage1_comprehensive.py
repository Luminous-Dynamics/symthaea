#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stage 1: Comprehensive PoGQ Validation

Runs 43 carefully designed experiments to comprehensively validate that PoGQ
beats all other baselines across diverse scenarios.

Mini-validation Results (Oct 7, 2025):
- Average improvement: +23.2 percentage points
- PoGQ dominates on extreme non-IID + attacks: +37-40 points
- Decision: PROCEED TO FULL STAGE 1 ✅

Experimental Design:
- Set A: Core Baselines (9 experiments)
  * FedAvg, FedProx, SCAFFOLD vs PoGQ
  * 3 heterogeneity levels × 3 baselines

- Set B: Byzantine-Resilient (14 experiments)
  * Krum, Multi-Krum vs PoGQ
  * 7 attack types × 2 baselines

- Set D: Comprehensive Attack Evaluation (20 experiments)
  * PoGQ deep dive across all scenarios
  * 7 attacks × 3 heterogeneity levels (minus 1 duplicate)

Total: 43 experiments × 50 rounds × ~6 min = ~100 GPU hours

Usage:
    python run_stage1_comprehensive.py [--set A|B|D|all]
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
import argparse
from datetime import datetime
from runner import ExperimentRunner

# Attack types
ALL_ATTACKS = [
    'gaussian_noise',
    'sign_flip',
    'label_flip',
    'targeted_poison',
    'model_replacement',
    'adaptive',
    'sybil'
]

# Heterogeneity levels
HETEROGENEITY = {
    'iid': 100.0,
    'moderate': 1.0,
    'extreme': 0.1
}


def create_experiment_config(
    set_name: str,
    exp_name: str,
    description: str,
    baseline: str,
    alpha: float,
    attack: str,
    num_byzantine: int
) -> dict:
    """Create experiment configuration."""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'stage1_comprehensive.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"stage1_{set_name}_{exp_name}"
    config['description'] = description

    # Set heterogeneity level
    config['data_split']['alpha'] = alpha

    # Set Byzantine configuration
    config['byzantine']['num_byzantine'] = num_byzantine
    config['federated']['num_byzantine'] = num_byzantine

    if attack is None:
        config['byzantine']['attack_types'] = []
    else:
        config['byzantine']['attack_types'] = [attack]

    # Set baseline to test (always include PoGQ for comparison)
    if baseline == 'pogq_only':
        config['baselines'] = [
            {'name': 'pogq', 'params': {
                'quality_threshold': 0.3,
                'reputation_decay': 0.95
            }}
        ]
    else:
        config['baselines'] = [
            {'name': baseline},
            {'name': 'pogq', 'params': {
                'quality_threshold': 0.3,
                'reputation_decay': 0.95
            }}
        ]

    # Add experiment metadata
    config['stage1_metadata'] = {
        'set': set_name,
        'experiment': exp_name,
        'baseline': baseline,
        'alpha': float(alpha),
        'attack': attack,
        'num_byzantine': num_byzantine
    }

    return config


def generate_set_a_experiments():
    """
    Set A: Core Baselines vs PoGQ (9 experiments)

    Tests PoGQ against standard FL baselines across different data heterogeneity.
    Uses adaptive attack as it's the hardest for non-robust methods.
    """
    experiments = []
    baselines = ['fedavg', 'fedprox', 'scaffold']

    for baseline in baselines:
        for het_name, alpha in HETEROGENEITY.items():
            exp_name = f"{baseline}_{het_name}_adaptive"
            description = f"Set A: {baseline.upper()} vs PoGQ ({het_name} α={alpha}, adaptive attack)"

            experiments.append({
                'set': 'A',
                'name': exp_name,
                'description': description,
                'baseline': baseline,
                'alpha': alpha,
                'attack': 'adaptive',
                'num_byzantine': 3
            })

    return experiments


def generate_set_b_experiments():
    """
    Set B: Byzantine-Resilient vs PoGQ (14 experiments)

    Tests PoGQ against state-of-the-art Byzantine-resilient baselines
    across all attack types. Uses extreme non-IID to stress-test robustness.
    """
    experiments = []
    baselines = ['krum', 'multikrum']

    for baseline in baselines:
        for attack in ALL_ATTACKS:
            exp_name = f"{baseline}_extreme_{attack}"
            description = f"Set B: {baseline.upper()} vs PoGQ (extreme α=0.1, {attack} attack)"

            experiments.append({
                'set': 'B',
                'name': exp_name,
                'description': description,
                'baseline': baseline,
                'alpha': 0.1,  # Extreme non-IID
                'attack': attack,
                'num_byzantine': 3
            })

    return experiments


def generate_set_d_experiments():
    """
    Set D: Comprehensive Attack Evaluation (20 experiments)

    Deep dive into PoGQ performance across all combinations of
    data heterogeneity and attack types. Excludes duplicate from
    mini-validation (extreme + adaptive).
    """
    experiments = []

    for het_name, alpha in HETEROGENEITY.items():
        for attack in ALL_ATTACKS:
            # Skip duplicate from mini-validation
            if het_name == 'extreme' and attack == 'adaptive':
                continue

            exp_name = f"pogq_{het_name}_{attack}"
            description = f"Set D: PoGQ deep dive ({het_name} α={alpha}, {attack} attack)"

            experiments.append({
                'set': 'D',
                'name': exp_name,
                'description': description,
                'baseline': 'pogq_only',  # No comparison baseline
                'alpha': alpha,
                'attack': attack,
                'num_byzantine': 3
            })

    # Add no-attack baselines for all heterogeneity levels
    for het_name, alpha in HETEROGENEITY.items():
        if het_name == 'extreme':
            continue  # Already have this from mini-validation

        exp_name = f"pogq_{het_name}_noattack"
        description = f"Set D: PoGQ baseline ({het_name} α={alpha}, no attack)"

        experiments.append({
            'set': 'D',
            'name': exp_name,
            'description': description,
            'baseline': 'pogq_only',
            'alpha': alpha,
            'attack': None,
            'num_byzantine': 0
        })

    return experiments


def main():
    """Run Stage 1 comprehensive validation."""
    parser = argparse.ArgumentParser(description='Run Stage 1 experiments')
    parser.add_argument(
        '--set',
        type=str,
        choices=['A', 'B', 'D', 'all'],
        default='all',
        help='Which experimental set to run (A, B, D, or all)'
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔬 STAGE 1: COMPREHENSIVE PoGQ VALIDATION")
    print("=" * 70)
    print("\nMini-validation Results:")
    print("  ✅ Average improvement: +23.2 percentage points")
    print("  ✅ PoGQ dominates on hard cases: +37-40 points")
    print("  ✅ Decision: PROCEED TO FULL STAGE 1")

    # Generate experiments
    set_a = generate_set_a_experiments() if args.set in ['A', 'all'] else []
    set_b = generate_set_b_experiments() if args.set in ['B', 'all'] else []
    set_d = generate_set_d_experiments() if args.set in ['D', 'all'] else []

    all_experiments = set_a + set_b + set_d

    print(f"\nExperimental Design:")
    if set_a:
        print(f"  Set A: Core Baselines - {len(set_a)} experiments")
    if set_b:
        print(f"  Set B: Byzantine-Resilient - {len(set_b)} experiments")
    if set_d:
        print(f"  Set D: Comprehensive Attack Eval - {len(set_d)} experiments")

    total_experiments = len(all_experiments)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 6 / 60:.1f} hours ({total_experiments * 6} minutes)")

    print("\n" + "=" * 70)
    print("EXPERIMENT LIST:")
    for i, exp in enumerate(all_experiments, 1):
        print(f"  {i:2d}. [{exp['set']}] {exp['name']:35s} - {exp['description']}")
    print("=" * 70)

    results = []
    start_time = time.time()

    for i, experiment in enumerate(all_experiments, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{total_experiments} [{experiment['set']}]: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Alpha: {experiment['alpha']} | Attack: {experiment['attack']} | Byzantine: {experiment['num_byzantine']}")
        print(f"{'=' * 70}")

        # Create config
        config = create_experiment_config(
            set_name=experiment['set'],
            exp_name=experiment['name'],
            description=experiment['description'],
            baseline=experiment['baseline'],
            alpha=experiment['alpha'],
            attack=experiment['attack'],
            num_byzantine=experiment['num_byzantine']
        )

        # Save config to temp file
        temp_config_path = f"/tmp/stage1_{experiment['set']}_{experiment['name']}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        try:
            runner = ExperimentRunner(temp_config_path)
            runner.run()

            # Extract results
            result = {
                'set': experiment['set'],
                'experiment': experiment['name'],
                'description': experiment['description'],
                'baseline': experiment['baseline'],
                'alpha': experiment['alpha'],
                'attack': experiment['attack'],
                'status': 'success',
                'results': {}
            }

            # Extract metrics for each baseline tested
            for baseline_name, baseline_results in runner.results['baselines'].items():
                final_test_acc = baseline_results['test_accuracy'][-1]
                result['results'][baseline_name] = {
                    'test_accuracy': final_test_acc,
                    'train_accuracy': baseline_results['train_accuracy'][-1],
                    'test_loss': baseline_results['test_loss'][-1]
                }

            results.append(result)

            print(f"\n✅ Results:")
            for baseline_name, metrics in result['results'].items():
                print(f"   {baseline_name:15s}: {metrics['test_accuracy'] * 100:.2f}%")

            # If comparing to PoGQ, show improvement
            if 'pogq' in result['results'] and len(result['results']) > 1:
                baseline_name = [k for k in result['results'].keys() if k != 'pogq'][0]
                baseline_acc = result['results'][baseline_name]['test_accuracy']
                pogq_acc = result['results']['pogq']['test_accuracy']
                improvement = pogq_acc - baseline_acc

                print(f"\n   Improvement: {improvement * 100:+.2f} percentage points")

                if improvement > 0.05:
                    print(f"   🌟 EXCELLENT: PoGQ wins by {improvement * 100:.2f} points!")
                elif improvement > 0:
                    print(f"   ✅ POSITIVE: PoGQ wins by {improvement * 100:.2f} points")
                else:
                    print(f"   ⚠️  CONCERN: PoGQ loses by {abs(improvement) * 100:.2f} points")

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'set': experiment['set'],
                'experiment': experiment['name'],
                'description': experiment['description'],
                'baseline': experiment['baseline'],
                'alpha': experiment['alpha'],
                'attack': experiment['attack'],
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏁 STAGE 1 COMPLETE!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # Save comprehensive results
    results_dir = project_root / 'experiments' / 'results' / 'stage1'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"stage1_summary_{timestamp}.yaml"

    with open(summary_path, 'w') as f:
        yaml.dump({
            'stage': 'stage1_comprehensive',
            'timestamp': datetime.now().isoformat(),
            'total_experiments': total_experiments,
            'elapsed_hours': elapsed_time / 3600,
            'experiments': results
        }, f)

    print(f"\n✅ Summary saved: {summary_path}")
    print("\nNext steps:")
    print("  1. Analyze results with statistical significance tests")
    print("  2. Generate figures and tables for paper")
    print("  3. Write Stage 1 paper draft")


if __name__ == "__main__":
    main()
