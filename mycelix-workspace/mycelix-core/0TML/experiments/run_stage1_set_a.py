#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stage 1 Set A Runner: Primary Comparison

Runs 21 experiments (7 attacks × 3 defenses) on extreme non-IID data.

Usage:
    python run_stage1_set_a.py

Estimated time: ~35 hours with parallel execution
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
from datetime import datetime
from runner import ExperimentRunner

# Attack types
ATTACKS = [
    'gaussian_noise',
    'sign_flip',
    'label_flip',
    'targeted_poison',
    'model_replacement',
    'adaptive',
    'sybil'
]

# Defenses to compare
DEFENSES = [
    'fedavg',      # Baseline (no defense)
    'multikrum',   # Best classical
    'pogq'         # Ours
]


def create_experiment_config(attack: str, defense: str) -> dict:
    """Create experiment configuration for specific attack and defense."""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'stage1_set_a_primary.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"set_a_{defense}_{attack}"
    config['byzantine']['attack_types'] = [attack]  # Single attack

    # Select single baseline
    if defense == 'fedavg':
        config['baselines'] = [{'name': 'fedavg'}]
    elif defense == 'multikrum':
        config['baselines'] = [{
            'name': 'multikrum',
            'params': {'num_byzantine': 3, 'k': 7}
        }]
    elif defense == 'pogq':
        config['baselines'] = [{
            'name': 'pogq',
            'params': {'quality_threshold': 0.3, 'reputation_decay': 0.95}
        }]

    return config


def main():
    """Run all Set A experiments."""
    print("\n" + "=" * 70)
    print("Stage 1 Set A: Primary Comparison")
    print("Extreme Non-IID | 30% Byzantine | 7 Attacks × 3 Defenses")
    print("=" * 70)

    total_experiments = len(ATTACKS) * len(DEFENSES)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 10} minutes (~{total_experiments * 10 / 60:.1f} hours)")

    results = []
    start_time = time.time()

    for i, (attack, defense) in enumerate(
        [(a, d) for a in ATTACKS for d in DEFENSES], 1
    ):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{total_experiments}: {defense.upper()} vs {attack}")
        print(f"{'=' * 70}")

        # Create config
        config = create_experiment_config(attack, defense)

        # Save config to temp file
        temp_config_path = f"/tmp/set_a_{defense}_{attack}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        try:
            runner = ExperimentRunner(temp_config_path)
            runner.run()
            results.append({
                'attack': attack,
                'defense': defense,
                'status': 'success',
                'final_accuracy': runner.results['baselines'][defense]['test_accuracy'][-1]
            })
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            results.append({
                'attack': attack,
                'defense': defense,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Set A Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # Save summary
    summary_path = project_root / 'experiments' / 'results' / f"set_a_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        yaml.dump({
            'set': 'A',
            'total_experiments': total_experiments,
            'elapsed_hours': elapsed_time / 3600,
            'results': results
        }, f)

    print(f"\n✅ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
