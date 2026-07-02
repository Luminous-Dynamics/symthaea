#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stage 1 Set B Runner: Heterogeneity Analysis

Runs 18 experiments (3 data levels × 3 attacks × 2 defenses).

Usage:
    python run_stage1_set_b.py

Estimated time: ~30 hours with parallel execution
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
from datetime import datetime
from runner import ExperimentRunner

# Heterogeneity levels (Dirichlet alpha values)
HETEROGENEITY_LEVELS = {
    'iid': 100,          # Alpha → ∞ approaches IID
    'moderate': 1.0,     # Moderate non-IID
    'extreme': 0.1       # Extreme non-IID
}

# Representative attacks
ATTACKS = [
    'gaussian_noise',  # Simple
    'adaptive',        # Sophisticated
    'sybil'            # Coordinated
]

# Defenses to compare
DEFENSES = [
    'multikrum',  # Best classical
    'pogq'        # Ours
]


def create_experiment_config(
    heterogeneity: str,
    alpha: float,
    attack: str,
    defense: str
) -> dict:
    """Create experiment configuration for specific parameters."""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'stage1_set_b_heterogeneity.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"set_b_{heterogeneity}_{defense}_{attack}"

    # Set heterogeneity level
    config['data_split']['alpha'] = alpha

    # Set single attack
    config['byzantine']['attack_types'] = [attack]

    # Select single baseline
    if defense == 'multikrum':
        config['baselines'] = [{
            'name': 'multikrum',
            'params': {'num_byzantine': 3, 'k': 7}
        }]
    elif defense == 'pogq':
        config['baselines'] = [{
            'name': 'pogq',
            'params': {'quality_threshold': 0.3, 'reputation_decay': 0.95}
        }]

    # Add metadata for analysis
    config['save']['heterogeneity_level'] = heterogeneity
    config['save']['alpha'] = alpha

    return config


def main():
    """Run all Set B experiments."""
    print("\n" + "=" * 70)
    print("Stage 1 Set B: Heterogeneity Analysis")
    print("3 Data Levels × 3 Attacks × 2 Defenses")
    print("=" * 70)

    total_experiments = len(HETEROGENEITY_LEVELS) * len(ATTACKS) * len(DEFENSES)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 10} minutes (~{total_experiments * 10 / 60:.1f} hours)")

    results = []
    start_time = time.time()

    # Iterate through all combinations
    experiment_num = 0
    for het_level, alpha in HETEROGENEITY_LEVELS.items():
        for attack in ATTACKS:
            for defense in DEFENSES:
                experiment_num += 1

                print(f"\n{'=' * 70}")
                print(f"Experiment {experiment_num}/{total_experiments}")
                print(f"Heterogeneity: {het_level} (α={alpha})")
                print(f"Defense: {defense.upper()} vs Attack: {attack}")
                print(f"{'=' * 70}")

                # Create config
                config = create_experiment_config(het_level, alpha, attack, defense)

                # Save config to temp file
                temp_config_path = f"/tmp/set_b_{het_level}_{defense}_{attack}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f)

                # Run experiment
                try:
                    runner = ExperimentRunner(temp_config_path)
                    runner.run()
                    results.append({
                        'heterogeneity': het_level,
                        'alpha': alpha,
                        'attack': attack,
                        'defense': defense,
                        'status': 'success',
                        'final_accuracy': runner.results['baselines'][defense]['test_accuracy'][-1]
                    })
                except Exception as e:
                    print(f"❌ Experiment failed: {e}")
                    results.append({
                        'heterogeneity': het_level,
                        'alpha': alpha,
                        'attack': attack,
                        'defense': defense,
                        'status': 'failed',
                        'error': str(e)
                    })

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Set B Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # Degradation analysis
    print("\n📊 Degradation Curve Preview:")
    for defense in DEFENSES:
        print(f"\n{defense.upper()}:")
        for het_level in ['iid', 'moderate', 'extreme']:
            avg_acc = sum(r['final_accuracy'] for r in results
                         if r['heterogeneity'] == het_level
                         and r['defense'] == defense
                         and r['status'] == 'success') / len(ATTACKS)
            print(f"  {het_level:10s}: {avg_acc:.2f}%")

    # Save summary
    summary_path = project_root / 'experiments' / 'results' / f"set_b_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        yaml.dump({
            'set': 'B',
            'total_experiments': total_experiments,
            'elapsed_hours': elapsed_time / 3600,
            'results': results
        }, f)

    print(f"\n✅ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
