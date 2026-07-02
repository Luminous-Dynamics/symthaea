#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stage 1 Set C Runner: Bulyan Theory Validation

Runs 7 experiments (7 attacks × Bulyan @ 20% Byzantine).

Purpose: Validate Bulyan achieves high accuracy when f < n/3 constraint is met.

Usage:
    python run_stage1_set_c.py

Estimated time: ~35 hours
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
from datetime import datetime
from runner import ExperimentRunner

# All attack types
ATTACKS = [
    'gaussian_noise',
    'sign_flip',
    'label_flip',
    'targeted_poison',
    'model_replacement',
    'adaptive',
    'sybil'
]


def create_experiment_config(attack: str) -> dict:
    """Create experiment configuration for specific attack."""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'stage1_set_c_bulyan.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"set_c_bulyan_{attack}"
    config['byzantine']['attack_types'] = [attack]

    return config


def main():
    """Run all Set C experiments."""
    print("\n" + "=" * 70)
    print("Stage 1 Set C: Bulyan Theory Validation")
    print("Extreme Non-IID | 20% Byzantine (f < n/3) | 7 Attacks")
    print("=" * 70)

    total_experiments = len(ATTACKS)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Byzantine fraction: 20% (2/10 clients)")
    print(f"Theoretical guarantee: f < n/3 (2 < 3.33 ✓)")
    print(f"Estimated time: ~{total_experiments * 5} hours")

    results = []
    start_time = time.time()

    for i, attack in enumerate(ATTACKS, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{total_experiments}: Bulyan vs {attack}")
        print(f"{'=' * 70}")

        # Create config
        config = create_experiment_config(attack)

        # Save config to temp file
        temp_config_path = f"/tmp/set_c_bulyan_{attack}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        try:
            runner = ExperimentRunner(temp_config_path)
            runner.run()

            # Extract results
            bulyan_results = runner.results['baselines']['bulyan']

            results.append({
                'attack': attack,
                'status': 'success',
                'final_accuracy': bulyan_results['test_accuracy'][-1],
                'final_train_loss': bulyan_results['train_loss'][-1]
            })

            print(f"✅ Final accuracy: {bulyan_results['test_accuracy'][-1]:.2f}%")

        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            results.append({
                'attack': attack,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Set C Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # Accuracy analysis
    if any(r['status'] == 'success' for r in results):
        print("\n📊 Bulyan Performance (20% Byzantine):")
        for result in results:
            if result['status'] == 'success':
                print(f"  {result['attack']:20s}: {result['final_accuracy']:.2f}%")

        avg_accuracy = sum(r['final_accuracy'] for r in results if r['status'] == 'success') / sum(1 for r in results if r['status'] == 'success')
        print(f"\n  Average accuracy: {avg_accuracy:.2f}%")

    # Save summary
    summary_path = project_root / 'experiments' / 'results' / f"set_c_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        yaml.dump({
            'set': 'C',
            'total_experiments': total_experiments,
            'byzantine_fraction': 0.20,
            'theory_compliance': 'f < n/3 (2 < 3.33)',
            'elapsed_hours': elapsed_time / 3600,
            'results': results
        }, f)

    print(f"\n✅ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
