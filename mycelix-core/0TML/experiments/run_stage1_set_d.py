#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Stage 1 Set D Runner: PoGQ Deep Dive

Runs 4 experiments (4 worst-case scenarios × PoGQ).

Purpose: Characterize PoGQ limits and failure modes.

Usage:
    python run_stage1_set_d.py

Estimated time: ~20 hours
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import time
from datetime import datetime
from runner import ExperimentRunner

# Worst-case scenarios
SCENARIOS = [
    {
        'name': 'no_attack',
        'attack_type': None,
        'description': 'Clean baseline - no Byzantine clients active'
    },
    {
        'name': 'adaptive_attack',
        'attack_type': 'adaptive',
        'description': 'Adaptive attack designed to evade quality metrics'
    },
    {
        'name': 'sybil_attack',
        'attack_type': 'sybil',
        'description': 'Coordinated attack by multiple Byzantine clients'
    },
    {
        'name': 'combined_adaptive_sybil',
        'attack_type': 'adaptive+sybil',
        'description': 'Combined adaptive evasion + coordination (worst case)'
    }
]


def create_experiment_config(scenario: dict) -> dict:
    """Create experiment configuration for specific scenario."""

    # Load base config
    config_path = project_root / 'experiments' / 'configs' / 'stage1_set_d_pogq_deep_dive.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for specific experiment
    config['experiment_name'] = f"set_d_pogq_{scenario['name']}"

    # Handle no-attack scenario
    if scenario['attack_type'] is None:
        # No Byzantine clients
        config['byzantine']['num_byzantine'] = 0
        config['federated']['num_byzantine'] = 0
        config['byzantine']['attack_types'] = []
    else:
        # Set attack type
        attack_types = scenario['attack_type'].split('+')
        config['byzantine']['attack_types'] = attack_types

    # Add scenario metadata
    config['scenario'] = scenario

    return config


def main():
    """Run all Set D experiments."""
    print("\n" + "=" * 70)
    print("Stage 1 Set D: PoGQ Deep Dive - Worst-Case Scenarios")
    print("Extreme Non-IID | 30% Byzantine | 4 Scenarios")
    print("=" * 70)

    total_experiments = len(SCENARIOS)
    print(f"\nTotal experiments: {total_experiments}")
    print("\nScenarios:")
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"  {i}. {scenario['name']:25s} - {scenario['description']}")

    print(f"\nEstimated time: ~{total_experiments * 5} hours")

    results = []
    start_time = time.time()

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{total_experiments}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'=' * 70}")

        # Create config
        config = create_experiment_config(scenario)

        # Save config to temp file
        temp_config_path = f"/tmp/set_d_pogq_{scenario['name']}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        try:
            runner = ExperimentRunner(temp_config_path)
            runner.run()

            # Extract PoGQ-specific results
            pogq_results = runner.results['baselines']['pogq']

            results.append({
                'scenario': scenario['name'],
                'attack_type': scenario['attack_type'],
                'status': 'success',
                'final_accuracy': pogq_results['test_accuracy'][-1],
                'final_train_loss': pogq_results['train_loss'][-1],
                'avg_quality_score': pogq_results.get('avg_quality_score', [])[-1] if pogq_results.get('avg_quality_score') else None,
                'num_accepted': pogq_results.get('num_accepted', [])[-1] if pogq_results.get('num_accepted') else None
            })

            print(f"✅ Final accuracy: {pogq_results['test_accuracy'][-1]:.2f}%")
            if 'avg_quality_score' in pogq_results and pogq_results['avg_quality_score']:
                print(f"   Avg quality score: {pogq_results['avg_quality_score'][-1]:.3f}")
            if 'num_accepted' in pogq_results and pogq_results['num_accepted']:
                print(f"   Clients accepted: {pogq_results['num_accepted'][-1]}/10")

        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            results.append({
                'scenario': scenario['name'],
                'attack_type': scenario['attack_type'],
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Set D Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{total_experiments}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total_experiments}")

    # PoGQ performance analysis
    if any(r['status'] == 'success' for r in results):
        print("\n📊 PoGQ Performance Across Scenarios:")
        for result in results:
            if result['status'] == 'success':
                print(f"\n  {result['scenario']}:")
                print(f"    Accuracy: {result['final_accuracy']:.2f}%")
                if result.get('avg_quality_score') is not None:
                    print(f"    Avg Quality: {result['avg_quality_score']:.3f}")
                if result.get('num_accepted') is not None:
                    print(f"    Accepted: {result['num_accepted']}/10 clients")

        # Calculate degradation from baseline
        baseline = next((r for r in results if r['scenario'] == 'no_attack' and r['status'] == 'success'), None)
        if baseline:
            print(f"\n📉 Degradation from Clean Baseline ({baseline['final_accuracy']:.2f}%):")
            for result in results:
                if result['status'] == 'success' and result['scenario'] != 'no_attack':
                    degradation = baseline['final_accuracy'] - result['final_accuracy']
                    print(f"  {result['scenario']:25s}: -{degradation:.2f}%")

    # Save summary
    summary_path = project_root / 'experiments' / 'results' / f"set_d_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        yaml.dump({
            'set': 'D',
            'total_experiments': total_experiments,
            'elapsed_hours': elapsed_time / 3600,
            'results': results
        }, f)

    print(f"\n✅ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
