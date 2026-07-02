#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Matrix Experiment Runner
========================

Executes full experimental matrix from sanity_slice.yaml config:
- 2 datasets × 8 attacks × 8 defenses × 2 seeds = 256 experiments

Generates individual experiment configs and runs them sequentially.

Author: Luminous Dynamics
Date: November 8, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
import itertools
from datetime import datetime
from typing import Dict, List
from experiments.runner import ExperimentRunner


def generate_matrix_configs(matrix_config: Dict) -> List[Dict]:
    """
    Generate all experiment configurations from matrix spec

    Args:
        matrix_config: Matrix configuration from sanity_slice.yaml

    Returns:
        List of individual experiment configs
    """
    configs = []

    # Extract matrix dimensions
    experiment_name = matrix_config['experiment']['name']
    seeds = matrix_config['experiment']['seeds']
    datasets = matrix_config['data']['datasets']
    attacks = matrix_config['attack_matrix']['attacks']
    defenses = matrix_config['defenses']
    byzantine_ratio = matrix_config['attack_matrix']['byzantine_ratio']

    # Generate all combinations
    for seed, dataset_config, attack, defense in itertools.product(
        seeds, datasets, attacks, defenses
    ):
        # Create individual experiment config
        config = {
            'experiment_name': f"{experiment_name}_{dataset_config['name']}_{attack}_{defense}_seed{seed}",
            'seed': seed,

            'dataset': {
                'name': dataset_config['name'],
                'data_dir': 'datasets'
            },

            'data_split': {
                'type': 'dirichlet' if dataset_config.get('non_iid', False) else 'iid',
                'alpha': dataset_config.get('dirichlet_alpha', 0.5),
                'seed': seed
            },

            'model': {
                # Use model_type from dataset config (allows per-dataset architectures)
                'type': dataset_config.get('model_type', 'simple_cnn'),
                'params': {
                    # Use num_classes from dataset config (allows per-dataset class counts)
                    'num_classes': dataset_config.get('num_classes', 10)
                }
            },

            'federated': {
                'num_clients': 100,
                'batch_size': 32,
                'learning_rate': 0.01,
                'local_epochs': 1,
                'fraction_clients': 1.0,
                'conformal_alpha': matrix_config.get('statistics', {}).get('alpha', 0.10)
            },

            'attack': {
                'type': attack,
                'byzantine_ratio': byzantine_ratio
            },

            'baselines': [defense],  # Single defense per experiment

            'training': {
                'num_rounds': 100,
                'eval_every': 10
            },

            'output_dir': 'results'
        }

        configs.append(config)

    return configs


def run_matrix(matrix_config_path: str, start: int = 1, end: int = None):
    """Execute full or partial experimental matrix

    Args:
        matrix_config_path: Path to matrix YAML config
        start: First experiment to run (1-based, default: 1)
        end: Last experiment to run (1-based, default: all)
    """
    # Load matrix config
    with open(matrix_config_path, 'r') as f:
        matrix_config = yaml.safe_load(f)

    # Generate all experiment configs
    print("=" * 70)
    print("🔬 Matrix Experiment Runner")
    print("=" * 70)
    print(f"\nMatrix Configuration: {matrix_config_path}")

    configs = generate_matrix_configs(matrix_config)
    total_experiments = len(configs)

    # Apply start/end filtering
    if end is None:
        end = total_experiments

    # Validate range
    if start < 1 or start > total_experiments:
        raise ValueError(f"Start must be between 1 and {total_experiments}")
    if end < start or end > total_experiments:
        raise ValueError(f"End must be between {start} and {total_experiments}")

    # Filter configs to requested range
    configs_to_run = configs[start-1:end]
    num_to_run = len(configs_to_run)

    print(f"\n📊 Experimental Matrix:")
    print(f"   Datasets: {len(matrix_config['data']['datasets'])}")
    print(f"   Attacks: {len(matrix_config['attack_matrix']['attacks'])}")
    print(f"   Defenses: {len(matrix_config['defenses'])}")
    print(f"   Seeds: {len(matrix_config['experiment']['seeds'])}")
    print(f"   Total Experiments: {total_experiments}")
    print(f"   Running: {start} to {end} ({num_to_run} experiments)")

    # Run experiments
    start_time = datetime.now()
    print(f"\n🚀 Starting matrix execution at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Estimated time: ~{num_to_run * 7.5 / 60:.1f} hours (sequential)\n")

    for idx, config in enumerate(configs_to_run, start):
        print("\n" + "=" * 70)
        print(f"Experiment {idx}/{total_experiments}")
        print(f"  Name: {config['experiment_name']}")
        print(f"  Dataset: {config['dataset']['name']}")
        print(f"  Attack: {config['attack']['type']}")
        print(f"  Defense: {config['baselines'][0]}")
        print(f"  Seed: {config['seed']}")
        print("=" * 70)

        try:
            # Create temp config file
            temp_config_path = f"/tmp/experiment_config_{idx}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)

            # Run experiment
            runner = ExperimentRunner(temp_config_path)
            runner.run()

            # Cleanup temp config
            Path(temp_config_path).unlink()

            print(f"\n✅ Experiment {idx}/{total_experiments} complete")

        except Exception as e:
            print(f"\n❌ Experiment {idx}/{total_experiments} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with remaining experiments...")
            # Auto-continue on error for background execution

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("🎉 Matrix Execution Complete!")
    print("=" * 70)
    print(f"   Total time: {duration}")
    print(f"   Experiments completed: {i}/{total_experiments}")
    print(f"   Results directory: results/")
    print("\n💡 Next steps:")
    print("   1. Run CI gates: pytest -v tests/test_ci_gates.py")
    print("   2. Check artifacts: ls -lh results/artifacts_*/")
    print("   3. Verify first draft Table II data")


def main():
    parser = argparse.ArgumentParser(description='Run full experimental matrix')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to matrix config YAML file (e.g., configs/sanity_slice.yaml)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Start experiment number (1-based, default: 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End experiment number (1-based, default: run all)'
    )
    args = parser.parse_args()

    run_matrix(args.config, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
