#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Suite Orchestrator

Runs all 35 Byzantine attack experiments (7 attacks × 5 baselines).
Each experiment is independent with its own config generated on-the-fly.

Total experiments: 35
Estimated time: ~3 hours on GPU (5-6 minutes per experiment)
"""

import sys
import os
from pathlib import Path
import yaml
import subprocess
import json
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ByzantineSuiteRunner:
    """Orchestrates 35 Byzantine attack experiments"""

    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
        self.results_dir = Path("/tmp/byzantine_results")
        self.results_dir.mkdir(exist_ok=True)

        # Attack types
        self.attacks = [
            "gaussian_noise",
            "sign_flip",
            "label_flip",
            "targeted_poison",
            "model_replacement",
            "adaptive",
            "sybil"
        ]

        # Baselines
        self.baselines = [
            "fedavg",
            "krum",
            "multikrum",
            "bulyan",
            "median"
        ]

        # Results tracking
        self.results = {}
        self.completed = 0
        self.failed = 0
        self.total = len(self.attacks) * len(self.baselines)

    def _load_base_config(self):
        """Load base configuration"""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)

    def _generate_experiment_config(self, attack: str, baseline: str) -> dict:
        """
        Generate config for specific attack × baseline combination.

        Args:
            attack: Attack type (e.g., "gaussian_noise")
            baseline: Baseline name (e.g., "fedavg")

        Returns:
            Config dict for this experiment
        """
        # Deep copy base config
        config = yaml.safe_load(yaml.dump(self.base_config))

        # Set experiment name
        config['experiment_name'] = f"byzantine_{attack}_{baseline}"
        config['description'] = f"Byzantine attack: {attack} vs {baseline}"

        # Set single attack type (not list)
        config['byzantine']['attack_type'] = attack

        # Set single baseline (not list)
        baseline_config = self._get_baseline_config(baseline)
        config['baselines'] = [baseline_config]

        # Add federated section (required by runner.py)
        if 'federated' not in config:
            config['federated'] = {
                'num_clients': config['data_split']['num_clients'],
                'batch_size': config['training']['batch_size'],
                'local_epochs': config['training']['local_epochs'],
                'num_rounds': config['training']['num_rounds'],
                'learning_rate': config['training']['learning_rate'],
                'fraction_clients': 1.0,
                'num_byzantine': config['byzantine']['num_byzantine']
            }

        return config

    def _get_baseline_config(self, baseline: str) -> dict:
        """Get configuration for specific baseline"""
        baseline_configs = {
            "fedavg": {
                "name": "fedavg",
                "params": {"mu": 0.0}
            },
            "krum": {
                "name": "krum",
                "params": {"num_byzantine": 3}
            },
            "multikrum": {
                "name": "multikrum",
                "params": {
                    "num_byzantine": 3,
                    "k": 7
                }
            },
            "bulyan": {
                "name": "bulyan",
                "params": {"num_byzantine": 3}
            },
            "median": {
                "name": "median",
                "params": {}
            }
        }
        return baseline_configs[baseline]

    def run_experiment(self, attack: str, baseline: str) -> bool:
        """
        Run single experiment.

        Args:
            attack: Attack type
            baseline: Baseline name

        Returns:
            True if successful, False if failed
        """
        experiment_num = self.completed + self.failed + 1

        print(f"\n{'='*70}")
        print(f"Experiment {experiment_num}/{self.total}")
        print(f"{'='*70}")
        print(f"🚀 Starting: Attack={attack} vs Defense={baseline}")

        # Generate config
        config = self._generate_experiment_config(attack, baseline)

        # Save temporary config
        config_path = self.results_dir / f"config_{attack}_{baseline}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        log_path = self.results_dir / f"log_{attack}_{baseline}.log"

        try:
            cmd = [
                "nix", "develop", "--command",
                "python", "-u", "experiments/runner.py",  # -u for unbuffered output
                "--config", str(config_path)
            ]

            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=1800,  # 30 minutes max per experiment (100 rounds can take time)
                    cwd=project_root
                )

            if result.returncode == 0:
                print(f"✅ Success: {attack} vs {baseline}")
                self.completed += 1

                # Try to extract results
                self._extract_results(attack, baseline, log_path)

                return True
            else:
                print(f"❌ Failed: {attack} vs {baseline} (exit code {result.returncode})")
                print(f"   Log: {log_path}")
                self.failed += 1
                return False

        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout: {attack} vs {baseline} (>10 minutes)")
            self.failed += 1
            return False
        except Exception as e:
            print(f"❌ Error: {attack} vs {baseline} - {e}")
            self.failed += 1
            return False
        finally:
            # Progress update
            progress = ((self.completed + self.failed) / self.total) * 100
            print(f"Progress: {progress:.0f}% ({self.completed}/{self.total} complete, {self.failed} failed)")

    def _extract_results(self, attack: str, baseline: str, log_path: Path):
        """Extract final accuracy from log file"""
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()

            # Look for final test accuracy
            import re
            accuracy_match = re.search(r'Final test accuracy: ([\d.]+)', log_content)

            if accuracy_match:
                accuracy = float(accuracy_match.group(1))

                if attack not in self.results:
                    self.results[attack] = {}

                self.results[attack][baseline] = {
                    "accuracy": accuracy,
                    "success": True
                }
            else:
                # No accuracy found - experiment may have failed
                if attack not in self.results:
                    self.results[attack] = {}

                self.results[attack][baseline] = {
                    "accuracy": 0.0,
                    "success": False
                }
        except Exception as e:
            print(f"   Warning: Could not extract results - {e}")

    def run_all(self):
        """Run all 35 experiments"""
        print("="*70)
        print("🔐 Byzantine Attack Robustness Test Suite")
        print("="*70)
        print(f"Total experiments: {self.total}")
        print(f"Attacks: {len(self.attacks)} types")
        print(f"Baselines: {len(self.baselines)} defenses")
        print(f"Estimated time: ~{self.total * 5} minutes on GPU")
        print()

        start_time = datetime.now()

        # Run all combinations
        for attack in self.attacks:
            for baseline in self.baselines:
                self.run_experiment(attack, baseline)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        # Summary
        print("\n" + "="*70)
        print("📊 BYZANTINE ATTACK SUITE COMPLETE")
        print("="*70)
        print(f"Total experiments: {self.total}")
        print(f"Completed: {self.completed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {(self.completed/self.total)*100:.1f}%")
        print(f"Total time: {duration:.1f} minutes")
        print()

        # Save results
        self._save_results()

        # Generate visualizations if we have results
        if self.completed > 0:
            self._generate_heatmap()

        return self.completed == self.total

    def _save_results(self):
        """Save results to JSON"""
        results_file = self.results_dir / "byzantine_suite_results.json"

        output = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "results": self.results
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"📁 Results saved to: {results_file}")

    def _generate_heatmap(self):
        """Generate attack effectiveness heatmap"""
        print("\n🎨 Generating attack effectiveness heatmap...")

        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create matrix
            matrix = np.zeros((len(self.attacks), len(self.baselines)))

            for i, attack in enumerate(self.attacks):
                for j, baseline in enumerate(self.baselines):
                    if attack in self.results and baseline in self.results[attack]:
                        matrix[i, j] = self.results[attack][baseline].get("accuracy", 0.0)

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.0,
                vmax=1.0,
                xticklabels=self.baselines,
                yticklabels=self.attacks,
                cbar_kws={'label': 'Final Test Accuracy'}
            )
            plt.title('Byzantine Attack Effectiveness Heatmap\n(Higher = Better Defense)')
            plt.xlabel('Defense Mechanism')
            plt.ylabel('Attack Type')
            plt.tight_layout()

            heatmap_path = self.results_dir / "byzantine_attack_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.savefig(heatmap_path.with_suffix('.pdf'), bbox_inches='tight')

            print(f"✅ Heatmap saved to: {heatmap_path}")

        except ImportError:
            print("⚠️  Matplotlib/seaborn not available - skipping heatmap")
        except Exception as e:
            print(f"⚠️  Could not generate heatmap: {e}")


def main():
    """Main entry point"""
    print("🚀 Starting Byzantine Suite Orchestrator...")
    print(f"Working directory: {Path.cwd()}")

    # Check if base config exists
    base_config = "experiments/configs/mnist_byzantine_attacks.yaml"
    print(f"Looking for config: {base_config}")

    if not Path(base_config).exists():
        print(f"❌ Base config not found: {base_config}")
        return 1

    print("✅ Config found, creating runner...")

    # Create runner
    runner = ByzantineSuiteRunner(base_config)

    print("✅ Runner created, starting experiments...")

    # Run all experiments
    success = runner.run_all()

    return 0 if success else 1


if __name__ == "__main__":
    print("=" * 70)
    print("BYZANTINE SUITE STARTING")
    print("=" * 70)
    sys.exit(main())
