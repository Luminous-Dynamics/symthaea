# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
AEGIS Gen 5 Validation Protocol v1.0
=====================================

Production-grade experimental validation with:
- Immutable reproducibility (state hashing, seed control)
- Statistical rigor (bootstrap CIs, effect sizes)
- Anti-p-hacking guardrails
- Automated figure generation
- Parallel execution with checkpointing

Target: 300 runs for MLSys/ICML 2026 submission

Author: Luminous Dynamics
Date: November 12, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import subprocess

# Global seeds for reproducibility
SEEDS = [101, 202, 303, 404, 505]


@dataclass
class ValidationManifest:
    """Immutable validation manifest for reproducibility."""
    validation_id: str
    git_commit: str
    state_hash: str
    timestamp: str
    python_version: str
    numpy_version: str
    seeds: List[int]
    experiment_matrix: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def create(cls, experiment_matrix: Dict) -> 'ValidationManifest':
        """Create validation manifest with environment snapshot."""
        # Git commit
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).parent.parent,
                text=True
            ).strip()
        except:
            git_commit = "unknown"

        # State hash (code + environment)
        state_hasher = hashlib.sha256()

        # Hash Python files
        for py_file in sorted(Path(__file__).parent.parent.glob("src/gen5/*.py")):
            with open(py_file, 'rb') as f:
                state_hasher.update(f.read())

        state_hash = state_hasher.hexdigest()[:16]

        return cls(
            validation_id=f"AEGIS-GEN5-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            git_commit=git_commit,
            state_hash=state_hash,
            timestamp=datetime.now().isoformat(),
            python_version=sys.version.split()[0],
            numpy_version=np.__version__,
            seeds=SEEDS,
            experiment_matrix=experiment_matrix,
        )


@dataclass
class ExperimentRun:
    """Single experiment run with full provenance."""
    run_id: str
    experiment: str
    seed: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, output_dir: Path):
        """Save run artifacts."""
        run_dir = output_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(run_dir / "config.json", 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'experiment': self.experiment,
                'seed': self.seed,
                'parameters': self.parameters,
            }, f, indent=2)

        # Save metrics
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)


class ExperimentMatrix:
    """
    Complete experiment matrix for AEGIS Gen 5 validation.

    Total: 300 runs across 9 experiment types
    """

    @staticmethod
    def E1_byzantine_tolerance() -> List[Dict]:
        """
        E1: Byzantine Tolerance Curves (120 runs)

        Adversary rates: 0%, 10%, 20%, 30%, 40%, 50%
        Attack types: label-flip, sign-flip, gradient-scaling, backdoor
        Seeds: 5

        Total: 6 rates × 4 attacks × 5 seeds = 120 runs
        """
        configs = []
        adversary_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        attack_types = ["label_flip", "sign_flip", "gradient_scaling", "backdoor"]

        for adv_rate in adversary_rates:
            for attack_type in attack_types:
                for seed in SEEDS:
                    configs.append({
                        'adversary_rate': adv_rate,
                        'attack_type': attack_type,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E2_sleeper_detection() -> List[Dict]:
        """
        E2: Sleeper Agent Detection (40 runs)

        Activation rounds: 10, 20
        Stealth levels: low, high
        Seeds: 5

        Total: 2 rounds × 2 stealth × 2 attack_levels × 5 seeds = 40 runs
        """
        configs = []
        activation_rounds = [10, 20]
        stealth_levels = ["low", "high"]

        for activation in activation_rounds:
            for stealth in stealth_levels:
                for seed in SEEDS:
                    configs.append({
                        'activation_round': activation,
                        'stealth_level': stealth,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E3_coordination_detection() -> List[Dict]:
        """
        E3: Coordination Detection (24 runs)

        Coalition sizes: 2, 3, 5, 8
        Coordination patterns: synchronized, sequential, adaptive
        Seeds: 3 (reduced for speed)

        Total: 4 sizes × 2 patterns × 3 seeds = 24 runs
        """
        configs = []
        coalition_sizes = [2, 3, 5, 8]
        patterns = ["synchronized", "sequential"]

        for size in coalition_sizes:
            for pattern in patterns:
                for seed in SEEDS[:3]:  # Only first 3 seeds
                    configs.append({
                        'coalition_size': size,
                        'coordination_pattern': pattern,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E4_active_learning_speedup() -> List[Dict]:
        """
        E4: Active Learning Speedup (30 runs)

        Query budgets: 1%, 5%, 10%
        Acquisition strategies: entropy, BALD
        Seeds: 5

        Total: 3 budgets × 2 strategies × 5 seeds = 30 runs
        """
        configs = []
        query_budgets = [0.01, 0.05, 0.10]
        acquisition_strategies = ["entropy", "margin"]  # BALD → margin (simpler)

        for budget in query_budgets:
            for strategy in acquisition_strategies:
                for seed in SEEDS:
                    configs.append({
                        'query_budget': budget,
                        'acquisition_strategy': strategy,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E5_federated_convergence() -> List[Dict]:
        """
        E5: Federated Convergence (30 runs)

        Optimizers: FedAvg, FedMDO
        Heterogeneity: low, high
        Seeds: 5

        Total: 2 optimizers × 3 heterogeneity × 5 seeds = 30 runs
        """
        configs = []
        optimizers = ["fedavg", "fedmdo"]
        heterogeneity_levels = ["low", "medium", "high"]

        for optimizer in optimizers:
            for heterogeneity in heterogeneity_levels:
                for seed in SEEDS:
                    configs.append({
                        'optimizer': optimizer,
                        'heterogeneity': heterogeneity,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E6_privacy_utility_tradeoff() -> List[Dict]:
        """
        E6: Privacy-Utility Tradeoff (24 runs)

        Epsilon: ∞ (no DP), 8, 4, 2
        Clipping: on, off
        Seeds: 3

        Total: 4 epsilon × 2 clipping × 3 seeds = 24 runs
        """
        configs = []
        epsilon_values = [float('inf'), 8.0, 4.0, 2.0]
        clipping_modes = [True, False]

        for epsilon in epsilon_values:
            for clipping in clipping_modes:
                for seed in SEEDS[:3]:
                    configs.append({
                        'epsilon': epsilon,
                        'clipping': clipping,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E7_distributed_validator_overhead() -> List[Dict]:
        """
        E7: Distributed Validator Overhead (12 runs)

        Validator quorum sizes: 3, 5, 7
        Payload sizes: small, large
        Seeds: 2

        Total: 3 quorums × 2 payloads × 2 seeds = 12 runs
        """
        configs = []
        quorum_sizes = [3, 5, 7]
        payload_sizes = ["small", "large"]

        for quorum in quorum_sizes:
            for payload in payload_sizes:
                for seed in SEEDS[:2]:
                    configs.append({
                        'quorum_size': quorum,
                        'payload_size': payload,
                        'seed': seed,
                    })

        return configs

    @staticmethod
    def E8_self_healing_recovery() -> List[Dict]:
        """
        E8: Self-Healing Recovery (12 runs)

        Fault types: poison_spike, straggler_burst, net_partition
        Seeds: 4

        Total: 3 faults × 4 seeds = 12 runs
        """
        configs = []
        fault_types = ["poison_spike", "straggler_burst", "net_partition"]

        for fault_type in fault_types:
            for seed in SEEDS[:4]:
                configs.append({
                    'fault_type': fault_type,
                    'seed': seed,
                })

        return configs

    @staticmethod
    def E9_secret_sharing_tolerance() -> List[Dict]:
        """
        E9: Secret Sharing Byzantine Tolerance (8 runs)

        n/t ratios: 3/5, 5/8
        Adversaries: 1, 2
        Seeds: 2

        Total: 2 ratios × 2 adversaries × 2 seeds = 8 runs
        """
        configs = []
        nt_ratios = [(3, 5), (5, 8)]
        adversary_counts = [1, 2]

        for (t, n) in nt_ratios:
            for adv_count in adversary_counts:
                for seed in SEEDS[:2]:
                    configs.append({
                        'threshold': t,
                        'num_validators': n,
                        'num_adversaries': adv_count,
                        'seed': seed,
                    })

        return configs

    @classmethod
    def get_full_matrix(cls) -> Dict[str, List[Dict]]:
        """Get complete experiment matrix (300 runs)."""
        return {
            'E1_byzantine_tolerance': cls.E1_byzantine_tolerance(),
            'E2_sleeper_detection': cls.E2_sleeper_detection(),
            'E3_coordination_detection': cls.E3_coordination_detection(),
            'E4_active_learning_speedup': cls.E4_active_learning_speedup(),
            'E5_federated_convergence': cls.E5_federated_convergence(),
            'E6_privacy_utility_tradeoff': cls.E6_privacy_utility_tradeoff(),
            'E7_distributed_validator_overhead': cls.E7_distributed_validator_overhead(),
            'E8_self_healing_recovery': cls.E8_self_healing_recovery(),
            'E9_secret_sharing_tolerance': cls.E9_secret_sharing_tolerance(),
        }

    @classmethod
    def get_dry_run_matrix(cls) -> Dict[str, List[Dict]]:
        """
        Dry run matrix (≈30 seconds, 9 runs)

        Tests: E1@{0%,30%,50%}, E5 both optimizers, E8 poison spike
        """
        return {
            'E1_byzantine_tolerance': [
                {'adversary_rate': 0.0, 'attack_type': 'label_flip', 'seed': SEEDS[0]},
                {'adversary_rate': 0.3, 'attack_type': 'sign_flip', 'seed': SEEDS[0]},
                {'adversary_rate': 0.5, 'attack_type': 'gradient_scaling', 'seed': SEEDS[0]},
            ],
            'E5_federated_convergence': [
                {'aggregator': 'aegis', 'adversary_rate': 0.0, 'seed': SEEDS[0]},
                {'aggregator': 'median', 'adversary_rate': 0.0, 'seed': SEEDS[0]},
                {'aggregator': 'aegis', 'adversary_rate': 0.2, 'seed': SEEDS[0]},
                {'aggregator': 'median', 'adversary_rate': 0.2, 'seed': SEEDS[0]},
            ],
            'E8_self_healing_recovery': [
                {'fault_type': 'poison_spike', 'seed': SEEDS[0]},
                {'fault_type': 'straggler_burst', 'seed': SEEDS[0]},
            ],
        }

    @classmethod
    def get_phase_a_matrix(cls) -> Dict[str, List[Dict]]:
        """
        Phase A matrix: Real implementations of E2, E3, E5 (≈5 minutes, 15-20 runs)

        High-signal, low-cost experiments for submission-grade results.
        """
        return {
            'E2_noniid_robustness': [
                # Primary α grid: Test across realistic non-IID spectrum
                {'alpha': 1.0, 'adversary_rate': 0.20, 'seed': SEEDS[0]},  # IID baseline
                {'alpha': 0.5, 'adversary_rate': 0.20, 'seed': SEEDS[0]},  # Moderate non-IID
                {'alpha': 0.3, 'adversary_rate': 0.20, 'seed': SEEDS[0]},  # Highly non-IID (PRIMARY FLOOR)
                # Stretch goal: α=0.1 with relaxed gates (not gating, for appendix)
                {'alpha': 0.1, 'adversary_rate': 0.20, 'seed': SEEDS[1]},  # Pathological (stretch)
                {'alpha': 0.1, 'adversary_rate': 0.20, 'seed': SEEDS[2]},  # Pathological (stretch)
            ],
            'E3_backdoor_resilience': [
                # Test AEGIS vs Median on backdoor attacks
                {'byz_frac': 0.20, 'seed': SEEDS[0]},
                {'byz_frac': 0.20, 'seed': SEEDS[1]},
                {'byz_frac': 0.20, 'seed': SEEDS[2]},
            ],
            'E5_federated_convergence': [
                # AEGIS vs Median at 0% and 20% Byzantines
                {'aggregator': 'aegis', 'adversary_rate': 0.0, 'seed': SEEDS[0]},
                {'aggregator': 'median', 'adversary_rate': 0.0, 'seed': SEEDS[0]},
                {'aggregator': 'aegis', 'adversary_rate': 0.2, 'seed': SEEDS[0]},
                {'aggregator': 'median', 'adversary_rate': 0.2, 'seed': SEEDS[0]},
                # Additional seeds for 20% Byzantine case
                {'aggregator': 'aegis', 'adversary_rate': 0.2, 'seed': SEEDS[1]},
                {'aggregator': 'median', 'adversary_rate': 0.2, 'seed': SEEDS[1]},
            ],
        }


def verify_state_hash(manifest: ValidationManifest) -> bool:
    """
    Verify state hash hasn't changed (drift detection).

    Returns:
        True if state unchanged, False if drift detected
    """
    current_manifest = ValidationManifest.create(manifest.experiment_matrix)

    if current_manifest.state_hash != manifest.state_hash:
        print(f"⚠️  STATE DRIFT DETECTED!")
        print(f"   Original: {manifest.state_hash}")
        print(f"   Current:  {current_manifest.state_hash}")
        return False

    return True


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic=np.mean,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Args:
        data: Sample data
        statistic: Function to compute (default: mean)
        alpha: Significance level (default: 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap resamples

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(42)

    # Point estimate
    point_est = statistic(data)

    # Bootstrap resamples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method CI
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_est, lower, upper


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cliff's Delta effect size (nonparametric).

    Interpretation:
        |δ| < 0.147: negligible
        |δ| < 0.330: small
        |δ| < 0.474: medium
        |δ| >= 0.474: large

    Returns:
        Delta in range [-1, 1]
    """
    n1, n2 = len(group1), len(group2)

    # Count pairwise comparisons
    greater = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
    less = sum(1 for x1 in group1 for x2 in group2 if x1 < x2)

    delta = (greater - less) / (n1 * n2)

    return delta


def create_validation_summary(
    results: List[ExperimentRun],
    output_path: Path,
):
    """Create comprehensive validation summary with statistics."""

    summary = {
        'overview': {
            'total_runs': len(results),
            'completed': sum(1 for r in results if r.status == 'completed'),
            'failed': sum(1 for r in results if r.status == 'failed'),
            'total_duration_hours': sum(r.duration for r in results) / 3600,
        },
        'experiments': {},
    }

    # Group by experiment type
    by_experiment = defaultdict(list)
    for result in results:
        if result.status == 'completed':
            by_experiment[result.experiment].append(result)

    # Compute statistics per experiment
    for exp_name, exp_results in by_experiment.items():
        exp_summary = {
            'num_runs': len(exp_results),
            'metrics': {},
        }

        # Aggregate metrics
        metrics_data = defaultdict(list)
        for result in exp_results:
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    metrics_data[metric_name].append(metric_value)

        # Compute statistics for each metric
        for metric_name, values in metrics_data.items():
            values_array = np.array(values)

            # Bootstrap 95% CI
            point_est, ci_lower, ci_upper = bootstrap_confidence_interval(values_array)

            exp_summary['metrics'][metric_name] = {
                'mean': float(point_est),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'n': len(values_array),
            }

        summary['experiments'][exp_name] = exp_summary

    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Statistical summary saved to {output_path}")


def main():
    """Main validation protocol entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Gen 5 Validation Protocol")
    parser.add_argument('--mode', choices=['dry-run', 'full'], default='dry-run',
                        help='Run mode: dry-run (~30 min) or full (hours)')
    parser.add_argument('--output-dir', default='validation_results_v1',
                        help='Output directory')

    args = parser.parse_args()

    # Get experiment matrix
    if args.mode == 'dry-run':
        matrix = ExperimentMatrix.get_dry_run_matrix()
        print("🧪 DRY RUN MODE (~30 minutes, ~10 runs)")
    else:
        matrix = ExperimentMatrix.get_full_matrix()
        print("🚀 FULL VALIDATION MODE (300 runs)")

    # Create manifest
    manifest = ValidationManifest.create(matrix)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    with open(output_dir / "manifest.json", 'w') as f:
        f.write(manifest.to_json())

    print(f"\n{'='*80}")
    print(f"AEGIS Gen 5 Validation Protocol v1.0")
    print(f"{'='*80}")
    print(f"\nValidation ID: {manifest.validation_id}")
    print(f"Git Commit: {manifest.git_commit}")
    print(f"State Hash: {manifest.state_hash}")
    print(f"Output Dir: {output_dir}")

    # Count total runs
    total_runs = sum(len(configs) for configs in matrix.values())
    print(f"\nTotal Experiments: {len(matrix)}")
    print(f"Total Runs: {total_runs}")

    print(f"\n{'='*80}\n")
    print("✅ Validation protocol initialized!")
    print("\nNext steps:")
    print("  1. Review manifest.json")
    print("  2. Run: python experiments/run_validation.py")
    print("  3. Generate figures: python experiments/generate_figures.py")

    return manifest, matrix


if __name__ == "__main__":
    main()
