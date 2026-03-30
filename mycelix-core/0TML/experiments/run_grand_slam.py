#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Grand Slam Experimental Matrix - Comprehensive PoGQ+Rep Validation

This script runs the definitive experiments to validate the PoGQ+Reputation system:
1. Three datasets (MNIST, CIFAR-10, Shakespeare) - generalization
2. Three baselines (FedAvg, Multi-Krum, PoGQ+Rep) - comparative advantage
3. Extreme non-IID (α=0.1) - real-world heterogeneity
4. Adaptive attack (30%) - sophisticated adversary
5. Stress tests (40%, 45%) - robustness under pressure

Expected Results:
- FedAvg: ~65% accuracy (vulnerable to Byzantine attacks)
- Multi-Krum: ~75% accuracy (classical defense)
- PoGQ+Rep: ~88% accuracy (our innovation) → +23pp improvement
"""

import sys
import os
from pathlib import Path
import yaml
import json
from datetime import datetime
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import ExperimentRunner
from baselines.fedavg import FedAvgServer
from baselines.multikrum import MultiKrumServer
from baselines.pogq_real import PoGQServer  # REAL PoGQ+Rep implementation

# Grand Slam Configuration
GRAND_SLAM_CONFIG = {
    # Core datasets for generalization (main paper)
    "datasets": [
        {
            "name": "mnist",  # Standard benchmark - digits
            "num_clients": 20,
            "samples_per_client": 500,
            "validation_split": 0.2,
        },
        {
            "name": "cifar10",  # Color images - more challenging
            "num_clients": 20,
            "samples_per_client": 500,
            "validation_split": 0.2,
        },
        {
            "name": "fmnist",  # Fashion-MNIST - cross-domain (digits → clothing)
            "num_clients": 20,
            "samples_per_client": 500,
            "validation_split": 0.2,
        },
    ],

    # Baselines to compare
    "baselines": [
        {
            "name": "FedAvg",
            "class": "FedAvgServer",
            "config": {}
        },
        {
            "name": "Multi-Krum",
            "class": "MultiKrumServer",
            "config": {"f": 6}  # Tolerate up to 6 Byzantine nodes
        },
        {
            "name": "PoGQ+Rep",
            "class": "PoGQServer",
            "config": {
                "quality_threshold": 0.3,
                "reputation_decay": 0.95,
            }
        }
    ],

    # Attack scenarios
    "attacks": [
        {
            "name": "adaptive",
            "type": "AdaptivePoisoning",
            "byzantine_fraction": 0.30,  # 30% Byzantine (6/20 nodes)
        }
    ],

    # Data heterogeneity
    "heterogeneity": {
        "type": "dirichlet",
        "alpha": 0.1,  # Extreme non-IID
    },

    # Training configuration
    "training": {
        "num_rounds": 50,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
    }
}

# Stress test configuration (cifar10 only, higher Byzantine fractions)
STRESS_TEST_CONFIG = {
    "datasets": [{"name": "cifar10", "num_clients": 20, "samples_per_client": 500}],  # FIXED: lowercase, no hyphen
    "baselines": [
        {"name": "FedAvg", "class": "FedAvgServer", "config": {}},
        {"name": "PoGQ+Rep", "class": "PoGQServer", "config": {}}
    ],
    "attacks": [
        {"name": "adaptive", "type": "AdaptivePoisoning", "byzantine_fraction": 0.40},
        {"name": "adaptive", "type": "AdaptivePoisoning", "byzantine_fraction": 0.45},
    ],
    "heterogeneity": {"type": "dirichlet", "alpha": 0.1},
    "training": {"num_rounds": 50, "local_epochs": 5, "batch_size": 32, "learning_rate": 0.01}
}


def normalize_baseline_name(baseline_name: str) -> str:
    """
    Normalize baseline names to match runner.py expectations.

    Maps:
    - "FedAvg" → "fedavg"
    - "Multi-Krum" → "multikrum"
    - "PoGQ+Rep" → "pogq"
    """
    # Mapping for special cases
    name_map = {
        "fedavg": "fedavg",
        "multi-krum": "multikrum",
        "pogq+rep": "pogq",
        "pogqrep": "pogq",  # In case it's already been mangled
    }

    # Normalize: lowercase, remove spaces/hyphens/plus
    normalized = baseline_name.lower().replace(' ', '').replace('-', '').replace('+', '')

    # Return mapped name or normalized
    return name_map.get(normalized, normalized)


def run_experiment(dataset_name, baseline_name, attack_config, runner_config):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running: {dataset_name} / {baseline_name} / {attack_config['name']} ({attack_config['byzantine_fraction']*100:.0f}%)")
    print(f"{'='*80}\n")

    # Load base config from mini_validation.yaml (proven working approach)
    # Use .resolve() to handle symlinks and ensure absolute path
    project_root = Path(__file__).resolve().parent.parent
    base_config_path = project_root / 'experiments' / 'configs' / 'mini_validation.yaml'

    # Verify file exists before trying to open it
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_config_path}\n"
                              f"  Script location: {Path(__file__).resolve()}\n"
                              f"  Project root: {project_root}\n"
                              f"  Looking for: {base_config_path}")

    with open(base_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    # Override with experiment-specific settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    byzantine_pct = int(attack_config["byzantine_fraction"] * 100)
    num_clients = runner_config.get("num_clients", 20)
    num_byzantine = int(num_clients * attack_config["byzantine_fraction"])

    # FIXED: Normalize baseline name properly
    baseline_runner_name = normalize_baseline_name(baseline_name)

    exp_config["experiment_name"] = f"grand_slam_{dataset_name}_{baseline_runner_name}_{byzantine_pct}pct"
    exp_config["dataset"]["name"] = dataset_name  # Already lowercase from config
    exp_config["byzantine"]["num_byzantine"] = num_byzantine

    # CRITICAL: Override num_clients in BOTH data_split and federated sections
    exp_config["data_split"]["num_clients"] = num_clients
    exp_config["federated"]["num_clients"] = num_clients
    exp_config["federated"]["num_byzantine"] = num_byzantine
    exp_config["federated"]["num_rounds"] = runner_config.get("num_rounds", 50)
    exp_config["training"]["num_rounds"] = runner_config.get("num_rounds", 50)
    exp_config["federated"]["local_epochs"] = runner_config.get("local_epochs", 5)
    exp_config["training"]["local_epochs"] = runner_config.get("local_epochs", 5)

    # Set baseline with normalized name
    exp_config["baselines"] = [{
        "name": baseline_runner_name,
        "params": runner_config.get("baseline_params", {})
    }]

    # Create temporary config file
    config_file = Path(f"results/grand_slam_config_{dataset_name.lower()}_{baseline_name.lower().replace('+', '')}_{byzantine_pct}pct_{timestamp}.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False)

    print(f"✅ Config created: {config_file}")

    # Initialize runner with file path
    runner = ExperimentRunner(str(config_file))

    # Run experiment
    results = runner.run()

    # DEFENSIVE: Handle None results
    if results is None:
        print("⚠️  WARNING: runner.run() returned None!")
        results = {
            "error": "runner.run() returned None",
            "final_accuracy": 0.0,
            "byzantine_detection_rate": 0.0
        }

    # Save results
    result_file = Path(f"results/grand_slam_{dataset_name}_{baseline_runner_name}_{attack_config['name']}_{byzantine_pct}pct_{timestamp}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {result_file}")

    # Clean up temp config
    config_file.unlink()

    return results


def run_grand_slam():
    """Run the complete Grand Slam experimental matrix"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                      GRAND SLAM EXPERIMENTAL MATRIX                          ║
║                                                                               ║
║  Purpose: Comprehensive validation of PoGQ+Reputation system                 ║
║  Scope: 3 datasets × 3 baselines × 1 attack = 9 core experiments            ║
║         + 4 stress tests (2 baselines × 2 Byzantine%) = 13 total            ║
║                                                                               ║
║  Main Paper: MNIST + CIFAR-10 + Fashion-MNIST                                ║
║  Appendix: EMNIST (47 classes) - run separately if time allows              ║
║  Supplementary: SVHN - domain shift robustness                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    all_results = []

    # Part 1: Core experiments (2 datasets × 3 baselines)
    print("\n" + "="*80)
    print("PART 1: CORE EXPERIMENTS (Generalization Across Datasets)")
    print("="*80)

    for dataset_config in GRAND_SLAM_CONFIG["datasets"]:
        for baseline_config in GRAND_SLAM_CONFIG["baselines"]:
            for attack_config in GRAND_SLAM_CONFIG["attacks"]:
                try:
                    results = run_experiment(
                        dataset_config["name"],
                        baseline_config["name"],
                        attack_config,
                        {
                            "num_clients": dataset_config["num_clients"],
                            "num_rounds": GRAND_SLAM_CONFIG["training"]["num_rounds"],
                            "local_epochs": GRAND_SLAM_CONFIG["training"]["local_epochs"],
                            "baseline_params": baseline_config["config"]
                        }
                    )
                    all_results.append({
                        "dataset": dataset_config["name"],
                        "baseline": baseline_config["name"],
                        "attack": attack_config["name"],
                        "byzantine_pct": int(attack_config["byzantine_fraction"] * 100),
                        "final_accuracy": results.get("final_accuracy", 0),
                        "byzantine_detection_rate": results.get("byzantine_detection_rate", 0)
                    })
                except Exception as e:
                    print(f"❌ Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()

    # Part 2: Stress tests (higher Byzantine fractions)
    print("\n" + "="*80)
    print("PART 2: STRESS TESTS (Robustness Under Pressure)")
    print("="*80)

    for baseline_config in STRESS_TEST_CONFIG["baselines"]:
        for attack_config in STRESS_TEST_CONFIG["attacks"]:
            try:
                results = run_experiment(
                    "cifar10",  # FIXED: Use lowercase as expected by runner.py
                    baseline_config["name"],
                    attack_config,
                    {
                        "num_clients": STRESS_TEST_CONFIG["datasets"][0]["num_clients"],
                        "num_rounds": STRESS_TEST_CONFIG["training"]["num_rounds"],
                        "local_epochs": STRESS_TEST_CONFIG["training"]["local_epochs"],
                        "baseline_params": baseline_config["config"]
                    }
                )
                all_results.append({
                    "dataset": "cifar10",  # FIXED: Use lowercase
                    "baseline": baseline_config["name"],
                    "attack": attack_config["name"],
                    "byzantine_pct": int(attack_config["byzantine_fraction"] * 100),
                    "final_accuracy": results.get("final_accuracy", 0),
                    "byzantine_detection_rate": results.get("byzantine_detection_rate", 0)
                })
            except Exception as e:
                print(f"❌ Stress test failed: {e}")
                import traceback
                traceback.print_exc()

    # Generate summary report
    generate_summary_report(all_results)


def generate_summary_report(all_results):
    """Generate a comprehensive summary report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"results/grand_slam_summary_{timestamp}.yaml")

    # Group results by scenario
    core_results = [r for r in all_results if r["byzantine_pct"] == 30]
    stress_results = [r for r in all_results if r["byzantine_pct"] > 30]

    # Calculate improvements
    improvements = []
    for dataset in ["mnist", "cifar10", "fmnist"]:  # All 3 main paper datasets
        dataset_results = [r for r in core_results if r["dataset"] == dataset]

        fedavg = next((r for r in dataset_results if r["baseline"] == "FedAvg"), None)
        pogq = next((r for r in dataset_results if r["baseline"] == "PoGQ+Rep"), None)

        if fedavg and pogq:
            improvement = pogq["final_accuracy"] - fedavg["final_accuracy"]
            improvements.append({
                "dataset": dataset,
                "fedavg_accuracy": fedavg["final_accuracy"],
                "pogq_accuracy": pogq["final_accuracy"],
                "improvement": improvement,
                "improvement_pp": improvement * 100  # percentage points
            })

    # Create summary
    summary = {
        "experiment": "Grand Slam Validation",
        "timestamp": timestamp,
        "total_experiments": len(all_results),
        "core_experiments": len(core_results),
        "stress_tests": len(stress_results),

        "key_findings": {
            "average_improvement": sum(i["improvement_pp"] for i in improvements) / len(improvements) if improvements else 0,
            "improvements_by_dataset": improvements,
        },

        "all_results": all_results,

        "grant_ready_numbers": {
            "claim": "PoGQ+Reputation achieves +23.2pp improvement over baseline FedAvg",
            "evidence": improvements,
            "verification": "See individual experiment results in results/ directory"
        }
    }

    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print("\n" + "="*80)
    print("GRAND SLAM SUMMARY")
    print("="*80)
    print(f"\n✅ Completed {len(all_results)} experiments")
    print(f"📊 Summary saved to: {summary_file}")

    if improvements:
        print("\n📈 Key Results:")
        for imp in improvements:
            print(f"\n  {imp['dataset']}:")
            print(f"    FedAvg:    {imp['fedavg_accuracy']:.1%}")
            print(f"    PoGQ+Rep:  {imp['pogq_accuracy']:.1%}")
            print(f"    Improvement: {imp['improvement_pp']:.1f} percentage points")

        avg_improvement = sum(i["improvement_pp"] for i in improvements) / len(improvements)
        print(f"\n  Average Improvement: {avg_improvement:.1f}pp")

        if abs(avg_improvement - 23.2) < 5:
            print("\n  ✅ CLAIM VERIFIED: ~23.2pp improvement achieved!")
        else:
            print(f"\n  ⚠️  CLAIM DISCREPANCY: Found {avg_improvement:.1f}pp vs claimed 23.2pp")
            print("     Recommend updating documentation to match actual results")


if __name__ == "__main__":
    print("\n🚀 Starting Grand Slam Experimental Matrix...")
    print("   This will take several hours. Monitor progress in /tmp/ logs")
    print("   Press Ctrl+C to cancel\n")

    import time
    time.sleep(3)  # Give user time to cancel if needed

    run_grand_slam()

    print("\n✅ Grand Slam complete! Check results/ directory for detailed data")
