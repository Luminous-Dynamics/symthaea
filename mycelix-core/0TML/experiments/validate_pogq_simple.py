#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Simplified PoGQ Validation - Uses Working Infrastructure

This creates a minimal config to test just FedAvg vs PoGQ+Rep
Leverages the proven working runner.py infrastructure
"""

import yaml
import json
from pathlib import Path
from datetime import datetime

# Configuration
CONFIG = {
    "experiment_name": "simple_pogq_validation",
    "dataset": {
        "name": "mnist",
        "num_train": 60000,
        "num_test": 10000,
        "data_dir": "datasets"
    },
    "data_split": {
        "type": "dirichlet",
        "alpha": 0.1,  # Extreme non-IID
        "num_clients": 10,
        "seed": 42
    },
    "training": {
        "num_rounds": 20,
        "local_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.01
    },
    "federated": {
        "num_clients": 10,
        "fraction_clients": 1.0,
        "batch_size": 32,
        "local_epochs": 3,
        "num_rounds": 20,
        "learning_rate": 0.01,
        "num_byzantine": 3,
        "quality_threshold": 0.3,
        "reputation_decay": 0.95
    },
    "byzantine": {
        "num_byzantine": 3,
        "attack_types": ["adaptivepoisoning"],
        "severity": 1.0,
        "enable_peer_context": True
    },
    "model": {
        "type": "simple_cnn",
        "params": {
            "num_classes": 10
        }
    },
    "device": "cuda",
    "save": {
        "metrics": [
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "num_accepted",
            "avg_quality_score"
        ],
        "per_round": True,
        "final_model": False,
        "attack_analysis": True
    }
}

def test_baseline(baseline_name, baseline_params):
    """Test a single baseline"""
    print(f"\n{'='*80}")
    print(f"Testing: {baseline_name}")
    print(f"{'='*80}\n")

    # Add baseline to config
    config = CONFIG.copy()
    config["baselines"] = [{
        "name": baseline_name.lower(),
        "params": baseline_params
    }]

    # Create config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = Path(f"results/simple_validation_{baseline_name.lower()}_{timestamp}.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Config created: {config_file}")

    # Import and run
    try:
        from runner import ExperimentRunner
        print("✅ Imported ExperimentRunner")

        runner = ExperimentRunner(str(config_file))
        print("✅ Runner initialized")

        print(f"\n🚀 Running {baseline_name} experiment...")
        results = runner.run()
        print("✅ Experiment completed")

        # Save results
        result_file = config_file.with_suffix('.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results:")
        print(f"   Final Accuracy: {results.get('final_accuracy', 0):.1%}")
        print(f"   Saved to: {result_file}")

        # Clean up config
        config_file.unlink()

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                   SIMPLE PoGQ VALIDATION                                     ║
║                                                                               ║
║  Quick test: FedAvg vs PoGQ+Rep                                              ║
║  Dataset: MNIST, 30% Byzantine, 20 rounds                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Test 1: FedAvg
    print("\n🔹 TEST 1: FedAvg (vulnerable baseline)")
    fedavg_results = test_baseline("FedAvg", {})

    if not fedavg_results:
        print("\n❌ FedAvg failed - cannot continue")
        return False

    # Test 2: PoGQ+Rep
    print("\n🔹 TEST 2: PoGQ+Rep (our innovation)")
    pogq_results = test_baseline("PoGQ", {
        "quality_threshold": 0.3,
        "reputation_decay": 0.95
    })

    if not pogq_results:
        print("\n❌ PoGQ+Rep failed")
        return False

    # Compare
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    fedavg_acc = fedavg_results.get("final_accuracy", 0)
    pogq_acc = pogq_results.get("final_accuracy", 0)
    improvement = (pogq_acc - fedavg_acc) * 100

    print(f"\n📊 Accuracy:")
    print(f"   FedAvg:    {fedavg_acc:.1%}")
    print(f"   PoGQ+Rep:  {pogq_acc:.1%}")
    print(f"   Improvement: {improvement:.1f}pp")

    # Success criteria
    print(f"\n✅ Validation Checks:")

    success = True
    if pogq_acc > fedavg_acc:
        print(f"   ✅ PoGQ+Rep outperforms FedAvg")
    else:
        print(f"   ❌ PoGQ+Rep does NOT outperform FedAvg")
        success = False

    if improvement > 5:
        print(f"   ✅ Improvement is significant (>{improvement:.1f}pp)")
    else:
        print(f"   ⚠️  Improvement is modest ({improvement:.1f}pp)")

    print("="*80)

    return success

if __name__ == "__main__":
    print("\n🚀 Starting simple PoGQ validation...")
    import time
    time.sleep(1)

    success = main()

    if success:
        print("\n✅ Validation successful - PoGQ+Rep integration works!")
    else:
        print("\n⚠️  Validation completed with issues")
