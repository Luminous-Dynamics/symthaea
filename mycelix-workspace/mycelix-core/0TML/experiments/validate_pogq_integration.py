#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick Validation: PoGQ+Rep Integration Test

This script runs a quick experiment to verify that:
1. The REAL PoGQ+Rep implementation is properly integrated
2. It can run without errors
3. It produces reasonable results

Single experiment: MNIST, FedAvg vs PoGQ+Rep, 30% Byzantine adaptive attack
Should complete in ~10-15 minutes
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing modules...")
try:
    from runner import ExperimentRunner
    from baselines.fedavg import FedAvgServer
    from baselines.pogq_real import PoGQServer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the nix develop shell")
    print("2. Check that baselines/pogq_real.py exists")
    print("3. Verify torch and other dependencies are installed")
    sys.exit(1)

# Quick validation configuration
CONFIG = {
    "dataset": "MNIST",
    "num_clients": 10,  # Fewer clients for speed
    "samples_per_client": 300,  # Fewer samples for speed
    "validation_split": 0.2,

    "byzantine_fraction": 0.30,  # 30% Byzantine (3/10 nodes)
    "attack_type": "AdaptivePoisoning",

    "heterogeneity": {
        "type": "dirichlet",
        "alpha": 0.1,  # Extreme non-IID
    },

    "training": {
        "num_rounds": 20,  # Fewer rounds for speed
        "local_epochs": 3,  # Fewer epochs for speed
        "batch_size": 32,
        "learning_rate": 0.01,
    }
}


def test_baseline(baseline_name, baseline_class, baseline_config):
    """Test a single baseline"""
    print(f"\n{'='*80}")
    print(f"Testing: {baseline_name}")
    print(f"{'='*80}\n")

    # Load base config from mini_validation.yaml (same approach as run_mini_validation.py)
    import yaml
    from pathlib import Path as PPath

    project_root = PPath(__file__).parent.parent
    config_path = project_root / 'experiments' / 'configs' / 'mini_validation.yaml'

    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    # Override specific fields for this validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_byzantine = int(CONFIG["num_clients"] * CONFIG["byzantine_fraction"])

    exp_config["experiment_name"] = f"validation_{baseline_name}"
    exp_config["data_split"]["alpha"] = CONFIG["heterogeneity"]["alpha"]
    exp_config["byzantine"]["num_byzantine"] = num_byzantine
    exp_config["federated"]["num_byzantine"] = num_byzantine
    exp_config["federated"]["num_rounds"] = CONFIG["training"]["num_rounds"]
    exp_config["training"]["num_rounds"] = CONFIG["training"]["num_rounds"]
    exp_config["federated"]["local_epochs"] = CONFIG["training"]["local_epochs"]
    exp_config["training"]["local_epochs"] = CONFIG["training"]["local_epochs"]

    # Set single baseline to test
    exp_config["baselines"] = [
        {
            "name": baseline_name.lower(),
            "params": baseline_config
        }
    ]

    # Save config to temporary file
    config_file = Path(f"results/validation_{baseline_name.lower().replace('+', '_')}_config_{timestamp}.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False)

    print(f"Created config file: {config_file}")

    # Initialize runner
    print("Initializing experiment runner...")
    try:
        runner = ExperimentRunner(str(config_file))
        print("✅ Runner initialized")
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Run experiment
    print("Running experiment (this will take ~10 minutes)...")
    try:
        results = runner.run()
        print("✅ Experiment completed successfully")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Save results
    result_file = Path(f"results/validation_{baseline_name.lower().replace('+', '_')}_{timestamp}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results:")
    print(f"   Final Accuracy: {results.get('final_accuracy', 0):.1%}")
    print(f"   Byzantine Detection Rate: {results.get('byzantine_detection_rate', 0):.1%}")
    print(f"   Saved to: {result_file}")

    # Clean up temp config
    config_file.unlink()

    return results


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                   PoGQ+Rep INTEGRATION VALIDATION                            ║
║                                                                               ║
║  Testing: REAL PoGQ+Reputation implementation                                ║
║  Dataset: MNIST (quick test)                                                 ║
║  Duration: ~10-15 minutes                                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Test 1: FedAvg baseline
    print("\n" + "="*80)
    print("TEST 1: FedAvg Baseline (should be vulnerable)")
    print("="*80)

    fedavg_results = test_baseline("FedAvg", FedAvgServer, {})

    if fedavg_results is None:
        print("\n❌ FedAvg test failed - cannot continue")
        return

    # Test 2: PoGQ+Rep (our innovation)
    print("\n" + "="*80)
    print("TEST 2: PoGQ+Rep (should be robust)")
    print("="*80)

    pogq_results = test_baseline("PoGQ+Rep", PoGQServer, {
        "quality_threshold": 0.3,
        "reputation_decay": 0.95,
    })

    if pogq_results is None:
        print("\n❌ PoGQ+Rep test failed")
        return

    # Compare results
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    fedavg_acc = fedavg_results.get("final_accuracy", 0)
    pogq_acc = pogq_results.get("final_accuracy", 0)
    improvement = (pogq_acc - fedavg_acc) * 100  # percentage points

    print(f"\n📊 Results:")
    print(f"   FedAvg:    {fedavg_acc:.1%}")
    print(f"   PoGQ+Rep:  {pogq_acc:.1%}")
    print(f"   Improvement: {improvement:.1f} percentage points")

    # Validation checks
    print(f"\n🔍 Validation Checks:")

    checks_passed = 0
    total_checks = 3

    # Check 1: PoGQ should be better than FedAvg
    if pogq_acc > fedavg_acc:
        print(f"   ✅ PoGQ+Rep outperforms FedAvg")
        checks_passed += 1
    else:
        print(f"   ❌ PoGQ+Rep does NOT outperform FedAvg")

    # Check 2: Improvement should be significant (>5pp)
    if improvement > 5:
        print(f"   ✅ Improvement is significant (>{improvement:.1f}pp)")
        checks_passed += 1
    else:
        print(f"   ❌ Improvement is NOT significant ({improvement:.1f}pp < 5pp)")

    # Check 3: Byzantine detection should be high
    pogq_detection = pogq_results.get("byzantine_detection_rate", 0)
    if pogq_detection > 0.8:
        print(f"   ✅ Byzantine detection is high ({pogq_detection:.1%})")
        checks_passed += 1
    else:
        print(f"   ❌ Byzantine detection is low ({pogq_detection:.1%})")

    # Overall assessment
    print(f"\n{'='*80}")
    if checks_passed == total_checks:
        print("✅ VALIDATION PASSED - Integration is working correctly!")
        print("\nReady to run full Grand Slam experiments:")
        print("   nix develop --command python run_grand_slam.py")
    elif checks_passed >= 2:
        print("⚠️  VALIDATION PARTIAL - Integration works but results suboptimal")
        print("\nConsider investigating before running full Grand Slam")
    else:
        print("❌ VALIDATION FAILED - Integration has issues")
        print("\nDo NOT run Grand Slam until issues are resolved")
    print(f"{'='*80}\n")

    return checks_passed == total_checks


if __name__ == "__main__":
    print("\n🚀 Starting PoGQ+Rep integration validation...")
    print("   This will take ~10-15 minutes")
    print("   Press Ctrl+C to cancel\n")

    import time
    time.sleep(2)

    success = main()

    if success:
        print("\n✅ All validation checks passed!")
        print("\n📋 Next Steps:")
        print("   1. Review results in results/ directory")
        print("   2. If satisfied, run: nix develop --command python run_grand_slam.py")
        print("   3. Grand Slam will generate all data needed for grant application")
    else:
        print("\n⚠️  Validation had issues - review results before proceeding")
