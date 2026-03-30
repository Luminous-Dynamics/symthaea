# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Quick integration test for real dataset + simulator.

Tests that EMNIST and CIFAR-10 datasets work with run_fl().
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.datasets.common import make_emnist, make_cifar10_backdoor
from experiments.simulator import FLScenario, run_fl

def test_emnist_integration():
    """Test EMNIST dataset with FL simulator."""
    print("=" * 60)
    print("Testing EMNIST Integration")
    print("=" * 60)

    # Create EMNIST dataset (small for quick test)
    print("\n1. Creating EMNIST dataset...")
    dataset = make_emnist(
        n_clients=10,
        noniid_alpha=0.5,
        n_train=1000,
        n_test=200,
        seed=42
    )
    print(f"✅ Dataset created: {dataset.dataset_name}")
    print(f"   Clients: {len(dataset.train_splits)}")
    print(f"   Features: {dataset.n_features}, Classes: {dataset.n_classes}")

    # Create scenario
    print("\n2. Creating FL scenario...")
    scenario = FLScenario(
        n_clients=10,
        byz_frac=0.2,
        noniid_alpha=0.5,
        attack="model_replacement",
        attack_lambda=10.0,
        participation_rate=0.6,
        seed=42
    )

    # Run FL simulation
    print("\n3. Running FL simulation (3 rounds)...")
    metrics = run_fl(
        scenario,
        aggregator="aegis",
        rounds=3,
        local_epochs=1,
        lr=0.05,
        dataset=dataset,
        aegis_cfg={
            "burn_in_rounds": 2,
            "ema_beta": 0.8,
            "use_robust_centroid": True,
            "per_layer_clip": 1.5,
        }
    )

    print("\n4. Results:")
    print(f"   Clean accuracy: {metrics['clean_acc']:.3f}")
    print(f"   Robust accuracy: {metrics['robust_acc']:.3f}")
    print(f"   AUC: {metrics['auc']:.3f}")
    print(f"   Runtime: {metrics['wall_s']:.2f}s")

    print("\n✅ EMNIST integration test passed!")
    return True


def test_cifar10_integration():
    """Test CIFAR-10 backdoor dataset with FL simulator."""
    print("\n" + "=" * 60)
    print("Testing CIFAR-10 Backdoor Integration")
    print("=" * 60)

    # Create CIFAR-10 backdoor dataset (small for quick test)
    print("\n1. Creating CIFAR-10 backdoor dataset...")
    dataset = make_cifar10_backdoor(
        n_clients=10,
        byz_frac=0.2,
        trigger_type="diagonal",
        trigger_value=2.0,
        poison_frac=0.2,
        target_label=0,
        n_train=500,
        n_test=100,
        seed=42
    )
    print(f"✅ Dataset created: {dataset.dataset_name}")
    print(f"   Clients: {len(dataset.train_splits)}")
    print(f"   Features: {dataset.n_features}, Classes: {dataset.n_classes}")
    print(f"   Test triggered: {dataset.test_triggered[0].shape[0]} samples")

    # Create scenario
    print("\n2. Creating FL scenario...")
    scenario = FLScenario(
        n_clients=10,
        byz_frac=0.2,
        noniid_alpha=1.0,  # IID for backdoor test
        attack="backdoor",
        participation_rate=0.6,
        seed=42
    )

    # Run FL simulation
    print("\n3. Running FL simulation (3 rounds)...")
    metrics = run_fl(
        scenario,
        aggregator="aegis",
        rounds=3,
        local_epochs=1,
        lr=0.05,
        dataset=dataset,
        aegis_cfg={
            "burn_in_rounds": 2,
            "ema_beta": 0.8,
            "use_robust_centroid": True,
            "per_layer_clip": 1.5,
        }
    )

    print("\n4. Results:")
    print(f"   Clean accuracy: {metrics['clean_acc']:.3f}")
    print(f"   Robust accuracy: {metrics['robust_acc']:.3f}")
    print(f"   ASR (backdoor): {metrics['asr']:.3f}")
    print(f"   AUC: {metrics['auc']:.3f}")
    print(f"   Runtime: {metrics['wall_s']:.2f}s")

    print("\n✅ CIFAR-10 integration test passed!")
    return True


if __name__ == "__main__":
    print("\n🧪 Real Dataset Integration Tests\n")

    try:
        # Test EMNIST
        test_emnist_integration()

        # Test CIFAR-10
        test_cifar10_integration()

        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
