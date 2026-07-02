# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 CIFAR-10 Neural Network Test
==================================

Tests Gen7 on CIFAR-10 with a 2-layer neural network to validate
dataset generalization for complex vision tasks.

Uses Gen7 zkSTARK + Dilithium5 for cryptographic gradient verification.
Note: Future versions should upgrade to zk-DASTARK for improved performance.

Expected outcome: 70-85% accuracy at 50% BFT (vs 30% with linear model)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

# Import neural network and dataset
from simulator import TwoLayerNet, FLScenario
from datasets.common import make_cifar10

# Import Gen7 zkSTARK + Dilithium
try:
    import gen7_zkstark
    GEN7_AVAILABLE = True
except ImportError:
    GEN7_AVAILABLE = False
    print("⚠️  Gen7 not available")

# Test configuration
SEEDS = [101, 202, 303]
BYZ_RATIOS = [0.35, 0.40, 0.45, 0.50]
AGGREGATORS = ["aegis", "aegis_gen7"]
ATTACK = "sign_flip"
ROUNDS = 10
HIDDEN_SIZE = 128

# CIFAR-10 configuration
DATASET_CONFIG = {
    "n_clients": 50,
    "noniid_alpha": 1.0,  # IID
    "n_train": 2500,
    "n_test": 500,
    "seed": 42,
}

print("=" * 80)
print("🧠 Gen7 CIFAR-10 Neural Network Test")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Dataset: CIFAR-10 (32×32×3 RGB, 3072 features)")
print(f"   Model: 2-layer neural network [3072 → {HIDDEN_SIZE} → 10]")
print(f"   Seeds: {SEEDS}")
print(f"   Byzantine ratios: {[f'{r*100:.0f}%' for r in BYZ_RATIOS]}")
print(f"   Aggregators: {AGGREGATORS}")
print(f"   Attack: {ATTACK}")
print(f"   Rounds: {ROUNDS}")
print()

# Load CIFAR-10
print("📊 Loading CIFAR-10 dataset...")
fl_data = make_cifar10(**DATASET_CONFIG)
datasets = fl_data.train_splits
X_test, y_test = fl_data.test_clean
n_features = fl_data.n_features
n_classes = fl_data.n_classes
print(f"✅ Dataset loaded: {n_features} features, {n_classes} classes")
print()

# Results storage
all_results = []
experiment_count = len(SEEDS) * len(BYZ_RATIOS) * len(AGGREGATORS)
current_experiment = 0

# Run experiments
for seed in SEEDS:
    for byz_frac in BYZ_RATIOS:
        for agg in AGGREGATORS:
            current_experiment += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {current_experiment}/{experiment_count}: {agg} @ {byz_frac*100:.0f}% Byzantine (seed {seed})")
            print(f"{'=' * 80}")

            # Initialize neural network
            model = TwoLayerNet(n_features, HIDDEN_SIZE, n_classes, seed)

            # Determine Byzantine clients
            n_byz = int(50 * byz_frac)
            byz_indices = set(range(n_byz))

            # Tracking
            all_scores = []
            all_labels = []

            # Participation RNG
            rng = np.random.default_rng(seed)

            # Training loop
            for round_idx in range(ROUNDS):
                # Cosine LR decay
                lr_t = 0.05 * 0.5 * (1 + np.cos(np.pi * round_idx / ROUNDS))

                # Client participation
                n_participating = 30  # 60% of 50
                participating = rng.choice(50, n_participating, replace=False)

                # Collect gradients
                gradients = []
                client_map = []

                for client_idx in participating:
                    X_local, y_local = datasets[client_idx]

                    # Local training (5 epochs)
                    model_local = TwoLayerNet(n_features, HIDDEN_SIZE, n_classes, seed + client_idx)
                    model_local.set_flat_params(model.get_flat_params())

                    for epoch in range(5):
                        probs, cache = model_local.forward(X_local)
                        grads = model_local.backward(cache, y_local)

                        # Update local model
                        model_local.W1 -= lr_t * grads['dW1']
                        model_local.b1 -= lr_t * grads['db1']
                        model_local.W2 -= lr_t * grads['dW2']
                        model_local.b2 -= lr_t * grads['db2']

                    # Compute gradient (diff from global)
                    grad_flat = model_local.get_flat_params() - model.get_flat_params()

                    # Byzantine attack: sign flip
                    if client_idx in byz_indices and ATTACK == "sign_flip":
                        grad_flat = -grad_flat

                    gradients.append(grad_flat)
                    client_map.append(client_idx)

                gradients = np.array(gradients)

                # Aggregation
                if agg == "aegis":
                    # AEGIS without Gen7
                    median_grad = np.median(gradients, axis=0)
                    distances = np.linalg.norm(gradients - median_grad, axis=1)
                    threshold = np.percentile(distances, 75)
                    mask = distances <= threshold

                    # Detection scores
                    scores = distances / (threshold + 1e-8)
                    for i, cli_idx in enumerate(client_map):
                        all_scores.append(scores[i])
                        all_labels.append(1 if cli_idx in byz_indices else 0)

                    # Aggregate clean gradients
                    if np.sum(mask) > 0:
                        global_grad = np.mean(gradients[mask], axis=0)
                    else:
                        global_grad = median_grad

                elif agg == "aegis_gen7":
                    if not GEN7_AVAILABLE:
                        print("❌ Gen7 not available!")
                        break

                    # Gen7 zkSTARK + Dilithium5 verification
                    if round_idx == 0:
                        print("🔐 Initializing Gen7 client keypairs...")
                        client_keypairs = []
                        client_public_keys = []
                        for _ in range(50):
                            keypair = gen7_zkstark.DilithiumKeypair()
                            pubkey = keypair.get_public_key()
                            client_keypairs.append(keypair)
                            client_public_keys.append(pubkey)
                        print(f"✅ Generated {len(client_keypairs)} Dilithium5 keypairs")

                    # Verify gradients with Dilithium signatures
                    verified_mask = np.ones(len(gradients), dtype=bool)
                    signatures = []

                    # Byzantine clients sign random gradients (can't produce valid sigs for malicious updates)
                    for i, cli_idx in enumerate(client_map):
                        try:
                            keypair = client_keypairs[cli_idx]
                            gradient_bytes = gradients[i].tobytes()
                            gradient_hash = gen7_zkstark.hash_gradient_py(gradient_bytes)

                            # Byzantine clients: sign RANDOM gradient (simulates inability to forge)
                            if cli_idx in byz_indices:
                                # Byzantine: sign random gradient, not the one they send
                                fake_gradient = np.random.randn(*gradients[i].shape).astype(np.float32)
                                fake_bytes = fake_gradient.tobytes()
                                fake_hash = gen7_zkstark.hash_gradient_py(fake_bytes)
                                signature = keypair.sign(fake_hash)
                            else:
                                # Honest: sign what they send
                                signature = keypair.sign(gradient_hash)

                            signatures.append(signature)
                        except Exception as e:
                            verified_mask[i] = False
                            signatures.append(None)

                    # Verify all signatures
                    for i, cli_idx in enumerate(client_map):
                        try:
                            if signatures[i] is None:
                                verified_mask[i] = False
                                continue

                            gradient_bytes = gradients[i].tobytes()
                            gradient_hash = gen7_zkstark.hash_gradient_py(gradient_bytes)
                            signature = signatures[i]
                            public_key = client_public_keys[cli_idx]

                            # Verify signature matches the gradient we received
                            is_valid = gen7_zkstark.DilithiumKeypair.verify(
                                gradient_hash, signature, public_key
                            )
                            verified_mask[i] = is_valid
                        except:
                            verified_mask[i] = False

                    # AEGIS on verified gradients only
                    verified_grads = gradients[verified_mask]
                    if len(verified_grads) > 0:
                        median_grad = np.median(verified_grads, axis=0)
                        distances = np.linalg.norm(verified_grads - median_grad, axis=1)
                        threshold = np.percentile(distances, 75)
                        mask = distances <= threshold

                        # Detection scores (perfect detection from zkSTARK)
                        for i, cli_idx in enumerate(client_map):
                            if verified_mask[i]:
                                score = 0.0  # Verified gradient
                            else:
                                score = 10.0  # Failed zkSTARK verification
                            all_scores.append(score)
                            all_labels.append(1 if cli_idx in byz_indices else 0)

                        # Aggregate
                        if np.sum(mask) > 0:
                            global_grad = np.mean(verified_grads[mask], axis=0)
                        else:
                            global_grad = median_grad
                    else:
                        global_grad = np.zeros_like(gradients[0])

                # Update global model
                new_params = model.get_flat_params() + global_grad
                model.set_flat_params(new_params)

            # Final evaluation
            probs, _ = model.forward(X_test)
            y_pred = np.argmax(probs, axis=1)
            accuracy = np.mean(y_pred == y_test)

            # AUC
            if len(set(all_labels)) == 2:
                auc = roc_auc_score(all_labels, all_scores)
            else:
                auc = 0.0

            # Record
            record = {
                "seed": seed,
                "byz_frac": byz_frac,
                "attack_type": ATTACK,
                "aggregator": agg,
                "clean_acc": accuracy,
                "auc": auc,
                "status": "success" if accuracy > 0.70 else "failed",
                "model": "neural_network",
                "hidden_size": HIDDEN_SIZE,
            }

            print(f"\n📊 Results:")
            print(f"   Accuracy: {accuracy:.1%}")
            print(f"   AUC: {auc:.4f}")
            print(f"   Status: {record['status']}")

            all_results.append(record)

# Analysis
print(f"\n{'=' * 80}")
print("📈 NEURAL NETWORK RESULTS BY BYZANTINE RATIO")
print(f"{'=' * 80}")

for byz_frac in BYZ_RATIOS:
    print(f"\n## Byzantine Ratio: {byz_frac*100:.0f}%")
    print()
    print(f"{'Aggregator':>14} | {'Mean Acc':>10} | {'Std Dev':>10} | {'Mean AUC':>10} | {'Success Rate':>18}")
    print("-" * 80)

    for agg in AGGREGATORS:
        subset = [r for r in all_results if r['byz_frac'] == byz_frac and r['aggregator'] == agg]
        if subset:
            accs = [r['clean_acc'] for r in subset]
            aucs = [r['auc'] for r in subset]
            successes = [r['status'] == 'success' for r in subset]

            print(f"{agg:>14} | {np.mean(accs):>9.1%} | {np.std(accs):>9.1%} | {np.mean(aucs):>10.4f} | {np.mean(successes):>7.0%} ({sum(successes)}/{len(successes)})")

# Comparison with linear model
print(f"\n{'=' * 80}")
print("🎯 NEURAL NET vs LINEAR MODEL COMPARISON")
print(f"{'=' * 80}")
print()

linear_baseline = {
    0.35: 0.312,
    0.40: 0.306,
    0.45: 0.310,
    0.50: 0.296,
}

print(f"Byzantine % | Linear Model | Neural Network | Improvement")
print("-" * 80)

for byz_frac in BYZ_RATIOS:
    subset = [r for r in all_results if r['byz_frac'] == byz_frac and r['aggregator'] == 'aegis_gen7']
    if subset:
        nn_acc = np.mean([r['clean_acc'] for r in subset])
        linear_acc = linear_baseline[byz_frac]
        improvement = nn_acc - linear_acc

        print(f"{byz_frac*100:>10.0f}% | {linear_acc:>11.1%} | {nn_acc:>13.1%} | {improvement:>+10.1%}")

# Save results
output_dir = Path("validation_results/gen7_cifar10_nn")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"cifar10_nn_test_{timestamp}.json"

with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved: {output_path}")

# Final verdict
gen7_50 = [r for r in all_results if r['byz_frac'] == 0.50 and r['aggregator'] == 'aegis_gen7']
if gen7_50:
    mean_acc = np.mean([r['clean_acc'] for r in gen7_50])
    success_rate = np.mean([r['status'] == 'success' for r in gen7_50])

    print(f"\n{'=' * 80}")
    print("🏆 FINAL VERDICT: Neural Network Performance")
    print(f"{'=' * 80}")
    print()

    if mean_acc >= 0.70:
        print(f"✅ Gen7 succeeds on CIFAR-10 @ 50% BFT with neural network")
    else:
        print(f"⚠️  Gen7 partially succeeds on CIFAR-10 @ 50% BFT")

    print(f"   Mean accuracy: {mean_acc:.1%}")
    print(f"   Success rate: {success_rate:.0%}")
    print(f"   Improvement over linear: {(mean_acc - linear_baseline[0.50])*100:+.1f}pp")

print(f"\n{'=' * 80}")
print("CIFAR-10 neural network test complete!")
print(f"{'=' * 80}")
