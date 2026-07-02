# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Experiment 1: MNIST Accuracy Comparison

Compare ZeroTrustML (Krum) vs FedAvg on MNIST dataset.

Metrics:
- Accuracy vs communication rounds
- Convergence speed
- Final model accuracy

Budget: $0 (local CPU training)
"""

import sys
from pathlib import Path

# Fix Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "benchmarks" / "datasets"))
sys.path.insert(0, str(project_root / "benchmarks" / "baselines"))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List

# Import datasets
from mnist_loader import MNISTFederated

# Import baselines
from fedavg import (
    FedAvgServer,
    FedAvgClient,
    FedAvgConfig,
    run_fedavg_round,
    evaluate_model
)

# Import ZeroTrustML
from zerotrustml.aggregation.algorithms import Krum
from zerotrustml.core.training import SimpleNN


class ExperimentConfig:
    """Experiment configuration"""
    num_clients = 10
    rounds = 50
    local_epochs = 1
    learning_rate = 0.01
    batch_size = 32
    data_split = "iid"  # or "non_iid"
    device = "cpu"
    results_dir = "../results"


def run_fedavg_experiment(
    config: ExperimentConfig,
    train_loaders: List,
    test_loader
) -> Dict:
    """Run FedAvg baseline"""
    print("\n🔄 Running FedAvg baseline...")

    # Initialize model
    model = SimpleNN(input_dim=784, hidden_dim=128, output_dim=10)

    # Initialize server
    fedavg_config = FedAvgConfig(
        num_clients=config.num_clients,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )
    server = FedAvgServer(model, fedavg_config)

    # Initialize clients
    clients = [
        FedAvgClient(
            client_id=i,
            model=SimpleNN(input_dim=784, hidden_dim=128, output_dim=10),
            train_loader=train_loaders[i],
            config=fedavg_config,
            device=config.device
        )
        for i in range(config.num_clients)
    ]

    # Training loop
    results = {
        "algorithm": "FedAvg",
        "rounds": [],
        "accuracies": [],
        "losses": [],
        "round_times": []
    }

    for round_num in range(config.rounds):
        start_time = time.time()

        # Run round
        round_stats = run_fedavg_round(server, clients)

        # Evaluate
        eval_stats = evaluate_model(server.global_model, test_loader, config.device)

        round_time = time.time() - start_time

        # Record results
        results["rounds"].append(round_num + 1)
        results["accuracies"].append(eval_stats["accuracy"])
        results["losses"].append(eval_stats["loss"])
        results["round_times"].append(round_time)

        if (round_num + 1) % 10 == 0:
            print(f"  Round {round_num + 1}: "
                  f"Accuracy = {eval_stats['accuracy']:.4f}, "
                  f"Loss = {eval_stats['loss']:.4f}, "
                  f"Time = {round_time:.2f}s")

    print(f"✅ FedAvg complete: Final accuracy = {results['accuracies'][-1]:.4f}")
    return results


def run_zerotrustml_krum_experiment(
    config: ExperimentConfig,
    train_loaders: List,
    test_loader
) -> Dict:
    """Run ZeroTrustML with Krum aggregation"""
    print("\n🔄 Running ZeroTrustML (Krum)...")

    # Initialize global model
    global_model = SimpleNN(input_dim=784, hidden_dim=128, output_dim=10)

    # Krum is a static class - no initialization needed

    # Initialize clients (using same RealMLNode structure)
    from zerotrustml.core.training import RealMLNode

    clients = [
        RealMLNode(
            node_id=i,
            model=SimpleNN(input_dim=784, hidden_dim=128, output_dim=10),
            device=config.device
        )
        for i in range(config.num_clients)
    ]

    # Replace synthetic data with real MNIST
    for i, client in enumerate(clients):
        client.train_loader = train_loaders[i]
        client.test_loader = test_loader

    # Training loop
    results = {
        "algorithm": "ZeroTrustML_Krum",
        "rounds": [],
        "accuracies": [],
        "losses": [],
        "round_times": []
    }

    for round_num in range(config.rounds):
        start_time = time.time()

        # 1. Clients compute gradients
        gradients = [client.compute_gradient() for client in clients]

        # 2. Aggregate with Krum (static method, equal reputations for fair comparison)
        reputations = [1.0] * len(gradients)
        aggregated_gradient = Krum.aggregate(gradients, reputations, num_byzantine=2)

        # 3. Update global model
        idx = 0
        with torch.no_grad():
            for param in global_model.parameters():
                param_size = param.numel()
                param_grad = aggregated_gradient[idx:idx+param_size].reshape(param.shape)
                param.data -= config.learning_rate * torch.from_numpy(param_grad).float()
                idx += param_size

        # 4. Distribute to clients
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())

        # 5. Evaluate
        eval_stats = evaluate_model(global_model, test_loader, config.device)

        round_time = time.time() - start_time

        # Record results
        results["rounds"].append(round_num + 1)
        results["accuracies"].append(eval_stats["accuracy"])
        results["losses"].append(eval_stats["loss"])
        results["round_times"].append(round_time)

        if (round_num + 1) % 10 == 0:
            print(f"  Round {round_num + 1}: "
                  f"Accuracy = {eval_stats['accuracy']:.4f}, "
                  f"Loss = {eval_stats['loss']:.4f}, "
                  f"Time = {round_time:.2f}s")

    print(f"✅ ZeroTrustML complete: Final accuracy = {results['accuracies'][-1]:.4f}")
    return results


def save_results(results: Dict, config: ExperimentConfig):
    """Save experiment results"""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_accuracy_{results['algorithm']}_{timestamp}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved: {filepath}")


def main():
    """Run complete experiment"""
    print("=" * 70)
    print("EXPERIMENT 1: MNIST Accuracy Comparison")
    print("Budget: $0 (local CPU training)")
    print("=" * 70)

    # Configuration
    config = ExperimentConfig()
    print(f"\n⚙️ Configuration:")
    print(f"  Clients: {config.num_clients}")
    print(f"  Rounds: {config.rounds}")
    print(f"  Local epochs: {config.local_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Data split: {config.data_split}")
    print(f"  Device: {config.device}")

    # Load dataset
    print(f"\n📥 Loading MNIST dataset...")
    mnist = MNISTFederated(data_dir="../data", download=True)

    # Create federated split
    if config.data_split == "iid":
        train_loaders = mnist.create_iid_split(num_clients=config.num_clients)
    else:
        train_loaders = mnist.create_non_iid_dirichlet(
            num_clients=config.num_clients,
            alpha=0.5
        )

    test_loader = mnist.get_test_loader()

    # Run experiments
    fedavg_results = run_fedavg_experiment(config, train_loaders, test_loader)
    zerotrustml_results = run_zerotrustml_krum_experiment(config, train_loaders, test_loader)

    # Save results
    save_results(fedavg_results, config)
    save_results(zerotrustml_results, config)

    # Summary comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFinal Accuracy:")
    print(f"  FedAvg:      {fedavg_results['accuracies'][-1]:.4f}")
    print(f"  ZeroTrustML:     {zerotrustml_results['accuracies'][-1]:.4f}")
    print(f"  Improvement: {(zerotrustml_results['accuracies'][-1] - fedavg_results['accuracies'][-1])*100:.2f}%")

    print(f"\nConvergence Speed (rounds to 90% accuracy):")
    fedavg_rounds = next((i for i, acc in enumerate(fedavg_results['accuracies']) if acc >= 0.90), config.rounds)
    zerotrustml_rounds = next((i for i, acc in enumerate(zerotrustml_results['accuracies']) if acc >= 0.90), config.rounds)
    print(f"  FedAvg:  {fedavg_rounds + 1} rounds")
    print(f"  ZeroTrustML: {zerotrustml_rounds + 1} rounds")

    print(f"\nAverage Round Time:")
    print(f"  FedAvg:  {np.mean(fedavg_results['round_times']):.2f}s")
    print(f"  ZeroTrustML: {np.mean(zerotrustml_results['round_times']):.2f}s")

    print("\n✅ Experiment complete! Results saved to benchmarks/results/")
    print("   Next: Generate paper figures from results")


if __name__ == "__main__":
    main()
