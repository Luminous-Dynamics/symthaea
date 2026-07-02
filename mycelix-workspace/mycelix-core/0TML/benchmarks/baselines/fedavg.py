# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FedAvg Baseline Implementation

Reference: Communication-Efficient Learning of Deep Networks from Decentralized Data
McMahan et al., 2017 (AISTATS)

Standard federated learning algorithm for comparison against ZeroTrustML.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import copy


@dataclass
class FedAvgConfig:
    """Configuration for FedAvg algorithm"""
    num_clients: int = 10
    clients_per_round: int = 10  # C * num_clients
    local_epochs: int = 1
    learning_rate: float = 0.01
    batch_size: int = 32
    aggregation_rule: str = "mean"  # mean (FedAvg) or weighted_mean


class FedAvgServer:
    """
    FedAvg Server Implementation

    Aggregates client models by averaging parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: FedAvgConfig
    ):
        self.global_model = model
        self.config = config
        self.round = 0

        # Statistics
        self.train_losses = []
        self.test_accuracies = []

    def aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None
    ) -> nn.Module:
        """
        Aggregate client models using FedAvg

        Args:
            client_models: List of trained client models
            client_weights: Optional weights for each client (e.g., dataset sizes)

        Returns:
            Updated global model
        """
        # Simple averaging (FedAvg)
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Get global model parameters
        global_dict = self.global_model.state_dict()

        # Average each parameter
        for key in global_dict.keys():
            # Stack client parameters
            client_params = torch.stack([
                client_models[i].state_dict()[key].float() * client_weights[i]
                for i in range(len(client_models))
            ])

            # Sum weighted parameters
            global_dict[key] = client_params.sum(0)

        # Update global model
        self.global_model.load_state_dict(global_dict)
        self.round += 1

        return self.global_model

    def aggregate_gradients(
        self,
        client_gradients: List[np.ndarray],
        client_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Aggregate gradients (alternative to model aggregation)

        Args:
            client_gradients: List of gradient arrays
            client_weights: Optional weights for each client

        Returns:
            Aggregated gradient
        """
        if client_weights is None:
            # Equal weighting (standard FedAvg)
            return np.mean(client_gradients, axis=0)
        else:
            # Weighted average
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
            return sum(
                grad * weight
                for grad, weight in zip(client_gradients, normalized_weights)
            )


class FedAvgClient:
    """
    FedAvg Client Implementation

    Performs local training and returns updated model.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader,
        config: FedAvgConfig,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Statistics
        self.train_loss = 0.0
        self.samples_trained = 0

    def local_train(self) -> nn.Module:
        """
        Perform local training for specified epochs

        Returns:
            Updated local model
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9
        )

        total_loss = 0.0
        total_samples = 0

        # Local epochs
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Flatten images if needed (for MNIST)
                if len(data.shape) == 4 and data.shape[1] == 1:  # Grayscale
                    data = data.view(data.size(0), -1)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        # Update statistics
        self.train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        self.samples_trained = total_samples

        return self.model

    def compute_gradient(self) -> np.ndarray:
        """
        Compute gradient (alternative to returning full model)

        Returns:
            Flattened gradient array
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        # Zero gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Accumulate gradients over one epoch
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Flatten if needed
            if len(data.shape) == 4 and data.shape[1] == 1:
                data = data.view(data.size(0), -1)

            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

        # Extract gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.cpu().numpy().flatten())
            else:
                gradients.append(np.zeros_like(param.cpu().numpy().flatten()))

        return np.concatenate(gradients)


def run_fedavg_round(
    server: FedAvgServer,
    clients: List[FedAvgClient],
    selected_clients: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Run one round of FedAvg

    Args:
        server: FedAvg server
        clients: List of FedAvg clients
        selected_clients: Indices of clients to train (None = all)

    Returns:
        Dict with round statistics
    """
    # Select clients (random sample if specified)
    if selected_clients is None:
        selected_clients = list(range(len(clients)))

    # Distribute global model to clients
    global_model_state = server.global_model.state_dict()
    for client_idx in selected_clients:
        clients[client_idx].model.load_state_dict(copy.deepcopy(global_model_state))

    # Local training
    trained_models = []
    client_weights = []
    total_loss = 0.0

    for client_idx in selected_clients:
        client = clients[client_idx]
        trained_model = client.local_train()
        trained_models.append(copy.deepcopy(trained_model))
        client_weights.append(client.samples_trained)
        total_loss += client.train_loss * client.samples_trained

    # Aggregate
    server.aggregate(trained_models, client_weights)

    # Statistics
    avg_loss = total_loss / sum(client_weights) if sum(client_weights) > 0 else 0.0

    return {
        "round": server.round,
        "avg_loss": avg_loss,
        "clients_trained": len(selected_clients)
    }


def evaluate_model(model: nn.Module, test_loader, device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate model on test set

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use

    Returns:
        Dict with accuracy and loss
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Flatten if needed
            if len(data.shape) == 4 and data.shape[1] == 1:
                data = data.view(data.size(0), -1)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total
    }


if __name__ == "__main__":
    print("✅ FedAvg baseline implementation ready")
    print("   Reference: McMahan et al. (2017)")
    print("   Use: Comparison baseline for ZeroTrustML")
