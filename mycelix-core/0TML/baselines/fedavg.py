# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FedAvg (Federated Averaging) Baseline Implementation

Reference: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
           McMahan et al., AISTATS 2017
           https://arxiv.org/abs/1602.05629

Algorithm:
1. Server initializes global model
2. For each round t:
   a) Server sends global model to K clients
   b) Each client trains locally on their data for E epochs
   c) Clients send gradients/weights back to server
   d) Server aggregates via weighted average: w_t+1 = Σ (n_k / n) * w_k
      where n_k is the number of samples at client k

This is the BASELINE that all FL methods compare against.
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FedAvgConfig:
    """Configuration for FedAvg algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1  # E in the paper
    batch_size: int = 10
    num_clients: int = 10  # K in the paper
    fraction_clients: float = 1.0  # Fraction of clients selected each round


class FedAvgServer:
    """
    Federated Averaging Server.

    Implements the server-side logic of FedAvg:
    - Distributes global model to clients
    - Aggregates client updates via weighted averaging
    - No Byzantine resistance, no security checks
    """

    def __init__(self, model: nn.Module, config: FedAvgConfig, device: str = 'cpu'):
        """
        Initialize FedAvg server.

        Args:
            model: PyTorch model to train federally
            config: FedAvg configuration
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.round = 0

        # Statistics tracking
        self.stats = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def get_model_weights(self) -> List[np.ndarray]:
        """
        Get current global model weights.

        Returns:
            List of numpy arrays (one per layer)
        """
        return [param.cpu().detach().numpy().copy() for param in self.model.parameters()]

    def set_model_weights(self, weights: List[np.ndarray]):
        """
        Set global model weights.

        Args:
            weights: List of numpy arrays (one per layer)
        """
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_weight, device=self.device))

    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate client updates using weighted averaging.

        FedAvg aggregation formula:
            w_global = Σ (n_k / n_total) * w_k

        Args:
            client_updates: List of dicts with keys:
                - 'weights': List[np.ndarray] - Client's local weights
                - 'num_samples': int - Number of samples used for training

        Returns:
            Aggregated weights as List[np.ndarray]
        """
        # Calculate total samples across all clients
        total_samples = sum(update['num_samples'] for update in client_updates)

        # Initialize aggregated weights with zeros
        num_layers = len(client_updates[0]['weights'])
        aggregated_weights = [
            np.zeros_like(client_updates[0]['weights'][i])
            for i in range(num_layers)
        ]

        # Weighted averaging
        for update in client_updates:
            weight = update['num_samples'] / total_samples

            for i, layer_weights in enumerate(update['weights']):
                aggregated_weights[i] += weight * layer_weights

        return aggregated_weights

    def train_round(self, clients: List['FedAvgClient']) -> Dict:
        """
        Execute one round of federated training.

        Args:
            clients: List of FedAvgClient instances

        Returns:
            Dictionary with round statistics
        """
        self.round += 1

        # Select clients (if fraction < 1.0)
        num_selected = max(1, int(self.config.fraction_clients * len(clients)))
        selected_clients = np.random.choice(clients, size=num_selected, replace=False)

        # Get current global weights
        global_weights = self.get_model_weights()

        # Each client trains locally
        client_updates = []
        for client in selected_clients:
            # Send global weights to client
            client.set_model_weights(global_weights)

            # Client trains locally
            client_update = client.train()
            client_updates.append(client_update)

        # Aggregate client updates
        aggregated_weights = self.aggregate(client_updates)

        # Update global model
        self.set_model_weights(aggregated_weights)

        # Calculate statistics
        avg_loss = np.mean([update['loss'] for update in client_updates])
        avg_accuracy = np.mean([update['accuracy'] for update in client_updates])

        # Record statistics
        self.stats['rounds'].append(self.round)
        self.stats['train_loss'].append(avg_loss)
        self.stats['train_accuracy'].append(avg_accuracy)

        return {
            'round': self.round,
            'num_clients': len(selected_clients),
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
        }


class FedAvgClient:
    """
    Federated Averaging Client.

    Implements the client-side logic of FedAvg:
    - Receives global model from server
    - Trains locally on private data for E epochs
    - Returns updated weights to server
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: FedAvgConfig,
        device: str = 'cpu'
    ):
        """
        Initialize FedAvg client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model (same architecture as server)
            train_data: DataLoader with client's local training data
            config: FedAvg configuration
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(self.device)

        # Calculate number of samples
        self.num_samples = len(train_data.dataset)

        # Optimizer for local training
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current local model weights."""
        return [param.cpu().detach().numpy().copy() for param in self.model.parameters()]

    def set_model_weights(self, weights: List[np.ndarray]):
        """Set local model weights."""
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_weight, device=self.device))

    def train(self) -> Dict:
        """
        Train local model for E epochs.

        Returns:
            Dictionary with:
                - 'weights': List[np.ndarray] - Updated model weights
                - 'num_samples': int - Number of training samples
                - 'loss': float - Average training loss
                - 'accuracy': float - Training accuracy
        """
        self.model.train()

        epoch_losses = []
        epoch_accuracies = []

        # Train for E local epochs
        for epoch in range(self.config.local_epochs):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track statistics
                batch_losses.append(loss.item())
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            # Epoch statistics
            epoch_loss = np.mean(batch_losses)
            epoch_accuracy = correct / total

            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

        # Return updated weights and statistics
        return {
            'weights': self.get_model_weights(),
            'num_samples': self.num_samples,
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies),
            'client_id': self.client_id,
        }


def create_fedavg_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[FedAvgConfig] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a FedAvg experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: FedAvg configuration (uses defaults if None)
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> def model_fn():
        ...     return SimpleCNN()
        >>>
        >>> server, clients, test_data = create_fedavg_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data
        ... )
        >>>
        >>> for round in range(num_rounds):
        ...     stats = server.train_round(clients)
        ...     print(f"Round {round}: Loss={stats['train_loss']:.4f}")
    """
    if config is None:
        config = FedAvgConfig()

    # Create server with global model
    global_model = model_fn()
    server = FedAvgServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()  # Each client has own model instance
        client = FedAvgClient(
            client_id=f"client_{i}",
            model=client_model,
            train_data=train_data,
            config=config,
            device=device
        )
        clients.append(client)

    return server, clients, test_data


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_global_model(
    model: nn.Module,
    test_data: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate global model on test data.

    Args:
        model: Global model to evaluate
        test_data: Test DataLoader
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with 'test_loss' and 'test_accuracy'
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return {
        'test_loss': test_loss / len(test_data),
        'test_accuracy': correct / total
    }


if __name__ == "__main__":
    print("✅ FedAvg baseline implementation loaded")
    print("   Reference: McMahan et al., AISTATS 2017")
    print("   Algorithm: Weighted averaging with no Byzantine resistance")
