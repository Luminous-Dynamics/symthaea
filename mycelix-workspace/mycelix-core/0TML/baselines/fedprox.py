# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FedProx (Federated Proximal) Implementation

Reference: "Federated Optimization in Heterogeneous Networks"
           Li et al., MLSys 2020
           https://arxiv.org/abs/1812.06127

Algorithm:
FedProx extends FedAvg by adding a proximal term to the local objective:
    min F_k(w) + (μ/2) ||w - w^t||²

This regularization helps with:
- Non-IID data (statistical heterogeneity)
- System heterogeneity (different device capabilities)
- Partial client participation

Key difference from FedAvg:
- Adds proximal term μ/2 ||w - w_global||² to local loss
- Controls divergence from global model during local training
- More stable convergence with heterogeneous data/systems
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FedProxConfig:
    """Configuration for FedProx algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1  # E in the paper
    batch_size: int = 10
    num_clients: int = 10  # K in the paper
    fraction_clients: float = 1.0  # Fraction of clients selected each round
    mu: float = 0.01  # Proximal term coefficient (higher = more regularization)


class FedProxServer:
    """
    FedProx Server.

    Identical to FedAvg server - the difference is in client-side training.
    Server still does weighted averaging of client updates.
    """

    def __init__(self, model: nn.Module, config: FedProxConfig, device: str = 'cpu'):
        """
        Initialize FedProx server.

        Args:
            model: PyTorch model to train federally
            config: FedProx configuration
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

        Same as FedAvg: w_global = Σ (n_k / n_total) * w_k

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

    def train_round(self, clients: List['FedProxClient']) -> Dict:
        """
        Execute one round of federated training.

        Args:
            clients: List of FedProxClient instances

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

            # Client trains locally (with proximal term)
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


class FedProxClient:
    """
    FedProx Client.

    Key difference from FedAvg:
    - Adds proximal term to loss: loss += (mu/2) * ||w - w_global||²
    - Prevents local model from diverging too far from global model
    - Especially helpful with non-IID data
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: FedProxConfig,
        device: str = 'cpu'
    ):
        """
        Initialize FedProx client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model (same architecture as server)
            train_data: DataLoader with client's local training data
            config: FedProx configuration
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

        # Global model weights (for proximal term)
        self.global_weights = None

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current local model weights."""
        return [param.cpu().detach().numpy().copy() for param in self.model.parameters()]

    def set_model_weights(self, weights: List[np.ndarray]):
        """
        Set local model weights.

        Also stores these as global_weights for proximal term calculation.
        """
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_weight, device=self.device))

        # Store global weights for proximal term
        self.global_weights = [w.copy() for w in weights]

    def proximal_term(self) -> torch.Tensor:
        """
        Calculate proximal term: (mu/2) * ||w - w_global||²

        Returns:
            Proximal term as scalar tensor
        """
        if self.global_weights is None:
            return torch.tensor(0.0, device=self.device)

        proximal_loss = 0.0
        for param, global_weight in zip(self.model.parameters(), self.global_weights):
            global_weight_tensor = torch.tensor(global_weight, device=self.device)
            proximal_loss += torch.sum((param - global_weight_tensor) ** 2)

        return (self.config.mu / 2.0) * proximal_loss

    def train(self) -> Dict:
        """
        Train local model for E epochs with proximal term.

        FedProx Loss = CrossEntropy + (mu/2) * ||w - w_global||²

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

                # Standard cross-entropy loss
                ce_loss = self.criterion(output, target)

                # Add proximal term (FedProx key difference)
                prox_term = self.proximal_term()

                # Total loss
                loss = ce_loss + prox_term

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


def create_fedprox_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[FedProxConfig] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a FedProx experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: FedProx configuration (uses defaults if None)
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> def model_fn():
        ...     return SimpleCNN()
        >>>
        >>> server, clients, test_data = create_fedprox_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data,
        ...     config=FedProxConfig(mu=0.01)  # Set proximal term
        ... )
        >>>
        >>> for round in range(num_rounds):
        ...     stats = server.train_round(clients)
        ...     print(f"Round {round}: Loss={stats['train_loss']:.4f}")
    """
    if config is None:
        config = FedProxConfig()

    # Create server with global model
    global_model = model_fn()
    server = FedProxServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()  # Each client has own model instance
        client = FedProxClient(
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
    print("✅ FedProx baseline implementation loaded")
    print("   Reference: Li et al., MLSys 2020")
    print("   Algorithm: FedAvg + proximal term for non-IID data")
    print(f"   Key: Adds (μ/2) ||w - w_global||² to local loss")
