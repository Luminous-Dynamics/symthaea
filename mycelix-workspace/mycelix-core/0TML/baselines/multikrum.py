# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Krum - Enhanced Byzantine-Robust Aggregation

Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
           Blanchard et al., NeurIPS 2017
           https://arxiv.org/abs/1703.02757

Algorithm:
Multi-Krum extends Krum by averaging the top-m gradients with lowest scores,
rather than selecting just one. This provides better robustness and smoother
convergence compared to single-gradient Krum.

Key Differences from Krum:
- Krum: Selects single best gradient (argmin of scores)
- Multi-Krum: Averages m gradients with lowest scores

Benefits:
- More robust than single Krum (averages reduce variance)
- Better convergence properties (smoother updates)
- Still Byzantine-tolerant up to f < n/2 - 1

Mathematical Formulation:
    1. Compute Krum score for each gradient i:
       score(i) = Σ_{j∈N(i,k)} ||g_i - g_j||²

    2. Select top-m gradients with lowest scores

    3. Average selected gradients:
       g_final = (1/m) Σ_{i∈S} g_i
       where S = set of m selected gradients
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MultiKrumConfig:
    """Configuration for Multi-Krum algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 10
    num_clients: int = 10
    fraction_clients: float = 1.0
    num_byzantine: int = 2  # f - number of Byzantine clients to tolerate
    multi_k: int = 5  # m - number of gradients to average (must be > f)


class MultiKrumServer:
    """
    Multi-Krum Server - Enhanced Byzantine-Robust Federated Learning.

    Averages top-m gradients with lowest Krum scores.
    """

    def __init__(self, model: nn.Module, config: MultiKrumConfig, device: str = 'cpu'):
        """
        Initialize Multi-Krum server.

        Args:
            model: PyTorch model to train federally
            config: Multi-Krum configuration
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.round = 0

        # Verify Byzantine tolerance constraint
        max_byzantine = (config.num_clients * config.fraction_clients) / 2 - 1
        if config.num_byzantine >= max_byzantine:
            raise ValueError(
                f"Cannot tolerate {config.num_byzantine} Byzantine clients. "
                f"Maximum: {int(max_byzantine)} for {config.num_clients} clients"
            )

        # Verify multi_k parameter
        if config.multi_k <= config.num_byzantine:
            raise ValueError(
                f"multi_k ({config.multi_k}) must be greater than "
                f"num_byzantine ({config.num_byzantine})"
            )

        # Statistics tracking
        self.stats = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'selected_clients': [],  # Track which clients were selected
        }

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current global model weights."""
        return [param.cpu().detach().numpy().copy() for param in self.model.parameters()]

    def set_model_weights(self, weights: List[np.ndarray]):
        """Set global model weights."""
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_weight, device=self.device))

    def flatten_weights(self, weights: List[np.ndarray]) -> np.ndarray:
        """Flatten list of weight arrays into single vector."""
        return np.concatenate([w.flatten() for w in weights])

    def compute_pairwise_distances(
        self,
        gradients: List[List[np.ndarray]]
    ) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between gradients.

        Args:
            gradients: List of gradient lists (one per client)

        Returns:
            Distance matrix of shape (n_clients, n_clients)
        """
        n = len(gradients)

        # Flatten all gradients to vectors
        flat_gradients = [self.flatten_weights(g) for g in gradients]

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((flat_gradients[i] - flat_gradients[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def multikrum_score(
        self,
        distances: np.ndarray,
        k: int
    ) -> tuple:
        """
        Compute Krum scores for all clients.

        Args:
            distances: Pairwise distance matrix (n x n)
            k: Number of nearest neighbors to consider

        Returns:
            (scores, selected_indices) tuple
            - scores: Array of Krum scores for each client
            - selected_indices: Indices of top-m clients with lowest scores
        """
        n = len(distances)
        scores = np.zeros(n)

        for i in range(n):
            # Get distances from client i to all others
            dists_i = distances[i].copy()

            # Exclude self (distance = 0)
            dists_i[i] = np.inf

            # Find k nearest neighbors
            nearest_indices = np.argpartition(dists_i, k-1)[:k]

            # Sum of squared distances to k nearest neighbors
            scores[i] = np.sum(dists_i[nearest_indices])

        # Select top-m clients with lowest scores
        m = min(self.config.multi_k, n)  # Don't select more than available
        selected_indices = np.argpartition(scores, m-1)[:m]

        return scores, selected_indices

    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate client updates using Multi-Krum selection.

        Multi-Krum averages the m gradients with lowest Krum scores.

        Args:
            client_updates: List of dicts with keys:
                - 'weights': Client's local weights
                - 'num_samples': Number of samples
                - 'client_id': Client identifier

        Returns:
            Averaged weights from selected clients
        """
        n = len(client_updates)
        f = self.config.num_byzantine

        # Calculate k (number of neighbors to consider)
        # From paper: k = n - f - 2
        k = n - f - 2
        if k < 1:
            k = max(1, n - 1)

        # Extract gradients
        gradients = [update['weights'] for update in client_updates]

        # Compute pairwise distances
        distances = self.compute_pairwise_distances(gradients)

        # Compute Krum scores and select top-m clients
        scores, selected_indices = self.multikrum_score(distances, k)

        # Record which clients were selected
        selected_client_ids = [
            client_updates[idx]['client_id']
            for idx in selected_indices
        ]
        self.stats['selected_clients'].append(selected_client_ids)

        # Average the selected gradients
        num_layers = len(gradients[0])
        aggregated_weights = [
            np.zeros_like(gradients[0][i])
            for i in range(num_layers)
        ]

        m = len(selected_indices)
        for idx in selected_indices:
            for i in range(num_layers):
                aggregated_weights[i] += gradients[idx][i] / m

        return aggregated_weights

    def train_round(self, clients: List['MultiKrumClient']) -> Dict:
        """
        Execute one round of federated training with Multi-Krum aggregation.

        Args:
            clients: List of MultiKrumClient instances (may include Byzantine)

        Returns:
            Dictionary with round statistics
        """
        self.round += 1

        # Select clients
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

        # Aggregate using Multi-Krum (averages top-m gradients)
        aggregated_weights = self.aggregate(client_updates)

        # Update global model
        self.set_model_weights(aggregated_weights)

        # Calculate statistics (average from selected clients only)
        selected_client_ids = self.stats['selected_clients'][-1]
        selected_updates = [
            u for u in client_updates
            if u['client_id'] in selected_client_ids
        ]

        avg_loss = np.mean([u['loss'] for u in selected_updates])
        avg_accuracy = np.mean([u['accuracy'] for u in selected_updates])

        # Record statistics
        self.stats['rounds'].append(self.round)
        self.stats['train_loss'].append(avg_loss)
        self.stats['train_accuracy'].append(avg_accuracy)

        return {
            'round': self.round,
            'num_clients': len(selected_clients),
            'num_selected_for_aggregation': len(selected_client_ids),
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'selected_clients': selected_client_ids,
        }


class MultiKrumClient:
    """
    Multi-Krum Client.

    Identical to standard FedAvg client - Multi-Krum defense is server-side only.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: MultiKrumConfig,
        device: str = 'cpu',
        is_byzantine: bool = False
    ):
        """
        Initialize Multi-Krum client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_data: DataLoader with client's local training data
            config: Multi-Krum configuration
            device: 'cpu' or 'cuda'
            is_byzantine: If True, client sends malicious updates
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(self.device)
        self.is_byzantine = is_byzantine

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

    def generate_byzantine_update(self) -> List[np.ndarray]:
        """
        Generate malicious update for Byzantine client.

        Uses random noise attack (same as Krum).
        """
        weights = self.get_model_weights()

        # Add large random noise
        byzantine_weights = [
            w + np.random.normal(0, 10.0, w.shape)
            for w in weights
        ]

        return byzantine_weights

    def train(self) -> Dict:
        """
        Train local model for E epochs.

        If Byzantine, returns malicious update instead.

        Returns:
            Dictionary with weights, num_samples, loss, accuracy, client_id
        """
        # Byzantine clients return malicious updates
        if self.is_byzantine:
            return {
                'weights': self.generate_byzantine_update(),
                'num_samples': self.num_samples,
                'loss': 0.0,  # Fake metrics
                'accuracy': 0.0,
                'client_id': self.client_id,
            }

        # Normal training for honest clients
        self.model.train()

        epoch_losses = []
        epoch_accuracies = []

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

            epoch_losses.append(np.mean(batch_losses))
            epoch_accuracies.append(correct / total)

        return {
            'weights': self.get_model_weights(),
            'num_samples': self.num_samples,
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies),
            'client_id': self.client_id,
        }


def create_multikrum_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[MultiKrumConfig] = None,
    byzantine_clients: Optional[List[int]] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a Multi-Krum experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: Multi-Krum configuration (uses defaults if None)
        byzantine_clients: List of client indices to make Byzantine
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> server, clients, test_data = create_multikrum_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data,
        ...     config=MultiKrumConfig(num_byzantine=2, multi_k=5),
        ...     byzantine_clients=[5, 7]
        ... )
    """
    if config is None:
        config = MultiKrumConfig()

    if byzantine_clients is None:
        byzantine_clients = []

    # Create server with global model
    global_model = model_fn()
    server = MultiKrumServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()
        is_byzantine = i in byzantine_clients

        client = MultiKrumClient(
            client_id=f"client_{i}",
            model=client_model,
            train_data=train_data,
            config=config,
            device=device,
            is_byzantine=is_byzantine
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
    """Evaluate global model on test data."""
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
    print("✅ Multi-Krum baseline implementation loaded")
    print("   Reference: Blanchard et al., NeurIPS 2017")
    print("   Algorithm: Byzantine-robust gradient averaging")
    print(f"   Key: Averages top-m gradients with lowest scores")
