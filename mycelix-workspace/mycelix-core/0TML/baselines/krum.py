# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Krum - Byzantine-Robust Aggregation

Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
           Blanchard et al., NeurIPS 2017
           https://arxiv.org/abs/1703.02757

Algorithm:
Krum is a Byzantine-robust aggregation rule that selects the gradient
closest to the majority of other gradients, based on sum of squared distances.

Key Idea:
- Byzantine clients may send arbitrary malicious updates
- Krum finds the gradient with smallest sum of distances to its k nearest neighbors
- This gradient is likely from an honest client (majority assumption)

Byzantine Tolerance:
- Can tolerate up to f < n/2 - 1 Byzantine clients
- Where n = total number of clients selected

Mathematical Formulation:
    For each gradient g_i, compute score:
        score(i) = Σ_{j∈N(i,k)} ||g_i - g_j||²

    Where N(i,k) = k nearest neighbors of g_i
    Select: argmin_i score(i)
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class KrumConfig:
    """Configuration for Krum algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 10
    num_clients: int = 10
    fraction_clients: float = 1.0
    num_byzantine: int = 2  # f - number of Byzantine clients to tolerate


class KrumServer:
    """
    Krum Server - Byzantine-Robust Federated Learning.

    Selects single most "representative" client update based on
    sum of squared distances to nearest neighbors.
    """

    def __init__(self, model: nn.Module, config: KrumConfig, device: str = 'cpu'):
        """
        Initialize Krum server.

        Args:
            model: PyTorch model to train federally
            config: Krum configuration
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

        # Statistics tracking
        self.stats = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'selected_clients': [],  # Track which clients were selected by Krum
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

    def krum_score(
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
            (scores, selected_index) tuple
            - scores: Array of Krum scores for each client
            - selected_index: Index of client with lowest score
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

        # Select client with minimum score
        selected_index = np.argmin(scores)

        return scores, selected_index

    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate client updates using Krum selection.

        Krum selects the single gradient with lowest score.

        Args:
            client_updates: List of dicts with keys:
                - 'weights': Client's local weights
                - 'num_samples': Number of samples
                - 'client_id': Client identifier

        Returns:
            Selected client's weights (not averaged)
        """
        n = len(client_updates)
        f = self.config.num_byzantine

        # Calculate k (number of neighbors to consider)
        # From paper: k = n - f - 2
        k = n - f - 2
        if k < 1:
            k = max(1, n - 1)

        # Extract gradients (compute as difference from initial weights)
        gradients = [update['weights'] for update in client_updates]

        # Compute pairwise distances
        distances = self.compute_pairwise_distances(gradients)

        # Compute Krum scores and select best client
        scores, selected_idx = self.krum_score(distances, k)

        # Record which client was selected
        selected_client_id = client_updates[selected_idx]['client_id']
        self.stats['selected_clients'].append(selected_client_id)

        # Return the selected client's weights
        return client_updates[selected_idx]['weights']

    def train_round(self, clients: List['KrumClient']) -> Dict:
        """
        Execute one round of federated training with Krum aggregation.

        Args:
            clients: List of KrumClient instances (may include Byzantine clients)

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

        # Aggregate using Krum (selects single best gradient)
        selected_weights = self.aggregate(client_updates)

        # Update global model with selected weights
        self.set_model_weights(selected_weights)

        # Calculate statistics (from selected client)
        selected_idx = len(self.stats['selected_clients']) - 1
        selected_update = [u for u in client_updates
                          if u['client_id'] == self.stats['selected_clients'][selected_idx]][0]

        # Record statistics
        self.stats['rounds'].append(self.round)
        self.stats['train_loss'].append(selected_update['loss'])
        self.stats['train_accuracy'].append(selected_update['accuracy'])

        return {
            'round': self.round,
            'num_clients': len(selected_clients),
            'train_loss': selected_update['loss'],
            'train_accuracy': selected_update['accuracy'],
            'selected_client': selected_update['client_id'],
        }


class KrumClient:
    """
    Krum Client.

    Identical to standard FedAvg client - Krum defense is server-side only.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: KrumConfig,
        device: str = 'cpu',
        is_byzantine: bool = False
    ):
        """
        Initialize Krum client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_data: DataLoader with client's local training data
            config: Krum configuration
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

        Common Byzantine attacks:
        - Random noise: Add large random values
        - Sign flip: Negate all weights
        - Label flip: Train on flipped labels

        This implementation uses random noise attack.
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


def create_krum_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[KrumConfig] = None,
    byzantine_clients: Optional[List[int]] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a Krum experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: Krum configuration (uses defaults if None)
        byzantine_clients: List of client indices to make Byzantine
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> server, clients, test_data = create_krum_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data,
        ...     config=KrumConfig(num_byzantine=2),
        ...     byzantine_clients=[5, 7]  # Make clients 5 and 7 Byzantine
        ... )
    """
    if config is None:
        config = KrumConfig()

    if byzantine_clients is None:
        byzantine_clients = []

    # Create server with global model
    global_model = model_fn()
    server = KrumServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()
        is_byzantine = i in byzantine_clients

        client = KrumClient(
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
    print("✅ Krum baseline implementation loaded")
    print("   Reference: Blanchard et al., NeurIPS 2017")
    print("   Algorithm: Byzantine-robust gradient selection")
    print(f"   Key: Selects gradient closest to majority")
