# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Coordinate-wise Median - Simple Byzantine-Robust Aggregation

Reference: "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
           Yin et al., ICML 2018
           https://arxiv.org/abs/1803.01498

Algorithm:
Coordinate-wise median is the simplest Byzantine-robust aggregation method.
For each coordinate (weight parameter), it takes the median value across all clients.

Key Idea:
- Byzantine clients cannot shift the median significantly unless they are > 50%
- Simple, fast, and parameter-free
- Provably robust up to f < n/2 Byzantine clients

Benefits:
- Simplest Byzantine-robust method (no hyperparameters)
- Fast computation (O(n log n) per coordinate)
- Provable guarantees: robust if f < n/2
- Works with any distribution of honest updates

Drawbacks:
- Less efficient than FedAvg with only honest clients
- Can be slower to converge than FedAvg
- Coordinate-wise operation ignores correlations between parameters

Mathematical Formulation:
    For each coordinate i:
        w_global[i] = median(w_1[i], w_2[i], ..., w_n[i])

    Where w_k = gradient/weights from client k
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MedianConfig:
    """Configuration for Median aggregation."""
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 10
    num_clients: int = 10
    fraction_clients: float = 1.0


class MedianServer:
    """
    Median Server - Simple Byzantine-Robust Federated Learning.

    Aggregates client updates using coordinate-wise median.
    """

    def __init__(self, model: nn.Module, config: MedianConfig, device: str = 'cpu'):
        """
        Initialize Median server.

        Args:
            model: PyTorch model to train federally
            config: Median configuration
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

    def unflatten_weights(
        self,
        flat_weights: np.ndarray,
        shapes: List[tuple]
    ) -> List[np.ndarray]:
        """Unflatten vector back to list of weight arrays."""
        weights = []
        offset = 0
        for shape in shapes:
            size = np.prod(shape)
            weights.append(flat_weights[offset:offset+size].reshape(shape))
            offset += size
        return weights

    def coordinate_wise_median(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Compute coordinate-wise median of gradients.

        For each coordinate, take the median value across all clients.

        Args:
            gradients: List of gradient lists (one per client)

        Returns:
            Aggregated gradient with median values
        """
        # Flatten all gradients to vectors
        flat_gradients = np.array([self.flatten_weights(g) for g in gradients])

        # Compute median along axis 0 (across clients)
        # Shape: (n_clients, n_coordinates) -> (n_coordinates,)
        median_gradient = np.median(flat_gradients, axis=0)

        # Unflatten back to weight shapes
        shapes = [w.shape for w in gradients[0]]
        return self.unflatten_weights(median_gradient, shapes)

    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate client updates using coordinate-wise median.

        Args:
            client_updates: List of dicts with keys:
                - 'weights': Client's local weights
                - 'num_samples': Number of samples (not used by median)
                - 'client_id': Client identifier

        Returns:
            Median-aggregated weights
        """
        # Extract gradients from all clients
        gradients = [update['weights'] for update in client_updates]

        # Compute coordinate-wise median
        aggregated_weights = self.coordinate_wise_median(gradients)

        return aggregated_weights

    def train_round(self, clients: List['MedianClient']) -> Dict:
        """
        Execute one round of federated training with Median aggregation.

        Args:
            clients: List of MedianClient instances (may include Byzantine)

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

        # Aggregate using coordinate-wise median
        aggregated_weights = self.aggregate(client_updates)

        # Update global model
        self.set_model_weights(aggregated_weights)

        # Calculate statistics (average from all clients)
        # Note: This includes Byzantine clients, which may report fake metrics
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


class MedianClient:
    """
    Median Client.

    Identical to standard FedAvg client - Median defense is server-side only.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: MedianConfig,
        device: str = 'cpu',
        is_byzantine: bool = False
    ):
        """
        Initialize Median client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_data: DataLoader with client's local training data
            config: Median configuration
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

        Common attacks:
        - Random noise: Add large random values
        - Sign flip: Negate all weights
        - Constant: Return constant malicious values

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


def create_median_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[MedianConfig] = None,
    byzantine_clients: Optional[List[int]] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a Median aggregation experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: Median configuration (uses defaults if None)
        byzantine_clients: List of client indices to make Byzantine
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> server, clients, test_data = create_median_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data,
        ...     byzantine_clients=[5, 7]  # Can tolerate f < n/2
        ... )
        >>>
        >>> for round in range(num_rounds):
        ...     stats = server.train_round(clients)
        ...     print(f"Round {round}: Loss={stats['train_loss']:.4f}")
    """
    if config is None:
        config = MedianConfig()

    if byzantine_clients is None:
        byzantine_clients = []

    # Create server with global model
    global_model = model_fn()
    server = MedianServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()
        is_byzantine = i in byzantine_clients

        client = MedianClient(
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
    print("✅ Median baseline implementation loaded")
    print("   Reference: Yin et al., ICML 2018")
    print("   Algorithm: Coordinate-wise median aggregation")
    print(f"   Key: Simplest Byzantine-robust method (robust if f < n/2)")
