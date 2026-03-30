# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Bulyan - Strongest Byzantine-Robust Aggregation

Reference: "The Hidden Vulnerability of Distributed Learning in Byzantium"
           El Mhamdi et al., ICML 2018
           https://arxiv.org/abs/1802.07927

Algorithm:
Bulyan combines Multi-Krum selection with coordinate-wise trimmed mean to
provide the strongest known Byzantine tolerance guarantees in federated learning.

Key Innovation:
1. Use Multi-Krum to select θ = n - 2f gradients
2. Apply coordinate-wise trimmed mean (remove β highest and lowest values per coordinate)
3. Provably robust up to f < n/3 Byzantine clients (stronger than Krum's f < n/2)

Benefits:
- Strongest Byzantine tolerance: f < n/3 (vs Krum's f < n/2)
- Coordinate-wise trimming prevents dimension-specific attacks
- Provable convergence guarantees
- Works with both IID and non-IID data

Mathematical Formulation:
    Phase 1 (Selection): Use Multi-Krum to select θ = n - 2f gradients
    Phase 2 (Trimming): For each coordinate i:
        - Sort selected values: g_1[i], g_2[i], ..., g_θ[i]
        - Remove β largest and β smallest
        - Average remaining values
    where β = f/2
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class BulyanConfig:
    """Configuration for Bulyan algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 10
    num_clients: int = 10
    fraction_clients: float = 1.0
    num_byzantine: int = 2  # f - must be < n/3 for Bulyan


class BulyanServer:
    """
    Bulyan Server - Strongest Byzantine-Robust Federated Learning.

    Two-phase aggregation:
    1. Multi-Krum selection of θ = n - 2f gradients
    2. Coordinate-wise trimmed mean with β = f/2
    """

    def __init__(self, model: nn.Module, config: BulyanConfig, device: str = 'cpu'):
        """
        Initialize Bulyan server.

        Args:
            model: PyTorch model to train federally
            config: Bulyan configuration
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.round = 0

        # Verify Byzantine tolerance constraint (f < n/3)
        max_byzantine = (config.num_clients * config.fraction_clients) / 3 - 1
        if config.num_byzantine >= max_byzantine:
            raise ValueError(
                f"Cannot tolerate {config.num_byzantine} Byzantine clients. "
                f"Maximum: {int(max_byzantine)} for {config.num_clients} clients "
                f"(Bulyan requires f < n/3)"
            )

        # Statistics tracking
        self.stats = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'selected_clients': [],  # Phase 1 selection
            'trimmed_clients': [],   # Phase 2 after trimming
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

    def compute_pairwise_distances(
        self,
        gradients: List[List[np.ndarray]]
    ) -> np.ndarray:
        """Compute pairwise squared Euclidean distances between gradients."""
        n = len(gradients)
        flat_gradients = [self.flatten_weights(g) for g in gradients]

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((flat_gradients[i] - flat_gradients[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def multikrum_selection(
        self,
        gradients: List[List[np.ndarray]],
        theta: int
    ) -> List[int]:
        """
        Phase 1: Multi-Krum selection of θ gradients.

        Args:
            gradients: List of all client gradients
            theta: Number of gradients to select (n - 2f)

        Returns:
            Indices of selected gradients
        """
        n = len(gradients)
        f = self.config.num_byzantine

        # Calculate k for Krum scoring
        k = n - f - 2
        if k < 1:
            k = max(1, n - 1)

        # Compute pairwise distances
        distances = self.compute_pairwise_distances(gradients)

        # Compute Krum scores
        scores = np.zeros(n)
        for i in range(n):
            dists_i = distances[i].copy()
            dists_i[i] = np.inf  # Exclude self
            nearest_indices = np.argpartition(dists_i, k-1)[:k]
            scores[i] = np.sum(dists_i[nearest_indices])

        # Select θ gradients with lowest scores
        theta = min(theta, n)  # Don't select more than available
        selected_indices = np.argpartition(scores, theta-1)[:theta]

        return selected_indices.tolist()

    def trimmed_mean(
        self,
        values: np.ndarray,
        beta: int
    ) -> float:
        """
        Compute trimmed mean of values.

        Args:
            values: Array of values for a single coordinate
            beta: Number of values to trim from each end

        Returns:
            Trimmed mean
        """
        if len(values) <= 2 * beta:
            # Not enough values to trim, return regular mean
            return np.mean(values)

        # Sort values
        sorted_values = np.sort(values)

        # Trim β smallest and β largest
        trimmed = sorted_values[beta:-beta]

        # Return mean of remaining values
        return np.mean(trimmed)

    def coordinate_wise_trimmed_mean(
        self,
        gradients: List[List[np.ndarray]],
        beta: int
    ) -> List[np.ndarray]:
        """
        Phase 2: Apply coordinate-wise trimmed mean.

        For each coordinate, remove β largest and β smallest values,
        then average the remaining values.

        Args:
            gradients: Selected gradients from Phase 1
            beta: Number of values to trim from each end (typically f/2)

        Returns:
            Aggregated gradient
        """
        # Flatten all gradients
        flat_gradients = np.array([self.flatten_weights(g) for g in gradients])
        n_coords = flat_gradients.shape[1]

        # Apply trimmed mean coordinate-wise
        result = np.zeros(n_coords)
        for coord in range(n_coords):
            values = flat_gradients[:, coord]
            result[coord] = self.trimmed_mean(values, beta)

        # Unflatten back to weight shapes
        shapes = [w.shape for w in gradients[0]]
        return self.unflatten_weights(result, shapes)

    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        """
        Aggregate client updates using Bulyan.

        Two-phase process:
        1. Multi-Krum: Select θ = n - 2f gradients
        2. Trimmed Mean: Apply coordinate-wise trimming with β = f/2

        Args:
            client_updates: List of dicts with keys:
                - 'weights': Client's local weights
                - 'num_samples': Number of samples
                - 'client_id': Client identifier

        Returns:
            Aggregated weights
        """
        n = len(client_updates)
        f = self.config.num_byzantine

        # Phase 1: Multi-Krum selection
        # Select θ = n - 2f gradients
        theta = n - 2 * f
        if theta < 1:
            theta = max(1, n)

        gradients = [update['weights'] for update in client_updates]
        selected_indices = self.multikrum_selection(gradients, theta)

        # Record Phase 1 selection
        selected_client_ids = [
            client_updates[idx]['client_id']
            for idx in selected_indices
        ]
        self.stats['selected_clients'].append(selected_client_ids)

        # Phase 2: Coordinate-wise trimmed mean
        # Trim β = f/2 from each end
        beta = max(1, f // 2)

        selected_gradients = [gradients[idx] for idx in selected_indices]
        aggregated_weights = self.coordinate_wise_trimmed_mean(
            selected_gradients,
            beta
        )

        return aggregated_weights

    def train_round(self, clients: List['BulyanClient']) -> Dict:
        """
        Execute one round of federated training with Bulyan aggregation.

        Args:
            clients: List of BulyanClient instances (may include Byzantine)

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
            client.set_model_weights(global_weights)
            client_update = client.train()
            client_updates.append(client_update)

        # Aggregate using Bulyan (Multi-Krum + Trimmed Mean)
        aggregated_weights = self.aggregate(client_updates)

        # Update global model
        self.set_model_weights(aggregated_weights)

        # Calculate statistics (from selected clients only)
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
            'num_selected_phase1': len(selected_client_ids),
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'selected_clients': selected_client_ids,
        }


class BulyanClient:
    """
    Bulyan Client.

    Identical to standard FedAvg client - Bulyan defense is server-side only.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: BulyanConfig,
        device: str = 'cpu',
        is_byzantine: bool = False
    ):
        """Initialize Bulyan client."""
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(self.device)
        self.is_byzantine = is_byzantine

        self.num_samples = len(train_data.dataset)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate
        )

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
        """Generate malicious update for Byzantine client."""
        weights = self.get_model_weights()
        return [w + np.random.normal(0, 10.0, w.shape) for w in weights]

    def train(self) -> Dict:
        """Train local model for E epochs."""
        if self.is_byzantine:
            return {
                'weights': self.generate_byzantine_update(),
                'num_samples': self.num_samples,
                'loss': 0.0,
                'accuracy': 0.0,
                'client_id': self.client_id,
            }

        self.model.train()

        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.config.local_epochs):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

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


def create_bulyan_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[BulyanConfig] = None,
    byzantine_clients: Optional[List[int]] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a Bulyan experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: Bulyan configuration (uses defaults if None)
        byzantine_clients: List of client indices to make Byzantine
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> server, clients, test_data = create_bulyan_experiment(
        ...     model_fn,
        ...     train_splits,
        ...     test_data,
        ...     config=BulyanConfig(num_byzantine=3),  # f < n/3
        ...     byzantine_clients=[5, 7, 9]
        ... )
    """
    if config is None:
        config = BulyanConfig()

    if byzantine_clients is None:
        byzantine_clients = []

    # Create server with global model
    global_model = model_fn()
    server = BulyanServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()
        is_byzantine = i in byzantine_clients

        client = BulyanClient(
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
    print("✅ Bulyan baseline implementation loaded")
    print("   Reference: El Mhamdi et al., ICML 2018")
    print("   Algorithm: Multi-Krum + coordinate-wise trimmed mean")
    print(f"   Key: Strongest Byzantine tolerance (f < n/3)")
