# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)

Reference: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
           Karimireddy et al., ICML 2020
           https://arxiv.org/abs/1910.06378

Algorithm:
SCAFFOLD uses control variates to correct for "client drift" - the phenomenon
where client models diverge significantly from each other during local training.

Key Innovation:
- Server maintains control variate c (tracks expected update direction)
- Each client maintains control variate c_i (tracks their local drift)
- During local training, client corrects gradients: g - c_i + c
- After training, client updates their control variate

Benefits:
- 3-4x fewer communication rounds vs FedAvg on non-IID data
- Provable convergence guarantees
- Works well with partial client participation

Mathematical Formulation:
    Local update: w_{i,t+1} = w_{i,t} - η(∇F_i(w_{i,t}) - c_i + c)
    Client control update: c_i^{new} = c_i - c + 1/(ηK)(w_i - w)
    Server control update: c^{new} = c + 1/n Σ(c_i^{new} - c_i)
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SCAFFOLDConfig:
    """Configuration for SCAFFOLD algorithm."""
    learning_rate: float = 0.01
    local_epochs: int = 1  # K in the paper
    batch_size: int = 10
    num_clients: int = 10
    fraction_clients: float = 1.0  # Fraction of clients selected each round


class SCAFFOLDServer:
    """
    SCAFFOLD Server.

    Maintains global model and global control variate.
    Aggregates client updates and control variate updates.
    """

    def __init__(self, model: nn.Module, config: SCAFFOLDConfig, device: str = 'cpu'):
        """
        Initialize SCAFFOLD server.

        Args:
            model: PyTorch model to train federally
            config: SCAFFOLD configuration
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.round = 0

        # Global control variate (c in paper)
        # Initialized to zero
        self.control_variate = [
            np.zeros_like(param.cpu().detach().numpy())
            for param in self.model.parameters()
        ]

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

    def get_control_variate(self) -> List[np.ndarray]:
        """Get current global control variate."""
        return [c.copy() for c in self.control_variate]

    def aggregate(self, client_updates: List[Dict]) -> tuple:
        """
        Aggregate client updates using SCAFFOLD algorithm.

        SCAFFOLD aggregation:
        1. Aggregate model updates (weighted average)
        2. Aggregate control variate updates

        Args:
            client_updates: List of dicts with keys:
                - 'weights': Client's updated weights
                - 'control_variate_delta': Change in client's control variate
                - 'num_samples': Number of training samples

        Returns:
            (aggregated_weights, control_variate_delta) tuple
        """
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)

        # Initialize aggregated weights and control variate delta
        num_layers = len(client_updates[0]['weights'])
        aggregated_weights = [
            np.zeros_like(client_updates[0]['weights'][i])
            for i in range(num_layers)
        ]
        control_variate_delta = [
            np.zeros_like(client_updates[0]['control_variate_delta'][i])
            for i in range(num_layers)
        ]

        # Weighted averaging for both weights and control variates
        for update in client_updates:
            weight = update['num_samples'] / total_samples

            for i in range(num_layers):
                aggregated_weights[i] += weight * update['weights'][i]
                control_variate_delta[i] += weight * update['control_variate_delta'][i]

        return aggregated_weights, control_variate_delta

    def update_control_variate(self, delta: List[np.ndarray]):
        """
        Update global control variate.

        c^{new} = c + delta

        Args:
            delta: Change in control variate from aggregated client updates
        """
        for i in range(len(self.control_variate)):
            self.control_variate[i] += delta[i]

    def train_round(self, clients: List['SCAFFOLDClient']) -> Dict:
        """
        Execute one round of SCAFFOLD training.

        Args:
            clients: List of SCAFFOLDClient instances

        Returns:
            Dictionary with round statistics
        """
        self.round += 1

        # Select clients
        num_selected = max(1, int(self.config.fraction_clients * len(clients)))
        selected_clients = np.random.choice(clients, size=num_selected, replace=False)

        # Get current global weights and control variate
        global_weights = self.get_model_weights()
        global_control = self.get_control_variate()

        # Each client trains locally
        client_updates = []
        for client in selected_clients:
            # Send global weights and control variate to client
            client.set_model_weights(global_weights)
            client.set_global_control_variate(global_control)

            # Client trains locally with SCAFFOLD correction
            client_update = client.train()
            client_updates.append(client_update)

        # Aggregate client updates
        aggregated_weights, control_variate_delta = self.aggregate(client_updates)

        # Update global model
        self.set_model_weights(aggregated_weights)

        # Update global control variate
        self.update_control_variate(control_variate_delta)

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


class SCAFFOLDClient:
    """
    SCAFFOLD Client.

    Key features:
    - Maintains local control variate c_i
    - Corrects gradients during training: g - c_i + c
    - Updates control variate after training
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: SCAFFOLDConfig,
        device: str = 'cpu'
    ):
        """
        Initialize SCAFFOLD client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model (same architecture as server)
            train_data: DataLoader with client's local training data
            config: SCAFFOLD configuration
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

        # Client control variate (c_i in paper)
        # Initialized to zero
        self.control_variate = [
            np.zeros_like(param.cpu().detach().numpy())
            for param in self.model.parameters()
        ]

        # Global control variate (received from server)
        self.global_control_variate = None

        # Store initial weights for control variate update
        self.initial_weights = None

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current local model weights."""
        return [param.cpu().detach().numpy().copy() for param in self.model.parameters()]

    def set_model_weights(self, weights: List[np.ndarray]):
        """Set local model weights."""
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_weight, device=self.device))

        # Store initial weights for control variate update
        self.initial_weights = [w.copy() for w in weights]

    def set_global_control_variate(self, control_variate: List[np.ndarray]):
        """Set global control variate from server."""
        self.global_control_variate = [c.copy() for c in control_variate]

    def apply_scaffold_correction(self):
        """
        Apply SCAFFOLD correction to gradients.

        For each parameter: grad = grad - c_i + c

        This corrects for client drift by:
        - Subtracting local drift direction (c_i)
        - Adding global expected direction (c)
        """
        if self.global_control_variate is None:
            return

        with torch.no_grad():
            for param, c_i, c in zip(
                self.model.parameters(),
                self.control_variate,
                self.global_control_variate
            ):
                if param.grad is not None:
                    # grad = grad - c_i + c
                    correction = torch.tensor(c - c_i, device=self.device)
                    param.grad.add_(correction)

    def update_control_variate(self):
        """
        Update client control variate after training.

        c_i^{new} = c_i - c + (w_initial - w_final) / (η * K)

        where:
        - w_initial: weights at start of local training
        - w_final: weights at end of local training
        - η: learning rate
        - K: number of local steps
        """
        if self.initial_weights is None or self.global_control_variate is None:
            return

        current_weights = self.get_model_weights()
        total_steps = self.config.local_epochs * len(self.train_data)

        control_variate_delta = []

        for i, (w_init, w_final, c_i, c) in enumerate(zip(
            self.initial_weights,
            current_weights,
            self.control_variate,
            self.global_control_variate
        )):
            # Calculate option 1: y_i (server gradient estimator)
            option_1 = (w_init - w_final) / (self.config.learning_rate * total_steps)

            # Update: c_i^{new} = c_i - c + option_1
            c_i_new = c_i - c + option_1

            # Store delta for server aggregation
            delta = c_i_new - c_i
            control_variate_delta.append(delta)

            # Update local control variate
            self.control_variate[i] = c_i_new

        return control_variate_delta

    def train(self) -> Dict:
        """
        Train local model for K epochs with SCAFFOLD correction.

        Returns:
            Dictionary with:
                - 'weights': Updated model weights
                - 'control_variate_delta': Change in control variate
                - 'num_samples': Number of training samples
                - 'loss': Average training loss
                - 'accuracy': Training accuracy
        """
        self.model.train()

        epoch_losses = []
        epoch_accuracies = []

        # Train for K local epochs
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

                # Apply SCAFFOLD correction to gradients
                self.apply_scaffold_correction()

                # Optimizer step
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

        # Update control variate based on training trajectory
        control_variate_delta = self.update_control_variate()

        # Return updated weights and statistics
        return {
            'weights': self.get_model_weights(),
            'control_variate_delta': control_variate_delta,
            'num_samples': self.num_samples,
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies),
            'client_id': self.client_id,
        }


def create_scaffold_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_data: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[SCAFFOLDConfig] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Create a SCAFFOLD experiment with server and clients.

    Args:
        model_fn: Function that returns a fresh PyTorch model
        train_data_splits: List of DataLoaders (one per client)
        test_data: Optional test DataLoader for evaluation
        config: SCAFFOLD configuration (uses defaults if None)
        device: 'cpu' or 'cuda'

    Returns:
        (server, clients, test_data) tuple

    Example:
        >>> def model_fn():
        ...     return SimpleCNN()
        >>>
        >>> server, clients, test_data = create_scaffold_experiment(
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
        config = SCAFFOLDConfig()

    # Create server with global model
    global_model = model_fn()
    server = SCAFFOLDServer(global_model, config, device)

    # Create clients
    clients = []
    for i, train_data in enumerate(train_data_splits):
        client_model = model_fn()
        client = SCAFFOLDClient(
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
    print("✅ SCAFFOLD baseline implementation loaded")
    print("   Reference: Karimireddy et al., ICML 2020")
    print("   Algorithm: Control variates for client drift correction")
    print(f"   Key: Corrects gradients with g - c_i + c")
