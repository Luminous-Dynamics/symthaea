# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real ML training module using PyTorch for federated learning.

Provides neural network models and training logic for actual gradient computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict


class SimpleNN(nn.Module):
    """
    Simple fully-connected neural network for federated learning

    Default architecture for MNIST-like tasks:
    - Input layer: 784 neurons (28x28 images)
    - Hidden layer: 128 neurons with ReLU and Dropout
    - Output layer: 10 neurons (10 classes)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as flat numpy array"""
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray):
        """Set parameters from flat numpy array"""
        idx = 0
        for param in self.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            param.data = torch.from_numpy(
                flat_params[idx:idx+param_size].reshape(param_shape)
            ).float()
            idx += param_size


class RealMLNode:
    """
    Federated learning node with real PyTorch training

    Handles:
    - Local dataset management
    - Gradient computation via backpropagation
    - Model evaluation
    - Gradient validation (for PoGQ)
    """

    def __init__(
        self,
        node_id: int,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None
    ):
        self.node_id = node_id

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        if model is None:
            self.model = SimpleNN().to(self.device)
        else:
            self.model = model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        # Generate local dataset
        self.train_loader, self.test_loader = self._generate_local_data()

        # Statistics
        self.train_loss = float('inf')
        self.test_accuracy = 0.0
        self.gradients_computed = 0

    def _generate_local_data(self) -> Tuple[DataLoader, DataLoader]:
        """Generate synthetic MNIST-like data for testing"""
        # Train data (1000 samples)
        n_train = 1000
        X_train = torch.randn(n_train, 784)

        # Create clustered data for more realistic learning
        for i in range(10):  # 10 classes
            class_samples = slice(i*100, (i+1)*100)
            X_train[class_samples] += torch.randn(1, 784) * 0.5

        y_train = torch.repeat_interleave(torch.arange(10), 100)

        # Add noise
        X_train += torch.randn_like(X_train) * 0.1

        # Test data (200 samples)
        n_test = 200
        X_test = torch.randn(n_test, 784)
        for i in range(10):
            class_samples = slice(i*20, (i+1)*20)
            X_test[class_samples] += torch.randn(1, 784) * 0.5
        y_test = torch.repeat_interleave(torch.arange(10), 20)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def compute_gradient(self, use_full_batch: bool = False) -> np.ndarray:
        """
        Compute real gradient through backpropagation

        Args:
            use_full_batch: If True, compute gradient over full epoch.
                          If False, use single batch (faster)

        Returns:
            Flat numpy array of gradients
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        batch_count = 0

        # Compute gradients
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Flatten images for SimpleNN (expects 784-dim input)
            data = data.view(data.size(0), -1)

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Accumulate gradients
            loss.backward()
            total_loss += loss.item()
            batch_count += 1

            # Use single batch unless full_batch specified
            if not use_full_batch:
                break

        # Extract gradients as numpy array
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.cpu().numpy().flatten())
            else:
                gradients.append(np.zeros_like(param.data.cpu().numpy().flatten()))

        flat_gradients = np.concatenate(gradients)

        # Update stats
        self.train_loss = total_loss / batch_count
        self.gradients_computed += 1

        return flat_gradients

    def apply_gradient_update(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """
        Apply gradient update to model parameters

        Args:
            gradient: Flat numpy array of gradients
            learning_rate: Learning rate for update
        """
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                param_size = param.numel()
                param_grad = gradient[idx:idx+param_size].reshape(param.shape)
                param.data -= learning_rate * torch.from_numpy(param_grad).float().to(self.device)
                idx += param_size

    def evaluate(self) -> float:
        """
        Evaluate model accuracy on test set

        Returns:
            Test accuracy (0.0-1.0)
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        self.test_accuracy = correct / total
        return self.test_accuracy

    def validate_gradient(self, gradient: np.ndarray) -> Dict[str, float]:
        """
        Validate gradient quality (for Proof of Quality Gradient)

        Creates temporary model, applies gradient, evaluates performance.
        Used by validators to score gradient quality.

        Args:
            gradient: Gradient to validate

        Returns:
            Dict with validation metrics:
                - test_loss: Loss after applying gradient
                - test_accuracy: Accuracy after applying gradient
                - gradient_norm: L2 norm of gradient
                - sparsity: Fraction of near-zero elements
        """
        # Create temporary model
        temp_model = SimpleNN().to(self.device)
        temp_model.load_state_dict(self.model.state_dict())

        # Apply gradient
        idx = 0
        with torch.no_grad():
            for param in temp_model.parameters():
                param_size = param.numel()
                param_grad = gradient[idx:idx+param_size].reshape(param.shape)
                param.data -= 0.01 * torch.from_numpy(param_grad).float().to(self.device)
                idx += param_size

        # Evaluate
        temp_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = temp_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return {
            'test_loss': total_loss / len(self.test_loader),
            'test_accuracy': correct / total,
            'gradient_norm': np.linalg.norm(gradient),
            'sparsity': np.mean(np.abs(gradient) < 1e-6)
        }


class Trainer:
    """
    High-level training interface for federated learning

    Manages training configuration and orchestrates local training.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Optional[DataLoader] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32
    ):
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train_local(self, epochs: int = 1) -> np.ndarray:
        """
        Train model locally for specified epochs

        Args:
            epochs: Number of local training epochs

        Returns:
            Gradient after local training
        """
        # Implementation depends on dataset format
        # For now, returns placeholder
        raise NotImplementedError("Custom dataset training not yet implemented")


# Training configuration dataclass
class TrainingConfig:
    """Configuration for federated training"""
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        aggregation: str = "krum"
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.aggregation = aggregation
