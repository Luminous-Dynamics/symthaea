# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Federated Learning Client

Client implementation for participating in federated learning.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FederatedClient:
    """Federated learning client

    Trains models locally and shares gradient updates with the federation.
    """

    def __init__(
        self,
        model: nn.Module,
        client_id: str,
        device: str = "cpu",
    ):
        """Initialize federated client

        Args:
            model: PyTorch model to train
            client_id: Unique identifier for this client
            device: Device for training ("cpu" or "cuda")
        """
        self.model = model
        self.client_id = client_id
        self.device = device
        self.model.to(device)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 0.01,
        criterion: Optional[nn.Module] = None,
    ) -> dict:
        """Train the model locally

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Training metrics (loss, accuracy)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": 100.0 * correct / total,
            "samples": total,
        }

        return metrics

    def get_gradients(self) -> dict:
        """Get current model gradients

        Returns:
            Dictionary of parameter gradients
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().detach().numpy()

        return gradients

    def set_parameters(self, parameters: dict):
        """Set model parameters

        Args:
            parameters: Dictionary of parameter values
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(torch.tensor(parameters[name]))
