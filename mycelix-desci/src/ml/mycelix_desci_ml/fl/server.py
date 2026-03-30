# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Federated Learning Server

Server implementation for coordinating federated learning.
"""

from typing import List, Dict
import torch
import torch.nn as nn


class FederatedServer:
    """Federated learning server

    Coordinates training across multiple clients and aggregates updates.
    """

    def __init__(self, model: nn.Module):
        """Initialize federated server

        Args:
            model: Global PyTorch model
        """
        self.model = model
        self.round = 0

    def get_parameters(self) -> dict:
        """Get current global model parameters

        Returns:
            Dictionary of parameter values
        """
        parameters = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                parameters[name] = param.cpu().detach().numpy()

        return parameters

    def aggregate_parameters(
        self,
        client_parameters: List[dict],
        client_weights: List[float],
    ):
        """Aggregate parameters from multiple clients

        Args:
            client_parameters: List of parameter dictionaries from clients
            client_weights: Weight for each client (e.g., based on dataset size)
        """
        if not client_parameters:
            raise ValueError("No client parameters to aggregate")

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Aggregate parameters
        aggregated = {}
        for name in client_parameters[0].keys():
            aggregated[name] = sum(
                weight * params[name]
                for weight, params in zip(normalized_weights, client_parameters)
            )

        # Update global model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated:
                    param.copy_(torch.tensor(aggregated[name]))

        self.round += 1

    def evaluate(self, test_loader) -> dict:
        """Evaluate global model

        Args:
            test_loader: DataLoader for test data

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            "loss": total_loss / len(test_loader),
            "accuracy": 100.0 * correct / total,
            "round": self.round,
        }

        return metrics
