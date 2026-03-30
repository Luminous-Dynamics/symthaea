# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FLTrust baseline adapter.

Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning via
Trust Bootstrapping", NDSS 2021.

This implementation reuses the FedAvg client training loop but replaces the
server aggregation step with the FLTrust weighted aggregation based on cosine
similarity to a trusted server gradient.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from baselines.fedavg import FedAvgConfig, FedAvgServer, FedAvgClient


@dataclass
class FLTrustConfig(FedAvgConfig):
    """FedAvg config with FLTrust extras."""

    val_batch_size: int = 256
    max_val_batches: int = 1


class FLTrustServer(FedAvgServer):
    """
    FedAvg server with FLTrust aggregation.

    Aggregation steps:
      1. Compute trusted gradient on the server validation loader
      2. Flatten each client update and compute cosine similarity to trusted grad
      3. ReLU the cosine (trust >= 0) and normalize
      4. Weighted average client weights using trust scores
    """

    def __init__(
        self,
        model: nn.Module,
        config: FLTrustConfig,
        trusted_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ):
        super().__init__(model, config, device)
        self.trusted_loader = trusted_loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_val_batches = max(1, config.max_val_batches)
        self.last_trust_scores: List[float] = []

    def _flatten_weights(self, weights: List[np.ndarray]) -> torch.Tensor:
        flat = np.concatenate([layer.reshape(-1) for layer in weights])
        return torch.tensor(flat, device=self.device, dtype=torch.float32)

    def _server_gradient(self) -> torch.Tensor:
        self.model.train(False)
        self.model.zero_grad(set_to_none=True)

        batches = 0
        for data, target in self.trusted_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
            batches += 1
            if batches >= self.max_val_batches:
                break

        grad_vector = parameters_to_vector([
            (param.grad if param.grad is not None else torch.zeros_like(param))
            for param in self.model.parameters()
        ])
        self.model.zero_grad(set_to_none=True)
        return grad_vector

    def aggregate(self, client_updates: List[dict]) -> List[np.ndarray]:
        server_grad = self._server_gradient()
        grad_norm = server_grad.norm() + 1e-12

        trusts = []
        client_vectors = []
        for update in client_updates:
            vec = self._flatten_weights(update["weights"])
            client_vectors.append(vec)
            cosine = torch.dot(vec, server_grad) / (vec.norm() * grad_norm + 1e-12)
            trusts.append(max(0.0, cosine.item()))

        trust_sum = sum(trusts)
        if trust_sum <= 1e-12:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            weights = [t / trust_sum for t in trusts]

        num_layers = len(client_updates[0]["weights"])
        aggregated_weights = [
            np.zeros_like(client_updates[0]["weights"][layer_idx])
            for layer_idx in range(num_layers)
        ]

        for weight, update in zip(weights, client_updates):
            for idx, layer in enumerate(update["weights"]):
                aggregated_weights[idx] += weight * layer

        self.last_trust_scores = trusts
        return aggregated_weights


def create_fltrust_experiment(
    model_fn: callable,
    train_data_splits: List[torch.utils.data.DataLoader],
    test_dataset,
    config: FLTrustConfig,
    device: str = "cpu",
) -> Tuple[FLTrustServer, List[FedAvgClient], torch.utils.data.DataLoader]:
    """
    Create FLTrust server, clients, and test loader.
    """
    trusted_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
    )

    server_model = model_fn()
    server = FLTrustServer(server_model, config, trusted_loader, device=device)

    clients: List[FedAvgClient] = []
    for idx, loader in enumerate(train_data_splits):
        client_model = model_fn()
        clients.append(
            FedAvgClient(
                client_id=f"client_{idx}",
                model=client_model,
                train_data=loader,
                config=config,
                device=device,
            )
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return server, clients, test_loader
