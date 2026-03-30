# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Defense Adapter: Compatibility Layer for Runner
================================================

Bridges the gap between:
- Old runner.py architecture (expects create_*_experiment functions)
- New defense registry (provides defense classes with aggregate() method)

This adapter wraps new defenses in the old experiment creation pattern,
allowing the runner to work without major refactoring.

Author: Luminous Dynamics
Date: November 8, 2025
"""

import torch
import torch.nn as nn
from typing import List, Callable, Any, Dict, Tuple
from dataclasses import dataclass
import numpy as np

# Import new defense registry
from src.defenses import get_defense, DEFENSE_REGISTRY


@dataclass
class BaseConfig:
    """Base configuration for all defenses"""
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 32
    num_clients: int = 10
    fraction_clients: float = 1.0


class GenericClient:
    """Generic FL client that works with any defense"""

    def __init__(self, client_id: int, data_loader, model_fn, config: BaseConfig, device):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model_fn().to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train(self, global_model: nn.Module) -> Dict[str, Any]:
        """Train locally and return update"""
        # Load global model
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(data)
                total_samples += len(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute gradient (difference from global model)
        gradient = []
        for local_param, global_param in zip(
            self.model.parameters(),
            global_model.parameters()
        ):
            gradient.append((global_param.data - local_param.data).cpu().numpy().flatten())

        gradient = np.concatenate(gradient)

        return {
            'gradient': gradient,
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples,
            'num_samples': total_samples
        }


class GenericServer:
    """Generic FL server that uses defense registry"""

    def __init__(
        self,
        model_fn: Callable,
        defense_name: str,
        defense_config: Dict,
        num_clients: int,
        fraction_clients: float,
        device
    ):
        self.model = model_fn().to(device)
        self.defense_name = defense_name
        self.defense = get_defense(defense_name, **defense_config)
        self.num_clients = num_clients
        self.fraction_clients = fraction_clients
        self.device = device
        self.round_num = 0

    def train_round(self, clients: List[GenericClient]) -> Dict[str, Any]:
        """Execute one training round"""
        # Select clients
        num_selected = max(1, int(self.num_clients * self.fraction_clients))
        selected_indices = np.random.choice(
            len(clients),
            size=min(num_selected, len(clients)),
            replace=False
        )
        selected_clients = [clients[i] for i in selected_indices]

        # Collect updates
        client_updates = []
        train_loss = 0.0
        train_accuracy = 0.0

        for client in selected_clients:
            result = client.train(self.model)
            client_updates.append(result['gradient'])
            train_loss += result['loss']
            train_accuracy += result['accuracy']

        # Aggregate using defense
        if len(client_updates) > 0:
            aggregated = self.defense.aggregate(
                client_updates,
                round_num=self.round_num,
                global_model=self.model,
                device=self.device
            )

            # Apply update to model
            self._apply_gradient(aggregated)

        self.round_num += 1

        return {
            'train_loss': train_loss / len(selected_clients),
            'train_accuracy': train_accuracy / len(selected_clients),
            'num_clients': len(selected_clients)
        }

    def _apply_gradient(self, gradient: np.ndarray):
        """Apply aggregated gradient to model"""
        offset = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_gradient = gradient[offset:offset+numel].reshape(param.shape)
                param.data -= torch.from_numpy(param_gradient).to(self.device)
                offset += numel


def evaluate_global_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

    return {
        'test_loss': test_loss / total,
        'test_accuracy': correct / total
    }


def create_generic_experiment(
    defense_name: str,
    model_fn: Callable,
    client_loaders: List,
    test_loader,
    config: BaseConfig,
    defense_config: Dict,
    device
) -> Tuple[GenericServer, List[GenericClient], Any]:
    """
    Generic experiment creator that works with any defense from registry

    Args:
        defense_name: Name from DEFENSE_REGISTRY
        model_fn: Function that creates model
        client_loaders: List of client DataLoaders
        test_loader: Test DataLoader
        config: Base configuration
        defense_config: Defense-specific configuration
        device: torch device

    Returns:
        (server, clients, test_loader) tuple
    """
    # Create clients
    clients = []
    for i, loader in enumerate(client_loaders):
        client = GenericClient(
            client_id=i,
            data_loader=loader,
            model_fn=model_fn,
            config=config,
            device=device
        )
        clients.append(client)

    # Create server with defense
    server = GenericServer(
        model_fn=model_fn,
        defense_name=defense_name,
        defense_config=defense_config,
        num_clients=config.num_clients,
        fraction_clients=config.fraction_clients,
        device=device
    )

    return server, clients, test_loader


# Specific experiment creators for backward compatibility
def create_fedavg_experiment(model_fn, client_loaders, test_loader, config, device):
    """FedAvg experiment"""
    return create_generic_experiment(
        defense_name="fedavg",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )


def create_coord_median_experiment(model_fn, client_loaders, test_loader, config, device):
    """Coordinate Median experiment"""
    return create_generic_experiment(
        defense_name="coord_median",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )


def create_rfa_experiment(model_fn, client_loaders, test_loader, config, device):
    """RFA experiment"""
    return create_generic_experiment(
        defense_name="rfa",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )


def create_fltrust_experiment(model_fn, client_loaders, test_dataset, config, device):
    """FLTrust experiment"""
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return create_generic_experiment(
        defense_name="fltrust",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )


def create_cbf_experiment(model_fn, client_loaders, test_loader, config, device):
    """CBF experiment"""
    return create_generic_experiment(
        defense_name="cbf",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={"conformal_alpha": getattr(config, 'conformal_alpha', 0.10)},
        device=device
    )


def create_pogq_v4_1_experiment(model_fn, client_loaders, test_loader, config, device):
    """PoGQ-v4.1 experiment"""
    return create_generic_experiment(
        defense_name="pogq_v4.1",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={
            "conformal_alpha": getattr(config, 'conformal_alpha', 0.10),
            "ema_beta": getattr(config, 'ema_beta', 0.85),
            "warmup_rounds": getattr(config, 'warmup_rounds', 3),
            "hysteresis_k": getattr(config, 'hysteresis_k', 2),
            "hysteresis_m": getattr(config, 'hysteresis_m', 3),
        },
        device=device
    )


def create_boba_experiment(model_fn, client_loaders, test_loader, config, device):
    """BOBA experiment"""
    return create_generic_experiment(
        defense_name="boba",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )


def create_coord_median_safe_experiment(model_fn, client_loaders, test_loader, config, device):
    """Coordinate Median Safe experiment"""
    return create_generic_experiment(
        defense_name="coord_median_safe",
        model_fn=model_fn,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=config,
        defense_config={},
        device=device
    )
