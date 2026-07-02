#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Smoke tests for the high-level EdgeClient SDK."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zerotrustml.experimental.edge_client import EdgeClient


def _build_linear_model(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def test_edge_client_generates_and_packages_proof(tmp_path):
    torch.manual_seed(123)

    input_dim = 6
    num_classes = 2
    data_x = torch.randn(64, input_dim)
    data_y = torch.randint(0, num_classes, (64,))
    loader = DataLoader(TensorDataset(data_x, data_y), batch_size=16, shuffle=True)

    model = _build_linear_model(input_dim, num_classes)
    loss_fn = nn.CrossEntropyLoss()

    client = EdgeClient(
        node_id="node-1",
        model=model,
        loss_fn=loss_fn,
        local_data=loader,
        reputation_store_path=str(tmp_path / "edge_reputation.db"),
    )

    # Align client model with a "global" snapshot
    client.update_model_state(model.state_dict())

    # Produce a synthetic gradient from one mini-batch
    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x
    batch_y = batch_y

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)
    loss.backward()

    gradients = {name: param.grad.clone() for name, param in model.named_parameters()}

    outcome = client.generate_proof(gradients, round_number=1)
    assert outcome.proof.node_id == "node-1"
    assert outcome.quality_score >= 0.0
    assert 0.0 <= outcome.reputation <= 1.0

    payload = client.package_submission(gradients, round_number=1)
    assert payload["node_id"] == "node-1"
    assert payload["round_number"] == 1
    assert "proof" in payload
