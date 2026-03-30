#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Exercise the full edge-proof → committee → storage pipeline locally."""

import asyncio
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zerotrustml.experimental.edge_client import EdgeClient
from zerotrustml.experimental.edge_validation import CommitteeVote, aggregate_committee_votes
from zerotrustml.modular_architecture import HolochainStorage, GradientMetadata
from zerotrustml.backends.holochain_backend import HolochainBackend
from zerotrustml.backends.ethereum_backend import EthereumBackend


def start_ganache_if_available():
    bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.npm/bin/ganache"))
    if not os.path.exists(bin_path):
        print("Ganache not found; skipping Ethereum attestation.")
        return None
    proc = subprocess.Popen(
        [
            bin_path,
            "--chain.chainId",
            "1337",
            "--wallet.totalAccounts",
            "1",
            "--miner.blockTime",
            "0.5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    return proc


async def main():
    # Step 1: simulate client gradient and proof using the new SDK
    torch.manual_seed(7)
    input_dim = 12
    num_classes = 2
    dataset = TensorDataset(
        torch.randn(256, input_dim),
        torch.randint(0, num_classes, (256,)),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
    loss_fn = nn.CrossEntropyLoss()

    client = EdgeClient(
        node_id="node-1",
        model=model,
        loss_fn=loss_fn,
        local_data=loader,
        reputation_store_path="edge_reputation_demo.db",
    )
    client.update_model_state(model.state_dict())

    batch_x, batch_y = next(iter(loader))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)
    loss.backward()

    gradient_tensors = {name: param.grad.clone() for name, param in model.named_parameters()}
    outcome = client.generate_proof(gradient_tensors, round_number=1)
    edge_proof_payload = outcome.proof.to_dict()
    gradient = np.concatenate([
        tensor.detach().cpu().numpy().ravel() for tensor in gradient_tensors.values()
    ])

    # Step 2: committee votes
    votes = [
        CommitteeVote(
            validator_id=f"validator-{i}",
            proof_hash=edge_proof_payload["gradient_hash"],
            quality_score=0.8,
            accepted=True,
            timestamp=datetime.utcnow().isoformat(),
            signature=None,
        )
        for i in range(5)
    ]
    consensus_score, accepted = aggregate_committee_votes(votes)
    print(f"Committee consensus: {consensus_score:.3f}, accepted={accepted}")

    # Step 3: store via HolochainStorage stub
    storage = HolochainStorage()
    metadata = GradientMetadata(
        node_id=1,
        round_num=1,
        timestamp=datetime.utcnow(),
        reputation_score=0.9,
        validation_passed=True,
        pogq_score=outcome.quality_score,
        edge_proof=edge_proof_payload,
        committee_votes=[vote.__dict__ for vote in votes],
    )
    gradient_id = await storage.store_gradient(gradient, metadata)
    print(f"Stored gradient id: {gradient_id}")

    # Step 4: optionally store to real Holochain backend
    try:
        backend = HolochainBackend()
        await backend.store_gradient({
            "id": "grad-1",
            "node_id": 1,
            "round_num": 1,
            "gradient_hash": edge_proof_payload["gradient_hash"],
            "pogq_score": 0.85,
            "zkpoc_verified": True,
            "timestamp": datetime.utcnow().timestamp(),
            "edge_proof": edge_proof_payload,
            "committee_votes": [vote.__dict__ for vote in votes],
        })
        print("Holochain backend call succeeded")
    except Exception as e:
        print(f"Holochain backend unavailable: {e}")

    # Step 5: Ethereum attestation (via Ganache)
    proc = start_ganache_if_available()
    try:
        backend_eth = EthereumBackend()
        await backend_eth.connect()
        await backend_eth.record_committee_attestation(
            edge_proof_payload["gradient_hash"],
            consensus_score,
            {"round": 1, "quality": outcome.quality_score},
        )
        await backend_eth.disconnect()
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
