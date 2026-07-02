#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Exercise the full edge-proof → committee → storage/shim pipeline."""

import asyncio
import os
import subprocess
import time
from datetime import datetime

import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "0TML" / "src"))

from zerotrustml.experimental.edge_validation import CommitteeVote, aggregate_committee_votes
from zerotrustml.modular_architecture import HolochainStorage, GradientMetadata
from zerotrustml.backends.holochain_backend import HolochainBackend
from zerotrustml.backends.ethereum_backend import EthereumBackend, BackendConnectionError

SCRIPT_DIR = Path(__file__).resolve().parent
GANACHE_CANDIDATES = [
    SCRIPT_DIR / "../0TML/.npm/bin/ganache",
    SCRIPT_DIR / "../.npm/bin/ganache",
]


def start_ganache():
    ganache_path = None
    for candidate in GANACHE_CANDIDATES:
        if candidate.exists():
            ganache_path = candidate.resolve()
            break
    if ganache_path is None:
        print("Ganache not installed; skipping Ethereum attestation (npm install ganache --prefix .npm).")
        return None
    proc = subprocess.Popen([str(ganache_path), "--chain.chainId", "1337", "--wallet.totalAccounts", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)
    return proc


def generate_votes(node_id: int, score: float) -> list[CommitteeVote]:
    score = float(max(0.0, min(1.0, score)))
    votes = []
    for validator in range(5):
        accepted = score >= 0.5
        votes.append(
            CommitteeVote(
                validator_id=f"validator-{validator}",
                proof_hash=f"proof-{node_id}",
                quality_score=score if accepted else score * 0.5,
                accepted=accepted,
                timestamp=datetime.utcnow().isoformat(),
                signature=None,
            )
        )
    return votes


async def main():
    gradient = np.random.randn(100).astype(np.float32)
    edge_proof = {
        "loss_before": 1.0,
        "loss_after": 0.8,
        "gradient_hash": "hash123",
        "timestamp": datetime.utcnow().isoformat(),
    }

    votes = generate_votes(1, 0.8)
    consensus_score, accepted = aggregate_committee_votes(votes)
    print(f"Committee consensus: {consensus_score:.3f}, accepted={accepted}")

    storage = HolochainStorage()
    metadata = GradientMetadata(
        node_id=1,
        round_num=1,
        timestamp=datetime.utcnow(),
        reputation_score=0.9,
        validation_passed=True,
        pogq_score=0.85,
        edge_proof=edge_proof,
        committee_votes=[vote.__dict__ for vote in votes],
    )
    gradient_id = await storage.store_gradient(gradient, metadata)
    print(f"Stored gradient id (stub): {gradient_id}")

    try:
        backend = HolochainBackend()
        await backend.store_gradient({
            "id": "grad-1",
            "node_id": 1,
            "round_num": 1,
            "gradient_hash": "hash123",
            "pogq_score": 0.85,
            "zkpoc_verified": True,
            "timestamp": datetime.utcnow().timestamp(),
            "edge_proof": edge_proof,
            "committee_votes": [vote.__dict__ for vote in votes],
        })
        print("Holochain backend call succeeded")
    except Exception as e:
        print(f"Holochain backend unavailable: {e}")

    proc = start_ganache()
    try:
        backend_eth = EthereumBackend()
        try:
            await backend_eth.connect()
            await backend_eth.record_committee_attestation("hash123", consensus_score, {"round": 1})
            await backend_eth.disconnect()
        except BackendConnectionError as e:
            print(f"Ethereum backend unavailable: {e}")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
