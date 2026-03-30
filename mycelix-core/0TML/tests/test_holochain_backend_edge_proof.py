#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for Holochain backend edge-proof payload handling."""

import json
import pytest
from datetime import datetime

from zerotrustml.backends.holochain_backend import HolochainBackend
from zerotrustml.backends.ethereum_backend import EthereumBackend


@pytest.mark.asyncio
async def test_store_gradient_includes_proof_json(monkeypatch):
    backend = HolochainBackend()

    captured = {}

    async def fake_call_zome(zome_name, fn_name, payload):
        captured["zome_name"] = zome_name
        captured["fn_name"] = fn_name
        captured["payload"] = payload
        return {"entry_hash": "hash123"}

    monkeypatch.setattr(backend, "_call_zome", fake_call_zome)

    edge_proof = {"loss_before": 1.0, "loss_after": 0.8}
    committee_votes = [
        {"validator_id": "validator-0", "quality_score": 0.8, "accepted": True},
        {"validator_id": "validator-1", "quality_score": 0.75, "accepted": True},
    ]

    await backend.store_gradient({
        "id": "grad-1",
        "node_id": 1,
        "round_num": 1,
        "gradient_hash": "deadbeef",
        "pogq_score": 0.9,
        "zkpoc_verified": True,
        "timestamp": datetime.utcnow().timestamp(),
        "edge_proof": edge_proof,
        "committee_votes": committee_votes,
    })

    payload = captured["payload"]
    assert payload["node_id"] == 1
    assert json.loads(payload["edge_proof"])["loss_after"] == 0.8
    votes = json.loads(payload["committee_votes"])
    assert votes[0]["validator_id"] == "validator-0"


@pytest.mark.asyncio
async def test_ethereum_backend_attestation_without_contract(caplog):
    backend = EthereumBackend()
    backend.contract = None
    caplog.set_level("INFO")
    result = await backend.record_committee_attestation("hash123", 0.75, {"round": 1})
    assert result is None
    assert "Committee attestation" in caplog.text
