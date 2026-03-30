#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Integration test for EthereumBackend using local Ganache."""

import asyncio
import os
import subprocess
import sys
import time

import pytest

from zerotrustml.backends.ethereum_backend import EthereumBackend

GANACHE_BIN = os.path.join(os.path.dirname(__file__), "../.npm/bin/ganache")
GANACHE_BIN = os.path.abspath(GANACHE_BIN)


@pytest.mark.asyncio
async def test_ethereum_backend_connects_to_ganache():
    if not os.path.exists(GANACHE_BIN):
        pytest.skip("Ganache binary not found; install with npm install ganache --prefix .npm")

    proc = subprocess.Popen(
        [
            GANACHE_BIN,
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

    try:
        # Give Ganache time to start
        time.sleep(2)

        backend = EthereumBackend(
            rpc_url="http://127.0.0.1:8545",
            chain_id=1337,
            contract_address=None,
            private_key=None,
        )

        connected = await backend.connect()
        assert connected is True

        result = await backend.record_committee_attestation(
            proof_hash="proof-hash",
            consensus_score=0.8,
            metadata={"round": 1},
        )
        assert result is None  # No contract loaded, but call should succeed

        await backend.disconnect()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
