#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Deploy ProofLog.sol to Anvil and record an attestation."""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

try:
    from solcx import compile_source, install_solc
    HAS_SOLCX = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SOLCX = False

from web3 import Web3

from zerotrustml.backends.ethereum_backend import EthereumBackend

ANVIL_BIN = "anvil"  # expected on PATH (via foundry)
PROOFLOG_SRC = Path(__file__).resolve().parents[1] / "contracts" / "ProofLog.sol"

@pytest.mark.asyncio
async def test_polygon_attestation_anvil():
    if not HAS_SOLCX:
        pytest.skip("py-solc-x (solcx) not installed; install via poetry add py-solc-x")

    try:
        proc = subprocess.Popen([ANVIL_BIN], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        pytest.skip("Anvil (foundry) not installed; run nix-shell -p foundry --run anvil")

    try:
        time.sleep(2)
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        assert w3.is_connected()
        account = w3.eth.accounts[0]

        install_solc("0.8.20")
        source = PROOFLOG_SRC.read_text()
        compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.20")
        _, contract_interface = compiled.popitem()
        prooflog = w3.eth.contract(abi=contract_interface["abi"], bytecode=contract_interface["bin"])

        tx_hash = prooflog.constructor().transact({"from": account})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress

        backend = EthereumBackend(contract_address=contract_address, chain_id=31337)  # anvil default
        await backend.connect()
        tx_hash = await backend.record_committee_attestation("integration-proof", 0.75, {"round": 42})
        await backend.disconnect()

        assert tx_hash is not None, "Expected a transaction hash from attestation write"

        # Verify contract state reflects the attestation
        prooflog_runtime = w3.eth.contract(
            address=contract_address,
            abi=contract_interface["abi"],
        )
        attestations = prooflog_runtime.functions.getAttestations("integration-proof").call()
        assert len(attestations) == 1, f"Expected one attestation entry, saw {len(attestations)}"

        proof_hash, consensus_score, timestamp, submitter = attestations[0]
        assert proof_hash == "integration-proof"
        assert consensus_score == 750  # scaled by 1000 inside backend
        assert timestamp > 0
        expected_submitter = backend.account.address if hasattr(backend.account, "address") else backend.account
        assert submitter == expected_submitter
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
