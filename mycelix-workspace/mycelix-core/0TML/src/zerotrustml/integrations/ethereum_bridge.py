# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Ethereum Bridge for ZeroTrustML

Connects the ZeroTrustML federated learning system to Ethereum smart contracts
for on-chain reputation anchoring and payment routing.

Usage:
    from zerotrustml.integrations.ethereum_bridge import EthereumBridge

    bridge = EthereumBridge(
        rpc_url="https://ethereum-sepolia-rpc.publicnode.com",
        private_key=os.environ.get("PRIVATE_KEY"),
    )

    # Anchor reputation to chain
    await bridge.anchor_reputation(node_id, score, proof_hash)

    # Create payment for FL contribution
    await bridge.create_contribution_payment(recipient, amount_eth)
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

# Contract addresses on Sepolia
SEPOLIA_CONTRACTS = {
    "registry": "0x556b810371e3d8D9E5753117514F03cC6C93b835",
    "reputation": "0xf3B343888a9b82274cEfaa15921252DB6c5f48C9",
    "payment": "0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB",
}

# Simplified ABIs
REPUTATION_ABI = [
    {
        "inputs": [
            {"name": "subject", "type": "address"},
            {"name": "contextHash", "type": "bytes32"},
            {"name": "score", "type": "int256"},
            {"name": "proof", "type": "bytes"},
        ],
        "name": "submitReputation",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "root", "type": "bytes32"},
            {"name": "epoch", "type": "uint256"},
        ],
        "name": "anchorMerkleRoot",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "subject", "type": "address"}],
        "name": "getReputation",
        "outputs": [
            {"name": "score", "type": "int256"},
            {"name": "submissions", "type": "uint256"},
            {"name": "lastUpdate", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

PAYMENT_ABI = [
    {
        "inputs": [
            {"name": "recipient", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "releaseTime", "type": "uint256"},
            {"name": "conditionHash", "type": "bytes32"},
        ],
        "name": "createPayment",
        "outputs": [{"name": "paymentId", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [{"name": "paymentId", "type": "uint256"}],
        "name": "releasePayment",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class ReputationScore:
    """On-chain reputation data"""
    score: int
    submissions: int
    last_update: datetime


@dataclass
class TransactionResult:
    """Result of a blockchain transaction"""
    success: bool
    tx_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    error: Optional[str] = None


class EthereumBridge:
    """
    Bridge between ZeroTrustML and Ethereum smart contracts.

    Provides methods for:
    - Anchoring PoGQ reputation scores on-chain
    - Creating escrow payments for FL contributions
    - Querying on-chain reputation data
    """

    def __init__(
        self,
        rpc_url: str = "https://ethereum-sepolia-rpc.publicnode.com",
        private_key: Optional[str] = None,
        contracts: Optional[Dict[str, str]] = None,
    ):
        if not HAS_WEB3:
            raise ImportError(
                "web3 package required for Ethereum integration. "
                "Install with: pip install web3"
            )

        self.rpc_url = rpc_url
        self.contracts = contracts or SEPOLIA_CONTRACTS

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        # Set up account if private key provided
        self.account = None
        if private_key:
            if not private_key.startswith("0x"):
                private_key = f"0x{private_key}"
            self.account = Account.from_key(private_key)

        # Initialize contracts
        self.reputation_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contracts["reputation"]),
            abi=REPUTATION_ABI,
        )
        self.payment_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contracts["payment"]),
            abi=PAYMENT_ABI,
        )

    @property
    def address(self) -> Optional[str]:
        """Get the connected wallet address"""
        return self.account.address if self.account else None

    async def get_reputation(self, address: str) -> ReputationScore:
        """
        Query on-chain reputation for an address.

        Args:
            address: Ethereum address to query

        Returns:
            ReputationScore with score, submission count, and last update time
        """
        address = Web3.to_checksum_address(address)
        result = self.reputation_contract.functions.getReputation(address).call()

        return ReputationScore(
            score=result[0],
            submissions=result[1],
            last_update=datetime.fromtimestamp(result[2]) if result[2] > 0 else None,
        )

    async def submit_reputation(
        self,
        subject: str,
        score: int,
        context_hash: bytes = b"\x00" * 32,
        proof: bytes = b"",
    ) -> TransactionResult:
        """
        Submit a reputation score on-chain.

        Args:
            subject: Address being rated
            score: Reputation score (-100 to 100)
            context_hash: Context identifier (32 bytes)
            proof: Optional proof data

        Returns:
            TransactionResult with tx hash and status
        """
        if not self.account:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error="No private key configured",
            )

        try:
            subject = Web3.to_checksum_address(subject)

            # Build transaction
            tx = self.reputation_contract.functions.submitReputation(
                subject,
                context_hash,
                score,
                proof,
            ).build_transaction({
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return TransactionResult(
                success=receipt["status"] == 1,
                tx_hash=tx_hash.hex(),
                block_number=receipt["blockNumber"],
                gas_used=receipt["gasUsed"],
            )

        except Exception as e:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error=str(e),
            )

    async def anchor_merkle_root(
        self,
        root: bytes,
        epoch: int,
    ) -> TransactionResult:
        """
        Anchor a Merkle root for batch reputation updates.

        Args:
            root: 32-byte Merkle root hash
            epoch: Epoch number for this batch

        Returns:
            TransactionResult with tx hash and status
        """
        if not self.account:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error="No private key configured",
            )

        try:
            tx = self.reputation_contract.functions.anchorMerkleRoot(
                root,
                epoch,
            ).build_transaction({
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return TransactionResult(
                success=receipt["status"] == 1,
                tx_hash=tx_hash.hex(),
                block_number=receipt["blockNumber"],
                gas_used=receipt["gasUsed"],
            )

        except Exception as e:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error=str(e),
            )

    async def create_contribution_payment(
        self,
        recipient: str,
        amount_wei: int,
        release_delay_seconds: int = 3600,
    ) -> TransactionResult:
        """
        Create an escrow payment for an FL contribution.

        Args:
            recipient: Address to receive payment
            amount_wei: Payment amount in wei
            release_delay_seconds: Time until payment can be released

        Returns:
            TransactionResult with tx hash and payment ID
        """
        if not self.account:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error="No private key configured",
            )

        try:
            recipient = Web3.to_checksum_address(recipient)
            release_time = int(datetime.now().timestamp()) + release_delay_seconds

            tx = self.payment_contract.functions.createPayment(
                recipient,
                amount_wei,
                release_time,
                b"\x00" * 32,  # No condition hash
            ).build_transaction({
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "value": amount_wei,
            })

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return TransactionResult(
                success=receipt["status"] == 1,
                tx_hash=tx_hash.hex(),
                block_number=receipt["blockNumber"],
                gas_used=receipt["gasUsed"],
            )

        except Exception as e:
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error=str(e),
            )

    def get_balance(self) -> int:
        """Get the wallet balance in wei"""
        if not self.account:
            return 0
        return self.w3.eth.get_balance(self.account.address)

    def get_balance_eth(self) -> float:
        """Get the wallet balance in ETH"""
        return self.w3.from_wei(self.get_balance(), "ether")


# Convenience function for quick integration
def create_bridge(
    private_key: Optional[str] = None,
    network: str = "sepolia",
) -> EthereumBridge:
    """
    Create an Ethereum bridge instance.

    Args:
        private_key: Wallet private key (or set PRIVATE_KEY env var)
        network: Network name (currently only 'sepolia' supported)

    Returns:
        Configured EthereumBridge instance
    """
    if private_key is None:
        private_key = os.environ.get("PRIVATE_KEY")

    rpc_urls = {
        "sepolia": "https://ethereum-sepolia-rpc.publicnode.com",
        "local": "http://127.0.0.1:8545",
    }

    return EthereumBridge(
        rpc_url=rpc_urls.get(network, rpc_urls["sepolia"]),
        private_key=private_key,
    )
