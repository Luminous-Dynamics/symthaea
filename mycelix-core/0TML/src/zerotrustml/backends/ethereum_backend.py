#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Ethereum Backend Implementation

Blockchain-based storage using Ethereum and EVM-compatible chains (Polygon, Arbitrum, etc.).
Provides immutable audit trail and smart contract-based storage.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import time
import hashlib
import json
import os

try:
    from web3 import Web3
    from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
except ImportError:
    print("Warning: web3 not installed. Run: pip install web3")
    Web3 = None

from .storage_backend import (
    StorageBackend,
    BackendType,
    GradientRecord,
    CreditRecord,
    ByzantineEvent,
    ConnectionError as BackendConnectionError,
    IntegrityError,
    NotFoundError
)

logger = logging.getLogger(__name__)


class EthereumBackend(StorageBackend):
    """
    Ethereum backend for blockchain-based FL deployments

    Features:
    - Immutable audit trail (blockchain)
    - Smart contract-based storage
    - Multi-chain support (Ethereum, Polygon, Arbitrum, Optimism, BSC)
    - Transparent and verifiable
    - Decentralized consensus

    Use cases:
    - Public blockchain FL networks
    - Token-based incentive systems
    - Maximum transparency requirements
    - Regulatory compliance (immutable records)
    """

    def __init__(
        self,
        rpc_url: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        chain_id: int = 1337,  # Local ganache default
        gas_price: Optional[int] = None,
        timeout: int = 30
    ):
        super().__init__(BackendType.BLOCKCHAIN)
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.private_key = private_key
        self.chain_id = chain_id
        self.gas_price = gas_price
        self.timeout = timeout

        self.w3 = None
        self.contract = None
        self.account = None
        self.connection_time = None

    async def connect(self) -> bool:
        """Connect to Ethereum node and smart contract"""
        if not Web3:
            raise BackendConnectionError("web3 required: pip install web3")

        try:
            # Connect to RPC endpoint
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={'timeout': self.timeout}))

            # Add PoA middleware for networks like Polygon (not needed for Sepolia)
            if self.chain_id in [80001, 80002, 137]:  # Mumbai (deprecated), Amoy testnet, Polygon mainnet
                self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            # Sepolia (11155111) uses PoS and does not require PoA middleware

            # Check connection
            if not self.w3.is_connected():
                raise BackendConnectionError(f"Failed to connect to {self.rpc_url}")

            # Set up account from private key
            if self.private_key:
                if not self.private_key.startswith('0x'):
                    self.private_key = '0x' + self.private_key
                self.account = self.w3.eth.account.from_key(self.private_key)
            else:
                # Use first account from node (for development)
                accounts = self.w3.eth.accounts
                if accounts:
                    self.account = accounts[0]
                else:
                    logger.warning("No account available - read-only mode")

            # Load smart contract if address provided
            if self.contract_address:
                # Load ABI from compiled contract
                abi = self._get_contract_abi()
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(self.contract_address),
                    abi=abi
                )

            self.connected = True
            self.connection_time = time.time()

            chain_name = self._get_chain_name(self.chain_id)
            logger.info(f"✅ Ethereum backend connected: {chain_name} ({self.rpc_url})")
            if self.account:
                logger.info(f"   Account: {self.account if isinstance(self.account, str) else self.account.address}")
            if self.contract:
                logger.info(f"   Contract: {self.contract_address}")

            return True

        except Exception as e:
            logger.error(f"Ethereum connection failed: {e}")
            raise BackendConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """Close connection (no persistent connection for HTTP)"""
        self.connected = False
        logger.info("Ethereum backend closed")

    async def health_check(self) -> Dict[str, Any]:
        """Check Ethereum node health"""
        if not self.connected or not self.w3:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": "Not connected"}
            }

        try:
            start = time.time()
            block_number = self.w3.eth.block_number
            latency = (time.time() - start) * 1000

            return {
                "healthy": True,
                "latency_ms": latency,
                "storage_available": True,
                "metadata": {
                    "rpc_url": self.rpc_url,
                    "chain_id": self.chain_id,
                    "chain_name": self._get_chain_name(self.chain_id),
                    "block_number": block_number,
                    "contract_address": self.contract_address,
                    "account": self.account.address if hasattr(self.account, 'address') else str(self.account)
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": str(e)}
            }

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """Store gradient on blockchain via smart contract"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        # Prepare data for blockchain (store hash only to save gas)
        gradient_hash = gradient_data["gradient_hash"]
        node_id_hash = Web3.keccak(text=gradient_data["node_id"])  # bytes32 (not .hex())

        # Build transaction
        try:
            # Get account address
            account_address = self.account.address if hasattr(self.account, 'address') else self.account

            # Call smart contract function with positional args
            # Signature: storeGradient(string,bytes32,uint256,string,uint256,bool)
            contract_function = self.contract.functions.storeGradient(
                gradient_data["id"],                                          # string gradient_id
                node_id_hash,                                                 # bytes32 node_id_hash
                gradient_data["round_num"],                                   # uint256 round_num
                gradient_hash,                                                # string gradient_hash
                int(gradient_data.get("pogq_score", 0) * 1000),             # uint256 pogq_score
                gradient_data.get("zkpoc_verified", False)                   # bool zkpoc_verified
            )

            # Estimate gas for the transaction
            try:
                estimated_gas = contract_function.estimate_gas({'from': account_address})
                gas_limit = int(estimated_gas * 1.5)  # Add 50% buffer
                logger.debug(f"Estimated gas: {estimated_gas}, using: {gas_limit}")
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}, using default 500000")
                gas_limit = 500000  # Increased default

            tx = contract_function.build_transaction({
                'from': account_address,
                'nonce': self.w3.eth.get_transaction_count(account_address),
                'gas': gas_limit,
                'gasPrice': self.gas_price or self.w3.eth.gas_price,
                'chainId': self.chain_id
            })

            # Sign and send transaction
            if hasattr(self.account, 'sign_transaction'):
                signed_tx = self.account.sign_transaction(tx)
                # web3.py v7.x changed rawTransaction to raw_transaction
                raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            else:
                # For accounts managed by node
                tx_hash = self.w3.eth.send_transaction(tx)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)

            if receipt['status'] != 1:
                raise IntegrityError(f"Transaction failed: {receipt}")

            logger.debug(f"Ethereum: Gradient stored {gradient_data['id'][:8]}... (tx: {tx_hash.hex()[:10]}...)")
            return gradient_data["id"]

        except Exception as e:
            logger.error(f"Failed to store gradient on blockchain: {e}")
            raise

    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """Retrieve gradient from blockchain"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            # Call smart contract view function
            result = self.contract.functions.getGradient(gradient_id).call()

            if not result or result[0] == '':  # Empty result
                return None

            # Parse result (depends on smart contract structure)
            return GradientRecord(
                id=gradient_id,
                node_id=result[1],  # Node ID hash (not recoverable)
                round_num=result[2],
                gradient=[],  # Not stored on-chain (too expensive)
                gradient_hash=result[3],
                pogq_score=result[4] / 1000.0 if result[4] else None,
                zkpoc_verified=result[5],
                reputation_score=0.0,  # Not stored on-chain
                timestamp=result[6],
                backend_metadata={
                    "tx_hash": result[7] if len(result) > 7 else None,
                    "block_number": result[8] if len(result) > 8 else None
                }
            )

        except Exception as e:
            logger.error(f"Failed to get gradient from blockchain: {e}")
            return None

    async def get_gradients_by_round(self, round_num: int) -> List[GradientRecord]:
        """Get all gradients for a round from blockchain"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            # Query smart contract for gradients by round
            gradient_ids = self.contract.functions.getGradientsByRound(round_num).call()

            gradients = []
            for gradient_id in gradient_ids:
                gradient = await self.get_gradient(gradient_id)
                if gradient:
                    gradients.append(gradient)

            return gradients

        except Exception as e:
            logger.error(f"Failed to get gradients by round from blockchain: {e}")
            return []

    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """
        Verify gradient integrity on blockchain

        For blockchain: Data is immutable by design.
        Just verify the entry exists and matches.
        """
        gradient = await self.get_gradient(gradient_id)
        return gradient is not None

    async def record_committee_attestation(
        self,
        proof_hash: str,
        consensus_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist committee verdict metadata on-chain (best effort).

        If the deployed contract exposes a `recordProofAttestation` function
        we call it; otherwise we log the attestation locally so the caller can
        decide how to handle persistence.
        """

        if not self.contract:
            logger.info(
                "Ethereum backend (no contract): Committee attestation %s score %.3f %s",
                proof_hash,
                consensus_score,
                metadata or {},
            )
            return None

        function = None
        payload = metadata or {}
        args = ()

        if hasattr(self.contract.functions, "recordProofAttestation"):
            function = self.contract.functions.recordProofAttestation
            args = (
                proof_hash,
                int(consensus_score * 1000),
                json.dumps(payload),
            )
        elif hasattr(self.contract.functions, "recordAttestation"):
            function = self.contract.functions.recordAttestation
            args = (
                proof_hash,
                int(consensus_score * 1000),
            )
        else:
            logger.info(
                "Smart contract missing attestation entrypoint; logging locally: %s",
                proof_hash,
            )
            return None

        try:
            account_address = self.account.address if hasattr(self.account, 'address') else self.account
            contract_function = function(*args)

            gas_limit = int(contract_function.estimate_gas({'from': account_address}) * 1.5)

            tx = contract_function.build_transaction({
                'from': account_address,
                'nonce': self.w3.eth.get_transaction_count(account_address),
                'gas': gas_limit,
                'gasPrice': self.gas_price or self.w3.eth.gas_price,
                'chainId': self.chain_id,
            })

            if hasattr(self.account, 'sign_transaction'):
                signed_tx = self.account.sign_transaction(tx)
                raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            else:
                tx_hash = self.w3.eth.send_transaction(tx)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)
            if receipt['status'] != 1:
                raise IntegrityError(f"Attestation transaction failed: {receipt}")

            logger.debug("Ethereum: Committee attestation recorded (%s)", tx_hash.hex())
            return tx_hash.hex()

        except Exception as e:
            logger.error("Failed to record committee attestation: %s", e)
            return None

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue credit on blockchain (as token or record)"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            holder_hash = Web3.keccak(text=holder)  # bytes32 (not .hex())
            account_address = self.account.address if hasattr(self.account, 'address') else self.account

            # Call smart contract function with positional args
            # Signature: issueCredit(bytes32,uint256,string)
            contract_function = self.contract.functions.issueCredit(
                holder_hash,    # bytes32 holder_hash
                amount,         # uint256 amount
                earned_from     # string earned_from
            )

            # Estimate gas for the transaction
            try:
                estimated_gas = contract_function.estimate_gas({'from': account_address})
                gas_limit = int(estimated_gas * 1.5)  # Add 50% buffer
                logger.debug(f"Estimated gas: {estimated_gas}, using: {gas_limit}")
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}, using default 300000")
                gas_limit = 300000  # Increased default

            tx = contract_function.build_transaction({
                'from': account_address,
                'nonce': self.w3.eth.get_transaction_count(account_address),
                'gas': gas_limit,
                'gasPrice': self.gas_price or self.w3.eth.gas_price,
                'chainId': self.chain_id
            })

            # Sign and send
            if hasattr(self.account, 'sign_transaction'):
                signed_tx = self.account.sign_transaction(tx)
                # web3.py v7.x changed rawTransaction to raw_transaction
                raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            else:
                tx_hash = self.w3.eth.send_transaction(tx)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)

            if receipt['status'] != 1:
                raise IntegrityError(f"Credit issuance failed: {receipt}")

            logger.debug(f"Ethereum: Credit issued (tx: {tx_hash.hex()[:10]}...) ({amount} to {holder})")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Failed to issue credit on blockchain: {e}")
            raise

    async def get_credit_balance(self, node_id: str) -> int:
        """Get credit balance from blockchain"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            node_id_hash = Web3.keccak(text=node_id).hex()
            balance = self.contract.functions.getCreditBalance(node_id_hash).call()
            return int(balance)

        except Exception as e:
            logger.error(f"Failed to get credit balance from blockchain: {e}")
            return 0

    async def get_credit_history(self, node_id: str) -> List[CreditRecord]:
        """Get credit history from blockchain"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            node_id_hash = Web3.keccak(text=node_id).hex()
            credit_txs = self.contract.functions.getCreditHistory(node_id_hash).call()

            credits = []
            for tx_data in credit_txs:
                credits.append(CreditRecord(
                    transaction_id=tx_data[0],
                    holder=node_id,  # Original node_id (hash not reversible)
                    amount=tx_data[1],
                    earned_from=tx_data[2],
                    timestamp=tx_data[3]
                ))

            return credits

        except Exception as e:
            logger.error(f"Failed to get credit history from blockchain: {e}")
            return []

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get reputation from blockchain"""
        if not self.contract:
            # Return default if no contract
            return {
                "node_id": node_id,
                "score": 0.5,
                "gradients_submitted": 0,
                "gradients_accepted": 0,
                "byzantine_events": 0,
                "total_credits": 0
            }

        try:
            node_id_hash = Web3.keccak(text=node_id).hex()
            rep_data = self.contract.functions.getReputation(node_id_hash).call()

            return {
                "node_id": node_id,
                "score": rep_data[0] / 1000.0,  # Unscale from int
                "gradients_submitted": rep_data[1],
                "gradients_accepted": rep_data[2],
                "byzantine_events": rep_data[3],
                "total_credits": rep_data[4]
            }

        except Exception as e:
            logger.error(f"Failed to get reputation from blockchain: {e}")
            return {
                "node_id": node_id,
                "score": 0.5,
                "gradients_submitted": 0,
                "gradients_accepted": 0,
                "byzantine_events": 0,
                "total_credits": 0
            }

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ) -> None:
        """Update reputation on blockchain"""
        logger.debug(f"Reputation update for {node_id}: {score_delta:+.3f} ({reason})")
        # Blockchain reputation is calculated from on-chain data
        # No explicit update needed (immutable record)

    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
        """Log Byzantine event to blockchain"""
        if not self.contract:
            raise BackendConnectionError("Smart contract not loaded")

        try:
            # Convert node_id to bytes32 hash (no .hex() - keep as HexBytes for Web3.py)
            node_id_hash = Web3.keccak(text=event["node_id"])
            account_address = self.account.address if hasattr(self.account, 'address') else self.account

            # IMPORTANT: Use positional args matching Solidity function signature
            # function logByzantineEvent(bytes32, uint256, string, string, string)
            contract_function = self.contract.functions.logByzantineEvent(
                node_id_hash,  # bytes32 nodeIdHash
                event["round_num"],  # uint256 roundNum
                event["detection_method"],  # string detectionMethod
                event["severity"],  # string severity
                json.dumps(event.get("details", {}))  # string details
            )

            # Estimate gas for the transaction
            try:
                estimated_gas = contract_function.estimate_gas({'from': account_address})
                gas_limit = int(estimated_gas * 1.5)  # Add 50% buffer
                logger.debug(f"Byzantine event gas - Estimated: {estimated_gas}, using: {gas_limit}")
            except Exception as e:
                logger.warning(f"Byzantine event gas estimation failed: {e}, using default 500000")
                gas_limit = 500000  # Increased default

            tx = contract_function.build_transaction({
                'from': account_address,
                'nonce': self.w3.eth.get_transaction_count(account_address),
                'gas': gas_limit,
                'gasPrice': self.gas_price or self.w3.eth.gas_price,
                'chainId': self.chain_id
            })

            # Sign and send
            if hasattr(self.account, 'sign_transaction'):
                signed_tx = self.account.sign_transaction(tx)
                # web3.py v7.x changed rawTransaction to raw_transaction
                raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            else:
                tx_hash = self.w3.eth.send_transaction(tx)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)

            if receipt['status'] != 1:
                raise IntegrityError(f"Byzantine event logging failed: {receipt}")

            logger.info(f"Ethereum: Byzantine event logged (tx: {tx_hash.hex()[:10]}...)")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Failed to log Byzantine event on blockchain: {e}")
            raise

    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[ByzantineEvent]:
        """Get Byzantine events from blockchain"""
        if not self.contract:
            return []

        try:
            # Query events from smart contract
            if node_id and round_num is not None:
                node_id_hash = Web3.keccak(text=node_id).hex()
                events_data = self.contract.functions.getByzantineEvents(
                    node_id_hash, round_num
                ).call()
            elif node_id:
                node_id_hash = Web3.keccak(text=node_id).hex()
                events_data = self.contract.functions.getByzantineEventsByNode(node_id_hash).call()
            elif round_num is not None:
                events_data = self.contract.functions.getByzantineEventsByRound(round_num).call()
            else:
                events_data = self.contract.functions.getAllByzantineEvents().call()

            events = []
            for event_data in events_data:
                events.append(ByzantineEvent(
                    event_id=event_data[0],
                    node_id=node_id or "unknown",  # Hash not reversible
                    round_num=event_data[1],
                    detection_method=event_data[2],
                    severity=event_data[3],
                    details=json.loads(event_data[4]) if event_data[4] else {},
                    timestamp=event_data[5]
                ))

            return events

        except Exception as e:
            logger.error(f"Failed to get Byzantine events from blockchain: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get Ethereum backend statistics"""
        try:
            block_number = self.w3.eth.block_number if self.w3 else 0
            uptime = time.time() - self.connection_time if self.connection_time else 0

            stats = {
                "backend_type": "ethereum",
                "total_gradients": 0,
                "total_credits_issued": 0,
                "total_byzantine_events": 0,
                "storage_size_bytes": 0,  # N/A for blockchain
                "uptime_seconds": uptime,
                "metadata": {
                    "rpc_url": self.rpc_url,
                    "chain_id": self.chain_id,
                    "chain_name": self._get_chain_name(self.chain_id),
                    "block_number": block_number,
                    "contract_address": self.contract_address
                }
            }

            # Get on-chain stats if contract available
            if self.contract:
                try:
                    contract_stats = self.contract.functions.getStats().call()
                    stats["total_gradients"] = contract_stats[0]
                    stats["total_credits_issued"] = contract_stats[1]
                    stats["total_byzantine_events"] = contract_stats[2]
                except:
                    pass

            return stats

        except Exception as e:
            logger.warning(f"Could not get stats from Ethereum: {e}")
            return {
                "backend_type": "ethereum",
                "total_gradients": 0,
                "total_credits_issued": 0,
                "total_byzantine_events": 0,
                "storage_size_bytes": 0,
                "uptime_seconds": 0,
                "metadata": {"error": str(e)}
            }

    # Helper methods

    def _get_chain_name(self, chain_id: int) -> str:
        """Get human-readable chain name"""
        chains = {
            1: "Ethereum Mainnet",
            5: "Goerli Testnet (deprecated)",
            11155111: "Sepolia Testnet",
            137: "Polygon Mainnet",
            80001: "Polygon Mumbai Testnet (deprecated)",
            80002: "Polygon Amoy Testnet",
            42161: "Arbitrum One",
            421614: "Arbitrum Sepolia",
            421613: "Arbitrum Goerli (deprecated)",
            10: "Optimism",
            11155420: "Optimism Sepolia",
            420: "Optimism Goerli (deprecated)",
            56: "BSC Mainnet",
            97: "BSC Testnet",
            1337: "Local Ganache",
            31337: "Local Anvil"
        }
        return chains.get(chain_id, f"Chain {chain_id}")

    def _get_contract_abi(self) -> List[Dict]:
        """
        Get smart contract ABI from generated artifacts

        Returns the complete ABI for ZeroTrustMLGradientStorage contract
        including all functions and events.
        """
        import sys
        from pathlib import Path

        # Add build directory to path to import generated ABI
        build_dir = Path(__file__).parent.parent.parent.parent / "build"
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))

        from contract_abi import CONTRACT_ABI
        return CONTRACT_ABI
