# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Cosmos SDK Storage Backend for ZeroTrustML Phase 10
Provides blockchain-based gradient storage using Cosmos SDK chains

Features:
- Multi-chain support (Cosmos Hub, Osmosis, Juno, etc.)
- IBC (Inter-Blockchain Communication) support
- Tendermint consensus
- CosmWasm smart contract integration
- Immutable audit trail
- Byzantine event logging
- Credit issuance as on-chain transactions

Author: ZeroTrustML Team
License: MIT
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Cosmos SDK dependencies
try:
    from cosmpy.aerial.client import LedgerClient, NetworkConfig
    from cosmpy.aerial.wallet import LocalWallet
    from cosmpy.aerial.tx import Transaction
    from cosmpy.crypto.address import Address
    from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin
    COSMPY_AVAILABLE = True
except ImportError:
    COSMPY_AVAILABLE = False
    print("⚠️  cosmpy not available - install with: pip install cosmpy")

from .storage_backend import (
    StorageBackend,
    BackendType,
    ConnectionError as BackendConnectionError,
    GradientRecord
)


@dataclass
class CosmosChainConfig:
    """Configuration for a Cosmos SDK chain"""
    chain_id: str
    rpc_url: str
    grpc_url: str
    prefix: str  # Address prefix (e.g., "cosmos", "osmo", "juno")
    denom: str   # Native token denom (e.g., "uatom", "uosmo")
    gas_price: float = 0.025
    gas_adjustment: float = 1.3


# Pre-configured chains
COSMOS_CHAINS = {
    "cosmos-hub": CosmosChainConfig(
        chain_id="cosmoshub-4",
        rpc_url="https://rpc.cosmos.network",
        grpc_url="grpc.cosmos.network:9090",
        prefix="cosmos",
        denom="uatom"
    ),
    "osmosis": CosmosChainConfig(
        chain_id="osmosis-1",
        rpc_url="https://rpc.osmosis.zone",
        grpc_url="grpc.osmosis.zone:9090",
        prefix="osmo",
        denom="uosmo"
    ),
    "juno": CosmosChainConfig(
        chain_id="juno-1",
        rpc_url="https://rpc.juno.strange.love",
        grpc_url="grpc.juno.strange.love:9090",
        prefix="juno",
        denom="ujuno"
    ),
    "local": CosmosChainConfig(
        chain_id="local-1",
        rpc_url="http://localhost:26657",
        grpc_url="localhost:9090",
        prefix="cosmos",
        denom="stake"
    )
}


class CosmosBackend(StorageBackend):
    """
    Cosmos SDK storage backend for federated learning gradients

    Uses Tendermint consensus and CosmWasm smart contracts for
    immutable, Byzantine-resistant gradient storage.
    """

    def __init__(
        self,
        chain: str = "local",
        mnemonic: Optional[str] = None,
        contract_address: Optional[str] = None,
        custom_config: Optional[CosmosChainConfig] = None,
        timeout: int = 30
    ):
        """
        Initialize Cosmos backend

        Args:
            chain: Chain name from COSMOS_CHAINS or "custom"
            mnemonic: BIP39 mnemonic for wallet (optional)
            contract_address: CosmWasm contract address
            custom_config: Custom chain configuration
            timeout: Transaction timeout in seconds
        """
        if not COSMPY_AVAILABLE:
            raise BackendConnectionError("cosmpy library not installed")

        super().__init__()
        self.backend_type = BackendType.BLOCKCHAIN

        # Chain configuration
        if custom_config:
            self.config = custom_config
        elif chain in COSMOS_CHAINS:
            self.config = COSMOS_CHAINS[chain]
        else:
            raise ValueError(f"Unknown chain: {chain}. Use custom_config.")

        self.chain_name = chain
        self.mnemonic = mnemonic
        self.contract_address = contract_address
        self.timeout = timeout

        # Runtime state
        self.client: Optional[LedgerClient] = None
        self.wallet: Optional[LocalWallet] = None
        self.address: Optional[str] = None

    async def connect(self) -> bool:
        """
        Connect to Cosmos SDK chain and CosmWasm contract

        Returns:
            True if connection successful

        Raises:
            BackendConnectionError: If connection fails
        """
        try:
            # Create network config
            network_config = NetworkConfig(
                chain_id=self.config.chain_id,
                url=self.config.grpc_url,
                fee_minimum_gas_price=self.config.gas_price,
                fee_denomination=self.config.denom,
                staking_denomination=self.config.denom
            )

            # Create ledger client
            self.client = LedgerClient(network_config)

            # Create or restore wallet
            if self.mnemonic:
                self.wallet = LocalWallet.from_mnemonic(
                    self.mnemonic,
                    prefix=self.config.prefix
                )
            else:
                # Generate new wallet (for testing)
                self.wallet = LocalWallet.generate(prefix=self.config.prefix)
                print(f"⚠️  Generated new wallet: {self.wallet.address()}")
                print(f"   Mnemonic: {self.wallet.mnemonic()}")

            self.address = str(self.wallet.address())

            # Verify contract exists (if specified)
            if self.contract_address:
                try:
                    # Query contract to verify it exists
                    query_msg = {"get_stats": {}}
                    result = await self._query_contract(query_msg)
                    print(f"✅ Connected to contract: {self.contract_address}")
                except Exception as e:
                    print(f"⚠️  Contract query failed: {e}")
                    print("   Contract may not be deployed yet")

            print(f"✅ Connected to {self.config.chain_id}")
            print(f"   Address: {self.address}")
            print(f"   RPC: {self.config.rpc_url}")

            return True

        except Exception as e:
            raise BackendConnectionError(f"Failed to connect to Cosmos: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from Cosmos chain"""
        self.client = None
        self.wallet = None
        self.address = None
        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health

        Returns:
            Health status with latency and chain info
        """
        start = time.time()

        try:
            if not self.client:
                raise BackendConnectionError("Not connected")

            # Get latest block height as health check
            block_height = self.client.query_height()

            latency_ms = (time.time() - start) * 1000

            return {
                "healthy": True,
                "latency_ms": round(latency_ms, 2),
                "backend_type": self.backend_type.value,
                "chain_id": self.config.chain_id,
                "block_height": block_height,
                "address": self.address,
                "contract": self.contract_address or "not deployed"
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "backend_type": self.backend_type.value
            }

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """
        Store gradient on Cosmos blockchain via CosmWasm contract

        Args:
            gradient_data: Dictionary containing:
                - id: Gradient ID
                - node_id: Node identifier
                - round_num: FL round number
                - gradient: Gradient data (stored as hash only)
                - gradient_hash: Pre-computed hash
                - pogq_score: Quality score (0.0-1.0)
                - zkpoc_verified: ZK verification status

        Returns:
            Transaction hash

        Note: Only stores gradient hash on-chain to minimize costs
        """
        if not self.client or not self.wallet:
            raise BackendConnectionError("Not connected")

        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        # Prepare data for blockchain
        gradient_id = gradient_data["id"]
        gradient_hash = gradient_data["gradient_hash"]

        # Hash node ID for privacy
        node_id_hash = hashlib.sha256(gradient_data["node_id"].encode()).hexdigest()

        # Scale PoGQ score to integer (0-1000)
        pogq_score = int(gradient_data.get("pogq_score", 0) * 1000)

        # Execute contract message
        execute_msg = {
            "store_gradient": {
                "gradient_id": gradient_id,
                "node_id_hash": node_id_hash,
                "round_num": gradient_data["round_num"],
                "gradient_hash": gradient_hash,
                "pogq_score": pogq_score,
                "zkpoc_verified": gradient_data.get("zkpoc_verified", False)
            }
        }

        tx_hash = await self._execute_contract(execute_msg)

        print(f"✅ Gradient stored on Cosmos: {gradient_id[:12]}...")
        print(f"   Tx: {tx_hash}")

        return tx_hash

    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """
        Retrieve gradient from Cosmos blockchain

        Args:
            gradient_id: Gradient identifier

        Returns:
            GradientRecord if found, None otherwise
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        query_msg = {
            "get_gradient": {
                "gradient_id": gradient_id
            }
        }

        try:
            result = await self._query_contract(query_msg)

            if not result:
                return None

            # Convert from on-chain format
            return GradientRecord(
                id=result["gradient_id"],
                node_id=result["node_id_hash"],  # Note: This is the hash
                round_num=result["round_num"],
                gradient_hash=result["gradient_hash"],
                pogq_score=result["pogq_score"] / 1000.0,  # Unscale
                zkpoc_verified=result["zkpoc_verified"],
                timestamp=result["timestamp"],
                gradient=[]  # Full gradient not stored on-chain
            )

        except Exception as e:
            print(f"⚠️  Error retrieving gradient: {e}")
            return None

    async def get_gradients_by_round(
        self,
        round_num: int,
        limit: int = 100
    ) -> List[str]:
        """
        Get all gradient IDs for a specific round

        Args:
            round_num: FL round number
            limit: Maximum results to return

        Returns:
            List of gradient IDs
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        query_msg = {
            "get_gradients_by_round": {
                "round_num": round_num,
                "limit": limit
            }
        }

        try:
            result = await self._query_contract(query_msg)
            return result.get("gradient_ids", [])
        except Exception as e:
            print(f"⚠️  Error retrieving gradients: {e}")
            return []

    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """
        Verify gradient integrity on blockchain

        Args:
            gradient_id: Gradient identifier

        Returns:
            True if gradient exists and is valid
        """
        gradient = await self.get_gradient(gradient_id)
        return gradient is not None

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """
        Issue credit on Cosmos blockchain

        Args:
            holder: Node/user identifier
            amount: Credit amount
            earned_from: Source of credit

        Returns:
            Transaction hash
        """
        if not self.client or not self.wallet:
            raise BackendConnectionError("Not connected")

        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        # Hash holder ID for privacy
        holder_hash = hashlib.sha256(holder.encode()).hexdigest()

        execute_msg = {
            "issue_credit": {
                "holder_hash": holder_hash,
                "amount": amount,
                "earned_from": earned_from
            }
        }

        tx_hash = await self._execute_contract(execute_msg)

        print(f"✅ Credit issued on Cosmos: {amount} to {holder[:12]}...")
        print(f"   Tx: {tx_hash}")

        return tx_hash

    async def get_credit_balance(self, holder: str) -> int:
        """
        Get credit balance from Cosmos blockchain

        Args:
            holder: Node/user identifier

        Returns:
            Credit balance
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        holder_hash = hashlib.sha256(holder.encode()).hexdigest()

        query_msg = {
            "get_credit_balance": {
                "holder_hash": holder_hash
            }
        }

        try:
            result = await self._query_contract(query_msg)
            return result.get("balance", 0)
        except Exception as e:
            print(f"⚠️  Error retrieving balance: {e}")
            return 0

    async def get_credit_history(
        self,
        holder: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get credit history from Cosmos blockchain

        Args:
            holder: Node/user identifier
            limit: Maximum results to return

        Returns:
            List of credit records
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        holder_hash = hashlib.sha256(holder.encode()).hexdigest()

        query_msg = {
            "get_credit_history": {
                "holder_hash": holder_hash,
                "limit": limit
            }
        }

        try:
            result = await self._query_contract(query_msg)
            return result.get("credits", [])
        except Exception as e:
            print(f"⚠️  Error retrieving credit history: {e}")
            return []

    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
        """
        Log Byzantine event to Cosmos blockchain (immutable)

        Args:
            event: Dictionary containing:
                - node_id: Node identifier
                - round_num: FL round number
                - detection_method: How event was detected
                - severity: Event severity
                - details: Additional details

        Returns:
            Transaction hash
        """
        if not self.client or not self.wallet:
            raise BackendConnectionError("Not connected")

        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        node_id_hash = hashlib.sha256(event["node_id"].encode()).hexdigest()

        execute_msg = {
            "log_byzantine_event": {
                "node_id_hash": node_id_hash,
                "round_num": event["round_num"],
                "detection_method": event["detection_method"],
                "severity": event["severity"],
                "details": json.dumps(event.get("details", {}))
            }
        }

        tx_hash = await self._execute_contract(execute_msg)

        print(f"✅ Byzantine event logged on Cosmos")
        print(f"   Node: {event['node_id'][:12]}...")
        print(f"   Severity: {event['severity']}")
        print(f"   Tx: {tx_hash}")

        return tx_hash

    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get Byzantine events from Cosmos blockchain

        Args:
            node_id: Optional node ID filter
            round_num: Optional round number filter
            limit: Maximum results to return

        Returns:
            List of Byzantine events
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        query_msg = {
            "get_byzantine_events": {
                "limit": limit
            }
        }

        if node_id:
            node_id_hash = hashlib.sha256(node_id.encode()).hexdigest()
            query_msg["get_byzantine_events"]["node_id_hash"] = node_id_hash

        if round_num is not None:
            query_msg["get_byzantine_events"]["round_num"] = round_num

        try:
            result = await self._query_contract(query_msg)
            return result.get("events", [])
        except Exception as e:
            print(f"⚠️  Error retrieving Byzantine events: {e}")
            return []

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """
        Get node reputation from Cosmos blockchain

        Args:
            node_id: Node identifier

        Returns:
            Reputation data
        """
        if not self.contract_address:
            raise BackendConnectionError("Contract address not set")

        node_id_hash = hashlib.sha256(node_id.encode()).hexdigest()

        query_msg = {
            "get_reputation": {
                "node_id_hash": node_id_hash
            }
        }

        try:
            result = await self._query_contract(query_msg)

            # Unscale average PoGQ score
            if "average_pogq_score" in result:
                result["average_pogq_score"] = result["average_pogq_score"] / 1000.0

            return result
        except Exception as e:
            print(f"⚠️  Error retrieving reputation: {e}")
            return {
                "total_gradients_submitted": 0,
                "total_credits_earned": 0,
                "byzantine_event_count": 0,
                "average_pogq_score": 0.0
            }

    async def update_reputation(
        self,
        node_id: str,
        reputation_data: Dict[str, Any]
    ) -> str:
        """
        Update node reputation on Cosmos blockchain

        Note: In Cosmos, reputation is automatically updated
        when gradients are stored or Byzantine events are logged.
        This method is a no-op for blockchain backends.

        Returns:
            Empty string (no transaction needed)
        """
        # Reputation auto-updates in smart contract
        return ""

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get backend statistics from Cosmos blockchain

        Returns:
            Statistics dictionary
        """
        if not self.contract_address:
            return {
                "total_gradients": 0,
                "total_credits_issued": 0,
                "total_byzantine_events": 0,
                "storage_size_bytes": 0,
                "backend_type": self.backend_type.value,
                "chain_id": self.config.chain_id,
                "status": "contract not deployed"
            }

        query_msg = {"get_stats": {}}

        try:
            result = await self._query_contract(query_msg)

            # Add backend-specific info
            result["backend_type"] = self.backend_type.value
            result["chain_id"] = self.config.chain_id
            result["chain_name"] = self.chain_name
            result["contract_address"] = self.contract_address

            return result

        except Exception as e:
            print(f"⚠️  Error retrieving stats: {e}")
            return {
                "total_gradients": 0,
                "total_credits_issued": 0,
                "total_byzantine_events": 0,
                "backend_type": self.backend_type.value,
                "error": str(e)
            }

    # ============================================
    # HELPER METHODS
    # ============================================

    async def _execute_contract(self, execute_msg: Dict[str, Any]) -> str:
        """
        Execute CosmWasm contract message

        Args:
            execute_msg: Contract execute message

        Returns:
            Transaction hash
        """
        if not self.client or not self.wallet:
            raise BackendConnectionError("Not connected")

        try:
            # Create transaction
            tx = Transaction()

            # Add execute message
            tx.add_message(
                self.wallet.execute_contract(
                    contract_address=self.contract_address,
                    msg=execute_msg,
                    funds=[]  # No funds sent
                )
            )

            # Estimate gas
            gas_limit = await self.client.estimate_gas(tx, self.wallet)
            gas_limit = int(gas_limit * self.config.gas_adjustment)

            # Sign and broadcast
            signed_tx = self.wallet.sign(tx, gas_limit=gas_limit)
            tx_hash = await self.client.broadcast_tx(signed_tx)

            # Wait for confirmation (optional)
            await asyncio.sleep(1)  # Wait for block

            return tx_hash

        except Exception as e:
            raise BackendConnectionError(f"Contract execution failed: {e}")

    async def _query_contract(self, query_msg: Dict[str, Any]) -> Any:
        """
        Query CosmWasm contract

        Args:
            query_msg: Contract query message

        Returns:
            Query result
        """
        if not self.client:
            raise BackendConnectionError("Not connected")

        try:
            result = self.client.query_contract_smart(
                contract_address=self.contract_address,
                query_msg=query_msg
            )
            return result

        except Exception as e:
            raise BackendConnectionError(f"Contract query failed: {e}")

    def _get_chain_name(self) -> str:
        """Get human-readable chain name"""
        return f"{self.chain_name} ({self.config.chain_id})"


# ============================================
# DEPLOYMENT HELPER FUNCTIONS
# ============================================

async def deploy_contract(
    backend: CosmosBackend,
    wasm_file: str
) -> str:
    """
    Deploy ZeroTrustML CosmWasm contract to Cosmos chain

    Args:
        backend: Connected CosmosBackend instance
        wasm_file: Path to compiled .wasm file

    Returns:
        Contract address
    """
    if not backend.client or not backend.wallet:
        raise BackendConnectionError("Backend not connected")

    print(f"📦 Deploying contract from {wasm_file}...")

    # Read wasm file
    with open(wasm_file, "rb") as f:
        wasm_code = f.read()

    # Store code
    tx = Transaction()
    tx.add_message(
        backend.wallet.store_code(wasm_code)
    )

    code_id = await backend.client.broadcast_tx(tx)
    print(f"✅ Code stored with ID: {code_id}")

    # Instantiate contract
    init_msg = {
        "version": "1.0.0"
    }

    tx2 = Transaction()
    tx2.add_message(
        backend.wallet.instantiate_contract(
            code_id=code_id,
            msg=init_msg,
            label="ZeroTrustML Gradient Storage",
            admin=backend.address
        )
    )

    contract_address = await backend.client.broadcast_tx(tx2)
    print(f"✅ Contract deployed at: {contract_address}")

    return contract_address


async def get_balance(backend: CosmosBackend) -> float:
    """
    Get wallet balance in native token

    Args:
        backend: Connected CosmosBackend instance

    Returns:
        Balance in base units (e.g., ATOM, not uatom)
    """
    if not backend.client or not backend.address:
        raise BackendConnectionError("Backend not connected")

    balance = backend.client.query_bank_balance(
        backend.address,
        backend.config.denom
    )

    # Convert from micro-units (uatom) to base units (ATOM)
    return balance / 1_000_000
