#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Client for ZeroTrustML Phase 10

Real WebSocket-based client for Holochain conductor operations.
Connects to Holochain conductor and calls zome functions for immutable audit trail.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import base64

# Unified logging and error handling
from zerotrustml.logging import (
    get_logger,
    async_log_operation,
    async_timed,
    async_correlation_context,
)
from zerotrustml.exceptions import (
    HolochainConnectionError,
    HolochainZomeError,
    ConfigurationError,
    ErrorCode,
)
from zerotrustml.error_handling import (
    async_retry,
    CircuitBreaker,
    Fallback,
)

try:
    import websockets
except ImportError:
    websockets = None

try:
    import msgpack
except ImportError:
    msgpack = None

logger = get_logger(__name__)

# Circuit breaker for Holochain operations
_holochain_breaker = CircuitBreaker(
    name="holochain_client",
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=2
)


@dataclass
class HolochainConfig:
    """Configuration for Holochain client"""
    admin_url: str = "ws://localhost:8888"
    app_url: str = "ws://localhost:8889"
    app_id: str = "zerotrustml"
    cell_id: Optional[str] = None
    timeout: int = 30


class HolochainClient:
    """
    Real Holochain client for Phase 10 immutable audit trail

    Connects to Holochain conductor via WebSocket and calls zome functions:
    - gradient_storage: Store and retrieve gradients
    - zerotrustml_credits: Issue credits and check balances
    - reputation_tracker: Update and query reputation
    """

    def __init__(
        self,
        admin_url: str = "ws://localhost:8888",
        app_url: str = "ws://localhost:8889",
        app_id: str = "zerotrustml",
        cell_id: Optional[str] = None
    ):
        self.config = HolochainConfig(
            admin_url=admin_url,
            app_url=app_url,
            app_id=app_id,
            cell_id=cell_id
        )
        self.admin_ws = None
        self.app_ws = None
        self.request_id = 0

    @async_timed(level=20, message="Holochain conductor connection established")
    async def connect(self):
        """Connect to Holochain conductor"""
        if not websockets:
            raise ConfigurationError(
                message="websockets library required for Holochain: pip install websockets",
                config_key="dependencies.websockets"
            )

        async with async_log_operation(
            "holochain_connect",
            admin_url=self.config.admin_url,
            app_url=self.config.app_url
        ):
            try:
                # Connect to admin interface with required Origin header
                logger.info("Connecting to Holochain admin interface", extra={
                    'admin_url': self.config.admin_url
                })
                self.admin_ws = await websockets.connect(
                    self.config.admin_url,
                    additional_headers={"Origin": "http://localhost"},
                    ping_interval=20,
                    ping_timeout=10
                )

                # Connect to app interface with required Origin header
                logger.info("Connecting to Holochain app interface", extra={
                    'app_url': self.config.app_url
                })
                self.app_ws = await websockets.connect(
                    self.config.app_url,
                    additional_headers={"Origin": "http://localhost"},
                    ping_interval=20,
                    ping_timeout=10
                )

                # Get app info and cell ID if not provided
                if not self.config.cell_id:
                    # list_apps expects status_filter parameter (null for all apps)
                    app_info = await self._call_admin("list_apps", {"status_filter": None})
                    # Find our app
                    for app in app_info.get("apps", []):
                        if app.get("installed_app_id") == self.config.app_id:
                            # Get first cell ID
                            cell_data = app.get("cell_data", [])
                            if cell_data:
                                self.config.cell_id = cell_data[0].get("cell_id")
                                break

                    if not self.config.cell_id:
                        logger.warning("Could not find cell ID for app", extra={
                            'app_id': self.config.app_id,
                            'recoverable': True
                        })

                logger.info("Connected to Holochain conductor", extra={
                    'cell_id': self.config.cell_id[:8] if self.config.cell_id else 'unknown',
                    'app_id': self.config.app_id
                })

            except Exception as e:
                logger.error("Failed to connect to Holochain", extra={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'admin_url': self.config.admin_url,
                    'app_url': self.config.app_url
                })
                raise HolochainConnectionError(
                    message=f"Failed to connect to Holochain conductor: {e}",
                    admin_url=self.config.admin_url,
                    app_url=self.config.app_url,
                    cause=e
                )

    async def disconnect(self):
        """Close WebSocket connections"""
        if self.admin_ws:
            await self.admin_ws.close()
            self.admin_ws = None

        if self.app_ws:
            await self.app_ws.close()
            self.app_ws = None

        logger.info("Disconnected from Holochain")

    async def _call_admin(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call admin interface method using Holochain WebSocket protocol.

        Message format: {"type": "method_name", "value": {...params...}}
        Response format: {"type": "response_type", "value": {...result...}}

        Uses MessagePack serialization (Holochain's native format).
        """
        if not self.admin_ws:
            raise RuntimeError("Not connected to admin interface")

        if not msgpack:
            raise ImportError("msgpack required for Holochain: pip install msgpack")

        self.request_id += 1

        # Holochain WebSocket protocol uses "value" not "data"
        request = {
            "type": method,
            "value": params
        }

        logger.debug(f"Admin request: {method} - {params}")

        # Serialize using MessagePack (Holochain's native format)
        await self.admin_ws.send(msgpack.packb(request))
        response_bytes = await asyncio.wait_for(
            self.admin_ws.recv(),
            timeout=self.config.timeout
        )

        # Deserialize MessagePack response (raw=False for string keys)
        response = msgpack.unpackb(response_bytes, raw=False)
        logger.debug(f"Admin response: {response}")

        # Check for error in response
        if isinstance(response, dict):
            if response.get("type") == "error":
                error_msg = response.get("value", {})
                raise RuntimeError(f"Holochain admin error: {error_msg}")

            # Return the value field (or entire response if no value)
            return response.get("value", response)

        return response

    async def _call_zome(
        self,
        zome_name: str,
        fn_name: str,
        payload: Any
    ) -> Any:
        """Call zome function via app interface (using MessagePack)"""
        if not self.app_ws:
            raise RuntimeError("Not connected to app interface")

        if not self.config.cell_id:
            raise RuntimeError("Cell ID not configured")

        if not msgpack:
            raise ImportError("msgpack required for Holochain: pip install msgpack")

        self.request_id += 1

        # Construct call_zome request
        request = {
            "type": "call_zome",
            "data": {
                "cell_id": self.config.cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": payload,
                "cap_secret": None,
                "provenance": None
            },
            "id": str(self.request_id)
        }

        try:
            # Serialize using MessagePack
            await self.app_ws.send(msgpack.packb(request))
            response_bytes = await asyncio.wait_for(
                self.app_ws.recv(),
                timeout=self.config.timeout
            )

            # Deserialize MessagePack response
            response = msgpack.unpackb(response_bytes, raw=False)

            if "error" in response:
                error_msg = response.get("error", {})
                raise RuntimeError(f"Holochain zome error ({zome_name}.{fn_name}): {error_msg}")

            # Return the result
            data = response.get("data", {})
            return data.get("payload") if "payload" in data else data

        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout calling {zome_name}.{fn_name}")

    # ============================================================
    # Gradient Storage Operations
    # ============================================================

    async def store_gradient(
        self,
        node_id: str,
        round_num: int,
        gradient: List[float],
        gradient_shape: List[int],
        reputation_score: float,
        pogq_score: Optional[float] = None,
        validation_passed: bool = True,
        anomaly_detected: bool = False,
        blacklisted: bool = False
    ) -> str:
        """
        Store gradient in Holochain DHT

        Args:
            node_id: Node identifier
            round_num: Training round number
            gradient: Gradient values (will be base64 encoded)
            gradient_shape: Shape of gradient tensor
            reputation_score: Node's reputation
            pogq_score: Proof of Gradient Quality score (optional for privacy)
            validation_passed: Whether gradient passed validation
            anomaly_detected: Whether anomaly was detected
            blacklisted: Whether node is blacklisted

        Returns:
            entry_hash: Holochain entry hash
        """
        # Encode gradient as base64 for storage
        gradient_bytes = json.dumps(gradient).encode()
        gradient_b64 = base64.b64encode(gradient_bytes).decode()

        payload = {
            "node_id": int(node_id.split("-")[-1]) if "-" in node_id else 0,  # Extract numeric ID
            "round_num": round_num,
            "gradient_data": gradient_b64,
            "gradient_shape": gradient_shape,
            "gradient_dtype": "float32",
            "reputation_score": reputation_score,
            "validation_passed": validation_passed,
            "pogq_score": pogq_score,
            "anomaly_detected": anomaly_detected,
            "blacklisted": blacklisted
        }

        result = await self._call_zome("gradient_storage", "store_gradient", payload)

        # Extract entry hash from result
        if isinstance(result, dict):
            entry_hash = result.get("entry_hash", "")
            logger.debug(f"Gradient stored: {entry_hash[:16]}...")
            return entry_hash

        return str(result)

    async def get_gradient(self, entry_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve gradient by entry hash"""
        result = await self._call_zome("gradient_storage", "get_gradient", entry_hash)

        if result and isinstance(result, dict):
            # Decode gradient data if present
            if "gradient_data" in result:
                gradient_b64 = result["gradient_data"]
                gradient_bytes = base64.b64decode(gradient_b64)
                result["gradient"] = json.loads(gradient_bytes.decode())

            return result

        return None

    async def get_gradients_by_round(self, round_num: int) -> List[Dict[str, Any]]:
        """Query gradients for a specific round"""
        result = await self._call_zome(
            "gradient_storage",
            "get_gradients_by_round",
            round_num
        )

        if isinstance(result, list):
            # Decode gradient data for each
            for gradient in result:
                if "gradient_data" in gradient:
                    gradient_b64 = gradient["gradient_data"]
                    gradient_bytes = base64.b64decode(gradient_b64)
                    gradient["gradient"] = json.loads(gradient_bytes.decode())

            return result

        return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get gradient storage statistics"""
        result = await self._call_zome("gradient_storage", "get_statistics", {})
        return result if isinstance(result, dict) else {}

    # ============================================================
    # Credit Operations
    # ============================================================

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str,
        pogq_score: Optional[float] = None,
        gradient_hash: Optional[str] = None,
        verifiers: Optional[List[str]] = None
    ) -> str:
        """
        Issue credit to a node

        Args:
            holder: Node receiving credit
            amount: Credit amount
            earned_from: Reason (gradient_quality, byzantine_detection, etc.)
            pogq_score: Optional PoGQ score for gradient credits
            gradient_hash: Optional gradient hash reference
            verifiers: Optional list of verifier agent IDs

        Returns:
            action_hash: Holochain action hash
        """
        # Convert holder to AgentPubKey format (simplified for demo)
        holder_key = self._node_id_to_agent_key(holder)

        # Construct EarnReason based on earned_from
        if earned_from == "gradient_quality":
            earn_reason = {
                "QualityGradient": {
                    "pogq_score": pogq_score or 0.7,
                    "gradient_hash": gradient_hash or ""
                }
            }
        elif earned_from == "byzantine_detection":
            earn_reason = {
                "ByzantineDetection": {
                    "caught_node_id": 0,
                    "evidence_hash": gradient_hash or ""
                }
            }
        elif earned_from == "peer_validation":
            earn_reason = {
                "PeerValidation": {
                    "validated_node_id": 0,
                    "gradient_hash": gradient_hash or ""
                }
            }
        else:
            earn_reason = {
                "NetworkContribution": {
                    "uptime_hours": amount
                }
            }

        # Convert verifiers to agent keys
        verifier_keys = [self._node_id_to_agent_key(v) for v in (verifiers or [])]

        payload = {
            "holder": holder_key,
            "amount": amount,
            "earned_from": earn_reason,
            "verifiers": verifier_keys
        }

        result = await self._call_zome("zerotrustml_credits", "create_credit", payload)

        logger.debug(f"Credit issued: {amount} to {holder}")
        return str(result)

    async def get_balance(self, holder: str) -> int:
        """Get credit balance for a node"""
        holder_key = self._node_id_to_agent_key(holder)

        result = await self._call_zome("zerotrustml_credits", "get_balance", holder_key)

        return int(result) if result else 0

    async def get_credit_statistics(self, holder: str) -> Dict[str, Any]:
        """Get credit statistics for a node"""
        holder_key = self._node_id_to_agent_key(holder)

        result = await self._call_zome("zerotrustml_credits", "get_credit_statistics", holder_key)

        return result if isinstance(result, dict) else {}

    # ============================================================
    # Reputation Operations (if reputation_tracker zome exists)
    # ============================================================

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ):
        """Update node reputation (placeholder for reputation_tracker zome)"""
        # This would call reputation_tracker zome if it has update functions
        logger.debug(f"Reputation update: {node_id} {score_delta:+.3f} ({reason})")
        # For now, reputation is tracked in PostgreSQL
        pass

    # ============================================================
    # Byzantine Event Logging
    # ============================================================

    async def log_byzantine_event(
        self,
        node_id: str,
        round_num: int,
        detection_method: str,
        severity: str,
        details: Dict[str, Any]
    ) -> str:
        """
        Log Byzantine detection event

        This stores the event in Holochain as immutable proof of misbehavior.
        Could be implemented as a separate zome or added to gradient_storage.
        """
        # For now, store as a gradient with special flags
        payload = {
            "node_id": int(node_id.split("-")[-1]) if "-" in node_id else 0,
            "round_num": round_num,
            "gradient_data": base64.b64encode(json.dumps(details).encode()).decode(),
            "gradient_shape": [0],
            "gradient_dtype": "byzantine_event",
            "reputation_score": 0.0,
            "validation_passed": False,
            "pogq_score": None,
            "anomaly_detected": True,
            "blacklisted": True
        }

        result = await self._call_zome("gradient_storage", "store_gradient", payload)

        logger.warning(f"Byzantine event logged: {node_id} ({detection_method})")
        return str(result)

    # ============================================================
    # Helper Methods
    # ============================================================

    def _node_id_to_agent_key(self, node_id: str) -> str:
        """
        Convert node ID to Holochain AgentPubKey format

        In production, this would be a proper agent public key.
        For demo, we generate a deterministic key from node ID.
        """
        # Simplified: use node ID as base64-encoded key
        # Real implementation would use proper Ed25519 public keys
        key_bytes = node_id.encode().ljust(32, b'\x00')[:32]
        return base64.b64encode(key_bytes).decode()

    async def health_check(self) -> bool:
        """Check if Holochain conductor is responsive"""
        try:
            if not self.admin_ws or not self.app_ws:
                return False

            # Try to get app info
            await self._call_admin("list_apps", {})
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# ============================================================
# Testing/Demo
# ============================================================

async def demo_holochain_client():
    """Demo Holochain client operations"""
    print("🧪 Testing Holochain Client\n")

    client = HolochainClient()

    try:
        # Connect
        print("Connecting to Holochain...")
        await client.connect()
        print("✅ Connected\n")

        # Health check
        healthy = await client.health_check()
        print(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}\n")

        # Store gradient
        print("Storing test gradient...")
        gradient_hash = await client.store_gradient(
            node_id="hospital-a",
            round_num=1,
            gradient=[0.1, 0.2, 0.3],
            gradient_shape=[3],
            reputation_score=0.8,
            pogq_score=0.95,
            validation_passed=True
        )
        print(f"✅ Gradient stored: {gradient_hash[:16]}...\n")

        # Retrieve gradient
        print("Retrieving gradient...")
        gradient = await client.get_gradient(gradient_hash)
        print(f"✅ Retrieved: {gradient}\n")

        # Issue credit
        print("Issuing credit...")
        credit_hash = await client.issue_credit(
            holder="hospital-a",
            amount=10,
            earned_from="gradient_quality",
            pogq_score=0.95,
            gradient_hash=gradient_hash
        )
        print(f"✅ Credit issued: {credit_hash[:16]}...\n")

        # Get balance
        balance = await client.get_balance("hospital-a")
        print(f"✅ Balance: {balance} credits\n")

        # Get statistics
        stats = await client.get_statistics()
        print(f"✅ Statistics: {stats}\n")

        await client.disconnect()
        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n(This is expected if Holochain conductor is not running)")
        print("Start Holochain with: docker-compose -f docker-compose.phase10.yml up -d holochain")


if __name__ == "__main__":
    asyncio.run(demo_holochain_client())
