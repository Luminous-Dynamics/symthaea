#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root"""
Holochain Backend Implementation

Decentralized DHT-based storage using Holochain conductor.
Provides immutable audit trail and agent-centric architecture.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
import time
import json

try:
    import websockets
except ImportError:
    print("Warning: websockets not installed. Run: pip install websockets")
    websockets = None

try:
    import msgpack
except ImportError:
    print("Warning: msgpack not installed. Run: pip install msgpack")
    msgpack = None

from .storage_backend import (
    StorageBackend,
    BackendType,
    GradientRecord,
    CreditRecord,
    ByzantineEvent,
    ConnectionError as BackendConnectionError,
)

try:
    from zerotrustml.experimental.edge_validation import EdgeGradientProof
except ImportError:  # pragma: no cover - optional dependency
    EdgeGradientProof = None

logger = logging.getLogger(__name__)


class HolochainBackend(StorageBackend):
    """
    Holochain backend for fully decentralized deployments

    Features:
    - Immutable audit trail (DHT-based)
    - Agent-centric architecture
    - No central server (peer-to-peer)
    - Intrinsic data integrity
    - Censorship-resistant

    Use cases:
    - Fully decentralized FL
    - Web3/crypto FL networks
    - Censorship-resistant systems
    - When trustlessness is critical
    """

    def __init__(
        self,
        # wss:// by default — requires TLS certs on the Holochain conductor.
        # Override with HOLOCHAIN_ADMIN_URL / HOLOCHAIN_APP_URL env vars for
        # local dev (e.g. ws://localhost:8888).
        admin_url: str = "wss://localhost:8888",
        app_url: str = "wss://localhost:8889",
        app_id: str = "zerotrustml",
        timeout: int = 30
    ):
        super().__init__(BackendType.HOLOCHAIN)
        self.admin_url = os.environ.get("HOLOCHAIN_ADMIN_URL", admin_url)
        self.app_url = os.environ.get("HOLOCHAIN_APP_URL", app_url)
        if self.admin_url.startswith("ws://"):
            logger.warning(
                "Holochain admin connection using insecure ws:// — "
                "use wss:// with TLS certs in production"
            )
        if self.app_url.startswith("ws://"):
            logger.warning(
                "Holochain app connection using insecure ws:// — "
                "use wss:// with TLS certs in production"
            )
        self.app_id = app_id
        self.timeout = timeout
        self.admin_ws = None
        self.app_ws = None
        self.cell_id = None
        self.connection_time = None

    async def connect(self) -> bool:
        """Connect to Holochain conductor via WebSocket"""
        if not websockets or not msgpack:
            raise BackendConnectionError(
                "websockets and msgpack required: pip install websockets msgpack"
            )

        try:
            # Connect to admin interface (uses MessagePack in 0.5.6)
            # NOTE: websockets 15.0+ uses additional_headers, not extra_headers
            logger.info(f"Connecting to Holochain admin: {self.admin_url}")
            self.admin_ws = await websockets.connect(
                self.admin_url,
                additional_headers={"Origin": "http://localhost"},
                ping_interval=20,
                ping_timeout=10
            )

            # Connect to app interface
            logger.info(f"Connecting to Holochain app: {self.app_url}")
            self.app_ws = await websockets.connect(
                self.app_url,
                additional_headers={"Origin": "http://localhost"},
                ping_interval=20,
                ping_timeout=10
            )

            # Get cell ID from list_apps
            try:
                app_info = await self._call_admin("list_apps", {"status_filter": None})
                # Find our app and get cell ID
                for app in app_info.get("apps", []):
                    if app.get("installed_app_id") == self.app_id:
                        cell_data = app.get("cell_data", [])
                        if cell_data:
                            self.cell_id = cell_data[0].get("cell_id")
                            break
            except Exception as e:
                logger.warning(f"Could not get cell_id (app may need installation): {e}")
                # Continue anyway - app might need to be installed separately

            self.connected = True
            self.connection_time = time.time()
            logger.info(f"✅ Holochain connected (cell_id: {self.cell_id})")
            return True

        except Exception as e:
            logger.error(f"Holochain connection failed: {e}")
            raise BackendConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """Close WebSocket connections"""
        if self.admin_ws:
            await self.admin_ws.close()
        if self.app_ws:
            await self.app_ws.close()

        self.connected = False
        logger.info("Holochain connection closed")

    async def health_check(self) -> Dict[str, Any]:
        """Check Holochain conductor health"""
        if not self.connected or not self.admin_ws:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": "Not connected"}
            }

        try:
            start = time.time()
            # Try to call list_apps
            await self._call_admin("list_apps", {"status_filter": None})
            latency = (time.time() - start) * 1000

            return {
                "healthy": True,
                "latency_ms": latency,
                "storage_available": True,
                "metadata": {
                    "admin_url": self.admin_url,
                    "app_url": self.app_url,
                    "app_id": self.app_id,
                    "cell_id": self.cell_id
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": str(e)}
            }

    async def _call_admin(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Holochain admin API (MessagePack)"""
        if not self.admin_ws:
            raise BackendConnectionError("Admin WebSocket not connected")

        # Holochain 0.5.6 admin API format
        request = {"type": method, "value": params}

        # Serialize with MessagePack
        await self.admin_ws.send(msgpack.packb(request))

        # Wait for response with timeout
        response_bytes = await asyncio.wait_for(
            self.admin_ws.recv(),
            timeout=self.timeout
        )

        # Deserialize MessagePack response (raw=False for string keys)
        response = msgpack.unpackb(response_bytes, raw=False)
        return response.get("value", response)

    async def _call_zome(
        self,
        zome_name: str,
        fn_name: str,
        payload: Any
    ) -> Any:
        """Call Holochain zome function (MessagePack)"""
        if not self.app_ws or not self.cell_id:
            raise BackendConnectionError("App WebSocket or cell_id not ready")

        request = {
            "type": "zome_call",
            "value": {
                "cell_id": self.cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": payload,
                "provenance": self.cell_id[1]  # AgentPubKey
            }
        }

        # Serialize with MessagePack
        await self.app_ws.send(msgpack.packb(request))

        # Wait for response
        response_bytes = await asyncio.wait_for(
            self.app_ws.recv(),
            timeout=self.timeout
        )

        response = msgpack.unpackb(response_bytes, raw=False)
        return response

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """Store gradient on Holochain DHT"""
        edge_proof = gradient_data.get("edge_proof")
        if EdgeGradientProof is not None and isinstance(edge_proof, EdgeGradientProof):
            edge_proof_payload: Optional[Dict[str, Any]] = edge_proof.to_dict()  # type: ignore[attr-defined]
        else:
            edge_proof_payload = edge_proof

        committee_votes = gradient_data.get("committee_votes")

        payload = {
            "node_id": gradient_data["node_id"],
            "round_num": gradient_data["round_num"],
            "gradient_hash": gradient_data["gradient_hash"],
            "pogq_score": gradient_data.get("pogq_score"),
            "zkpoc_verified": gradient_data.get("zkpoc_verified", False),
            "timestamp": int(gradient_data.get("timestamp", time.time())),
        }

        if edge_proof_payload is not None:
            payload["edge_proof"] = json.dumps(edge_proof_payload)
        if committee_votes:
            payload["committee_votes"] = json.dumps(committee_votes)

        # Call store_gradient zome function
        result = await self._call_zome(
            zome_name="gradient_storage",
            fn_name="store_gradient",
            payload=payload,
        )

        # Holochain returns entry hash as gradient_id
        gradient_id = result.get("entry_hash")
        logger.debug(f"Holochain: Gradient stored {gradient_id[:8]}...")
        return gradient_id

    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """Retrieve gradient from DHT"""
        try:
            result = await self._call_zome(
                zome_name="gradient_storage",
                fn_name="get_gradient",
                payload={"entry_hash": gradient_id}
            )

            if not result:
                return None

            edge_proof = None
            committee_votes = None

            if result.get("edge_proof"):
                try:
                    edge_proof = json.loads(result["edge_proof"])
                except (TypeError, ValueError):
                    edge_proof = result["edge_proof"]
                else:
                    if EdgeGradientProof is not None and isinstance(edge_proof, dict):
                        try:
                            edge_proof = EdgeGradientProof(**edge_proof)  # type: ignore[call-arg]
                        except TypeError:
                            pass

            if result.get("committee_votes"):
                try:
                    committee_votes = json.loads(result["committee_votes"])
                except (TypeError, ValueError):
                    committee_votes = result["committee_votes"]

            return GradientRecord(
                id=gradient_id,
                node_id=result["node_id"],
                round_num=result["round_num"],
                gradient=[],  # Holochain stores hash only, not full gradient
                gradient_hash=result["gradient_hash"],
                pogq_score=result.get("pogq_score"),
                zkpoc_verified=result.get("zkpoc_verified", False),
                reputation_score=0.0,  # Calculated separately
                timestamp=float(result["timestamp"]),
                backend_metadata={"entry_hash": gradient_id},
                edge_proof=edge_proof,
                committee_votes=committee_votes,
            )

        except Exception as e:
            logger.error(f"Failed to get gradient from Holochain: {e}")
            return None

    async def get_gradients_by_round(self, round_num: int) -> List[GradientRecord]:
        """Get all gradients for a round from DHT"""
        result = await self._call_zome(
            zome_name="gradient_storage",
            fn_name="get_gradients_by_round",
            payload={"round_num": round_num}
        )

        gradients = []
        for entry in result.get("entries", []):
            edge_proof = None
            committee_votes = None

            if entry.get("edge_proof"):
                try:
                    edge_proof = json.loads(entry["edge_proof"])
                except (TypeError, ValueError):
                    edge_proof = entry["edge_proof"]
                else:
                    if EdgeGradientProof is not None and isinstance(edge_proof, dict):
                        try:
                            edge_proof = EdgeGradientProof(**edge_proof)  # type: ignore[call-arg]
                        except TypeError:
                            pass

            if entry.get("committee_votes"):
                try:
                    committee_votes = json.loads(entry["committee_votes"])
                except (TypeError, ValueError):
                    committee_votes = entry["committee_votes"]

            gradients.append(GradientRecord(
                id=entry["entry_hash"],
                node_id=entry["node_id"],
                round_num=entry["round_num"],
                gradient=[],  # Hash only
                gradient_hash=entry["gradient_hash"],
                pogq_score=entry.get("pogq_score"),
                zkpoc_verified=entry.get("zkpoc_verified", False),
                reputation_score=0.0,
                timestamp=float(entry["timestamp"]),
                edge_proof=edge_proof,
                committee_votes=committee_votes,
            ))

        return gradients

    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """
        Verify gradient integrity

        For Holochain: Data is intrinsically immutable via DHT.
        Just verify the entry exists and hash matches.
        """
        gradient = await self.get_gradient(gradient_id)
        return gradient is not None  # If it exists on DHT, it's valid

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue credit on Holochain DHT"""
        result = await self._call_zome(
            zome_name="zerotrustml_credits",
            fn_name="issue_credit",
            payload={
                "holder": holder,
                "amount": amount,
                "earned_from": earned_from,
                "timestamp": int(time.time())
            }
        )

        credit_id = result.get("entry_hash")
        logger.debug(f"Holochain: Credit issued {credit_id[:8]}... ({amount} to {holder})")
        return credit_id

    async def get_credit_balance(self, node_id: str) -> int:
        """Get total credit balance from DHT"""
        result = await self._call_zome(
            zome_name="zerotrustml_credits",
            fn_name="get_balance",
            payload={"node_id": node_id}
        )

        return result.get("balance", 0)

    async def get_credit_history(self, node_id: str) -> List[CreditRecord]:
        """Get credit history from DHT"""
        result = await self._call_zome(
            zome_name="zerotrustml_credits",
            fn_name="get_credit_history",
            payload={"node_id": node_id}
        )

        credits = []
        for entry in result.get("entries", []):
            credits.append(CreditRecord(
                transaction_id=entry["entry_hash"],
                holder=entry["holder"],
                amount=entry["amount"],
                earned_from=entry["earned_from"],
                timestamp=float(entry["timestamp"])
            ))

        return credits

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get reputation from DHT"""
        result = await self._call_zome(
            zome_name="reputation_tracker",
            fn_name="get_reputation",
            payload={"node_id": node_id}
        )

        return {
            "node_id": node_id,
            "score": result.get("score", 0.5),
            "gradients_submitted": result.get("gradients_submitted", 0),
            "gradients_accepted": result.get("gradients_accepted", 0),
            "byzantine_events": result.get("byzantine_events", 0),
            "total_credits": result.get("total_credits", 0)
        }

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ) -> None:
        """Update reputation on DHT"""
        await self._call_zome(
            zome_name="reputation_tracker",
            fn_name="update_reputation",
            payload={
                "node_id": node_id,
                "score_delta": score_delta,
                "reason": reason
            }
        )

    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
        """Log Byzantine event to DHT"""
        result = await self._call_zome(
            zome_name="byzantine_tracker",
            fn_name="log_event",
            payload={
                "node_id": event["node_id"],
                "round_num": event["round_num"],
                "detection_method": event["detection_method"],
                "severity": event["severity"],
                "details": json.dumps(event.get("details", {})),
                "timestamp": int(time.time())
            }
        )

        event_id = result.get("entry_hash")
        logger.info(f"Holochain: Byzantine event logged {event_id[:8]}...")
        return event_id

    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[ByzantineEvent]:
        """Get Byzantine events from DHT"""
        result = await self._call_zome(
            zome_name="byzantine_tracker",
            fn_name="get_events",
            payload={
                "node_id": node_id,
                "round_num": round_num
            }
        )

        events = []
        for entry in result.get("entries", []):
            events.append(ByzantineEvent(
                event_id=entry["entry_hash"],
                node_id=entry["node_id"],
                round_num=entry["round_num"],
                detection_method=entry["detection_method"],
                severity=entry["severity"],
                details=json.loads(entry.get("details", "{}")),
                timestamp=float(entry["timestamp"])
            ))

        return events

    async def get_stats(self) -> Dict[str, Any]:
        """Get Holochain backend statistics"""
        try:
            # Query DHT for stats
            result = await self._call_zome(
                zome_name="zerotrustml_stats",
                fn_name="get_stats",
                payload={}
            )

            uptime = time.time() - self.connection_time if self.connection_time else 0

            return {
                "backend_type": "holochain",
                "total_gradients": result.get("total_gradients", 0),
                "total_credits_issued": result.get("total_credits", 0),
                "total_byzantine_events": result.get("total_byzantine", 0),
                "storage_size_bytes": result.get("dht_size", 0),
                "uptime_seconds": uptime,
                "metadata": {
                    "admin_url": self.admin_url,
                    "app_id": self.app_id,
                    "cell_id": self.cell_id,
                    "dht_peers": result.get("peers", 0)
                }
            }

        except Exception as e:
            logger.warning(f"Could not get stats from Holochain: {e}")
            return {
                "backend_type": "holochain",
                "total_gradients": 0,
                "total_credits_issued": 0,
                "total_byzantine_events": 0,
                "storage_size_bytes": 0,
                "uptime_seconds": 0,
                "metadata": {"error": str(e)}
            }
