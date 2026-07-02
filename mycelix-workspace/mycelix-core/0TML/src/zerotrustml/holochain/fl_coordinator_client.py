#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Federated Learning Coordinator Client for Holochain

Provides a Python client to interact with the FL Coordinator zome for
Byzantine-resistant federated learning on Holochain.

This is the primary integration point between 0TML and Holochain, enabling:
- Gradient submission and retrieval (FREE on Holochain DHT)
- PoGQ validation and reputation management
- Byzantine detection and cartel detection
- HyperFeel compressed gradient storage
- Round coordination and completion

Usage:
    from zerotrustml.holochain import FLCoordinatorClient

    async with FLCoordinatorClient() as client:
        # Submit a gradient for a round
        result = await client.submit_gradient(
            node_id="node-1",
            round_num=1,
            gradient_data=gradient_bytes,
            gradient_shape=[100, 10],
        )

        # Get all gradients for a round
        gradients = await client.get_round_gradients(round_num=1)
"""

import asyncio
import json
import base64
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Unified logging and error handling
from zerotrustml.logging import get_logger, async_timed
from zerotrustml.exceptions import (
    HolochainConnectionError,
    HolochainZomeError,
    ConfigurationError,
)
from zerotrustml.error_handling import CircuitBreaker

try:
    import websockets
except ImportError:
    websockets = None

try:
    import msgpack
except ImportError:
    msgpack = None

logger = get_logger(__name__)

# Circuit breaker for FL operations
_fl_breaker = CircuitBreaker(
    name="fl_coordinator",
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=2
)


class ByzantineType(Enum):
    """Types of Byzantine behavior"""
    DATA_POISONING = "DataPoisoning"
    MODEL_POISONING = "ModelPoisoning"
    FREE_RIDING = "FreeRiding"
    SYBIL_ATTACK = "SybilAttack"
    GRADIENT_TAMPERING = "GradientTampering"
    CARTEL_COLLUSION = "CartelCollusion"


@dataclass
class GradientSubmission:
    """Result of a gradient submission"""
    action_hash: str
    node_id: str
    round_num: int
    timestamp: int
    pogq_score: Optional[float] = None
    reputation_delta: Optional[float] = None


@dataclass
class ModelGradient:
    """Gradient data retrieved from Holochain"""
    node_id: str
    round_num: int
    gradient_data: bytes
    gradient_shape: List[int]
    gradient_dtype: str
    hyperfeel_hv: Optional[bytes] = None
    hyperfeel_scale: Optional[float] = None
    signature: Optional[str] = None
    timestamp: int = 0
    pogq_score: Optional[float] = None
    reputation_score: float = 0.5


@dataclass
class NodeReputation:
    """Node reputation data"""
    node_id: str
    score: float
    total_rounds: int
    successful_rounds: int
    byzantine_flags: int
    last_update: int
    trend: str = "stable"


@dataclass
class ByzantineDetectionResult:
    """Result of Byzantine detection"""
    detected: bool
    byzantine_nodes: List[str]
    detection_method: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CartelDetectionResult:
    """Result of cartel detection"""
    detected: bool
    cartels: List[List[str]]
    collusion_scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundStats:
    """Statistics for an FL round"""
    round_num: int
    total_gradients: int
    avg_pogq_score: float
    avg_reputation: float
    byzantine_count: int
    hyperfeel_count: int
    completion_time: Optional[int] = None


class FLCoordinatorClient:
    """
    Client for the Holochain Federated Learning Coordinator zome.

    This client provides the Python interface to the FL coordinator,
    enabling Byzantine-resistant federated learning with:
    - FREE gradient storage and retrieval via Holochain DHT
    - PoGQ (Proof of Gradient Quality) validation
    - Hierarchical Byzantine detection
    - HyperFeel gradient compression
    - Reputation-based aggregation

    The FL coordinator zome is the primary layer for high-frequency
    operations, with Ethereum/Polygon used only for periodic anchoring.
    """

    def __init__(
        self,
        admin_url: str = "ws://localhost:8888",
        app_url: str = "ws://localhost:8889",
        app_id: str = "mycelix-fl",
        cell_id: Optional[str] = None,
        zome_name: str = "federated_learning_coordinator",
        timeout: int = 30,
    ):
        self.admin_url = admin_url
        self.app_url = app_url
        self.app_id = app_id
        self.cell_id = cell_id
        self.zome_name = zome_name
        self.timeout = timeout

        self.admin_ws = None
        self.app_ws = None
        self.request_id = 0
        self._connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    @async_timed(level=20, message="FL coordinator connection established")
    async def connect(self):
        """Connect to Holochain conductor"""
        if not websockets:
            raise ConfigurationError(
                message="websockets library required: pip install websockets",
                config_key="dependencies.websockets"
            )
        if not msgpack:
            raise ConfigurationError(
                message="msgpack library required: pip install msgpack",
                config_key="dependencies.msgpack"
            )

        try:
            logger.info("Connecting to Holochain FL coordinator", extra={
                'admin_url': self.admin_url,
                'app_url': self.app_url,
                'app_id': self.app_id,
            })

            # Connect to admin interface
            self.admin_ws = await websockets.connect(
                self.admin_url,
                additional_headers={"Origin": "http://localhost"},
                ping_interval=20,
                ping_timeout=10
            )

            # Connect to app interface
            self.app_ws = await websockets.connect(
                self.app_url,
                additional_headers={"Origin": "http://localhost"},
                ping_interval=20,
                ping_timeout=10
            )

            # Get cell ID if not provided
            if not self.cell_id:
                await self._discover_cell_id()

            self._connected = True
            logger.info("Connected to FL coordinator", extra={
                'cell_id': self.cell_id[:16] + "..." if self.cell_id else 'unknown',
            })

        except Exception as e:
            logger.error("Failed to connect to FL coordinator", extra={
                'error': str(e),
                'admin_url': self.admin_url,
            })
            raise HolochainConnectionError(
                message=f"Failed to connect to FL coordinator: {e}",
                admin_url=self.admin_url,
                app_url=self.app_url,
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

        self._connected = False
        logger.info("Disconnected from FL coordinator")

    async def _discover_cell_id(self):
        """Discover cell ID from conductor"""
        try:
            apps = await self._call_admin("list_apps", {"status_filter": None})
            for app in apps.get("apps", []):
                if app.get("installed_app_id") == self.app_id:
                    cell_data = app.get("cell_data", [])
                    if cell_data:
                        self.cell_id = cell_data[0].get("cell_id")
                        return

            logger.warning("Could not find cell ID for app", extra={
                'app_id': self.app_id
            })
        except Exception as e:
            logger.warning(f"Failed to discover cell ID: {e}")

    async def _call_admin(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call admin interface method"""
        if not self.admin_ws:
            raise RuntimeError("Not connected to admin interface")

        self.request_id += 1
        request = {"type": method, "value": params}

        await self.admin_ws.send(msgpack.packb(request))
        response_bytes = await asyncio.wait_for(
            self.admin_ws.recv(),
            timeout=self.timeout
        )

        response = msgpack.unpackb(response_bytes, raw=False)

        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"Holochain admin error: {response.get('value')}")

        return response.get("value", response)

    async def _call_zome(self, fn_name: str, payload: Any) -> Any:
        """Call FL coordinator zome function"""
        if not self.app_ws:
            raise RuntimeError("Not connected to app interface")

        if not self.cell_id:
            raise RuntimeError("Cell ID not configured")

        self.request_id += 1
        request = {
            "type": "call_zome",
            "data": {
                "cell_id": self.cell_id,
                "zome_name": self.zome_name,
                "fn_name": fn_name,
                "payload": payload,
                "cap_secret": None,
                "provenance": None
            },
            "id": str(self.request_id)
        }

        try:
            await self.app_ws.send(msgpack.packb(request))
            response_bytes = await asyncio.wait_for(
                self.app_ws.recv(),
                timeout=self.timeout
            )

            response = msgpack.unpackb(response_bytes, raw=False)

            if "error" in response:
                raise HolochainZomeError(
                    message=f"Zome error in {fn_name}: {response['error']}",
                    zome_name=self.zome_name,
                    fn_name=fn_name
                )

            data = response.get("data", {})
            return data.get("payload") if "payload" in data else data

        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout calling {self.zome_name}.{fn_name}")

    # =========================================================================
    # Gradient Operations
    # =========================================================================

    async def submit_gradient(
        self,
        node_id: str,
        round_num: int,
        gradient_data: bytes,
        gradient_shape: List[int],
        gradient_dtype: str = "float32",
        pogq_score: Optional[float] = None,
    ) -> GradientSubmission:
        """
        Submit a gradient to the DHT (FREE operation).

        Args:
            node_id: Unique node identifier
            round_num: Training round number
            gradient_data: Serialized gradient bytes
            gradient_shape: Shape of the gradient tensor
            gradient_dtype: Data type (float32, float16, etc.)
            pogq_score: Pre-computed PoGQ score (optional)

        Returns:
            GradientSubmission with action hash and metadata
        """
        payload = {
            "node_id": node_id,
            "round_num": round_num,
            "gradient_data": base64.b64encode(gradient_data).decode(),
            "gradient_shape": gradient_shape,
            "gradient_dtype": gradient_dtype,
        }

        result = await self._call_zome("submit_gradient", payload)

        action_hash = result if isinstance(result, str) else result.get("action_hash", "")

        logger.debug(f"Gradient submitted: {node_id} round {round_num}", extra={
            'action_hash': action_hash[:16] + "..." if action_hash else 'unknown'
        })

        return GradientSubmission(
            action_hash=action_hash,
            node_id=node_id,
            round_num=round_num,
            timestamp=int(datetime.now().timestamp()),
            pogq_score=pogq_score,
        )

    async def submit_gradient_with_pogq(
        self,
        node_id: str,
        round_num: int,
        gradient_data: bytes,
        gradient_shape: List[int],
        gradient_dtype: str = "float32",
        loss_before: float = 0.0,
        loss_after: float = 0.0,
        accuracy: float = 0.0,
        dataset_size: int = 0,
    ) -> GradientSubmission:
        """
        Submit gradient with PoGQ (Proof of Gradient Quality) validation.

        The zome computes PoGQ score and updates reputation atomically.

        Args:
            node_id: Unique node identifier
            round_num: Training round number
            gradient_data: Serialized gradient bytes
            gradient_shape: Shape of the gradient tensor
            gradient_dtype: Data type
            loss_before: Loss before applying gradient
            loss_after: Loss after applying gradient
            accuracy: Model accuracy
            dataset_size: Size of local dataset

        Returns:
            GradientSubmission with computed PoGQ score
        """
        payload = {
            "node_id": node_id,
            "round_num": round_num,
            "gradient_data": base64.b64encode(gradient_data).decode(),
            "gradient_shape": gradient_shape,
            "gradient_dtype": gradient_dtype,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "accuracy": accuracy,
            "dataset_size": dataset_size,
        }

        result = await self._call_zome("submit_gradient_with_pogq", payload)

        return GradientSubmission(
            action_hash=result.get("action_hash", ""),
            node_id=node_id,
            round_num=round_num,
            timestamp=int(datetime.now().timestamp()),
            pogq_score=result.get("pogq_score"),
            reputation_delta=result.get("reputation_delta"),
        )

    async def submit_compressed_gradient(
        self,
        node_id: str,
        round_num: int,
        hyperfeel_hv: bytes,
        hyperfeel_scale: float,
        gradient_shape: List[int],
    ) -> GradientSubmission:
        """
        Submit HyperFeel-compressed gradient (10-50x compression).

        HyperFeel encoding converts full gradients to compact hypervectors
        while preserving semantic similarity for Byzantine detection.

        Args:
            node_id: Unique node identifier
            round_num: Training round number
            hyperfeel_hv: HyperFeel hypervector bytes (typically 2KB)
            hyperfeel_scale: Scale factor for reconstruction
            gradient_shape: Original gradient shape

        Returns:
            GradientSubmission for the compressed gradient
        """
        payload = {
            "node_id": node_id,
            "round_num": round_num,
            "hyperfeel_hv": base64.b64encode(hyperfeel_hv).decode(),
            "hyperfeel_scale": hyperfeel_scale,
            "gradient_shape": gradient_shape,
        }

        result = await self._call_zome("submit_compressed_gradient", payload)

        return GradientSubmission(
            action_hash=result.get("action_hash", ""),
            node_id=node_id,
            round_num=round_num,
            timestamp=int(datetime.now().timestamp()),
            pogq_score=result.get("similarity_to_median"),
        )

    async def get_round_gradients(self, round_num: int) -> List[ModelGradient]:
        """
        Retrieve all gradients for a round (FREE operation).

        Args:
            round_num: Training round number

        Returns:
            List of ModelGradient objects with gradient data
        """
        result = await self._call_zome("get_round_gradients", round_num)

        gradients = []
        for item in (result if isinstance(result, list) else []):
            action_hash, gradient_data = item if isinstance(item, tuple) else (None, item)

            gradients.append(ModelGradient(
                node_id=gradient_data.get("node_id", ""),
                round_num=gradient_data.get("round_num", round_num),
                gradient_data=base64.b64decode(gradient_data.get("gradient_data", "")),
                gradient_shape=gradient_data.get("gradient_shape", []),
                gradient_dtype=gradient_data.get("gradient_dtype", "float32"),
                hyperfeel_hv=base64.b64decode(gradient_data.get("hyperfeel_hv", ""))
                    if gradient_data.get("hyperfeel_hv") else None,
                hyperfeel_scale=gradient_data.get("hyperfeel_scale"),
                timestamp=gradient_data.get("timestamp", 0),
                pogq_score=gradient_data.get("pogq_score"),
                reputation_score=gradient_data.get("reputation_score", 0.5),
            ))

        return gradients

    async def get_node_gradients(self, node_id: str) -> List[ModelGradient]:
        """
        Get all gradients submitted by a specific node.

        Args:
            node_id: Node identifier

        Returns:
            List of ModelGradient objects
        """
        result = await self._call_zome("get_node_gradients", node_id)

        gradients = []
        for item in (result if isinstance(result, list) else []):
            _, gradient_data = item if isinstance(item, tuple) else (None, item)

            gradients.append(ModelGradient(
                node_id=node_id,
                round_num=gradient_data.get("round_num", 0),
                gradient_data=base64.b64decode(gradient_data.get("gradient_data", "")),
                gradient_shape=gradient_data.get("gradient_shape", []),
                gradient_dtype=gradient_data.get("gradient_dtype", "float32"),
                timestamp=gradient_data.get("timestamp", 0),
            ))

        return gradients

    # =========================================================================
    # Round Management
    # =========================================================================

    async def complete_round(
        self,
        round_num: int,
        aggregated_model_hash: str,
        participants: List[str],
        success: bool = True,
    ) -> str:
        """
        Mark a round as complete.

        Args:
            round_num: Round number
            aggregated_model_hash: Hash of aggregated model
            participants: List of participant node IDs
            success: Whether round completed successfully

        Returns:
            Action hash of round completion record
        """
        payload = {
            "round_num": round_num,
            "aggregated_model_hash": aggregated_model_hash,
            "participants": participants,
            "success": success,
        }

        result = await self._call_zome("complete_round", payload)
        return result if isinstance(result, str) else result.get("action_hash", "")

    async def get_round_stats(self, round_num: int) -> RoundStats:
        """
        Get statistics for a round.

        Args:
            round_num: Round number

        Returns:
            RoundStats with aggregated metrics
        """
        result = await self._call_zome("get_round_matl_stats", round_num)

        return RoundStats(
            round_num=round_num,
            total_gradients=result.get("total_gradients", 0),
            avg_pogq_score=result.get("avg_pogq_score", 0.0),
            avg_reputation=result.get("avg_reputation", 0.5),
            byzantine_count=result.get("byzantine_count", 0),
            hyperfeel_count=result.get("hyperfeel_count", 0),
            completion_time=result.get("completion_time"),
        )

    # =========================================================================
    # Byzantine Detection
    # =========================================================================

    async def record_byzantine(
        self,
        node_id: str,
        round_num: int,
        byzantine_type: ByzantineType,
        evidence_hash: str,
        detected_by: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Record a Byzantine detection event.

        Args:
            node_id: ID of the Byzantine node
            round_num: Round where detected
            byzantine_type: Type of Byzantine behavior
            evidence_hash: Hash of evidence data
            detected_by: ID of detecting node
            confidence: Detection confidence (0-1)

        Returns:
            Action hash of the Byzantine record
        """
        payload = {
            "node_id": node_id,
            "round_num": round_num,
            "byzantine_type": byzantine_type.value,
            "evidence_hash": evidence_hash,
            "detected_by": detected_by,
            "confidence": confidence,
        }

        result = await self._call_zome("record_byzantine", payload)

        logger.warning(f"Byzantine recorded: {node_id} ({byzantine_type.value})", extra={
            'round': round_num,
            'confidence': confidence,
        })

        return result if isinstance(result, str) else result.get("action_hash", "")

    async def detect_byzantine_hierarchical(
        self,
        round_num: int,
        threshold: float = 0.3,
    ) -> ByzantineDetectionResult:
        """
        Run hierarchical Byzantine detection on round gradients.

        Uses the MATL hierarchical detector:
        1. Compute pairwise gradient similarities
        2. Identify statistical outliers
        3. Apply reputation-weighted voting

        Args:
            round_num: Round to analyze
            threshold: Detection threshold (0-1)

        Returns:
            ByzantineDetectionResult with detected nodes
        """
        payload = {
            "round_num": round_num,
            "threshold": threshold,
        }

        result = await self._call_zome("detect_byzantine_hierarchical", payload)

        return ByzantineDetectionResult(
            detected=result.get("detected", False),
            byzantine_nodes=result.get("byzantine_nodes", []),
            detection_method="hierarchical",
            confidence=result.get("confidence", 0.0),
            details=result.get("details", {}),
        )

    async def detect_cartels(
        self,
        round_num: int,
        min_cartel_size: int = 2,
    ) -> CartelDetectionResult:
        """
        Detect coordinated Byzantine attacks (cartels).

        Identifies groups of nodes submitting suspiciously similar
        gradients that deviate from the honest majority.

        Args:
            round_num: Round to analyze
            min_cartel_size: Minimum cartel size to detect

        Returns:
            CartelDetectionResult with identified cartels
        """
        payload = {
            "round_num": round_num,
            "min_cartel_size": min_cartel_size,
        }

        result = await self._call_zome("detect_cartels", payload)

        return CartelDetectionResult(
            detected=result.get("detected", False),
            cartels=result.get("cartels", []),
            collusion_scores=result.get("collusion_scores", {}),
            evidence=result.get("evidence", {}),
        )

    async def get_round_byzantine_records(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Get all Byzantine records for a round.

        Args:
            round_num: Round number

        Returns:
            List of Byzantine record dictionaries
        """
        result = await self._call_zome("get_round_byzantine_records", round_num)
        return result if isinstance(result, list) else []

    # =========================================================================
    # Reputation Management
    # =========================================================================

    async def get_reputation(self, node_id: str) -> Optional[NodeReputation]:
        """
        Get reputation for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeReputation or None if not found
        """
        result = await self._call_zome("get_reputation", node_id)

        if not result:
            return None

        return NodeReputation(
            node_id=node_id,
            score=result.get("score", 0.5),
            total_rounds=result.get("total_rounds", 0),
            successful_rounds=result.get("successful_rounds", 0),
            byzantine_flags=result.get("byzantine_flags", 0),
            last_update=result.get("last_update", 0),
            trend=result.get("trend", "stable"),
        )

    async def update_reputation(
        self,
        node_id: str,
        delta: float,
        reason: str,
        round_num: Optional[int] = None,
    ) -> str:
        """
        Update a node's reputation.

        Args:
            node_id: Node identifier
            delta: Reputation change (-1 to +1)
            reason: Reason for update
            round_num: Related round (optional)

        Returns:
            Action hash of reputation update
        """
        payload = {
            "node_id": node_id,
            "delta": delta,
            "reason": reason,
        }
        if round_num is not None:
            payload["round_num"] = round_num

        result = await self._call_zome("update_reputation", payload)
        return result if isinstance(result, str) else result.get("action_hash", "")

    async def update_reputation_with_matl(
        self,
        node_id: str,
        pogq_score: float,
        byzantine_detected: bool,
        round_num: int,
    ) -> str:
        """
        Update reputation using MATL metrics.

        Args:
            node_id: Node identifier
            pogq_score: PoGQ score from gradient quality analysis
            byzantine_detected: Whether Byzantine behavior was detected
            round_num: Round number

        Returns:
            Action hash of reputation update
        """
        payload = {
            "node_id": node_id,
            "pogq_score": pogq_score,
            "byzantine_detected": byzantine_detected,
            "round_num": round_num,
        }

        result = await self._call_zome("update_reputation_with_matl", payload)
        return result if isinstance(result, str) else result.get("action_hash", "")

    # =========================================================================
    # HyperFeel Operations
    # =========================================================================

    async def get_round_hypervectors(self, round_num: int) -> List[Tuple[str, bytes]]:
        """
        Get all HyperFeel hypervectors for a round.

        Args:
            round_num: Round number

        Returns:
            List of (node_id, hypervector) tuples
        """
        result = await self._call_zome("get_round_hypervectors", round_num)

        hypervectors = []
        for item in (result if isinstance(result, list) else []):
            node_id, hv_bytes = item
            hypervectors.append((
                node_id,
                base64.b64decode(hv_bytes) if isinstance(hv_bytes, str) else hv_bytes
            ))

        return hypervectors

    async def aggregate_round_hypervectors(self, round_num: int) -> bytes:
        """
        Aggregate all hypervectors for a round into a single vector.

        Args:
            round_num: Round number

        Returns:
            Aggregated hypervector bytes
        """
        result = await self._call_zome("aggregate_round_hypervectors", round_num)

        if isinstance(result, str):
            return base64.b64decode(result)
        return bytes(result) if result else b""

    async def compute_hypervector_similarity(
        self,
        hv1: bytes,
        hv2: bytes,
    ) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Args:
            hv1: First hypervector
            hv2: Second hypervector

        Returns:
            Similarity score (0-1)
        """
        payload = [
            base64.b64encode(hv1).decode(),
            base64.b64encode(hv2).decode(),
        ]

        result = await self._call_zome("compute_hypervector_similarity", payload)
        return float(result) if result else 0.0

    # =========================================================================
    # Coordinator Bootstrap (for guardian ceremonies)
    # =========================================================================

    async def get_bootstrap_config(self) -> Optional[Dict[str, Any]]:
        """Get the current bootstrap configuration"""
        result = await self._call_zome("get_bootstrap_config", None)
        return result if isinstance(result, dict) else None

    async def get_active_coordinators(self) -> List[str]:
        """Get list of active coordinator public keys"""
        result = await self._call_zome("get_active_coordinators", None)
        return result if isinstance(result, list) else []

    async def get_active_guardians(self) -> List[Dict[str, Any]]:
        """Get list of active guardians"""
        result = await self._call_zome("get_active_guardians", None)
        return result if isinstance(result, list) else []

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if the FL coordinator is responsive"""
        try:
            if not self._connected:
                return False
            await self._call_admin("list_apps", {"status_filter": None})
            return True
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connected


# Factory function for easy creation
def create_fl_coordinator(
    admin_url: str = "ws://localhost:8888",
    app_url: str = "ws://localhost:8889",
    app_id: str = "mycelix-fl",
) -> FLCoordinatorClient:
    """
    Create a new FL Coordinator client.

    Example:
        async with create_fl_coordinator() as client:
            await client.submit_gradient(...)
    """
    return FLCoordinatorClient(
        admin_url=admin_url,
        app_url=app_url,
        app_id=app_id,
    )
