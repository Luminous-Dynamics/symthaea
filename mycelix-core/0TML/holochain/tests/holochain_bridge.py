#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain WebSocket Bridge for Byzantine-Robust Federated Learning

This module provides a Python bridge to call defense_coordinator zome functions
through the Holochain conductor WebSocket API.

Usage:
    bridge = HolochainBridge(app_port=9002)
    await bridge.connect()
    result = await bridge.run_defense(gradients, round_num)
    await bridge.close()
"""

import asyncio
import json
import websockets
import msgpack
import base64
import struct
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


# Default conductor ports (matches test_live_conductor.py)
DEFAULT_ADMIN_PORT = 9010
DEFAULT_APP_PORT = 9011


class AggregationMethod(Enum):
    """Mirrors defense_coordinator's AggregationMethod enum."""
    MEAN = 0
    MEDIAN = 1
    TRIMMED_MEAN = 2
    KRUM = 3
    MULTI_KRUM = 4
    GEOMETRIC_MEDIAN = 5


@dataclass
class CBDAnalysisResult:
    """Result from CBD (Causal Byzantine Detection) analysis."""
    node_id: str
    z_score: float
    is_byzantine: bool
    causal_evidence_strength: float
    reasoning: str


@dataclass
class DefenseResult:
    """Result from running the three-layer defense."""
    aggregated_gradient: List[float]
    method_used: AggregationMethod
    detected_byzantine: List[str]
    layer1_flagged: List[str]  # Statistical detection
    layer2_flagged: List[str]  # Reputation filtering
    confidence: float
    processing_time_ms: float


class HolochainBridge:
    """
    WebSocket bridge to Holochain conductor for Byzantine FL.

    This class provides a Python interface to call the defense_coordinator
    zome functions through the Holochain conductor's App WebSocket API.
    """

    def __init__(
        self,
        admin_port: int = DEFAULT_ADMIN_PORT,
        app_port: int = DEFAULT_APP_PORT,
        app_id: str = "byzantine-fl-agent1",
    ):
        """
        Initialize the Holochain bridge.

        Args:
            admin_port: Conductor admin WebSocket port
            app_port: Conductor app WebSocket port
            app_id: Installed app ID to use for zome calls
        """
        self.admin_port = admin_port
        self.app_port = app_port
        self.app_id = app_id
        self.admin_ws = None
        self.app_ws = None
        self.call_id = 0
        self.cell_id = None
        self.agent_pubkey = None

    async def connect(self):
        """Connect to the Holochain conductor."""
        # Connect to admin interface to get cell info
        admin_uri = f"ws://127.0.0.1:{self.admin_port}"
        self.admin_ws = await websockets.connect(
            admin_uri,
            additional_headers={"Origin": "http://127.0.0.1"},
            subprotocols=["holochain-ws-v1"]
        )

        # Get cell ID for our app
        await self._get_cell_info()

        # Connect to app interface
        app_uri = f"ws://127.0.0.1:{self.app_port}"
        self.app_ws = await websockets.connect(
            app_uri,
            additional_headers={"Origin": "http://127.0.0.1"},
            subprotocols=["holochain-ws-v1"]
        )

        print(f"Connected to Holochain conductor (app: {self.app_id})")

    async def close(self):
        """Close all WebSocket connections."""
        if self.admin_ws:
            await self.admin_ws.close()
        if self.app_ws:
            await self.app_ws.close()

    async def _admin_call(self, request_type: str, data: dict) -> dict:
        """Make an admin API call."""
        self.call_id += 1
        request = {
            "id": self.call_id,
            "type": request_type,
            "data": data
        }
        await self.admin_ws.send(json.dumps(request))
        response = await self.admin_ws.recv()
        return json.loads(response)

    async def _get_cell_info(self):
        """Get cell ID and agent pubkey for our app."""
        result = await self._admin_call("list_apps", {"status_filter": None})
        apps = result.get("data", [])

        for app in apps:
            if app.get("installed_app_id") == self.app_id:
                # Extract cell info from app
                cells = app.get("cell_info", {})
                for role_name, role_cells in cells.items():
                    if role_cells:
                        cell = role_cells[0]  # Get first cell
                        self.cell_id = cell.get("cell_id")
                        self.agent_pubkey = self.cell_id[1] if self.cell_id else None
                        return

        raise ValueError(f"App '{self.app_id}' not found or has no cells")

    async def call_zome(
        self,
        zome_name: str,
        fn_name: str,
        payload: Any
    ) -> Any:
        """
        Call a zome function on the conductor.

        Args:
            zome_name: Name of the zome (e.g., "defense_coordinator")
            fn_name: Function name to call
            payload: Input data (will be msgpack encoded)

        Returns:
            Decoded response from the zome function
        """
        if not self.app_ws or not self.cell_id:
            raise RuntimeError("Not connected. Call connect() first.")

        self.call_id += 1

        # Serialize payload with msgpack
        payload_bytes = msgpack.packb(payload)
        payload_b64 = base64.b64encode(payload_bytes).decode('utf-8')

        # Build zome call request
        # Note: Format depends on Holochain version
        request = {
            "id": self.call_id,
            "type": "call_zome",
            "data": {
                "cell_id": self.cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": payload_b64,
                "provenance": self.agent_pubkey,
                "cap_secret": None  # Use unrestricted capability
            }
        }

        await self.app_ws.send(json.dumps(request))
        response = await self.app_ws.recv()
        result = json.loads(response)

        # Handle response
        if result.get("type") == "error":
            error_data = result.get("data", {})
            raise RuntimeError(f"Zome call failed: {error_data}")

        # Decode msgpack response
        response_data = result.get("data")
        if response_data:
            if isinstance(response_data, str):
                # Base64 encoded msgpack
                response_bytes = base64.b64decode(response_data)
                return msgpack.unpackb(response_bytes, raw=False)
            return response_data

        return None

    # ========== Defense Coordinator API ==========

    async def run_defense(
        self,
        gradients: List[Tuple[str, List[int]]],
        round_num: int,
        reputation_scores: Optional[List[Tuple[str, float]]] = None
    ) -> DefenseResult:
        """
        Run three-layer Byzantine defense on submitted gradients.

        This calls the defense_coordinator's process_round function which
        implements RBFT v2.4 with:
        - Layer 1: Statistical detection (CBD)
        - Layer 2: Reputation-weighted filtering
        - Layer 3: Robust aggregation (Multi-Krum/Trimmed Mean)

        Args:
            gradients: List of (node_id, gradient_fp) tuples
                       where gradient_fp is fixed-point (Q16.16) integers
            round_num: Training round number
            reputation_scores: Optional list of (node_id, reputation) tuples
                               If None, uses default reputation of 0.5

        Returns:
            DefenseResult with aggregated gradient and detection info
        """
        start_time = time.time()

        # Convert to GradientVector format expected by zome
        # The zome expects AgentPubKey but we use node_id strings
        # The conductor will handle the conversion
        gradient_vectors = [
            {
                "values": grad,
                "node_id": nid,  # Will be converted to AgentPubKey
                "round": round_num
            }
            for nid, grad in gradients
        ]

        # Prepare reputation scores (default to 0.5 if not provided)
        fp_scale = 65536
        if reputation_scores is None:
            rep_scores = [
                (nid, int(0.5 * fp_scale))  # Default reputation
                for nid, _ in gradients
            ]
        else:
            rep_scores = [
                (nid, int(rep * fp_scale))
                for nid, rep in reputation_scores
            ]

        # Build ProcessRoundInput
        input_data = {
            "round": round_num,
            "gradients": gradient_vectors,
            "config": None,  # Use default config
            "reputation_scores": rep_scores
        }

        result = await self.call_zome(
            "defense_coordinator",
            "process_round",
            input_data
        )

        processing_time = (time.time() - start_time) * 1000

        # Parse DefenseResult from zome
        return DefenseResult(
            aggregated_gradient=result.get("aggregated_gradient", []),
            method_used=AggregationMethod(result.get("aggregation_method", 0)),
            detected_byzantine=result.get("flagged_nodes", []),
            layer1_flagged=result.get("layer1_flagged", []),
            layer2_flagged=result.get("layer2_flagged", []),
            confidence=result.get("confidence", 0.0) / fp_scale,
            processing_time_ms=processing_time
        )

    async def analyze_gradient(
        self,
        node_id: str,
        gradient: List[int],
        all_gradients: List[List[int]],
        round_num: int
    ) -> CBDAnalysisResult:
        """
        Analyze a single gradient for Byzantine behavior.

        Uses CBD (Causal Byzantine Detection) from the defense_coordinator.
        """
        input_data = {
            "node_id": node_id,
            "gradient": gradient,
            "all_gradients": all_gradients,
            "round_num": round_num
        }

        result = await self.call_zome(
            "defense_coordinator",
            "analyze_gradient_cbd",
            input_data
        )

        return CBDAnalysisResult(
            node_id=result.get("node_id", node_id),
            z_score=result.get("z_score", 0.0),
            is_byzantine=result.get("is_byzantine", False),
            causal_evidence_strength=result.get("causal_evidence", 0.0),
            reasoning=result.get("reasoning", "")
        )

    async def update_reputation(
        self,
        node_id: str,
        contribution_quality: float,
        was_byzantine: bool
    ):
        """Update a node's reputation after a round."""
        input_data = {
            "node_id": node_id,
            "contribution_quality": contribution_quality,
            "was_byzantine": was_byzantine
        }

        await self.call_zome(
            "defense_coordinator",
            "update_reputation_entry",
            input_data
        )

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get a node's current reputation."""
        result = await self.call_zome(
            "defense_coordinator",
            "get_reputation",
            {"node_id": node_id}
        )
        return result or {}

    async def aggregate_gradients(
        self,
        gradients: List[List[int]],
        method: AggregationMethod = AggregationMethod.MULTI_KRUM,
        weights: Optional[List[float]] = None
    ) -> List[int]:
        """
        Aggregate gradients using specified method.

        Args:
            gradients: List of fixed-point gradients
            method: Aggregation method to use
            weights: Optional reputation-based weights

        Returns:
            Aggregated gradient in fixed-point format
        """
        input_data = {
            "gradients": gradients,
            "method": method.value,
            "weights": weights
        }

        result = await self.call_zome(
            "defense_coordinator",
            "aggregate_with_method",
            input_data
        )

        return result.get("aggregated", [])

    async def store_gradient(
        self,
        node_id: str,
        round_num: int,
        gradient: List[int],
        validation_passed: bool,
        pogq_score: Optional[float] = None
    ) -> str:
        """
        Store a gradient in the DHT.

        Returns:
            Entry hash of stored gradient
        """
        input_data = {
            "node_id": node_id,
            "round_num": round_num,
            "gradient_data": base64.b64encode(
                msgpack.packb(gradient)
            ).decode('utf-8'),
            "gradient_shape": [len(gradient)],
            "gradient_dtype": "fixed32",
            "validation_passed": validation_passed,
            "pogq_score": pogq_score
        }

        result = await self.call_zome(
            "gradient_storage",
            "store_gradient",
            input_data
        )

        return result.get("entry_hash", "")


class FederatedLearningCoordinator:
    """
    High-level coordinator for Byzantine-robust federated learning.

    This class orchestrates FL rounds using the Holochain-based
    defense_coordinator for Byzantine resilience.
    """

    def __init__(self, bridge: HolochainBridge):
        """
        Initialize the FL coordinator.

        Args:
            bridge: Connected HolochainBridge instance
        """
        self.bridge = bridge
        self.round_num = 0
        self.fp_scale = 65536  # Q16.16 fixed-point scale

    def float_to_fixed(self, x: float) -> int:
        """Convert float to Q16.16 fixed-point."""
        return int(x * self.fp_scale)

    def fixed_to_float(self, x: int) -> float:
        """Convert Q16.16 fixed-point to float."""
        return x / self.fp_scale

    def gradient_to_fixed(self, gradient: np.ndarray) -> List[int]:
        """Convert numpy gradient to fixed-point list."""
        return [self.float_to_fixed(float(x)) for x in gradient.flatten()]

    def fixed_to_gradient(self, fixed: List[int], shape: tuple) -> np.ndarray:
        """Convert fixed-point list back to numpy gradient."""
        floats = [self.fixed_to_float(x) for x in fixed]
        return np.array(floats).reshape(shape)

    async def run_round(
        self,
        node_gradients: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Run one federated learning round with Byzantine defense.

        Args:
            node_gradients: Dict mapping node_id to gradient (numpy array)

        Returns:
            Tuple of (aggregated_gradient, detected_byzantine_nodes)
        """
        self.round_num += 1

        # Get gradient shape from first entry
        first_grad = next(iter(node_gradients.values()))
        grad_shape = first_grad.shape

        # Convert to fixed-point for Holochain
        gradients_fp = [
            (node_id, self.gradient_to_fixed(grad))
            for node_id, grad in node_gradients.items()
        ]

        # Run defense through Holochain
        result = await self.bridge.run_defense(gradients_fp, self.round_num)

        # Convert result back to numpy
        aggregated = self.fixed_to_gradient(result.aggregated_gradient, grad_shape)

        print(f"Round {self.round_num}: "
              f"detected {len(result.detected_byzantine)} Byzantine nodes, "
              f"method={result.method_used.name}, "
              f"time={result.processing_time_ms:.1f}ms")

        return aggregated, result.detected_byzantine


# ========== Utility Functions ==========

async def create_connected_bridge(
    app_port: int = DEFAULT_APP_PORT,
    app_id: str = "byzantine-fl-agent1"
) -> HolochainBridge:
    """Create and connect a HolochainBridge instance."""
    bridge = HolochainBridge(app_port=app_port, app_id=app_id)
    await bridge.connect()
    return bridge


# ========== Demo / Test ==========

async def demo():
    """Demonstrate the bridge functionality."""
    print("="*60)
    print("HOLOCHAIN BRIDGE DEMO")
    print("="*60)

    try:
        # Create and connect bridge
        bridge = await create_connected_bridge()

        # Create FL coordinator
        fl = FederatedLearningCoordinator(bridge)

        # Simulate gradients from 5 nodes (3 honest, 2 Byzantine)
        np.random.seed(42)
        honest_gradient = np.random.randn(100) * 0.1  # Small honest gradient

        gradients = {
            "node1": honest_gradient + np.random.randn(100) * 0.01,
            "node2": honest_gradient + np.random.randn(100) * 0.01,
            "node3": honest_gradient + np.random.randn(100) * 0.01,
            "node4": np.random.randn(100) * 10,  # Byzantine: large random
            "node5": -honest_gradient * 5,  # Byzantine: sign flip attack
        }

        # Run FL round
        aggregated, detected = await fl.run_round(gradients)

        print(f"\nAggregated gradient shape: {aggregated.shape}")
        print(f"Aggregated mean: {np.mean(aggregated):.6f}")
        print(f"Detected Byzantine nodes: {detected}")
        print(f"Expected Byzantine: ['node4', 'node5']")

        await bridge.close()
        print("\n✅ Demo complete!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nMake sure the Holochain conductor is running with:")
        print("  - Admin port: 9001")
        print("  - App port: 9002")
        print("  - byzantine-fl-agent1 app installed")


if __name__ == "__main__":
    asyncio.run(demo())
