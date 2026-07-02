# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Event Listener for Federated Learning.

Subscribes to DHT signals from Holochain conductor for:
- Gradient broadcasts from participants
- Reputation updates
- Byzantine detection alerts
- Round coordination signals

Integrates with the FL aggregator to process incoming gradients.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

try:
    import websockets
except ImportError:
    websockets = None

try:
    import msgpack
except ImportError:
    msgpack = None

from ..observability.logging import get_logger, FLLogger
from ..security.input_validator import GradientValidator, ValidationError

logger = get_logger(__name__)


@dataclass
class GradientSignal:
    """Represents a gradient signal from Holochain."""
    round_id: int
    participant_id: str
    gradient_hash: str
    dimension: int
    timestamp: float
    signature: Optional[str] = None
    raw_gradient: Optional[List[float]] = None


@dataclass
class ReputationSignal:
    """Represents a reputation update signal."""
    participant_id: str
    old_score: float
    new_score: float
    reason: str
    timestamp: float


@dataclass
class RoundSignal:
    """Represents a round coordination signal."""
    round_id: int
    action: str  # "start", "end", "aggregate"
    participants: List[str] = field(default_factory=list)
    timestamp: float = 0.0


class HolochainEventListener:
    """
    Listens for Holochain DHT signals and dispatches to handlers.
    
    Provides:
    - Signal subscription and routing
    - Gradient validation before processing
    - Reputation verification
    - Rate limiting per participant
    """
    
    def __init__(
        self,
        app_url: str = "ws://localhost:8889",
        cell_id: Optional[bytes] = None,
        gradient_validator: Optional[GradientValidator] = None,
        min_reputation: float = 0.0,
    ):
        """
        Initialize the event listener.
        
        Args:
            app_url: Holochain app WebSocket URL
            cell_id: Cell ID to listen to (auto-detected if None)
            gradient_validator: Validator for incoming gradients
            min_reputation: Minimum reputation to accept gradients
        """
        self.app_url = app_url
        self.cell_id = cell_id
        self.gradient_validator = gradient_validator or GradientValidator()
        self.min_reputation = min_reputation
        
        self.ws = None
        self.running = False
        self._handlers: Dict[str, List[Callable]] = {
            "gradient": [],
            "reputation": [],
            "round": [],
            "byzantine": [],
        }
        self._pending_gradients: Dict[int, Dict[str, GradientSignal]] = {}
        self._participant_reputation: Dict[str, float] = {}
        self._processed_hashes: Set[str] = set()
        
        self.fl_logger = FLLogger("holochain_listener")
    
    def on_gradient(self, handler: Callable[[GradientSignal], None]) -> None:
        """Register a gradient signal handler."""
        self._handlers["gradient"].append(handler)
    
    def on_reputation(self, handler: Callable[[ReputationSignal], None]) -> None:
        """Register a reputation update handler."""
        self._handlers["reputation"].append(handler)
    
    def on_round(self, handler: Callable[[RoundSignal], None]) -> None:
        """Register a round coordination handler."""
        self._handlers["round"].append(handler)
    
    def on_byzantine(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a Byzantine detection handler."""
        self._handlers["byzantine"].append(handler)
    
    async def connect(self) -> bool:
        """Connect to Holochain app WebSocket."""
        if not websockets or not msgpack:
            logger.error("websockets and msgpack required for Holochain listener")
            return False
        
        try:
            logger.info(f"Connecting to Holochain at {self.app_url}")
            self.ws = await websockets.connect(
                self.app_url,
                additional_headers={"Origin": "http://localhost"},
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Connected to Holochain conductor")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Holochain: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Holochain."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("Disconnected from Holochain")
    
    async def start_listening(self) -> None:
        """Start listening for signals."""
        if not self.ws:
            if not await self.connect():
                raise RuntimeError("Cannot start listening: not connected")
        
        self.running = True
        logger.info("Started listening for Holochain signals")
        
        try:
            async for message in self.ws:
                if not self.running:
                    break
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Holochain connection closed")
        except Exception as e:
            logger.error(f"Error in signal listener: {e}", exc_info=True)
        finally:
            self.running = False
    
    async def _process_message(self, message: bytes) -> None:
        """Process an incoming WebSocket message."""
        try:
            # Decode msgpack
            data = msgpack.unpackb(message, raw=False)
            
            if not isinstance(data, dict):
                return
            
            msg_type = data.get("type")
            
            if msg_type == "signal":
                await self._handle_signal(data.get("data", {}))
            elif msg_type == "app_signal":
                await self._handle_app_signal(data.get("signal", {}))
        except Exception as e:
            logger.debug(f"Failed to process message: {e}")
    
    async def _handle_signal(self, signal_data: Dict[str, Any]) -> None:
        """Handle a raw signal from the conductor."""
        signal_type = signal_data.get("signal_type")
        payload = signal_data.get("payload", {})
        
        if signal_type == "GradientSubmitted":
            await self._handle_gradient_signal(payload)
        elif signal_type == "ReputationUpdated":
            await self._handle_reputation_signal(payload)
        elif signal_type == "RoundEvent":
            await self._handle_round_signal(payload)
        elif signal_type == "ByzantineDetected":
            await self._handle_byzantine_signal(payload)
    
    async def _handle_app_signal(self, signal: Dict[str, Any]) -> None:
        """Handle an app-level signal."""
        # App signals have a different structure
        if "gradient" in signal:
            await self._handle_gradient_signal(signal["gradient"])
        elif "reputation" in signal:
            await self._handle_reputation_signal(signal["reputation"])
    
    async def _handle_gradient_signal(self, payload: Dict[str, Any]) -> None:
        """Process an incoming gradient signal."""
        try:
            participant_id = payload.get("participant_id", "")
            round_id = payload.get("round_id", 0)
            gradient_data = payload.get("gradient", [])
            
            # Compute hash for deduplication
            gradient_hash = hashlib.sha256(
                json.dumps(gradient_data, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            # Check for duplicate
            if gradient_hash in self._processed_hashes:
                logger.debug(f"Duplicate gradient {gradient_hash} from {participant_id}")
                return
            
            # Check reputation
            reputation = self._participant_reputation.get(participant_id, 1.0)
            if reputation < self.min_reputation:
                self.fl_logger.gradient_rejected(
                    round_id, participant_id, f"Low reputation: {reputation}"
                )
                return
            
            # Validate gradient
            try:
                validated = self.gradient_validator.validate(gradient_data)
            except ValidationError as e:
                self.fl_logger.gradient_rejected(round_id, participant_id, str(e))
                return
            
            # Create signal
            signal = GradientSignal(
                round_id=round_id,
                participant_id=participant_id,
                gradient_hash=gradient_hash,
                dimension=len(validated),
                timestamp=payload.get("timestamp", asyncio.get_event_loop().time()),
                signature=payload.get("signature"),
                raw_gradient=validated,
            )
            
            # Track for deduplication
            self._processed_hashes.add(gradient_hash)
            if len(self._processed_hashes) > 10000:
                # Prune old hashes
                self._processed_hashes = set(list(self._processed_hashes)[-5000:])
            
            # Store pending gradient
            if round_id not in self._pending_gradients:
                self._pending_gradients[round_id] = {}
            self._pending_gradients[round_id][participant_id] = signal
            
            # Log and dispatch
            self.fl_logger.gradient_submitted(round_id, participant_id, len(validated))
            
            for handler in self._handlers["gradient"]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    logger.error(f"Gradient handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to handle gradient signal: {e}")
    
    async def _handle_reputation_signal(self, payload: Dict[str, Any]) -> None:
        """Process a reputation update signal."""
        try:
            participant_id = payload.get("participant_id", "")
            old_score = payload.get("old_score", 0.0)
            new_score = payload.get("new_score", 0.0)
            
            # Update local cache
            self._participant_reputation[participant_id] = new_score
            
            signal = ReputationSignal(
                participant_id=participant_id,
                old_score=old_score,
                new_score=new_score,
                reason=payload.get("reason", ""),
                timestamp=payload.get("timestamp", asyncio.get_event_loop().time()),
            )
            
            self.fl_logger.reputation_updated(participant_id, old_score, new_score)
            
            for handler in self._handlers["reputation"]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    logger.error(f"Reputation handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to handle reputation signal: {e}")
    
    async def _handle_round_signal(self, payload: Dict[str, Any]) -> None:
        """Process a round coordination signal."""
        try:
            signal = RoundSignal(
                round_id=payload.get("round_id", 0),
                action=payload.get("action", ""),
                participants=payload.get("participants", []),
                timestamp=payload.get("timestamp", asyncio.get_event_loop().time()),
            )
            
            if signal.action == "start":
                self.fl_logger.round_started(signal.round_id, len(signal.participants))
            
            for handler in self._handlers["round"]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    logger.error(f"Round handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to handle round signal: {e}")
    
    async def _handle_byzantine_signal(self, payload: Dict[str, Any]) -> None:
        """Process a Byzantine detection signal."""
        try:
            participant_ids = payload.get("participant_ids", [])
            method = payload.get("detection_method", "unknown")
            round_id = payload.get("round_id", 0)
            
            self.fl_logger.byzantine_detected(round_id, participant_ids, method)
            
            # Update reputation cache
            for pid in participant_ids:
                self._participant_reputation[pid] = 0.0
            
            for handler in self._handlers["byzantine"]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)
                except Exception as e:
                    logger.error(f"Byzantine handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to handle byzantine signal: {e}")
    
    def get_pending_gradients(self, round_id: int) -> Dict[str, GradientSignal]:
        """Get all pending gradients for a round."""
        return self._pending_gradients.get(round_id, {})
    
    def clear_round(self, round_id: int) -> None:
        """Clear pending gradients for a completed round."""
        if round_id in self._pending_gradients:
            del self._pending_gradients[round_id]
    
    def get_participant_reputation(self, participant_id: str) -> float:
        """Get cached reputation for a participant."""
        return self._participant_reputation.get(participant_id, 1.0)
    
    def set_participant_reputation(self, participant_id: str, score: float) -> None:
        """Set cached reputation for a participant."""
        self._participant_reputation[participant_id] = score


async def create_gradient_aggregator_integration(
    listener: HolochainEventListener,
    aggregator: Any,  # FL aggregator instance
    ethereum_client: Any = None,  # Ethereum client for reputation lookup
) -> None:
    """
    Create integration between Holochain listener and FL aggregator.
    
    This connects the Holochain DHT signals to the FL aggregation pipeline.
    """
    
    async def on_gradient(signal: GradientSignal):
        """Handle incoming gradient."""
        if signal.raw_gradient:
            # Submit to aggregator
            await aggregator.submit_gradient(
                round_id=signal.round_id,
                node_id=signal.participant_id,
                gradient=signal.raw_gradient,
            )
    
    async def on_round_end(signal: RoundSignal):
        """Handle round end - trigger aggregation."""
        if signal.action == "aggregate":
            await aggregator.aggregate_round(signal.round_id)
    
    async def on_byzantine(data: Dict[str, Any]):
        """Handle Byzantine detection - report to chain."""
        if ethereum_client:
            # Report Byzantine nodes to ReputationAnchor contract
            for node_id in data.get("participant_ids", []):
                try:
                    await ethereum_client.report_byzantine(node_id, data.get("round_id", 0))
                except Exception as e:
                    logger.error(f"Failed to report Byzantine node to chain: {e}")
    
    # Register handlers
    listener.on_gradient(on_gradient)
    listener.on_round(on_round_end)
    listener.on_byzantine(on_byzantine)
    
    logger.info("Holochain-Aggregator integration configured")
