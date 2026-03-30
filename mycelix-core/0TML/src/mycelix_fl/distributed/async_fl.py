# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Async Federated Learning

Asynchronous FL operations for distributed scenarios.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod

import numpy as np


class MessageType(Enum):
    """Types of FL messages."""
    GRADIENT = "gradient"
    AGGREGATION = "aggregation"
    HEARTBEAT = "heartbeat"
    ROUND_START = "round_start"
    ROUND_END = "round_end"


@dataclass
class GradientMessage:
    """Message containing a gradient from a node."""
    node_id: str
    gradient: np.ndarray
    round_num: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationMessage:
    """Message containing aggregated result."""
    round_num: int
    aggregated_gradient: np.ndarray
    participating_nodes: List[str]
    byzantine_nodes: Set[str]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncFLConfig:
    """Configuration for async FL."""
    # Timing
    round_timeout_seconds: float = 60.0
    heartbeat_interval_seconds: float = 5.0
    gradient_collection_timeout: float = 30.0

    # Aggregation
    min_nodes_per_round: int = 3
    max_stragglers: int = 2
    aggregation_strategy: str = "wait_for_threshold"  # or "timeout"

    # Byzantine handling
    byzantine_threshold: float = 0.45
    use_detection: bool = True
    use_healing: bool = True

    # Backoff
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class NodeConnection(ABC):
    """Abstract base class for node connections."""

    @abstractmethod
    async def send_gradient(self, gradient: np.ndarray, round_num: int) -> bool:
        """Send gradient to coordinator."""
        pass

    @abstractmethod
    async def receive_aggregation(self) -> AggregationMessage:
        """Receive aggregated result from coordinator."""
        pass

    @abstractmethod
    async def heartbeat(self) -> bool:
        """Send heartbeat to coordinator."""
        pass


class FLProtocol:
    """
    Protocol handler for FL communication.

    Manages message serialization and protocol logic.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.message_handlers: Dict[MessageType, Callable] = {}

    def register_handler(self, msg_type: MessageType, handler: Callable) -> None:
        """Register a message handler."""
        self.message_handlers[msg_type] = handler

    def serialize_gradient(self, gradient: np.ndarray, round_num: int) -> bytes:
        """Serialize a gradient message."""
        import struct

        # Simple binary format: [round_num (4 bytes)] [shape (4 bytes each)] [data]
        header = struct.pack(">I", round_num)
        shape = struct.pack(">I", len(gradient.shape)) + b"".join(
            struct.pack(">I", d) for d in gradient.shape
        )
        data = gradient.astype(np.float32).tobytes()
        return header + shape + data

    def deserialize_gradient(self, data: bytes) -> GradientMessage:
        """Deserialize a gradient message."""
        import struct

        round_num = struct.unpack(">I", data[:4])[0]
        ndims = struct.unpack(">I", data[4:8])[0]
        shape = tuple(
            struct.unpack(">I", data[8 + i * 4 : 12 + i * 4])[0] for i in range(ndims)
        )
        offset = 8 + ndims * 4
        gradient = np.frombuffer(data[offset:], dtype=np.float32).reshape(shape)

        return GradientMessage(
            node_id=self.node_id,
            gradient=gradient,
            round_num=round_num,
        )


class AsyncMycelixFL:
    """
    Asynchronous Federated Learning Coordinator.

    Handles async gradient collection and aggregation.
    """

    def __init__(self, config: Optional[AsyncFLConfig] = None):
        from mycelix_fl import MycelixFL, FLConfig

        self.config = config or AsyncFLConfig()
        self.fl_config = FLConfig(
            byzantine_threshold=self.config.byzantine_threshold,
            use_detection=self.config.use_detection,
            use_healing=self.config.use_healing,
        )
        self.fl = MycelixFL(config=self.fl_config)

        # State
        self.current_round = 0
        self.pending_gradients: Dict[str, GradientMessage] = {}
        self.connected_nodes: Set[str] = set()
        self.lock = asyncio.Lock()

    async def start_round(self, round_num: int) -> None:
        """Start a new FL round."""
        async with self.lock:
            self.current_round = round_num
            self.pending_gradients = {}

    async def submit_gradient(
        self,
        node_id: str,
        gradient: np.ndarray,
        round_num: int,
    ) -> bool:
        """
        Submit a gradient from a node.

        Args:
            node_id: ID of the submitting node
            gradient: Gradient array
            round_num: Round number

        Returns:
            True if accepted, False if rejected
        """
        async with self.lock:
            if round_num != self.current_round:
                return False

            self.pending_gradients[node_id] = GradientMessage(
                node_id=node_id,
                gradient=gradient,
                round_num=round_num,
            )
            return True

    async def can_aggregate(self) -> bool:
        """Check if we have enough gradients to aggregate."""
        return len(self.pending_gradients) >= self.config.min_nodes_per_round

    async def aggregate(self) -> Optional[AggregationMessage]:
        """
        Perform aggregation on collected gradients.

        Returns:
            Aggregation result or None if not enough gradients
        """
        async with self.lock:
            if len(self.pending_gradients) < self.config.min_nodes_per_round:
                return None

            # Convert to format expected by FL
            gradients = {
                msg.node_id: msg.gradient
                for msg in self.pending_gradients.values()
            }

            # Run aggregation (blocking, but in practice this is fast)
            result = self.fl.execute_round(gradients, round_num=self.current_round)

            return AggregationMessage(
                round_num=self.current_round,
                aggregated_gradient=result.aggregated_gradient,
                participating_nodes=result.participating_nodes,
                byzantine_nodes=result.byzantine_nodes,
            )

    async def run_round_with_timeout(
        self,
        expected_nodes: Set[str],
    ) -> Optional[AggregationMessage]:
        """
        Run a complete round with timeout.

        Args:
            expected_nodes: Set of expected node IDs

        Returns:
            Aggregation result or None if timeout/failure
        """
        start_time = time.time()
        await self.start_round(self.current_round + 1)

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > self.config.round_timeout_seconds:
                break

            # Check if we can aggregate
            if await self.can_aggregate():
                received = set(self.pending_gradients.keys())
                missing = expected_nodes - received

                # Wait for stragglers or timeout
                if len(missing) <= self.config.max_stragglers:
                    break

            await asyncio.sleep(0.1)

        return await self.aggregate()

    async def wait_for_gradients(
        self,
        min_count: int,
        timeout: Optional[float] = None,
    ) -> Dict[str, GradientMessage]:
        """
        Wait for a minimum number of gradients.

        Args:
            min_count: Minimum number of gradients needed
            timeout: Timeout in seconds (None = use config)

        Returns:
            Dict of received gradients
        """
        timeout = timeout or self.config.gradient_collection_timeout
        start_time = time.time()

        while True:
            if len(self.pending_gradients) >= min_count:
                return dict(self.pending_gradients)

            if time.time() - start_time > timeout:
                return dict(self.pending_gradients)

            await asyncio.sleep(0.05)


class AsyncNode:
    """
    Async FL node (client).

    Handles gradient computation and submission.
    """

    def __init__(
        self,
        node_id: str,
        coordinator: AsyncMycelixFL,
    ):
        self.node_id = node_id
        self.coordinator = coordinator
        self.current_model: Optional[np.ndarray] = None

    async def train_and_submit(
        self,
        compute_gradient: Callable[[], np.ndarray],
        round_num: int,
    ) -> bool:
        """
        Compute gradient and submit to coordinator.

        Args:
            compute_gradient: Function that computes and returns gradient
            round_num: Current round number

        Returns:
            True if submission accepted
        """
        # Compute gradient (could be async in real scenario)
        gradient = compute_gradient()

        # Submit to coordinator
        return await self.coordinator.submit_gradient(
            node_id=self.node_id,
            gradient=gradient,
            round_num=round_num,
        )

    async def apply_aggregation(self, message: AggregationMessage) -> None:
        """Apply aggregated gradient to model."""
        if self.current_model is not None:
            self.current_model = self.current_model - 0.01 * message.aggregated_gradient
