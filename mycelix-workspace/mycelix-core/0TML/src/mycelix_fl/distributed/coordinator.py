# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FL Coordinator

Central coordinator for distributed federated learning.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np


class RoundState(Enum):
    """State of an FL round."""
    PENDING = "pending"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class CoordinatorConfig:
    """Configuration for FL coordinator."""
    # Round settings
    num_rounds: int = 100
    min_nodes: int = 3
    max_nodes: Optional[int] = None

    # Timing
    round_timeout_seconds: float = 120.0
    collection_timeout_seconds: float = 60.0

    # Byzantine
    byzantine_threshold: float = 0.45
    use_detection: bool = True
    use_healing: bool = True

    # Callbacks
    on_round_start: Optional[Callable[[int], None]] = None
    on_round_end: Optional[Callable[[int, Any], None]] = None
    on_node_join: Optional[Callable[[str], None]] = None
    on_node_leave: Optional[Callable[[str], None]] = None


@dataclass
class RoundInfo:
    """Information about a single round."""
    round_num: int
    state: RoundState = RoundState.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    participating_nodes: List[str] = field(default_factory=list)
    byzantine_nodes: Set[str] = field(default_factory=set)
    healed_nodes: Set[str] = field(default_factory=set)
    aggregated_gradient: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class FLCoordinator:
    """
    Federated Learning Coordinator.

    Manages distributed FL training across multiple nodes.
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        from mycelix_fl import MycelixFL, FLConfig

        self.config = config or CoordinatorConfig()

        # Create FL instance
        fl_config = FLConfig(
            num_rounds=self.config.num_rounds,
            min_nodes=self.config.min_nodes,
            byzantine_threshold=self.config.byzantine_threshold,
            use_detection=self.config.use_detection,
            use_healing=self.config.use_healing,
        )
        self.fl = MycelixFL(config=fl_config)

        # State
        self.current_round = 0
        self.round_history: List[RoundInfo] = []
        self.registered_nodes: Set[str] = set()
        self.node_gradients: Dict[str, np.ndarray] = {}
        self.lock = asyncio.Lock()

    def register_node(self, node_id: str) -> bool:
        """Register a new node."""
        if self.config.max_nodes and len(self.registered_nodes) >= self.config.max_nodes:
            return False

        self.registered_nodes.add(node_id)

        if self.config.on_node_join:
            self.config.on_node_join(node_id)

        return True

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node."""
        self.registered_nodes.discard(node_id)

        if self.config.on_node_leave:
            self.config.on_node_leave(node_id)

    async def start_round(self) -> RoundInfo:
        """Start a new FL round."""
        async with self.lock:
            self.current_round += 1
            round_info = RoundInfo(
                round_num=self.current_round,
                state=RoundState.COLLECTING,
                start_time=time.time(),
            )
            self.node_gradients = {}

            if self.config.on_round_start:
                self.config.on_round_start(self.current_round)

            return round_info

    async def submit_gradient(
        self,
        node_id: str,
        gradient: np.ndarray,
    ) -> bool:
        """Submit a gradient from a node."""
        async with self.lock:
            if node_id not in self.registered_nodes:
                return False

            self.node_gradients[node_id] = gradient
            return True

    async def complete_round(self) -> RoundInfo:
        """Complete the current round with aggregation."""
        async with self.lock:
            round_info = RoundInfo(
                round_num=self.current_round,
                state=RoundState.AGGREGATING,
                start_time=self.round_history[-1].start_time if self.round_history else time.time(),
            )

            if len(self.node_gradients) < self.config.min_nodes:
                round_info.state = RoundState.FAILED
                round_info.error = f"Insufficient nodes: {len(self.node_gradients)} < {self.config.min_nodes}"
                round_info.end_time = time.time()
                self.round_history.append(round_info)
                return round_info

            try:
                # Run aggregation
                result = self.fl.execute_round(
                    self.node_gradients,
                    round_num=self.current_round,
                )

                round_info.state = RoundState.COMPLETE
                round_info.participating_nodes = result.participating_nodes
                round_info.byzantine_nodes = result.byzantine_nodes
                round_info.healed_nodes = result.healed_nodes
                round_info.aggregated_gradient = result.aggregated_gradient
                round_info.end_time = time.time()

                if self.config.on_round_end:
                    self.config.on_round_end(self.current_round, result)

            except Exception as e:
                round_info.state = RoundState.FAILED
                round_info.error = str(e)
                round_info.end_time = time.time()

            self.round_history.append(round_info)
            return round_info

    async def run_round(
        self,
        get_gradients: Callable[[], Dict[str, np.ndarray]],
    ) -> RoundInfo:
        """
        Run a complete round.

        Args:
            get_gradients: Callback to get gradients from nodes

        Returns:
            Round information
        """
        await self.start_round()

        # Collect gradients (with timeout)
        try:
            gradients = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, get_gradients),
                timeout=self.config.collection_timeout_seconds,
            )
            for node_id, gradient in gradients.items():
                await self.submit_gradient(node_id, gradient)
        except asyncio.TimeoutError:
            pass

        return await self.complete_round()

    async def run_training(
        self,
        get_gradients: Callable[[int], Dict[str, np.ndarray]],
        num_rounds: Optional[int] = None,
    ) -> List[RoundInfo]:
        """
        Run full training loop.

        Args:
            get_gradients: Callback that takes round number and returns gradients
            num_rounds: Number of rounds (None = use config)

        Returns:
            List of round info
        """
        num_rounds = num_rounds or self.config.num_rounds
        results = []

        for round_num in range(1, num_rounds + 1):
            # Get gradients for this round
            async def get_round_gradients():
                return get_gradients(round_num)

            round_info = await self.run_round(lambda: get_gradients(round_num))
            results.append(round_info)

            if round_info.state == RoundState.FAILED:
                break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        completed = [r for r in self.round_history if r.state == RoundState.COMPLETE]
        failed = [r for r in self.round_history if r.state == RoundState.FAILED]

        durations = [r.duration_seconds for r in completed if r.duration_seconds]
        byzantine_counts = [len(r.byzantine_nodes) for r in completed]

        return {
            "total_rounds": len(self.round_history),
            "completed_rounds": len(completed),
            "failed_rounds": len(failed),
            "registered_nodes": len(self.registered_nodes),
            "avg_round_duration_seconds": np.mean(durations) if durations else 0,
            "avg_byzantine_per_round": np.mean(byzantine_counts) if byzantine_counts else 0,
        }
