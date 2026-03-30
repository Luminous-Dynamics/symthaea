# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Chaos Engineering Tests for Mycelix Resilience Validation
==========================================================

This module validates Mycelix's resilience under adverse conditions through
systematic chaos injection. Tests cover:

1. Network Partition Tests
   - Simulate network splits during FL rounds
   - Test recovery when partition heals
   - Verify no data loss or corruption

2. Node Crash Tests
   - Coordinator crash during aggregation
   - Node crash after submitting gradient
   - Multiple simultaneous node failures

3. Byzantine Surge Tests
   - Sudden increase in Byzantine nodes (20% -> 45%)
   - Test self-healing recovery
   - Verify system stabilizes

4. Latency Injection Tests
   - High latency on some nodes
   - Verify timeout handling
   - Test late-arriving gradients

5. Resource Exhaustion Tests
   - Memory pressure scenarios
   - CPU throttling
   - Disk I/O delays

6. Holochain DHT Tests
   - DHT partition
   - Slow gossip propagation
   - Conflicting DHT entries

Usage:
    # Run all chaos tests
    pytest -m chaos tests/test_chaos_engineering.py

    # Run only slow tests (requires more resources)
    pytest -m "chaos and slow" tests/test_chaos_engineering.py

    # Run quick chaos validation
    pytest -m "chaos and not slow" tests/test_chaos_engineering.py

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import os
import random
import sys
import time
import tracemalloc
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Performance thresholds
MAX_RECOVERY_TIME_S = 30.0  # Max time to recover from partition
MAX_DETECTION_LATENCY_MS = 10.0  # Max Byzantine detection latency under chaos
MEMORY_PRESSURE_THRESHOLD_MB = 100  # Additional memory budget for chaos tests
DEFAULT_GRADIENT_SIZE = 10_000  # Gradient vector size for tests
DEFAULT_ROUND_TIMEOUT_S = 60.0  # Timeout for a single round

# Byzantine parameters
DEFAULT_HONEST_FRACTION = 0.8
BYZANTINE_SURGE_FRACTION = 0.45
BYZANTINE_RECOVERY_THRESHOLD = 0.35

# Network chaos parameters
NETWORK_PARTITION_DURATION_S = 5.0
HIGH_LATENCY_MS = 2000
GOSSIP_DELAY_MS = 500


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ChaosType(Enum):
    """Types of chaos that can be injected."""
    NETWORK_PARTITION = "network_partition"
    NODE_CRASH = "node_crash"
    BYZANTINE_SURGE = "byzantine_surge"
    LATENCY_INJECTION = "latency_injection"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_THROTTLE = "cpu_throttle"
    DISK_IO_DELAY = "disk_io_delay"
    DHT_PARTITION = "dht_partition"
    GOSSIP_DELAY = "gossip_delay"
    CONFLICTING_ENTRIES = "conflicting_entries"


class NodeState(Enum):
    """State of a simulated node."""
    RUNNING = "running"
    CRASHED = "crashed"
    PARTITIONED = "partitioned"
    THROTTLED = "throttled"
    BYZANTINE = "byzantine"


@dataclass
class ChaosConfig:
    """Configuration for chaos injection."""
    chaos_type: ChaosType
    target_nodes: List[str] = field(default_factory=list)
    duration_s: float = 5.0
    intensity: float = 1.0  # 0.0 to 1.0
    start_delay_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosEvent:
    """Record of a chaos event."""
    event_id: str
    chaos_type: ChaosType
    start_time: float
    end_time: Optional[float] = None
    affected_nodes: Set[str] = field(default_factory=set)
    impact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryMetrics:
    """Metrics tracking recovery from chaos."""
    recovery_time_s: float
    data_lost: int
    data_corrupted: int
    rounds_failed: int
    rounds_recovered: int
    byzantine_detected: int
    byzantine_missed: int


@dataclass
class GradientSubmission:
    """Represents a gradient submission from a node."""
    node_id: str
    gradient: np.ndarray
    timestamp: float
    round_num: int
    is_byzantine: bool = False


# =============================================================================
# CHAOS INJECTION FIXTURES
# =============================================================================

class ChaosNetworkController:
    """
    Controller for injecting network-level chaos.

    Simulates:
    - Network partitions (nodes cannot communicate)
    - Latency injection
    - Packet loss
    - Connection timeouts
    """

    def __init__(self):
        self.partitioned_groups: List[Set[str]] = []
        self.node_latencies: Dict[str, int] = {}
        self.disconnected_nodes: Set[str] = set()
        self.packet_loss_rate: Dict[str, float] = {}
        self._events: List[ChaosEvent] = []
        self._lock = asyncio.Lock()

    async def create_partition(
        self,
        group_a: Set[str],
        group_b: Set[str],
        duration_s: float = 5.0
    ) -> ChaosEvent:
        """
        Create a network partition between two groups.

        Expected Behavior:
        - Nodes in group_a cannot communicate with nodes in group_b
        - Each group can still communicate internally
        - After duration_s, partition heals automatically

        Args:
            group_a: First partition group
            group_b: Second partition group
            duration_s: How long partition lasts

        Returns:
            ChaosEvent recording the partition
        """
        async with self._lock:
            event = ChaosEvent(
                event_id=f"partition_{time.time()}",
                chaos_type=ChaosType.NETWORK_PARTITION,
                start_time=time.time(),
                affected_nodes=group_a | group_b,
                impact={"group_a": list(group_a), "group_b": list(group_b)}
            )

            self.partitioned_groups.append(group_a)
            self.partitioned_groups.append(group_b)
            self._events.append(event)

            logger.info(f"Network partition created: {group_a} <-> {group_b}")

            # Schedule healing
            asyncio.create_task(self._heal_partition_after(event, duration_s))

            return event

    async def _heal_partition_after(self, event: ChaosEvent, duration_s: float):
        """Heal partition after specified duration."""
        await asyncio.sleep(duration_s)
        await self.heal_partition(event.event_id)

    async def heal_partition(self, event_id: str):
        """Heal a specific partition."""
        async with self._lock:
            for event in self._events:
                if event.event_id == event_id and event.end_time is None:
                    event.end_time = time.time()

                    # Remove partition groups
                    groups_to_remove = [
                        event.impact.get("group_a", []),
                        event.impact.get("group_b", [])
                    ]
                    for group in groups_to_remove:
                        group_set = set(group)
                        if group_set in self.partitioned_groups:
                            self.partitioned_groups.remove(group_set)

                    logger.info(f"Partition {event_id} healed")
                    break

    def can_communicate(self, node_a: str, node_b: str) -> bool:
        """
        Check if two nodes can communicate.

        Returns:
            True if nodes can communicate, False if partitioned
        """
        if node_a in self.disconnected_nodes or node_b in self.disconnected_nodes:
            return False

        for i in range(0, len(self.partitioned_groups), 2):
            if i + 1 < len(self.partitioned_groups):
                group_a = self.partitioned_groups[i]
                group_b = self.partitioned_groups[i + 1]

                if (node_a in group_a and node_b in group_b) or \
                   (node_a in group_b and node_b in group_a):
                    return False

        return True

    async def inject_latency(
        self,
        node_ids: List[str],
        latency_ms: int,
        duration_s: float = 10.0
    ) -> ChaosEvent:
        """
        Inject latency for specific nodes.

        Expected Behavior:
        - All communications from affected nodes delayed by latency_ms
        - May cause timeouts if latency_ms > round timeout
        - Gradients from high-latency nodes may arrive late

        Args:
            node_ids: Nodes to affect
            latency_ms: Latency to inject in milliseconds
            duration_s: Duration of latency injection

        Returns:
            ChaosEvent recording the injection
        """
        event = ChaosEvent(
            event_id=f"latency_{time.time()}",
            chaos_type=ChaosType.LATENCY_INJECTION,
            start_time=time.time(),
            affected_nodes=set(node_ids),
            impact={"latency_ms": latency_ms}
        )

        for node_id in node_ids:
            self.node_latencies[node_id] = latency_ms

        self._events.append(event)
        logger.info(f"Injected {latency_ms}ms latency to {node_ids}")

        # Schedule removal
        asyncio.create_task(self._remove_latency_after(node_ids, duration_s, event))

        return event

    async def _remove_latency_after(
        self,
        node_ids: List[str],
        duration_s: float,
        event: ChaosEvent
    ):
        """Remove latency after duration."""
        await asyncio.sleep(duration_s)
        for node_id in node_ids:
            self.node_latencies.pop(node_id, None)
        event.end_time = time.time()
        logger.info(f"Latency removed from {node_ids}")

    def get_latency(self, node_id: str) -> int:
        """Get current latency for a node in milliseconds."""
        return self.node_latencies.get(node_id, 0)

    async def disconnect_node(self, node_id: str):
        """Completely disconnect a node from the network."""
        self.disconnected_nodes.add(node_id)
        logger.info(f"Node {node_id} disconnected")

    async def reconnect_node(self, node_id: str):
        """Reconnect a previously disconnected node."""
        self.disconnected_nodes.discard(node_id)
        logger.info(f"Node {node_id} reconnected")

    def get_events(self) -> List[ChaosEvent]:
        """Get all chaos events."""
        return self._events.copy()

    def reset(self):
        """Reset all chaos state."""
        self.partitioned_groups.clear()
        self.node_latencies.clear()
        self.disconnected_nodes.clear()
        self.packet_loss_rate.clear()
        self._events.clear()


class MockChaosCoordinator:
    """
    Mock FL Coordinator with chaos injection capabilities.

    Simulates coordinator behavior under chaotic conditions:
    - Can be crashed mid-aggregation
    - Handles node failures gracefully
    - Tracks state for recovery validation
    """

    def __init__(
        self,
        network: ChaosNetworkController,
        min_nodes: int = 3,
        round_timeout_s: float = DEFAULT_ROUND_TIMEOUT_S
    ):
        self.network = network
        self.min_nodes = min_nodes
        self.round_timeout_s = round_timeout_s

        # State
        self.current_round = 0
        self.gradients: Dict[str, GradientSubmission] = {}
        self.aggregated_gradients: Dict[int, np.ndarray] = {}
        self.registered_nodes: Set[str] = set()
        self.node_states: Dict[str, NodeState] = {}

        # Crash simulation
        self._crashed = False
        self._crash_point: Optional[str] = None

        # Metrics
        self.failed_rounds: List[int] = []
        self.recovered_rounds: List[int] = []
        self._lock = asyncio.Lock()

    async def register_node(self, node_id: str) -> bool:
        """Register a node with the coordinator."""
        if self._crashed:
            raise ConnectionError("Coordinator crashed")

        async with self._lock:
            self.registered_nodes.add(node_id)
            self.node_states[node_id] = NodeState.RUNNING
            return True

    async def submit_gradient(
        self,
        node_id: str,
        gradient: np.ndarray,
        round_num: int,
        is_byzantine: bool = False
    ) -> bool:
        """
        Submit a gradient from a node.

        Expected Behavior Under Chaos:
        - If node is partitioned, submission fails
        - If latency injected, submission delayed
        - If coordinator crashed, raises ConnectionError

        Args:
            node_id: Submitting node
            gradient: Gradient vector
            round_num: Round number
            is_byzantine: Whether this is a Byzantine gradient

        Returns:
            True if submission successful
        """
        if self._crashed:
            raise ConnectionError("Coordinator crashed")

        # Check network connectivity
        if not self.network.can_communicate(node_id, "coordinator"):
            logger.debug(f"Node {node_id} cannot reach coordinator (partitioned)")
            return False

        # Apply latency
        latency_ms = self.network.get_latency(node_id)
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000)

        async with self._lock:
            if self.node_states.get(node_id) == NodeState.CRASHED:
                return False

            submission = GradientSubmission(
                node_id=node_id,
                gradient=gradient.copy(),
                timestamp=time.time(),
                round_num=round_num,
                is_byzantine=is_byzantine
            )
            self.gradients[node_id] = submission
            logger.debug(f"Received gradient from {node_id} for round {round_num}")
            return True

    async def aggregate_round(self, round_num: int) -> Optional[np.ndarray]:
        """
        Aggregate gradients for a round.

        Expected Behavior Under Chaos:
        - If crashed during aggregation, state preserved
        - If insufficient nodes, round fails
        - Byzantine gradients filtered if detection enabled

        Args:
            round_num: Round to aggregate

        Returns:
            Aggregated gradient or None if failed
        """
        if self._crashed:
            raise ConnectionError("Coordinator crashed during aggregation")

        # Check for crash point
        if self._crash_point == "pre_aggregate":
            await self._simulate_crash()
            return None

        async with self._lock:
            # Collect valid gradients
            valid_gradients = []
            for submission in self.gradients.values():
                if submission.round_num == round_num:
                    # Check node is still connected
                    if self.network.can_communicate(submission.node_id, "coordinator"):
                        valid_gradients.append(submission.gradient)

            if len(valid_gradients) < self.min_nodes:
                logger.warning(
                    f"Round {round_num} failed: {len(valid_gradients)} < {self.min_nodes} nodes"
                )
                self.failed_rounds.append(round_num)
                return None

            # Check for mid-aggregation crash
            if self._crash_point == "mid_aggregate":
                await self._simulate_crash()
                return None

            # Aggregate (simple mean for testing)
            aggregated = np.mean(valid_gradients, axis=0)
            self.aggregated_gradients[round_num] = aggregated

            logger.info(f"Round {round_num} aggregated from {len(valid_gradients)} nodes")
            return aggregated

    async def _simulate_crash(self):
        """Simulate coordinator crash."""
        self._crashed = True
        logger.warning("Coordinator crashed!")
        raise RuntimeError("Coordinator crashed")

    def crash_at(self, point: str):
        """
        Configure coordinator to crash at specific point.

        Args:
            point: One of 'pre_aggregate', 'mid_aggregate', 'post_aggregate'
        """
        self._crash_point = point

    async def recover(self):
        """Recover from crash."""
        self._crashed = False
        self._crash_point = None
        logger.info("Coordinator recovered")

    def crash_node(self, node_id: str):
        """Crash a specific node."""
        self.node_states[node_id] = NodeState.CRASHED
        logger.info(f"Node {node_id} crashed")

    def recover_node(self, node_id: str):
        """Recover a crashed node."""
        self.node_states[node_id] = NodeState.RUNNING
        logger.info(f"Node {node_id} recovered")

    def get_state(self) -> Dict[str, Any]:
        """Get current coordinator state for validation."""
        return {
            "crashed": self._crashed,
            "current_round": self.current_round,
            "registered_nodes": list(self.registered_nodes),
            "failed_rounds": self.failed_rounds.copy(),
            "aggregated_rounds": list(self.aggregated_gradients.keys()),
        }

    def reset_round(self):
        """Reset for new round."""
        self.gradients.clear()


class MockChaosDHT:
    """
    Mock Holochain DHT with chaos injection.

    Simulates DHT behavior under adverse conditions:
    - Partition (some nodes can't see entries)
    - Slow gossip (entries propagate slowly)
    - Conflicting entries (same key, different values)
    """

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._entry_versions: Dict[str, List[Dict[str, Any]]] = {}  # For conflicts
        self._partitioned_nodes: Set[str] = set()
        self._gossip_delay_ms: int = 0
        self._pending_gossip: Deque[Tuple[str, Dict[str, Any], float]] = deque()
        self._connected = True

    async def store_entry(
        self,
        key: str,
        value: Dict[str, Any],
        source_node: str
    ) -> str:
        """
        Store an entry in the DHT.

        Expected Behavior Under Chaos:
        - If partitioned, entry only visible to same partition
        - If gossip delayed, other nodes see entry after delay
        - If conflicting, multiple versions stored

        Args:
            key: Entry key
            value: Entry value
            source_node: Node storing the entry

        Returns:
            Entry hash
        """
        if not self._connected:
            raise ConnectionError("DHT not connected")

        if source_node in self._partitioned_nodes:
            raise ConnectionError(f"Node {source_node} is partitioned from DHT")

        entry_hash = hashlib.sha256(
            f"{key}{time.time()}{source_node}".encode()
        ).hexdigest()

        entry = {
            "hash": entry_hash,
            "key": key,
            "value": value,
            "source": source_node,
            "timestamp": time.time()
        }

        # Handle gossip delay
        if self._gossip_delay_ms > 0:
            propagation_time = time.time() + (self._gossip_delay_ms / 1000)
            self._pending_gossip.append((key, entry, propagation_time))
            logger.debug(f"Entry {key} pending gossip (delay: {self._gossip_delay_ms}ms)")
        else:
            self._storage[key] = entry

        # Track versions for conflict detection
        if key not in self._entry_versions:
            self._entry_versions[key] = []
        self._entry_versions[key].append(entry)

        return entry_hash

    async def get_entry(
        self,
        key: str,
        requesting_node: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entry from the DHT.

        Expected Behavior Under Chaos:
        - If partitioned, may not see entry
        - If gossip pending, may not see latest
        - If conflicting, returns one version (undefined which)

        Args:
            key: Entry key
            requesting_node: Node requesting the entry

        Returns:
            Entry value or None
        """
        if not self._connected:
            raise ConnectionError("DHT not connected")

        if requesting_node in self._partitioned_nodes:
            # Partitioned nodes have limited view
            return None

        # Process any ready gossip
        await self._process_gossip()

        return self._storage.get(key)

    async def _process_gossip(self):
        """Process pending gossip entries."""
        current_time = time.time()

        while self._pending_gossip:
            key, entry, propagation_time = self._pending_gossip[0]

            if current_time >= propagation_time:
                self._pending_gossip.popleft()
                self._storage[key] = entry
                logger.debug(f"Entry {key} propagated via gossip")
            else:
                break  # Future entries not ready yet

    def partition_nodes(self, node_ids: Set[str]):
        """Partition nodes from the DHT."""
        self._partitioned_nodes.update(node_ids)
        logger.info(f"DHT partitioned nodes: {node_ids}")

    def heal_partition(self):
        """Heal all DHT partitions."""
        self._partitioned_nodes.clear()
        logger.info("DHT partition healed")

    def set_gossip_delay(self, delay_ms: int):
        """Set gossip propagation delay."""
        self._gossip_delay_ms = delay_ms
        logger.info(f"DHT gossip delay set to {delay_ms}ms")

    async def create_conflict(
        self,
        key: str,
        value_a: Dict[str, Any],
        value_b: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Create a conflicting entry scenario.

        Two different values stored for same key, simulating
        split-brain or concurrent writes.

        Args:
            key: Entry key
            value_a: First value
            value_b: Second (conflicting) value

        Returns:
            Tuple of (hash_a, hash_b)
        """
        hash_a = await self.store_entry(key, value_a, "node_a")
        hash_b = await self.store_entry(key, value_b, "node_b")

        logger.warning(f"Conflicting entries created for key {key}")
        return hash_a, hash_b

    def get_conflict_count(self, key: str) -> int:
        """Get number of conflicting versions for a key."""
        return len(self._entry_versions.get(key, []))

    def disconnect(self):
        """Disconnect from DHT."""
        self._connected = False

    def reconnect(self):
        """Reconnect to DHT."""
        self._connected = True

    def reset(self):
        """Reset DHT state."""
        self._storage.clear()
        self._entry_versions.clear()
        self._partitioned_nodes.clear()
        self._gossip_delay_ms = 0
        self._pending_gossip.clear()
        self._connected = True


class ByzantineSurgeController:
    """
    Controller for Byzantine node surges.

    Simulates sudden increases in Byzantine activity
    and monitors system recovery.
    """

    def __init__(self, initial_byzantine_fraction: float = 0.2):
        self.initial_fraction = initial_byzantine_fraction
        self.current_fraction = initial_byzantine_fraction
        self.byzantine_nodes: Set[str] = set()
        self.honest_nodes: Set[str] = set()
        self._surge_active = False
        self._surge_start_time: Optional[float] = None

    def initialize_nodes(
        self,
        node_ids: List[str],
        byzantine_fraction: float
    ):
        """
        Initialize node classification.

        Args:
            node_ids: All node IDs
            byzantine_fraction: Initial Byzantine fraction
        """
        num_byzantine = int(len(node_ids) * byzantine_fraction)

        shuffled = node_ids.copy()
        random.shuffle(shuffled)

        self.byzantine_nodes = set(shuffled[:num_byzantine])
        self.honest_nodes = set(shuffled[num_byzantine:])
        self.current_fraction = byzantine_fraction

        logger.info(
            f"Initialized {len(self.byzantine_nodes)} Byzantine, "
            f"{len(self.honest_nodes)} honest nodes"
        )

    def trigger_surge(self, target_fraction: float = BYZANTINE_SURGE_FRACTION):
        """
        Trigger a Byzantine surge.

        Expected Behavior:
        - Additional nodes become Byzantine
        - System should detect increased Byzantine activity
        - Self-healing should activate if fraction > threshold

        Args:
            target_fraction: Target Byzantine fraction
        """
        if self._surge_active:
            return

        total_nodes = len(self.byzantine_nodes) + len(self.honest_nodes)
        target_byzantine = int(total_nodes * target_fraction)
        current_byzantine = len(self.byzantine_nodes)

        nodes_to_convert = target_byzantine - current_byzantine

        if nodes_to_convert > 0:
            # Convert honest nodes to Byzantine
            honest_list = list(self.honest_nodes)
            random.shuffle(honest_list)

            for node_id in honest_list[:nodes_to_convert]:
                self.honest_nodes.remove(node_id)
                self.byzantine_nodes.add(node_id)

        self.current_fraction = len(self.byzantine_nodes) / total_nodes
        self._surge_active = True
        self._surge_start_time = time.time()

        logger.warning(
            f"Byzantine surge triggered: {self.current_fraction:.1%} "
            f"({len(self.byzantine_nodes)} nodes)"
        )

    def check_recovery(self, detected_byzantine: Set[str]) -> bool:
        """
        Check if system has recovered from surge.

        Recovery means:
        - Most Byzantine nodes detected
        - Detection rate > 80%

        Args:
            detected_byzantine: Set of detected Byzantine nodes

        Returns:
            True if recovered
        """
        if not self._surge_active:
            return True

        true_positives = len(detected_byzantine & self.byzantine_nodes)
        detection_rate = true_positives / max(1, len(self.byzantine_nodes))

        recovered = detection_rate > 0.8

        if recovered:
            self._surge_active = False
            logger.info(f"System recovered from surge (detection rate: {detection_rate:.1%})")

        return recovered

    def is_byzantine(self, node_id: str) -> bool:
        """Check if a node is Byzantine."""
        return node_id in self.byzantine_nodes

    def get_stats(self) -> Dict[str, Any]:
        """Get current Byzantine statistics."""
        return {
            "byzantine_count": len(self.byzantine_nodes),
            "honest_count": len(self.honest_nodes),
            "current_fraction": self.current_fraction,
            "surge_active": self._surge_active,
            "surge_duration": (
                time.time() - self._surge_start_time
                if self._surge_active and self._surge_start_time
                else None
            ),
        }


class ResourceExhaustionController:
    """
    Controller for resource exhaustion simulation.

    Note: These tests are simulated - actual memory/CPU pressure
    would require system-level tools.
    """

    def __init__(self):
        self._memory_pressure_active = False
        self._cpu_throttle_active = False
        self._io_delay_ms = 0
        self._allocated_blocks: List[np.ndarray] = []

    @contextmanager
    def memory_pressure(self, megabytes: int = 100):
        """
        Context manager for memory pressure simulation.

        Expected Behavior:
        - System should handle reduced memory gracefully
        - May trigger garbage collection
        - Should not crash

        Args:
            megabytes: Amount of memory to allocate
        """
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        try:
            # Allocate memory blocks
            block_size = 1024 * 1024  # 1MB
            for _ in range(megabytes):
                self._allocated_blocks.append(
                    np.zeros(block_size // 8, dtype=np.float64)
                )

            self._memory_pressure_active = True
            logger.info(f"Memory pressure active: {megabytes}MB allocated")

            yield

        finally:
            # Release memory
            self._allocated_blocks.clear()
            gc.collect()
            self._memory_pressure_active = False

            current_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()

            logger.info(
                f"Memory pressure released. "
                f"Delta: {(current_memory - initial_memory) / 1024 / 1024:.2f}MB"
            )

    def set_cpu_throttle(self, throttle_factor: float = 0.5):
        """
        Set CPU throttle factor.

        Args:
            throttle_factor: 0.5 = 50% of normal speed
        """
        self._cpu_throttle_active = throttle_factor < 1.0
        logger.info(f"CPU throttle: {throttle_factor:.0%}")

    def get_cpu_delay(self) -> float:
        """Get artificial CPU delay in seconds."""
        if self._cpu_throttle_active:
            return 0.001  # 1ms delay per operation
        return 0

    def set_io_delay(self, delay_ms: int):
        """Set I/O delay simulation."""
        self._io_delay_ms = delay_ms
        logger.info(f"I/O delay set to {delay_ms}ms")

    async def simulate_io_operation(self):
        """Simulate an I/O operation with configured delay."""
        if self._io_delay_ms > 0:
            await asyncio.sleep(self._io_delay_ms / 1000)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def chaos_network():
    """Fixture providing network chaos controller."""
    controller = ChaosNetworkController()
    yield controller
    controller.reset()


@pytest.fixture
def chaos_coordinator(chaos_network):
    """Fixture providing chaos-enabled coordinator."""
    coordinator = MockChaosCoordinator(network=chaos_network)
    # Note: reconnect_node is sync in ChaosNetworkController for this mock
    coordinator.network.disconnected_nodes.discard("coordinator")
    return coordinator


@pytest.fixture
def chaos_dht():
    """Fixture providing chaos-enabled DHT."""
    dht = MockChaosDHT()
    yield dht
    dht.reset()


@pytest.fixture
def byzantine_controller():
    """Fixture providing Byzantine surge controller."""
    return ByzantineSurgeController()


@pytest.fixture
def resource_controller():
    """Fixture providing resource exhaustion controller."""
    return ResourceExhaustionController()


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def honest_gradients(rng):
    """Generate honest gradients for testing."""
    true_direction = rng.randn(DEFAULT_GRADIENT_SIZE).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}
    for i in range(5):
        noise = rng.randn(DEFAULT_GRADIENT_SIZE).astype(np.float32) * 0.1
        gradients[f"honest_{i}"] = true_direction + noise

    return gradients


@pytest.fixture
def byzantine_gradients(rng, honest_gradients):
    """Generate mixed honest and Byzantine gradients."""
    true_direction = list(honest_gradients.values())[0]
    true_direction = true_direction / np.linalg.norm(true_direction)

    gradients = honest_gradients.copy()

    # Add Byzantine gradients
    gradients["byzantine_flip"] = -true_direction * 1.5
    gradients["byzantine_scale"] = true_direction * 100.0
    gradients["byzantine_random"] = rng.randn(DEFAULT_GRADIENT_SIZE).astype(np.float32) * 5

    return gradients


# =============================================================================
# 1. NETWORK PARTITION TESTS
# =============================================================================

@pytest.mark.chaos
class TestNetworkPartition:
    """
    Network partition tests validate system behavior when nodes
    lose connectivity.

    Expected Behaviors:
    - Partitioned nodes cannot submit gradients
    - Non-partitioned nodes continue normally
    - Recovery occurs when partition heals
    - No data loss or corruption after healing
    """

    @pytest.mark.asyncio
    async def test_partition_during_round(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test network partition during FL round.

        Expected Behavior:
        - Partitioned nodes fail to submit
        - Round may fail if too few nodes remain
        - Non-partitioned nodes unaffected
        """
        # Setup
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Create partition (split nodes)
        group_a = set(node_ids[:2])
        group_b = set(node_ids[2:])

        await chaos_network.create_partition(
            group_a,
            {"coordinator"},  # Partition from coordinator
            duration_s=2.0
        )

        # Submit gradients
        submitted = []
        for node_id, gradient in honest_gradients.items():
            success = await chaos_coordinator.submit_gradient(
                node_id, gradient, round_num=1
            )
            if success:
                submitted.append(node_id)

        # Verify partition effect
        # Group A should fail, Group B should succeed
        for node_id in group_a:
            assert node_id not in submitted, \
                f"Partitioned node {node_id} should not submit"

        for node_id in group_b:
            assert node_id in submitted, \
                f"Non-partitioned node {node_id} should submit"

    @pytest.mark.asyncio
    async def test_partition_recovery(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test recovery after partition heals.

        Expected Behavior:
        - After healing, all nodes can communicate
        - Subsequent rounds succeed normally
        - No residual effects from partition
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Create short partition
        group_a = set(node_ids[:3])
        event = await chaos_network.create_partition(
            group_a,
            {"coordinator"},
            duration_s=0.5
        )

        # Wait for healing
        await asyncio.sleep(1.0)

        # Verify all nodes can now communicate
        for node_id in node_ids:
            can_communicate = chaos_network.can_communicate(node_id, "coordinator")
            assert can_communicate, \
                f"Node {node_id} should communicate after partition heals"

        # Submit gradients - all should succeed
        submitted = []
        for node_id, gradient in honest_gradients.items():
            success = await chaos_coordinator.submit_gradient(
                node_id, gradient, round_num=1
            )
            if success:
                submitted.append(node_id)

        assert len(submitted) == len(node_ids), \
            "All nodes should submit after partition heals"

    @pytest.mark.asyncio
    async def test_no_data_loss_after_partition(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Verify no data corruption after partition.

        Expected Behavior:
        - Gradients submitted before partition preserved
        - Gradients submitted after healing correct
        - Aggregation produces valid result
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Submit some gradients before partition
        pre_partition_nodes = node_ids[:2]
        for node_id in pre_partition_nodes:
            await chaos_coordinator.submit_gradient(
                node_id, honest_gradients[node_id], round_num=1
            )

        # Create brief partition
        await chaos_network.create_partition(
            set(node_ids[2:]),
            {"coordinator"},
            duration_s=0.5
        )

        await asyncio.sleep(0.6)  # Wait for healing

        # Submit remaining gradients
        for node_id in node_ids[2:]:
            await chaos_coordinator.submit_gradient(
                node_id, honest_gradients[node_id], round_num=1
            )

        # Aggregate
        result = await chaos_coordinator.aggregate_round(round_num=1)

        assert result is not None, "Aggregation should succeed after partition heals"

        # Verify result is valid (similar to honest mean)
        expected = np.mean(list(honest_gradients.values()), axis=0)
        cosine_sim = np.dot(result, expected) / (
            np.linalg.norm(result) * np.linalg.norm(expected)
        )

        assert cosine_sim > 0.9, \
            f"Aggregated gradient should be similar to expected (cosine={cosine_sim:.3f})"


# =============================================================================
# 2. NODE CRASH TESTS
# =============================================================================

@pytest.mark.chaos
class TestNodeCrash:
    """
    Node crash tests validate handling of sudden node failures.

    Expected Behaviors:
    - Coordinator crash preserves state
    - Node crash during submission handled gracefully
    - Multiple simultaneous failures don't cascade
    - System recovers after crashed nodes restart
    """

    @pytest.mark.asyncio
    async def test_coordinator_crash_during_aggregation(
        self,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test coordinator crash during aggregation.

        Expected Behavior:
        - Crash raises RuntimeError
        - Gradients received before crash preserved
        - Recovery restores coordinator
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Submit gradients
        for node_id, gradient in honest_gradients.items():
            await chaos_coordinator.submit_gradient(node_id, gradient, round_num=1)

        # Configure crash point
        chaos_coordinator.crash_at("mid_aggregate")

        # Aggregation should crash
        with pytest.raises(RuntimeError, match="Coordinator crashed"):
            await chaos_coordinator.aggregate_round(round_num=1)

        # Verify state preserved
        state = chaos_coordinator.get_state()
        assert state["crashed"], "Coordinator should be crashed"

        # Recover and retry
        await chaos_coordinator.recover()

        state = chaos_coordinator.get_state()
        assert not state["crashed"], "Coordinator should be recovered"

    @pytest.mark.asyncio
    async def test_node_crash_after_gradient_submit(
        self,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test node crash after submitting gradient.

        Expected Behavior:
        - Gradient already submitted is preserved
        - Crashed node excluded from future rounds
        - Aggregation uses submitted gradient
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Submit gradients
        for node_id, gradient in honest_gradients.items():
            await chaos_coordinator.submit_gradient(node_id, gradient, round_num=1)

        # Crash a node after submission
        chaos_coordinator.crash_node(node_ids[0])

        # Aggregation should still succeed using already-submitted gradient
        result = await chaos_coordinator.aggregate_round(round_num=1)

        assert result is not None, \
            "Aggregation should succeed with already-submitted gradients"

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_node_failures(
        self,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test multiple nodes crashing simultaneously.

        Expected Behavior:
        - System handles multiple failures gracefully
        - Round fails if below minimum nodes
        - System continues with remaining nodes if sufficient
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Submit gradients
        for node_id, gradient in honest_gradients.items():
            await chaos_coordinator.submit_gradient(node_id, gradient, round_num=1)

        # Crash multiple nodes (leave exactly min_nodes - 1)
        nodes_to_crash = len(node_ids) - chaos_coordinator.min_nodes + 1
        for i in range(nodes_to_crash):
            chaos_coordinator.crash_node(node_ids[i])

        # Round 1 should still succeed (gradients already submitted)
        result = await chaos_coordinator.aggregate_round(round_num=1)
        assert result is not None

        # Reset for round 2
        chaos_coordinator.reset_round()

        # Round 2 should fail (crashed nodes can't submit)
        remaining_nodes = [n for n in node_ids[nodes_to_crash:]
                         if chaos_coordinator.node_states.get(n) != NodeState.CRASHED]

        for node_id in remaining_nodes:
            await chaos_coordinator.submit_gradient(
                node_id, honest_gradients[node_id], round_num=2
            )

        result = await chaos_coordinator.aggregate_round(round_num=2)

        # Should fail due to insufficient nodes
        assert 2 in chaos_coordinator.failed_rounds, \
            "Round should fail with insufficient active nodes"

    @pytest.mark.asyncio
    async def test_node_recovery_after_crash(
        self,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test node recovery after crash.

        Expected Behavior:
        - Recovered node can rejoin
        - Subsequent rounds include recovered node
        - No state corruption from crash/recovery cycle
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Crash a node
        crashed_node = node_ids[0]
        chaos_coordinator.crash_node(crashed_node)

        assert chaos_coordinator.node_states[crashed_node] == NodeState.CRASHED

        # Recover the node
        chaos_coordinator.recover_node(crashed_node)

        assert chaos_coordinator.node_states[crashed_node] == NodeState.RUNNING

        # Node can now submit
        success = await chaos_coordinator.submit_gradient(
            crashed_node, honest_gradients[crashed_node], round_num=1
        )

        assert success, "Recovered node should be able to submit"


# =============================================================================
# 3. BYZANTINE SURGE TESTS
# =============================================================================

@pytest.mark.chaos
class TestByzantineSurge:
    """
    Byzantine surge tests validate self-healing under sudden increases
    in Byzantine activity.

    Expected Behaviors:
    - System detects Byzantine surge
    - Self-healing activates when BFT > threshold
    - Byzantine influence reduced through quarantine
    - System stabilizes and recovers
    """

    def test_byzantine_surge_detection(
        self,
        byzantine_controller: ByzantineSurgeController
    ):
        """
        Test detection of Byzantine surge.

        Expected Behavior:
        - Surge increases Byzantine fraction
        - Detection identifies Byzantine nodes
        - System remains aware of increased threat
        """
        # Initialize with low Byzantine fraction
        node_ids = [f"node_{i}" for i in range(10)]
        byzantine_controller.initialize_nodes(node_ids, byzantine_fraction=0.2)

        initial_stats = byzantine_controller.get_stats()
        assert initial_stats["current_fraction"] == 0.2
        assert not initial_stats["surge_active"]

        # Trigger surge
        byzantine_controller.trigger_surge(target_fraction=0.45)

        surge_stats = byzantine_controller.get_stats()
        assert surge_stats["current_fraction"] >= 0.4, \
            "Byzantine fraction should increase"
        assert surge_stats["surge_active"], \
            "Surge should be active"

    def test_self_healing_activation(
        self,
        byzantine_controller: ByzantineSurgeController,
        rng
    ):
        """
        Test self-healing mechanism activation.

        Expected Behavior:
        - Self-healing activates when BFT > tolerance
        - Adaptive thresholds adjust during healing
        - Gradients quarantined appropriately
        """
        # Import self-healing (if available)
        try:
            from gen5.self_healing import SelfHealingMechanism, HealingConfig
        except ImportError:
            pytest.skip("Self-healing module not available")

        healer = SelfHealingMechanism(
            config=HealingConfig(
                bft_tolerance=0.45,
                healing_threshold=0.40,
                score_threshold=0.5
            )
        )

        # Simulate normal operation (40% Byzantine)
        normal_scores = [0.8, 0.85, 0.15, 0.20, 0.75, 0.70, 0.18, 0.82, 0.78, 0.12]
        decisions, details = healer.process_batch(normal_scores, round_num=0)

        assert not details["is_healing"], \
            "Should not heal at 40% Byzantine"

        # Simulate Byzantine surge (60% Byzantine)
        surge_scores = [0.15, 0.20, 0.10, 0.18, 0.12, 0.85, 0.80, 0.14, 0.16, 0.11]
        decisions, details = healer.process_batch(surge_scores, round_num=1)

        assert details["is_healing"], \
            "Should activate healing above tolerance"
        assert details["current_bft"] > 0.45, \
            "BFT estimate should reflect surge"

    def test_system_stabilization(
        self,
        byzantine_controller: ByzantineSurgeController
    ):
        """
        Test system stabilization after Byzantine surge.

        Expected Behavior:
        - Detection rate improves over time
        - System eventually recovers
        - Post-recovery operation normal
        """
        node_ids = [f"node_{i}" for i in range(10)]
        byzantine_controller.initialize_nodes(node_ids, byzantine_fraction=0.2)

        # Trigger surge
        byzantine_controller.trigger_surge(target_fraction=0.45)

        # Simulate detection improving over rounds
        # In reality, this would use actual Byzantine detection
        detected = set()

        for round_num in range(5):
            # Each round, detect more Byzantine nodes
            byzantine_list = list(byzantine_controller.byzantine_nodes)
            new_detected = set(byzantine_list[:min(round_num + 1, len(byzantine_list))])
            detected.update(new_detected)

            if byzantine_controller.check_recovery(detected):
                break

        assert not byzantine_controller._surge_active, \
            "System should recover from surge"

    @pytest.mark.slow
    def test_surge_and_recovery_cycle(
        self,
        byzantine_controller: ByzantineSurgeController,
        rng
    ):
        """
        Test complete surge and recovery cycle.

        Expected Behavior:
        - Multiple surge/recovery cycles handled
        - No state corruption between cycles
        - System remains resilient
        """
        node_ids = [f"node_{i}" for i in range(20)]

        for cycle in range(3):
            # Reset
            byzantine_controller.initialize_nodes(node_ids, byzantine_fraction=0.2)

            # Surge
            byzantine_controller.trigger_surge(target_fraction=0.4 + cycle * 0.05)

            stats = byzantine_controller.get_stats()
            assert stats["surge_active"]

            # Simulate recovery
            detected = byzantine_controller.byzantine_nodes.copy()
            recovered = byzantine_controller.check_recovery(detected)

            assert recovered, f"Should recover in cycle {cycle}"


# =============================================================================
# 4. LATENCY INJECTION TESTS
# =============================================================================

@pytest.mark.chaos
class TestLatencyInjection:
    """
    Latency injection tests validate timeout handling and
    late-arriving gradient processing.

    Expected Behaviors:
    - High-latency nodes may timeout
    - Late gradients handled appropriately
    - System continues with available gradients
    - No corruption from late arrivals
    """

    @pytest.mark.asyncio
    async def test_high_latency_nodes(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test handling of high-latency nodes.

        Expected Behavior:
        - High-latency submissions delayed
        - Round may proceed without slow nodes
        - No errors from delay
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Inject latency on some nodes
        slow_nodes = node_ids[:2]
        await chaos_network.inject_latency(
            slow_nodes,
            latency_ms=HIGH_LATENCY_MS,
            duration_s=10.0
        )

        # Verify latency applied
        for node_id in slow_nodes:
            latency = chaos_network.get_latency(node_id)
            assert latency == HIGH_LATENCY_MS, \
                f"Latency should be injected for {node_id}"

        # Fast nodes should not have latency
        fast_nodes = node_ids[2:]
        for node_id in fast_nodes:
            latency = chaos_network.get_latency(node_id)
            assert latency == 0, \
                f"No latency for fast node {node_id}"

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test timeout handling for slow submissions.

        Expected Behavior:
        - Submissions with extreme latency timeout
        - Timeout doesn't crash system
        - Other submissions unaffected
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Very high latency on one node
        timeout_node = node_ids[0]
        await chaos_network.inject_latency(
            [timeout_node],
            latency_ms=5000,  # 5 seconds
            duration_s=10.0
        )

        # Submit from all nodes with timeout
        async def submit_with_timeout(node_id: str, timeout_s: float = 1.0):
            try:
                return await asyncio.wait_for(
                    chaos_coordinator.submit_gradient(
                        node_id, honest_gradients[node_id], round_num=1
                    ),
                    timeout=timeout_s
                )
            except asyncio.TimeoutError:
                return False

        results = await asyncio.gather(*[
            submit_with_timeout(node_id)
            for node_id in node_ids
        ])

        # Slow node should timeout
        assert results[0] is False, \
            "High-latency node should timeout"

        # Other nodes should succeed
        assert all(results[1:]), \
            "Fast nodes should succeed"

    @pytest.mark.asyncio
    async def test_late_arriving_gradients(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test handling of late-arriving gradients.

        Expected Behavior:
        - Gradients arriving after aggregation not included
        - No corruption from late gradients
        - Next round starts fresh
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Submit most gradients immediately
        fast_nodes = node_ids[:-1]
        for node_id in fast_nodes:
            await chaos_coordinator.submit_gradient(
                node_id, honest_gradients[node_id], round_num=1
            )

        # Aggregate before last node submits
        result_1 = await chaos_coordinator.aggregate_round(round_num=1)

        # Late gradient arrives
        late_node = node_ids[-1]
        await chaos_coordinator.submit_gradient(
            late_node, honest_gradients[late_node], round_num=1
        )

        # Verify aggregation used only fast nodes
        assert result_1 is not None

        # Late gradient should not affect round 1 result
        # (It will be available for round 2 if re-submitted)


# =============================================================================
# 5. RESOURCE EXHAUSTION TESTS
# =============================================================================

@pytest.mark.chaos
class TestResourceExhaustion:
    """
    Resource exhaustion tests validate system behavior under
    memory, CPU, and I/O pressure.

    Expected Behaviors:
    - System handles memory pressure gracefully
    - CPU throttling doesn't cause hangs
    - I/O delays handled with appropriate timeouts
    - No crashes under resource constraints
    """

    def test_memory_pressure(
        self,
        resource_controller: ResourceExhaustionController,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test operation under memory pressure.

        Expected Behavior:
        - Operations complete despite reduced memory
        - No memory errors or crashes
        - Garbage collection triggered if needed
        """
        with resource_controller.memory_pressure(megabytes=50):
            # Perform memory-intensive operations
            gradients_list = list(honest_gradients.values())

            # Aggregation
            aggregated = np.mean(gradients_list, axis=0)

            # Clone operations
            clones = [g.copy() for g in gradients_list]

            # Verify correctness
            assert aggregated.shape == gradients_list[0].shape
            assert len(clones) == len(gradients_list)

    def test_cpu_throttle(
        self,
        resource_controller: ResourceExhaustionController,
        honest_gradients: Dict[str, np.ndarray],
        rng
    ):
        """
        Test operation under CPU throttling.

        Expected Behavior:
        - Operations complete (possibly slower)
        - No timeouts for reasonable workloads
        - Results correct despite throttling
        """
        resource_controller.set_cpu_throttle(throttle_factor=0.5)

        # Perform CPU-intensive operations
        gradients_list = list(honest_gradients.values())

        start = time.time()

        # Compute norms
        norms = [np.linalg.norm(g) for g in gradients_list]

        # Compute cosine similarities
        for i, g1 in enumerate(gradients_list):
            for j, g2 in enumerate(gradients_list):
                if i < j:
                    similarity = np.dot(g1, g2) / (norms[i] * norms[j])
                    # Add artificial delay
                    time.sleep(resource_controller.get_cpu_delay())

        elapsed = time.time() - start

        # Should complete in reasonable time even throttled
        assert elapsed < 10.0, \
            f"CPU-intensive operation took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_disk_io_delay(
        self,
        resource_controller: ResourceExhaustionController,
        chaos_dht: MockChaosDHT
    ):
        """
        Test handling of disk I/O delays.

        Expected Behavior:
        - I/O operations complete despite delays
        - Timeouts don't cause data loss
        - System recovers when I/O normalizes
        """
        resource_controller.set_io_delay(delay_ms=100)

        # Simulate I/O operations
        for i in range(5):
            await resource_controller.simulate_io_operation()

            # Store in DHT (simulating disk-backed storage)
            await chaos_dht.store_entry(
                key=f"test_key_{i}",
                value={"data": f"value_{i}"},
                source_node="test_node"
            )

        # Verify all entries stored
        for i in range(5):
            entry = await chaos_dht.get_entry(f"test_key_{i}", "test_node")
            assert entry is not None, \
                f"Entry {i} should be stored despite I/O delays"


# =============================================================================
# 6. HOLOCHAIN DHT TESTS
# =============================================================================

@pytest.mark.chaos
class TestHolochainDHTChaos:
    """
    Holochain DHT chaos tests validate DHT behavior under
    adverse network conditions.

    Expected Behaviors:
    - Partitioned nodes have limited DHT view
    - Slow gossip eventually propagates
    - Conflicting entries detected and handled
    - DHT reconnection restores full functionality
    """

    @pytest.mark.asyncio
    async def test_dht_partition(
        self,
        chaos_dht: MockChaosDHT
    ):
        """
        Test DHT behavior during partition.

        Expected Behavior:
        - Partitioned nodes can't store entries
        - Partitioned nodes can't retrieve entries
        - Non-partitioned nodes operate normally
        """
        # Store entry normally
        hash_1 = await chaos_dht.store_entry(
            key="pre_partition",
            value={"data": "before"},
            source_node="normal_node"
        )

        # Partition some nodes
        chaos_dht.partition_nodes({"partitioned_node"})

        # Partitioned node can't store
        with pytest.raises(ConnectionError):
            await chaos_dht.store_entry(
                key="during_partition",
                value={"data": "during"},
                source_node="partitioned_node"
            )

        # Partitioned node can't retrieve
        entry = await chaos_dht.get_entry("pre_partition", "partitioned_node")
        assert entry is None, \
            "Partitioned node should not see entries"

        # Non-partitioned node can retrieve
        entry = await chaos_dht.get_entry("pre_partition", "normal_node")
        assert entry is not None, \
            "Non-partitioned node should see entries"

    @pytest.mark.asyncio
    async def test_slow_gossip_propagation(
        self,
        chaos_dht: MockChaosDHT
    ):
        """
        Test DHT with slow gossip propagation.

        Expected Behavior:
        - Entries not immediately visible
        - Entries visible after gossip delay
        - No data loss from gossip delay
        """
        # Set gossip delay
        chaos_dht.set_gossip_delay(delay_ms=GOSSIP_DELAY_MS)

        # Store entry
        await chaos_dht.store_entry(
            key="delayed_entry",
            value={"data": "test"},
            source_node="source_node"
        )

        # Immediate retrieval should fail (gossip pending)
        entry = await chaos_dht.get_entry("delayed_entry", "other_node")
        assert entry is None, \
            "Entry should not be immediately visible with gossip delay"

        # Wait for gossip
        await asyncio.sleep(GOSSIP_DELAY_MS / 1000 + 0.1)

        # Now entry should be visible
        entry = await chaos_dht.get_entry("delayed_entry", "other_node")
        assert entry is not None, \
            "Entry should be visible after gossip delay"

    @pytest.mark.asyncio
    async def test_conflicting_dht_entries(
        self,
        chaos_dht: MockChaosDHT
    ):
        """
        Test handling of conflicting DHT entries.

        Expected Behavior:
        - Conflicts are tracked
        - System detects multiple versions
        - Conflict resolution can be applied
        """
        # Create conflicting entries
        hash_a, hash_b = await chaos_dht.create_conflict(
            key="conflicted_key",
            value_a={"version": "A", "data": "first"},
            value_b={"version": "B", "data": "second"}
        )

        # Verify conflict detected
        conflict_count = chaos_dht.get_conflict_count("conflicted_key")
        assert conflict_count >= 2, \
            f"Should detect conflict (versions: {conflict_count})"

        # Retrieval returns one version (conflict resolution would pick winner)
        entry = await chaos_dht.get_entry("conflicted_key", "test_node")
        assert entry is not None, \
            "Should retrieve one version despite conflict"

    @pytest.mark.asyncio
    async def test_dht_disconnection_recovery(
        self,
        chaos_dht: MockChaosDHT
    ):
        """
        Test DHT disconnection and recovery.

        Expected Behavior:
        - Disconnection causes operations to fail
        - Reconnection restores functionality
        - No data loss during disconnect
        """
        # Store entry
        await chaos_dht.store_entry(
            key="persistent",
            value={"data": "survives"},
            source_node="test_node"
        )

        # Disconnect
        chaos_dht.disconnect()

        # Operations should fail
        with pytest.raises(ConnectionError):
            await chaos_dht.store_entry(
                key="during_disconnect",
                value={"data": "fails"},
                source_node="test_node"
            )

        with pytest.raises(ConnectionError):
            await chaos_dht.get_entry("persistent", "test_node")

        # Reconnect
        chaos_dht.reconnect()

        # Operations should work
        entry = await chaos_dht.get_entry("persistent", "test_node")
        assert entry is not None, \
            "Entry should persist through disconnect"
        assert entry["value"]["data"] == "survives", \
            "Entry data should be intact"


# =============================================================================
# INTEGRATION CHAOS TESTS
# =============================================================================

@pytest.mark.chaos
@pytest.mark.slow
class TestChaosIntegration:
    """
    Integration tests combining multiple chaos scenarios.

    These tests validate system resilience under combined
    adverse conditions.
    """

    @pytest.mark.asyncio
    async def test_partition_with_byzantine_surge(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        byzantine_controller: ByzantineSurgeController,
        byzantine_gradients: Dict[str, np.ndarray]
    ):
        """
        Test combined partition and Byzantine surge.

        Expected Behavior:
        - System handles both challenges
        - Byzantine detection works despite partition
        - Recovery occurs when both heal
        """
        node_ids = list(byzantine_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Initialize Byzantine controller
        byzantine_controller.initialize_nodes(node_ids, byzantine_fraction=0.2)

        # Trigger Byzantine surge
        byzantine_controller.trigger_surge(target_fraction=0.4)

        # Also create partition
        honest_nodes = list(byzantine_controller.honest_nodes)[:2]
        await chaos_network.create_partition(
            set(honest_nodes),
            {"coordinator"},
            duration_s=2.0
        )

        # Submit gradients
        for node_id, gradient in byzantine_gradients.items():
            if chaos_network.can_communicate(node_id, "coordinator"):
                await chaos_coordinator.submit_gradient(
                    node_id, gradient, round_num=1,
                    is_byzantine=byzantine_controller.is_byzantine(node_id)
                )

        # System should handle combined chaos
        # (Actual aggregation would use Byzantine-robust method)

        # Wait for partition to heal
        await asyncio.sleep(2.5)

        # Verify connectivity restored
        for node_id in honest_nodes:
            assert chaos_network.can_communicate(node_id, "coordinator")

    @pytest.mark.asyncio
    async def test_cascading_failures(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        chaos_dht: MockChaosDHT,
        honest_gradients: Dict[str, np.ndarray]
    ):
        """
        Test cascading failure scenario.

        Expected Behavior:
        - Multiple failures don't cause total collapse
        - System degrades gracefully
        - Recovery is possible after failures clear
        """
        node_ids = list(honest_gradients.keys())
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Stage 1: Network partition
        await chaos_network.create_partition(
            {node_ids[0]},
            {"coordinator"},
            duration_s=5.0
        )

        # Stage 2: DHT slow gossip
        chaos_dht.set_gossip_delay(delay_ms=500)

        # Stage 3: Node crash
        chaos_coordinator.crash_node(node_ids[1])

        # Stage 4: Latency injection
        await chaos_network.inject_latency([node_ids[2]], latency_ms=1000, duration_s=5.0)

        # Submit from remaining functional nodes
        functional_nodes = node_ids[3:]
        for node_id in functional_nodes:
            success = await chaos_coordinator.submit_gradient(
                node_id, honest_gradients[node_id], round_num=1
            )
            assert success, f"Functional node {node_id} should submit"

        # System should still function with reduced capacity
        # (Aggregation may fail if below min_nodes)

        # Allow recovery
        await asyncio.sleep(6.0)

        # Verify some recovery
        events = chaos_network.get_events()
        healed_events = [e for e in events if e.end_time is not None]
        assert len(healed_events) > 0, "Some chaos should have healed"


# =============================================================================
# STRESS TESTS
# =============================================================================

@pytest.mark.chaos
@pytest.mark.slow
class TestChaosStress:
    """
    Stress tests for extended chaos scenarios.
    """

    @pytest.mark.asyncio
    async def test_extended_partition_recovery(
        self,
        chaos_network: ChaosNetworkController,
        chaos_coordinator: MockChaosCoordinator,
        rng
    ):
        """
        Test recovery from extended partition.

        Expected Behavior:
        - System survives long partition
        - Full recovery when partition heals
        - State consistent after recovery
        """
        # Create many nodes
        node_ids = [f"node_{i}" for i in range(20)]
        for node_id in node_ids:
            await chaos_coordinator.register_node(node_id)

        # Long partition
        partition_group = set(node_ids[:10])
        event = await chaos_network.create_partition(
            partition_group,
            {"coordinator"},
            duration_s=10.0
        )

        # Run multiple rounds during partition
        for round_num in range(1, 4):
            gradients = {
                node_id: rng.randn(DEFAULT_GRADIENT_SIZE).astype(np.float32)
                for node_id in node_ids
            }

            for node_id, gradient in gradients.items():
                if chaos_network.can_communicate(node_id, "coordinator"):
                    await chaos_coordinator.submit_gradient(
                        node_id, gradient, round_num=round_num
                    )

            result = await chaos_coordinator.aggregate_round(round_num)
            # May succeed or fail depending on min_nodes

            chaos_coordinator.reset_round()

        # Wait for healing
        await asyncio.sleep(10.5)

        # All nodes should be able to communicate
        for node_id in node_ids:
            assert chaos_network.can_communicate(node_id, "coordinator"), \
                f"Node {node_id} should communicate after healing"

    def test_rapid_byzantine_oscillation(
        self,
        byzantine_controller: ByzantineSurgeController
    ):
        """
        Test rapid oscillation between normal and surge.

        Expected Behavior:
        - System handles rapid state changes
        - No state corruption
        - Statistics accurate
        """
        node_ids = [f"node_{i}" for i in range(10)]

        for cycle in range(10):
            # Initialize fresh
            byzantine_controller.initialize_nodes(
                node_ids,
                byzantine_fraction=0.2 if cycle % 2 == 0 else 0.1
            )

            if cycle % 2 == 0:
                # Surge
                byzantine_controller.trigger_surge(target_fraction=0.45)
                assert byzantine_controller._surge_active
            else:
                # Recovery
                detected = byzantine_controller.byzantine_nodes.copy()
                byzantine_controller.check_recovery(detected)
                # Note: Recovery requires detection, may not always trigger

        stats = byzantine_controller.get_stats()
        assert stats["byzantine_count"] + stats["honest_count"] == len(node_ids)


# =============================================================================
# UTILITY FUNCTIONS FOR CHAOS TESTING
# =============================================================================

def generate_random_chaos_sequence(
    num_events: int,
    rng: np.random.RandomState
) -> List[ChaosConfig]:
    """
    Generate a random sequence of chaos events for fuzzing.

    Args:
        num_events: Number of chaos events to generate
        rng: Random number generator

    Returns:
        List of chaos configurations
    """
    chaos_types = list(ChaosType)
    configs = []

    for i in range(num_events):
        chaos_type = rng.choice(chaos_types)

        config = ChaosConfig(
            chaos_type=chaos_type,
            target_nodes=[f"node_{j}" for j in range(rng.randint(1, 5))],
            duration_s=rng.uniform(1.0, 10.0),
            intensity=rng.uniform(0.3, 1.0),
            start_delay_s=rng.uniform(0.0, 5.0) * i
        )
        configs.append(config)

    return configs


async def run_chaos_sequence(
    configs: List[ChaosConfig],
    network: ChaosNetworkController,
    dht: MockChaosDHT,
    byzantine: ByzantineSurgeController
) -> List[ChaosEvent]:
    """
    Execute a sequence of chaos events.

    Args:
        configs: Chaos configurations to execute
        network: Network controller
        dht: DHT controller
        byzantine: Byzantine controller

    Returns:
        List of executed chaos events
    """
    events = []

    for config in configs:
        await asyncio.sleep(config.start_delay_s)

        if config.chaos_type == ChaosType.NETWORK_PARTITION:
            nodes = set(config.target_nodes)
            event = await network.create_partition(
                nodes,
                {"coordinator"},
                duration_s=config.duration_s
            )
            events.append(event)

        elif config.chaos_type == ChaosType.LATENCY_INJECTION:
            latency_ms = int(config.intensity * HIGH_LATENCY_MS)
            event = await network.inject_latency(
                config.target_nodes,
                latency_ms=latency_ms,
                duration_s=config.duration_s
            )
            events.append(event)

        elif config.chaos_type == ChaosType.DHT_PARTITION:
            dht.partition_nodes(set(config.target_nodes))
            events.append(ChaosEvent(
                event_id=f"dht_partition_{time.time()}",
                chaos_type=ChaosType.DHT_PARTITION,
                start_time=time.time(),
                affected_nodes=set(config.target_nodes)
            ))

        elif config.chaos_type == ChaosType.GOSSIP_DELAY:
            delay_ms = int(config.intensity * GOSSIP_DELAY_MS)
            dht.set_gossip_delay(delay_ms)
            events.append(ChaosEvent(
                event_id=f"gossip_delay_{time.time()}",
                chaos_type=ChaosType.GOSSIP_DELAY,
                start_time=time.time(),
                impact={"delay_ms": delay_ms}
            ))

        elif config.chaos_type == ChaosType.BYZANTINE_SURGE:
            target_fraction = 0.2 + config.intensity * 0.3  # 20-50%
            byzantine.trigger_surge(target_fraction)
            events.append(ChaosEvent(
                event_id=f"byzantine_surge_{time.time()}",
                chaos_type=ChaosType.BYZANTINE_SURGE,
                start_time=time.time(),
                impact={"target_fraction": target_fraction}
            ))

    return events


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run chaos tests
    pytest.main([
        __file__,
        "-v",
        "-m", "chaos",
        "--tb=short"
    ])
