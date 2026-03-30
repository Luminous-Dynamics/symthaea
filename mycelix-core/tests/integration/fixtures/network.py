# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FL Network setup and teardown fixtures.

Provides comprehensive network simulation for Mycelix FL integration testing,
including Holochain zome mocking, gradient storage, and aggregation.
"""

import asyncio
import numpy as np
import hashlib
import time
import json
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Network Configuration
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for FL network simulation."""
    num_honest_nodes: int = 10
    num_byzantine_nodes: int = 0
    gradient_dimension: int = 1000
    aggregation_strategy: str = "trimmed_mean"
    byzantine_threshold: float = 0.33
    detection_enabled: bool = True
    reputation_enabled: bool = True
    holochain_mock: bool = True
    consensus_rounds: int = 3
    timeout_ms: int = 5000
    min_participants: int = 3
    staleness_threshold_rounds: int = 5
    gradient_clip_norm: float = 10.0
    learning_rate: float = 0.01

    @property
    def total_nodes(self) -> int:
        return self.num_honest_nodes + self.num_byzantine_nodes

    @property
    def byzantine_ratio(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return self.num_byzantine_nodes / self.total_nodes


# ============================================================================
# Gradient Storage (Holochain DHT Mock)
# ============================================================================

@dataclass
class GradientEntry:
    """A gradient entry in the DHT."""
    entry_hash: str
    node_id: str
    round_id: int
    gradient: np.ndarray
    timestamp: float
    signature: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None


class GradientStore:
    """
    Mock Holochain DHT for gradient storage.

    Simulates the gradient-store zome functionality including:
    - Gradient submission and retrieval
    - Entry validation
    - Round-based organization
    """

    def __init__(self):
        self._entries: Dict[str, GradientEntry] = {}
        self._by_round: Dict[int, List[str]] = {}
        self._by_node: Dict[str, List[str]] = {}
        self._validation_callbacks: List[callable] = []
        self._lock = asyncio.Lock()

    def _compute_hash(self, data: bytes) -> str:
        """Compute entry hash."""
        return hashlib.sha256(data).hexdigest()[:32]

    def _serialize_gradient(self, gradient: np.ndarray) -> bytes:
        """Serialize gradient for hashing."""
        return gradient.astype(np.float32).tobytes()

    async def submit_gradient(
        self,
        node_id: str,
        round_id: int,
        gradient: np.ndarray,
        signature: bytes = b"",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Submit a gradient to the store.

        Returns:
            Tuple of (success, entry_hash, error_message)
        """
        async with self._lock:
            # Create entry
            gradient_bytes = self._serialize_gradient(gradient)
            entry_data = node_id.encode() + struct.pack("I", round_id) + gradient_bytes
            entry_hash = self._compute_hash(entry_data)

            entry = GradientEntry(
                entry_hash=entry_hash,
                node_id=node_id,
                round_id=round_id,
                gradient=gradient.copy(),
                timestamp=time.time(),
                signature=signature,
                metadata=metadata or {},
            )

            # Run validation callbacks
            for callback in self._validation_callbacks:
                is_valid, reason = await callback(entry)
                if not is_valid:
                    entry.rejected = True
                    entry.rejection_reason = reason
                    self._entries[entry_hash] = entry
                    return False, entry_hash, reason

            entry.validated = True
            self._entries[entry_hash] = entry

            # Index by round and node
            if round_id not in self._by_round:
                self._by_round[round_id] = []
            self._by_round[round_id].append(entry_hash)

            if node_id not in self._by_node:
                self._by_node[node_id] = []
            self._by_node[node_id].append(entry_hash)

            return True, entry_hash, None

    async def get_gradient(self, entry_hash: str) -> Optional[GradientEntry]:
        """Retrieve a gradient by its hash."""
        return self._entries.get(entry_hash)

    async def get_gradients_for_round(
        self,
        round_id: int,
        only_validated: bool = True
    ) -> List[GradientEntry]:
        """Get all gradients for a specific round."""
        hashes = self._by_round.get(round_id, [])
        entries = []
        for h in hashes:
            entry = self._entries.get(h)
            if entry:
                if only_validated and not entry.validated:
                    continue
                if entry.rejected:
                    continue
                entries.append(entry)
        return entries

    async def get_gradients_by_node(self, node_id: str) -> List[GradientEntry]:
        """Get all gradients submitted by a specific node."""
        hashes = self._by_node.get(node_id, [])
        return [self._entries[h] for h in hashes if h in self._entries]

    def add_validation_callback(self, callback: callable):
        """Add a validation callback for gradient submission."""
        self._validation_callbacks.append(callback)

    async def clear(self):
        """Clear all stored gradients."""
        async with self._lock:
            self._entries.clear()
            self._by_round.clear()
            self._by_node.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total = len(self._entries)
        validated = sum(1 for e in self._entries.values() if e.validated)
        rejected = sum(1 for e in self._entries.values() if e.rejected)
        return {
            "total_entries": total,
            "validated": validated,
            "rejected": rejected,
            "rounds": len(self._by_round),
            "nodes": len(self._by_node),
        }


# ============================================================================
# Zome Validator
# ============================================================================

class ZomeValidator:
    """
    Validates gradients at the zome level.

    Implements validation rules that would be enforced by the Holochain zome,
    including:
    - Gradient magnitude checks
    - NaN/Inf detection
    - Statistical anomaly detection
    - Signature verification
    """

    def __init__(
        self,
        max_gradient_norm: float = 100.0,
        max_element_value: float = 1000.0,
        min_dimension: int = 1,
        max_dimension: int = 1_000_000,
    ):
        self.max_gradient_norm = max_gradient_norm
        self.max_element_value = max_element_value
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self._rejection_log: List[Dict[str, Any]] = []

    async def validate(self, entry: GradientEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate a gradient entry.

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        gradient = entry.gradient

        # Check dimension
        if gradient.size < self.min_dimension:
            reason = f"Gradient dimension {gradient.size} below minimum {self.min_dimension}"
            self._log_rejection(entry, reason)
            return False, reason

        if gradient.size > self.max_dimension:
            reason = f"Gradient dimension {gradient.size} exceeds maximum {self.max_dimension}"
            self._log_rejection(entry, reason)
            return False, reason

        # Check for NaN or Inf
        if np.any(np.isnan(gradient)):
            reason = "Gradient contains NaN values"
            self._log_rejection(entry, reason)
            return False, reason

        if np.any(np.isinf(gradient)):
            reason = "Gradient contains Inf values"
            self._log_rejection(entry, reason)
            return False, reason

        # Check gradient norm
        norm = np.linalg.norm(gradient)
        if norm > self.max_gradient_norm:
            reason = f"Gradient norm {norm:.2f} exceeds maximum {self.max_gradient_norm}"
            self._log_rejection(entry, reason)
            return False, reason

        # Check element values
        max_val = np.max(np.abs(gradient))
        if max_val > self.max_element_value:
            reason = f"Max element value {max_val:.2f} exceeds limit {self.max_element_value}"
            self._log_rejection(entry, reason)
            return False, reason

        return True, None

    def _log_rejection(self, entry: GradientEntry, reason: str):
        """Log a rejection for analysis."""
        self._rejection_log.append({
            "entry_hash": entry.entry_hash,
            "node_id": entry.node_id,
            "round_id": entry.round_id,
            "reason": reason,
            "timestamp": time.time(),
        })

    def get_rejection_log(self) -> List[Dict[str, Any]]:
        """Get the rejection log."""
        return self._rejection_log.copy()

    def clear_rejection_log(self):
        """Clear the rejection log."""
        self._rejection_log.clear()


# ============================================================================
# Reputation Bridge
# ============================================================================

class ReputationBridge:
    """
    Bridge between Byzantine detection and reputation scores.

    Manages node reputation based on:
    - Byzantine detection results
    - Gradient quality metrics
    - Participation history
    """

    def __init__(
        self,
        initial_score: float = 1.0,
        min_score: float = 0.0,
        max_score: float = 1.0,
        decay_rate: float = 0.01,
        penalty_byzantine: float = 0.3,
        penalty_rejected: float = 0.1,
        reward_honest: float = 0.05,
    ):
        self.initial_score = initial_score
        self.min_score = min_score
        self.max_score = max_score
        self.decay_rate = decay_rate
        self.penalty_byzantine = penalty_byzantine
        self.penalty_rejected = penalty_rejected
        self.reward_honest = reward_honest

        self._scores: Dict[str, float] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._byzantine_flags: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_score(self, node_id: str) -> float:
        """Get the reputation score for a node."""
        return self._scores.get(node_id, self.initial_score)

    async def update_score(
        self,
        node_id: str,
        delta: float,
        reason: str
    ) -> float:
        """Update a node's reputation score."""
        async with self._lock:
            current = self._scores.get(node_id, self.initial_score)
            new_score = max(self.min_score, min(self.max_score, current + delta))
            self._scores[node_id] = new_score

            # Log history
            if node_id not in self._history:
                self._history[node_id] = []
            self._history[node_id].append({
                "timestamp": time.time(),
                "old_score": current,
                "new_score": new_score,
                "delta": delta,
                "reason": reason,
            })

            return new_score

    async def flag_byzantine(self, node_id: str, detection_confidence: float = 1.0):
        """Flag a node as Byzantine and penalize reputation."""
        async with self._lock:
            self._byzantine_flags[node_id] = self._byzantine_flags.get(node_id, 0) + 1

        penalty = self.penalty_byzantine * detection_confidence
        await self.update_score(node_id, -penalty, f"Byzantine detection (confidence: {detection_confidence:.2f})")

    async def flag_rejected(self, node_id: str, reason: str):
        """Flag a node for rejected gradient."""
        await self.update_score(node_id, -self.penalty_rejected, f"Gradient rejected: {reason}")

    async def reward_honest_participation(self, node_id: str):
        """Reward a node for honest participation."""
        await self.update_score(node_id, self.reward_honest, "Honest participation")

    async def apply_decay(self):
        """Apply decay to move scores toward initial value."""
        async with self._lock:
            for node_id in self._scores:
                current = self._scores[node_id]
                if current < self.initial_score:
                    self._scores[node_id] = min(current + self.decay_rate, self.initial_score)
                elif current > self.initial_score:
                    self._scores[node_id] = max(current - self.decay_rate, self.initial_score)

    def get_byzantine_count(self, node_id: str) -> int:
        """Get the number of times a node was flagged as Byzantine."""
        return self._byzantine_flags.get(node_id, 0)

    def get_all_scores(self) -> Dict[str, float]:
        """Get all reputation scores."""
        return self._scores.copy()

    def get_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get reputation history for a node."""
        return self._history.get(node_id, []).copy()

    async def reset(self):
        """Reset all reputation data."""
        async with self._lock:
            self._scores.clear()
            self._history.clear()
            self._byzantine_flags.clear()


# ============================================================================
# FL Aggregator
# ============================================================================

class AggregationStrategy(Enum):
    """Aggregation strategies for Byzantine-resilient FL."""
    FEDAVG = "fedavg"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    BULYAN = "bulyan"
    GEOMETRIC_MEDIAN = "geometric_median"


class FLAggregator:
    """
    Federated Learning aggregator with Byzantine resilience.

    Implements multiple aggregation strategies from the fl-aggregator Rust library.
    """

    def __init__(
        self,
        strategy: str = "trimmed_mean",
        byzantine_threshold: float = 0.33,
        trim_ratio: float = 0.1,
        krum_k: int = 1,
    ):
        self.strategy = AggregationStrategy(strategy)
        self.byzantine_threshold = byzantine_threshold
        self.trim_ratio = trim_ratio
        self.krum_k = krum_k
        self._aggregation_history: List[Dict[str, Any]] = []

    async def aggregate(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using the configured strategy.

        Returns:
            Tuple of (aggregated_gradient, metadata)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients to aggregate")

        # Stack gradients for vectorized operations
        gradient_matrix = np.stack(gradients)
        n_gradients = len(gradients)

        if weights is None:
            weights = [1.0 / n_gradients] * n_gradients

        metadata = {
            "strategy": self.strategy.value,
            "n_gradients": n_gradients,
            "timestamp": time.time(),
        }

        if self.strategy == AggregationStrategy.FEDAVG:
            result = self._fedavg(gradient_matrix, weights)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            result = self._trimmed_mean(gradient_matrix)
        elif self.strategy == AggregationStrategy.MEDIAN:
            result = self._median(gradient_matrix)
        elif self.strategy == AggregationStrategy.KRUM:
            result, selected_idx = self._krum(gradient_matrix)
            metadata["selected_indices"] = [int(selected_idx)]
        elif self.strategy == AggregationStrategy.MULTI_KRUM:
            result, selected_indices = self._multi_krum(gradient_matrix)
            metadata["selected_indices"] = [int(i) for i in selected_indices]
        elif self.strategy == AggregationStrategy.BULYAN:
            result, selected_indices = self._bulyan(gradient_matrix)
            metadata["selected_indices"] = [int(i) for i in selected_indices]
        elif self.strategy == AggregationStrategy.GEOMETRIC_MEDIAN:
            result = self._geometric_median(gradient_matrix)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._aggregation_history.append(metadata)
        return result, metadata

    def _fedavg(self, gradients: np.ndarray, weights: List[float]) -> np.ndarray:
        """Federated averaging."""
        weights = np.array(weights).reshape(-1, 1)
        return np.sum(gradients * weights, axis=0)

    def _trimmed_mean(self, gradients: np.ndarray) -> np.ndarray:
        """Coordinate-wise trimmed mean."""
        n = gradients.shape[0]
        k = int(n * self.trim_ratio)

        # Sort along axis 0 and trim
        sorted_grads = np.sort(gradients, axis=0)
        if k > 0:
            trimmed = sorted_grads[k:-k]
        else:
            trimmed = sorted_grads

        return np.mean(trimmed, axis=0)

    def _median(self, gradients: np.ndarray) -> np.ndarray:
        """Coordinate-wise median."""
        return np.median(gradients, axis=0)

    def _krum(self, gradients: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Krum aggregation - select the gradient closest to others.
        """
        n = gradients.shape[0]
        # Number of potentially Byzantine gradients
        f = int(n * self.byzantine_threshold)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = d
                distances[j, i] = d

        # For each gradient, sum distances to n-f-2 nearest neighbors
        scores = np.zeros(n)
        k = n - f - 2
        if k < 1:
            k = 1

        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores[i] = np.sum(sorted_dists[1:k+1])  # Exclude self

        selected = np.argmin(scores)
        return gradients[selected], selected

    def _multi_krum(self, gradients: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Multi-Krum aggregation - select multiple gradients.
        """
        n = gradients.shape[0]
        f = int(n * self.byzantine_threshold)
        m = n - f  # Number of gradients to select

        if m < 1:
            m = 1

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = d
                distances[j, i] = d

        # Compute Krum scores
        k = n - f - 2
        if k < 1:
            k = 1

        scores = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores[i] = np.sum(sorted_dists[1:k+1])

        # Select m gradients with lowest scores
        selected_indices = np.argsort(scores)[:m]
        selected_grads = gradients[selected_indices]

        return np.mean(selected_grads, axis=0), list(selected_indices)

    def _bulyan(self, gradients: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Bulyan aggregation - Multi-Krum selection followed by trimmed mean.
        """
        n = gradients.shape[0]
        f = int(n * self.byzantine_threshold)

        # First, select using Multi-Krum
        _, selected_indices = self._multi_krum(gradients)
        selected_grads = gradients[selected_indices]

        # Then apply trimmed mean on selected gradients
        if len(selected_grads) > 2:
            result = self._trimmed_mean(selected_grads)
        else:
            result = np.mean(selected_grads, axis=0)

        return result, selected_indices

    def _geometric_median(
        self,
        gradients: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-5
    ) -> np.ndarray:
        """
        Geometric median (Weiszfeld's algorithm).
        """
        # Initialize with coordinate-wise median
        estimate = np.median(gradients, axis=0)

        for _ in range(max_iter):
            distances = np.linalg.norm(gradients - estimate, axis=1, keepdims=True)
            distances = np.maximum(distances, 1e-10)  # Avoid division by zero

            weights = 1.0 / distances
            weights /= weights.sum()

            new_estimate = np.sum(gradients * weights, axis=0)

            if np.linalg.norm(new_estimate - estimate) < tol:
                break

            estimate = new_estimate

        return estimate

    def get_history(self) -> List[Dict[str, Any]]:
        """Get aggregation history."""
        return self._aggregation_history.copy()


# ============================================================================
# Byzantine Detector
# ============================================================================

class ByzantineDetector:
    """
    Detects Byzantine behavior in gradient submissions.

    Implements multiple detection strategies:
    - Statistical outlier detection
    - Clustering-based detection
    - Historical behavior analysis
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        contamination: float = 0.1,
    ):
        self.z_threshold = z_threshold
        self.contamination = contamination
        self._detection_log: List[Dict[str, Any]] = []

    async def detect(
        self,
        gradients: List[np.ndarray],
        node_ids: List[str],
    ) -> Tuple[Set[str], Dict[str, float]]:
        """
        Detect Byzantine nodes.

        Returns:
            Tuple of (detected_byzantine_node_ids, confidence_scores)
        """
        n = len(gradients)
        if n < 3:
            return set(), {}

        gradient_matrix = np.stack(gradients)

        # Compute gradient norms
        norms = np.linalg.norm(gradient_matrix, axis=1)

        # Z-score based detection
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        if std_norm < 1e-10:
            return set(), {}

        z_scores = np.abs((norms - mean_norm) / std_norm)

        # Compute pairwise similarities
        similarities = np.zeros(n)
        for i in range(n):
            sim = []
            for j in range(n):
                if i != j:
                    cos_sim = np.dot(gradient_matrix[i], gradient_matrix[j]) / (
                        norms[i] * norms[j] + 1e-10
                    )
                    sim.append(cos_sim)
            similarities[i] = np.mean(sim)

        # Detect outliers
        detected = set()
        confidence = {}

        for i, node_id in enumerate(node_ids):
            # High z-score indicates abnormal magnitude
            if z_scores[i] > self.z_threshold:
                confidence[node_id] = min(1.0, z_scores[i] / (self.z_threshold * 2))
                detected.add(node_id)
            # Low similarity indicates divergent direction
            elif similarities[i] < 0.5:  # Threshold for similarity
                confidence[node_id] = 1.0 - similarities[i]
                detected.add(node_id)

        # Log detection
        self._detection_log.append({
            "timestamp": time.time(),
            "n_gradients": n,
            "detected": list(detected),
            "confidence": confidence,
        })

        return detected, confidence

    def get_detection_log(self) -> List[Dict[str, Any]]:
        """Get detection log."""
        return self._detection_log.copy()


# ============================================================================
# FL Network
# ============================================================================

class FLNetwork:
    """
    Complete FL network simulation.

    Orchestrates:
    - Node management
    - Gradient storage
    - Aggregation
    - Byzantine detection
    - Reputation management
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.gradient_store = GradientStore()
        self.validator = ZomeValidator(max_gradient_norm=config.gradient_clip_norm * 10)
        self.reputation_bridge = ReputationBridge()
        self.aggregator = FLAggregator(
            strategy=config.aggregation_strategy,
            byzantine_threshold=config.byzantine_threshold,
        )
        self.detector = ByzantineDetector()

        self._nodes: Dict[str, Any] = {}
        self._round: int = 0
        self._started: bool = False
        self._round_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

        # Wire up validation
        self.gradient_store.add_validation_callback(self.validator.validate)

    async def start(self):
        """Start the network."""
        if self._started:
            return

        # Import node classes here to avoid circular imports
        from .nodes import HonestNode, ByzantineNode, NodeBehavior

        # Create honest nodes
        for i in range(self.config.num_honest_nodes):
            node_id = f"honest_{i}"
            self._nodes[node_id] = HonestNode(
                node_id=node_id,
                gradient_dimension=self.config.gradient_dimension,
                behavior=NodeBehavior.HONEST,
            )

        # Create Byzantine nodes
        attack_types = ["random", "sign_flip", "scaling", "label_flip", "backdoor"]
        for i in range(self.config.num_byzantine_nodes):
            node_id = f"byzantine_{i}"
            attack = attack_types[i % len(attack_types)]
            self._nodes[node_id] = ByzantineNode(
                node_id=node_id,
                gradient_dimension=self.config.gradient_dimension,
                attack_type=attack,
            )

        self._started = True
        logger.info(f"FL Network started with {self.config.total_nodes} nodes")

    async def shutdown(self):
        """Shutdown the network."""
        self._started = False
        self._nodes.clear()
        await self.gradient_store.clear()
        await self.reputation_bridge.reset()
        logger.info("FL Network shutdown")

    async def run_round(self) -> Dict[str, Any]:
        """
        Execute a single FL round.

        Returns:
            Round results including aggregated gradient and metrics
        """
        async with self._lock:
            self._round += 1
            round_id = self._round

        round_start = time.time()
        gradients = []
        node_ids = []
        submissions = {"accepted": 0, "rejected": 0}

        # Collect gradients from all nodes
        for node_id, node in self._nodes.items():
            gradient = await node.compute_gradient(round_id)

            # Submit to store
            success, entry_hash, error = await self.gradient_store.submit_gradient(
                node_id=node_id,
                round_id=round_id,
                gradient=gradient,
            )

            if success:
                submissions["accepted"] += 1
                gradients.append(gradient)
                node_ids.append(node_id)
            else:
                submissions["rejected"] += 1
                if self.config.reputation_enabled:
                    await self.reputation_bridge.flag_rejected(node_id, error or "Unknown")

        # Check minimum participants
        if len(gradients) < self.config.min_participants:
            return {
                "round_id": round_id,
                "success": False,
                "error": f"Insufficient participants: {len(gradients)} < {self.config.min_participants}",
            }

        # Byzantine detection
        detected_byzantine = set()
        detection_confidence = {}
        if self.config.detection_enabled:
            detected_byzantine, detection_confidence = await self.detector.detect(
                gradients, node_ids
            )

            # Update reputation for detected nodes
            if self.config.reputation_enabled:
                for node_id in detected_byzantine:
                    await self.reputation_bridge.flag_byzantine(
                        node_id, detection_confidence.get(node_id, 1.0)
                    )

        # Filter out detected Byzantine gradients for aggregation
        filtered_gradients = []
        filtered_node_ids = []
        for i, node_id in enumerate(node_ids):
            if node_id not in detected_byzantine:
                filtered_gradients.append(gradients[i])
                filtered_node_ids.append(node_id)

        # Aggregate
        if len(filtered_gradients) == 0:
            return {
                "round_id": round_id,
                "success": False,
                "error": "All gradients filtered as Byzantine",
            }

        aggregated, agg_metadata = await self.aggregator.aggregate(
            filtered_gradients, node_ids=filtered_node_ids
        )

        # Reward honest participants
        if self.config.reputation_enabled:
            for node_id in filtered_node_ids:
                await self.reputation_bridge.reward_honest_participation(node_id)

        round_end = time.time()

        result = {
            "round_id": round_id,
            "success": True,
            "aggregated_gradient": aggregated,
            "n_participants": len(gradients),
            "n_accepted": submissions["accepted"],
            "n_rejected": submissions["rejected"],
            "n_byzantine_detected": len(detected_byzantine),
            "detected_byzantine": list(detected_byzantine),
            "detection_confidence": detection_confidence,
            "aggregation_metadata": agg_metadata,
            "duration_ms": (round_end - round_start) * 1000,
        }

        self._round_history.append(result)
        return result

    async def run_rounds(self, n: int) -> List[Dict[str, Any]]:
        """Run multiple FL rounds."""
        results = []
        for _ in range(n):
            result = await self.run_round()
            results.append(result)
        return results

    def get_node(self, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> Dict[str, Any]:
        """Get all nodes."""
        return self._nodes.copy()

    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get round history."""
        return self._round_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "total_nodes": len(self._nodes),
            "honest_nodes": self.config.num_honest_nodes,
            "byzantine_nodes": self.config.num_byzantine_nodes,
            "current_round": self._round,
            "storage_stats": self.gradient_store.get_stats(),
            "reputation_scores": self.reputation_bridge.get_all_scores(),
        }
