# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Honest and Byzantine node simulators for FL testing.

Provides realistic node behavior simulation including:
- Honest nodes with configurable noise
- Byzantine nodes with various attack strategies
- Node lifecycle management
"""

import asyncio
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Node Behavior Types
# ============================================================================

class NodeBehavior(Enum):
    """Node behavior classification."""
    HONEST = "honest"
    BYZANTINE = "byzantine"
    SLOW = "slow"
    INTERMITTENT = "intermittent"


class AttackType(Enum):
    """Byzantine attack types."""
    RANDOM = "random"
    SIGN_FLIP = "sign_flip"
    SCALING = "scaling"
    LABEL_FLIP = "label_flip"
    BACKDOOR = "backdoor"
    GAUSSIAN = "gaussian"
    SAME_VALUE = "same_value"
    LITTLE = "little"  # A Little Is Enough attack
    EMPIRE = "empire"  # Fall of Empires attack
    MIMIC = "mimic"  # Mimic honest behavior with subtle poison


# ============================================================================
# Node Metrics
# ============================================================================

@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    gradients_submitted: int = 0
    gradients_accepted: int = 0
    gradients_rejected: int = 0
    byzantine_detections: int = 0
    total_compute_time_ms: float = 0.0
    average_gradient_norm: float = 0.0
    last_active_round: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "gradients_submitted": self.gradients_submitted,
            "gradients_accepted": self.gradients_accepted,
            "gradients_rejected": self.gradients_rejected,
            "byzantine_detections": self.byzantine_detections,
            "total_compute_time_ms": self.total_compute_time_ms,
            "average_gradient_norm": self.average_gradient_norm,
            "last_active_round": self.last_active_round,
        }


# ============================================================================
# Base Node
# ============================================================================

class BaseNode:
    """Base class for FL nodes."""

    def __init__(
        self,
        node_id: str,
        gradient_dimension: int,
        behavior: NodeBehavior,
    ):
        self.node_id = node_id
        self.gradient_dimension = gradient_dimension
        self.behavior = behavior
        self.metrics = NodeMetrics(node_id=node_id)
        self._local_model: Optional[np.ndarray] = None
        self._gradient_history: List[np.ndarray] = []
        self._rng = np.random.RandomState()

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute gradient for a round. Must be implemented by subclasses."""
        raise NotImplementedError

    async def receive_aggregated_gradient(self, gradient: np.ndarray, round_id: int):
        """Receive and apply aggregated gradient."""
        if self._local_model is None:
            self._local_model = np.zeros(self.gradient_dimension)
        self._local_model += gradient

    def get_metrics(self) -> NodeMetrics:
        """Get node metrics."""
        return self.metrics

    def reset(self):
        """Reset node state."""
        self._local_model = None
        self._gradient_history.clear()
        self.metrics = NodeMetrics(node_id=self.node_id)


# ============================================================================
# Honest Node
# ============================================================================

class HonestNode(BaseNode):
    """
    Simulates an honest FL participant.

    Generates gradients based on:
    - Local model state
    - Simulated local data characteristics
    - Configurable noise levels
    """

    def __init__(
        self,
        node_id: str,
        gradient_dimension: int,
        behavior: NodeBehavior = NodeBehavior.HONEST,
        noise_scale: float = 0.01,
        base_gradient_mean: float = 0.0,
        base_gradient_std: float = 1.0,
        local_data_size: int = 1000,
        dropout_probability: float = 0.0,
        latency_ms: float = 0.0,
    ):
        super().__init__(node_id, gradient_dimension, behavior)
        self.noise_scale = noise_scale
        self.base_gradient_mean = base_gradient_mean
        self.base_gradient_std = base_gradient_std
        self.local_data_size = local_data_size
        self.dropout_probability = dropout_probability
        self.latency_ms = latency_ms

        # Initialize with consistent base gradient direction
        self._base_direction = self._rng.randn(gradient_dimension)
        self._base_direction /= np.linalg.norm(self._base_direction)

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute an honest gradient."""
        start_time = time.time()

        # Simulate dropout
        if self._rng.random() < self.dropout_probability:
            raise RuntimeError(f"Node {self.node_id} dropped out of round {round_id}")

        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        # Generate base gradient
        gradient = self._base_direction * self.base_gradient_std + self.base_gradient_mean

        # Add noise proportional to data size (less data = more noise)
        noise = self._rng.randn(self.gradient_dimension) * self.noise_scale
        noise *= np.sqrt(1000 / self.local_data_size)  # Scale noise by data size
        gradient += noise

        # Round-dependent evolution (simulates convergence)
        decay = 1.0 / (1.0 + 0.01 * round_id)
        gradient *= decay

        # Update metrics
        compute_time = (time.time() - start_time) * 1000
        self.metrics.gradients_submitted += 1
        self.metrics.total_compute_time_ms += compute_time
        self.metrics.last_active_round = round_id

        norm = np.linalg.norm(gradient)
        if self.metrics.average_gradient_norm == 0:
            self.metrics.average_gradient_norm = norm
        else:
            # Running average
            self.metrics.average_gradient_norm = (
                0.9 * self.metrics.average_gradient_norm + 0.1 * norm
            )

        self._gradient_history.append(gradient)
        return gradient.astype(np.float32)


# ============================================================================
# Byzantine Node
# ============================================================================

class ByzantineNode(BaseNode):
    """
    Simulates a Byzantine (malicious) FL participant.

    Implements various attack strategies:
    - Random: Submit random gradients
    - Sign Flip: Flip the sign of honest gradients
    - Scaling: Scale gradients by large factor
    - Label Flip: Simulate label flipping attack
    - Backdoor: Inject backdoor patterns
    - Little: "A Little Is Enough" attack
    - Empire: "Fall of Empires" attack
    """

    def __init__(
        self,
        node_id: str,
        gradient_dimension: int,
        attack_type: str = "random",
        attack_params: Optional[Dict[str, Any]] = None,
        stealth_mode: bool = False,
    ):
        super().__init__(node_id, gradient_dimension, NodeBehavior.BYZANTINE)
        self.attack_type = AttackType(attack_type)
        self.attack_params = attack_params or {}
        self.stealth_mode = stealth_mode

        # For attacks that need to know honest behavior
        self._observed_gradients: List[np.ndarray] = []
        self._attack_round = 0

    def observe_gradient(self, gradient: np.ndarray):
        """Observe an honest gradient (for adaptive attacks)."""
        self._observed_gradients.append(gradient.copy())
        if len(self._observed_gradients) > 10:
            self._observed_gradients.pop(0)

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute a malicious gradient based on attack type."""
        start_time = time.time()
        self._attack_round = round_id

        if self.attack_type == AttackType.RANDOM:
            gradient = self._random_attack()
        elif self.attack_type == AttackType.SIGN_FLIP:
            gradient = self._sign_flip_attack()
        elif self.attack_type == AttackType.SCALING:
            gradient = self._scaling_attack()
        elif self.attack_type == AttackType.LABEL_FLIP:
            gradient = self._label_flip_attack()
        elif self.attack_type == AttackType.BACKDOOR:
            gradient = self._backdoor_attack()
        elif self.attack_type == AttackType.GAUSSIAN:
            gradient = self._gaussian_attack()
        elif self.attack_type == AttackType.SAME_VALUE:
            gradient = self._same_value_attack()
        elif self.attack_type == AttackType.LITTLE:
            gradient = self._little_attack()
        elif self.attack_type == AttackType.EMPIRE:
            gradient = self._empire_attack()
        elif self.attack_type == AttackType.MIMIC:
            gradient = self._mimic_attack()
        else:
            gradient = self._random_attack()

        # Apply stealth mode (make attack less detectable)
        if self.stealth_mode:
            gradient = self._apply_stealth(gradient)

        # Update metrics
        compute_time = (time.time() - start_time) * 1000
        self.metrics.gradients_submitted += 1
        self.metrics.total_compute_time_ms += compute_time
        self.metrics.last_active_round = round_id

        self._gradient_history.append(gradient)
        return gradient.astype(np.float32)

    def _random_attack(self) -> np.ndarray:
        """Generate random gradient."""
        scale = self.attack_params.get("scale", 10.0)
        return self._rng.randn(self.gradient_dimension) * scale

    def _sign_flip_attack(self) -> np.ndarray:
        """Flip sign of estimated honest gradient."""
        if self._observed_gradients:
            honest = np.mean(self._observed_gradients, axis=0)
        else:
            honest = self._rng.randn(self.gradient_dimension)

        scale = self.attack_params.get("scale", 1.0)
        return -honest * scale

    def _scaling_attack(self) -> np.ndarray:
        """Scale gradient by large factor."""
        scale = self.attack_params.get("scale", 100.0)
        base = self._rng.randn(self.gradient_dimension)
        return base * scale

    def _label_flip_attack(self) -> np.ndarray:
        """Simulate label flipping - produce gradient in wrong direction."""
        if self._observed_gradients:
            honest = np.mean(self._observed_gradients, axis=0)
            # Rotate gradient 180 degrees
            return -honest
        else:
            return self._rng.randn(self.gradient_dimension)

    def _backdoor_attack(self) -> np.ndarray:
        """Inject backdoor pattern."""
        backdoor_pattern = self.attack_params.get("pattern", None)
        if backdoor_pattern is None:
            # Create a subtle pattern that affects specific indices
            backdoor_pattern = np.zeros(self.gradient_dimension)
            trigger_indices = self.attack_params.get(
                "trigger_indices",
                list(range(0, min(100, self.gradient_dimension), 10))
            )
            backdoor_pattern[trigger_indices] = self.attack_params.get("trigger_value", 5.0)

        # Mix with honest-looking gradient
        base = self._rng.randn(self.gradient_dimension) * 0.1
        blend = self.attack_params.get("blend", 0.5)

        return base * (1 - blend) + backdoor_pattern * blend

    def _gaussian_attack(self) -> np.ndarray:
        """Gaussian noise attack."""
        mean = self.attack_params.get("mean", 0.0)
        std = self.attack_params.get("std", 5.0)
        return self._rng.normal(mean, std, self.gradient_dimension)

    def _same_value_attack(self) -> np.ndarray:
        """All elements have same value."""
        value = self.attack_params.get("value", 1.0)
        return np.full(self.gradient_dimension, value)

    def _little_attack(self) -> np.ndarray:
        """
        'A Little Is Enough' attack.
        Slightly modifies the average gradient to cause slow poisoning.
        """
        if not self._observed_gradients:
            return self._random_attack()

        honest_mean = np.mean(self._observed_gradients, axis=0)
        honest_std = np.std(self._observed_gradients, axis=0)

        # Stay within detection threshold but push in adversarial direction
        epsilon = self.attack_params.get("epsilon", 0.5)
        direction = self.attack_params.get("direction", -1)  # -1 for opposite

        return honest_mean + direction * epsilon * honest_std

    def _empire_attack(self) -> np.ndarray:
        """
        'Fall of Empires' attack.
        Collude to shift the aggregate in adversarial direction.
        """
        if not self._observed_gradients:
            return self._random_attack()

        honest_mean = np.mean(self._observed_gradients, axis=0)

        # Compute attack gradient that will shift aggregate
        n_byzantine = self.attack_params.get("n_byzantine", 1)
        n_total = self.attack_params.get("n_total", 10)
        target_shift = self.attack_params.get("target_shift", None)

        if target_shift is None:
            # Default: shift in opposite direction
            target_shift = -honest_mean * 2

        # Calculate required Byzantine gradient to achieve shift
        # Aggregate = (n_honest * honest_mean + n_byzantine * byzantine) / n_total
        # We want: byzantine = (target * n_total - n_honest * honest_mean) / n_byzantine
        n_honest = n_total - n_byzantine
        if n_byzantine > 0:
            byzantine = (target_shift * n_total - n_honest * honest_mean) / n_byzantine
        else:
            byzantine = target_shift

        return byzantine

    def _mimic_attack(self) -> np.ndarray:
        """
        Mimic attack - behave honestly but with subtle poison.
        Hard to detect because it looks like honest behavior.
        """
        if not self._observed_gradients:
            return self._rng.randn(self.gradient_dimension)

        # Start with honest-looking gradient
        honest_mean = np.mean(self._observed_gradients, axis=0)
        honest_std = np.std(self._observed_gradients, axis=0)

        # Add honest-level noise
        gradient = honest_mean + self._rng.randn(self.gradient_dimension) * np.mean(honest_std)

        # Add very subtle poison
        poison_strength = self.attack_params.get("poison_strength", 0.01)
        poison_direction = self._rng.randn(self.gradient_dimension)
        poison_direction /= np.linalg.norm(poison_direction)

        return gradient + poison_direction * poison_strength * np.linalg.norm(gradient)

    def _apply_stealth(self, gradient: np.ndarray) -> np.ndarray:
        """Apply stealth modifications to avoid detection."""
        if not self._observed_gradients:
            return gradient

        honest_mean = np.mean(self._observed_gradients, axis=0)
        honest_std = np.std(self._observed_gradients, axis=0) + 1e-10

        # Clip to be within reasonable range of honest gradients
        z_limit = self.attack_params.get("z_limit", 2.0)

        clipped = np.clip(
            gradient,
            honest_mean - z_limit * honest_std,
            honest_mean + z_limit * honest_std
        )

        return clipped


# ============================================================================
# Node Factory
# ============================================================================

class NodeFactory:
    """Factory for creating various node types."""

    @staticmethod
    def create_honest_node(
        node_id: str,
        gradient_dim: int,
        **kwargs
    ) -> HonestNode:
        """Create an honest node."""
        return HonestNode(
            node_id=node_id,
            gradient_dimension=gradient_dim,
            **kwargs
        )

    @staticmethod
    def create_byzantine_node(
        node_id: str,
        gradient_dim: int,
        attack_type: str = "random",
        **kwargs
    ) -> ByzantineNode:
        """Create a Byzantine node."""
        return ByzantineNode(
            node_id=node_id,
            gradient_dimension=gradient_dim,
            attack_type=attack_type,
            **kwargs
        )

    @staticmethod
    def create_mixed_network(
        n_honest: int,
        n_byzantine: int,
        gradient_dim: int,
        attack_types: Optional[List[str]] = None,
    ) -> Tuple[List[HonestNode], List[ByzantineNode]]:
        """Create a mixed network of honest and Byzantine nodes."""
        if attack_types is None:
            attack_types = ["random", "sign_flip", "scaling"]

        honest_nodes = [
            NodeFactory.create_honest_node(f"honest_{i}", gradient_dim)
            for i in range(n_honest)
        ]

        byzantine_nodes = [
            NodeFactory.create_byzantine_node(
                f"byzantine_{i}",
                gradient_dim,
                attack_type=attack_types[i % len(attack_types)]
            )
            for i in range(n_byzantine)
        ]

        return honest_nodes, byzantine_nodes


# ============================================================================
# Slow/Intermittent Nodes
# ============================================================================

class SlowNode(HonestNode):
    """A node that is slow to respond."""

    def __init__(
        self,
        node_id: str,
        gradient_dimension: int,
        delay_ms: float = 1000.0,
        delay_variance: float = 200.0,
        **kwargs
    ):
        super().__init__(node_id, gradient_dimension, behavior=NodeBehavior.SLOW, **kwargs)
        self.delay_ms = delay_ms
        self.delay_variance = delay_variance

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute gradient with artificial delay."""
        delay = self.delay_ms + self._rng.randn() * self.delay_variance
        delay = max(0, delay)
        await asyncio.sleep(delay / 1000)
        return await super().compute_gradient(round_id)


class IntermittentNode(HonestNode):
    """A node that occasionally fails to respond."""

    def __init__(
        self,
        node_id: str,
        gradient_dimension: int,
        failure_probability: float = 0.2,
        **kwargs
    ):
        super().__init__(
            node_id, gradient_dimension, behavior=NodeBehavior.INTERMITTENT, **kwargs
        )
        self.failure_probability = failure_probability

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute gradient with chance of failure."""
        if self._rng.random() < self.failure_probability:
            raise ConnectionError(f"Node {self.node_id} failed in round {round_id}")
        return await super().compute_gradient(round_id)
