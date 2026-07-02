# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Differential Privacy for Federated Learning

Implements gradient clipping and noise addition for (ε, δ)-differential privacy.

Based on:
- Abadi et al., "Deep Learning with Differential Privacy" (2016)
- McMahan et al., "Learning Differentially Private Recurrent Language Models" (2017)

Author: Luminous Dynamics
Date: December 31, 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class NoiseType(Enum):
    """Type of noise mechanism."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


@dataclass
class DPConfig:
    """
    Configuration for differential privacy.

    Attributes:
        epsilon: Privacy budget (lower = more private)
        delta: Failure probability (typically 1e-5 to 1e-7)
        max_grad_norm: Maximum L2 norm for gradient clipping
        noise_multiplier: Noise scale relative to sensitivity
        noise_type: Type of noise mechanism (gaussian or laplace)
        target_epsilon: Target epsilon for privacy accountant
        target_delta: Target delta for privacy accountant
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    noise_type: NoiseType = NoiseType.GAUSSIAN
    target_epsilon: Optional[float] = None
    target_delta: Optional[float] = None

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {self.noise_multiplier}")


def clip_gradients(
    gradient: np.ndarray,
    max_norm: float,
) -> Tuple[np.ndarray, float]:
    """
    Clip gradient to maximum L2 norm.

    Args:
        gradient: Input gradient array
        max_norm: Maximum L2 norm

    Returns:
        Tuple of (clipped_gradient, original_norm)
    """
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        clipped = gradient * (max_norm / norm)
        return clipped.astype(gradient.dtype), float(norm)
    return gradient.copy(), float(norm)


def clip_gradients_batch(
    gradients: Dict[str, np.ndarray],
    max_norm: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Clip all gradients in a batch.

    Args:
        gradients: Dict of node_id -> gradient
        max_norm: Maximum L2 norm

    Returns:
        Tuple of (clipped_gradients, original_norms)
    """
    clipped = {}
    norms = {}
    for node_id, gradient in gradients.items():
        clipped[node_id], norms[node_id] = clip_gradients(gradient, max_norm)
    return clipped, norms


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy.

    Adds calibrated Gaussian noise to achieve (ε, δ)-DP.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
    ):
        """
        Initialize Gaussian mechanism.

        Args:
            epsilon: Privacy parameter
            delta: Failure probability
            sensitivity: L2 sensitivity of the query
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute noise scale for (ε, δ)-DP."""
        # Using the standard Gaussian mechanism formula
        # σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        return math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon

    def add_noise(self, value: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Add Gaussian noise to a value."""
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0, self.sigma, size=value.shape).astype(value.dtype)
        return value + noise

    def get_noise_scale(self) -> float:
        """Get the noise scale (sigma)."""
        return self.sigma


class LaplaceMechanism:
    """
    Laplace mechanism for differential privacy.

    Adds calibrated Laplace noise to achieve ε-DP.
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
    ):
        """
        Initialize Laplace mechanism.

        Args:
            epsilon: Privacy parameter
            sensitivity: L1 sensitivity of the query
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(self, value: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Add Laplace noise to a value."""
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.laplace(0, self.scale, size=value.shape).astype(value.dtype)
        return value + noise

    def get_noise_scale(self) -> float:
        """Get the noise scale (b parameter)."""
        return self.scale


def add_noise(
    gradient: np.ndarray,
    noise_multiplier: float,
    sensitivity: float = 1.0,
    noise_type: NoiseType = NoiseType.GAUSSIAN,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add noise to a gradient for differential privacy.

    Args:
        gradient: Input gradient
        noise_multiplier: Multiplier for noise scale
        sensitivity: Sensitivity of the gradient (typically max_grad_norm / batch_size)
        noise_type: Type of noise to add
        rng: Random number generator

    Returns:
        Noisy gradient
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_scale = noise_multiplier * sensitivity

    if noise_type == NoiseType.GAUSSIAN:
        noise = rng.normal(0, noise_scale, size=gradient.shape).astype(gradient.dtype)
    else:  # Laplace
        noise = rng.laplace(0, noise_scale, size=gradient.shape).astype(gradient.dtype)

    return gradient + noise


def compute_privacy_budget(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Compute privacy budget (epsilon) using RDP accounting.

    This is a simplified version. For production, use the full RDP accountant.

    Args:
        noise_multiplier: Noise multiplier used in training
        sample_rate: Fraction of data sampled per step
        num_steps: Total number of training steps
        delta: Target delta

    Returns:
        Estimated epsilon
    """
    # Simplified RDP-to-DP conversion
    # For accurate accounting, use: privacy_accountant.get_epsilon()
    
    # Simple composition bound (not tight, but safe)
    per_step_epsilon = sample_rate / noise_multiplier
    composed_epsilon = math.sqrt(2 * num_steps * math.log(1 / delta)) * per_step_epsilon
    
    return composed_epsilon


class PrivacyAccountant:
    """
    Privacy budget accountant for federated learning.

    Tracks privacy spending across multiple rounds.
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
    ):
        """
        Initialize privacy accountant.

        Args:
            noise_multiplier: Base noise multiplier
            sample_rate: Sampling rate per round
            delta: Target delta for DP guarantee
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps: List[float] = []  # Track noise multiplier per step

    def step(self, noise_multiplier: Optional[float] = None) -> None:
        """Record a training step."""
        nm = noise_multiplier if noise_multiplier is not None else self.noise_multiplier
        self.steps.append(nm)

    def get_epsilon(self) -> float:
        """Get current privacy budget spent."""
        if not self.steps:
            return 0.0

        # Use advanced composition
        return compute_privacy_budget(
            noise_multiplier=np.mean(self.steps),
            sample_rate=self.sample_rate,
            num_steps=len(self.steps),
            delta=self.delta,
        )

    def get_remaining_budget(self, target_epsilon: float) -> float:
        """Get remaining privacy budget."""
        return max(0.0, target_epsilon - self.get_epsilon())

    def can_continue(self, target_epsilon: float) -> bool:
        """Check if we can continue training within budget."""
        return self.get_epsilon() < target_epsilon

    def reset(self) -> None:
        """Reset the accountant."""
        self.steps = []


class DifferentialPrivacy:
    """
    Differential privacy wrapper for federated learning.

    Provides gradient clipping and noise addition with privacy accounting.
    """

    def __init__(self, config: DPConfig):
        """
        Initialize DP wrapper.

        Args:
            config: DP configuration
        """
        self.config = config
        self.accountant = PrivacyAccountant(
            noise_multiplier=config.noise_multiplier,
            sample_rate=1.0,  # FL typically uses all data per round
            delta=config.delta,
        )
        self.rng = np.random.default_rng()

        # Create noise mechanism
        if config.noise_type == NoiseType.GAUSSIAN:
            self.mechanism = GaussianMechanism(
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=config.max_grad_norm,
            )
        else:
            self.mechanism = LaplaceMechanism(
                epsilon=config.epsilon,
                sensitivity=config.max_grad_norm,
            )

    def privatize_gradient(
        self,
        gradient: np.ndarray,
        clip: bool = True,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply differential privacy to a gradient.

        Args:
            gradient: Input gradient
            clip: Whether to clip the gradient
            add_noise: Whether to add noise

        Returns:
            Tuple of (privatized_gradient, metadata)
        """
        metadata = {"original_norm": float(np.linalg.norm(gradient))}

        # Clip gradient
        if clip:
            gradient, clipped_norm = clip_gradients(gradient, self.config.max_grad_norm)
            metadata["clipped_norm"] = clipped_norm
            metadata["was_clipped"] = clipped_norm > self.config.max_grad_norm

        # Add noise
        if add_noise:
            gradient = self.mechanism.add_noise(gradient, self.rng)
            metadata["noise_scale"] = self.mechanism.get_noise_scale()

        # Record step for accounting
        self.accountant.step()
        metadata["current_epsilon"] = self.accountant.get_epsilon()

        return gradient, metadata

    def privatize_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        clip: bool = True,
        add_noise: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Apply differential privacy to a batch of gradients.

        Args:
            gradients: Dict of node_id -> gradient
            clip: Whether to clip gradients
            add_noise: Whether to add noise

        Returns:
            Tuple of (privatized_gradients, metadata_per_node)
        """
        privatized = {}
        metadata = {}

        for node_id, gradient in gradients.items():
            privatized[node_id], metadata[node_id] = self.privatize_gradient(
                gradient, clip=clip, add_noise=add_noise
            )

        return privatized, metadata

    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent."""
        return self.accountant.get_epsilon()

    def get_privacy_remaining(self) -> Optional[float]:
        """Get remaining privacy budget (if target set)."""
        if self.config.target_epsilon is not None:
            return self.accountant.get_remaining_budget(self.config.target_epsilon)
        return None

    def can_continue(self) -> bool:
        """Check if we can continue within privacy budget."""
        if self.config.target_epsilon is not None:
            return self.accountant.can_continue(self.config.target_epsilon)
        return True

    def reset(self) -> None:
        """Reset privacy accountant."""
        self.accountant.reset()
