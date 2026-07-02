# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Advanced Sybil attack generators used by the BFT validation harness.

These behaviours simulate sophisticated adversaries that coordinate gradients
across rounds to avoid detection by temporal or entropy-based heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class _DriftState:
    direction: np.ndarray
    progress: float = 0.0


@dataclass
class _EntropyState:
    window: List[np.ndarray] = field(default_factory=list)


class AdvancedSybilGenerator:
    """Coordinator for temporal-drift and entropy-smoothing Sybil behaviours."""

    def __init__(self, *, drift_steps: int = 6, entropy_window: int = 4):
        self.drift_steps = max(1, drift_steps)
        self.entropy_window = max(1, entropy_window)
        self._drift: Dict[int, _DriftState] = {}
        self._entropy: Dict[int, _EntropyState] = {}

    def reset(self) -> None:
        """Clear all per-node state."""
        self._drift.clear()
        self._entropy.clear()

    def temporal_drift(self, node_id: int, honest_gradient: np.ndarray) -> np.ndarray:
        """
        Gradually rotate the gradient away from the honest direction over several
        rounds while injecting small noise to mimic slow concept drift.
        """
        state = self._drift.get(node_id)
        if state is None:
            # Pick an opposing direction with slight randomisation to avoid
            # perfectly mirrored gradients.
            target = -honest_gradient + np.random.normal(0.0, 0.01, size=honest_gradient.shape)
            norm = np.linalg.norm(target) + 1e-9
            state = _DriftState(direction=target / norm)
            self._drift[node_id] = state

        state.progress = min(1.0, state.progress + 1.0 / self.drift_steps)
        blended = (1.0 - state.progress) * honest_gradient + state.progress * state.direction
        jitter = np.random.normal(0.0, 0.02, size=honest_gradient.shape)
        return blended + jitter

    def entropy_smoothing(self, node_id: int, honest_gradient: np.ndarray) -> np.ndarray:
        """
        Maintain a low-entropy moving average of previously sent gradients so the
        distribution appears benign while gradually biasing updates.
        """
        state = self._entropy.setdefault(node_id, _EntropyState())
        if state.window:
            averaged = np.mean(state.window, axis=0)
        else:
            averaged = honest_gradient.copy()

        # Blend toward the honest gradient but damp magnitude to stay near the
        # historical mean, then add slight correlated noise.
        smoothed = 0.7 * averaged + 0.3 * honest_gradient
        correlated_noise = np.sign(smoothed) * 0.01
        advanced = smoothed + correlated_noise

        state.window.append(advanced)
        if len(state.window) > self.entropy_window:
            state.window.pop(0)

        return advanced

    def generate(
        self,
        attack_type: str,
        node_id: int,
        honest_gradient: np.ndarray,
    ) -> np.ndarray:
        attack_type = attack_type.lower()
        if attack_type == "temporal_drift":
            return self.temporal_drift(node_id, honest_gradient)
        if attack_type == "entropy_smoothing":
            return self.entropy_smoothing(node_id, honest_gradient)
        raise ValueError(f"Unknown advanced Sybil attack '{attack_type}'")


__all__ = ["AdvancedSybilGenerator"]
