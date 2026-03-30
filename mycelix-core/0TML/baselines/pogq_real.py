#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Robust Proof of Gradient Quality (PoGQ) implementation used across the 0TML
experimental suite. This module replaces the archived mean-based variant with
coordinate-wise robust statistics so detection remains stable even when nearly
half of participating nodes are Byzantine.

Exports:
    - analyze_gradient_quality: Standalone robust PoGQ scoring helper
    - PoGQServer: FedAvg-compatible server that filters client updates using
      PoGQ scores and lightweight reputation tracking
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from baselines.fedavg import FedAvgConfig, FedAvgServer


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1] with zero-norm protection."""
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0

    return float(np.dot(a, b) / denom)


def _robust_deviation(target: np.ndarray, centre: np.ndarray, mad: np.ndarray) -> float:
    """
    Compute a robust (MAD-normalised) deviation score.

    Returns median absolute deviation scaled to be comparable across layers.
    Larger values imply the gradient diverges significantly from the honest
    cluster. The result is non-negative.
    """
    abs_distance = np.abs(target - centre)
    safe_mad = np.where(mad < 1e-12, 1e-12, mad)
    normalised = abs_distance / safe_mad
    return float(np.median(normalised))


def analyze_gradient_quality(
    gradient: np.ndarray,
    reference_gradients: Sequence[np.ndarray],
) -> float:
    """
    Compute a PoGQ score in [0, 1] for *gradient* relative to a reference set.

    The score combines:
        • Cosine similarity against the coordinate-wise median (robust centre)
        • Median absolute deviation (MAD) penalty for outliers
        • Magnitude penalty discouraging extremely large gradients

    The function is intentionally stateless so it can be reused in tests,
    demos, and production aggregation.
    """
    if not reference_gradients:
        # Neutral score when no reference exists (e.g., first round)
        return 0.5

    gradient = np.asarray(gradient, dtype=np.float64)
    references = np.stack([np.asarray(g, dtype=np.float64) for g in reference_gradients], axis=0)

    # Coordinate-wise robust statistics
    median_grad = np.median(references, axis=0)
    mad = np.median(np.abs(references - median_grad), axis=0)

    # 1) Directional agreement
    cosine = _cosine_similarity(gradient, median_grad)
    direction_score = (cosine + 1.0) / 2.0  # map [-1, 1] → [0, 1]

    # 2) Deviation penalty using MAD (robust z-score analogue)
    deviation = _robust_deviation(gradient, median_grad, mad)
    deviation_penalty = float(np.exp(-max(0.0, deviation - 1.0)))  # tolerate mild noise

    # 3) Magnitude penalty (protect against gradient explosion)
    median_norm = float(np.linalg.norm(median_grad) + 1e-12)
    gradient_norm = float(np.linalg.norm(gradient))
    magnitude_ratio = gradient_norm / median_norm if median_norm > 0 else 1.0
    magnitude_penalty = float(np.exp(-max(0.0, magnitude_ratio - 1.25)))

    score = direction_score * deviation_penalty * magnitude_penalty
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class PoGQConfig:
    """Configuration parameters for PoGQServer."""

    quality_threshold: float = 0.7
    adaptive_threshold: bool = True
    min_threshold: float = 0.35
    max_threshold: float = 0.95
    min_reference: int = 3
    history_window: int = 5
    reputation_decay: float = 0.9
    initial_reputation: float = 0.7
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 32
    num_clients: int = 10
    fraction_clients: float = 1.0


class PoGQServer(FedAvgServer):
    """
    FedAvg-compatible server that filters client updates using PoGQ scores
    and lightweight reputation tracking.
    """

    def __init__(self, model, config: Optional[Dict] = None, device: str = "cpu"):
        cfg = PoGQConfig(**(config or {}))

        fedavg_cfg = FedAvgConfig(
            learning_rate=cfg.learning_rate,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            num_clients=cfg.num_clients,
            fraction_clients=cfg.fraction_clients,
        )

        super().__init__(model, fedavg_cfg, device=device)

        self.cfg = cfg
        self.reputations: defaultdict[str, float] = defaultdict(lambda: cfg.initial_reputation)
        self.gradient_history: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=cfg.history_window)
        )
        self.last_detection_summary: Dict[str, object] = {}

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _flatten_layers(self, layer_deltas: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(layer).ravel() for layer in layer_deltas])

    def _temporal_consistency(self, node_id: str, gradient: np.ndarray) -> float:
        history = self.gradient_history[node_id]
        if not history:
            return 1.0

        sims = []
        for past in history:
            sims.append((_cosine_similarity(gradient, past) + 1.0) / 2.0)

        return float(np.mean(sims)) if sims else 1.0

    def _update_reputation(self, node_id: str, quality: float):
        current = self.reputations[node_id]
        new_score = (
            self.cfg.reputation_decay * current + (1.0 - self.cfg.reputation_decay) * quality
        )
        self.reputations[node_id] = float(np.clip(new_score, 0.0, 1.0))

    # --------------------------------------------------------------------- #
    # FedAvg overrides
    # --------------------------------------------------------------------- #
    def aggregate(self, client_updates: List[Dict]) -> List[np.ndarray]:
        if not client_updates:
            return self.get_model_weights()

        global_weights = self.get_model_weights()

        # Compute flattened gradients per client (global - local weights)
        flattened_gradients: List[np.ndarray] = []
        layer_deltas: List[List[np.ndarray]] = []

        for update in client_updates:
            deltas = [
                np.asarray(global_layer, dtype=np.float64) - np.asarray(layer, dtype=np.float64)
                for global_layer, layer in zip(global_weights, update["weights"])
            ]
            layer_deltas.append(deltas)
            flattened_gradients.append(self._flatten_layers(deltas))

        quality_scores: List[float] = []
        combined_scores: List[float] = []
        node_ids: List[str] = []

        for idx, gradient in enumerate(flattened_gradients):
            update = client_updates[idx]
            node_id = str(update.get("client_id", idx))
            node_ids.append(node_id)

            references = [g for j, g in enumerate(flattened_gradients) if j != idx]
            if len(references) < self.cfg.min_reference:
                quality = analyze_gradient_quality(gradient, flattened_gradients)
            else:
                quality = analyze_gradient_quality(gradient, references)

            temporal = self._temporal_consistency(node_id, gradient)
            # Combine directional quality with temporal consistency (30% weight)
            combined = 0.7 * quality + 0.3 * temporal

            quality_scores.append(float(quality))
            combined_scores.append(float(combined))

            self._update_reputation(node_id, combined)
            self.gradient_history[node_id].append(gradient)

        # Adaptive thresholding (IQR + MAD blend)
        threshold = self.cfg.quality_threshold
        if self.cfg.adaptive_threshold and len(combined_scores) >= 3:
            scores_arr = np.asarray(combined_scores)
            q1 = float(np.percentile(scores_arr, 25))
            q3 = float(np.percentile(scores_arr, 75))
            iqr_threshold = q1 - 1.5 * (q3 - q1)

            median = float(np.median(scores_arr))
            mad = float(np.median(np.abs(scores_arr - median)))
            mad_threshold = median - 3.0 * mad

            threshold = max(iqr_threshold, mad_threshold, threshold)
            threshold = float(
                np.clip(threshold, self.cfg.min_threshold, self.cfg.max_threshold)
            )

        accepted_indices: List[int] = [
            idx for idx, score in enumerate(combined_scores) if score >= threshold
        ]
        if not accepted_indices:
            # Fall back to single best contributor to preserve progress
            accepted_indices = [int(np.argmax(combined_scores))]

        # Weighted aggregation using sample count × reputation
        total_weight = 0.0
        aggregated_layers = [
            np.zeros_like(client_updates[0]["weights"][layer_idx])
            for layer_idx in range(len(global_weights))
        ]

        for idx in accepted_indices:
            update = client_updates[idx]
            node_id = node_ids[idx]
            rep = self.reputations[node_id]
            quality = combined_scores[idx]
            num_samples = float(update.get("num_samples", 1))

            contribution_weight = max(1e-6, num_samples * (0.5 + 0.5 * rep) * (0.5 + 0.5 * quality))
            total_weight += contribution_weight

            for layer_idx, layer_weights in enumerate(update["weights"]):
                aggregated_layers[layer_idx] += contribution_weight * np.asarray(layer_weights)

        if total_weight == 0.0:
            # Should rarely trigger; revert to vanilla FedAvg aggregation
            return super().aggregate(client_updates)

        aggregated = [layer / total_weight for layer in aggregated_layers]

        self.last_detection_summary = {
            "threshold": threshold,
            "quality_scores": quality_scores,
            "combined_scores": combined_scores,
            "accepted_clients": [node_ids[idx] for idx in accepted_indices],
            "rejected_clients": [
                node_ids[idx] for idx in range(len(node_ids)) if idx not in accepted_indices
            ],
        }

        return aggregated


__all__ = ["analyze_gradient_quality", "PoGQServer"]
