# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Meta-detector ensemble that wraps PoGQ (GroundTruthDetector) with
directional, density, temporal, and conformal calibration signals.

The detector is intentionally lightweight—no external ML dependencies—so it
can run inside existing experiment scripts. It relies on per-parameter L2
norm features (cheap to compute) to approximate FedGuard-style density scores
and uses a rolling conformal buffer to cap the false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ground_truth_detector import GroundTruthDetector, QualityScore


@dataclass
class MetaScore:
    """Container for the fused signals used by the meta-detector."""

    node_id: int
    pogq_quality: float
    temporal_quality: float
    dirsim: float
    density: float
    outlier_z: float
    reputation: float
    decision: str = "pending"  # accept | reject | abstain
    reason: Optional[str] = None
    flags: Dict[str, bool] = field(default_factory=dict)


class ConformalCalibrator:
    """
    Rolling quantile estimator to cap false-positive rates without extra passes.
    """

    def __init__(self, alpha: float = 0.10, buffer_size: int = 256):
        self.alpha = float(np.clip(alpha, 1e-4, 0.5))
        self.buffer: Deque[float] = deque(maxlen=max(32, buffer_size))

    def update(self, scores: List[float]) -> None:
        """Append reference scores that are believed to be honest."""
        for score in scores:
            if not np.isfinite(score):
                continue
            self.buffer.append(float(score))

    def threshold(self, default: float = 0.0) -> float:
        """Return the quantile used for decisioning."""
        if not self.buffer:
            return default

        arr = np.sort(np.asarray(self.buffer, dtype=np.float64))
        # Empirical quantile: ceil(alpha * (n + 1)) - 1 clipped to [0, n-1]
        pos = int(np.ceil(self.alpha * (len(arr) + 1))) - 1
        pos = min(max(pos, 0), len(arr) - 1)
        return float(arr[pos])


class MetaDetector:
    """
    Ensemble detector that fuses PoGQ utility with FLTrust-style direction,
    lightweight density scoring, temporal smoothing, and conformal calibration.
    """

    def __init__(
        self,
        base_detector: GroundTruthDetector,
        *,
        conformal_alpha: float = 0.10,
        conformal_buffer: int = 256,
        temporal_alpha: float = 0.6,
        dirsim_floor: float = 0.0,
        dirsim_warning: float = 0.05,
        z_max: float = 3.0,
        density_min: float = 0.15,
        density_warning: float = 0.25,
        density_neighbors: int = 3,
        abstain_on_conflict: bool = True,
        teacher_ema: float = 0.7,
        reputation_penalty: float = 0.6,
        reputation_recovery: float = 0.05,
    ):
        self.base = base_detector
        self.calibrator = ConformalCalibrator(conformal_alpha, conformal_buffer)
        self.temporal_alpha = float(np.clip(temporal_alpha, 0.0, 1.0))
        self.dirsim_floor = dirsim_floor
        self.dirsim_warning = dirsim_warning
        self.z_max = z_max
        self.density_min = density_min
        self.density_warning = density_warning
        self.density_neighbors = max(1, density_neighbors)
        self.abstain_on_conflict = abstain_on_conflict
        self.teacher_ema = float(np.clip(teacher_ema, 0.0, 1.0))
        self.reputation_penalty = reputation_penalty
        self.reputation_recovery = reputation_recovery

        self._temporal_scores: Dict[int, float] = {}
        self._reputations: Dict[int, float] = {}
        self._teacher_gradient: Optional[np.ndarray] = None
        self._param_names: List[str] = [
            name for name, _ in self.base.global_model.named_parameters()
        ]
        self._last_summary: Dict[str, float] = {}

    # ------------------------------------------------------------------ Utils
    def _update_temporal(self, node_id: int, instant_quality: float) -> float:
        prev = self._temporal_scores.get(node_id, instant_quality)
        blended = self.temporal_alpha * instant_quality + (1.0 - self.temporal_alpha) * prev
        self._temporal_scores[node_id] = blended
        return blended

    def _update_teacher_gradient(self) -> Optional[np.ndarray]:
        ref = self.base._compute_reference_gradient()
        if ref is None:
            return None
        if self._teacher_gradient is None:
            self._teacher_gradient = ref
        else:
            beta = self.teacher_ema
            self._teacher_gradient = beta * ref + (1.0 - beta) * self._teacher_gradient
        return self._teacher_gradient

    def _dirsim_scores(self, gradients: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, float]:
        teacher = self._update_teacher_gradient()
        if teacher is None:
            return {nid: 0.0 for nid in gradients}

        norm_teacher = np.linalg.norm(teacher) + 1e-12
        scores = {}
        for node_id, grad in gradients.items():
            vec = self.base._flatten_gradient_dict(grad)
            denom = (np.linalg.norm(vec) * norm_teacher) + 1e-12
            cosine = float(np.dot(vec, teacher) / denom)
            scores[node_id] = max(0.0, (cosine + 1.0) / 2.0)  # scale to [0,1]
        return scores

    def _layer_norm_features(self, gradient: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        for name in self._param_names:
            tensor = gradient.get(name)
            if tensor is None:
                feats.append(0.0)
            else:
                feats.append(float(np.linalg.norm(tensor)))
        return np.asarray(feats, dtype=np.float32)

    def _density_scores(
        self,
        feature_matrix: List[np.ndarray],
        node_ids: List[int],
    ) -> Dict[int, float]:
        if not feature_matrix:
            return {nid: 1.0 for nid in node_ids}

        feats = np.vstack(feature_matrix)
        n_clients = feats.shape[0]
        dists = np.zeros((n_clients, n_clients), dtype=np.float32)
        for i in range(n_clients):
            diff = feats - feats[i]
            dists[i] = np.linalg.norm(diff, axis=1)

        densities = []
        for i in range(n_clients):
            row = np.delete(dists[i], i)
            if row.size == 0:
                score = 1.0
            else:
                k = min(self.density_neighbors, row.size)
                nearest = np.partition(row, k - 1)[:k]
                score = 1.0 / (1.0 + float(np.mean(nearest)))
            densities.append(score)

        densities = np.asarray(densities)
        if densities.max() - densities.min() < 1e-6:
            normed = np.ones_like(densities)
        else:
            normed = (densities - densities.min()) / (densities.max() - densities.min())

        return {node_id: float(val) for node_id, val in zip(node_ids, normed)}

    def _compute_outlier_z(self, scores: Dict[int, QualityScore]) -> Dict[int, float]:
        norms = np.asarray([qs.gradient_norm for qs in scores.values()], dtype=np.float64)
        mean = float(np.mean(norms))
        std = float(np.std(norms) + 1e-12)
        return {
            node_id: float(abs(score.gradient_norm - mean) / std)
            for node_id, score in scores.items()
        }

    # ---------------------------------------------------------------- Decisions
    def _decide(
        self,
        score: MetaScore,
        quality_threshold: float,
    ) -> Tuple[str, str]:
        """Apply the rule-based ensemble decision."""
        if score.dirsim <= self.dirsim_floor:
            score.flags["dirsim_reject"] = True
            return "reject", "dirsim"

        if score.outlier_z > self.z_max:
            score.flags["outlier_reject"] = True
            return "reject", "outlier"

        if score.density < self.density_min:
            score.flags["density_reject"] = True
            return "reject", "density"

        signal = 0.6 * score.pogq_quality + 0.4 * score.temporal_quality
        if signal < quality_threshold:
            score.flags["utility_reject"] = True
            return "reject", "utility"

        conflict = (
            score.dirsim < self.dirsim_warning
            and score.density < self.density_warning
        )
        if conflict and self.abstain_on_conflict:
            score.flags["conflict_abstain"] = True
            return "abstain", "conflict"

        return "accept", "ok"

    def detect(
        self,
        gradients: Dict[int, Dict[str, np.ndarray]],
        baseline_stats: Optional[Tuple[float, Dict[int, float], Dict[int, int]]] = None,
    ) -> Tuple[Dict[int, str], Dict[int, MetaScore]]:
        """
        Run the meta-detector. Returns (decision_map, meta_scores).
        """
        if not gradients:
            return {}, {}

        # Compute PoGQ quality scores once.
        if baseline_stats is None:
            baseline_stats = self.base.compute_validation_loss(self.base.global_model)

        pogq_scores: Dict[int, QualityScore] = {}
        feature_vectors: List[np.ndarray] = []
        node_ids: List[int] = []
        for node_id, gradient in gradients.items():
            qscore = self.base.compute_quality(gradient, node_id, baseline_stats)
            pogq_scores[node_id] = qscore
            # Mirror GroundTruthDetector bookkeeping so downstream metrics work.
            self.base.quality_scores.append(qscore)
            self.base.total_evaluated += 1
            if qscore.is_byzantine:
                self.base.detection_count += 1
            feature_vectors.append(self._layer_norm_features(gradient))
            node_ids.append(node_id)

        dirsim = self._dirsim_scores(gradients)
        density = self._density_scores(feature_vectors, node_ids)
        outlier_z = self._compute_outlier_z(pogq_scores)

        meta_scores: Dict[int, MetaScore] = {}
        for node_id, qscore in pogq_scores.items():
            temporal = self._update_temporal(node_id, qscore.quality)
            rep = self._reputations.get(node_id, 1.0)
            meta_scores[node_id] = MetaScore(
                node_id=node_id,
                pogq_quality=qscore.quality,
                temporal_quality=temporal,
                dirsim=dirsim.get(node_id, 0.0),
                density=density.get(node_id, 1.0),
                outlier_z=outlier_z.get(node_id, 0.0),
                reputation=rep,
            )

        threshold = self.calibrator.threshold(default=0.0)

        decisions: Dict[int, str] = {}
        accepted_scores: List[float] = []
        counts = {"accept": 0, "reject": 0, "abstain": 0}
        reasons = {"dirsim": 0, "outlier": 0, "density": 0, "utility": 0, "conflict": 0}

        for node_id, meta in meta_scores.items():
            decision, reason = self._decide(meta, threshold)
            meta.decision = decision
            meta.reason = reason
            decisions[node_id] = decision
            counts[decision] = counts.get(decision, 0) + 1
            reasons[reason] = reasons.get(reason, 0) + 1

            if decision == "accept":
                accepted_scores.append(meta.pogq_quality)
                self._reputations[node_id] = min(1.0, meta.reputation + self.reputation_recovery)
            elif decision == "reject":
                self._reputations[node_id] = self.reputation_penalty * meta.reputation

        self.calibrator.update(accepted_scores)
        self._last_summary = {
            "threshold": threshold,
            "accept": counts["accept"],
            "reject": counts["reject"],
            "abstain": counts["abstain"],
            "reason_dirsim": reasons["dirsim"],
            "reason_outlier": reasons["outlier"],
            "reason_density": reasons["density"],
            "reason_utility": reasons["utility"],
            "reason_conflict": reasons["conflict"],
        }
        return decisions, meta_scores

    def summary(self) -> Dict[str, float]:
        """Return stats from the last detect() call."""
        return dict(self._last_summary)
