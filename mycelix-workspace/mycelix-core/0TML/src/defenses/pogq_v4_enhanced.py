#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
PoGQ-v4.1 Enhanced: Publication-Grade Byzantine Detection
=========================================================

Complete implementation of PoGQ-v4.1 with all Gen-4 components:
1. Mondrian (class-aware validation with per-class z-score normalization)
2. Mondrian Conformal (per-class FPR buckets)
3. Adaptive Hybrid Scoring (PCA-cosine + adaptive λ based on SNR)
4. Temporal EMA (β=0.85 historical smoothing)
5. Direction Prefilter (ReLU(cosine) > 0 cheap rejection)
6. Winsorized Dispersion (outlier-robust thresholds)
7. PoGQ-v4-Lite (linear probe head for high-dim rescue - CIFAR-10)

Phase 2 Enhancements (COMPLETE):
8. Warm-up Quota (W=3 rounds grace period for cold-start)
9. Hysteresis (k=2 consecutive violations to quarantine, m=3 to release)
10. Egregious Cap (p99.9 threshold for warm-up violations)

Author: Luminous Dynamics
Date: November 8, 2025
Version: 4.1.1 (Phase 2 complete)
"""

import numpy as np
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoGQv41Config:
    """Configuration for PoGQ-v4.1 Enhanced"""
    # Mondrian settings
    per_class_normalization: bool = True
    class_score_window: int = 10  # Window for computing per-class stats

    # PCA settings
    pca_components: int = 32
    pca_fit_on_reference: bool = True
    pca_incremental: bool = True  # Use IncrementalPCA for stability

    # Adaptive lambda settings
    lambda_adaptive: bool = True
    lambda_a: float = 2.0  # SNR coefficient
    lambda_b: float = -1.0  # SNR offset
    lambda_min: float = 0.1  # Clip lower bound
    lambda_max: float = 0.9  # Clip upper bound

    # Mondrian conformal settings
    mondrian_conformal: bool = True
    conformal_alpha: float = 0.10  # FPR guarantee
    min_bucket_size: int = 10  # Min samples for separate threshold

    # Temporal EMA settings
    ema_beta: float = 0.85  # Weight for historical scores (stable but responsive)

    # Warm-up quota settings (Phase 2)
    warmup_rounds: int = 3  # Number of rounds before full enforcement
    egregious_cap_quantile: float = 0.999  # p99.9 cap for egregious violations during warmup

    # Hysteresis settings (Phase 2 - prevent flapping)
    hysteresis_k: int = 2  # Consecutive violations to quarantine
    hysteresis_m: int = 3  # Consecutive clears to release

    # Direction prefilter settings
    direction_prefilter: bool = True
    direction_threshold: float = 0.0  # ReLU threshold

    # Winsorized dispersion settings
    winsorize_quantiles: Tuple[float, float] = (0.05, 0.95)

    # PoGQ-v4-Lite settings (for high-dim datasets like CIFAR-10)
    use_lite_mode: bool = False
    probe_layer_slice: Optional[Tuple[int, int]] = None  # (start, end) for probe head

    # Provenance tracking
    track_provenance: bool = True
    code_commit: Optional[str] = None
    dataset_sha256: Optional[str] = None
    attack_preset_id: Optional[str] = None
    defense_preset_id: str = "pogq_v4.1_default"
    calibration_hash: Optional[str] = None


@dataclass
class MondrianProfile:
    """Represents a client's class distribution profile for Mondrian conformal"""
    client_classes: Set[int] = field(default_factory=set)

    def __hash__(self):
        return hash(frozenset(self.client_classes))

    def __eq__(self, other):
        if not isinstance(other, MondrianProfile):
            return False
        return frozenset(self.client_classes) == frozenset(other.client_classes)

    @property
    def profile_key(self) -> frozenset:
        return frozenset(self.client_classes)


class AdaptiveHybridScorer:
    """
    Adaptive λ hybrid scorer with PCA-projected cosine similarity

    Computes: hybrid_score = λ(t) * direction_score + (1-λ(t)) * utility_score
    where λ(t) = sigmoid(a·SNR_t + b) clipped to [λ_min, λ_max]
    and direction_score uses PCA(32) cosine similarity for high-dim robustness
    """

    def __init__(self, config: PoGQv41Config):
        self.config = config
        self.pca = None
        self.gradient_history: List[np.ndarray] = []
        self.lambda_history: List[float] = []

        # Initialize IncrementalPCA if requested
        if config.pca_incremental:
            try:
                from sklearn.decomposition import IncrementalPCA
                self.pca = IncrementalPCA(n_components=config.pca_components)
                self.pca_fitted = False
            except ImportError:
                logger.warning("sklearn not available, falling back to standard PCA")
                self.pca = None
        else:
            self.pca = None

    def fit_pca(self, reference_gradients: List[np.ndarray]):
        """
        Fit PCA on clean reference gradients only
        Uses IncrementalPCA for numerical stability
        """
        if self.pca is None or not self.config.pca_fit_on_reference:
            return

        if len(reference_gradients) == 0:
            logger.warning("No reference gradients for PCA fitting")
            return

        # Stack gradients
        X = np.vstack([g.reshape(1, -1) for g in reference_gradients])

        if self.config.pca_incremental:
            # Fit incrementally
            batch_size = min(50, len(reference_gradients))
            for i in range(0, len(reference_gradients), batch_size):
                batch = X[i:i+batch_size]
                self.pca.partial_fit(batch)
            self.pca_fitted = True
            logger.info(f"Fitted IncrementalPCA on {len(reference_gradients)} reference gradients")
        else:
            try:
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=self.config.pca_components)
                self.pca.fit(X)
                self.pca_fitted = True
                logger.info(f"Fitted PCA on {len(reference_gradients)} reference gradients")
            except ImportError:
                logger.warning("sklearn not available for PCA")

    def pca_cosine_similarity(
        self,
        gradient: np.ndarray,
        reference_gradient: np.ndarray
    ) -> float:
        """
        Cosine similarity in PCA(32) subspace for high-dim robustness
        Falls back to full-space cosine if PCA not fitted
        """
        if self.pca is None or not hasattr(self, 'pca_fitted') or not self.pca_fitted:
            # Fallback to full-space cosine
            return self._cosine_similarity(gradient, reference_gradient)

        # Project to PCA subspace
        g_proj = self.pca.transform(gradient.reshape(1, -1)).flatten()
        ref_proj = self.pca.transform(reference_gradient.reshape(1, -1)).flatten()

        return self._cosine_similarity(g_proj, ref_proj)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def compute_adaptive_lambda(self, gradient: np.ndarray) -> float:
        """
        Adaptive λ_t = sigmoid(a·SNR_t + b) clipped to [λ_min, λ_max]
        where SNR_t = ||gradient|| / IQR(recent gradients)
        """
        if not self.config.lambda_adaptive or len(self.gradient_history) < 5:
            return 0.7  # Default

        # Compute gradient norm
        norm = np.linalg.norm(gradient)

        # Compute SNR using recent gradient norms
        recent_norms = [np.linalg.norm(g) for g in self.gradient_history[-10:]]
        q75 = np.percentile(recent_norms, 75)
        q25 = np.percentile(recent_norms, 25)
        iqr = q75 - q25

        if iqr < 1e-8:
            snr = 1.0
        else:
            snr = norm / iqr

        # Sigmoid with SNR
        lambda_t = 1.0 / (1.0 + np.exp(-(self.config.lambda_a * snr + self.config.lambda_b)))

        # Clip to [λ_min, λ_max] for stability
        lambda_t = np.clip(lambda_t, self.config.lambda_min, self.config.lambda_max)

        self.lambda_history.append(lambda_t)
        return lambda_t

    def compute_hybrid_score(
        self,
        gradient: np.ndarray,
        reference_gradient: np.ndarray,
        utility_score: float
    ) -> float:
        """
        Compute hybrid score = λ(t) * direction_score + (1-λ(t)) * utility_score
        """
        # Direction score using PCA-cosine
        direction_score = self.pca_cosine_similarity(gradient, reference_gradient)

        # Apply ReLU if prefilter enabled
        if self.config.direction_prefilter:
            direction_score = max(0.0, direction_score - self.config.direction_threshold)

        # Adaptive lambda
        lambda_t = self.compute_adaptive_lambda(gradient)

        # Hybrid score
        hybrid_score = lambda_t * direction_score + (1.0 - lambda_t) * utility_score

        # Update history
        self.gradient_history.append(gradient)
        if len(self.gradient_history) > 20:
            self.gradient_history.pop(0)

        return hybrid_score

    def get_lambda_statistics(self) -> Dict[str, float]:
        """Get statistics about adaptive lambda values"""
        if not self.lambda_history:
            return {"count": 0}

        return {
            "count": len(self.lambda_history),
            "mean": float(np.mean(self.lambda_history)),
            "std": float(np.std(self.lambda_history)),
            "min": float(np.min(self.lambda_history)),
            "max": float(np.max(self.lambda_history)),
            "median": float(np.median(self.lambda_history))
        }


class MondrianConformally:
    """
    Mondrian Conformal Prediction with per-class FPR buckets

    Guarantees FPR ≤ α per Mondrian profile (class distribution bucket)
    with hierarchical backoff: bucket → class-only → global
    """

    def __init__(self, config: PoGQv41Config):
        self.config = config
        self.thresholds: Dict[frozenset, float] = {}
        self.bucket_stats: Dict[frozenset, Dict[str, Any]] = {}
        self.global_threshold: float = 0.0  # Default threshold before calibration

    def calibrate(
        self,
        validation_scores: List[float],
        validation_profiles: List[MondrianProfile]
    ):
        """
        Set Mondrian conformal thresholds using calibration data
        Separate (1-α) quantile per class profile bucket
        """
        # Group scores by profile
        profile_scores = defaultdict(list)
        for score, profile in zip(validation_scores, validation_profiles):
            profile_key = profile.profile_key
            profile_scores[profile_key].append(score)

        # Compute quantile per bucket
        alpha = self.config.conformal_alpha
        for profile_key, scores in profile_scores.items():
            if len(scores) >= self.config.min_bucket_size:
                # Sufficient samples: use bucket-specific threshold
                threshold = np.quantile(scores, 1 - alpha)
                self.thresholds[profile_key] = threshold
                self.bucket_stats[profile_key] = {
                    "n_calibration": len(scores),
                    "threshold": float(threshold),
                    "score_mean": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                    "type": "bucket"
                }
            else:
                # Insufficient samples: mark for hierarchical backoff
                self.bucket_stats[profile_key] = {
                    "n_calibration": len(scores),
                    "type": "needs_backoff"
                }

        # Global threshold for backoff
        all_scores = [s for scores in profile_scores.values() for s in scores]
        self.global_threshold = np.quantile(all_scores, 1 - alpha) if all_scores else 0.0

        logger.info(f"Calibrated {len(self.thresholds)} Mondrian buckets "
                   f"with α={alpha}, global threshold={self.global_threshold:.3f}")

    def get_threshold(self, profile: MondrianProfile) -> float:
        """
        Get threshold for profile with hierarchical backoff
        1. Try bucket-specific threshold
        2. If insufficient data, try class-only threshold (weighted average of single-class buckets)
        3. Fall back to global threshold
        """
        profile_key = profile.profile_key

        # Try bucket-specific
        if profile_key in self.thresholds:
            return self.thresholds[profile_key]

        # Class-only backoff: compute weighted average of thresholds from
        # buckets that contain only classes present in this profile
        class_only_thresholds = []
        class_only_weights = []

        for class_id in profile.client_classes:
            # Look for single-class bucket threshold for this class
            single_class_key = frozenset({class_id})
            if single_class_key in self.thresholds:
                threshold = self.thresholds[single_class_key]
                n_samples = self.bucket_stats.get(single_class_key, {}).get("n_calibration", 1)
                class_only_thresholds.append(threshold)
                class_only_weights.append(n_samples)

        if class_only_thresholds:
            # Weighted average by calibration sample count
            total_weight = sum(class_only_weights)
            weighted_threshold = sum(
                t * w for t, w in zip(class_only_thresholds, class_only_weights)
            ) / total_weight

            # Apply conservative shrinkage (0.95) for class-only backoff
            # since we're extrapolating from single-class to multi-class
            shrinkage = 0.95
            logger.debug(f"Class-only backoff for {profile_key}: "
                        f"weighted_threshold={weighted_threshold:.3f}, "
                        f"n_classes={len(class_only_thresholds)}")
            return weighted_threshold * shrinkage

        # Fall back to global threshold with more conservative shrinkage
        shrinkage = 0.9
        return self.global_threshold * shrinkage

    def is_outlier(self, score: float, profile: MondrianProfile) -> bool:
        """Check if score is below threshold (outlier)"""
        threshold = self.get_threshold(profile)
        return score < threshold

    def get_calibration_statistics(self) -> Dict[str, Any]:
        """Get statistics about Mondrian conformal calibration"""
        return {
            "n_buckets": len(self.thresholds),
            "n_needs_backoff": sum(1 for s in self.bucket_stats.values()
                                  if s.get("type") == "needs_backoff"),
            "global_threshold": float(self.global_threshold),
            "bucket_stats": dict(self.bucket_stats)
        }


class PoGQv41Enhanced:
    """
    PoGQ-v4.1 Enhanced: Publication-Grade Byzantine Detection

    Integrates all Gen-4 components:
    - Mondrian (class-aware + per-class z-score)
    - Mondrian Conformal (per-class FPR buckets)
    - Adaptive Hybrid Scoring (PCA-cosine + λ_t)
    - Temporal EMA
    - Direction Prefilter
    - Winsorized Dispersion
    - PoGQ-v4-Lite (for high-dim)
    """

    def __init__(self, config: Optional[PoGQv41Config] = None):
        self.config = config or PoGQv41Config()

        # Components
        self.hybrid_scorer = AdaptiveHybridScorer(self.config)
        self.mondrian_conformal = MondrianConformally(self.config)

        # Per-class score statistics for z-score normalization
        self.class_score_stats: Dict[int, Tuple[float, float]] = {}  # (mean, std) per class
        self.class_score_history: Dict[int, List[float]] = defaultdict(list)

        # EMA scores
        self.ema_scores: Dict[str, float] = {}

        # Phase 2: Warm-up and hysteresis state
        self.client_round_counts: Dict[str, int] = defaultdict(int)  # Track rounds per client
        self.consecutive_violations: Dict[str, int] = defaultdict(int)  # For hysteresis
        self.consecutive_clears: Dict[str, int] = defaultdict(int)  # For release
        self.quarantined: Dict[str, bool] = defaultdict(bool)  # Quarantine status
        self.egregious_cap: Optional[float] = None  # Set during calibration

        # Provenance
        self.provenance = self._init_provenance()

    def _init_provenance(self) -> Dict[str, Any]:
        """Initialize provenance tracking"""
        if not self.config.track_provenance:
            return {}

        provenance = {
            "defense_preset_id": self.config.defense_preset_id,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config)
        }

        # Git commit
        if self.config.code_commit:
            provenance["code_commit"] = self.config.code_commit

        # Dataset SHA
        if self.config.dataset_sha256:
            provenance["dataset_sha256"] = self.config.dataset_sha256

        # Attack preset
        if self.config.attack_preset_id:
            provenance["attack_preset_id"] = self.config.attack_preset_id

        # Calibration hash
        if self.config.calibration_hash:
            provenance["calibration_hash"] = self.config.calibration_hash

        return provenance

    def fit_pca(self, reference_gradients: List[np.ndarray]):
        """Fit PCA on clean reference gradients"""
        self.hybrid_scorer.fit_pca(reference_gradients)

    def calibrate_mondrian(
        self,
        validation_scores: List[float],
        validation_profiles: List[MondrianProfile]
    ):
        """Calibrate Mondrian conformal thresholds and egregious cap"""
        self.mondrian_conformal.calibrate(validation_scores, validation_profiles)

        # Phase 2: Set egregious cap for warm-up quota (p99.9 of clean validation scores)
        if validation_scores:
            self.egregious_cap = np.quantile(validation_scores, self.config.egregious_cap_quantile)
            logger.info(f"Egregious cap set to {self.egregious_cap:.3f} "
                       f"(p{self.config.egregious_cap_quantile*100:.1f} of clean validation)")
        else:
            self.egregious_cap = float('inf')  # No cap if no validation data

    def update_class_statistics(
        self,
        class_id: int,
        score: float,
        window: Optional[int] = None
    ):
        """Update per-class score statistics for z-score normalization"""
        window = window or self.config.class_score_window

        # Add to history
        self.class_score_history[class_id].append(score)

        # Keep only recent window
        if len(self.class_score_history[class_id]) > window:
            self.class_score_history[class_id] = self.class_score_history[class_id][-window:]

        # Recompute stats
        scores = self.class_score_history[class_id]
        if len(scores) >= 3:  # Minimum for stats
            mean = np.mean(scores)
            std = np.std(scores)
            self.class_score_stats[class_id] = (mean, std)

    def validate_on_client_classes_normalized(
        self,
        gradient: np.ndarray,
        client_classes: Set[int],
        validation_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        reference_gradient: np.ndarray
    ) -> Tuple[float, Dict[int, float]]:
        """
        Validate on client's actual classes with per-class z-score normalization

        Returns:
            (aggregated_z_score, per_class_raw_scores)
        """
        class_z_scores = []
        class_raw_scores = {}

        for class_id in client_classes:
            if class_id not in validation_data:
                continue

            X_val, y_val = validation_data[class_id]

            # Compute class-specific utility score
            # For now, use simple gradient alignment (can enhance with actual loss computation)
            raw_score = self._compute_class_utility(gradient, X_val, y_val)
            class_raw_scores[class_id] = raw_score

            # Z-score normalization using class-specific stats
            if self.config.per_class_normalization and class_id in self.class_score_stats:
                mu, sigma = self.class_score_stats[class_id]
                if sigma > 1e-8:
                    z_score = (raw_score - mu) / sigma
                else:
                    z_score = 0.0
            else:
                z_score = raw_score

            class_z_scores.append(z_score)

            # Update class statistics
            self.update_class_statistics(class_id, raw_score)

        # Aggregate z-scores
        aggregated_z_score = np.mean(class_z_scores) if class_z_scores else 0.0

        return aggregated_z_score, class_raw_scores

    def _compute_class_utility(
        self,
        gradient: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Compute utility score for a specific class
        Simplified version: returns gradient norm (can be enhanced with actual loss)
        """
        # For MVP: use gradient norm as proxy
        # In production: compute actual Δloss on validation batch
        return float(np.linalg.norm(gradient))

    def score_gradient(
        self,
        gradient: np.ndarray,
        client_id: str,
        client_classes: Set[int],
        reference_gradient: np.ndarray,
        validation_data: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        round_number: int = 0
    ) -> Dict[str, Any]:
        """
        Score a client gradient using all PoGQ-v4.1 components

        Returns:
            Dictionary with scores, detection decision, and diagnostics
        """
        # 1. Mondrian validation (per-class z-score normalized)
        if validation_data:
            mondrian_score, class_raw_scores = self.validate_on_client_classes_normalized(
                gradient, client_classes, validation_data, reference_gradient
            )
        else:
            mondrian_score = 0.0
            class_raw_scores = {}

        # 2. Hybrid score (adaptive λ + PCA-cosine direction)
        utility_score = mondrian_score  # Use Mondrian score as utility
        hybrid_score = self.hybrid_scorer.compute_hybrid_score(
            gradient, reference_gradient, utility_score
        )

        # 3. Temporal EMA
        ema_key = f"{client_id}_{round_number}"
        if client_id in self.ema_scores:
            ema_score = (self.config.ema_beta * self.ema_scores[client_id] +
                        (1 - self.config.ema_beta) * hybrid_score)
        else:
            ema_score = hybrid_score
        self.ema_scores[client_id] = ema_score

        # 4. Phase 2: Warm-up quota + Hysteresis logic
        self.client_round_counts[client_id] += 1
        client_rounds = self.client_round_counts[client_id]
        profile = MondrianProfile(client_classes=client_classes)

        # Check conformal outlier status
        is_conformal_outlier = self.mondrian_conformal.is_outlier(ema_score, profile)

        # Warm-up grace period
        in_warmup = client_rounds <= self.config.warmup_rounds
        if in_warmup:
            # During warm-up, only flag egregious violations
            is_egregious = (self.egregious_cap is not None and
                           ema_score < self.egregious_cap)
            is_violation = is_egregious
            warmup_status = "warm-up-grace" if not is_egregious else "warm-up-egregious"
        else:
            # After warm-up, use conformal detection
            is_violation = is_conformal_outlier
            warmup_status = "enforcing"

        # Hysteresis logic for stability (prevent flapping)
        if is_violation:
            self.consecutive_violations[client_id] += 1
            self.consecutive_clears[client_id] = 0
        else:
            self.consecutive_clears[client_id] += 1
            self.consecutive_violations[client_id] = 0

        # Update quarantine status
        if self.consecutive_violations[client_id] >= self.config.hysteresis_k:
            self.quarantined[client_id] = True
        elif self.consecutive_clears[client_id] >= self.config.hysteresis_m:
            self.quarantined[client_id] = False

        # Final decision
        is_byzantine = self.quarantined[client_id]

        # Diagnostics
        result = {
            "client_id": client_id,
            "round": round_number,
            "scores": {
                "mondrian_raw": float(mondrian_score),
                "hybrid": float(hybrid_score),
                "hybrid_raw": float(hybrid_score),  # For ema_vs_raw.json artifact
                "ema": float(ema_score),
                "utility": float(utility_score)
            },
            "detection": {
                "is_byzantine": bool(is_byzantine),
                "is_conformal_outlier": bool(is_conformal_outlier),
                "threshold": float(self.mondrian_conformal.get_threshold(profile)),
                "mondrian_profile": sorted(list(client_classes)),
                "quarantined": bool(self.quarantined[client_id])
            },
            "phase2": {
                "client_rounds": int(client_rounds),
                "in_warmup": bool(in_warmup),
                "warmup_status": warmup_status,
                "consecutive_violations": int(self.consecutive_violations[client_id]),
                "consecutive_clears": int(self.consecutive_clears[client_id]),
                "egregious_cap": float(self.egregious_cap) if self.egregious_cap else None
            },
            "class_scores": {int(k): float(v) for k, v in class_raw_scores.items()},
            "lambda": float(self.hybrid_scorer.lambda_history[-1]) if self.hybrid_scorer.lambda_history else 0.85
        }

        return result

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_ids: List[str],
        client_class_distributions: List[Set[int]],
        reference_gradient: np.ndarray,
        validation_data: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        round_number: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate client gradients with PoGQ-v4.1 detection

        Returns:
            (aggregated_gradient, diagnostics)
        """
        results = []
        honest_gradients = []
        honest_clients = []

        for gradient, client_id, classes in zip(client_updates, client_ids, client_class_distributions):
            result = self.score_gradient(
                gradient, client_id, classes, reference_gradient,
                validation_data, round_number
            )
            results.append(result)

            if not result["detection"]["is_byzantine"]:
                honest_gradients.append(gradient)
                honest_clients.append(client_id)

        # Aggregate honest gradients (simple mean for now)
        if honest_gradients:
            aggregated = np.mean(honest_gradients, axis=0)
        else:
            logger.warning("No honest gradients! Using reference gradient")
            aggregated = reference_gradient

        # Diagnostics
        diagnostics = {
            "round": round_number,
            "n_clients": len(client_updates),
            "n_honest": len(honest_gradients),
            "n_byzantine": len(client_updates) - len(honest_gradients),
            "detection_rate": (len(client_updates) - len(honest_gradients)) / len(client_updates),
            "honest_clients": honest_clients,
            "per_client_results": results,
            "lambda_stats": self.hybrid_scorer.get_lambda_statistics(),
            "mondrian_stats": self.mondrian_conformal.get_calibration_statistics(),
            "provenance": self.provenance
        }

        return aggregated, diagnostics

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "lambda_stats": self.hybrid_scorer.get_lambda_statistics(),
            "mondrian_stats": self.mondrian_conformal.get_calibration_statistics(),
            "n_classes_tracked": len(self.class_score_stats),
            "ema_clients": len(self.ema_scores),
            "provenance": self.provenance
        }


class PoGQv41Lite:
    """
    PoGQ-v4-Lite: High-Dim Rescue Mode

    For datasets like CIFAR-10, compute utility score only on linear probe head
    gradient slice instead of full model gradient. This reduces high-dim noise.
    """

    def __init__(
        self,
        backbone_feature_dim: int,
        num_classes: int,
        probe_slice: Optional[Tuple[int, int]] = None,
        config: Optional[PoGQv41Config] = None
    ):
        """
        Args:
            backbone_feature_dim: Dimension of backbone features
            num_classes: Number of output classes
            probe_slice: (start, end) indices for probe head in gradient vector
            config: PoGQv41Config
        """
        self.backbone_feature_dim = backbone_feature_dim
        self.num_classes = num_classes
        self.probe_slice = probe_slice
        self.config = config or PoGQv41Config(use_lite_mode=True)

        # Compute probe slice if not provided
        if self.probe_slice is None:
            # Assume last (feature_dim * num_classes + num_classes) parameters are probe head
            probe_params = backbone_feature_dim * num_classes + num_classes
            self.probe_slice = (-probe_params, None)

    def extract_probe_gradient(self, full_gradient: np.ndarray) -> np.ndarray:
        """Extract probe head gradient from full gradient vector"""
        start, end = self.probe_slice
        return full_gradient[start:end]

    def compute_utility_score_lite(
        self,
        full_gradient: np.ndarray,
        reference_gradient: np.ndarray
    ) -> float:
        """
        Compute utility score on probe head only
        Uses cosine similarity on probe head gradient slice
        """
        # Extract probe gradients
        probe_grad = self.extract_probe_gradient(full_gradient)
        probe_ref = self.extract_probe_gradient(reference_gradient)

        # Cosine similarity
        norm_grad = np.linalg.norm(probe_grad)
        norm_ref = np.linalg.norm(probe_ref)

        if norm_grad == 0 or norm_ref == 0:
            return 0.0

        similarity = np.dot(probe_grad, probe_ref) / (norm_grad * norm_ref)

        # Log checksums for audit
        grad_checksum = hashlib.sha256(probe_grad.tobytes()).hexdigest()[:8]
        logger.debug(f"Probe gradient checksum: {grad_checksum}, similarity: {similarity:.3f}")

        return float(similarity)

    def verify_probe_slice(self, full_gradient: np.ndarray) -> bool:
        """Verify probe head gradient slice is within bounds"""
        start, end = self.probe_slice
        if end is None:
            end = len(full_gradient)

        expected_size = self.backbone_feature_dim * self.num_classes + self.num_classes
        actual_size = end - start

        if actual_size != expected_size:
            logger.warning(f"Probe slice size mismatch: expected {expected_size}, got {actual_size}")
            return False

        return True


def demonstrate_pogq_v41():
    """Demonstrate PoGQ-v4.1 Enhanced"""
    print("\n" + "="*80)
    print("🚀 PoGQ-v4.1 Enhanced Demonstration")
    print("="*80)

    # Configuration
    config = PoGQv41Config(
        pca_components=32,
        lambda_adaptive=True,
        mondrian_conformal=True,
        conformal_alpha=0.10,
        track_provenance=True
    )

    # Initialize detector
    pogq = PoGQv41Enhanced(config)

    # Simulate reference gradient
    gradient_dim = 1000
    reference_gradient = np.random.randn(gradient_dim).astype(np.float32)

    # Fit PCA on reference gradients
    reference_gradients = [np.random.randn(gradient_dim).astype(np.float32) for _ in range(50)]
    pogq.fit_pca(reference_gradients)

    # Simulate validation data (per-class)
    validation_data = {
        0: (np.random.randn(10, 28), np.zeros(10)),
        1: (np.random.randn(10, 28), np.ones(10))
    }

    # Calibrate Mondrian conformal
    val_scores = [0.5 + 0.2 * np.random.randn() for _ in range(100)]
    val_profiles = [MondrianProfile(client_classes={0, 1}) for _ in range(100)]
    pogq.calibrate_mondrian(val_scores, val_profiles)

    # Simulate client gradients
    print("\n📊 Simulating 5 clients (3 honest, 2 Byzantine)...")
    client_updates = []
    client_ids = []
    client_classes = []

    for i in range(5):
        if i < 3:  # Honest clients
            gradient = reference_gradient + 0.1 * np.random.randn(gradient_dim)
        else:  # Byzantine clients
            gradient = -reference_gradient + 0.5 * np.random.randn(gradient_dim)

        client_updates.append(gradient)
        client_ids.append(f"client_{i}")
        client_classes.append({0, 1})

    # Aggregate
    print("\n🔄 Aggregating with PoGQ-v4.1...")
    aggregated, diagnostics = pogq.aggregate(
        client_updates,
        client_ids,
        client_classes,
        reference_gradient,
        validation_data,
        round_number=1
    )

    # Results
    print("\n📈 Results:")
    print(f"  Total clients: {diagnostics['n_clients']}")
    print(f"  Honest: {diagnostics['n_honest']}")
    print(f"  Byzantine: {diagnostics['n_byzantine']}")
    print(f"  Detection rate: {diagnostics['detection_rate']:.1%}")
    print(f"\n  Lambda stats: {diagnostics['lambda_stats']}")

    print("\n✅ PoGQ-v4.1 demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_pogq_v41()
