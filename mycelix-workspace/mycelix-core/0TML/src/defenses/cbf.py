#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
CBF: Conformal Behavioral Filter for Byzantine Detection
=========================================================

A representation-based pre-aggregation filter with conformal guarantees.

Core Approach:
- Learn gradient representations via PCA on clean calibration set
- Detect anomalies via cosine similarity to behavioral history
- Conformal prediction thresholding (FPR ≤ α guarantee)
- Complete provenance tracking for reproducibility

Design Principles:
- Frozen extractor (trained only on clean server set)
- Strict train/test separation (calibration != validation != evaluation)
- Per-client behavioral history with sliding window

Note: This is NOT the FedGuard method from "FedGuard: Byzantine-Robust
Federated Learning via Trust-based Aggregation" which uses membership-inference.
CBF is an internal baseline for comparison.

Author: Luminous Dynamics
Date: November 8, 2025
Version: 1.0.0
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CBFConfig:
    """Configuration for Conformal Behavioral Filter"""
    # Representation extractor settings
    representation_dim: int = 64
    freeze_extractor: bool = True  # Always frozen after calibration
    use_pca_extractor: bool = True  # IncrementalPCA (default) vs random projection

    # Behavioral filter settings
    history_window: int = 5
    anomaly_threshold: float = 0.8  # Fallback threshold if conformal not calibrated

    # Conformal thresholding
    use_conformal_threshold: bool = True
    conformal_alpha: float = 0.10  # Target FPR (10%)
    min_conformal_samples: int = 20  # Minimum samples for conformal calibration

    # Normalization
    normalize_gradients: bool = True
    normalizer_type: str = "l2"  # "l2" or "standard"

    # Provenance
    track_provenance: bool = True


class SimpleRepresentationExtractor:
    """
    Simple representation extractor for gradient embeddings
    Uses random projection for efficiency (deprecated in favor of PCA)
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_matrix = None
        self.fitted = False

    def fit(self, gradients: List[np.ndarray], labels: Optional[np.ndarray] = None):
        """
        Fit extractor on clean gradients
        For simplicity, uses random projection (can enhance with PCA or neural net)
        """
        if len(gradients) == 0:
            logger.warning("No gradients for extractor fitting")
            return

        # Initialize random projection matrix
        np.random.seed(42)  # Deterministic for provenance
        self.projection_matrix = np.random.randn(self.input_dim, self.output_dim).astype(np.float32)

        # Normalize columns
        for j in range(self.output_dim):
            self.projection_matrix[:, j] /= np.linalg.norm(self.projection_matrix[:, j])

        self.fitted = True
        logger.info(f"Fitted representation extractor (random): {self.input_dim} → {self.output_dim}")

    def transform(self, gradient: np.ndarray) -> np.ndarray:
        """Extract representation from gradient"""
        if not self.fitted:
            raise RuntimeError("Extractor not fitted")

        # Project gradient
        representation = self.projection_matrix.T @ gradient

        return representation

    def get_hash(self) -> str:
        """Get hash of projection matrix for provenance"""
        if self.projection_matrix is None:
            return "not_fitted"

        return hashlib.sha256(self.projection_matrix.tobytes()).hexdigest()[:16]


class PCARepresentationExtractor:
    """
    Gen-4 PCA-based representation extractor
    Uses IncrementalPCA for numerical stability and memory efficiency
    """

    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.pca = None
        self.fitted = False

    def fit(self, gradients: List[np.ndarray], labels: Optional[np.ndarray] = None):
        """
        Fit PCA on clean gradients
        Uses IncrementalPCA for stability
        """
        if len(gradients) == 0:
            logger.warning("No gradients for PCA fitting")
            return

        try:
            from sklearn.decomposition import IncrementalPCA
        except ImportError:
            logger.error("sklearn required for PCA extractor")
            raise

        # Stack gradients
        X = np.vstack([g.reshape(1, -1) for g in gradients])

        # Fit IncrementalPCA
        self.pca = IncrementalPCA(n_components=self.output_dim)
        self.pca.fit(X)

        self.fitted = True
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"Fitted PCA extractor: {X.shape[1]} → {self.output_dim} "
                   f"(explained variance: {explained_var:.2%})")

    def transform(self, gradient: np.ndarray) -> np.ndarray:
        """Extract PCA representation from gradient"""
        if not self.fitted:
            raise RuntimeError("PCA extractor not fitted")

        # Transform
        representation = self.pca.transform(gradient.reshape(1, -1))[0]

        return representation.astype(np.float32)

    def get_hash(self) -> str:
        """Get hash of PCA components for provenance"""
        if self.pca is None:
            return "not_fitted"

        # Hash the principal components
        components_bytes = self.pca.components_.tobytes()
        return hashlib.sha256(components_bytes).hexdigest()[:16]


class CBF:
    """
    Conformal Behavioral Filter (CBF)

    Representation-based pre-aggregation filter for Byzantine detection.
    Uses PCA embeddings + conformal thresholding for FPR guarantees.

    Note: Not related to FedGuard (MIA-based method). This is an internal baseline.
    """

    def __init__(self, config: Optional[CBFConfig] = None):
        self.config = config or CBFConfig()

        self.extractor = None
        self.gradient_dim = None

        # Behavioral history
        self.client_representations: Dict[str, List[np.ndarray]] = {}

        # Normalization parameters
        self.normalizer_params: Optional[Dict[str, float]] = None

        # Gen-4: Conformal threshold
        self.conformal_threshold: Optional[float] = None

        # Provenance
        self.provenance = {
            "defense": "fedguard_strict",
            "extractor_type": "pca" if self.config.use_pca_extractor else "random",
            "extractor_hash": None,
            "training_set_hash": None,
            "normalizer_params": None,
            "conformal_threshold": None,
            "conformal_alpha": self.config.conformal_alpha if self.config.use_conformal_threshold else None,
            "calibration_timestamp": None
        }

    def calibrate_on_clean_server_set(
        self,
        clean_gradients: List[np.ndarray],
        client_ids: Optional[List[str]] = None
    ):
        """
        Calibrate FedGuard-strict on clean server-side gradients
        This is the ONLY time the extractor is trained

        Args:
            clean_gradients: List of known-clean gradients
            client_ids: Optional client IDs for provenance
        """
        if len(clean_gradients) == 0:
            raise ValueError("No clean gradients for calibration")

        # Get gradient dimension
        self.gradient_dim = clean_gradients[0].shape[0]

        # Initialize extractor (Gen-4: PCA or random projection)
        if self.config.use_pca_extractor:
            self.extractor = PCARepresentationExtractor(
                output_dim=self.config.representation_dim
            )
        else:
            self.extractor = SimpleRepresentationExtractor(
                input_dim=self.gradient_dim,
                output_dim=self.config.representation_dim
            )

        # Fit extractor on clean gradients
        self.extractor.fit(clean_gradients)

        # Compute normalizer parameters
        if self.config.normalize_gradients:
            self._fit_normalizer(clean_gradients)

        # Provenance tracking
        if self.config.track_provenance:
            # Hash training set
            training_set_bytes = b"".join([g.tobytes() for g in clean_gradients])
            training_set_hash = hashlib.sha256(training_set_bytes).hexdigest()[:16]

            self.provenance.update({
                "extractor_hash": self.extractor.get_hash(),
                "training_set_hash": training_set_hash,
                "normalizer_params": self.normalizer_params,
                "calibration_timestamp": np.datetime64('now').astype(str),
                "n_calibration_samples": len(clean_gradients)
            })

        logger.info(f"FedGuard-strict calibrated on {len(clean_gradients)} clean gradients")
        logger.info(f"Extractor hash: {self.provenance['extractor_hash']}")
        logger.info(f"Training set hash: {self.provenance['training_set_hash']}")

    def calibrate_conformal_threshold(
        self,
        clean_validation_gradients: List[np.ndarray],
        client_ids: Optional[List[str]] = None
    ):
        """
        Gen-4: Calibrate conformal prediction threshold on separate validation set

        Sets threshold as the (1-α) quantile of anomaly scores on clean validation data.
        This guarantees FPR ≤ α on similar distributions (conformal prediction).

        Args:
            clean_validation_gradients: Separate clean set (NOT used for extractor training)
            client_ids: Optional client IDs for validation samples
        """
        if self.extractor is None:
            raise RuntimeError("Must calibrate extractor first (call calibrate_on_clean_server_set)")

        if len(clean_validation_gradients) < self.config.min_conformal_samples:
            logger.warning(
                f"Insufficient validation samples for conformal calibration: "
                f"{len(clean_validation_gradients)} < {self.config.min_conformal_samples}. "
                f"Using fixed threshold instead."
            )
            return

        # Generate client IDs if not provided
        if client_ids is None:
            client_ids = [f"val_client_{i}" for i in range(len(clean_validation_gradients))]

        # Score all validation gradients
        validation_scores = []
        for gradient, client_id in zip(clean_validation_gradients, client_ids):
            representation = self._extract_representation(gradient)
            score = self._compute_anomaly_score(client_id, representation)
            validation_scores.append(score)
            # Update history (so we have baselines)
            self._update_history(client_id, representation)

        # Compute (1-α) quantile as conformal threshold
        alpha = self.config.conformal_alpha
        threshold = np.quantile(validation_scores, alpha)  # Lower α-quantile for anomaly scores

        self.conformal_threshold = float(threshold)

        # Update provenance
        self.provenance["conformal_threshold"] = float(threshold)
        self.provenance["n_conformal_samples"] = len(validation_scores)

        logger.info(f"Conformal threshold calibrated: {threshold:.3f} (α={alpha:.2f}, n={len(validation_scores)})")
        logger.info(f"  Validation score range: [{min(validation_scores):.3f}, {max(validation_scores):.3f}]")

    def _fit_normalizer(self, gradients: List[np.ndarray]):
        """Fit gradient normalizer on clean data"""
        if self.config.normalizer_type == "l2":
            # L2 normalization: no parameters needed
            self.normalizer_params = {"type": "l2"}

        elif self.config.normalizer_type == "standard":
            # Standard normalization: compute mean and std per coordinate
            X = np.vstack(gradients)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

            self.normalizer_params = {
                "type": "standard",
                "mean_hash": hashlib.sha256(mean.tobytes()).hexdigest()[:8],
                "std_hash": hashlib.sha256(std.tobytes()).hexdigest()[:8]
            }

            # Store actual values (not in provenance, too large)
            self._normalizer_mean = mean
            self._normalizer_std = std

        else:
            raise ValueError(f"Unknown normalizer type: {self.config.normalizer_type}")

    def _normalize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Normalize gradient using fitted normalizer"""
        if not self.config.normalize_gradients:
            return gradient

        if self.normalizer_params["type"] == "l2":
            norm = np.linalg.norm(gradient)
            if norm < 1e-8:
                return gradient
            return gradient / norm

        elif self.normalizer_params["type"] == "standard":
            return (gradient - self._normalizer_mean) / (self._normalizer_std + 1e-8)

        return gradient

    def _extract_representation(self, gradient: np.ndarray) -> np.ndarray:
        """Extract representation from gradient"""
        if self.extractor is None:
            raise RuntimeError("FedGuard-strict not calibrated. Call calibrate_on_clean_server_set() first")

        # Normalize
        normalized = self._normalize_gradient(gradient)

        # Extract representation
        representation = self.extractor.transform(normalized)

        return representation

    def _compute_anomaly_score(
        self,
        client_id: str,
        representation: np.ndarray
    ) -> float:
        """
        Compute anomaly score based on historical representations
        Uses average cosine similarity to history
        """
        if client_id not in self.client_representations:
            # First gradient from this client: not anomalous
            return 1.0

        history = self.client_representations[client_id]
        if len(history) == 0:
            return 1.0

        # Compute cosine similarities to historical representations
        similarities = []
        for hist_repr in history:
            sim = np.dot(representation, hist_repr) / (
                np.linalg.norm(representation) * np.linalg.norm(hist_repr) + 1e-8
            )
            similarities.append(sim)

        # Average similarity
        avg_similarity = np.mean(similarities)

        return float(avg_similarity)

    def _update_history(self, client_id: str, representation: np.ndarray):
        """Update client representation history"""
        if client_id not in self.client_representations:
            self.client_representations[client_id] = []

        self.client_representations[client_id].append(representation)

        # Keep only recent window
        if len(self.client_representations[client_id]) > self.config.history_window:
            self.client_representations[client_id] = \
                self.client_representations[client_id][-self.config.history_window:]

    def score_gradient(
        self,
        gradient: np.ndarray,
        client_id: str
    ) -> Dict[str, Any]:
        """
        Score a gradient for Byzantine behavior (Gen-4 enhanced)

        Returns:
            Dict with score, is_byzantine flag, and diagnostics
        """
        # Extract representation
        representation = self._extract_representation(gradient)

        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(client_id, representation)

        # Gen-4: Use conformal threshold if available, otherwise fixed threshold
        if self.config.use_conformal_threshold and self.conformal_threshold is not None:
            threshold = self.conformal_threshold
            threshold_type = "conformal"
        else:
            threshold = self.config.anomaly_threshold
            threshold_type = "fixed"

        # Detection decision
        is_byzantine = anomaly_score < threshold

        # Update history
        self._update_history(client_id, representation)

        return {
            "client_id": client_id,
            "anomaly_score": float(anomaly_score),
            "is_byzantine": bool(is_byzantine),
            "threshold": float(threshold),
            "threshold_type": threshold_type,
            "representation_norm": float(np.linalg.norm(representation))
        }

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_ids: Optional[List[str]] = None,
        **context
    ) -> np.ndarray:
        """
        Aggregate client gradients with FedGuard-strict filtering

        Args:
            client_updates: List of client gradients
            client_ids: List of client identifiers
            **context: Additional context (ignored)

        Returns:
            Aggregated gradient
        """
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_updates))]

        # Score all gradients
        honest_gradients = []
        honest_clients = []
        scores = []

        for gradient, client_id in zip(client_updates, client_ids):
            result = self.score_gradient(gradient, client_id)
            scores.append(result)

            if not result["is_byzantine"]:
                honest_gradients.append(gradient)
                honest_clients.append(client_id)

        # Aggregate honest gradients (simple mean)
        if honest_gradients:
            aggregated = np.mean(honest_gradients, axis=0)
            logger.info(f"FedGuard-strict: {len(honest_gradients)}/{len(client_updates)} honest")
        else:
            logger.warning("FedGuard-strict: No honest gradients! Using mean of all")
            aggregated = np.mean(client_updates, axis=0)

        return aggregated

    def explain(self) -> Dict[str, Any]:
        """
        Return per-client diagnostics and provenance

        Returns:
            Dictionary with scores, flags, and provenance
        """
        return {
            "provenance": self.provenance,
            "config": {
                "representation_dim": self.config.representation_dim,
                "history_window": self.config.history_window,
                "anomaly_threshold": self.config.anomaly_threshold,
                "normalizer_type": self.config.normalizer_type
            },
            "n_clients_tracked": len(self.client_representations)
        }


def demonstrate_fedguard_strict():
    """Demonstrate FedGuard-strict"""
    print("\n" + "="*80)
    print("🛡️  FedGuard-strict Demonstration")
    print("="*80)

    # Initialize
    config = FedGuardStrictConfig(
        representation_dim=32,
        anomaly_threshold=0.7,
        track_provenance=True
    )
    fedguard = FedGuardStrict(config)

    # Simulate clean calibration set
    print("\n📊 Calibrating on 50 clean gradients...")
    gradient_dim = 1000
    clean_gradients = [np.random.randn(gradient_dim).astype(np.float32) for _ in range(50)]

    fedguard.calibrate_on_clean_server_set(clean_gradients)

    # Simulate client gradients (3 honest, 2 Byzantine)
    print("\n🔄 Testing on 5 clients (3 honest, 2 Byzantine)...")
    client_updates = []
    client_ids = []

    for i in range(5):
        client_id = f"client_{i}"
        client_ids.append(client_id)

        if i < 3:  # Honest
            gradient = clean_gradients[0] + 0.1 * np.random.randn(gradient_dim)
        else:  # Byzantine
            gradient = -clean_gradients[0] + 0.5 * np.random.randn(gradient_dim)

        client_updates.append(gradient)

    # Aggregate
    aggregated = fedguard.aggregate(client_updates, client_ids)

    # Show results
    print("\n📈 Results:")
    explanation = fedguard.explain()
    print(f"  Extractor hash: {explanation['provenance']['extractor_hash']}")
    print(f"  Training set hash: {explanation['provenance']['training_set_hash']}")
    print(f"  Clients tracked: {explanation['n_clients_tracked']}")

    print("\n✅ FedGuard-strict demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_fedguard_strict()
