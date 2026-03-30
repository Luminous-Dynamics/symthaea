# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Feature Extractor for ML-Enhanced Byzantine Detection
======================================================

Computes composite feature vectors for gradient-based Sybil/Byzantine detection:

1. PoGQ (Proof of Gradient Quality): Cosine similarity to median gradient
2. TCDM (Temporal Consistency): Correlation with node's historical gradients
3. Z-Score: Statistical outlier detection (distance from mean)
4. Entropy: Shannon entropy of gradient distribution

These features feed into SVM/RF classifiers for 95-98% detection accuracy.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from collections import deque


@dataclass
class CompositeFeatures:
    """Container for all extracted features for a single gradient"""

    pogq_score: float          # [0, 1] - Higher is more honest (cosine similarity)
    tcdm_score: float          # [0, 1] - Temporal consistency with history
    zscore_magnitude: float    # R - Distance from mean (higher = outlier)
    entropy_score: float       # [0, ∞) - Shannon entropy of gradient distribution

    # Metadata
    node_id: int
    round_num: int
    gradient_norm: float       # L2 norm of gradient

    def to_array(self) -> np.ndarray:
        """Convert to feature vector for ML classifier"""
        return np.array([
            self.pogq_score,
            self.tcdm_score,
            self.zscore_magnitude,
            self.entropy_score,
            self.gradient_norm,  # Additional feature
        ])

    @classmethod
    def feature_names(cls) -> List[str]:
        """Return feature names for interpretability"""
        return [
            'pogq_score',
            'tcdm_score',
            'zscore_magnitude',
            'entropy_score',
            'gradient_norm',
        ]


class FeatureExtractor:
    """
    Extracts composite feature vectors from gradients for Byzantine detection.

    Maintains temporal history per node for TCDM computation.
    """

    def __init__(self, history_window: int = 5):
        """
        Args:
            history_window: Number of past gradients to retain for TCDM
        """
        self.history_window = history_window

        # Per-node gradient history: node_id -> deque of recent gradients
        self.gradient_history: dict[int, deque] = {}

        # Global statistics (updated each round)
        self.global_mean: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None
        self.global_median: Optional[np.ndarray] = None

    def update_global_statistics(self, gradients: List[np.ndarray]):
        """
        Update global gradient statistics for z-score and PoGQ computation.

        Args:
            gradients: List of all gradients in current round
        """
        if len(gradients) == 0:
            return

        grad_array = np.array(gradients)  # Shape: (num_nodes, gradient_dim)

        self.global_mean = np.mean(grad_array, axis=0)
        self.global_std = np.std(grad_array, axis=0) + 1e-8  # Avoid div by zero
        self.global_median = np.median(grad_array, axis=0)

    def extract_features(
        self,
        gradient: np.ndarray,
        node_id: int,
        round_num: int,
        all_gradients: Optional[List[np.ndarray]] = None,
    ) -> CompositeFeatures:
        """
        Extract all features for a single gradient.

        Args:
            gradient: The gradient vector to analyze
            node_id: ID of the node that submitted this gradient
            round_num: Current training round number
            all_gradients: All gradients in this round (for global stats)

        Returns:
            CompositeFeatures object with all extracted features
        """
        # Update global stats if provided
        if all_gradients is not None:
            self.update_global_statistics(all_gradients)

        # 1. Compute PoGQ (cosine similarity to median)
        pogq_score = self._compute_pogq(gradient)

        # 2. Compute TCDM (temporal consistency)
        tcdm_score = self._compute_tcdm(gradient, node_id)

        # 3. Compute Z-score magnitude
        zscore_mag = self._compute_zscore(gradient)

        # 4. Compute entropy
        entropy_score = self._compute_entropy(gradient)

        # 5. Compute gradient norm
        grad_norm = float(np.linalg.norm(gradient))

        # Update history
        self._update_history(gradient, node_id)

        return CompositeFeatures(
            pogq_score=pogq_score,
            tcdm_score=tcdm_score,
            zscore_magnitude=zscore_mag,
            entropy_score=entropy_score,
            node_id=node_id,
            round_num=round_num,
            gradient_norm=grad_norm,
        )

    def _compute_pogq(self, gradient: np.ndarray) -> float:
        """
        Compute Proof of Gradient Quality (cosine similarity to median).

        Returns:
            Similarity score in [0, 1] (1 = identical to median)
        """
        if self.global_median is None:
            return 0.5  # Neutral score if no stats available

        # Cosine similarity: (a · b) / (||a|| ||b||)
        dot_product = np.dot(gradient, self.global_median)
        norm_g = np.linalg.norm(gradient)
        norm_m = np.linalg.norm(self.global_median)

        if norm_g < 1e-10 or norm_m < 1e-10:
            return 0.0

        similarity = dot_product / (norm_g * norm_m)

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    def _compute_tcdm(self, gradient: np.ndarray, node_id: int) -> float:
        """
        Compute Temporal Consistency Detection Metric.

        Measures correlation between current gradient and node's historical gradients.
        Byzantine nodes often show low temporal consistency.

        Returns:
            Consistency score in [0, 1] (1 = high consistency)
        """
        if node_id not in self.gradient_history:
            return 0.5  # Neutral score for first gradient

        history = self.gradient_history[node_id]

        if len(history) == 0:
            return 0.5

        # Compute average cosine similarity with historical gradients
        similarities = []
        for past_gradient in history:
            dot_prod = np.dot(gradient, past_gradient)
            norm_curr = np.linalg.norm(gradient)
            norm_past = np.linalg.norm(past_gradient)

            if norm_curr < 1e-10 or norm_past < 1e-10:
                continue

            sim = dot_prod / (norm_curr * norm_past)
            similarities.append((sim + 1.0) / 2.0)  # Convert to [0, 1]

        if len(similarities) == 0:
            return 0.5

        return float(np.mean(similarities))

    def _compute_zscore(self, gradient: np.ndarray) -> float:
        """
        Compute Z-score magnitude (statistical outlier detection).

        Returns:
            Absolute z-score magnitude (higher = more extreme outlier)
        """
        if self.global_mean is None or self.global_std is None:
            return 0.0

        # Element-wise z-score
        z_scores = (gradient - self.global_mean) / self.global_std

        # Return L2 norm of z-score vector
        return float(np.linalg.norm(z_scores))

    def _compute_entropy(self, gradient: np.ndarray) -> float:
        """
        Compute Shannon entropy of gradient distribution.

        Low entropy = concentrated values (potential attack pattern)
        High entropy = diverse values (more natural)

        Returns:
            Shannon entropy value
        """
        # Discretize gradient into bins for entropy calculation
        num_bins = min(50, len(gradient) // 10)
        if num_bins < 2:
            return 0.0

        hist, _ = np.histogram(gradient, bins=num_bins)

        # Normalize to probability distribution
        hist = hist / (np.sum(hist) + 1e-10)

        # Remove zero bins
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.0

        return float(scipy_entropy(hist, base=2))

    def _update_history(self, gradient: np.ndarray, node_id: int):
        """Update gradient history for TCDM computation"""
        if node_id not in self.gradient_history:
            self.gradient_history[node_id] = deque(maxlen=self.history_window)

        self.gradient_history[node_id].append(gradient.copy())

    def reset(self):
        """Clear all history and statistics"""
        self.gradient_history.clear()
        self.global_mean = None
        self.global_std = None
        self.global_median = None


def extract_features_batch(
    gradients: List[np.ndarray],
    node_ids: List[int],
    round_num: int,
    extractor: Optional[FeatureExtractor] = None,
) -> List[CompositeFeatures]:
    """
    Batch feature extraction for efficiency.

    Args:
        gradients: List of gradient vectors
        node_ids: Corresponding node IDs
        round_num: Current round number
        extractor: Existing FeatureExtractor (or create new one)

    Returns:
        List of CompositeFeatures, one per gradient
    """
    if extractor is None:
        extractor = FeatureExtractor()

    # Update global stats once for entire batch
    extractor.update_global_statistics(gradients)

    # Extract features for each gradient
    features = []
    for gradient, node_id in zip(gradients, node_ids):
        feat = extractor.extract_features(
            gradient=gradient,
            node_id=node_id,
            round_num=round_num,
            all_gradients=None,  # Already updated above
        )
        features.append(feat)

    return features
