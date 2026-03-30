# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gradient Dimensionality Analyzer

Analyzes gradient characteristics to inform Byzantine detection strategy.
Uses PCA to compute effective dimensionality and recommend appropriate
detection parameters based on gradient structure.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class GradientProfile:
    """Profile of gradient characteristics"""

    # Dimensionality metrics
    nominal_dimensionality: int  # Total number of gradient elements
    effective_dimensionality: float  # PCA-based effective dimension
    explained_variance_ratio: float  # Variance captured by top components

    # Distribution metrics
    mean_gradient_norm: float
    std_gradient_norm: float
    min_gradient_norm: float
    max_gradient_norm: float

    # Cosine similarity metrics (between gradients)
    mean_cosine_similarity: float
    std_cosine_similarity: float
    min_cosine_similarity: float
    max_cosine_similarity: float

    # Recommended parameters
    recommended_cos_min: float
    recommended_cos_max: float
    recommended_recovery_threshold: int
    recommended_recovery_bonus: float
    recommended_committee_floor: float
    recommended_reputation_floor: float

    # Detection strategy
    detection_strategy: str  # "low_dim", "mid_dim", or "high_dim"

    def __str__(self) -> str:
        return (
            f"GradientProfile(\n"
            f"  Strategy: {self.detection_strategy}\n"
            f"  Nominal Dim: {self.nominal_dimensionality}\n"
            f"  Effective Dim: {self.effective_dimensionality:.1f}\n"
            f"  Variance Explained: {self.explained_variance_ratio:.1%}\n"
            f"  Cosine Similarity: {self.mean_cosine_similarity:.3f} ± {self.std_cosine_similarity:.3f}\n"
            f"  Recommended thresholds: [{self.recommended_cos_min:.2f}, {self.recommended_cos_max:.2f}]\n"
            f")"
        )


class GradientDimensionalityAnalyzer:
    """
    Analyzes gradient characteristics to inform detection strategy.

    Uses PCA to compute effective dimensionality and recommends appropriate
    Byzantine detection parameters based on gradient structure.
    """

    def __init__(
        self,
        variance_threshold: float = 0.95,  # Variance to capture in PCA
        min_samples_for_pca: int = 5,  # Minimum gradients needed for PCA
    ):
        """
        Args:
            variance_threshold: Amount of variance to capture when computing
                effective dimensionality (default: 0.95 = 95%)
            min_samples_for_pca: Minimum number of gradient samples needed
                to perform PCA analysis
        """
        self.variance_threshold = variance_threshold
        self.min_samples_for_pca = min_samples_for_pca

    def analyze_gradients(
        self, gradients: List[torch.Tensor]
    ) -> GradientProfile:
        """
        Analyze gradient characteristics and recommend detection parameters.

        Args:
            gradients: List of gradient tensors from different nodes

        Returns:
            GradientProfile with dimensionality metrics and recommended parameters
        """
        if len(gradients) < 2:
            raise ValueError(
                f"Need at least 2 gradients for analysis, got {len(gradients)}"
            )

        # Flatten all gradients to vectors
        flat_grads = [g.flatten() for g in gradients]
        nominal_dim = flat_grads[0].numel()

        # Stack into matrix: [num_nodes, gradient_dim]
        grad_matrix = torch.stack(flat_grads).detach().cpu().numpy()

        # Compute effective dimensionality via PCA
        effective_dim, variance_ratio = self._compute_effective_dimensionality(
            grad_matrix
        )

        # Compute gradient norm statistics
        norms = [torch.norm(g).item() for g in flat_grads]
        norm_stats = {
            "mean": np.mean(norms),
            "std": np.std(norms),
            "min": np.min(norms),
            "max": np.max(norms),
        }

        # Compute cosine similarity statistics
        cosine_stats = self._compute_cosine_statistics(flat_grads)

        # WEEK 2: Detect label skew characteristics
        label_skew_detected = self.detect_label_skew(flat_grads)

        # Determine detection strategy and recommend parameters
        strategy, params = self._recommend_parameters(
            effective_dim, nominal_dim, cosine_stats, label_skew_detected
        )

        return GradientProfile(
            nominal_dimensionality=nominal_dim,
            effective_dimensionality=effective_dim,
            explained_variance_ratio=variance_ratio,
            mean_gradient_norm=norm_stats["mean"],
            std_gradient_norm=norm_stats["std"],
            min_gradient_norm=norm_stats["min"],
            max_gradient_norm=norm_stats["max"],
            mean_cosine_similarity=cosine_stats["mean"],
            std_cosine_similarity=cosine_stats["std"],
            min_cosine_similarity=cosine_stats["min"],
            max_cosine_similarity=cosine_stats["max"],
            recommended_cos_min=params["cos_min"],
            recommended_cos_max=params["cos_max"],
            recommended_recovery_threshold=params["recovery_threshold"],
            recommended_recovery_bonus=params["recovery_bonus"],
            recommended_committee_floor=params["committee_floor"],
            recommended_reputation_floor=params["reputation_floor"],
            detection_strategy=strategy,
        )

    def _compute_effective_dimensionality(
        self, grad_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute effective dimensionality using PCA.

        Effective dimensionality = number of components needed to capture
        variance_threshold (e.g., 95%) of the total variance.

        Args:
            grad_matrix: [num_samples, gradient_dim] numpy array

        Returns:
            (effective_dim, variance_ratio): effective dimensionality and
                fraction of variance explained
        """
        num_samples, gradient_dim = grad_matrix.shape

        if num_samples < self.min_samples_for_pca:
            # Not enough samples for meaningful PCA, use nominal dim
            return float(gradient_dim), 1.0

        # Center the data
        grad_centered = grad_matrix - grad_matrix.mean(axis=0, keepdims=True)

        # Compute SVD (more numerically stable than eigendecomposition)
        try:
            # Use randomized SVD for large matrices (faster)
            if gradient_dim > 1000:
                from sklearn.decomposition import TruncatedSVD

                n_components = min(
                    num_samples - 1, gradient_dim, 1000
                )  # Limit components
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd.fit(grad_centered)
                explained_variance_ratio = svd.explained_variance_ratio_
            else:
                # Full SVD for smaller matrices
                U, S, Vt = np.linalg.svd(grad_centered, full_matrices=False)
                explained_variance = (S**2) / (num_samples - 1)
                total_variance = explained_variance.sum()
                explained_variance_ratio = explained_variance / total_variance

            # Find number of components needed for variance_threshold
            cumsum_variance = np.cumsum(explained_variance_ratio)
            n_components = (
                np.searchsorted(cumsum_variance, self.variance_threshold) + 1
            )

            # Effective dimensionality is number of components
            effective_dim = float(n_components)
            variance_captured = cumsum_variance[n_components - 1]

            return effective_dim, variance_captured

        except Exception as e:
            # Fallback: use nominal dimensionality if PCA fails
            print(f"Warning: PCA failed ({e}), using nominal dimensionality")
            return float(gradient_dim), 1.0

    def _compute_cosine_statistics(
        self, gradients: List[torch.Tensor]
    ) -> dict:
        """
        Compute pairwise cosine similarity statistics between gradients.

        Args:
            gradients: List of flattened gradient tensors

        Returns:
            dict with mean, std, min, max cosine similarities
        """
        n = len(gradients)
        if n < 2:
            return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}

        # Compute all pairwise cosine similarities
        cosine_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = F.cosine_similarity(
                    gradients[i].unsqueeze(0),
                    gradients[j].unsqueeze(0),
                    dim=1,
                ).item()
                cosine_sims.append(cos_sim)

        return {
            "mean": np.mean(cosine_sims),
            "std": np.std(cosine_sims),
            "min": np.min(cosine_sims),
            "max": np.max(cosine_sims),
        }

    def _compute_pairwise_cosines(
        self, gradients: List[torch.Tensor]
    ) -> List[float]:
        """
        Compute all pairwise cosine similarities between gradients.

        Args:
            gradients: List of flattened gradient tensors

        Returns:
            List of pairwise cosine similarities
        """
        n = len(gradients)
        cosine_sims = []

        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = F.cosine_similarity(
                    gradients[i].unsqueeze(0),
                    gradients[j].unsqueeze(0),
                    dim=1,
                ).item()
                cosine_sims.append(cos_sim)

        return cosine_sims

    def _detect_bimodal_distribution(self, cosine_sims: List[float]) -> bool:
        """
        Detect if cosine similarities form two clusters (indicating skew).

        Uses simple heuristic: check if there are both high and low similarities.

        Args:
            cosine_sims: List of pairwise cosine similarities

        Returns:
            True if both (max - mean) > 0.2 AND (mean - min) > 0.2
        """
        if len(cosine_sims) < 2:
            return False

        cosine_arr = np.array(cosine_sims)
        mean_val = np.mean(cosine_arr)
        max_val = np.max(cosine_arr)
        min_val = np.min(cosine_arr)

        # Check if there's a large spread on both sides of the mean
        high_cluster = (max_val - mean_val) > 0.2
        low_cluster = (mean_val - min_val) > 0.2

        return high_cluster and low_cluster

    def detect_label_skew(self, gradients: List[torch.Tensor]) -> bool:
        """
        Detect if gradients exhibit label skew characteristics.

        Label skew is detected if ANY of the following conditions are met:
        1. High variance in cosine similarities (std > 0.3)
        2. Bimodal distribution of cosine similarities
        3. Low mean cosine similarity (mean < 0.5) + high std (std > 0.25)

        Args:
            gradients: List of gradient tensors

        Returns:
            True if label skew is detected
        """
        if len(gradients) < 5:
            # Need sufficient samples to detect label skew
            return False

        cosine_sims = self._compute_pairwise_cosines(gradients)
        cosine_stats = self._compute_cosine_statistics(gradients)

        # Condition 1: High variance in cosine similarities
        high_variance = cosine_stats["std"] > 0.3

        # Condition 2: Bimodal distribution
        bimodal = self._detect_bimodal_distribution(cosine_sims)

        # Condition 3: Low mean + moderate variance
        low_mean_high_var = (
            cosine_stats["mean"] < 0.5 and cosine_stats["std"] > 0.25
        )

        label_skew_detected = high_variance or bimodal or low_mean_high_var

        return label_skew_detected

    def _recommend_parameters(
        self,
        effective_dim: float,
        nominal_dim: int,
        cosine_stats: dict,
        label_skew_detected: bool = False,  # NEW: Week 2 enhancement
    ) -> Tuple[str, dict]:
        """
        Recommend detection parameters based on gradient characteristics and skew.

        FIXED: Use nominal_dim as primary signal since effective_dim is
        sample-count-limited (can't exceed n_samples - 1).

        Args:
            effective_dim: PCA-based effective dimensionality
            nominal_dim: Total number of gradient elements
            cosine_stats: Statistics on pairwise cosine similarities
            label_skew_detected: Whether label skew was detected (Week 2)

        Returns:
            (strategy_name, parameters_dict)
        """
        # Determine strategy based on NOMINAL dimensionality (primary signal)
        # effective_dim is unreliable with small sample sizes
        if nominal_dim < 100:
            strategy = "low_dim"
            # Low-dimensional: Use MAD-based detection, wider thresholds
            # For low-dim tabular data under label skew
            params = {
                "cos_min": -0.8,  # Very wide range
                "cos_max": 0.95,
                "recovery_threshold": 2,
                "recovery_bonus": 0.15,  # Higher bonus to recover FPs
                "committee_floor": 0.20,  # Lower floor (more lenient)
                "reputation_floor": 0.01,
            }
        elif nominal_dim < 1000:
            strategy = "mid_dim"
            # Mid-dimensional: Moderate thresholds
            # For EMNIST-like grayscale CNNs
            params = {
                "cos_min": -0.4,
                "cos_max": 0.96,
                "recovery_threshold": 3,
                "recovery_bonus": 0.10,
                "committee_floor": 0.30,
                "reputation_floor": 0.01,
            }
        else:
            strategy = "high_dim"
            # High-dimensional: Current optimal parameters
            # For CIFAR-10-like RGB CNNs
            params = {
                "cos_min": -0.5,
                "cos_max": 0.95,
                "recovery_threshold": 2,
                "recovery_bonus": 0.12,
                "committee_floor": 0.25,
                "reputation_floor": 0.01,
            }

        # WEEK 2: Apply label skew adjustments if detected
        if label_skew_detected:
            strategy += "_label_skew"

            # Widen cos_min range to accommodate legitimate diversity
            # Based on parameter sweep results: cos_min=-0.5 optimal for label skew
            if nominal_dim < 100:
                # Low-dim: widen more aggressively
                params["cos_min"] = max(-0.9, params["cos_min"] - 0.2)
            elif nominal_dim < 1000:
                # Mid-dim: widen moderately
                params["cos_min"] = max(-0.6, params["cos_min"] - 0.2)
            else:
                # High-dim: optimal at -0.5 for label skew (verified by parameter sweep)
                # Current default is already -0.5, so no change needed
                pass

            # Increase recovery bonus to quickly recover from false flags
            params["recovery_bonus"] = min(0.15, params["recovery_bonus"] + 0.03)

            # Lower recovery threshold for faster recovery
            params["recovery_threshold"] = max(2, params["recovery_threshold"] - 1)

        # Adjust based on observed cosine similarity distribution
        # If gradients are very similar (high mean cosine), widen range
        if cosine_stats["mean"] > 0.8:
            params["cos_max"] = min(0.99, params["cos_max"] + 0.03)
        elif cosine_stats["mean"] < 0.3:
            params["cos_min"] = max(-0.9, params["cos_min"] - 0.1)

        return strategy, params
