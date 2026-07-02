# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mode 1: Ground Truth (PoGQ) Byzantine Detection

This detector validates gradients against a held-out validation dataset,
enabling Byzantine tolerance beyond the 33-35% peer-comparison ceiling.

Key Innovation: Uses semantic validation (does gradient improve model loss?)
rather than peer comparison (is gradient similar to others?).

Author: Zero-TrustML Research Team
Date: November 4, 2025
Status: Production-ready implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, DefaultDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class QualityScore:
    """Gradient quality assessment"""
    node_id: int
    quality: float  # Possibly smoothed value used for thresholding
    baseline_loss: float
    updated_loss: float
    improvement: float
    is_byzantine: bool
    timestamp: float
    raw_quality: Optional[float] = None  # Raw (unsmoothed) score for diagnostics
    relative_improvement: Optional[float] = None
    gradient_norm: Optional[float] = None
    hybrid_component: Optional[float] = None
    class_improvements: Optional[Dict[int, float]] = None
    dominant_class: Optional[int] = None
    trust_score: Optional[float] = None
    mondrian_quality: Optional[float] = None
    stage1_flag: Optional[bool] = None
    stage2_flag: Optional[bool] = None


class GroundTruthDetector:
    """
    Mode 1: Proof of Gradient Quality (PoGQ) Detector

    Validates gradients by applying them to the global model and measuring
    improvement on a held-out validation dataset. Gradients that degrade
    performance are rejected as Byzantine.

    This approach exceeds the 33% BFT limit of peer-comparison because it
    uses an independent source of truth (validation data) rather than
    comparing gradients to each other.
    """

    def __init__(
        self,
        global_model: nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        quality_threshold: float = 0.5,
        learning_rate: float = 0.01,
        device: str = "cpu",
        adaptive_threshold: bool = True,
        mad_multiplier: float = 3.0,
        max_validation_batches: Optional[int] = None,
        loss_mode: str = "ce_logits",
        winsorize_p: Optional[float] = None,
        dispersion: str = "mad",
        threshold_mode: str = "gap",
        ema_alpha: Optional[float] = None,
        freeze_batchnorm: bool = False,
        relative_improvement: bool = False,
        gradient_clip: Optional[float] = None,
        hybrid_lambda: Optional[float] = None,
        prefilter_fltrust: bool = False,
        conformal_fpr: Optional[float] = None,
        expected_bft_ratio: Optional[float] = None,
    ):
        """
        Initialize Ground Truth detector.

        Args:
            global_model: The current global model
            validation_loader: DataLoader with held-out validation data
            quality_threshold: Minimum quality score (used if adaptive_threshold=False)
            learning_rate: LR for test gradient application
            device: Device for computation (cpu/cuda)
            adaptive_threshold: Use MAD-based adaptive threshold (recommended)
            mad_multiplier: Number of MADs below median for threshold (default: 3.0)
        """
        self.global_model = global_model
        self.validation_loader = validation_loader
        self.quality_threshold = quality_threshold
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.adaptive_threshold = adaptive_threshold
        self.mad_multiplier = mad_multiplier
        self.max_validation_batches = max_validation_batches
        self.loss_mode = loss_mode
        self.winsorize_p = winsorize_p
        self.dispersion = dispersion
        self.threshold_mode = threshold_mode
        self.ema_alpha = ema_alpha
        self.freeze_batchnorm = freeze_batchnorm
        self.relative_improvement = relative_improvement
        self.gradient_clip = gradient_clip
        self.hybrid_lambda = hybrid_lambda
        self.prefilter_fltrust = prefilter_fltrust
        self.conformal_fpr = conformal_fpr
        self.expected_bft_ratio = expected_bft_ratio
        self.last_mondrian_stats: Dict[int, Dict[str, float]] = {}

        # Statistics
        self.quality_scores: List[QualityScore] = []
        self.detection_count = 0
        self.total_evaluated = 0
        self.computed_threshold = quality_threshold  # Will be updated if adaptive

        # Internal state
        self._ema_scores: Dict[int, float] = {}
        self._reference_gradient: Optional[np.ndarray] = None

        if self.freeze_batchnorm:
            self._set_batchnorm_eval(self.global_model)

    def _flatten_gradient_dict(self, gradient: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten gradient dictionary into a single vector."""
        return np.concatenate([np.asarray(values, dtype=np.float64).ravel() for values in gradient.values()])

    def _scale_gradient_dict(self, gradient: Dict[str, np.ndarray], scale: float) -> Dict[str, np.ndarray]:
        """Scale every tensor in the gradient dictionary."""
        if scale == 1.0:
            return gradient
        return {name: np.asarray(values) * scale for name, values in gradient.items()}

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar loss according to configured mode."""
        if self.loss_mode in ("ce", "ce_logits"):
            criterion = nn.CrossEntropyLoss()
            return criterion(logits, target)

        if self.loss_mode == "mse_logits":
            target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
            return F.mse_loss(logits, target_one_hot)

        raise ValueError(f"Unsupported loss_mode '{self.loss_mode}'")

    def _set_batchnorm_eval(self, model: nn.Module) -> None:
        """Freeze BatchNorm statistics for deterministic evaluation."""
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.track_running_stats = False

    def compute_validation_loss(self, model: nn.Module) -> Tuple[float, Dict[int, float], Dict[int, int]]:
        """
        Compute loss on validation dataset.

        Args:
            model: Model to evaluate

        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        per_class_loss: DefaultDict[int, float] = defaultdict(float)
        per_class_count: DefaultDict[int, int] = defaultdict(int)

        with torch.no_grad():
            batch_count = 0
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                per_sample = F.cross_entropy(output, target, reduction="none")

                batch_loss = float(per_sample.mean().item())
                total_loss += batch_loss * len(data)
                total_samples += len(data)

                for cls, loss_value in zip(target.tolist(), per_sample.tolist()):
                    per_class_loss[cls] += loss_value
                    per_class_count[cls] += 1
                batch_count += 1

                if self.max_validation_batches and batch_count >= self.max_validation_batches:
                    break

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        class_means = {
            cls: per_class_loss[cls] / max(1, per_class_count[cls])
            for cls in per_class_loss
        }
        return avg_loss, class_means, dict(per_class_count)

    def apply_gradient(
        self,
        model: nn.Module,
        gradient: Dict[str, np.ndarray]
    ) -> nn.Module:
        """
        Apply a gradient to create an updated model (without modifying original).

        Args:
            model: Current model
            gradient: Gradient dictionary (param_name -> gradient_array)

        Returns:
            New model with gradient applied
        """
        # Create a copy of the model
        import copy
        updated_model = copy.deepcopy(model)
        if self.freeze_batchnorm:
            self._set_batchnorm_eval(updated_model)

        # Apply gradient
        with torch.no_grad():
            for name, param in updated_model.named_parameters():
                if name in gradient:
                    grad_tensor = torch.as_tensor(
                        gradient[name],
                        device=self.device,
                        dtype=param.data.dtype,
                    )
                    param.data -= self.learning_rate * grad_tensor

        return updated_model

    def _validate_gradient_shapes(self, gradient: Dict[str, np.ndarray]) -> None:
        """Ensure gradients align with model parameter shapes and contain finite values."""
        model_params = {name: param for name, param in self.global_model.named_parameters()}

        for name, array in gradient.items():
            if name not in model_params:
                raise KeyError(f"Gradient provided for unknown parameter '{name}'")

            expected_shape = tuple(model_params[name].data.cpu().numpy().shape)
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Gradient for '{name}' has shape {array.shape}, expected {expected_shape}"
                )

            if not np.isfinite(array).all():
                raise ValueError(f"Gradient for '{name}' contains NaN or Inf values")

            if array.dtype not in (np.float16, np.float32, np.float64):
                raise TypeError(
                    f"Gradient for '{name}' must be floating point, received {array.dtype}"
                )

    def compute_quality(
        self,
        gradient: Dict[str, np.ndarray],
        node_id: int,
        baseline_stats: Optional[Tuple[float, Dict[int, float], Dict[int, int]]] = None,
    ) -> QualityScore:
        """
        Compute gradient quality by measuring validation loss improvement.

        Args:
            gradient: Gradient to evaluate
            node_id: ID of node that submitted gradient

        Returns:
            QualityScore with assessment
        """
        self._validate_gradient_shapes(gradient)
        flat_gradient = self._flatten_gradient_dict(gradient)
        gradient_norm = float(np.linalg.norm(flat_gradient) + 1e-12)

        if self.gradient_clip and gradient_norm > self.gradient_clip:
            scale = self.gradient_clip / gradient_norm
            gradient = self._scale_gradient_dict(gradient, scale)
            flat_gradient = flat_gradient * scale
            gradient_norm = float(np.linalg.norm(flat_gradient) + 1e-12)

        # Measure baseline loss (cached per round when provided)
        if baseline_stats is None:
            baseline_loss, baseline_classes, baseline_counts = self.compute_validation_loss(self.global_model)
        else:
            baseline_loss, baseline_classes, baseline_counts = baseline_stats

        # Apply gradient and measure new loss
        updated_model = self.apply_gradient(self.global_model, gradient)
        updated_loss, updated_classes, _ = self.compute_validation_loss(updated_model)

        # Compute improvement
        improvement = baseline_loss - updated_loss  # Positive = better
        relative_improvement = None
        improvement_metric = improvement
        if self.relative_improvement:
            denom = max(abs(baseline_loss), 1e-8)
            relative_improvement = improvement / denom
            improvement_metric = relative_improvement

        # Normalize to 0-1 quality score using sigmoid
        raw_quality = 1.0 / (1.0 + np.exp(-10.0 * improvement_metric))

        # Optional exponential moving average smoothing
        if self.ema_alpha is not None:
            previous = self._ema_scores.get(node_id, raw_quality)
            quality = self.ema_alpha * raw_quality + (1.0 - self.ema_alpha) * previous
            self._ema_scores[node_id] = quality
        else:
            quality = raw_quality

        class_improvements = None
        dominant_class = None
        if baseline_classes and updated_classes:
            class_improvements = {}
            for cls, base_loss in baseline_classes.items():
                updated = updated_classes.get(cls, base_loss)
                class_improvements[cls] = base_loss - updated
            if class_improvements:
                dominant_class = max(class_improvements, key=class_improvements.get)

        hybrid_component = None
        if self.hybrid_lambda is not None:
            if self._reference_gradient is None:
                self._reference_gradient = self._compute_reference_gradient()
            ref_vec = self._reference_gradient
            if ref_vec is not None and ref_vec.size == flat_gradient.size:
                denom = (np.linalg.norm(flat_gradient) * np.linalg.norm(ref_vec)) + 1e-12
                cosine = float(np.dot(flat_gradient, ref_vec) / denom)
                hybrid_component = (cosine + 1.0) / 2.0
                quality = self.hybrid_lambda * quality + (1.0 - self.hybrid_lambda) * hybrid_component

        return QualityScore(
            node_id=node_id,
            quality=quality,
            baseline_loss=baseline_loss,
            updated_loss=updated_loss,
            improvement=improvement,
            is_byzantine=False,
            timestamp=time.time(),
            raw_quality=raw_quality,
            relative_improvement=relative_improvement,
            gradient_norm=gradient_norm,
            hybrid_component=hybrid_component,
            class_improvements=class_improvements,
            dominant_class=dominant_class,
        )

    def compute_adaptive_threshold(self, qualities: List[float]) -> float:
        """
        Compute adaptive threshold using the configured strategy.

        Strategies:
            - "gap" (default): Find the largest gap between sorted scores
            - "robust": Median minus MAD/biweight-based dispersion (winsorized)
        """
        values = np.asarray(qualities, dtype=np.float64)
        if values.size == 0:
            return self.quality_threshold

        if self.threshold_mode == "robust":
            return self._compute_robust_threshold(values)

        return self._compute_gap_threshold(values)

    def _compute_gap_threshold(self, qualities_array: np.ndarray) -> float:
        """Gap-based adaptive threshold (legacy behavior)."""
        n = len(qualities_array)
        if n < 2:
            return float(np.median(qualities_array))

        sorted_qualities = np.sort(qualities_array)
        gaps = np.diff(sorted_qualities)

        start_idx = max(1, int(n * 0.1))
        end_idx = min(len(gaps) - 1, int(n * 0.9))
        middle_gaps = gaps[start_idx:end_idx]

        if len(middle_gaps) == 0:
            return float(np.median(qualities_array))

        max_gap_idx = start_idx + int(np.argmax(middle_gaps))
        threshold = (sorted_qualities[max_gap_idx] + sorted_qualities[max_gap_idx + 1]) / 2.0
        return float(threshold)

    def _compute_robust_threshold(self, qualities_array: np.ndarray) -> float:
        """Median minus dispersion (MAD/Biweight) with optional winsorization."""
        if self.winsorize_p:
            qualities_array = self._winsorize_values(qualities_array, self.winsorize_p)

        median = float(np.median(qualities_array))
        dispersion = self._robust_dispersion(qualities_array, self.dispersion)
        threshold = median - self.mad_multiplier * dispersion
        return float(threshold)

    @staticmethod
    def _winsorize_values(values: np.ndarray, proportion: float) -> np.ndarray:
        """Clip both tails of the distribution to reduce outlier influence."""
        if values.size == 0:
            return values
        lower = np.quantile(values, proportion)
        upper = np.quantile(values, 1.0 - proportion)
        return np.clip(values, lower, upper)

    @staticmethod
    def _robust_dispersion(values: np.ndarray, mode: str) -> float:
        """Return dispersion estimate using MAD or biweight midvariance."""
        mode = mode.lower()
        if mode == "mad":
            mad = np.median(np.abs(values - np.median(values)))
            return float(mad + 1e-12)

        if mode == "biweight":
            median = np.median(values)
            mad = np.median(np.abs(values - median)) + 1e-12
            u = (values - median) / (9.0 * mad)
            mask = np.abs(u) < 1
            if not np.any(mask):
                return float(mad)
            numerator = ((values[mask] - median) ** 2 * (1 - u[mask] ** 2) ** 4).sum()
            denominator = ((1 - u[mask] ** 2) * (1 - 5 * u[mask] ** 2)).sum() + 1e-12
            return float(np.sqrt(numerator / denominator + 1e-12))

        raise ValueError(f"Unsupported dispersion mode '{mode}'")

    def _compute_reference_gradient(self) -> Optional[np.ndarray]:
        """Compute reference gradient on the validation set for hybrid scoring."""
        self.global_model.zero_grad()
        self.global_model.eval()
        if self.freeze_batchnorm:
            self._set_batchnorm_eval(self.global_model)

        criterion = nn.CrossEntropyLoss()
        accumulators = [torch.zeros_like(param.data) for param in self.global_model.parameters()]
        batches = 0
        max_batches = self.max_validation_batches or len(self.validation_loader)

        for data, target in self.validation_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.global_model.zero_grad()
            with torch.enable_grad():
                output = self.global_model(data)
                if self.loss_mode == "mse_logits":
                    target_one_hot = F.one_hot(target, num_classes=output.shape[1]).float()
                    loss = F.mse_loss(output, target_one_hot)
                else:
                    loss = criterion(output, target)
                loss.backward()

            for acc, param in zip(accumulators, self.global_model.parameters()):
                if param.grad is not None:
                    acc.add_(param.grad.detach())
            batches += 1
            if batches >= max_batches:
                break

        if batches == 0:
            return None

        flat = np.concatenate([acc.cpu().numpy().ravel() / batches for acc in accumulators])
        return flat.astype(np.float64)

    def _compute_fltrust_scores(self, gradients: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, float]:
        """Compute FLTrust trust (ReLU cosine) for each gradient."""
        if not gradients:
            return {}

        if self._reference_gradient is None:
            self._reference_gradient = self._compute_reference_gradient()

        ref_vec = self._reference_gradient
        if ref_vec is None:
            return {node_id: 0.0 for node_id in gradients}

        ref_norm = np.linalg.norm(ref_vec) + 1e-12
        trust_scores: Dict[int, float] = {}
        for node_id, gradient in gradients.items():
            flat = self._flatten_gradient_dict(gradient)
            denom = (np.linalg.norm(flat) * ref_norm) + 1e-12
            cosine = float(np.dot(flat, ref_vec) / denom)
            trust_scores[node_id] = max(0.0, cosine)
        return trust_scores

    def _min_survivor_count(self, population: int) -> int:
        """Minimum number of FLTrust survivors before falling back to PoGQ-only."""
        if population <= 0:
            return 0
        if self.expected_bft_ratio is None:
            return max(1, population // 4)
        honest_estimate = max(1, int(round((1.0 - self.expected_bft_ratio) * population)))
        return max(1, honest_estimate // 2)

    def _prepare_threshold_inputs(
        self,
        scores: List[QualityScore],
    ) -> Tuple[List[float], Dict[int, float]]:
        """Return (values, mapping) for the configured threshold mode."""
        if not scores:
            return [], {}

        if self.threshold_mode == "mondrian":
            values, mapping = self._compute_mondrian_scores(scores)
            return values, mapping

        values = [score.quality for score in scores]
        mapping = {score.node_id: score.quality for score in scores}
        return values, mapping

    def _compute_mondrian_scores(
        self,
        scores: List[QualityScore],
    ) -> Tuple[List[float], Dict[int, float]]:
        """Compute class-aware normalized scores for Mondrian thresholding."""
        class_values: Dict[int, List[float]] = defaultdict(list)
        for score in scores:
            if not score.class_improvements:
                continue
            for cls, value in score.class_improvements.items():
                class_values[cls].append(value)

        class_stats: Dict[int, Dict[str, float]] = {}
        for cls, values in class_values.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            dispersion = self._robust_dispersion(arr, self.dispersion)
            class_stats[cls] = {
                "median": float(np.median(arr)),
                "dispersion": float(dispersion),
            }

        normalized: List[float] = []
        mapping: Dict[int, float] = {}
        for score in scores:
            z_values = []
            if score.class_improvements:
                for cls, value in score.class_improvements.items():
                    stats = class_stats.get(cls)
                    if not stats:
                        continue
                    dispersion = stats["dispersion"] if stats["dispersion"] > 0 else 1e-12
                    z = (value - stats["median"]) / (dispersion + 1e-12)
                    z_values.append(z)
            if z_values:
                mondrian_value = float(np.mean(z_values))
            else:
                mondrian_value = score.quality
            score.mondrian_quality = mondrian_value
            normalized.append(mondrian_value)
            mapping[score.node_id] = mondrian_value

        self.last_mondrian_stats = class_stats
        return normalized, mapping

    @staticmethod
    def _compute_conformal_threshold(values: np.ndarray, alpha: float) -> float:
        """Quantile-based threshold ensuring ≤ alpha false-positive rate."""
        if values.size == 0:
            return 0.0
        clipped_alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-6))
        sorted_vals = np.sort(values)
        index = int(np.floor(clipped_alpha * len(sorted_vals)))
        index = min(max(index, 0), len(sorted_vals) - 1)
        return float(sorted_vals[index])

    def detect_byzantine(
        self,
        gradients: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, bool]:
        """
        Detect Byzantine nodes using FLTrust prefilter + PoGQ adaptive thresholding.
        """
        self._reference_gradient = None
        round_scores: List[QualityScore] = []
        score_by_node: Dict[int, QualityScore] = {}
        baseline_stats = self.compute_validation_loss(self.global_model)

        for node_id, gradient in gradients.items():
            score = self.compute_quality(gradient, node_id, baseline_stats)
            round_scores.append(score)
            score_by_node[node_id] = score

        # Stage 1: Optional FLTrust prefilter
        survivors_mask = {node_id: True for node_id in gradients}
        stage1_flags = {node_id: False for node_id in gradients}
        trust_scores: Dict[int, float] = {}

        if self.prefilter_fltrust:
            trust_scores = self._compute_fltrust_scores(gradients)
            min_survivors = self._min_survivor_count(len(gradients))
            survivors = 0
            for node_id, trust in trust_scores.items():
                score_by_node[node_id].trust_score = trust
                keep = trust > 0.0
                survivors_mask[node_id] = keep
                stage1_flags[node_id] = not keep
                if keep:
                    survivors += 1
            if survivors < min_survivors:
                survivors_mask = {node_id: True for node_id in gradients}
                stage1_flags = {node_id: False for node_id in gradients}
        else:
            for score in round_scores:
                score.trust_score = None

        survivor_scores = [
            score_by_node[node_id]
            for node_id, keep in survivors_mask.items()
            if keep
        ]

        threshold_values, value_map = self._prepare_threshold_inputs(survivor_scores)
        values_array = np.asarray(threshold_values, dtype=np.float64)

        if survivor_scores:
            if self.adaptive_threshold:
                adaptive_threshold = self.compute_adaptive_threshold(values_array)
            else:
                adaptive_threshold = self.quality_threshold

            if self.conformal_fpr is not None:
                conformal_threshold = self._compute_conformal_threshold(values_array, self.conformal_fpr)
                if self.adaptive_threshold:
                    self.computed_threshold = min(adaptive_threshold, conformal_threshold)
                else:
                    self.computed_threshold = conformal_threshold
            else:
                self.computed_threshold = adaptive_threshold
        else:
            self.computed_threshold = self.quality_threshold

        detections: Dict[int, bool] = {}
        for node_id, score in score_by_node.items():
            stage1_flag = stage1_flags.get(node_id, False)
            stage2_flag = False
            threshold_input = value_map.get(node_id)
            if threshold_input is not None:
                stage2_flag = threshold_input < self.computed_threshold

            score.stage1_flag = stage1_flag
            score.stage2_flag = stage2_flag
            score.is_byzantine = stage1_flag or stage2_flag

            self.quality_scores.append(score)
            detections[node_id] = score.is_byzantine
            self.total_evaluated += 1
            if score.is_byzantine:
                self.detection_count += 1

        self._reference_gradient = None
        return detections

    def get_quality_statistics(self) -> Dict:
        """Get statistics about quality scores."""
        if not self.quality_scores:
            return {
                'mean_quality': 0.0,
                'std_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
                'detection_rate': 0.0,
                'total_evaluated': 0
            }

        qualities = [s.quality for s in self.quality_scores]

        return {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'detection_rate': self.detection_count / self.total_evaluated,
            'total_evaluated': self.total_evaluated
        }

    def get_quality_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quality score distribution for honest vs Byzantine nodes.

        Returns:
            Tuple of (honest_qualities, byzantine_qualities)
        """
        honest = [s.quality for s in self.quality_scores if not s.is_byzantine]
        byzantine = [s.quality for s in self.quality_scores if s.is_byzantine]

        return np.array(honest), np.array(byzantine)

    def reset_statistics(self):
        """Reset detection statistics."""
        self.quality_scores = []
        self.detection_count = 0
        self.total_evaluated = 0
        self._ema_scores = {}


class HybridGroundTruthDetector(GroundTruthDetector):
    """
    Enhanced Mode 1 detector with additional signals for robustness.

    Combines:
    - Quality score (primary): Validation loss improvement
    - Reputation (secondary): Historical behavior
    - Magnitude (safety): Catch extreme outliers
    """

    def __init__(
        self,
        global_model: nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        quality_threshold: float = 0.5,
        quality_weight: float = 0.7,
        reputation_weight: float = 0.2,
        magnitude_weight: float = 0.1,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(
            global_model,
            validation_loader,
            quality_threshold,
            learning_rate,
            device
        )

        # Ensemble weights
        self.quality_weight = quality_weight
        self.reputation_weight = reputation_weight
        self.magnitude_weight = magnitude_weight

        # Reputation tracking
        self.reputations: Dict[int, float] = {}
        self.reputation_penalty = 0.5  # Multiply by this on detection
        self.reputation_recovery = 0.05  # Add this on honest behavior

    def compute_magnitude_confidence(
        self,
        gradient: Dict[str, np.ndarray],
        gradients: Dict[int, Dict[str, np.ndarray]]
    ) -> float:
        """
        Compute confidence that gradient has abnormal magnitude.

        Args:
            gradient: Gradient to check
            gradients: All gradients for comparison

        Returns:
            Confidence 0-1 (1 = definitely abnormal)
        """
        # Compute L2 norm of this gradient
        norm = 0.0
        for param_name, param_grad in gradient.items():
            norm += np.sum(param_grad ** 2)
        norm = np.sqrt(norm)

        # Compute norms of all gradients
        all_norms = []
        for g in gradients.values():
            g_norm = 0.0
            for param_grad in g.values():
                g_norm += np.sum(param_grad ** 2)
            all_norms.append(np.sqrt(g_norm))

        # Z-score analysis
        mean_norm = np.mean(all_norms)
        std_norm = np.std(all_norms) + 1e-10
        z_score = abs((norm - mean_norm) / std_norm)

        # Confidence: z > 3 → high confidence it's abnormal
        return min(1.0, z_score / 3.0)

    def get_reputation(self, node_id: int) -> float:
        """Get reputation for a node (default: 1.0)."""
        return self.reputations.get(node_id, 1.0)

    def update_reputation(self, node_id: int, is_byzantine: bool):
        """Update reputation based on detection result."""
        current = self.get_reputation(node_id)

        if is_byzantine:
            # Exponential penalty
            self.reputations[node_id] = current * self.reputation_penalty
        else:
            # Gradual recovery
            self.reputations[node_id] = min(
                1.0,
                current * 0.95 + self.reputation_recovery
            )

    def detect_byzantine(
        self,
        gradients: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, bool]:
        """
        Hybrid detection using quality + reputation + magnitude.

        Args:
            gradients: Dictionary mapping node_id -> gradient_dict

        Returns:
            Dictionary mapping node_id -> is_byzantine (True = Byzantine)
        """
        detections = {}

        baseline_stats = self.compute_validation_loss(self.global_model)

        for node_id, gradient in gradients.items():
            # 1. Quality score (primary signal)
            quality_score = self.compute_quality(gradient, node_id, baseline_stats)
            quality_conf = 1.0 - quality_score.quality  # Invert for "Byzantine confidence"

            # 2. Reputation signal
            reputation = self.get_reputation(node_id)
            reputation_conf = 1.0 - reputation  # Low reputation → high suspicion

            # 3. Magnitude signal
            magnitude_conf = self.compute_magnitude_confidence(gradient, gradients)

            # 4. Ensemble decision
            ensemble_confidence = (
                self.quality_weight * quality_conf +
                self.reputation_weight * reputation_conf +
                self.magnitude_weight * magnitude_conf
            )

            is_byzantine = ensemble_confidence > 0.5

            # Update reputation
            self.update_reputation(node_id, is_byzantine)

            # Record
            quality_score.is_byzantine = is_byzantine  # Update with ensemble result
            self.quality_scores.append(quality_score)
            detections[node_id] = is_byzantine

            # Update statistics
            self.total_evaluated += 1
            if is_byzantine:
                self.detection_count += 1

        return detections


# Convenience factory function
def create_ground_truth_detector(
    model: nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    mode: str = "simple",
    **kwargs
) -> GroundTruthDetector:
    """
    Factory function to create appropriate detector.

    Args:
        model: Global model
        validation_loader: Validation data
        mode: "simple" or "hybrid"
        **kwargs: Additional detector arguments

    Returns:
        Configured detector
    """
    if mode == "simple":
        return GroundTruthDetector(model, validation_loader, **kwargs)
    elif mode == "hybrid":
        return HybridGroundTruthDetector(model, validation_loader, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
