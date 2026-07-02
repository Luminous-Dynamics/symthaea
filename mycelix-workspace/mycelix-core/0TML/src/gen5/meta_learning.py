# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Meta-Learning Ensemble - Gen 5 Layer 1
=======================================

Auto-optimizing Byzantine detection ensemble that learns optimal weights
via online gradient descent.

Key Features:
- Online weight learning from labeled feedback
- Momentum-based optimization for fast convergence
- L2 weight decay for regularization
- Softmax normalization for numerical stability
- Method importance tracking for explainability

Algorithm:
----------
1. Initialize weights uniformly: w_i = 1/n
2. For each batch of gradients with labels:
   a. Compute signals from all base methods
   b. Ensemble score = Σ(w_i × signal_i) with softmax(w)
   c. Loss = Binary cross-entropy
   d. Gradient descent: w_t = w_{t-1} - lr × ∇L + momentum
3. Converges in ~50 iterations

Author: Luminous Dynamics
Date: November 11, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import json


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning ensemble."""

    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_iterations: int = 1000
    convergence_threshold: float = 1e-4
    min_batch_size: int = 10


class MetaLearningEnsemble:
    """
    Meta-learning ensemble that auto-optimizes detection weights.

    Combines multiple Byzantine detection methods (PoGQ, FLTrust, Krum, etc.)
    with learnable weights that adapt via online gradient descent.

    Example:
        >>> from defenses import get_defense
        >>> base_methods = [
        ...     get_defense("pogq_v4.1"),
        ...     get_defense("fltrust"),
        ...     get_defense("krum"),
        ... ]
        >>> ensemble = MetaLearningEnsemble(base_methods)
        >>>
        >>> # Training loop
        >>> for batch in training_data:
        ...     signals_batch, labels_batch = extract_batch(batch)
        ...     loss = ensemble.update_weights(signals_batch, labels_batch)
        ...     print(f"Loss: {loss:.4f}")
        >>>
        >>> # Inference
        >>> signals = ensemble.compute_signals(gradient)
        >>> score = ensemble.compute_ensemble_score(signals)
        >>> decision = "HONEST" if score >= 0.5 else "BYZANTINE"
    """

    def __init__(
        self,
        base_methods: List[Any],
        config: Optional[MetaLearningConfig] = None,
    ):
        """
        Initialize meta-learning ensemble.

        Args:
            base_methods: List of detection method instances.
                         Each must have a .score(gradient) method.
            config: Configuration parameters (optional)
        """
        if not base_methods:
            raise ValueError("Must provide at least one base detection method")

        self.base_methods = base_methods
        self.n_methods = len(base_methods)
        self.config = config or MetaLearningConfig()

        # Initialize weights uniformly
        self.weights = np.ones(self.n_methods) / self.n_methods

        # Momentum velocity
        self.velocity = np.zeros(self.n_methods)

        # Training state
        self.iteration = 0
        self.loss_history: List[float] = []
        self.weight_history: List[np.ndarray] = [self.weights.copy()]

        # Statistics
        self.stats = {
            "total_updates": 0,
            "total_samples": 0,
            "converged": False,
            "best_loss": float("inf"),
        }

    def compute_signals(self, gradient: np.ndarray) -> Dict[str, float]:
        """
        Extract detection signals from all base methods.

        Args:
            gradient: Client gradient vector (shape: (d,))

        Returns:
            Dict mapping method_name → signal_value ∈ [0.0, 1.0]

        Example:
            >>> signals = ensemble.compute_signals(gradient)
            >>> # Returns: {
            >>> #     'pogq_v4.1': 0.85,
            >>> #     'fltrust': 0.92,
            >>> #     'krum': 0.78,
            >>> # }
        """
        signals = {}

        for i, method in enumerate(self.base_methods):
            try:
                # Get method name
                if hasattr(method, "name"):
                    method_name = method.name
                elif hasattr(method, "__class__"):
                    method_name = method.__class__.__name__
                else:
                    method_name = f"method_{i}"

                # Compute signal
                if hasattr(method, "score"):
                    signal_value = method.score(gradient)
                elif hasattr(method, "detect"):
                    # Some methods have detect() instead of score()
                    result = method.detect(gradient)
                    if isinstance(result, dict) and "score" in result:
                        signal_value = result["score"]
                    elif isinstance(result, float):
                        signal_value = result
                    else:
                        signal_value = 0.5  # Neutral
                else:
                    raise AttributeError(
                        f"Method {method_name} has no score() or detect() method"
                    )

                # Clip to [0, 1]
                signals[method_name] = float(np.clip(signal_value, 0.0, 1.0))

            except Exception as e:
                # If method fails, use neutral score
                method_name = f"method_{i}"
                print(f"Warning: Method {method_name} failed: {e}")
                signals[method_name] = 0.5

        return signals

    def compute_ensemble_score(self, signals: Dict[str, float]) -> float:
        """
        Compute weighted ensemble score.

        Algorithm:
            1. Normalize weights via softmax: w_norm = exp(w) / Σ exp(w)
            2. Weighted average: score = Σ(w_norm_i × signal_i)

        Args:
            signals: Dict of method_name → signal_value

        Returns:
            Ensemble score ∈ [0.0, 1.0]
                - 0.0 = definitely Byzantine
                - 1.0 = definitely honest

        Example:
            >>> score = ensemble.compute_ensemble_score({
            ...     'pogq_v4.1': 0.85,
            ...     'fltrust': 0.92,
            ...     'krum': 0.78
            ... })
            >>> # Returns: ~0.85 (weighted average)
        """
        # Convert signals dict to array (same order as base_methods)
        signal_values = []
        for i, method in enumerate(self.base_methods):
            # Get method name
            if hasattr(method, "name"):
                method_name = method.name
            elif hasattr(method, "__class__"):
                method_name = method.__class__.__name__
            else:
                method_name = f"method_{i}"

            # Get signal value (default 0.5 if missing)
            signal_value = signals.get(method_name, 0.5)
            signal_values.append(signal_value)

        signal_values = np.array(signal_values)

        # Softmax normalization for numerical stability
        exp_weights = np.exp(self.weights - np.max(self.weights))  # Subtract max for stability
        normalized_weights = exp_weights / np.sum(exp_weights)

        # Weighted average
        ensemble_score = np.dot(normalized_weights, signal_values)

        return float(np.clip(ensemble_score, 0.0, 1.0))

    def update_weights(
        self,
        signals_batch: np.ndarray,
        labels_batch: np.ndarray,
    ) -> float:
        """
        Update ensemble weights via online gradient descent.

        Algorithm:
            Loss = Binary cross-entropy:
                L = -Σ[y log(p) + (1-y) log(1-p)]
            where p = ensemble_score

            Gradient:
                ∇L/∂w_i = Σ_j [(p_j - y_j) × signal_{j,i}] / n

            Update (with momentum):
                v_t = β × v_{t-1} - α × ∇L
                w_t = w_{t-1} + v_t - λ × w_{t-1}

        Args:
            signals_batch: Matrix of signals (n_samples × n_methods)
            labels_batch: True labels (n_samples,) where:
                         1.0 = HONEST, 0.0 = BYZANTINE

        Returns:
            Current loss value

        Example:
            >>> # Collect batch of 50 gradients
            >>> signals_batch = []  # Will be 50 × 3 matrix
            >>> labels_batch = []   # Will be 50 × 1 vector
            >>>
            >>> for gradient, label in batch:
            ...     signals = ensemble.compute_signals(gradient)
            ...     signals_batch.append([signals[m.name] for m in base_methods])
            ...     labels_batch.append(1.0 if label == "HONEST" else 0.0)
            >>>
            >>> loss = ensemble.update_weights(
            ...     np.array(signals_batch),
            ...     np.array(labels_batch)
            ... )
            >>> print(f"Loss: {loss:.4f}")
        """
        if len(signals_batch) != len(labels_batch):
            raise ValueError(
                f"Batch size mismatch: signals={len(signals_batch)}, "
                f"labels={len(labels_batch)}"
            )

        if len(signals_batch) < self.config.min_batch_size:
            raise ValueError(
                f"Batch too small: {len(signals_batch)} < "
                f"{self.config.min_batch_size}"
            )

        n_samples = len(labels_batch)

        # Normalize weights via softmax
        exp_weights = np.exp(self.weights - np.max(self.weights))
        normalized_weights = exp_weights / np.sum(exp_weights)

        # Compute ensemble predictions for each sample
        ensemble_scores = signals_batch @ normalized_weights

        # Binary cross-entropy loss
        eps = 1e-7  # Prevent log(0)
        ensemble_scores = np.clip(ensemble_scores, eps, 1.0 - eps)
        loss = -np.mean(
            labels_batch * np.log(ensemble_scores)
            + (1.0 - labels_batch) * np.log(1.0 - ensemble_scores)
        )

        # Gradient of loss w.r.t. weights
        # ∇L = signals^T @ (predictions - labels) / n
        residuals = ensemble_scores - labels_batch
        gradient = signals_batch.T @ residuals / n_samples

        # Momentum update
        self.velocity = (
            self.config.momentum * self.velocity - self.config.learning_rate * gradient
        )

        # Weight update with L2 decay
        self.weights = (
            self.weights + self.velocity - self.config.weight_decay * self.weights
        )

        # Track convergence
        self.iteration += 1
        self.loss_history.append(float(loss))
        self.weight_history.append(self.weights.copy())

        # Update statistics
        self.stats["total_updates"] += 1
        self.stats["total_samples"] += n_samples
        if loss < self.stats["best_loss"]:
            self.stats["best_loss"] = float(loss)

        # Check convergence
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_std = np.std(recent_losses)
            if loss_std < self.config.convergence_threshold:
                self.stats["converged"] = True

        return float(loss)

    def federated_update_weights(
        self,
        local_gradients: List[np.ndarray],
        agent_reputations: Optional[np.ndarray] = None,
        aggregation_method: str = "krum",
    ) -> float:
        """
        Federated weight update via Byzantine-robust aggregation.

        Applies Gen 5's own defenses to meta-gradients ("meta on meta").
        Agents compute local gradients with DP, coordinator aggregates.

        Algorithm:
            1. Collect local_gradients from agents
            2. Aggregate via Byzantine-robust method:
               - "krum": Multi-Krum (tolerates f < n/3)
               - "trimmed_mean": Remove top/bottom 10%, average
               - "median": Coordinate-wise median (tolerates 50%)
               - "reputation_weighted": Weight by agent reputation
            3. Apply aggregated gradient to global weights
            4. Update with momentum and weight decay

        Byzantine Tolerance:
            - krum: f < n/3 Byzantine agents
            - trimmed_mean: 10% Byzantine agents
            - median: Up to 50% Byzantine agents
            - reputation_weighted: Depends on reputation quality

        Args:
            local_gradients: List of noised gradients from agents
                            Each: (n_methods,) array
            agent_reputations: Optional reputation scores for weighting
                             Shape: (n_agents,)
            aggregation_method: Byzantine-robust aggregation method

        Returns:
            Estimated global loss (based on gradient norm)

        Example:
            >>> from gen5.federated_meta import LocalMetaLearner
            >>>
            >>> # Agents compute local gradients
            >>> agents = [
            ...     LocalMetaLearner(method_names=["pogq", "fltrust", "krum"])
            ...     for _ in range(10)
            ... ]
            >>>
            >>> local_grads = []
            >>> for agent, (local_signals, local_labels) in zip(agents, agent_data):
            ...     grad = agent.compute_local_gradient(
            ...         signals=local_signals,
            ...         labels=local_labels,
            ...         current_weights=ensemble.weights
            ...     )
            ...     local_grads.append(grad)
            >>>
            >>> # Coordinator aggregates
            >>> loss = ensemble.federated_update_weights(
            ...     local_gradients=local_grads,
            ...     aggregation_method="krum"
            ... )
        """
        # Import aggregation methods
        from .federated_meta import (
            aggregate_gradients_krum,
            aggregate_gradients_trimmed_mean,
            aggregate_gradients_median,
            aggregate_gradients_reputation_weighted,
        )

        if len(local_gradients) == 0:
            raise ValueError("No local gradients provided")

        # Aggregate gradients based on method
        if aggregation_method == "krum":
            aggregated_gradient = aggregate_gradients_krum(
                local_gradients, f=len(local_gradients) // 3
            )
        elif aggregation_method == "trimmed_mean":
            aggregated_gradient = aggregate_gradients_trimmed_mean(
                local_gradients, trim_ratio=0.1
            )
        elif aggregation_method == "median":
            aggregated_gradient = aggregate_gradients_median(local_gradients)
        elif aggregation_method == "reputation_weighted":
            if agent_reputations is None:
                raise ValueError(
                    "agent_reputations required for reputation_weighted aggregation"
                )
            aggregated_gradient = aggregate_gradients_reputation_weighted(
                local_gradients, agent_reputations
            )
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregation_method}. "
                f"Choose from: krum, trimmed_mean, median, reputation_weighted"
            )

        # Apply aggregated gradient via momentum update
        self.velocity = (
            self.config.momentum * self.velocity
            - self.config.learning_rate * aggregated_gradient
        )

        # Weight update with L2 decay
        self.weights = (
            self.weights + self.velocity - self.config.weight_decay * self.weights
        )

        # Track convergence
        self.iteration += 1

        # Estimate loss from gradient norm (no true labels available)
        estimated_loss = float(np.linalg.norm(aggregated_gradient))
        self.loss_history.append(estimated_loss)
        self.weight_history.append(self.weights.copy())

        # Update statistics
        self.stats["total_updates"] += 1
        self.stats["total_samples"] += len(local_gradients)  # Count agents
        if estimated_loss < self.stats["best_loss"]:
            self.stats["best_loss"] = estimated_loss

        # Check convergence
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_std = np.std(recent_losses)
            if loss_std < self.config.convergence_threshold:
                self.stats["converged"] = True

        return estimated_loss

    def get_method_importances(self) -> Dict[str, float]:
        """
        Get normalized method importances for explainability.

        Importance = softmax(weight_i) = exp(w_i) / Σ_j exp(w_j)

        These importances sum to 1.0 and represent each method's
        contribution to the final ensemble decision.

        Returns:
            Dict mapping method_name → importance ∈ [0.0, 1.0]

        Example:
            >>> importances = ensemble.get_method_importances()
            >>> # Returns: {
            >>> #     'pogq_v4.1': 0.35,
            >>> #     'fltrust': 0.45,
            >>> #     'krum': 0.20
            >>> # }
            >>> assert abs(sum(importances.values()) - 1.0) < 1e-6
        """
        # Softmax normalization
        exp_weights = np.exp(self.weights - np.max(self.weights))
        normalized = exp_weights / np.sum(exp_weights)

        importances = {}
        for i, method in enumerate(self.base_methods):
            # Get method name
            if hasattr(method, "name"):
                method_name = method.name
            elif hasattr(method, "__class__"):
                method_name = method.__class__.__name__
            else:
                method_name = f"method_{i}"

            importances[method_name] = float(normalized[i])

        return importances

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Get convergence metrics for monitoring training.

        Returns:
            Dict with:
                - current_loss: Latest loss value
                - best_loss: Minimum loss achieved
                - iterations: Number of updates
                - converged: Whether weights have converged
                - loss_std: Std dev of recent 10 losses
        """
        if not self.loss_history:
            return {
                "current_loss": None,
                "best_loss": float("inf"),
                "iterations": 0,
                "converged": False,
                "loss_std": None,
            }

        # Compute loss std from recent history
        recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
        loss_std = float(np.std(recent_losses))

        return {
            "current_loss": self.loss_history[-1],
            "best_loss": self.stats["best_loss"],
            "iterations": self.iteration,
            "converged": self.stats["converged"],
            "loss_std": loss_std,
            "total_samples": self.stats["total_samples"],
        }

    def save_weights(self, path: str | Path) -> None:
        """
        Save learned weights to file.

        Args:
            path: File path to save weights (.json)

        Example:
            >>> ensemble.save_weights("gen5_weights.json")
        """
        path = Path(path)

        # Get method names
        method_names = []
        for i, method in enumerate(self.base_methods):
            if hasattr(method, "name"):
                method_names.append(method.name)
            elif hasattr(method, "__class__"):
                method_names.append(method.__class__.__name__)
            else:
                method_names.append(f"method_{i}")

        # Create save data
        save_data = {
            "weights": self.weights.tolist(),
            "method_names": method_names,
            "iteration": self.iteration,
            "loss_history": self.loss_history,
            "stats": self.stats,
            "config": {
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
            },
        }

        # Save to JSON
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Weights saved to {path}")

    def load_weights(self, path: str | Path) -> None:
        """
        Load learned weights from file.

        Args:
            path: File path to load weights from (.json)

        Example:
            >>> ensemble.load_weights("gen5_weights.json")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        # Load from JSON
        with open(path, "r") as f:
            save_data = json.load(f)

        # Validate method names match
        saved_methods = save_data["method_names"]
        current_methods = [
            m.name if hasattr(m, "name") else m.__class__.__name__
            for m in self.base_methods
        ]

        if saved_methods != current_methods:
            raise ValueError(
                f"Method mismatch. Saved: {saved_methods}, "
                f"Current: {current_methods}"
            )

        # Load weights
        self.weights = np.array(save_data["weights"])
        self.iteration = save_data["iteration"]
        self.loss_history = save_data["loss_history"]
        self.stats = save_data["stats"]

        print(f"Weights loaded from {path} (iteration {self.iteration})")

    def __repr__(self) -> str:
        """String representation."""
        method_names = [
            m.name if hasattr(m, "name") else m.__class__.__name__
            for m in self.base_methods
        ]

        return (
            f"MetaLearningEnsemble(\n"
            f"  methods={method_names},\n"
            f"  iterations={self.iteration},\n"
            f"  converged={self.stats['converged']},\n"
            f"  best_loss={self.stats['best_loss']:.4f}\n"
            f")"
        )
