# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Input Validation Utilities

Provides validation functions for gradients, configurations, and other inputs.
All functions either return validated data or raise descriptive exceptions.

Author: Luminous Dynamics
Date: December 31, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging

from .config import (
    EPSILON,
    MAX_GRADIENT_VALUE,
    MIN_GRADIENT_NORM,
    BYZANTINE_THRESHOLD_MAX,
    validate_dimension,
    get_nearest_valid_dimension,
)
from .exceptions import (
    InvalidGradientError,
    GradientShapeMismatchError,
    InsufficientGradientsError,
    ConfigurationError,
    InvalidDimensionError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GRADIENT VALIDATION
# =============================================================================

def validate_gradient(
    gradient: np.ndarray,
    node_id: Optional[str] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    max_value: float = MAX_GRADIENT_VALUE,
) -> np.ndarray:
    """
    Validate a single gradient array.

    Args:
        gradient: Gradient array to validate
        node_id: Optional node ID for error messages
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow Inf values
        max_value: Maximum allowed absolute value

    Returns:
        Validated gradient (same array if valid)

    Raises:
        InvalidGradientError: If gradient is invalid
    """
    node_str = f" from node '{node_id}'" if node_id else ""

    # Check type
    if not isinstance(gradient, np.ndarray):
        raise InvalidGradientError(
            f"Gradient{node_str} must be numpy array, got {type(gradient).__name__}",
            node_id=node_id,
            issue="wrong_type",
        )

    # Check not empty
    if gradient.size == 0:
        raise InvalidGradientError(
            f"Gradient{node_str} is empty",
            node_id=node_id,
            gradient_shape=gradient.shape,
            issue="empty",
        )

    # Check for NaN
    if not allow_nan and np.any(np.isnan(gradient)):
        nan_count = np.sum(np.isnan(gradient))
        raise InvalidGradientError(
            f"Gradient{node_str} contains {nan_count} NaN values",
            node_id=node_id,
            gradient_shape=gradient.shape,
            issue="contains_nan",
        )

    # Check for Inf
    if not allow_inf and np.any(np.isinf(gradient)):
        inf_count = np.sum(np.isinf(gradient))
        raise InvalidGradientError(
            f"Gradient{node_str} contains {inf_count} Inf values",
            node_id=node_id,
            gradient_shape=gradient.shape,
            issue="contains_inf",
        )

    # Check value range
    max_abs = np.max(np.abs(gradient[~np.isnan(gradient) & ~np.isinf(gradient)]))
    if max_abs > max_value:
        raise InvalidGradientError(
            f"Gradient{node_str} has values exceeding {max_value:.0e} (max={max_abs:.2e})",
            node_id=node_id,
            gradient_shape=gradient.shape,
            issue="value_overflow",
        )

    return gradient


def validate_gradients(
    gradients: Dict[str, np.ndarray],
    min_nodes: int = 1,
    require_same_shape: bool = True,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Validate a dictionary of gradients from multiple nodes.

    Args:
        gradients: Dict of node_id -> gradient array
        min_nodes: Minimum required number of nodes
        require_same_shape: Whether all gradients must have same shape
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow Inf values

    Returns:
        Validated gradients dict

    Raises:
        InsufficientGradientsError: If not enough gradients
        InvalidGradientError: If any gradient is invalid
        GradientShapeMismatchError: If shapes don't match
    """
    # Check minimum nodes
    if len(gradients) < min_nodes:
        raise InsufficientGradientsError(
            received=len(gradients),
            minimum=min_nodes,
        )

    # Validate each gradient
    reference_shape = None
    for node_id, gradient in gradients.items():
        validate_gradient(
            gradient,
            node_id=node_id,
            allow_nan=allow_nan,
            allow_inf=allow_inf,
        )

        # Check shape consistency
        if require_same_shape:
            flat_shape = (gradient.size,)
            if reference_shape is None:
                reference_shape = flat_shape
            elif flat_shape != reference_shape:
                raise GradientShapeMismatchError(
                    expected_shape=reference_shape,
                    actual_shape=flat_shape,
                    node_id=node_id,
                )

    return gradients


def sanitize_gradient(
    gradient: np.ndarray,
    replace_nan: float = 0.0,
    replace_inf: Optional[float] = None,
    clip_value: Optional[float] = None,
) -> np.ndarray:
    """
    Sanitize a gradient by replacing invalid values.

    Args:
        gradient: Gradient to sanitize
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with (None = clip to max finite)
        clip_value: Optional value to clip all values to

    Returns:
        Sanitized gradient (new array, doesn't modify input)
    """
    result = gradient.copy()

    # Replace NaN
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        result[nan_mask] = replace_nan
        logger.debug(f"Replaced {np.sum(nan_mask)} NaN values with {replace_nan}")

    # Replace Inf
    inf_mask = np.isinf(result)
    if np.any(inf_mask):
        if replace_inf is not None:
            result[inf_mask] = np.sign(result[inf_mask]) * replace_inf
        else:
            # Clip to max finite value
            finite_max = np.max(np.abs(result[~inf_mask])) if np.any(~inf_mask) else 1.0
            result[inf_mask] = np.sign(result[inf_mask]) * finite_max
        logger.debug(f"Replaced {np.sum(inf_mask)} Inf values")

    # Clip values
    if clip_value is not None:
        result = np.clip(result, -clip_value, clip_value)

    return result


def compute_gradient_stats(gradient: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a gradient.

    Returns:
        Dict with norm, mean, std, min, max, sparsity
    """
    # Handle NaN/Inf for stats
    finite_grad = gradient[np.isfinite(gradient)]

    if len(finite_grad) == 0:
        return {
            "norm": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sparsity": 1.0,
            "nan_ratio": 1.0,
            "inf_ratio": 0.0,
        }

    return {
        "norm": float(np.linalg.norm(finite_grad)),
        "mean": float(np.mean(finite_grad)),
        "std": float(np.std(finite_grad)),
        "min": float(np.min(finite_grad)),
        "max": float(np.max(finite_grad)),
        "sparsity": float(np.sum(np.abs(finite_grad) < EPSILON) / len(finite_grad)),
        "nan_ratio": float(np.sum(np.isnan(gradient)) / gradient.size),
        "inf_ratio": float(np.sum(np.isinf(gradient)) / gradient.size),
    }


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_byzantine_threshold(threshold: float) -> float:
    """
    Validate Byzantine threshold.

    Args:
        threshold: Threshold value (0.0 to 0.5)

    Returns:
        Validated threshold

    Raises:
        ConfigurationError: If threshold invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ConfigurationError(
            "Byzantine threshold must be a number",
            param_name="byzantine_threshold",
            param_value=threshold,
        )

    if threshold < 0.0:
        raise ConfigurationError(
            "Byzantine threshold cannot be negative",
            param_name="byzantine_threshold",
            param_value=threshold,
            valid_range="0.0 to 0.5",
        )

    if threshold > BYZANTINE_THRESHOLD_MAX:
        raise ConfigurationError(
            f"Byzantine threshold exceeds maximum ({BYZANTINE_THRESHOLD_MAX})",
            param_name="byzantine_threshold",
            param_value=threshold,
            valid_range=f"0.0 to {BYZANTINE_THRESHOLD_MAX}",
        )

    return float(threshold)


def validate_hv_dimension(dimension: int) -> int:
    """
    Validate hypervector dimension.

    Args:
        dimension: Dimension value (must be power of 2)

    Returns:
        Validated dimension

    Raises:
        InvalidDimensionError: If dimension invalid
    """
    if not isinstance(dimension, int):
        raise ConfigurationError(
            "Dimension must be an integer",
            param_name="dimension",
            param_value=dimension,
        )

    if not validate_dimension(dimension):
        nearest = get_nearest_valid_dimension(dimension)
        raise InvalidDimensionError(dimension, nearest_valid=nearest)

    return dimension


def validate_positive_int(value: int, name: str) -> int:
    """Validate a positive integer parameter."""
    if not isinstance(value, int) or value <= 0:
        raise ConfigurationError(
            f"{name} must be a positive integer",
            param_name=name,
            param_value=value,
            valid_range="> 0",
        )
    return value


def validate_positive_float(value: float, name: str) -> float:
    """Validate a positive float parameter."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(
            f"{name} must be a positive number",
            param_name=name,
            param_value=value,
            valid_range="> 0",
        )
    return float(value)


def validate_probability(value: float, name: str) -> float:
    """Validate a probability value (0 to 1)."""
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ConfigurationError(
            f"{name} must be between 0 and 1",
            param_name=name,
            param_value=value,
            valid_range="0.0 to 1.0",
        )
    return float(value)


# =============================================================================
# NODE ID VALIDATION
# =============================================================================

def validate_node_id(node_id: str) -> str:
    """
    Validate a node ID string.

    Args:
        node_id: Node identifier

    Returns:
        Validated node ID

    Raises:
        ConfigurationError: If node ID invalid
    """
    if not isinstance(node_id, str):
        raise ConfigurationError(
            "Node ID must be a string",
            param_name="node_id",
            param_value=node_id,
        )

    if not node_id or node_id.isspace():
        raise ConfigurationError(
            "Node ID cannot be empty or whitespace",
            param_name="node_id",
            param_value=repr(node_id),
        )

    # Check for problematic characters
    problematic = set('<>:"/\\|?*')
    if any(c in node_id for c in problematic):
        raise ConfigurationError(
            f"Node ID contains invalid characters: {problematic}",
            param_name="node_id",
            param_value=node_id,
        )

    return node_id


def validate_node_ids(node_ids: Union[List[str], Set[str]]) -> Set[str]:
    """
    Validate a collection of node IDs.

    Returns:
        Set of validated node IDs
    """
    validated = set()
    for node_id in node_ids:
        validated.add(validate_node_id(node_id))
    return validated


# =============================================================================
# RESULT VALIDATION
# =============================================================================

def validate_shapley_values(
    shapley_values: Dict[str, float],
    node_ids: Optional[Set[str]] = None,
    require_normalized: bool = False,
) -> Dict[str, float]:
    """
    Validate Shapley values.

    Args:
        shapley_values: Dict of node_id -> Shapley value
        node_ids: Optional set of expected node IDs
        require_normalized: Whether values should sum to ~1

    Returns:
        Validated Shapley values
    """
    # Check all values are finite
    for node_id, value in shapley_values.items():
        if not np.isfinite(value):
            raise ConfigurationError(
                f"Shapley value for '{node_id}' is not finite",
                param_name="shapley_values",
                param_value=value,
            )

    # Check expected nodes present
    if node_ids:
        missing = node_ids - set(shapley_values.keys())
        if missing:
            logger.warning(f"Missing Shapley values for nodes: {missing}")

    # Check normalization
    if require_normalized:
        total = sum(shapley_values.values())
        if not 0.9 <= total <= 1.1:
            logger.warning(
                f"Shapley values sum to {total:.3f}, expected ~1.0"
            )

    return shapley_values


# =============================================================================
# BATCH VALIDATION
# =============================================================================

class GradientValidator:
    """
    Stateful gradient validator for batch processing.

    Tracks validation statistics and can enforce consistency
    across multiple batches.

    Example:
        >>> validator = GradientValidator(reference_shape=(10000,))
        >>> for batch in gradient_batches:
        ...     validated = validator.validate_batch(batch)
        >>> print(validator.stats)
    """

    def __init__(
        self,
        reference_shape: Optional[Tuple[int, ...]] = None,
        min_nodes: int = 1,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ):
        self.reference_shape = reference_shape
        self.min_nodes = min_nodes
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

        # Statistics
        self.batches_validated = 0
        self.gradients_validated = 0
        self.gradients_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}

    def validate_batch(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Validate a batch of gradients.

        Args:
            gradients: Dict of node_id -> gradient

        Returns:
            Dict of validated gradients (invalid ones removed)
        """
        validated = {}

        for node_id, gradient in gradients.items():
            try:
                # Validate individual gradient
                validate_gradient(
                    gradient,
                    node_id=node_id,
                    allow_nan=self.allow_nan,
                    allow_inf=self.allow_inf,
                )

                # Check shape consistency
                if self.reference_shape is not None:
                    flat_shape = (gradient.size,)
                    if flat_shape != self.reference_shape:
                        raise GradientShapeMismatchError(
                            expected_shape=self.reference_shape,
                            actual_shape=flat_shape,
                            node_id=node_id,
                        )

                validated[node_id] = gradient
                self.gradients_validated += 1

            except Exception as e:
                self.gradients_rejected += 1
                reason = type(e).__name__
                self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
                logger.warning(f"Rejected gradient from {node_id}: {e}")

        # Set reference shape from first valid batch
        if self.reference_shape is None and validated:
            first_grad = next(iter(validated.values()))
            self.reference_shape = (first_grad.size,)

        # Check minimum nodes
        if len(validated) < self.min_nodes:
            raise InsufficientGradientsError(
                received=len(validated),
                minimum=self.min_nodes,
            )

        self.batches_validated += 1
        return validated

    @property
    def stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.gradients_validated + self.gradients_rejected
        return {
            "batches_validated": self.batches_validated,
            "gradients_validated": self.gradients_validated,
            "gradients_rejected": self.gradients_rejected,
            "rejection_rate": self.gradients_rejected / total if total > 0 else 0.0,
            "rejection_reasons": self.rejection_reasons,
            "reference_shape": self.reference_shape,
        }
