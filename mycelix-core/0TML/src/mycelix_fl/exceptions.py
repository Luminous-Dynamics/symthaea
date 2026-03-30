# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Custom Exceptions

Structured exception hierarchy for clear error handling and debugging.
All exceptions include context information to aid troubleshooting.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from typing import Any, Dict, List, Optional, Set


class MycelixFLError(Exception):
    """
    Base exception for all MycelixFL errors.

    All custom exceptions inherit from this, allowing:
    - Catching all MycelixFL errors with `except MycelixFLError`
    - Distinguishing MycelixFL errors from system errors
    - Consistent error formatting and context

    Attributes:
        message: Human-readable error message
        context: Additional context for debugging
        suggestion: Optional suggested fix
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with context."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return " | ".join(parts)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(MycelixFLError):
    """
    Raised when configuration is invalid.

    Examples:
        - Invalid Byzantine threshold (>0.5)
        - Invalid hypervector dimension (not power of 2)
        - Conflicting configuration options
    """

    def __init__(
        self,
        message: str,
        param_name: Optional[str] = None,
        param_value: Any = None,
        valid_range: Optional[str] = None,
    ):
        context = {}
        if param_name:
            context["parameter"] = param_name
        if param_value is not None:
            context["value"] = param_value
        if valid_range:
            context["valid_range"] = valid_range

        suggestion = None
        if valid_range:
            suggestion = f"Use a value in range {valid_range}"

        super().__init__(message, context, suggestion)


class InvalidDimensionError(ConfigurationError):
    """Raised when hypervector dimension is invalid."""

    def __init__(self, dimension: int, nearest_valid: Optional[int] = None):
        suggestion = None
        if nearest_valid:
            suggestion = f"Use {nearest_valid} instead (nearest valid dimension)"

        super().__init__(
            f"Hypervector dimension must be a positive power of 2",
            param_name="dimension",
            param_value=dimension,
            valid_range="64, 128, 256, 512, 1024, 2048, 4096, ...",
        )
        if suggestion:
            self.suggestion = suggestion


# =============================================================================
# GRADIENT ERRORS
# =============================================================================

class GradientError(MycelixFLError):
    """Base class for gradient-related errors."""
    pass


class InvalidGradientError(GradientError):
    """
    Raised when a gradient is invalid.

    Examples:
        - Contains NaN or Inf values
        - Wrong shape/dtype
        - Empty gradient
    """

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        gradient_shape: Optional[tuple] = None,
        issue: Optional[str] = None,
    ):
        context = {}
        if node_id:
            context["node_id"] = node_id
        if gradient_shape:
            context["shape"] = gradient_shape
        if issue:
            context["issue"] = issue

        super().__init__(message, context)


class GradientShapeMismatchError(GradientError):
    """Raised when gradients have inconsistent shapes."""

    def __init__(
        self,
        expected_shape: tuple,
        actual_shape: tuple,
        node_id: Optional[str] = None,
    ):
        super().__init__(
            f"Gradient shape mismatch",
            context={
                "expected": expected_shape,
                "actual": actual_shape,
                "node_id": node_id or "unknown",
            },
            suggestion="Ensure all gradients have the same shape",
        )


class InsufficientGradientsError(GradientError):
    """Raised when not enough gradients are provided."""

    def __init__(self, received: int, minimum: int):
        super().__init__(
            f"Insufficient gradients for secure aggregation",
            context={"received": received, "minimum_required": minimum},
            suggestion=f"Wait for at least {minimum} nodes to submit gradients",
        )


# =============================================================================
# BYZANTINE DETECTION ERRORS
# =============================================================================

class ByzantineDetectionError(MycelixFLError):
    """Base class for Byzantine detection errors."""
    pass


class DetectionLayerError(ByzantineDetectionError):
    """Raised when a detection layer fails."""

    def __init__(
        self,
        layer_name: str,
        error: Exception,
        gradients_affected: Optional[int] = None,
    ):
        super().__init__(
            f"Detection layer '{layer_name}' failed",
            context={
                "layer": layer_name,
                "original_error": str(error),
                "gradients_affected": gradients_affected or "unknown",
            },
            suggestion="Check layer configuration and input gradients",
        )
        self.__cause__ = error


class TooManyByzantineError(ByzantineDetectionError):
    """Raised when Byzantine ratio exceeds threshold."""

    def __init__(
        self,
        detected_ratio: float,
        threshold: float,
        byzantine_nodes: Set[str],
    ):
        super().__init__(
            f"Byzantine ratio ({detected_ratio:.1%}) exceeds threshold ({threshold:.1%})",
            context={
                "detected_ratio": f"{detected_ratio:.2%}",
                "threshold": f"{threshold:.2%}",
                "byzantine_count": len(byzantine_nodes),
                "byzantine_nodes": list(byzantine_nodes)[:10],  # First 10 only
            },
            suggestion="Increase threshold or investigate Byzantine nodes",
        )
        self.byzantine_nodes = byzantine_nodes


class HealingFailedError(ByzantineDetectionError):
    """Raised when self-healing fails."""

    def __init__(
        self,
        node_id: str,
        healing_ratio: float,
        threshold: float,
    ):
        super().__init__(
            f"Cannot heal node '{node_id}' - deviation too large",
            context={
                "node_id": node_id,
                "healing_ratio": f"{healing_ratio:.3f}",
                "threshold": f"{threshold:.3f}",
            },
            suggestion="Consider excluding this node instead",
        )


# =============================================================================
# COMPRESSION ERRORS
# =============================================================================

class CompressionError(MycelixFLError):
    """Base class for compression errors."""
    pass


class EncodingError(CompressionError):
    """Raised when hypervector encoding fails."""

    def __init__(
        self,
        message: str,
        gradient_shape: Optional[tuple] = None,
        target_dimension: Optional[int] = None,
    ):
        context = {}
        if gradient_shape:
            context["gradient_shape"] = gradient_shape
        if target_dimension:
            context["target_dimension"] = target_dimension

        super().__init__(message, context)


class DecodingError(CompressionError):
    """Raised when hypervector decoding fails."""

    def __init__(
        self,
        message: str,
        hypervector_shape: Optional[tuple] = None,
    ):
        super().__init__(
            message,
            context={"hypervector_shape": hypervector_shape} if hypervector_shape else None,
        )


class CompressionRatioError(CompressionError):
    """Raised when compression ratio is below target."""

    def __init__(
        self,
        achieved_ratio: float,
        target_ratio: float,
    ):
        super().__init__(
            f"Compression ratio {achieved_ratio:.1f}x below target {target_ratio:.1f}x",
            context={
                "achieved": f"{achieved_ratio:.1f}x",
                "target": f"{target_ratio:.1f}x",
            },
            suggestion="Increase hypervector dimension or adjust encoding parameters",
        )


# =============================================================================
# ML BRIDGE ERRORS
# =============================================================================

class MLBridgeError(MycelixFLError):
    """Base class for ML framework bridge errors."""
    pass


class UnsupportedFrameworkError(MLBridgeError):
    """Raised when ML framework is not supported."""

    def __init__(self, framework: str, supported: List[str]):
        super().__init__(
            f"ML framework '{framework}' is not supported",
            context={"framework": framework, "supported": supported},
            suggestion=f"Use one of: {', '.join(supported)}",
        )


class ModelExtractionError(MLBridgeError):
    """Raised when gradient extraction from model fails."""

    def __init__(
        self,
        model_type: str,
        error: Exception,
    ):
        super().__init__(
            f"Failed to extract gradients from {model_type} model",
            context={
                "model_type": model_type,
                "original_error": str(error),
            },
            suggestion="Ensure model has been trained with backward pass",
        )
        self.__cause__ = error


# =============================================================================
# RUST BRIDGE ERRORS
# =============================================================================

class RustBridgeError(MycelixFLError):
    """Base class for Rust backend errors."""
    pass


class RustModuleNotFoundError(RustBridgeError):
    """Raised when Rust module is not available."""

    def __init__(self, module_name: str):
        super().__init__(
            f"Rust module '{module_name}' not found",
            context={"module": module_name},
            suggestion="Install gen7-zkstark Rust extension or disable Rust backend",
        )


class RustExecutionError(RustBridgeError):
    """Raised when Rust execution fails."""

    def __init__(
        self,
        operation: str,
        error: Exception,
    ):
        super().__init__(
            f"Rust operation '{operation}' failed",
            context={
                "operation": operation,
                "original_error": str(error),
            },
            suggestion="Falling back to Python implementation",
        )
        self.__cause__ = error


# =============================================================================
# TIMEOUT ERRORS
# =============================================================================

class TimeoutError(MycelixFLError):
    """Raised when an operation times out."""

    def __init__(
        self,
        operation: str,
        timeout_ms: float,
        elapsed_ms: float,
    ):
        super().__init__(
            f"Operation '{operation}' timed out",
            context={
                "operation": operation,
                "timeout_ms": f"{timeout_ms:.1f}",
                "elapsed_ms": f"{elapsed_ms:.1f}",
            },
            suggestion="Increase timeout or optimize operation",
        )


class RoundTimeoutError(TimeoutError):
    """Raised when FL round times out."""

    def __init__(
        self,
        round_num: int,
        timeout_ms: float,
        waiting_for: Optional[List[str]] = None,
    ):
        super().__init__(
            operation=f"FL round {round_num}",
            timeout_ms=timeout_ms,
            elapsed_ms=timeout_ms,  # At timeout
        )
        if waiting_for:
            self.context["waiting_for_nodes"] = waiting_for[:10]


# =============================================================================
# SERIALIZATION ERRORS
# =============================================================================

class SerializationError(MycelixFLError):
    """Raised when serialization/deserialization fails."""

    def __init__(
        self,
        operation: str,  # "serialize" or "deserialize"
        data_type: str,
        error: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to {operation} {data_type}",
            context={
                "operation": operation,
                "data_type": data_type,
                "original_error": str(error) if error else None,
            },
        )
        if error:
            self.__cause__ = error
