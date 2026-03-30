# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Universal Federated Learning System

A unified, production-ready FL framework combining:
- HyperFeel v2 for 2000x gradient compression
- Multi-layer Byzantine detection (99%+ @ 45% adversarial)
- Real Φ measurement (integrated information theory)
- Shapley-weighted fair aggregation
- Self-healing for recoverable Byzantine errors
- ML framework bridges (PyTorch, TensorFlow, NumPy)
- Optional Rust acceleration (100-1000x speedup) via zerotrustml-core

Example:
    >>> from mycelix_fl import MycelixFL, FLConfig
    >>> fl = MycelixFL(config=FLConfig(num_rounds=10))
    >>> result = fl.execute_round(gradients, round_num=1)
    >>> print(f"Byzantine detected: {len(result.byzantine_nodes)}")
    >>> print(f"Healed: {len(result.healed_nodes)}")
    >>> print(f"Φ gain: {result.phi_after - result.phi_before:.6f}")

    # Check for Rust acceleration
    >>> from mycelix_fl import has_rust_backend
    >>> print(f"Rust acceleration: {has_rust_backend()}")

    # Enable logging for debugging
    >>> from mycelix_fl import setup_logging
    >>> setup_logging(level="DEBUG")

Author: Luminous Dynamics
Version: 1.1.0
Date: December 30, 2025
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Union

__version__ = "1.3.0"
__author__ = "Luminous Dynamics"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
#
# Following Python library best practices:
# 1. Add NullHandler to prevent "No handler found" warnings
# 2. Provide setup_logging() for users who want output
# 3. Never configure root logger automatically (let apps decide)
#

# Add NullHandler to library root logger (standard library practice)
logging.getLogger(__name__).addHandler(logging.NullHandler())


def setup_logging(
    level: Union[int, str] = logging.INFO,
    format: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """
    Configure logging for the mycelix_fl package.

    This function sets up logging output for all mycelix_fl modules.
    By default, libraries don't produce output - call this to enable it.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Can be int or string name.
        format: Custom format string. Default shows time, level, module, message.
        handler: Custom handler. Default is StreamHandler to stderr.

    Returns:
        The configured logger for mycelix_fl package.

    Example:
        >>> import mycelix_fl
        >>> mycelix_fl.setup_logging(level="DEBUG")  # Verbose output
        >>> mycelix_fl.setup_logging(level="WARNING")  # Only warnings+

        # Custom format
        >>> mycelix_fl.setup_logging(
        ...     level="INFO",
        ...     format="%(levelname)s: %(message)s"
        ... )

        # Log to file
        >>> import logging
        >>> file_handler = logging.FileHandler("mycelix.log")
        >>> mycelix_fl.setup_logging(handler=file_handler)
    """
    logger = logging.getLogger(__name__)

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for h in logger.handlers[:]:
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)

    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)

    # Set format
    if format is None:
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)

    # Also configure child loggers to propagate
    logger.propagate = False

    logger.debug(f"mycelix_fl logging configured at level {logging.getLevelName(level)}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for mycelix_fl or a submodule.

    Args:
        name: Submodule name (e.g., "detection", "hyperfeel").
              If None, returns the root mycelix_fl logger.

    Returns:
        Logger instance.

    Example:
        >>> from mycelix_fl import get_logger
        >>> logger = get_logger("detection")
        >>> logger.info("Detection module initialized")
    """
    if name is None:
        return logging.getLogger(__name__)
    return logging.getLogger(f"{__name__}.{name}")

# Core exports
from mycelix_fl.core.unified_fl import MycelixFL, FLConfig, RoundResult
from mycelix_fl.core.phi_measurement import (
    HypervectorPhiMeasurer,
    PhiMetrics,
    measure_phi,
)

# Detection exports
from mycelix_fl.detection.multi_layer_stack import (
    MultiLayerByzantineDetector,
    DetectionResult,
    DetectionLayer,
)
from mycelix_fl.detection.shapley_detector import ShapleyByzantineDetector
from mycelix_fl.detection.self_healing import SelfHealingDetector

# HyperFeel exports
from mycelix_fl.hyperfeel.encoder_v2 import (
    HyperFeelEncoderV2,
    HyperGradient,
    EncodingConfig,
)

# ML Bridge exports
from mycelix_fl.ml.bridge import (
    MLBridge,
    PyTorchBridge,
    TensorFlowBridge,
    NumPyBridge,
    create_bridge,
)

# Configuration exports
from mycelix_fl.config import (
    RuntimeConfig,
    runtime_config,
    HV_DIMENSION_DEFAULT,
    HV_DIMENSION_HIGH,
    HV_DIMENSION_COMPACT,
    BYZANTINE_THRESHOLD_MAX,
    BYZANTINE_THRESHOLD_DEFAULT,
    DETECTION_LATENCY_TARGET_MS,
    EPSILON,
    validate_dimension,
    get_nearest_valid_dimension,
)

# Exception exports
from mycelix_fl.exceptions import (
    MycelixFLError,
    ConfigurationError,
    InvalidDimensionError,
    GradientError,
    InvalidGradientError,
    GradientShapeMismatchError,
    InsufficientGradientsError,
    ByzantineDetectionError,
    DetectionLayerError,
    TooManyByzantineError,
    HealingFailedError,
    CompressionError,
    EncodingError,
    DecodingError,
    MLBridgeError,
    UnsupportedFrameworkError,
    RustBridgeError,
    RustModuleNotFoundError,
    SerializationError,
)

# Validation exports
from mycelix_fl.validation import (
    validate_gradient,
    validate_gradients,
    sanitize_gradient,
    compute_gradient_stats,
    validate_byzantine_threshold,
    validate_hv_dimension,
    validate_positive_int,
    validate_positive_float,
    validate_probability,
    validate_node_id,
    validate_node_ids,
    validate_shapley_values,
    GradientValidator,
)

# Observability exports
from mycelix_fl.observability import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    metrics,
    StructuredLogger,
    timed,
    timed_block,
    PerformanceTracker,
    PerformanceSnapshot,
)

# Attack module exports (for testing Byzantine resilience)
from mycelix_fl.attacks import (
    AttackType,
    AttackConfig,
    ByzantineAttack,
    GradientScalingAttack,
    SignFlipAttack,
    GaussianNoiseAttack,
    LittleIsEnoughAttack,
    IPMAttack,
    FreeRiderAttack,
    BackdoorAttack,
    AdaptiveAttack,
    CartelAttack,
    create_attack,
    AttackOrchestrator,
    create_gradient_scaling_scenario,
    create_adaptive_attack_scenario,
    create_cartel_scenario,
    create_mixed_attack_scenario,
)

# Privacy module exports (v1.2.0)
from mycelix_fl.privacy import (
    DifferentialPrivacy,
    DPConfig,
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
    add_noise,
    compute_privacy_budget,
    PrivacyAccountant,
)

# Benchmark exports (v1.2.0)
from mycelix_fl.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    run_standard_benchmarks,
    run_scalability_test,
    run_byzantine_detection_benchmark,
    compare_backends,
)

# Distributed/Async exports (v1.2.0)
from mycelix_fl.distributed import (
    AsyncMycelixFL,
    AsyncFLConfig,
    NodeConnection,
    GradientMessage,
    AggregationMessage,
    FLProtocol,
    FLCoordinator,
    CoordinatorConfig,
    RoundState,
)

# CLI exports (v1.2.0)
from mycelix_fl.cli import main as cli_main, app as cli_app

# Proof exports (v1.3.0 - zkSTARK gradient proofs)
from mycelix_fl.proofs import (
    ProofOfGradientQuality,
    GradientProofConfig,
    GradientStatistics,
    VerificationResult as ProofVerificationResult,
    BatchProofVerifier,
    compute_gradient_commitment,
    has_rust_proofs,
)

# Rust Bridge exports (optional, for 100-1000x speedup)
try:
    from mycelix_fl.rust_bridge import (
        RUST_AVAILABLE,
        get_encoder,
        get_detector,
        get_phi_measurer,
        get_shapley_computer,
        detect_byzantine as rust_detect_byzantine,
    )
except ImportError:
    RUST_AVAILABLE = False
    get_encoder = None
    get_detector = None
    get_phi_measurer = None
    get_shapley_computer = None
    rust_detect_byzantine = None


def has_rust_backend() -> bool:
    """Check if the Rust acceleration backend (zerotrustml-core) is available."""
    return RUST_AVAILABLE


def get_backend_info() -> dict:
    """Get information about the available backends."""
    info = {
        "version": __version__,
        "rust_available": RUST_AVAILABLE,
        "python_version": "3.11+",
        "features": {
            "hyperfeel_compression": True,
            "multi_layer_detection": True,
            "phi_measurement": True,
            "shapley_aggregation": True,
            "self_healing": True,
        },
    }
    if RUST_AVAILABLE:
        from mycelix_fl.rust_bridge import get_version, get_dimension
        info["rust_version"] = get_version()
        info["hv_dimension"] = get_dimension()
        info["rust_paradigm_shifts"] = 36
    return info


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Logging
    "setup_logging",
    "get_logger",
    # Core
    "MycelixFL",
    "FLConfig",
    "RoundResult",
    # Phi measurement
    "HypervectorPhiMeasurer",
    "PhiMetrics",
    "measure_phi",
    # Detection
    "MultiLayerByzantineDetector",
    "DetectionResult",
    "DetectionLayer",
    "ShapleyByzantineDetector",
    "SelfHealingDetector",
    # HyperFeel
    "HyperFeelEncoderV2",
    "HyperGradient",
    "EncodingConfig",
    # ML Bridge
    "MLBridge",
    "PyTorchBridge",
    "TensorFlowBridge",
    "NumPyBridge",
    "create_bridge",
    # Configuration (v1.2.0)
    "RuntimeConfig",
    "runtime_config",
    "HV_DIMENSION_DEFAULT",
    "HV_DIMENSION_HIGH",
    "HV_DIMENSION_COMPACT",
    "BYZANTINE_THRESHOLD_MAX",
    "BYZANTINE_THRESHOLD_DEFAULT",
    "DETECTION_LATENCY_TARGET_MS",
    "EPSILON",
    "validate_dimension",
    "get_nearest_valid_dimension",
    # Exceptions (v1.2.0)
    "MycelixFLError",
    "ConfigurationError",
    "InvalidDimensionError",
    "GradientError",
    "InvalidGradientError",
    "GradientShapeMismatchError",
    "InsufficientGradientsError",
    "ByzantineDetectionError",
    "DetectionLayerError",
    "TooManyByzantineError",
    "HealingFailedError",
    "CompressionError",
    "EncodingError",
    "DecodingError",
    "MLBridgeError",
    "UnsupportedFrameworkError",
    "RustBridgeError",
    "RustModuleNotFoundError",
    "SerializationError",
    # Validation (v1.2.0)
    "validate_gradient",
    "validate_gradients",
    "sanitize_gradient",
    "compute_gradient_stats",
    "validate_byzantine_threshold",
    "validate_hv_dimension",
    "validate_positive_int",
    "validate_positive_float",
    "validate_probability",
    "validate_node_id",
    "validate_node_ids",
    "validate_shapley_values",
    "GradientValidator",
    # Observability (v1.2.0)
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "metrics",
    "StructuredLogger",
    "timed",
    "timed_block",
    "PerformanceTracker",
    "PerformanceSnapshot",
    # Attacks (v1.2.0 - for testing)
    "AttackType",
    "AttackConfig",
    "ByzantineAttack",
    "GradientScalingAttack",
    "SignFlipAttack",
    "GaussianNoiseAttack",
    "LittleIsEnoughAttack",
    "IPMAttack",
    "FreeRiderAttack",
    "BackdoorAttack",
    "AdaptiveAttack",
    "CartelAttack",
    "create_attack",
    "AttackOrchestrator",
    "create_gradient_scaling_scenario",
    "create_adaptive_attack_scenario",
    "create_cartel_scenario",
    "create_mixed_attack_scenario",
    # Privacy (v1.2.0)
    "DifferentialPrivacy",
    "DPConfig",
    "GaussianMechanism",
    "LaplaceMechanism",
    "clip_gradients",
    "add_noise",
    "compute_privacy_budget",
    "PrivacyAccountant",
    # Benchmarks (v1.2.0)
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_standard_benchmarks",
    "run_scalability_test",
    "run_byzantine_detection_benchmark",
    "compare_backends",
    # Distributed/Async (v1.2.0)
    "AsyncMycelixFL",
    "AsyncFLConfig",
    "NodeConnection",
    "GradientMessage",
    "AggregationMessage",
    "FLProtocol",
    "FLCoordinator",
    "CoordinatorConfig",
    "RoundState",
    # CLI (v1.2.0)
    "cli_main",
    "cli_app",
    # Proofs (v1.3.0)
    "ProofOfGradientQuality",
    "GradientProofConfig",
    "GradientStatistics",
    "ProofVerificationResult",
    "BatchProofVerifier",
    "compute_gradient_commitment",
    "has_rust_proofs",
    # Rust Bridge
    "RUST_AVAILABLE",
    "has_rust_backend",
    "get_backend_info",
    "get_encoder",
    "get_detector",
    "get_phi_measurer",
    "get_shapley_computer",
    "rust_detect_byzantine",
]
