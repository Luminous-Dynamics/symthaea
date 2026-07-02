# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Configuration Constants

Centralized configuration for all MycelixFL modules.
This ensures consistency across the system and makes tuning easier.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from dataclasses import dataclass
from typing import Optional
import os


# =============================================================================
# HYPERVECTOR DIMENSIONS
# =============================================================================

# Standard hypervector dimension (power of 2 for efficiency)
HV_DIMENSION_DEFAULT = 2048

# High-fidelity dimension for maximum accuracy
HV_DIMENSION_HIGH = 4096

# Compact dimension for bandwidth-constrained environments
HV_DIMENSION_COMPACT = 1024

# Ultra-high fidelity (16K) for research/benchmarking
HV_DIMENSION_ULTRA = 16384


# =============================================================================
# BYZANTINE DETECTION THRESHOLDS
# =============================================================================

# Maximum Byzantine tolerance (our breakthrough: 45% vs classical 33%)
BYZANTINE_THRESHOLD_MAX = 0.45

# Default Byzantine threshold for most deployments
BYZANTINE_THRESHOLD_DEFAULT = 0.45

# Conservative threshold for high-security environments
BYZANTINE_THRESHOLD_CONSERVATIVE = 0.33

# Shapley value threshold for flagging low contributors
SHAPLEY_THRESHOLD_DEFAULT = 0.1

# PoGQ quality threshold
POGQ_THRESHOLD_DEFAULT = 0.5

# Self-healing deviation thresholds
HEALING_MINOR_THRESHOLD = 0.3
HEALING_MODERATE_THRESHOLD = 0.7


# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

# Target latency for full detection pipeline (ms)
DETECTION_LATENCY_TARGET_MS = 100.0

# Maximum acceptable latency before warning
DETECTION_LATENCY_WARN_MS = 500.0

# Compression ratio target for HyperFeel
COMPRESSION_RATIO_TARGET = 2000.0

# Minimum compression ratio to be considered effective
COMPRESSION_RATIO_MIN = 100.0


# =============================================================================
# FEDERATED LEARNING DEFAULTS
# =============================================================================

# Default number of FL rounds
FL_ROUNDS_DEFAULT = 100

# Minimum nodes required for secure aggregation
FL_MIN_NODES_DEFAULT = 3

# Default learning rate
FL_LEARNING_RATE_DEFAULT = 0.01

# Maximum gradient norm for clipping
FL_MAX_GRADIENT_NORM = 10.0


# =============================================================================
# DETECTION LAYERS
# =============================================================================

# Number of layers required to confirm Byzantine
MIN_LAYERS_FOR_BYZANTINE = 2

# Confidence thresholds per layer
CONFIDENCE_ZKSTARK = 1.0
CONFIDENCE_POGQ = 0.95
CONFIDENCE_SHAPLEY = 0.98
CONFIDENCE_HYPERVECTOR = 0.95
CONFIDENCE_SELF_HEALING = 0.85


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Small epsilon for numerical stability
EPSILON = 1e-8

# Larger epsilon for float comparisons
FLOAT_COMPARE_EPSILON = 1e-6

# Maximum value before considered overflow
MAX_GRADIENT_VALUE = 1e10

# Minimum norm before considered zero
MIN_GRADIENT_NORM = 1e-10


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


# Environment-overridable settings
MYCELIX_DEBUG = get_env_bool('MYCELIX_DEBUG', False)
MYCELIX_HV_DIMENSION = get_env_int('MYCELIX_HV_DIMENSION', HV_DIMENSION_DEFAULT)
MYCELIX_BYZANTINE_THRESHOLD = get_env_float('MYCELIX_BYZANTINE_THRESHOLD', BYZANTINE_THRESHOLD_DEFAULT)
MYCELIX_USE_RUST = get_env_bool('MYCELIX_USE_RUST', True)


# =============================================================================
# RUNTIME CONFIGURATION CLASS
# =============================================================================

@dataclass
class RuntimeConfig:
    """
    Runtime configuration that can be modified during execution.

    This allows dynamic tuning without restarting the system.

    Example:
        >>> config = RuntimeConfig()
        >>> config.enable_debug_logging()
        >>> config.set_byzantine_threshold(0.4)
    """

    # Detection settings
    hv_dimension: int = HV_DIMENSION_DEFAULT
    byzantine_threshold: float = BYZANTINE_THRESHOLD_DEFAULT
    shapley_threshold: float = SHAPLEY_THRESHOLD_DEFAULT
    pogq_threshold: float = POGQ_THRESHOLD_DEFAULT

    # Performance settings
    use_rust_backend: bool = True
    enable_compression: bool = True
    enable_detection: bool = True
    enable_healing: bool = True

    # Logging settings
    debug_mode: bool = False
    log_gradients: bool = False
    log_shapley_values: bool = True

    # Monitoring settings
    enable_metrics: bool = True
    metrics_prefix: str = "mycelix_fl"

    def enable_debug_logging(self) -> None:
        """Enable verbose debug logging."""
        self.debug_mode = True
        self.log_gradients = True
        self.log_shapley_values = True

    def set_byzantine_threshold(self, threshold: float) -> None:
        """
        Set Byzantine threshold with validation.

        Args:
            threshold: New threshold (0.0 to 0.5)

        Raises:
            ValueError: If threshold out of range
        """
        if not 0.0 <= threshold <= 0.5:
            raise ValueError(f"Byzantine threshold must be 0.0-0.5, got {threshold}")
        self.byzantine_threshold = threshold

    def set_high_security(self) -> None:
        """Configure for high-security environments."""
        self.byzantine_threshold = BYZANTINE_THRESHOLD_CONSERVATIVE
        self.enable_detection = True
        self.enable_healing = True
        self.use_rust_backend = True

    def set_high_performance(self) -> None:
        """Configure for maximum performance (lower security)."""
        self.hv_dimension = HV_DIMENSION_COMPACT
        self.enable_healing = False
        self.use_rust_backend = True


# Global runtime config (can be modified at runtime)
runtime_config = RuntimeConfig(
    hv_dimension=MYCELIX_HV_DIMENSION,
    byzantine_threshold=MYCELIX_BYZANTINE_THRESHOLD,
    use_rust_backend=MYCELIX_USE_RUST,
    debug_mode=MYCELIX_DEBUG,
)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_dimension(dimension: int) -> bool:
    """Check if dimension is valid (positive power of 2)."""
    return dimension > 0 and (dimension & (dimension - 1)) == 0


def get_nearest_valid_dimension(dimension: int) -> int:
    """Get nearest valid power-of-2 dimension."""
    if dimension <= 0:
        return HV_DIMENSION_DEFAULT

    # Find nearest power of 2
    power = 1
    while power < dimension:
        power *= 2

    # Return closer of power and power/2
    if power - dimension > dimension - power // 2 and power // 2 > 0:
        return power // 2
    return power
