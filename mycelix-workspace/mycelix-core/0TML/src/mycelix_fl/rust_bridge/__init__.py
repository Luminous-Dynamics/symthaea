# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Rust Bridge

This module provides seamless integration between the pure-Python mycelix_fl
implementation and the high-performance zerotrustml-core Rust backend.

When the Rust module is available:
- 100-1000x speedup on hypervector operations
- Native SIMD-optimized 16,384-dimensional binary vectors
- O(n) exact Shapley computation
- 36 paradigm shifts for Byzantine detection

When not available:
- Falls back gracefully to Python implementations
- Full functionality preserved

Usage:
    from mycelix_fl.rust_bridge import get_encoder, get_detector

    encoder = get_encoder()  # Returns Rust or Python encoder
    detector = get_detector()  # Returns Rust or Python detector
"""

from mycelix_fl.rust_bridge.core import (
    RUST_AVAILABLE,
    RUST_IMPORT_ERROR,
    get_version,
    get_dimension,
    get_rust_status,
    # Encoder
    get_encoder,
    RustHypervectorEncoder,
    # Detector
    get_detector,
    get_unified_detector,
    RustUnifiedDetector,
    # Phi
    get_phi_measurer,
    RustPhiMeasurer,
    # Shapley
    get_shapley_computer,
    RustShapleyComputer,
    # Orchestrator
    get_adaptive_orchestrator,
    RustAdaptiveOrchestrator,
    # Utilities
    detect_byzantine,
    bundle_hypervectors,
)

__all__ = [
    # Status & Diagnostics
    "RUST_AVAILABLE",
    "RUST_IMPORT_ERROR",
    "get_version",
    "get_dimension",
    "get_rust_status",
    # Encoder
    "get_encoder",
    "RustHypervectorEncoder",
    # Detector
    "get_detector",
    "get_unified_detector",
    "RustUnifiedDetector",
    # Phi
    "get_phi_measurer",
    "RustPhiMeasurer",
    # Shapley
    "get_shapley_computer",
    "RustShapleyComputer",
    # Orchestrator
    "get_adaptive_orchestrator",
    "RustAdaptiveOrchestrator",
    # Utilities
    "detect_byzantine",
    "bundle_hypervectors",
]
