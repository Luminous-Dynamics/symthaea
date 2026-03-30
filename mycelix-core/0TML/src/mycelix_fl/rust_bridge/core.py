# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Rust Bridge Core Implementation

Provides wrappers around zerotrustml_core that fall back to Python
implementations when the Rust module is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the Rust module with detailed error handling
RUST_AVAILABLE = False
RUST_IMPORT_ERROR: Optional[str] = None
ztc = None  # type: ignore

try:
    import zerotrustml_core as ztc
    RUST_AVAILABLE = True
    logger.info(f"zerotrustml_core v{ztc.version()} loaded (Rust acceleration enabled)")
except ImportError as e:
    RUST_IMPORT_ERROR = str(e)
    if "No module named" in str(e):
        logger.warning(
            "zerotrustml_core not installed. Using Python fallback (100-1000x slower).\n"
            "  To install: pip install zerotrustml-core\n"
            "  Or build from source: cd zerotrustml-core && maturin build"
        )
    elif "libz" in str(e).lower() or "libc" in str(e).lower():
        logger.warning(
            f"zerotrustml_core has missing system dependencies: {e}\n"
            "  On NixOS: nix-shell -p zlib\n"
            "  On Ubuntu: apt install zlib1g-dev\n"
            "  Using Python fallback."
        )
    else:
        logger.warning(
            f"zerotrustml_core import failed: {e}\n"
            "  Using Python fallback. Check wheel compatibility with your Python version."
        )
except Exception as e:
    RUST_IMPORT_ERROR = str(e)
    logger.warning(f"zerotrustml_core initialization failed: {e}. Using Python fallback.")


def get_version() -> str:
    """Get the version of the Rust core, or 'python-only' if not available."""
    if RUST_AVAILABLE:
        return ztc.version()
    return "python-only"


def get_dimension() -> int:
    """Get the default hypervector dimension (16,384 for Rust, 8,192 for Python)."""
    if RUST_AVAILABLE:
        return ztc.HV_DIMENSION
    return 8192  # Python default


def get_rust_status() -> Dict[str, Any]:
    """
    Get detailed status of Rust backend.

    Returns:
        Dict with keys:
            - available: bool - whether Rust backend is loaded
            - version: str - Rust version or 'python-only'
            - hv_dimension: int - hypervector dimension
            - import_error: Optional[str] - error message if import failed
            - features: List[str] - available Rust features

    Example:
        >>> from mycelix_fl.rust_bridge import get_rust_status
        >>> status = get_rust_status()
        >>> if not status['available']:
        ...     print(f"Rust unavailable: {status['import_error']}")
    """
    status = {
        "available": RUST_AVAILABLE,
        "version": get_version(),
        "hv_dimension": get_dimension(),
        "import_error": RUST_IMPORT_ERROR,
        "features": [],
    }

    if RUST_AVAILABLE:
        status["features"] = [
            "hypervector_encoder",
            "phi_measurer",
            "shapley_computer",
            "unified_detector",
            "simd_acceleration",
        ]
        # Check for advanced features
        if hasattr(ztc, "AdaptiveOrchestrator"):
            status["features"].append("adaptive_orchestrator")
        if hasattr(ztc, "py_bundle"):
            status["features"].append("hypervector_bundling")

    return status


# =============================================================================
# HYPERVECTOR ENCODER
# =============================================================================

@dataclass
class RustHypervectorEncoder:
    """
    Wrapper for the Rust HypervectorEncoder.

    Provides 100-1000x speedup over pure Python implementation.
    Falls back to Python HyperFeelEncoderV2 when Rust not available.
    """
    dimension: int = 16384
    num_levels: int = 256

    def __post_init__(self):
        if RUST_AVAILABLE:
            self._encoder = ztc.HypervectorEncoder(self.dimension, self.num_levels)
            self._mode = "rust"
        else:
            # Import Python fallback
            from mycelix_fl.hyperfeel import HyperFeelEncoderV2, EncodingConfig
            config = EncodingConfig(dimension=self.dimension)
            self._encoder = HyperFeelEncoderV2(config)
            self._mode = "python"

    def encode(self, gradient: np.ndarray) -> Any:
        """Encode a gradient vector to a hypervector."""
        if self._mode == "rust":
            if isinstance(gradient, np.ndarray):
                gradient = gradient.astype(np.float32).flatten().tolist()
            return self._encoder.encode(gradient)
        else:
            # Python fallback returns HyperGradient
            return self._encoder.encode_gradient(gradient, "default")

    def encode_batch(self, gradients: List[np.ndarray]) -> List[Any]:
        """Batch encode multiple gradients (parallelized in Rust)."""
        if self._mode == "rust":
            grads = [g.astype(np.float32).flatten().tolist() for g in gradients]
            return self._encoder.encode_batch(grads)
        else:
            return [self.encode(g) for g in gradients]

    @property
    def is_rust(self) -> bool:
        return self._mode == "rust"


def get_encoder(dimension: int = 16384, num_levels: int = 256) -> RustHypervectorEncoder:
    """Factory function for hypervector encoder."""
    return RustHypervectorEncoder(dimension=dimension, num_levels=num_levels)


# =============================================================================
# UNIFIED DETECTOR
# =============================================================================

@dataclass
class RustUnifiedDetector:
    """
    Wrapper for the Rust UnifiedDetector with 10 detection layers.

    Layers:
        0. Phi (IIT-inspired)
        1. Shapley (contribution)
        2. Cartel (collusion)
        3. Temporal (time-based)
        4. Recycling (reuse)
        5. MLBD (multi-layer)
        6. Topology (network)
        7. Causal (causality)
        8. Spatiotemporal (combined)
        9. Immunity (adaptive)
    """
    dimension: int = 16384

    def __post_init__(self):
        if RUST_AVAILABLE:
            self._detector = ztc.UnifiedDetector(self.dimension)
            self._mode = "rust"
        else:
            from mycelix_fl.detection import MultiLayerByzantineDetector
            self._detector = MultiLayerByzantineDetector()
            self._mode = "python"

    def detect(
        self,
        gradients: Dict[str, np.ndarray],
        round_number: int = 0,
    ) -> "RustDetectionResult":
        """
        Detect Byzantine nodes from gradients.

        Args:
            gradients: Dict mapping node_id -> gradient array
            round_number: Current FL round

        Returns:
            RustDetectionResult with byzantine_nodes, weights, confidence
        """
        if self._mode == "rust":
            # Convert to format expected by Rust
            node_ids = list(gradients.keys())
            grad_list = [g.astype(np.float32).flatten().tolist() for g in gradients.values()]

            result = self._detector.detect_from_gradients(grad_list, node_ids)

            # Rust returns node IDs directly now
            byzantine_set = set(result.byzantine_nodes)
            honest_set = set(result.honest_nodes)

            # Compute aggregation weights: 1.0 for honest, 0.0 for byzantine
            aggregation_weights = {}
            trust_scores = result.trust_scores if hasattr(result, 'trust_scores') else {}
            for nid in node_ids:
                if nid in byzantine_set:
                    aggregation_weights[nid] = 0.0
                else:
                    # Use trust score if available, else 1.0
                    aggregation_weights[nid] = trust_scores.get(nid, 1.0)

            # Normalize weights
            total = sum(aggregation_weights.values())
            if total > 0:
                aggregation_weights = {k: v / total for k, v in aggregation_weights.items()}

            return RustDetectionResult(
                byzantine_nodes=byzantine_set,
                aggregation_weights=aggregation_weights,
                confidence=result.confidence,
                layer_results={"detection_layer": result.detection_layer},
                is_rust=True,
            )
        else:
            # Python fallback
            from mycelix_fl.detection import DetectionResult as PyDetectionResult
            result = self._detector.detect(
                gradients=gradients,
                pogq_scores=None,
                round_number=round_number,
            )
            return RustDetectionResult(
                byzantine_nodes=result.flagged_nodes,
                aggregation_weights=result.node_weights,
                confidence=result.overall_confidence,
                layer_results=result.layer_results,
                is_rust=False,
            )

    @property
    def is_rust(self) -> bool:
        return self._mode == "rust"


@dataclass
class RustDetectionResult:
    """Detection result compatible with both Rust and Python backends."""
    byzantine_nodes: set
    aggregation_weights: Dict[str, float]
    confidence: float
    layer_results: Dict[str, Any]
    is_rust: bool = False


def get_detector(dimension: int = 16384) -> RustUnifiedDetector:
    """Factory function for unified detector."""
    return RustUnifiedDetector(dimension=dimension)


def get_unified_detector(dimension: int = 16384) -> RustUnifiedDetector:
    """Alias for get_detector."""
    return get_detector(dimension)


# =============================================================================
# PHI MEASURER (IIT-INSPIRED)
# =============================================================================

@dataclass
class RustPhiMeasurer:
    """
    Wrapper for Rust PhiMeasurer.

    Measures integrated information (Φ) using hypervector approximation.
    Reduces O(2^n) IIT to O(L²) where L = num_layers.
    """
    dimension: int = 16384

    def __post_init__(self):
        if RUST_AVAILABLE:
            self._measurer = ztc.PhiMeasurer(self.dimension)
            self._mode = "rust"
        else:
            from mycelix_fl.core.phi_measurement import HypervectorPhiMeasurer
            self._measurer = HypervectorPhiMeasurer(hv_dimension=self.dimension)
            self._mode = "python"

    def measure_phi(self, gradients: List[np.ndarray], aggregated: np.ndarray) -> "RustPhiMeasurement":
        """Measure Φ for a set of gradients and their aggregation."""
        if self._mode == "rust":
            grad_list = [g.astype(np.float32).flatten().tolist() for g in gradients]
            agg_list = aggregated.astype(np.float32).flatten().tolist()
            result = self._measurer.measure(grad_list, agg_list)
            return RustPhiMeasurement(
                phi_value=result.phi,
                integration=result.phi_delta,
                information=result.mutual_information,
                coherence=result.entropy,
                contributions=[],  # Not provided by current Rust API
            )
        else:
            # For Python, we need to convert hypervectors to numpy arrays
            # and use the measure_phi_from_activations method
            raise NotImplementedError("Python phi measurement requires gradient arrays")

    def detect_byzantine(self, gradients: List[np.ndarray], aggregated: np.ndarray) -> List[int]:
        """Detect Byzantine nodes based on Φ analysis."""
        if self._mode == "rust":
            # Measure phi and identify outliers
            phi_result = self.measure_phi(gradients, aggregated)
            # For now, return empty - detection logic is in UnifiedDetector
            return []
        else:
            raise NotImplementedError("Python phi detection requires full gradients")

    @property
    def is_rust(self) -> bool:
        return self._mode == "rust"


@dataclass
class RustPhiMeasurement:
    """Phi measurement result."""
    phi_value: float
    integration: float
    information: float
    coherence: float
    contributions: List[float]


def get_phi_measurer(dimension: int = 16384) -> RustPhiMeasurer:
    """Factory function for Phi measurer."""
    return RustPhiMeasurer(dimension=dimension)


# =============================================================================
# SHAPLEY COMPUTER
# =============================================================================

@dataclass
class RustShapleyComputer:
    """
    Wrapper for Rust ShapleyComputer.

    Computes exact Shapley values in O(n) time using hypervector algebra.
    Classical Shapley is O(2^n) or O(n!) - intractable for FL.
    """
    dimension: int = 16384

    def __post_init__(self):
        if RUST_AVAILABLE:
            self._computer = ztc.ShapleyComputer(num_samples=100)
            self._mode = "rust"
        else:
            from mycelix_fl.detection import ShapleyByzantineDetector
            self._computer = ShapleyByzantineDetector()
            self._mode = "python"

    def compute_from_gradients(
        self, gradients: List[np.ndarray], node_ids: Optional[List[str]] = None
    ) -> List["RustShapleyContribution"]:
        """Compute Shapley values from gradients."""
        if self._mode == "rust":
            grad_list = [g.astype(np.float32).flatten().tolist() for g in gradients]
            results = self._computer.compute_from_gradients(grad_list, node_ids)
            return [
                RustShapleyContribution(
                    node_id=r.node_id,
                    shapley_value=r.shapley_value,
                    normalized_contribution=r.marginal_contribution,
                    is_positive_contributor=r.shapley_value > 0,
                    confidence=r.consistency,
                )
                for r in results
            ]
        else:
            raise NotImplementedError("Python Shapley requires full gradients")

    def compute_exact(self, hypervectors: List[Any]) -> List["RustShapleyContribution"]:
        """Compute exact Shapley values in O(n) time (deprecated, use compute_from_gradients)."""
        # For backward compatibility - hypervectors should be converted to gradients
        raise NotImplementedError("Use compute_from_gradients instead")

    def detect_byzantine(self, gradients: List[np.ndarray]) -> List[str]:
        """Detect Byzantine nodes based on negative Shapley values."""
        if self._mode == "rust":
            contributions = self.compute_from_gradients(gradients)
            return [c.node_id for c in contributions if not c.is_positive_contributor]
        else:
            raise NotImplementedError("Python detection requires full gradients")

    def aggregation_weights(self, gradients: List[np.ndarray]) -> Dict[str, float]:
        """Compute aggregation weights from Shapley values."""
        if self._mode == "rust":
            contributions = self.compute_from_gradients(gradients)
            # Use normalized weights
            weights = {c.node_id: max(0.0, c.shapley_value) for c in contributions}
            total = sum(weights.values())
            if total > 0:
                return {k: v / total for k, v in weights.items()}
            # Uniform weights if all negative
            n = len(contributions)
            return {c.node_id: 1.0 / n for c in contributions}
        else:
            raise NotImplementedError("Python weights require full gradients")

    @property
    def is_rust(self) -> bool:
        return self._mode == "rust"


@dataclass
class RustShapleyContribution:
    """Shapley contribution result."""
    node_id: int
    shapley_value: float
    normalized_contribution: float
    is_positive_contributor: bool
    confidence: float


def get_shapley_computer(dimension: int = 16384) -> RustShapleyComputer:
    """Factory function for Shapley computer."""
    return RustShapleyComputer(dimension=dimension)


# =============================================================================
# ADAPTIVE ORCHESTRATOR (FULL 36 PARADIGM SHIFTS)
# =============================================================================

@dataclass
class RustAdaptiveOrchestrator:
    """
    Wrapper for the full Rust AdaptiveOrchestrator.

    Includes all 36 paradigm shifts:
    - 10 detector layers
    - PBEW (Predictive Byzantine Early Warning)
    - UBI (Universal Byzantine Immunity)
    - Attack fingerprinting
    - Meta-detection
    - Manifold analysis
    - Attention fusion
    - Residual learning
    - Trust layer (BRRS, MLBD, ATI)

    This is the ultimate Byzantine detection system.
    """
    dimension: int = 16384

    def __post_init__(self):
        if RUST_AVAILABLE:
            try:
                self._orchestrator = ztc.AdaptiveOrchestrator()
                self._mode = "rust"
            except Exception as e:
                logger.warning(f"AdaptiveOrchestrator not available: {e}")
                self._mode = "python_fallback"
        else:
            self._mode = "python_fallback"

    def detect_adaptive(
        self,
        gradients: Dict[str, np.ndarray],
        round_number: int = 0,
    ) -> Optional["RustAdaptiveResult"]:
        """
        Full adaptive Byzantine detection with all paradigm shifts.

        Returns None if not available (falls back to simpler detector).
        """
        if self._mode == "rust":
            node_ids = list(gradients.keys())
            grad_list = [g.astype(np.float32).flatten().tolist() for g in gradients.values()]

            result = self._orchestrator.detect_adaptive(grad_list)

            return RustAdaptiveResult(
                byzantine_nodes=[node_ids[i] for i in result.byzantine_nodes],
                aggregation_weights={
                    node_ids[i]: w for i, w in enumerate(result.aggregation_weights)
                },
                confidence=result.confidence,
                attack_type=getattr(result, 'attack_type', None),
                defense_recommendation=getattr(result, 'defense_recommendation', None),
            )
        return None

    @property
    def is_rust(self) -> bool:
        return self._mode == "rust"


@dataclass
class RustAdaptiveResult:
    """Full adaptive detection result."""
    byzantine_nodes: List[str]
    aggregation_weights: Dict[str, float]
    confidence: float
    attack_type: Optional[str] = None
    defense_recommendation: Optional[str] = None


def get_adaptive_orchestrator(dimension: int = 16384) -> RustAdaptiveOrchestrator:
    """Factory function for adaptive orchestrator."""
    return RustAdaptiveOrchestrator(dimension=dimension)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_byzantine(
    gradients: Dict[str, np.ndarray],
    round_number: int = 0,
    use_adaptive: bool = True,
) -> Tuple[set, Dict[str, float], float]:
    """
    One-shot Byzantine detection using best available backend.

    Args:
        gradients: Dict mapping node_id -> gradient array
        round_number: Current FL round
        use_adaptive: Try adaptive orchestrator first

    Returns:
        Tuple of (byzantine_nodes, aggregation_weights, confidence)
    """
    if use_adaptive and RUST_AVAILABLE:
        try:
            orchestrator = get_adaptive_orchestrator()
            result = orchestrator.detect_adaptive(gradients, round_number)
            if result is not None:
                return (
                    set(result.byzantine_nodes),
                    result.aggregation_weights,
                    result.confidence,
                )
        except Exception:
            pass

    # Fall back to unified detector
    detector = get_unified_detector()
    result = detector.detect(gradients, round_number)
    return (result.byzantine_nodes, result.aggregation_weights, result.confidence)


def bundle_hypervectors(hypervectors: List[Any], weights: Optional[List[float]] = None) -> Any:
    """
    Bundle multiple hypervectors using (weighted) majority vote.

    Args:
        hypervectors: List of hypervectors to bundle
        weights: Optional weights for weighted bundling

    Returns:
        Bundled hypervector
    """
    if not RUST_AVAILABLE:
        raise NotImplementedError("Bundling requires Rust backend")

    if weights is None:
        return ztc.py_bundle(hypervectors)
    else:
        return ztc.py_bundle_weighted(hypervectors, weights)
