"""
Experimental bridge to the Rust trust stack (HyperFeel + Symthaea + MATL).

This module provides a thin, optional wrapper around the PyO3-powered
`holochain_credits_bridge` library so that 0TML experiments can:

1. Encode gradients with HyperFeel
2. Assess updates with Symthaea-HLB
3. Evaluate node trust with the Rust MatlEngine

All functions degrade gracefully when the Rust bridge is not available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from zerotrustml.logging import get_logger

logger = get_logger(__name__)

try:
    # Provided by Mycelix-Core/0TML/rust-bridge (built via maturin)
    import holochain_credits_bridge as rust_trust_bridge  # type: ignore[import]

    RUST_TRUST_BRIDGE_AVAILABLE = True
except ImportError:
    RUST_TRUST_BRIDGE_AVAILABLE = False
    logger.warning(
        "Rust trust bridge (holochain_credits_bridge) not available. "
        "Run `maturin develop --release` in 0TML/rust-bridge to enable "
        "HyperFeel + Symthaea + MATL integration."
    )


def encode_gradient_hyperfeel(
    gradient: Sequence[float],
    round_num: int,
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Encode a dense gradient via the Rust HyperFeel encoder.

    Returns a dict compatible with the Rust HyperGradient serialization
    or None if the Rust bridge is unavailable.
    """
    if not RUST_TRUST_BRIDGE_AVAILABLE:
        return None

    try:
        return rust_trust_bridge.encode_gradient_hyperfeel(
            list(float(x) for x in gradient),
            int(round_num),
            str(node_id),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("encode_gradient_hyperfeel failed: %s", exc, extra={"node_id": node_id})
        return None


def assess_update_with_symthaea(
    gradient: Sequence[float],
    round_num: int,
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Assess an update using Symthaea-HLB via the Rust bridge.

    The returned dict includes:
    - accuracy, loss
    - phi_before, phi_after, phi_gain
    - epistemic_confidence, is_anomalous, similarity, is_ambiguous, severity, causes
    - hypervector, original_size, compression_ratio
    """
    if not RUST_TRUST_BRIDGE_AVAILABLE:
        return None

    try:
        assessment = rust_trust_bridge.assess_update_symthaea(
            list(float(x) for x in gradient),
            int(round_num),
            str(node_id),
        )
        return dict(assessment)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("assess_update_symthaea failed: %s", exc, extra={"node_id": node_id})
        return None


def evaluate_node_matl(
    node_id: str,
    quality: float,
    consistency: float,
    entropy: float,
    reputation: float,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a node contribution using the Rust MatlEngine.

    Returns a dict of the form:
    {
        "node": { ... },
        "network": { ... },
    }
    or None if the bridge is unavailable.
    """
    if not RUST_TRUST_BRIDGE_AVAILABLE:
        return None

    try:
        result = rust_trust_bridge.evaluate_node_matl(
            str(node_id),
            float(quality),
            float(consistency),
            float(entropy),
            float(reputation),
        )
        return dict(result)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("evaluate_node_matl failed: %s", exc, extra={"node_id": node_id})
        return None


def log_symthaea_and_matl_metrics(
    *,
    node_id: str,
    round_num: int,
    gradient: Sequence[float],
    reputation: float,
    logger_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience helper for experiments: run Symthaea + MATL evaluation and
    emit a single structured log event with all relevant metrics.

    This function is intentionally side-effect-free beyond logging; it does
    not alter FL control flow.
    """
    if not RUST_TRUST_BRIDGE_AVAILABLE:
        return

    assessment = assess_update_with_symthaea(gradient, round_num, node_id)
    if not assessment:
        return

    # Derive a simple PoGQ triple from Symthaea metrics for demonstration.
    quality = float(assessment.get("epistemic_confidence", 0.0))
    consistency = max(float(assessment.get("phi_gain", 0.0)), 0.0)
    entropy = 0.5 if assessment.get("is_anomalous") else 0.1

    matl_result = evaluate_node_matl(
        node_id=node_id,
        quality=quality,
        consistency=consistency,
        entropy=entropy,
        reputation=reputation,
    )
    if not matl_result:
        return

    node_eval = matl_result.get("node", {})
    net_eval = matl_result.get("network", {})

    extra = {
        "node_id": node_id,
        "round": int(round_num),
        "symthaea_phi_before": assessment.get("phi_before"),
        "symthaea_phi_after": assessment.get("phi_after"),
        "symthaea_phi_gain": assessment.get("phi_gain"),
        "symthaea_confidence": assessment.get("epistemic_confidence"),
        "symthaea_is_anomalous": assessment.get("is_anomalous"),
        "symthaea_severity": assessment.get("severity"),
        "matl_composite_score": node_eval.get("composite_score"),
        "matl_node_threshold": node_eval.get("node_threshold"),
        "matl_is_node_anomalous": node_eval.get("is_node_anomalous"),
        "matl_is_in_suspicious_cluster": node_eval.get("is_in_suspicious_cluster"),
        "matl_estimated_byzantine_fraction": net_eval.get("estimated_byzantine_fraction"),
        "matl_adaptive_byzantine_threshold": net_eval.get("adaptive_byzantine_threshold"),
        "matl_network_status": net_eval.get("status"),
    }

    if logger_extra:
        extra.update(logger_extra)

    logger.info("Symthaea + MATL evaluation", extra=extra)

