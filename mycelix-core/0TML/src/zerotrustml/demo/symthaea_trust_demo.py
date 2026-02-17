"""
Symthaea + MATL Trust Demo (Experimental)

This small demo shows how to:

1. Encode a gradient with the Rust HyperFeel encoder,
2. Assess it with Symthaea-HLB via the Rust bridge,
3. Evaluate trust with the Rust MatlEngine, and
4. Emit structured logs via the unified logging system.

Requirements:
    - The `holochain_credits_bridge` PyO3 module built and installed
      (see 0TML/rust-bridge docs, typically via `maturin develop --release`).
"""

from __future__ import annotations

import numpy as np

from zerotrustml.logging import configure_logging, get_logger, correlation_context
from zerotrustml.experimental.symthaea_trust_bridge import (
    RUST_TRUST_BRIDGE_AVAILABLE,
    encode_gradient_hyperfeel,
    assess_update_with_symthaea,
    evaluate_node_matl,
    log_symthaea_and_matl_metrics,
)


def main() -> None:
    configure_logging(level="INFO", json_format=True)
    logger = get_logger(__name__)

    if not RUST_TRUST_BRIDGE_AVAILABLE:
        logger.warning("Rust trust bridge unavailable; demo cannot run.")
        return

    node_id = "demo-node-1"
    round_num = 1
    reputation = 0.8

    # Synthetic gradient for demonstration purposes only.
    gradient = np.sin(np.linspace(0.0, 10.0, 2_000, dtype=np.float32))

    with correlation_context(node_id=node_id, round=round_num):
        logger.info("Starting Symthaea + MATL trust demo")

        # 1) Encode with HyperFeel
        hg = encode_gradient_hyperfeel(gradient, round_num, node_id)
        if hg is None:
            logger.error("Failed to encode gradient with HyperFeel")
            return

        logger.info(
            "Encoded gradient with HyperFeel",
            extra={
                "hyperfeel_original_size": hg.get("original_size"),
                "hyperfeel_compression_ratio": hg.get("compression_ratio"),
            },
        )

        # 2) Assess with Symthaea
        assessment = assess_update_with_symthaea(gradient, round_num, node_id)
        if assessment is None:
            logger.error("Symthaea assessment failed")
            return

        logger.info(
            "Symthaea assessment",
            extra={
                "phi_before": assessment.get("phi_before"),
                "phi_after": assessment.get("phi_after"),
                "phi_gain": assessment.get("phi_gain"),
                "epistemic_confidence": assessment.get("epistemic_confidence"),
                "symthaea_is_anomalous": assessment.get("is_anomalous"),
            },
        )

        # 3) Evaluate node via MATL
        quality = float(assessment.get("epistemic_confidence", 0.0))
        consistency = max(float(assessment.get("phi_gain", 0.0)), 0.0)
        entropy = 0.5 if assessment.get("is_anomalous") else 0.1

        matl_eval = evaluate_node_matl(
            node_id=node_id,
            quality=quality,
            consistency=consistency,
            entropy=entropy,
            reputation=reputation,
        )

        if matl_eval is None:
            logger.error("MATL evaluation failed")
            return

        node_eval = matl_eval.get("node", {})
        net_eval = matl_eval.get("network", {})

        logger.info(
            "MATL evaluation",
            extra={
                "matl_composite_score": node_eval.get("composite_score"),
                "matl_node_threshold": node_eval.get("node_threshold"),
                "matl_is_node_anomalous": node_eval.get("is_node_anomalous"),
                "matl_estimated_byzantine_fraction": net_eval.get("estimated_byzantine_fraction"),
                "matl_adaptive_byzantine_threshold": net_eval.get("adaptive_byzantine_threshold"),
                "matl_network_status": net_eval.get("status"),
            },
        )

        # 4) End-to-end helper (also logs a single combined record).
        log_symthaea_and_matl_metrics(
            node_id=node_id,
            round_num=round_num,
            gradient=gradient,
            reputation=reputation,
            logger_extra={"demo": True},
        )

        logger.info("Symthaea + MATL trust demo complete")


if __name__ == "__main__":
    main()

