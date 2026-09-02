#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("verify_vart_002_shadow_characterization.py")
spec = importlib.util.spec_from_file_location("shadow_gate", SCRIPT)
shadow_gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shadow_gate)

SHA1_A = "a" * 40
SHA1_B = "b" * 40
SHA1_C = "c" * 40
SHA1_D = "d" * 40
TRACE = "1" * 64


def mode() -> dict:
    return {
        "queries_observed": 100,
        "raw_returned_total": 250,
        "would_return_total": 180,
        "selection_changed_queries": 42,
        "excluded_unknown_total": 40,
        "excluded_taint_total": 20,
        "excluded_domain_total": 10,
    }


def valid_packet() -> dict:
    return {
        "schema": "symthaea.vart-002.shadow-characterization.v1",
        "status": "development_measurement_only",
        "campaign_id": "DEVART-SHADOW-001",
        "benchmark_domain": "DEVART",
        "subject_source": {"head": SHA1_A, "tree": SHA1_B},
        "instrument_source": {"head": SHA1_C, "tree": SHA1_D},
        "paired_behavior_neutrality": {
            "shadow_disabled_decision_trace_sha256": TRACE,
            "shadow_enabled_decision_trace_sha256": TRACE,
            "same_worlds": True,
            "same_seeds": True,
            "same_subject_source": True,
            "same_compute_budget": True,
            "same_actions_and_receipts": True,
        },
        "coverage": {"audit_cycles": 1000, "queries_observed": 100},
        "grounded_history": mode(),
        "grounded_or_imported": mode(),
        "counterfactual_only": mode(),
        "hidden_vart_feedback_used": False,
        "used_to_optimize_hidden_vart": False,
        "enforcement_enabled_during_measurement": False,
        "claim_authorized": False,
        "confirmatory_execution_authorized": False,
    }


def expect_reject(packet: dict, needle: str) -> None:
    try:
        shadow_gate.verify(packet)
    except shadow_gate.VerificationError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected rejection containing {needle!r}")


def main() -> None:
    result = shadow_gate.verify(valid_packet())
    assert result["verdict"] == "SHADOW_CHARACTERIZATION_PASS"
    assert result["behavior_neutral"] is True

    p = valid_packet()
    p["paired_behavior_neutrality"]["shadow_enabled_decision_trace_sha256"] = "2" * 64
    expect_reject(p, "decision traces differ")

    p = valid_packet()
    p["hidden_vart_feedback_used"] = True
    expect_reject(p, "hidden VART feedback")

    p = valid_packet()
    p["used_to_optimize_hidden_vart"] = True
    expect_reject(p, "must not optimize hidden VART")

    p = valid_packet()
    p["enforcement_enabled_during_measurement"] = True
    expect_reject(p, "shadow-only")

    p = valid_packet()
    p["paired_behavior_neutrality"]["same_actions_and_receipts"] = False
    expect_reject(p, "same_actions_and_receipts")

    p = valid_packet()
    p["grounded_history"]["queries_observed"] = 99
    expect_reject(p, "must equal coverage")

    p = valid_packet()
    p["grounded_history"]["selection_changed_queries"] = 101
    expect_reject(p, "cannot exceed queries")

    p = valid_packet()
    p["grounded_history"]["would_return_total"] = 251
    expect_reject(p, "cannot exceed raw_returned_total")

    print("PASS: shadow characterization acceptance + SH1-SH8 fail-closed cases")


if __name__ == "__main__":
    main()
