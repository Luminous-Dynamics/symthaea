#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("verify_vart_002_provenance_enforcement_preparation.py")
spec = importlib.util.spec_from_file_location("gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gate)

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40
SHA_D = "d" * 40


def valid_packet() -> dict:
    return {
        "schema": "symthaea.vart-002.provenance-enforcement-preparation.v1",
        "status": "candidate_not_authorized",
        "subject_source": {"head": SHA_A, "tree": SHA_B},
        "instrument_source": {"head": SHA_C, "tree": SHA_D},
        "shadow_characterization": {
            "fresh_devart_campaign_id": "DEVART-EPISTEMIC-SHADOW-001",
            "hidden_vart_feedback_used": False,
            "raw_decision_path_unchanged": True,
            "audit_cycles": 1000,
            "queries_observed": 350,
            "raw_vs_grounded_divergence_characterized": True,
        },
        "component_model": {
            "perception_subject_binding_complete": True,
            "cognition_subject_binding_complete": True,
            "composite_derivation_receipts_complete": True,
            "composite_never_implicitly_grounded": True,
            "legacy_unknown_preserved": True,
            "sidecars_transactionally_consistent": True,
            "replay_replacement_clears_sidecars": True,
            "paired_persistence_restore_qualified": True,
            "occurrence_truth_separate_from_content_truth": True,
        },
        "promotion_scope": {
            "fresh_devart_only": True,
            "hidden_vart_execution_allowed": False,
            "automatic_production_rollout_allowed": False,
        },
        "enforcement_preparation_eligible": True,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def expect_reject(packet: dict, needle: str) -> None:
    try:
        gate.verify(packet)
    except gate.GateError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected rejection containing {needle!r}")


def main() -> None:
    result = gate.verify(valid_packet())
    assert result["verdict"] == "PROVENANCE_ENFORCEMENT_PREPARATION_ELIGIBLE"
    assert result["hidden_vart_execution_allowed"] is False
    assert result["confirmatory_execution_authorized"] is False
    assert result["claim_authorized"] is False

    p = valid_packet()
    p["shadow_characterization"]["hidden_vart_feedback_used"] = True
    expect_reject(p, "hidden VART feedback")

    p = valid_packet()
    p["shadow_characterization"]["raw_decision_path_unchanged"] = False
    expect_reject(p, "behavior-neutral")

    p = valid_packet()
    p["shadow_characterization"]["queries_observed"] = 0
    expect_reject(p, "queries_observed")

    p = valid_packet()
    p["component_model"]["paired_persistence_restore_qualified"] = False
    expect_reject(p, "paired_persistence_restore_qualified")

    p = valid_packet()
    p["component_model"]["occurrence_truth_separate_from_content_truth"] = False
    expect_reject(p, "occurrence_truth_separate_from_content_truth")

    p = valid_packet()
    p["promotion_scope"]["hidden_vart_execution_allowed"] = True
    expect_reject(p, "hidden VART execution")

    p = valid_packet()
    p["confirmatory_execution_authorized"] = True
    expect_reject(p, "confirmatory execution")

    p = valid_packet()
    p["enforcement_preparation_eligible"] = False
    expect_reject(p, "declared eligibility")

    print("PASS: provenance enforcement preparation acceptance + E1-E8 fail-closed cases")


if __name__ == "__main__":
    main()
