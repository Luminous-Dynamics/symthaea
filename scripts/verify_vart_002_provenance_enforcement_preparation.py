#!/usr/bin/env python3
"""Fail-closed preparation gate for promoting VART-002 provenance from shadow to DEVART enforcement.

This verifier never authorizes hidden VART execution, confirmatory execution, production
rollout, or scientific claims. It answers only whether the implementation and fresh
DEVART shadow characterization are mature enough to begin a bounded enforcement test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "symthaea.vart-002.provenance-enforcement-preparation.v1"
SHA1 = re.compile(r"^[0-9a-f]{40}$")


class GateError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def verify(packet: dict) -> dict:
    require(packet.get("schema") == SCHEMA, "schema mismatch")
    require(packet.get("status") == "candidate_not_authorized", "status must remain candidate_not_authorized")
    require(packet.get("confirmatory_execution_authorized") is False, "confirmatory execution must remain unauthorized")
    require(packet.get("claim_authorized") is False, "scientific claims must remain unauthorized")

    for label in ("subject_source", "instrument_source"):
        source = packet.get(label)
        require(isinstance(source, dict), f"{label} missing")
        require(SHA1.fullmatch(str(source.get("head", ""))) is not None, f"{label}.head must be SHA-1")
        require(SHA1.fullmatch(str(source.get("tree", ""))) is not None, f"{label}.tree must be SHA-1")

    shadow = packet.get("shadow_characterization")
    require(isinstance(shadow, dict), "shadow_characterization missing")
    require(bool(str(shadow.get("fresh_devart_campaign_id", "")).strip()), "fresh DEVART campaign id required")
    require(shadow.get("hidden_vart_feedback_used") is False, "hidden VART feedback must not be used")
    require(shadow.get("raw_decision_path_unchanged") is True, "shadow mode must be behavior-neutral")
    require(isinstance(shadow.get("audit_cycles"), int) and shadow["audit_cycles"] > 0, "audit_cycles must be > 0")
    require(isinstance(shadow.get("queries_observed"), int) and shadow["queries_observed"] > 0, "queries_observed must be > 0")
    require(shadow.get("raw_vs_grounded_divergence_characterized") is True, "raw-vs-grounded divergence must be characterized")

    component = packet.get("component_model")
    require(isinstance(component, dict), "component_model missing")
    required_true = (
        "perception_subject_binding_complete",
        "cognition_subject_binding_complete",
        "composite_derivation_receipts_complete",
        "composite_never_implicitly_grounded",
        "legacy_unknown_preserved",
        "sidecars_transactionally_consistent",
        "replay_replacement_clears_sidecars",
        "paired_persistence_restore_qualified",
        "occurrence_truth_separate_from_content_truth",
    )
    for field in required_true:
        require(component.get(field) is True, f"component_model.{field} must be true")

    scope = packet.get("promotion_scope")
    require(isinstance(scope, dict), "promotion_scope missing")
    require(scope.get("fresh_devart_only") is True, "promotion scope must be fresh DEVART only")
    require(scope.get("hidden_vart_execution_allowed") is False, "hidden VART execution must remain blocked")
    require(scope.get("automatic_production_rollout_allowed") is False, "automatic production rollout must remain blocked")

    computed_eligible = True
    require(packet.get("enforcement_preparation_eligible") is computed_eligible, "declared eligibility must match computed eligibility")

    return {
        "schema": SCHEMA,
        "verdict": "PROVENANCE_ENFORCEMENT_PREPARATION_ELIGIBLE",
        "fresh_devart_campaign_id": shadow["fresh_devart_campaign_id"],
        "audit_cycles": shadow["audit_cycles"],
        "queries_observed": shadow["queries_observed"],
        "hidden_vart_execution_allowed": False,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        packet = json.loads(args.packet.read_text())
        result = verify(packet)
    except (OSError, json.JSONDecodeError, GateError) as exc:
        if args.json:
            print(json.dumps({"verdict": "PROVENANCE_ENFORCEMENT_PREPARATION_REJECT", "error": str(exc)}, sort_keys=True))
        else:
            print(f"PROVENANCE_ENFORCEMENT_PREPARATION_REJECT: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
