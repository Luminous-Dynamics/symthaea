#!/usr/bin/env python3
"""Verify behavior-neutral VART-002 provenance shadow characterization receipts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "symthaea.vart-002.shadow-characterization.v1"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODES = ("grounded_history", "grounded_or_imported", "counterfactual_only")


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def nonnegative_int(value, label: str) -> int:
    require(isinstance(value, int) and value >= 0, f"{label} must be a non-negative integer")
    return value


def verify(packet: dict) -> dict:
    require(packet.get("schema") == SCHEMA, "schema mismatch")
    require(packet.get("status") == "development_measurement_only", "invalid status")
    require(packet.get("benchmark_domain") == "DEVART", "shadow characterization must use DEVART")
    require(bool(str(packet.get("campaign_id", "")).strip()), "campaign_id required")
    require(packet.get("hidden_vart_feedback_used") is False, "hidden VART feedback must not be used")
    require(packet.get("used_to_optimize_hidden_vart") is False, "shadow data must not optimize hidden VART")
    require(packet.get("enforcement_enabled_during_measurement") is False, "measurement must remain shadow-only")
    require(packet.get("claim_authorized") is False, "claim authority must remain false")
    require(packet.get("confirmatory_execution_authorized") is False, "confirmatory execution must remain false")

    for label in ("subject_source", "instrument_source"):
        source = packet.get(label)
        require(isinstance(source, dict), f"{label} missing")
        require(SHA1.fullmatch(str(source.get("head", ""))) is not None, f"{label}.head must be SHA-1")
        require(SHA1.fullmatch(str(source.get("tree", ""))) is not None, f"{label}.tree must be SHA-1")

    neutrality = packet.get("paired_behavior_neutrality")
    require(isinstance(neutrality, dict), "paired_behavior_neutrality missing")
    disabled = str(neutrality.get("shadow_disabled_decision_trace_sha256", ""))
    enabled = str(neutrality.get("shadow_enabled_decision_trace_sha256", ""))
    require(SHA256.fullmatch(disabled) is not None, "shadow-disabled trace hash must be SHA-256")
    require(SHA256.fullmatch(enabled) is not None, "shadow-enabled trace hash must be SHA-256")
    require(disabled == enabled, "shadow-on/off decision traces differ")
    for field in ("same_worlds", "same_seeds", "same_subject_source", "same_compute_budget", "same_actions_and_receipts"):
        require(neutrality.get(field) is True, f"paired_behavior_neutrality.{field} must be true")

    coverage = packet.get("coverage")
    require(isinstance(coverage, dict), "coverage missing")
    audit_cycles = nonnegative_int(coverage.get("audit_cycles"), "coverage.audit_cycles")
    queries_observed = nonnegative_int(coverage.get("queries_observed"), "coverage.queries_observed")
    require(audit_cycles > 0, "audit_cycles must be > 0")
    require(queries_observed > 0, "queries_observed must be > 0")

    mode_results = {}
    for mode in MODES:
        values = packet.get(mode)
        require(isinstance(values, dict), f"{mode} missing")
        q = nonnegative_int(values.get("queries_observed"), f"{mode}.queries_observed")
        require(q == queries_observed, f"{mode}.queries_observed must equal coverage.queries_observed")
        raw = nonnegative_int(values.get("raw_returned_total"), f"{mode}.raw_returned_total")
        would = nonnegative_int(values.get("would_return_total"), f"{mode}.would_return_total")
        changed = nonnegative_int(values.get("selection_changed_queries"), f"{mode}.selection_changed_queries")
        unknown = nonnegative_int(values.get("excluded_unknown_total"), f"{mode}.excluded_unknown_total")
        taint = nonnegative_int(values.get("excluded_taint_total"), f"{mode}.excluded_taint_total")
        domain = nonnegative_int(values.get("excluded_domain_total"), f"{mode}.excluded_domain_total")
        require(changed <= q, f"{mode}.selection_changed_queries cannot exceed queries")
        require(would <= raw, f"{mode}.would_return_total cannot exceed raw_returned_total")
        mode_results[mode] = {
            "queries": q,
            "raw_returned_total": raw,
            "would_return_total": would,
            "selection_changed_queries": changed,
            "excluded_total": unknown + taint + domain,
        }

    return {
        "verdict": "SHADOW_CHARACTERIZATION_PASS",
        "campaign_id": packet["campaign_id"],
        "audit_cycles": audit_cycles,
        "queries_observed": queries_observed,
        "decision_trace_sha256": disabled,
        "modes": mode_results,
        "behavior_neutral": True,
        "hidden_vart_feedback_used": False,
        "claim_authorized": False,
        "confirmatory_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        packet = json.loads(args.packet.read_text())
        result = verify(packet)
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        if args.json:
            print(json.dumps({"verdict": "SHADOW_CHARACTERIZATION_REJECT", "error": str(exc)}, sort_keys=True))
        else:
            print(f"SHADOW_CHARACTERIZATION_REJECT: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
