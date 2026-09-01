#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import verify_vart_world_creative_001_post_pilot as v


def write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def base() -> dict:
    return {
        "schema": v.SCHEMA,
        "experiment_id": v.EXPERIMENT_ID,
        "status": "dispositioned",
        "pilot": {
            "pilot_receipt_sha256": "a" * 64,
            "pilot_evidence_closure_sha256": "b" * 64,
            "pilot_design_sha256": "c" * 64,
            "source_head": "d" * 40,
            "source_tree": "e" * 40,
            "audit_verdict": "PILOT_AUDIT_PASS",
            "paired_block_semantics": "PASS"
        },
        "inspection": {
            "inspection_started_utc": "2026-09-01T00:00:00+00:00",
            "inspection_completed_utc": "2026-09-01T00:01:00+00:00",
            "inspected_paths": ["_orchestrator/resolved_plan.json"],
            "outcome_magnitudes_viewed": False,
            "comparative_policy_rankings_viewed": False,
            "human_preference_values_viewed": False,
            "inspection_purpose": "instrumentation_and_protocol_only"
        },
        "defects": [],
        "resolution": {
            "all_defects_dispositioned": True,
            "unresolved_defect_count": 0,
            "pilot_rerun_required": False,
            "pilot_rerun_complete": False,
            "new_preregistration_lineage_required": False,
            "new_preregistration_lineage_created": False,
            "confirmatory_source_fetchable": True,
            "confirmatory_source_reproducible": True
        },
        "confirmatory_freeze_eligible": True,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "bounded_statement": "test"
    }


def expect_reject(obj: dict, code: str) -> None:
    with tempfile.TemporaryDirectory(prefix="vart-disposition-") as td:
        path = Path(td) / "d.json"
        write(path, obj)
        try:
            v.verify(path)
        except v.Reject as exc:
            assert exc.code == code, f"expected {code}, got {exc.code}: {exc.detail}"
            return
    raise AssertionError(f"expected {code}")


with tempfile.TemporaryDirectory(prefix="vart-disposition-pass-") as td:
    path = Path(td) / "d.json"
    write(path, base())
    assert v.verify(path)["verdict"] == "POST_PILOT_DISPOSITION_PASS"

x = base()
x["pilot"]["audit_verdict"] = "PILOT_AUDIT_REJECT"
expect_reject(x, "POST_PILOT_AUDIT_NOT_PASS")

x = base()
x["defects"] = [{"id": "D1", "class": "instrumentation_plumbing", "status": "unresolved"}]
x["resolution"]["unresolved_defect_count"] = 1
x["resolution"]["all_defects_dispositioned"] = False
x["confirmatory_freeze_eligible"] = False
with tempfile.TemporaryDirectory(prefix="vart-disposition-unresolved-") as td:
    path = Path(td) / "d.json"
    write(path, x)
    result = v.verify(path)
    assert result["confirmatory_freeze_eligible"] is False

x = base()
x["resolution"]["pilot_rerun_required"] = True
x["resolution"]["pilot_rerun_complete"] = False
x["confirmatory_freeze_eligible"] = False
expect_reject(x, "POST_PILOT_RERUN_OUTSTANDING")

x = base()
x["defects"] = [{"id": "D2", "class": "scientific_mechanism", "status": "resolved"}]
x["resolution"]["new_preregistration_lineage_required"] = False
x["confirmatory_freeze_eligible"] = True
expect_reject(x, "POST_PILOT_LINEAGE_CLASSIFICATION_MISMATCH")

x = base()
x["confirmatory_execution_authorized"] = True
expect_reject(x, "POST_PILOT_AUTHORITY_VIOLATION")

x = base()
x["resolution"]["confirmatory_source_fetchable"] = False
expect_reject(x, "POST_PILOT_SOURCE_NOT_CLOSED")

print("PASS: post-pilot disposition acceptance + D1-D5 rejection/eligibility cases")
