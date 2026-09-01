#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
SCHEMA = "symthaea.vart-world-creative-001.post-pilot-disposition.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_CLASSES = {"instrumentation_plumbing", "scientific_mechanism", "scientific_contract"}


class Reject(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise Reject(code, detail)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Reject("POST_PILOT_DISPOSITION_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise Reject("POST_PILOT_DISPOSITION_INVALID", str(exc)) from exc


def sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            "POST_PILOT_DISPOSITION_INVALID", label)
    return value


def verify(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    require(isinstance(obj, dict), "POST_PILOT_DISPOSITION_INVALID", "root")
    require(obj.get("schema") == SCHEMA and obj.get("experiment_id") == EXPERIMENT_ID,
            "POST_PILOT_DISPOSITION_INVALID", "identity")
    require(obj.get("status") == "dispositioned",
            "POST_PILOT_DISPOSITION_INCOMPLETE", "status must be dispositioned")
    require(obj.get("confirmatory_execution_authorized") is False,
            "POST_PILOT_AUTHORITY_VIOLATION", "confirmatory execution")
    require(obj.get("claim_authorized") is False,
            "POST_PILOT_AUTHORITY_VIOLATION", "claim authorization")

    pilot = obj.get("pilot")
    require(isinstance(pilot, dict), "POST_PILOT_DISPOSITION_INVALID", "pilot")
    sha(pilot.get("pilot_receipt_sha256"), "pilot_receipt_sha256")
    sha(pilot.get("pilot_evidence_closure_sha256"), "pilot_evidence_closure_sha256")
    sha(pilot.get("pilot_design_sha256"), "pilot_design_sha256")
    require(isinstance(pilot.get("source_head"), str) and HEX40.fullmatch(pilot["source_head"]) is not None,
            "POST_PILOT_DISPOSITION_INVALID", "source_head")
    require(isinstance(pilot.get("source_tree"), str) and HEX40.fullmatch(pilot["source_tree"]) is not None,
            "POST_PILOT_DISPOSITION_INVALID", "source_tree")
    require(pilot.get("audit_verdict") == "PILOT_AUDIT_PASS",
            "POST_PILOT_AUDIT_NOT_PASS", str(pilot.get("audit_verdict")))
    require(pilot.get("paired_block_semantics") == "PASS",
            "POST_PILOT_PAIRING_NOT_PASS", str(pilot.get("paired_block_semantics")))

    inspection = obj.get("inspection")
    require(isinstance(inspection, dict), "POST_PILOT_DISPOSITION_INVALID", "inspection")
    require(inspection.get("inspection_purpose") == "instrumentation_and_protocol_only",
            "POST_PILOT_INSPECTION_SCOPE_VIOLATION", "inspection_purpose")
    require(isinstance(inspection.get("inspected_paths"), list),
            "POST_PILOT_DISPOSITION_INVALID", "inspected_paths")

    defects = obj.get("defects")
    require(isinstance(defects, list), "POST_PILOT_DISPOSITION_INVALID", "defects")
    serious = False
    unresolved = 0
    for i, defect in enumerate(defects):
        require(isinstance(defect, dict), "POST_PILOT_DISPOSITION_INVALID", f"defects[{i}]")
        cls = defect.get("class")
        require(cls in ALLOWED_CLASSES, "POST_PILOT_DEFECT_CLASS_INVALID", f"defects[{i}].class")
        require(isinstance(defect.get("id"), str) and defect["id"],
                "POST_PILOT_DISPOSITION_INVALID", f"defects[{i}].id")
        status = defect.get("status")
        require(status in {"resolved", "unresolved"},
                "POST_PILOT_DISPOSITION_INVALID", f"defects[{i}].status")
        if status != "resolved":
            unresolved += 1
        if cls in {"scientific_mechanism", "scientific_contract"}:
            serious = True

    resolution = obj.get("resolution")
    require(isinstance(resolution, dict), "POST_PILOT_DISPOSITION_INVALID", "resolution")
    require(resolution.get("unresolved_defect_count") == unresolved,
            "POST_PILOT_DEFECT_ACCOUNTING_MISMATCH", "unresolved_defect_count")
    require(resolution.get("all_defects_dispositioned") is (unresolved == 0),
            "POST_PILOT_DEFECT_ACCOUNTING_MISMATCH", "all_defects_dispositioned")

    rerun_required = resolution.get("pilot_rerun_required")
    rerun_complete = resolution.get("pilot_rerun_complete")
    require(isinstance(rerun_required, bool) and isinstance(rerun_complete, bool),
            "POST_PILOT_DISPOSITION_INVALID", "pilot rerun booleans")
    require(not rerun_required or rerun_complete,
            "POST_PILOT_RERUN_OUTSTANDING", "pilot rerun required but incomplete")

    lineage_required = resolution.get("new_preregistration_lineage_required")
    lineage_created = resolution.get("new_preregistration_lineage_created")
    require(isinstance(lineage_required, bool) and isinstance(lineage_created, bool),
            "POST_PILOT_DISPOSITION_INVALID", "lineage booleans")
    require(lineage_required is serious,
            "POST_PILOT_LINEAGE_CLASSIFICATION_MISMATCH", f"serious={serious}")
    require(not lineage_required or lineage_created,
            "POST_PILOT_NEW_LINEAGE_REQUIRED", "new preregistration lineage not created")

    require(resolution.get("confirmatory_source_fetchable") is True,
            "POST_PILOT_SOURCE_NOT_CLOSED", "confirmatory source not fetchable")
    require(resolution.get("confirmatory_source_reproducible") is True,
            "POST_PILOT_SOURCE_NOT_CLOSED", "confirmatory source not reproducible")

    eligible = (
        unresolved == 0
        and (not rerun_required or rerun_complete)
        and (not lineage_required or lineage_created)
    )
    require(obj.get("confirmatory_freeze_eligible") is eligible,
            "POST_PILOT_ELIGIBILITY_MISMATCH", f"expected {eligible}")

    return {
        "verdict": "POST_PILOT_DISPOSITION_PASS",
        "experiment_id": EXPERIMENT_ID,
        "defect_count": len(defects),
        "unresolved_defect_count": unresolved,
        "new_preregistration_lineage_required": lineage_required,
        "confirmatory_freeze_eligible": eligible,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify VART-WORLD-CREATIVE-001 post-pilot disposition")
    parser.add_argument("disposition", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.disposition)
    except Reject as exc:
        payload = {
            "verdict": "POST_PILOT_DISPOSITION_REJECT",
            "reason_class": exc.code,
            "detail": exc.detail,
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print("POST_PILOT_DISPOSITION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
