#!/usr/bin/env python3
"""Source-level adversarial integrity checks for WCARE-38.

These checks prove structural properties of the frozen qualification surface.
They do not execute or qualify any real panel.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_SCHEMA = ROOT / "docs/release/evidence/WCARE38_ATTESTATION_PACKAGE_MANIFEST_SCHEMA_V1.json"
RESULT_SCHEMA = ROOT / "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_RESULT_SCHEMA_V1.json"
ENGINE = ROOT / "scripts/wcare38_qualify_authenticated_panel.py"
FRONTDOOR = ROOT / "scripts/wcare38-qualify.py"


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def main() -> int:
    manifest = json.loads(MANIFEST_SCHEMA.read_text())
    result = json.loads(RESULT_SCHEMA.read_text())
    engine = ENGINE.read_text()
    frontdoor = FRONTDOOR.read_text()

    entry = manifest["properties"]["entries"]["items"]
    entry_properties = set(entry["properties"])
    require(entry.get("additionalProperties") is False, "manifest_entry_allows_unknown_fields")
    require(
        entry_properties == {"subject_receipt_sha256", "envelope_path", "issuer_policy_path"},
        f"manifest_entry_surface_changed:{sorted(entry_properties)}",
    )
    require("result" not in " ".join(entry_properties).lower(), "manifest_accepts_precomputed_result")

    branches = result.get("oneOf")
    require(isinstance(branches, list) and len(branches) == 2, "result_schema_not_two_shape_union")
    full = branches[0]
    failure = branches[1]
    require(full.get("additionalProperties") is False, "full_result_allows_unknown_fields")
    require(failure.get("additionalProperties") is True, "failure_status_cannot_carry_diagnostics")
    result_properties = full["properties"]
    require(
        result_properties["supplemental_packages_contribute_weight"].get("const") is False,
        "supplemental_packages_can_contribute_weight",
    )
    require(
        result_properties["provenance_authentication_partition_complete"].get("const") is True,
        "provenance_partition_not_required_complete",
    )
    require(
        result_properties["relation_authentication_partition_complete"].get("const") is True,
        "relation_partition_not_required_complete",
    )
    failure_properties = failure["properties"]
    require(
        set(failure_properties["disposition"]["enum"])
        == {"AUTHENTICATION_INVALID", "INFRASTRUCTURE_INDETERMINATE"},
        "failure_receipt_disposition_surface_changed",
    )
    require("detail" in failure.get("required", []), "failure_receipt_does_not_require_detail")

    for sentinel in (
        "duplicate_attestation_package_for_receipt",
        "manifest_subject_not_wcare36_receipt",
        "classify_w37_execution(",
        "subject_path",
        "w37_qualifier",
        "ATTESTATION_ACCEPTED",
    ):
        require(sentinel in engine, f"engine_missing_sentinel:{sentinel}")

    require("result_path" not in engine, "engine_accepts_precomputed_wcare37_result_path")
    require("verify_partition(" in frontdoor, "frontdoor_missing_partition_verifier")
    require("supplemental_packages_contribute_weight" in frontdoor, "frontdoor_missing_supplemental_zero_weight")
    require("manifest_subject_duplicate" in frontdoor, "frontdoor_missing_duplicate_manifest_guard")
    require("wcare38_monotonicity_selftest.py" in frontdoor, "frontdoor_missing_monotonicity_gate")
    require("wcare38_frontdoor_sha256" in frontdoor, "frontdoor_not_self_bound")
    require("wcare38_overlay_sha256" in frontdoor, "overlay_not_plan_bound")

    payload = {
        "authority": "MeasurementOnly",
        "classification": "PASS_ADVERSARIAL_SOURCE_SELFTEST",
        "precomputed_green_json_consumable": False,
        "duplicate_package_can_amplify_weight": False,
        "supplemental_package_can_amplify_weight": False,
        "required_partition_may_be_incomplete": False,
        "failure_receipt_must_fabricate_full_census": False,
        "runtime_authority_granted": False,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
