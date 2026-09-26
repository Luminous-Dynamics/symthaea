#!/usr/bin/env python3
"""Independent reference oracle for SEMI-EQP-MET-001E2.

Imports no Symthaea production code. It validates the exact frozen synthetic
bench-subject corpus and derives each expected relation from fixture inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

EXPECTED_SHA256 = "496f5686df93d5995c82f982f9c164d7a2d775b2657395eca62235c9271ba384"
EXPECTED_SCHEMA = "semi-eqp-met-001e2-bench-subject-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_IDS = {
    "exact_as_built_subject_complete",
    "design_only_without_physical_article",
    "friendly_names_not_instance_identity",
    "same_design_two_physical_articles_distinct",
    "camera_replacement_creates_new_as_built_generation",
    "optics_change_creates_new_configuration_context",
    "parser_or_firmware_change_preserves_hardware_but_changes_acquisition_context",
    "stale_calibration_blocks_current_campaign_subject",
    "reference_target_binding_mismatch",
    "unordered_subsystem_refs_canonicalize",
    "duplicate_subsystem_ref_rejected",
    "imported_subsystems_preserve_import_dependence",
    "missing_campaign_or_custody_binding_incomplete",
    "remount_without_calibration_transfer_requires_new_context",
    "exploratory_prototype_manifest_not_claim_bearing",
    "bench_subject_manifest_cannot_mint_execution_authority",
}

CORPUS = Path("docs/release/evidence/semi-eqp-met-001e2-bench-subject-corpus-v1.json")


def result(kind: str, value):
    return kind, value


def derive(case: dict):
    cid = case["id"]

    if cid == "exact_as_built_subject_complete":
        complete = all(
            case.get(k)
            for k in (
                "physical_article_ref",
                "as_built_configuration_ref",
                "installed_subsystem_refs",
                "acquisition_profile_ref",
                "calibration_ref",
                "campaign_ref",
                "reference_target_ref",
            )
        ) and case.get("calibration_currentness") == "current"
        return result("expected_disposition", "CampaignSubjectManifestComplete" if complete else "CampaignBindingIncomplete")

    if cid == "design_only_without_physical_article":
        return result("expected_error", "PhysicalSubjectMissing" if not case.get("physical_article_ref") else None)

    if cid == "friendly_names_not_instance_identity":
        weak_only = (
            bool(case.get("friendly_name"))
            and bool(case.get("camera_name"))
            and bool(case.get("stage_name"))
            and not case.get("physical_article_ref")
            and not case.get("installed_subsystem_refs")
        )
        return result("expected_error", "WeakIdentityInsufficient" if weak_only else None)

    if cid == "same_design_two_physical_articles_distinct":
        distinct = case.get("physical_article_ref_a") != case.get("physical_article_ref_b")
        return result("expected_identity_relation", "distinct" if distinct else "same")

    if cid == "camera_replacement_creates_new_as_built_generation":
        changed = (
            case.get("as_built_configuration_ref_a") != case.get("as_built_configuration_ref_b")
            and case.get("removed_subsystem_ref") != case.get("installed_subsystem_ref")
        )
        return result("expected_identity_relation", "distinct_configuration_generation" if changed else "same")

    if cid == "optics_change_creates_new_configuration_context":
        changed = case.get("optics_ref_a") != case.get("optics_ref_b")
        return result("expected_identity_relation", "distinct_configuration_context" if changed else "same")

    if cid == "parser_or_firmware_change_preserves_hardware_but_changes_acquisition_context":
        changed = case.get("acquisition_profile_ref_a") != case.get("acquisition_profile_ref_b")
        same_article = bool(case.get("physical_article_ref")) and bool(case.get("as_built_configuration_ref"))
        value = "same_physical_article_changed_acquisition_context" if changed and same_article else "unresolved"
        return result("expected_relation", value)

    if cid == "stale_calibration_blocks_current_campaign_subject":
        value = "CampaignSubjectNotCurrent" if case.get("calibration_currentness") != "current" else "CampaignSubjectCurrent"
        return result("expected_disposition", value)

    if cid == "reference_target_binding_mismatch":
        mismatch = case.get("manifest_reference_target_ref") != case.get("campaign_reference_target_ref")
        return result("expected_error", "TargetBindingMismatch" if mismatch else None)

    if cid == "unordered_subsystem_refs_canonicalize":
        same = (
            case.get("declared_order_semantics") == "set"
            and sorted(case.get("subsystem_refs_a", [])) == sorted(case.get("subsystem_refs_b", []))
        )
        return result("expected_identity_relation", "same" if same else "distinct")

    if cid == "duplicate_subsystem_ref_rejected":
        refs = case.get("subsystem_refs", [])
        return result("expected_error", "DuplicateReference" if len(refs) != len(set(refs)) else None)

    if cid == "imported_subsystems_preserve_import_dependence":
        imported = bool(case.get("imported_subsystem_refs"))
        operational = case.get("instrument_status") == "operational_under_profile"
        value = "operational_but_import_dependent" if imported and operational else "unresolved"
        return result("expected_closure_relation", value)

    if cid == "missing_campaign_or_custody_binding_incomplete":
        incomplete = not case.get("campaign_ref") or not case.get("custody_context_ref")
        return result("expected_disposition", "CampaignBindingIncomplete" if incomplete else "CampaignBindingComplete")

    if cid == "remount_without_calibration_transfer_requires_new_context":
        remounted = case.get("pre_mount_ref") != case.get("post_mount_ref")
        missing_transfer = not case.get("calibration_transfer_evidence_ref")
        value = "CalibrationTransferRequired" if remounted and missing_transfer else "TransferEvidencePresent"
        return result("expected_disposition", value)

    if cid == "exploratory_prototype_manifest_not_claim_bearing":
        value = "ExploratorySubjectOnly" if case.get("manifest_mode") == "exploratory" else "ClaimBearingCandidate"
        return result("expected_disposition", value)

    if cid == "bench_subject_manifest_cannot_mint_execution_authority":
        value = "none" if case.get("execution_authority") is False else "invalid"
        return result("expected_authority", value)

    raise AssertionError(f"unhandled fixture: {cid}")


def main() -> int:
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"corpus digest drift: {digest}")
    corpus = json.loads(raw)
    if corpus.get("schema") != EXPECTED_SCHEMA:
        raise AssertionError("schema drift")
    if corpus.get("authority") != EXPECTED_AUTHORITY:
        raise AssertionError("authority drift")

    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise AssertionError("case-count drift")
    ids = [c.get("id") for c in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate fixture id")
    if set(ids) != EXPECTED_IDS:
        raise AssertionError("fixture-id drift")

    for case in cases:
        field, actual = derive(case)
        expected = case.get(field)
        if actual != expected:
            raise AssertionError(
                f"{case['id']}: derived {field}={actual!r}, frozen={expected!r}"
            )

    print(f"ok fixtures={len(cases)} digest={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
