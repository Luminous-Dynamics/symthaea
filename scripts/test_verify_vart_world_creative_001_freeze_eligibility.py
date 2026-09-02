#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import verify_vart_world_creative_001_freeze_eligibility as gate

H64_A = "a" * 64
H64_B = "b" * 64
H64_C = "c" * 64
H64_D = "d" * 64
H64_E = "e" * 64
H64_F = "f" * 64
H40_A = "a" * 40
H40_B = "b" * 40
H40_C = "c" * 40
H40_D = "d" * 40
H40_E = "e" * 40
H40_F = "f" * 40
V05_HEAD = gate.V05_HEAD
V05_TREE = gate.V05_TREE


def dump(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return gate.sha256_file(path)


def disposition(anchor_sha: str) -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.post-pilot-disposition.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "status": "dispositioned",
        "pilot": {
            "pilot_receipt_sha256": H64_A,
            "pilot_evidence_closure_sha256": H64_B,
            "pilot_design_sha256": H64_C,
            "source_head": H40_A,
            "source_tree": H40_B,
            "dual_source_bound": True,
            "instrument_source_head": H40_E,
            "instrument_source_tree": H40_F,
            "preexecution_anchor_sha256": anchor_sha,
            "pilot_config_sha256": H64_D,
            "audit_verdict": "PILOT_AUDIT_PASS",
            "paired_block_semantics": "PASS",
        },
        "inspection": {
            "inspection_purpose": "instrumentation_and_protocol_only",
            "inspected_paths": ["_orchestrator/resolved_plan.json"],
            "outcome_magnitudes_viewed": False,
            "comparative_policy_rankings_viewed": False,
            "human_preference_values_viewed": False,
        },
        "defects": [],
        "resolution": {
            "unresolved_defect_count": 0,
            "all_defects_dispositioned": True,
            "pilot_rerun_required": False,
            "pilot_rerun_complete": False,
            "new_preregistration_lineage_required": False,
            "new_preregistration_lineage_created": False,
        },
        "source_closure_eligible": True,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def anchor() -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.pilot-preexecution-anchor.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "pilot_config_sha256": H64_D,
        "pilot_design_sha256": H64_C,
        "subject_source_head": H40_A,
        "subject_source_tree": H40_B,
        "instrument_source_head": H40_E,
        "instrument_source_tree": H40_F,
        "runner_source_sha256": H64_E,
        "auditor_source_sha256": H64_F,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def attestation(anchor_sha: str) -> dict[str, object]:
    obj = dict(anchor())
    obj["schema"] = "symthaea.vart-world-creative-001.pilot-anchor-attestation.v1"
    obj["preexecution_anchor_sha256"] = anchor_sha
    obj["pilot_receipt_sha256"] = H64_A
    obj["pilot_evidence_closure_sha256"] = H64_B
    obj["audit_verdict"] = "PILOT_AUDIT_PASS"
    return obj


def subject_source_closure() -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.source-closure.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "status": "qualified",
        "confirmatory_source": {
            "head": H40_C,
            "tree": H40_D,
            "parent_v05a_head": V05_HEAD,
            "parent_v05a_tree": V05_TREE,
        },
        "pilot_predecessor": {
            "head": H40_A,
            "tree": H40_B,
            "is_ancestor_of_confirmatory_source": True,
        },
        "remote": {
            "repository_full_name": "Luminous-Dynamics/symthaea",
            "ref": "refs/heads/research/vart-confirmatory-subject",
            "fetch_verified": True,
            "fetched_head": H40_C,
            "fetched_tree": H40_D,
            "fresh_checkout_verified": True,
            "fresh_checkout_head": H40_C,
            "fresh_checkout_tree": H40_D,
        },
        "reproduction": {
            "environment_digest": H64_A,
            "lock_manifest_sha256": H64_B,
            "qualification_receipt_sha256": H64_C,
            "independent_checkout_gate": True,
        },
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def instrument_qualification() -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.instrument-qualification.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "status": "qualified",
        "instrument_source": {"head": H40_E, "tree": H40_F, "dirty": False},
        "instrument_manifest_sha256": H64_D,
        "instrument_environment_digest": H64_E,
        "all_suites_pass": True,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def instrument_source_closure(qualification_sha: str) -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.instrument-source-closure.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "status": "qualified",
        "instrument_source": {"head": H40_E, "tree": H40_F},
        "remote": {
            "repository_full_name": "Luminous-Dynamics/symthaea",
            "ref": "refs/heads/research/vart-world-creative-001-execution",
            "fetch_verified": True,
            "fetched_head": H40_E,
            "fetched_tree": H40_F,
            "fresh_checkout_verified": True,
            "fresh_checkout_head": H40_E,
            "fresh_checkout_tree": H40_F,
        },
        "qualification": {
            "instrument_qualification_receipt_sha256": qualification_sha,
            "instrument_manifest_sha256": H64_D,
            "instrument_environment_digest": H64_E,
            "all_suites_pass": True,
        },
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def expect_reject(paths: tuple[Path, ...], code: str) -> None:
    try:
        gate.verify(*paths)
    except gate.Reject as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected {code}")


with tempfile.TemporaryDirectory(prefix="vart-freeze-eligibility-") as td:
    root = Path(td)
    disposition_path = root / "disposition.json"
    anchor_path = root / "anchor.json"
    attestation_path = root / "attestation.json"
    subject_path = root / "subject-source.json"
    instrument_q_path = root / "instrument-qualification.json"
    instrument_source_path = root / "instrument-source.json"

    anchor_sha = dump(anchor_path, anchor())
    dump(disposition_path, disposition(anchor_sha))
    dump(attestation_path, attestation(anchor_sha))
    dump(subject_path, subject_source_closure())
    instrument_q_sha = dump(instrument_q_path, instrument_qualification())
    dump(instrument_source_path, instrument_source_closure(instrument_q_sha))
    paths = (
        disposition_path,
        anchor_path,
        attestation_path,
        subject_path,
        instrument_q_path,
        instrument_source_path,
    )

    result = gate.verify(*paths)
    assert result["verdict"] == "CONFIRMATORY_FREEZE_PREPARATION_ELIGIBLE"
    assert result["instrument_source_head"] == H40_E
    assert result["confirmatory_execution_authorized"] is False
    assert result["claim_authorized"] is False

    # F1 — post-run attestation cannot substitute a different pre-execution anchor.
    obj = attestation(anchor_sha)
    obj["preexecution_anchor_sha256"] = H64_A
    dump(attestation_path, obj)
    expect_reject(paths, "PILOT_ANCHOR_DIGEST_MISMATCH")
    dump(attestation_path, attestation(anchor_sha))

    # F2 — pilot design may not drift between pre-anchor and post-attestation.
    obj = attestation(anchor_sha)
    obj["pilot_design_sha256"] = H64_F
    dump(attestation_path, obj)
    expect_reject(paths, "PILOT_ANCHOR_DIGEST_MISMATCH")
    dump(attestation_path, attestation(anchor_sha))

    # F3 — subject source closure must descend from the dispositioned pilot subject.
    obj = subject_source_closure()
    obj["pilot_predecessor"]["head"] = H40_D
    dump(subject_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_PILOT_PREDECESSOR_MISMATCH")
    dump(subject_path, subject_source_closure())

    # F4 — subject remote must resolve to exact confirmatory HEAD/TREE.
    obj = subject_source_closure()
    obj["remote"]["fetched_head"] = H40_A
    dump(subject_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_REMOTE_IDENTITY_MISMATCH")
    dump(subject_path, subject_source_closure())

    # F5 — the qualified v0.5-A baseline identity is exact, not any 40-hex pair.
    obj = subject_source_closure()
    obj["confirmatory_source"]["parent_v05a_head"] = H40_A
    dump(subject_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_BASELINE_MISMATCH")
    dump(subject_path, subject_source_closure())

    # F6 — source closure cannot authorize execution itself.
    obj = subject_source_closure()
    obj["confirmatory_execution_authorized"] = True
    dump(subject_path, obj)
    expect_reject(paths, "FREEZE_ELIGIBILITY_AUTHORITY_VIOLATION")
    dump(subject_path, subject_source_closure())

    # F7 — instrument qualification must belong to the anchored instrument source.
    obj = instrument_qualification()
    obj["instrument_source"]["head"] = H40_A
    new_q_sha = dump(instrument_q_path, obj)
    dump(instrument_source_path, instrument_source_closure(new_q_sha))
    expect_reject(paths, "INSTRUMENT_QUALIFICATION_SOURCE_MISMATCH")
    instrument_q_sha = dump(instrument_q_path, instrument_qualification())
    dump(instrument_source_path, instrument_source_closure(instrument_q_sha))

    # F8 — instrument source closure must bind exact qualification receipt bytes.
    obj = instrument_source_closure(instrument_q_sha)
    obj["qualification"]["instrument_qualification_receipt_sha256"] = H64_A
    dump(instrument_source_path, obj)
    expect_reject(paths, "INSTRUMENT_SOURCE_CLOSURE_QUALIFICATION_MISMATCH")
    dump(instrument_source_path, instrument_source_closure(instrument_q_sha))

    # F9 — instrument durable ref/fresh checkout identity cannot be substituted.
    obj = instrument_source_closure(instrument_q_sha)
    obj["remote"]["fresh_checkout_tree"] = H40_A
    dump(instrument_source_path, obj)
    expect_reject(paths, "INSTRUMENT_SOURCE_CLOSURE_REMOTE_IDENTITY_MISMATCH")
    dump(instrument_source_path, instrument_source_closure(instrument_q_sha))

    # F10 — instrument closure cannot escalate authority.
    obj = instrument_source_closure(instrument_q_sha)
    obj["claim_authorized"] = True
    dump(instrument_source_path, obj)
    expect_reject(paths, "FREEZE_ELIGIBILITY_AUTHORITY_VIOLATION")
    dump(instrument_source_path, instrument_source_closure(instrument_q_sha))

    # F11 — instrument identity is part of the pre/post anchor, not a mutable label.
    obj = attestation(anchor_sha)
    obj["instrument_source_tree"] = H40_A
    dump(attestation_path, obj)
    expect_reject(paths, "PILOT_ANCHOR_DIGEST_MISMATCH")

print("PASS: scoped-disposition dual-source freeze eligibility + F1-F11 rejection")
