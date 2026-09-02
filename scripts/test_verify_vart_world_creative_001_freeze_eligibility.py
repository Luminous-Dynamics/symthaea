#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import verify_vart_world_creative_001_freeze_eligibility as gate

H64_A = "a" * 64
H64_B = "b" * 64
H64_C = "c" * 64
H64_D = "d" * 64
H40_A = "a" * 40
H40_B = "b" * 40
H40_C = "c" * 40
H40_D = "d" * 40
V05_HEAD = "33820b3d9e904280e6264719fe7717cb2e5dd5bb"
V05_TREE = "e93c6dbfa05b602100ff924efaa5d95f92ef5a65"


def dump(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disposition() -> dict[str, object]:
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
            "audit_verdict": "PILOT_AUDIT_PASS",
            "paired_block_semantics": "PASS",
        },
        "inspection": {
            "inspection_purpose": "instrumentation_and_protocol_only",
            "inspected_paths": ["_orchestrator/resolved_plan.json"],
        },
        "defects": [],
        "resolution": {
            "unresolved_defect_count": 0,
            "all_defects_dispositioned": True,
            "pilot_rerun_required": False,
            "pilot_rerun_complete": False,
            "new_preregistration_lineage_required": False,
            "new_preregistration_lineage_created": False,
            "confirmatory_source_fetchable": True,
            "confirmatory_source_reproducible": True,
        },
        "confirmatory_freeze_eligible": True,
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
        "source_head": H40_A,
        "source_tree": H40_B,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def attestation(anchor_sha: str) -> dict[str, object]:
    return {
        "schema": "symthaea.vart-world-creative-001.pilot-anchor-attestation.v1",
        "experiment_id": gate.EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "preexecution_anchor_sha256": anchor_sha,
        "pilot_config_sha256": H64_D,
        "pilot_design_sha256": H64_C,
        "pilot_receipt_sha256": H64_A,
        "pilot_evidence_closure_sha256": H64_B,
        "audit_verdict": "PILOT_AUDIT_PASS",
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def source_closure() -> dict[str, object]:
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
            "ref": "refs/heads/research/vart-confirmatory",
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


def expect_reject(paths: tuple[Path, Path, Path, Path], code: str) -> None:
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
    source_path = root / "source.json"

    dump(disposition_path, disposition())
    anchor_sha = dump(anchor_path, anchor())
    dump(attestation_path, attestation(anchor_sha))
    dump(source_path, source_closure())
    paths = (disposition_path, anchor_path, attestation_path, source_path)

    result = gate.verify(*paths)
    assert result["verdict"] == "CONFIRMATORY_FREEZE_PREPARATION_ELIGIBLE"
    assert result["confirmatory_execution_authorized"] is False
    assert result["claim_authorized"] is False

    # F1 — attestation may not substitute a different pre-execution anchor.
    obj = attestation(anchor_sha)
    obj["preexecution_anchor_sha256"] = H64_A
    dump(attestation_path, obj)
    expect_reject(paths, "PILOT_ANCHOR_DIGEST_MISMATCH")
    dump(attestation_path, attestation(anchor_sha))

    # F2 — attested semantic design must equal the dispositioned pilot design.
    obj = attestation(anchor_sha)
    obj["pilot_design_sha256"] = H64_D
    dump(attestation_path, obj)
    expect_reject(paths, "PILOT_ANCHOR_DIGEST_MISMATCH")
    dump(attestation_path, attestation(anchor_sha))

    # F3 — source closure must descend from the dispositioned pilot source identity.
    obj = source_closure()
    obj["pilot_predecessor"]["head"] = H40_D
    dump(source_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_PILOT_PREDECESSOR_MISMATCH")
    dump(source_path, source_closure())

    # F4 — a named remote that resolves to a different HEAD is not closure.
    obj = source_closure()
    obj["remote"]["fetched_head"] = H40_A
    dump(source_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_REMOTE_IDENTITY_MISMATCH")
    dump(source_path, source_closure())

    # F5 — fresh-checkout equivalence is required, not merely ls-remote reachability.
    obj = source_closure()
    obj["remote"]["fresh_checkout_verified"] = False
    dump(source_path, obj)
    expect_reject(paths, "SOURCE_CLOSURE_REMOTE_NOT_VERIFIED")
    dump(source_path, source_closure())

    # F6 — transition evidence can never authorize confirmatory execution itself.
    obj = source_closure()
    obj["confirmatory_execution_authorized"] = True
    dump(source_path, obj)
    expect_reject(paths, "FREEZE_ELIGIBILITY_AUTHORITY_VIOLATION")
    dump(source_path, source_closure())

    # F7 — reproduction context must contain real digests and independent-checkout gate.
    obj = source_closure()
    obj["reproduction"]["environment_digest"] = None
    dump(source_path, obj)
    expect_reject(paths, "FREEZE_ELIGIBILITY_INVALID")
    dump(source_path, source_closure())

print("PASS: freeze-eligibility canonical acceptance + F1-F7 deterministic rejection")
