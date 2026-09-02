#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001_post_pilot as post_pilot

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
        raise Reject("FREEZE_ELIGIBILITY_EVIDENCE_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise Reject("FREEZE_ELIGIBILITY_JSON_INVALID", f"{path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hex40(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX40.fullmatch(value) is not None,
            "FREEZE_ELIGIBILITY_INVALID", label)
    return value


def hex64(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            "FREEZE_ELIGIBILITY_INVALID", label)
    return value


def verify_anchor(anchor_path: Path, attestation_path: Path, disposition: dict[str, Any]) -> dict[str, Any]:
    anchor = read_json(anchor_path)
    attestation = read_json(attestation_path)
    require(isinstance(anchor, dict) and isinstance(attestation, dict),
            "PILOT_ANCHOR_INVALID", "anchor/attestation roots")
    require(anchor.get("schema") == "symthaea.vart-world-creative-001.pilot-preexecution-anchor.v1"
            and anchor.get("experiment_id") == EXPERIMENT_ID,
            "PILOT_ANCHOR_INVALID", "preexecution anchor identity")
    require(attestation.get("schema") == "symthaea.vart-world-creative-001.pilot-anchor-attestation.v1"
            and attestation.get("experiment_id") == EXPERIMENT_ID,
            "PILOT_ANCHOR_INVALID", "attestation identity")
    for obj, name in ((anchor, "anchor"), (attestation, "attestation")):
        require(obj.get("campaign") == "pilot" and obj.get("noncanonical") is True,
                "PILOT_ANCHOR_INVALID", f"{name} campaign")
        require(obj.get("confirmatory_execution_authorized") is False and obj.get("claim_authorized") is False,
                "FREEZE_ELIGIBILITY_AUTHORITY_VIOLATION", name)

    anchor_sha = sha256_file(anchor_path)
    require(attestation.get("preexecution_anchor_sha256") == anchor_sha,
            "PILOT_ANCHOR_DIGEST_MISMATCH", "preexecution anchor")
    require(attestation.get("pilot_config_sha256") == anchor.get("pilot_config_sha256"),
            "PILOT_ANCHOR_DIGEST_MISMATCH", "pilot config")
    require(attestation.get("pilot_design_sha256") == anchor.get("pilot_design_sha256"),
            "PILOT_ANCHOR_DIGEST_MISMATCH", "pilot design")
    require(attestation.get("audit_verdict") == "PILOT_AUDIT_PASS",
            "PILOT_ANCHOR_AUDIT_NOT_PASS", str(attestation.get("audit_verdict")))

    pilot = disposition.get("pilot")
    require(isinstance(pilot, dict), "FREEZE_ELIGIBILITY_INVALID", "disposition.pilot")
    require(attestation.get("pilot_design_sha256") == pilot.get("pilot_design_sha256"),
            "PILOT_ANCHOR_DISPOSITION_MISMATCH", "pilot_design_sha256")
    require(attestation.get("pilot_receipt_sha256") == pilot.get("pilot_receipt_sha256"),
            "PILOT_ANCHOR_DISPOSITION_MISMATCH", "pilot_receipt_sha256")
    require(attestation.get("pilot_evidence_closure_sha256") == pilot.get("pilot_evidence_closure_sha256"),
            "PILOT_ANCHOR_DISPOSITION_MISMATCH", "pilot_evidence_closure_sha256")

    return {
        "preexecution_anchor_sha256": anchor_sha,
        "anchor_attestation_sha256": sha256_file(attestation_path),
        "pilot_config_sha256": hex64(anchor.get("pilot_config_sha256"), "pilot_config_sha256"),
        "pilot_design_sha256": hex64(anchor.get("pilot_design_sha256"), "pilot_design_sha256"),
    }


def verify_source_closure(path: Path, disposition: dict[str, Any]) -> dict[str, Any]:
    obj = read_json(path)
    require(isinstance(obj, dict), "SOURCE_CLOSURE_INVALID", "root")
    require(obj.get("schema") == "symthaea.vart-world-creative-001.source-closure.v1"
            and obj.get("experiment_id") == EXPERIMENT_ID
            and obj.get("status") == "qualified",
            "SOURCE_CLOSURE_INVALID", "identity/status")
    require(obj.get("confirmatory_execution_authorized") is False and obj.get("claim_authorized") is False,
            "FREEZE_ELIGIBILITY_AUTHORITY_VIOLATION", "source closure")

    source = obj.get("confirmatory_source")
    predecessor = obj.get("pilot_predecessor")
    remote = obj.get("remote")
    reproduction = obj.get("reproduction")
    for value, label in ((source, "confirmatory_source"), (predecessor, "pilot_predecessor"),
                         (remote, "remote"), (reproduction, "reproduction")):
        require(isinstance(value, dict), "SOURCE_CLOSURE_INVALID", label)

    head = hex40(source.get("head"), "confirmatory_source.head")
    tree = hex40(source.get("tree"), "confirmatory_source.tree")
    hex40(source.get("parent_v05a_head"), "parent_v05a_head")
    hex40(source.get("parent_v05a_tree"), "parent_v05a_tree")
    pred_head = hex40(predecessor.get("head"), "pilot_predecessor.head")
    pred_tree = hex40(predecessor.get("tree"), "pilot_predecessor.tree")
    require(predecessor.get("is_ancestor_of_confirmatory_source") is True,
            "SOURCE_CLOSURE_ANCESTRY_NOT_PROVEN", "pilot predecessor")

    pilot = disposition.get("pilot")
    require(isinstance(pilot, dict), "FREEZE_ELIGIBILITY_INVALID", "disposition.pilot")
    require(pred_head == pilot.get("source_head") and pred_tree == pilot.get("source_tree"),
            "SOURCE_CLOSURE_PILOT_PREDECESSOR_MISMATCH", "pilot source identity")

    require(isinstance(remote.get("repository_full_name"), str) and remote.get("repository_full_name"),
            "SOURCE_CLOSURE_INVALID", "remote.repository_full_name")
    require(isinstance(remote.get("ref"), str) and remote.get("ref"),
            "SOURCE_CLOSURE_INVALID", "remote.ref")
    require(remote.get("fetch_verified") is True and remote.get("fresh_checkout_verified") is True,
            "SOURCE_CLOSURE_REMOTE_NOT_VERIFIED", "fetch/fresh checkout")
    require(remote.get("fetched_head") == head and remote.get("fresh_checkout_head") == head,
            "SOURCE_CLOSURE_REMOTE_IDENTITY_MISMATCH", "HEAD")
    require(remote.get("fetched_tree") == tree and remote.get("fresh_checkout_tree") == tree,
            "SOURCE_CLOSURE_REMOTE_IDENTITY_MISMATCH", "TREE")

    hex64(reproduction.get("environment_digest"), "reproduction.environment_digest")
    hex64(reproduction.get("lock_manifest_sha256"), "reproduction.lock_manifest_sha256")
    hex64(reproduction.get("qualification_receipt_sha256"), "reproduction.qualification_receipt_sha256")
    require(reproduction.get("independent_checkout_gate") is True,
            "SOURCE_CLOSURE_REPRODUCTION_NOT_VERIFIED", "independent checkout gate")

    return {
        "source_closure_sha256": sha256_file(path),
        "confirmatory_source_head": head,
        "confirmatory_source_tree": tree,
        "environment_digest": reproduction["environment_digest"],
        "remote_repository_full_name": remote["repository_full_name"],
        "remote_ref": remote["ref"],
    }


def verify(disposition_path: Path, anchor_path: Path, attestation_path: Path, source_closure_path: Path) -> dict[str, Any]:
    try:
        post_result = post_pilot.verify(disposition_path)
    except post_pilot.Reject as exc:
        raise Reject("POST_PILOT_DISPOSITION_NOT_QUALIFIED", f"{exc.code}: {exc.detail}") from exc
    require(post_result.get("confirmatory_freeze_eligible") is True,
            "POST_PILOT_FREEZE_NOT_ELIGIBLE", "disposition")
    disposition = read_json(disposition_path)
    require(isinstance(disposition, dict), "FREEZE_ELIGIBILITY_INVALID", "disposition root")

    anchor = verify_anchor(anchor_path, attestation_path, disposition)
    source = verify_source_closure(source_closure_path, disposition)

    return {
        "verdict": "CONFIRMATORY_FREEZE_PREPARATION_ELIGIBLE",
        "experiment_id": EXPERIMENT_ID,
        "post_pilot_disposition_sha256": sha256_file(disposition_path),
        **anchor,
        **source,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "bounded_statement": (
            "Transition evidence is coherent enough to prepare a prospective confirmatory freeze. "
            "This verdict does not authorize confirmatory execution or scientific claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify VART transition eligibility to prepare a confirmatory freeze")
    parser.add_argument("--disposition", type=Path, required=True)
    parser.add_argument("--pilot-anchor", type=Path, required=True)
    parser.add_argument("--pilot-attestation", type=Path, required=True)
    parser.add_argument("--source-closure", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.disposition, args.pilot_anchor, args.pilot_attestation, args.source_closure)
    except Reject as exc:
        payload = {
            "verdict": "CONFIRMATORY_FREEZE_PREPARATION_REJECT",
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
        print("CONFIRMATORY_FREEZE_PREPARATION_ELIGIBLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
