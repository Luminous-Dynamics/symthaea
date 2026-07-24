#!/usr/bin/env python3
"""Independent standard-library verifier for V12 confirmatory releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

FILES = {
    "readiness_release_sha256": ("confirmatory_readiness_release.json", "release_sha256"),
    "collection_protocol_sha256": ("collection_protocol.json", "protocol_sha256"),
    "collection_close_sha256": ("collection_close.json", "receipt_sha256"),
    "unblinding_receipt_sha256": ("unblinding_receipt.json", "receipt_sha256"),
    "analysis_execution_sha256": ("analysis_execution.json", "record_sha256"),
    "publication_record_sha256": ("publication_record.json", "record_sha256"),
    "post_publication_audit_sha256": ("post_publication_audit.json", "ledger_sha256"),
    "study_release_bundle_sha256": ("study_release_bundle.json", "bundle_sha256"),
    "orchestration_log_sha256": ("orchestration.json", "log_sha256"),
}

SELF_FIELDS = {
    "confirmatory_readiness_release.json": "release_sha256",
    "collection_protocol.json": "protocol_sha256",
    "collection_close.json": "receipt_sha256",
    "unblinding_receipt.json": "receipt_sha256",
    "analysis_execution.json": "record_sha256",
    "publication_record.json": "record_sha256",
    "post_publication_audit.json": "ledger_sha256",
    "study_release_bundle.json": "bundle_sha256",
    "orchestration.json": "log_sha256",
    "confirmatory_final_release.json": "bundle_sha256",
}

GENESIS = "0" * 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_self(path: Path, value: dict[str, Any], errors: list[str]) -> None:
    field = SELF_FIELDS[path.name]
    observed = value.get(field)
    if not is_sha256(observed):
        errors.append(f"{path.name}: invalid {field}")
        return
    payload = dict(value)
    payload.pop(field, None)
    if digest(payload) != observed:
        errors.append(f"{path.name}: {field} mismatch")


def verify_orchestration(value: dict[str, Any], errors: list[str]) -> None:
    if value.get("orchestration_version") != "symthaea-muse-study-orchestration-v3":
        errors.append("orchestration: wrong version")
    if value.get("current_phase") != "Published":
        errors.append("orchestration: current phase is not Published")
    previous = GENESIS
    orchestration_id = value.get("orchestration_id")
    for index, transition in enumerate(value.get("transitions", []), start=1):
        if transition.get("sequence") != index:
            errors.append(f"orchestration: transition {index} sequence mismatch")
        if transition.get("previous_transition_sha256") != previous:
            errors.append(f"orchestration: transition {index} chain broken")
        observed = transition.get("transition_sha256")
        payload = dict(transition)
        payload.pop("transition_sha256", None)
        payload = {"orchestration_id": orchestration_id, **payload}
        if not is_sha256(observed) or digest(payload) != observed:
            errors.append(f"orchestration: transition {index} digest mismatch")
        previous = observed


def verify_audit(value: dict[str, Any], publication: dict[str, Any], errors: list[str]) -> None:
    if value.get("publication_sha256") != publication.get("record_sha256"):
        errors.append("post-publication audit: publication binding mismatch")
    previous = GENESIS
    retracted = False
    for index, event in enumerate(value.get("events", []), start=1):
        if retracted:
            errors.append(f"post-publication audit: event {index} occurs after retraction")
        if event.get("sequence") != index:
            errors.append(f"post-publication audit: event {index} sequence mismatch")
        if event.get("previous_event_sha256") != previous:
            errors.append(f"post-publication audit: event {index} chain broken")
        observed = event.get("event_sha256")
        payload = dict(event)
        payload.pop("event_sha256", None)
        payload = {"publication_sha256": value.get("publication_sha256"), **payload}
        if not is_sha256(observed) or digest(payload) != observed:
            errors.append(f"post-publication audit: event {index} digest mismatch")
        previous = observed
        retracted = event.get("event_kind") == "Retraction"
    expected = "Retracted" if retracted else ("Corrected" if value.get("events") else "Active")
    if value.get("current_publication_status") != expected:
        errors.append("post-publication audit: status mismatch")


def verify_semantics(values: dict[str, dict[str, Any]], errors: list[str]) -> None:
    protocol = values["collection_protocol.json"]
    close = values["collection_close.json"]
    unblinding = values["unblinding_receipt.json"]
    analysis = values["analysis_execution.json"]
    publication = values["publication_record.json"]
    audit = values["post_publication_audit.json"]
    release = values["study_release_bundle.json"]
    final = values["confirmatory_final_release.json"]

    if not protocol.get("outcome_monitoring_prohibited"):
        errors.append("collection protocol: outcome monitoring is not prohibited")
    if not protocol.get("codebook_access_prohibited"):
        errors.append("collection protocol: codebook access is not prohibited")
    if close.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("collection close: protocol binding mismatch")
    if not close.get("collection_irreversibly_closed"):
        errors.append("collection close: closure is not irreversible")
    if not close.get("codebook_never_accessed_during_collection"):
        errors.append("collection close: codebook access declaration failed")
    if not close.get("outcome_statistics_never_computed_during_collection"):
        errors.append("collection close: outcome-monitor declaration failed")
    if unblinding.get("collection_close_sha256") != close.get("receipt_sha256"):
        errors.append("unblinding: close binding mismatch")
    if unblinding.get("revealed_key_commitment_sha256") != unblinding.get(
        "randomization_commitment_sha256"
    ):
        errors.append("unblinding: revealed key does not open commitment")
    if analysis.get("collection_close_sha256") != close.get("receipt_sha256"):
        errors.append("analysis: close binding mismatch")
    if analysis.get("unblinding_receipt_sha256") != unblinding.get("receipt_sha256"):
        errors.append("analysis: unblinding binding mismatch")
    deviations = analysis.get("deviations", [])
    expected_claim = (
        "Confirmatory"
        if analysis.get("crosscheck_passed") and not deviations
        else "DescriptiveOnly"
    )
    if analysis.get("claim_status") != expected_claim:
        errors.append("analysis: claim status mismatch")
    expected_conclusion = (
        "DescriptiveOnly"
        if expected_claim == "DescriptiveOnly"
        else ("ConfirmedBenefit" if analysis.get("primary_success") else "DidNotConfirmBenefit")
    )
    if publication.get("primary_conclusion") != expected_conclusion:
        errors.append("publication: primary conclusion mismatch")
    if publication.get("analysis_execution_sha256") != analysis.get("record_sha256"):
        errors.append("publication: analysis binding mismatch")
    endpoints = publication.get("endpoint_disclosures", [])
    names = [json.dumps(item.get("endpoint"), sort_keys=True) for item in endpoints]
    if len(names) != len(set(names)):
        errors.append("publication: duplicate endpoint disclosure")
    verify_audit(audit, publication, errors)
    if final.get("source_revision") != release.get("source_revision"):
        errors.append("final release: source revision mismatch")
    if final.get("workspace_tree_sha256") != release.get("workspace_tree_sha256"):
        errors.append("final release: workspace tree mismatch")
    if final.get("execution_environment_sha256") != release.get(
        "execution_environment_sha256"
    ):
        errors.append("final release: environment mismatch")


def verify_root(root: Path) -> list[str]:
    errors: list[str] = []
    values: dict[str, dict[str, Any]] = {}
    for filename in [*{item[0] for item in FILES.values()}, "confirmatory_final_release.json"]:
        path = root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        value = read_json(path)
        if not isinstance(value, dict):
            errors.append(f"{filename}: expected object")
            continue
        values[filename] = value
        verify_self(path, value, errors)
    if errors:
        return errors
    final = values["confirmatory_final_release.json"]
    for field, (filename, self_field) in FILES.items():
        if final.get(field) != values[filename].get(self_field):
            errors.append(f"final release: {field} mismatch")
    verify_orchestration(values["orchestration.json"], errors)
    verify_semantics(values, errors)
    return errors


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = digest(value)
    return result


def self_test() -> None:
    assert digest({"b": 2, "a": 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        readiness = seal({"release_version": "test"}, "release_sha256")
        protocol = seal(
            {
                "study_id": "study",
                "outcome_monitoring_prohibited": True,
                "codebook_access_prohibited": True,
            },
            "protocol_sha256",
        )
        close = seal(
            {
                "study_id": "study",
                "protocol_sha256": protocol["protocol_sha256"],
                "collection_irreversibly_closed": True,
                "codebook_never_accessed_during_collection": True,
                "outcome_statistics_never_computed_during_collection": True,
            },
            "receipt_sha256",
        )
        unblinding = seal(
            {
                "study_id": "study",
                "collection_close_sha256": close["receipt_sha256"],
                "randomization_commitment_sha256": "1" * 64,
                "revealed_key_commitment_sha256": "1" * 64,
            },
            "receipt_sha256",
        )
        analysis = seal(
            {
                "study_id": "study",
                "collection_close_sha256": close["receipt_sha256"],
                "unblinding_receipt_sha256": unblinding["receipt_sha256"],
                "crosscheck_passed": True,
                "primary_success": False,
                "claim_status": "Confirmatory",
                "deviations": [],
            },
            "record_sha256",
        )
        publication = seal(
            {
                "study_id": "study",
                "analysis_execution_sha256": analysis["record_sha256"],
                "primary_conclusion": "DidNotConfirmBenefit",
                "endpoint_disclosures": [],
            },
            "record_sha256",
        )
        audit = seal(
            {
                "study_id": "study",
                "publication_sha256": publication["record_sha256"],
                "events": [],
                "current_publication_status": "Active",
            },
            "ledger_sha256",
        )
        study_release = seal(
            {
                "source_revision": "abc",
                "workspace_tree_sha256": "2" * 64,
                "execution_environment_sha256": "3" * 64,
            },
            "bundle_sha256",
        )
        orchestration = seal(
            {
                "orchestration_version": "symthaea-muse-study-orchestration-v3",
                "orchestration_id": "study",
                "current_phase": "Published",
                "authorities": [],
                "transitions": [],
            },
            "log_sha256",
        )
        values = {
            "confirmatory_readiness_release.json": readiness,
            "collection_protocol.json": protocol,
            "collection_close.json": close,
            "unblinding_receipt.json": unblinding,
            "analysis_execution.json": analysis,
            "publication_record.json": publication,
            "post_publication_audit.json": audit,
            "study_release_bundle.json": study_release,
            "orchestration.json": orchestration,
        }
        final_payload = {
            field: values[filename][self_field]
            for field, (filename, self_field) in FILES.items()
        }
        final_payload.update(
            {
                "study_id": "study",
                "source_revision": study_release["source_revision"],
                "workspace_tree_sha256": study_release["workspace_tree_sha256"],
                "execution_environment_sha256": study_release[
                    "execution_environment_sha256"
                ],
                "public_release_uri": "https://example.invalid/release",
                "released_at_utc": "now",
                "release_version": "symthaea-muse-confirmatory-final-release-v1",
            }
        )
        values["confirmatory_final_release.json"] = seal(
            final_payload, "bundle_sha256"
        )
        for filename, value in values.items():
            (root / filename).write_text(json.dumps(value), encoding="utf-8")
        assert verify_root(root) == []
        values["analysis_execution.json"]["primary_success"] = True
        (root / "analysis_execution.json").write_text(
            json.dumps(values["analysis_execution.json"]), encoding="utf-8"
        )
        assert verify_root(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("V12 verifier self-test passed")
        return 0
    if args.root is None:
        parser.error("ROOT is required unless --self-test is used")
    errors = verify_root(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("V12 confirmatory execution release verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
