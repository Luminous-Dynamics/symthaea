#!/usr/bin/env python3
"""Independent standard-library verifier for V11 readiness evidence.

This verifier intentionally does not import Rust code. It validates the V11
self-commitments and the cross-file root release in a conventional directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

FILES = {
    "study_operations_release_sha256": "operations_release.json",
    "external_review_protocol_sha256": "external_review_protocol.json",
    "review_evidence_index_sha256": "review_evidence_index.json",
    "review_resolution_ledger_sha256": "review_resolution.json",
    "review_completion_sha256": "review_completion.json",
    "confirmatory_amendment_ledger_sha256": "confirmatory_amendments.json",
    "workspace_validation_sha256": "workspace_validation.json",
    "human_governance_sha256": "human_governance.json",
    "dry_run_sha256": "dry_run.json",
    "independent_reproduction_sha256": "reproduction_readiness.json",
    "readiness_report_sha256": "confirmatory_readiness.json",
}

SELF_FIELDS = {
    "external_review_protocol.json": "protocol_sha256",
    "review_evidence_index.json": "index_sha256",
    "review_resolution.json": "ledger_sha256",
    "review_completion.json": "completion_sha256",
    "confirmatory_amendments.json": "ledger_sha256",
    "workspace_validation.json": "evidence_sha256",
    "human_governance.json": "governance_sha256",
    "dry_run.json": "dry_run_sha256",
    "reproduction_readiness.json": "reproduction_sha256",
    "confirmatory_readiness.json": "report_sha256",
    "confirmatory_readiness_release.json": "release_sha256",
}


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


def verify_self_commitment(path: Path, value: dict[str, Any], errors: list[str]) -> None:
    field = SELF_FIELDS.get(path.name)
    if field is None:
        return
    observed = value.get(field)
    if not is_sha256(observed):
        errors.append(f"{path.name}: invalid {field}")
        return
    payload = dict(value)
    payload.pop(field, None)
    expected = digest(payload)
    if observed != expected:
        errors.append(f"{path.name}: {field} mismatch")


def verify_authority_snapshot(amendments: dict[str, Any], errors: list[str]) -> None:
    snapshot = amendments.get("baseline_authority")
    if not isinstance(snapshot, dict):
        errors.append("confirmatory_amendments.json: missing baseline_authority")
        return
    observed = snapshot.get("snapshot_sha256")
    payload = dict(snapshot)
    payload.pop("snapshot_sha256", None)
    if not is_sha256(observed) or observed != digest(payload):
        errors.append("confirmatory_amendments.json: baseline snapshot mismatch")


def verify_package_and_response_sets(root: Path, release: dict[str, Any], errors: list[str]) -> None:
    packages = read_json(root / "review_packages.json")
    responses = read_json(root / "review_responses.json")
    if not isinstance(packages, list) or not isinstance(responses, list):
        errors.append("review package/response sets must be arrays")
        return
    package_digests: list[str] = []
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            errors.append(f"review_packages.json[{index}] is not an object")
            continue
        observed = package.get("package_sha256")
        payload = dict(package)
        payload.pop("package_sha256", None)
        if not is_sha256(observed) or observed != digest(payload):
            errors.append(f"review_packages.json[{index}] commitment mismatch")
        else:
            package_digests.append(observed)
    response_digests: list[str] = []
    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            errors.append(f"review_responses.json[{index}] is not an object")
            continue
        observed = response.get("response_sha256")
        payload = dict(response)
        payload.pop("response_sha256", None)
        if not is_sha256(observed) or observed != digest(payload):
            errors.append(f"review_responses.json[{index}] commitment mismatch")
        else:
            response_digests.append(observed)
    if digest(sorted(package_digests)) != release.get("review_package_set_sha256"):
        errors.append("release: review_package_set_sha256 mismatch")
    if digest(sorted(response_digests)) != release.get("review_response_set_sha256"):
        errors.append("release: review_response_set_sha256 mismatch")


def verify_readiness_semantics(readiness: dict[str, Any], errors: list[str]) -> None:
    checks = readiness.get("checks")
    if not isinstance(checks, list):
        errors.append("confirmatory_readiness.json: checks must be an array")
        return
    nonblocking = [
        check.get("gate")
        for check in checks
        if isinstance(check, dict) and not check.get("blocking")
    ]
    if nonblocking:
        errors.append(f"readiness contains nonblocking required gates: {nonblocking}")
    failed = [
        check.get("gate")
        for check in checks
        if isinstance(check, dict) and check.get("blocking") and not check.get("passed")
    ]
    decision = readiness.get("decision")
    if decision == "ReadyForConfirmatoryCollection" and failed:
        errors.append(f"readiness decision is Ready with failed gates: {failed}")
    if decision == "NotReady" and not failed:
        errors.append("readiness decision is NotReady without a failed blocking gate")


def verify_root(root: Path) -> list[str]:
    errors: list[str] = []
    release_path = root / "confirmatory_readiness_release.json"
    release = read_json(release_path)
    if not isinstance(release, dict):
        return ["confirmatory_readiness_release.json is not an object"]
    verify_self_commitment(release_path, release, errors)

    for field, filename in FILES.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        value = read_json(path)
        if not isinstance(value, dict):
            errors.append(f"{filename} is not an object")
            continue
        verify_self_commitment(path, value, errors)
        if digest(value) != release.get(field):
            errors.append(f"release: {field} mismatch")
        if filename == "confirmatory_amendments.json":
            verify_authority_snapshot(value, errors)
        if filename == "confirmatory_readiness.json":
            verify_readiness_semantics(value, errors)

    for filename in ["review_packages.json", "review_responses.json"]:
        if not (root / filename).is_file():
            errors.append(f"missing {filename}")
    if (root / "review_packages.json").is_file() and (root / "review_responses.json").is_file():
        verify_package_and_response_sets(root, release, errors)

    for field in [
        "source_archive_sha256",
        "flake_lock_sha256",
        "toolchain_evidence_sha256",
        "external_timestamp_receipt_sha256",
        "release_sha256",
    ]:
        if not is_sha256(release.get(field)):
            errors.append(f"release: invalid {field}")
    return errors


def seal_object(value: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field] = digest(value)
    return sealed


def self_test() -> None:
    sample = {"b": 2, "a": 1}
    assert digest(sample) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    assert is_sha256("a" * 64)
    assert not is_sha256("g" * 64)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values: dict[str, dict[str, Any]] = {
            "operations_release.json": {"release_version": "test"},
            "external_review_protocol.json": seal_object(
                {"protocol_version": "test"}, "protocol_sha256"
            ),
            "review_evidence_index.json": seal_object(
                {"entries": []}, "index_sha256"
            ),
            "review_resolution.json": seal_object(
                {"resolutions": []}, "ledger_sha256"
            ),
            "review_completion.json": seal_object(
                {"all_findings_resolved": True}, "completion_sha256"
            ),
            "workspace_validation.json": seal_object(
                {"workspace_tree_clean": True}, "evidence_sha256"
            ),
            "human_governance.json": seal_object(
                {"participant_withdrawal_tested": True}, "governance_sha256"
            ),
            "dry_run.json": seal_object(
                {"synthetic_data_only": True}, "dry_run_sha256"
            ),
            "reproduction_readiness.json": seal_object(
                {"independent_organization_count": 1}, "reproduction_sha256"
            ),
        }
        snapshot = seal_object(
            {"preregistration_receipt_sha256": "1" * 64},
            "snapshot_sha256",
        )
        values["confirmatory_amendments.json"] = seal_object(
            {"baseline_authority": snapshot, "amendments": []},
            "ledger_sha256",
        )
        checks = [
            {"gate": "WorkspaceValidation", "blocking": True, "passed": True}
        ]
        values["confirmatory_readiness.json"] = seal_object(
            {
                "checks": checks,
                "decision": "ReadyForConfirmatoryCollection",
            },
            "report_sha256",
        )
        packages = [seal_object({"reviewer": "one"}, "package_sha256")]
        responses = [seal_object({"reviewer_id": "one"}, "response_sha256")]
        for filename, value in values.items():
            (root / filename).write_text(json.dumps(value), encoding="utf-8")
        (root / "review_packages.json").write_text(
            json.dumps(packages), encoding="utf-8"
        )
        (root / "review_responses.json").write_text(
            json.dumps(responses), encoding="utf-8"
        )
        release = {
            field: digest(values[filename]) for field, filename in FILES.items()
        }
        release.update(
            {
                "review_package_set_sha256": digest(
                    sorted(package["package_sha256"] for package in packages)
                ),
                "review_response_set_sha256": digest(
                    sorted(response["response_sha256"] for response in responses)
                ),
                "source_archive_sha256": "2" * 64,
                "flake_lock_sha256": "3" * 64,
                "toolchain_evidence_sha256": "4" * 64,
                "external_timestamp_receipt_sha256": "5" * 64,
            }
        )
        release = seal_object(release, "release_sha256")
        (root / "confirmatory_readiness_release.json").write_text(
            json.dumps(release), encoding="utf-8"
        )
        assert verify_root(root) == []
        values["dry_run.json"]["synthetic_data_only"] = False
        (root / "dry_run.json").write_text(
            json.dumps(values["dry_run.json"]), encoding="utf-8"
        )
        assert verify_root(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("V11 verifier self-test passed")
        return 0
    if args.root is None:
        parser.error("ROOT is required unless --self-test is used")
    errors = verify_root(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("V11 confirmatory readiness evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
