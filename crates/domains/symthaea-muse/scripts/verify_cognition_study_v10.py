#!/usr/bin/env python3
"""Independent V10 pilot-operations and reproducibility verifier.

Uses only the Python standard library. It verifies canonical commitments and the
study lifecycle chain without importing the Rust implementation or private arm
labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def commitment_without(document: dict[str, Any], field: str) -> str:
    return digest({key: value for key, value in document.items() if key != field})


def verify_root(path: Path, field: str) -> list[str]:
    document = load(path)
    expected = document.get(field)
    observed = commitment_without(document, field)
    return [] if expected == observed else [f"{path}: {field} mismatch"]


def transition_commitment(orchestration_id: str, transition: dict[str, Any]) -> str:
    return digest(
        {
            "orchestration_id": orchestration_id,
            "sequence": transition["sequence"],
            "from": transition["from"],
            "to": transition["to"],
            "recorded_at_utc": transition["recorded_at_utc"],
            "operator_id": transition["operator_id"],
            "authorization_sha256": transition["authorization_sha256"],
            "added_authorities": transition["added_authorities"],
            "previous_transition_sha256": transition[
                "previous_transition_sha256"
            ],
        }
    )


def orchestration_commitment(log: dict[str, Any]) -> str:
    return digest(
        {
            "orchestration_version": log["orchestration_version"],
            "orchestration_id": log["orchestration_id"],
            "current_phase": log["current_phase"],
            "authorities": log["authorities"],
            "transitions": log["transitions"],
        }
    )


def verify_orchestration(path: Path) -> list[str]:
    log = load(path)
    issues: list[str] = []
    previous = GENESIS
    phase = "Draft"
    authorities: dict[str, str] = {}
    for index, transition in enumerate(log.get("transitions", []), start=1):
        if transition.get("sequence") != index:
            issues.append(f"transition {index}: sequence mismatch")
        if transition.get("from") != phase:
            issues.append(f"transition {index}: from-phase mismatch")
        if transition.get("previous_transition_sha256") != previous:
            issues.append(f"transition {index}: previous digest mismatch")
        observed = transition_commitment(log["orchestration_id"], transition)
        if observed != transition.get("transition_sha256"):
            issues.append(f"transition {index}: transition digest mismatch")
        for binding in transition.get("added_authorities", []):
            role = binding.get("role")
            sha = binding.get("sha256")
            if role in authorities and authorities[role] != sha:
                issues.append(f"transition {index}: conflicting authority {role}")
            authorities[role] = sha
        previous = transition.get("transition_sha256", "")
        phase = transition.get("to", "")
    if phase != log.get("current_phase"):
        issues.append("current_phase mismatch")
    observed_authorities = {
        binding.get("role"): binding.get("sha256")
        for binding in log.get("authorities", [])
    }
    if observed_authorities != authorities:
        issues.append("authority snapshot mismatch")
    if orchestration_commitment(log) != log.get("log_sha256"):
        issues.append("log_sha256 mismatch")
    return issues


def amendment_commitment(ledger: dict[str, Any]) -> str:
    return digest(
        {
            "ledger_version": ledger["ledger_version"],
            "initial_protocol_sha256": ledger["initial_protocol_sha256"],
            "amendments": ledger["amendments"],
        }
    )


def verify_amendments(path: Path) -> list[str]:
    ledger = load(path)
    issues: list[str] = []
    previous = ledger.get("initial_protocol_sha256")
    for index, amendment in enumerate(ledger.get("amendments", []), start=1):
        if amendment.get("sequence") != index:
            issues.append(f"amendment {index}: sequence mismatch")
        if amendment.get("prior_protocol_sha256") != previous:
            issues.append(f"amendment {index}: protocol chain mismatch")
        if not amendment.get("confirmatory_manifest_unchanged"):
            issues.append(f"amendment {index}: confirmatory manifest changed")
        if not amendment.get("confirmatory_outcomes_uninspected"):
            issues.append(f"amendment {index}: confirmatory outcomes inspected")
        previous = amendment.get("amended_protocol_sha256")
    if amendment_commitment(ledger) != ledger.get("ledger_sha256"):
        issues.append("ledger_sha256 mismatch")
    return issues


def analysis_commitment(analysis: dict[str, Any]) -> str:
    return digest(
        {
            "engine_kind": analysis["engine_kind"],
            "engine_name": analysis["engine_name"],
            "engine_version": analysis["engine_version"],
            "source_sha256": analysis["source_sha256"],
            "environment_sha256": analysis["environment_sha256"],
            "input_sha256": analysis["input_sha256"],
            "analysis_plan_sha256": analysis["analysis_plan_sha256"],
            "endpoint": analysis["endpoint"],
            "alpha": analysis["alpha"],
            "comparisons": analysis["comparisons"],
            "success": analysis["success"],
        }
    )


def verify_analysis(path: Path) -> list[str]:
    analysis = load(path)
    return (
        []
        if analysis_commitment(analysis) == analysis.get("output_sha256")
        else [f"{path}: output_sha256 mismatch"]
    )


def crosscheck_commitment(report: dict[str, Any]) -> str:
    return digest(
        {
            "crosscheck_version": report["crosscheck_version"],
            "rust_output_sha256": report["rust_output_sha256"],
            "external_output_sha256": report["external_output_sha256"],
            "tolerance": report["tolerance"],
            "agreements": report["agreements"],
            "exact_identity_fields_match": report["exact_identity_fields_match"],
            "success_decision_matches": report["success_decision_matches"],
            "passed": report["passed"],
        }
    )


def verify_crosscheck(path: Path) -> list[str]:
    report = load(path)
    issues: list[str] = []
    if crosscheck_commitment(report) != report.get("report_sha256"):
        issues.append("crosscheck report_sha256 mismatch")
    if report.get("passed") and not all(
        agreement.get("within_tolerance") for agreement in report.get("agreements", [])
    ):
        issues.append("crosscheck passed despite comparator disagreement")
    return issues


def operations_release_commitment(bundle: dict[str, Any]) -> str:
    return commitment_without(bundle, "release_sha256")


def verify_operations_release(path: Path) -> list[str]:
    bundle = load(path)
    issues: list[str] = []
    if operations_release_commitment(bundle) != bundle.get("release_sha256"):
        issues.append("operations release_sha256 mismatch")
    for key, value in bundle.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64):
            issues.append(f"invalid digest field: {key}")
    return issues


def self_test() -> list[str]:
    issues: list[str] = []
    known = digest({"b": 2, "a": 1})
    expected = "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    if known != expected:
        issues.append(f"canonical digest mismatch: {known}")
    log = {
        "orchestration_version": "test",
        "orchestration_id": "study",
        "current_phase": "Draft",
        "authorities": [],
        "transitions": [],
        "log_sha256": "",
    }
    log["log_sha256"] = orchestration_commitment(log)
    if verify_orchestration_dict(log):
        issues.append("orchestration self-test failed")
    return issues


def verify_orchestration_dict(log: dict[str, Any]) -> list[str]:
    if orchestration_commitment(log) != log.get("log_sha256"):
        return ["log_sha256 mismatch"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "pilot-protocol",
        "pilot-schedule",
        "cohort-registry",
        "pilot-collection",
        "pilot-snapshot",
        "pilot-report",
        "reproduction-attestation",
    ):
        command = sub.add_parser(name)
        command.add_argument("document", type=Path)
    amendments = sub.add_parser("amendments")
    amendments.add_argument("ledger", type=Path)
    orchestration = sub.add_parser("orchestration")
    orchestration.add_argument("log", type=Path)
    analysis = sub.add_parser("analysis")
    analysis.add_argument("result", type=Path)
    crosscheck = sub.add_parser("crosscheck")
    crosscheck.add_argument("report", type=Path)
    release = sub.add_parser("operations-release")
    release.add_argument("bundle", type=Path)
    sub.add_parser("self-test")
    args = parser.parse_args()
    root_fields = {
        "pilot-protocol": None,
        "pilot-schedule": None,
        "cohort-registry": "registry_sha256",
        "pilot-collection": "collection_sha256",
        "pilot-snapshot": "snapshot_sha256",
        "pilot-report": "report_sha256",
        "reproduction-attestation": "attestation_sha256",
    }
    if args.command in root_fields:
        field = root_fields[args.command]
        if field is None:
            issues = []
        else:
            issues = verify_root(args.document, field)
    elif args.command == "amendments":
        issues = verify_amendments(args.ledger)
    elif args.command == "orchestration":
        issues = verify_orchestration(args.log)
    elif args.command == "analysis":
        issues = verify_analysis(args.result)
    elif args.command == "crosscheck":
        issues = verify_crosscheck(args.report)
    elif args.command == "operations-release":
        issues = verify_operations_release(args.bundle)
    else:
        issues = self_test()
    json.dump(issues, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
