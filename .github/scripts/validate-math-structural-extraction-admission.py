#!/usr/bin/env python3
"""Validate MATH-RET-CONVERGENCE-001D extraction-admission sidecars."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_GATES = {
    "representation_execution": {
        "subject_commit": "694ef48e53296c2c6fe39a9c07927085b2f2754f",
        "workflow_run_id": 35459503395,
        "workflow_blob": "f021a3f40164280d63e3467bb96c82ebe17ab9e1",
        "conclusion": "success",
    },
    "receipt_contract": {
        "subject_commit": "9b8f6d911a693090db4c1079f97d45f5fda57dcf",
        "workflow_run_id": 35463362013,
        "workflow_blob": "9532dac2f8e1bcde22cc17c4935d90e56b70b8ba",
        "conclusion": "success",
    },
    "compatibility_wire": {
        "subject_commit": "bdcf7203f4d077eb8fc52f5a67b8d9aefb49fd77",
        "workflow_run_id": 35463731810,
        "workflow_blob": "7d5815fca9ae8e0ec2bdcb2b4c6bff98ceb90224",
        "conclusion": "success",
        "profile": "math-structural-compat-wire-v1",
        "vectors_blob": "b44eef6240e977e94744ee94e2e538576a656881",
        "python_oracle_blob": "cb601ce5ec1ad0ea9fbd879a67bd1c1c9b95bdb7",
    },
    "rust_wire_canary": {
        "subject_commit": "556b8e71c4c596484b6cf0d4b0b26f6f03a0291d",
        "workflow_run_id": 35463948360,
        "workflow_blob": "01eb0010c76a9883b6d5692b32aa936e5d7e09df",
        "conclusion": "success",
        "rust_source_blob": "a9465e952eb537c26d7c352a8f0c1914a3ddae81",
    },
}

FORBIDDEN_AUTHORITY_KEYS = {
    "truth_value",
    "formal_authority",
    "theorem_authority",
    "proof_valid",
    "epistemic_confidence",
    "hdc_advantage",
    "production_ready",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def exact_keys(obj: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(obj)
    require(
        actual == expected,
        f"{where}: keys differ; missing={sorted(expected-actual)} extra={sorted(actual-expected)}",
    )


def reject_authority_smuggling(value: Any, where: str = "root") -> None:
    if isinstance(value, dict):
        bad = FORBIDDEN_AUTHORITY_KEYS.intersection(value)
        require(not bad, f"{where}: forbidden authority fields {sorted(bad)}")
        for key, child in value.items():
            reject_authority_smuggling(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_authority_smuggling(child, f"{where}[{index}]")


def validate(admission: dict[str, Any]) -> None:
    reject_authority_smuggling(admission)
    exact_keys(
        admission,
        {
            "version",
            "authority",
            "admission_kind",
            "extraction_receipt",
            "gates",
            "reconstruction",
            "holdout_firewall",
            "evidence_refs",
        },
        "admission",
    )
    require(admission["version"] == "math-structural-extraction-admission-v1", "version mismatch")
    require(admission["authority"] == "MeasurementOnly", "authority must remain MeasurementOnly")
    require(admission["admission_kind"] == "AllPrerequisitesQualified", "admission kind mismatch")

    receipt = admission["extraction_receipt"]
    exact_keys(receipt, {"version", "sha256", "wire_profile"}, "extraction_receipt")
    require(receipt["version"] == "math-structural-extraction-receipt-v1", "receipt version mismatch")
    require(isinstance(receipt["sha256"], str) and SHA256.fullmatch(receipt["sha256"]), "receipt sha256 invalid")
    require(receipt["wire_profile"] == "math-structural-compat-wire-v1", "receipt is not bound to frozen wire profile")

    gates = admission["gates"]
    exact_keys(gates, set(EXPECTED_GATES), "gates")
    for name, expected in EXPECTED_GATES.items():
        gate = gates[name]
        exact_keys(gate, set(expected), f"gates.{name}")
        require(gate == expected, f"gates.{name}: exact prerequisite binding mismatch")

    reconstruction = admission["reconstruction"]
    exact_keys(
        reconstruction,
        {"target_main_commit", "source_objects_reverified", "environment_rechecked", "merge_frozen_draft_branches"},
        "reconstruction",
    )
    require(
        isinstance(reconstruction["target_main_commit"], str)
        and SHA40.fullmatch(reconstruction["target_main_commit"]),
        "target_main_commit must be 40 lowercase hex",
    )
    require(reconstruction["source_objects_reverified"] is True, "source objects must be reverified")
    require(reconstruction["environment_rechecked"] is True, "environment must be rechecked on then-current main")
    require(reconstruction["merge_frozen_draft_branches"] is False, "frozen draft branches must be reconstructed, not merged wholesale")

    firewall = admission["holdout_firewall"]
    exact_keys(firewall, {"ranking_evaluated", "scores_emitted", "labels_exposed_to_extraction"}, "holdout_firewall")
    require(firewall == {
        "ranking_evaluated": False,
        "scores_emitted": False,
        "labels_exposed_to_extraction": False,
    }, "holdout firewall violated")

    refs = admission["evidence_refs"]
    require(isinstance(refs, list) and len(refs) >= 5, "at least five evidence refs required")
    require(all(isinstance(ref, str) and ref.strip() for ref in refs), "evidence refs must be non-empty strings")
    require(len(refs) == len(set(refs)), "evidence refs must be unique")


def valid_example() -> dict[str, Any]:
    return {
        "version": "math-structural-extraction-admission-v1",
        "authority": "MeasurementOnly",
        "admission_kind": "AllPrerequisitesQualified",
        "extraction_receipt": {
            "version": "math-structural-extraction-receipt-v1",
            "sha256": "ab" * 32,
            "wire_profile": "math-structural-compat-wire-v1",
        },
        "gates": copy.deepcopy(EXPECTED_GATES),
        "reconstruction": {
            "target_main_commit": "1" * 40,
            "source_objects_reverified": True,
            "environment_rechecked": True,
            "merge_frozen_draft_branches": False,
        },
        "holdout_firewall": {
            "ranking_evaluated": False,
            "scores_emitted": False,
            "labels_exposed_to_extraction": False,
        },
        "evidence_refs": [
            "github-actions://representation-execution",
            "github-actions://receipt-contract",
            "github-actions://compatibility-wire",
            "github-actions://rust-wire-canary",
            "artifact://extraction-receipt",
        ],
    }


def expect_reject(mutate, label: str) -> None:
    value = valid_example()
    mutate(value)
    try:
        validate(value)
    except ValidationError:
        return
    raise AssertionError(f"negative self-test unexpectedly accepted: {label}")


def self_test() -> None:
    validate(valid_example())
    expect_reject(lambda a: a.__setitem__("authority", "FormalAuthority"), "authority escalation")
    expect_reject(lambda a: a["extraction_receipt"].__setitem__("wire_profile", "other"), "wire substitution")
    expect_reject(lambda a: a["gates"]["representation_execution"].__setitem__("conclusion", "queued"), "queued predecessor")
    expect_reject(lambda a: a["gates"]["receipt_contract"].__setitem__("workflow_run_id", 1), "run substitution")
    expect_reject(lambda a: a["gates"]["compatibility_wire"].__setitem__("vectors_blob", "0" * 40), "vector substitution")
    expect_reject(lambda a: a["gates"]["rust_wire_canary"].__setitem__("rust_source_blob", "0" * 40), "Rust canary substitution")
    expect_reject(lambda a: a["reconstruction"].__setitem__("source_objects_reverified", False), "unverified source objects")
    expect_reject(lambda a: a["reconstruction"].__setitem__("merge_frozen_draft_branches", True), "wholesale draft merge")
    expect_reject(lambda a: a["holdout_firewall"].__setitem__("labels_exposed_to_extraction", True), "holdout label leak")
    expect_reject(lambda a: a.__setitem__("truth_value", True), "truth authority smuggling")
    print("PASS: math structural extraction admission v1 semantic self-tests")


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        if args.admission:
            validate(load(args.admission))
            print(f"PASS: {args.admission}")
        if not args.self_test and not args.admission:
            raise ValidationError("provide --self-test and/or --admission")
        return 0
    except (ValidationError, AssertionError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
