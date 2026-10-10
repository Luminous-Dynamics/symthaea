#!/usr/bin/env python3
"""Validate the public Cyber Crucible v1 manifest without third-party packages.

This is a semantic contract check, not a benchmark run or a security verdict.
The corpus is intentionally public calibration data and MUST NOT be described
as held-out qualification evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "validation" / "cyber_crucible_public_v1.json"
EXPECTED_SCHEMA_VERSION = "cyber-crucible-public-calibration.v1"
EXPECTED_CORRECT_AND_SECURE_REQUIRES = [
    "functional_status=pass",
    "security_status=pass",
    "execution_mode=real",
    "required_evidence_checks=passed",
]
VALID_STATUSES = ["pass", "fail", "inconclusive", "not_run"]
VALID_BASES = {"direct", "control_plane_record", "derived", "flow_derived"}


def canonical_scenario_bytes(scenario: dict[str, Any]) -> bytes:
    """Canonical JSON used for scenario_digest; digest excludes its own field."""
    payload = copy.deepcopy(scenario)
    payload.pop("scenario_digest", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append("schema_version must be the supported v1 identifier")
    if manifest.get("lane") != "public_calibration":
        errors.append("lane must remain public_calibration")
    if manifest.get("non_authorizing") is not True:
        errors.append("corpus must remain explicitly non-authorizing")

    policy = manifest.get("scoring_contract", {})
    if policy.get("functional_status_values") != VALID_STATUSES:
        errors.append("functional statuses must preserve the v1 enum")
    if policy.get("security_status_values") != VALID_STATUSES:
        errors.append("security statuses must be separately declared with the v1 enum")
    if policy.get("correct_and_secure_requires") != EXPECTED_CORRECT_AND_SECURE_REQUIRES:
        errors.append("correct_and_secure must require both gates, real execution, and evidence")
    if policy.get("simulated_execution_authorizes_pass") is not False:
        errors.append("simulated execution must never authorize a pass")
    if policy.get("diagnosis_grants_execution_authority") is not False:
        errors.append("diagnosis must not grant live-system execution authority")

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < 3:
        errors.append("v1 corpus must include at least three calibration scenarios")
        return errors

    seen_ids: set[str] = set()
    for index, scenario in enumerate(scenarios):
        label = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            errors.append(f"{label} must be an object")
            continue

        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            errors.append(f"{label}.scenario_id must be non-empty")
        elif scenario_id in seen_ids:
            errors.append(f"duplicate scenario_id: {scenario_id}")
        else:
            seen_ids.add(scenario_id)

        if not isinstance(scenario.get("revision"), int) or scenario["revision"] < 1:
            errors.append(f"{label}.revision must be a positive integer")

        supplied_digest = scenario.get("scenario_digest")
        if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
            errors.append(f"{label}.scenario_digest must be a 64-character SHA-256 hex digest")
        elif any(ch not in "0123456789abcdef" for ch in supplied_digest):
            errors.append(f"{label}.scenario_digest must be lowercase hexadecimal")
        else:
            expected_digest = hashlib.sha256(canonical_scenario_bytes(scenario)).hexdigest()
            if supplied_digest != expected_digest:
                errors.append(
                    f"{label} digest mismatch: expected {expected_digest}, got {supplied_digest}"
                )

        visible = scenario.get("solver_visible", {})
        evidence = visible.get("evidence", []) if isinstance(visible, dict) else []
        if not isinstance(evidence, list) or len(evidence) < 2:
            errors.append(f"{label} must provide at least two solver-visible evidence records")
            evidence = []

        evidence_ids: set[str] = set()
        required_ids: set[str] = set()
        for evidence_index, item in enumerate(evidence):
            item_label = f"{label}.solver_visible.evidence[{evidence_index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_label} must be an object")
                continue
            evidence_id = item.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                errors.append(f"{item_label}.evidence_id must be non-empty")
            elif evidence_id in evidence_ids:
                errors.append(f"{label} has duplicate evidence ID {evidence_id}")
            else:
                evidence_ids.add(evidence_id)
            if item.get("basis") not in VALID_BASES:
                errors.append(f"{item_label}.basis is not a supported evidence basis")
            if item.get("required_for_claim") is True and isinstance(evidence_id, str):
                required_ids.add(evidence_id)

        oracle = scenario.get("oracle", {})
        if not isinstance(oracle, dict) or oracle.get("visibility") != "public_training":
            errors.append(f"{label} oracle visibility must say public_training explicitly")
            oracle = {}
        refs = oracle.get("required_evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            errors.append(f"{label}.oracle.required_evidence_refs must be a string list")
            refs = []
        if set(refs) != required_ids:
            errors.append(
                f"{label} required_evidence_refs must exactly match records marked required_for_claim"
            )
        if not set(refs).issubset(evidence_ids):
            errors.append(f"{label} required evidence refs must resolve to visible evidence IDs")
        alternatives = oracle.get("plausible_alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) < 2:
            errors.append(f"{label} must preserve at least two competing hypotheses")

        evaluation = scenario.get("evaluation", {})
        if not isinstance(evaluation, dict):
            errors.append(f"{label}.evaluation must be an object")
            continue
        for key in (
            "functional_requirements",
            "security_requirements",
            "allowed_observation_classes",
            "allowed_action_classes",
            "forbidden_action_classes",
        ):
            value = evaluation.get(key)
            if not isinstance(value, list) or not value or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                errors.append(f"{label}.evaluation.{key} must be a non-empty string list")
        if not evaluation.get("claim_maturity_ceiling"):
            errors.append(f"{label} must declare a claim maturity ceiling")
        control = evaluation.get("positive_control", {})
        if not isinstance(control, dict) or control.get("capability_status") not in {
            "established",
            "not_established",
            "ambiguous",
        }:
            errors.append(f"{label} must carry an explicit positive-control status")

    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read JSON manifest {path}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(manifest, dict):
        print("FAIL: manifest root must be a JSON object", file=sys.stderr)
        return 2

    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"FAIL: {len(errors)} contract violation(s)", file=sys.stderr)
        return 1

    print(
        f"PASS: {manifest['corpus_id']} revision {manifest['revision']} "
        f"({len(manifest['scenarios'])} public calibration scenarios); "
        "structural checks only, not qualification evidence"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
