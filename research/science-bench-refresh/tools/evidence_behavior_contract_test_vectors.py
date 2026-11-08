#!/usr/bin/env python3
"""Test-only generator for end-to-end evidence behavior contract CLI controls.

This script creates only synthetic artifacts. It is not a production runner,
does not call an evaluated model, and must never be used as scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"
PLAN_SCHEMA = "symthaea.science_bench_refresh_evidence_behavior_contract_plan.v1"
REPORT_SCHEMA = "symthaea.science_bench_refresh_evidence_behavior_contract_report.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def run_cli(
    manifest: Path,
    args: list[str],
    expected_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    command = [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expected_exit:
        raise RuntimeError(
            f"command {args[0]} exited {result.returncode}, expected {expected_exit}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed: dict[str, Any] | None = None
    if result.stdout.strip():
        try:
            candidate = json.loads(result.stdout)
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{args[0]} emitted non-JSON stdout: {error}") from error
    return result, parsed


def plan_items() -> list[dict[str, Any]]:
    definitions = [
        (
            "update-001",
            "update_compliance",
            "coherent_update",
            "dependent_claims_update_without_unrelated_drift",
            "not_applicable",
        ),
        (
            "conflict-001",
            "conflict_disclosure",
            "source_conflict",
            "explicit_conflict_and_provenance",
            "not_applicable",
        ),
        (
            "decoy-001",
            "decoy_invariance",
            "irrelevant_decoy",
            "supported_claims_stable",
            "not_applicable",
        ),
        (
            "deletion-001",
            "deletion_calibration",
            "key_evidence_deletion",
            "claim_strength_tracks_remaining_support",
            "not_applicable",
        ),
        (
            "permutation-001",
            "permutation_invariance",
            "evidence_permutation",
            "semantic_answer_invariant",
            "not_applicable",
        ),
        (
            "force-001",
            "evidence_force_monotonicity",
            "evidence_strength_reduction",
            "weaker_evidence_not_stronger_claim",
            "numeric_specificity",
        ),
        (
            "completeness-001",
            "claim_evidence_completeness",
            "joint_claim_evidence_audit",
            "all_material_claims_warranted",
            "not_applicable",
        ),
    ]
    items: list[dict[str, Any]] = []
    for case_id, family, intervention, behavior, force_axis in definitions:
        items.append(
            {
                "case_id": case_id,
                "pair_id": f"pair-{case_id}",
                "family": family,
                "intervention": intervention,
                "baseline_condition": f"{case_id}-baseline",
                "intervention_condition": f"{case_id}-intervention",
                "expected_behavior": behavior,
                "force_axis": force_axis,
                "decision_rule_id": f"rule-{family}-v1",
                "evidence_manifest_sha256": digest(f"manifest:{case_id}"),
                "evaluator_gold_sha256": digest(f"gold:{case_id}"),
            }
        )
    return items


def outcome_records() -> list[dict[str, Any]]:
    definitions = [
        ("update-001", "pass", [], None),
        ("conflict-001", "violation", ["silent_conflict"], None),
        ("decoy-001", "pass", [], None),
        (
            "deletion-001",
            "not_evaluable",
            [],
            "Synthetic negative control: gold sufficiency intentionally withheld.",
        ),
        ("permutation-001", "pass", [], None),
        ("force-001", "violation", ["unsupported_force_increase"], None),
        ("completeness-001", "pass", [], None),
    ]
    records: list[dict[str, Any]] = []
    for case_id, status, violation_codes, reason in definitions:
        unavailable = status == "not_evaluable"
        records.append(
            {
                "case_id": case_id,
                "status": status,
                "baseline_response_artifact_sha256": (
                    None if unavailable else digest(f"baseline-response:{case_id}")
                ),
                "intervention_response_artifact_sha256": (
                    None if unavailable else digest(f"intervention-response:{case_id}")
                ),
                "adjudication_record_sha256": digest(f"adjudication:{case_id}"),
                "violation_codes": violation_codes,
                "not_evaluable_reason": reason,
            }
        )
    return records


def require_invalid_plan(
    manifest: Path,
    path: Path,
    label: str,
) -> dict[str, Any]:
    _, report = run_cli(
        manifest,
        ["verify-evidence-behavior-contract-plan", str(path)],
        expected_exit=11,
    )
    if report is None or report.get("valid") is not False:
        raise RuntimeError(f"{label} was not rejected as an invalid plan")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    source_revision = os.environ.get("SBR_GENERATOR_SOURCE_REVISION", "")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if source_revision != git_revision:
        raise RuntimeError(
            "test-vector revision must equal the exact checked-out source revision"
        )
    if len(source_revision) != 40 or set(source_revision) == {"0"}:
        raise RuntimeError("test-vector source revision is not a valid full Git SHA")

    fields = {
        "schema": PLAN_SCHEMA,
        "purpose": "qualified_run",
        "contract_set_id": "synthetic-cli-positive-control-v1",
        "evaluator_source_revision": source_revision,
        "source_authority_policy_sha256": digest("synthetic-authority-policy"),
        "decision_rules_sha256": digest("synthetic-decision-rules"),
        "items": plan_items(),
    }
    fields_path = out / "plan-fields.json"
    write_json(fields_path, fields)
    _, commit_result = run_cli(
        args.manifest_path,
        ["commit-evidence-behavior-contract-plan", str(fields_path)],
    )
    if commit_result is None:
        raise RuntimeError("plan commitment command returned no JSON")
    plan = dict(fields)
    plan["plan_commitment_sha256"] = commit_result["plan_commitment_sha256"]
    plan_path = out / "plan.json"
    plan_bytes = write_json(plan_path, plan)

    _, plan_report = run_cli(
        args.manifest_path,
        ["verify-evidence-behavior-contract-plan", str(plan_path)],
    )
    if plan_report is None or plan_report.get("valid") is not True:
        raise RuntimeError(f"qualified positive-control plan did not verify: {plan_report}")

    report_fields = {
        "schema": REPORT_SCHEMA,
        "purpose": "qualified_run",
        "plan_sha256": sha256_bytes(plan_bytes),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "evaluator_source_revision": source_revision,
        "decision_rules_sha256": fields["decision_rules_sha256"],
        "adjudication_mode": "deterministic_synthetic_oracle",
        "blind_to_system_identity": True,
        "records": outcome_records(),
    }
    report_fields_path = out / "report-fields.json"
    write_json(report_fields_path, report_fields)
    _, report_commit = run_cli(
        args.manifest_path,
        [
            "commit-evidence-behavior-contract-report",
            str(plan_path),
            str(report_fields_path),
        ],
    )
    if report_commit is None:
        raise RuntimeError("report commitment command returned no JSON")
    report = dict(report_fields)
    report["report_commitment_sha256"] = report_commit["report_commitment_sha256"]
    report_path = out / "report.json"
    write_json(report_path, report)

    first_result, verified = run_cli(
        args.manifest_path,
        [
            "verify-evidence-behavior-contract-report",
            str(plan_path),
            str(report_path),
        ],
    )
    second_result, _ = run_cli(
        args.manifest_path,
        [
            "verify-evidence-behavior-contract-report",
            str(plan_path),
            str(report_path),
        ],
    )
    if verified is None or verified.get("valid") is not True:
        raise RuntimeError(f"qualified synthetic report failed verification: {verified}")
    if first_result.stdout != second_result.stdout:
        raise RuntimeError("repeat verification was not byte-deterministic")

    scorecard = verified.get("scorecard") or {}
    expected_counts = {
        "pass_count": 4,
        "violation_count": 2,
        "not_evaluable_count": 1,
        "planned_case_count": 7,
        "observed_record_count": 7,
        "missing_case_count": 0,
    }
    for key, value in expected_counts.items():
        if scorecard.get(key) != value:
            raise RuntimeError(
                f"scorecard {key}={scorecard.get(key)!r}, expected {value!r}"
            )
    if scorecard.get("runner_execution_proven") is not False:
        raise RuntimeError("synthetic receipt falsely claims runner execution proof")
    if scorecard.get("external_anchor_required") is not True:
        raise RuntimeError("qualified report must require an external pre-run anchor")
    families = scorecard.get("family_summaries", [])
    if len(families) != 7 or any(item.get("planned_cases") != 1 for item in families):
        raise RuntimeError("scorecard did not preserve all seven family denominators")

    # A modified typed plan must fail both structural and commitment checks.
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["items"][0]["expected_behavior"] = "supported_claims_stable"
    tampered_plan_path = out / "tampered-plan.json"
    write_json(tampered_plan_path, tampered_plan)
    require_invalid_plan(args.manifest_path, tampered_plan_path, "tampered plan")

    # A plan cannot be retargeted to another evaluator source revision.
    wrong_revision = json.loads(plan_path.read_text(encoding="utf-8"))
    wrong_revision["evaluator_source_revision"] = "a" * 40
    wrong_revision_path = out / "wrong-revision-plan.json"
    write_json(wrong_revision_path, wrong_revision)
    require_invalid_plan(args.manifest_path, wrong_revision_path, "wrong-revision plan")

    # A duplicate record also makes a planned case missing; the commit CLI rejects it.
    duplicate_fields = json.loads(report_fields_path.read_text(encoding="utf-8"))
    duplicate_fields["records"][1]["case_id"] = duplicate_fields["records"][0]["case_id"]
    duplicate_path = out / "duplicate-report-fields.json"
    write_json(duplicate_path, duplicate_fields)
    run_cli(
        args.manifest_path,
        [
            "commit-evidence-behavior-contract-report",
            str(plan_path),
            str(duplicate_path),
        ],
        expected_exit=2,
    )

    missing_fields = json.loads(report_fields_path.read_text(encoding="utf-8"))
    missing_fields["records"].pop()
    missing_path = out / "missing-report-fields.json"
    write_json(missing_path, missing_fields)
    run_cli(
        args.manifest_path,
        [
            "commit-evidence-behavior-contract-report",
            str(plan_path),
            str(missing_path),
        ],
        expected_exit=2,
    )

    # Tampering with the report after commitment must not pass verification.
    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["records"][0]["intervention_response_artifact_sha256"] = digest(
        "tampered-response-reference"
    )
    tampered_report_path = out / "tampered-report.json"
    write_json(tampered_report_path, tampered_report)
    _, rejected_report = run_cli(
        args.manifest_path,
        [
            "verify-evidence-behavior-contract-report",
            str(plan_path),
            str(tampered_report_path),
        ],
        expected_exit=11,
    )
    if rejected_report is None or rejected_report.get("valid") is not False:
        raise RuntimeError("tampered outcome report was not rejected")

    summary = {
        "schema": "symthaea.science_bench_refresh_evidence_behavior_contract_test_summary.v1",
        "source_revision": source_revision,
        "synthetic_positive_control": True,
        "planned_cases": 7,
        "pass_count": scorecard["pass_count"],
        "violation_count": scorecard["violation_count"],
        "not_evaluable_count": scorecard["not_evaluable_count"],
        "repeat_verification_deterministic": True,
        "tampered_plan_rejected": True,
        "wrong_revision_rejected": True,
        "duplicate_and_missing_records_rejected": True,
        "tampered_report_rejected": True,
        "runner_execution_proven": False,
        "external_anchor_required": True,
        "empirical_system_result": False,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"evidence behavior contract test vectors failed: {error}", file=sys.stderr)
        raise SystemExit(1)
