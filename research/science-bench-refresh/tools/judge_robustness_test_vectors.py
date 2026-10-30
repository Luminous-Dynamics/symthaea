#!/usr/bin/env python3
"""Synthetic end-to-end controls for the judge robustness audit CLI.

This creates no model result. It validates the audit contract, scorecard math,
repeatability controls and fail-closed artifact accounting on synthetic decisions.
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
PLAN_SCHEMA = "symthaea.science_bench_refresh_judge_robustness_plan.v1"
REPORT_SCHEMA = "symthaea.science_bench_refresh_judge_robustness_report.v1"
ZERO = "0" * 64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def run_cli(
    manifest: Path, args: list[str], expected_exit: int = 0
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    command = [
        "cargo", "run", "--locked", "--quiet", "--manifest-path",
        str(manifest), "--", *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expected_exit:
        raise RuntimeError(
            f"{args[0]} exited {result.returncode}, expected {expected_exit}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed = None
    if result.stdout.strip():
        try:
            candidate = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{args[0]} emitted non-JSON stdout: {error}") from error
        if isinstance(candidate, dict):
            parsed = candidate
    return result, parsed


def case_definitions() -> list[dict[str, Any]]:
    def case(
        case_id: str,
        family: str,
        baseline_input: str,
        variant_input: str,
        baseline_order: str,
        variant_order: str,
        baseline_gold: str = "candidate_a",
        variant_gold: str = "candidate_a",
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "family": family,
            "expected_relation": (
                "semantic_decision_must_change"
                if family == "semantic_sensitivity_control"
                else "semantic_decision_invariant"
            ),
            "baseline_input_sha256": digest(baseline_input),
            "variant_input_sha256": digest(variant_input),
            "baseline_order": baseline_order,
            "variant_order": variant_order,
            "reference_baseline_decision_sha256": digest(baseline_gold),
            "reference_variant_decision_sha256": digest(variant_gold),
        }

    return [
        case("repeat-001", "repeatability", "same-input", "same-input",
             "candidate_a_first", "candidate_a_first"),
        case("position-001", "position_swap", "a-first-input", "b-first-input",
             "candidate_a_first", "candidate_b_first"),
        case("format-001", "formatting_invariance", "plain-format", "structured-format",
             "candidate_a_first", "candidate_a_first"),
        case("confidence-001", "confidence_style_invariance", "neutral-wording", "confident-wording",
             "candidate_a_first", "candidate_a_first"),
        case("authority-001", "authority_cue_invariance", "plain-source", "fabricated-authority-cue",
             "candidate_a_first", "candidate_a_first"),
        case("sensitivity-001", "semantic_sensitivity_control", "evidence-for-a", "evidence-for-b",
             "candidate_a_first", "candidate_a_first", "candidate_a", "candidate_b"),
    ]


def raw_decision_for(winner: str, order: str) -> str:
    if winner == "tie":
        return "tie"
    if winner == "abstain":
        return "abstain"
    winner_on_left = (winner == "candidate_a") == (order == "candidate_a_first")
    return "left" if winner_on_left else "right"


def report_cases(plan: dict[str, Any], variant_bias_case: str | None = None) -> list[dict[str, Any]]:
    records = []
    for case in plan["cases"]:
        family = case["family"]
        gold_baseline = (
            "candidate_a" if family != "semantic_sensitivity_control" else "candidate_a"
        )
        gold_variant = (
            "candidate_b" if family == "semantic_sensitivity_control" else "candidate_a"
        )
        observations = []
        for condition in ("baseline", "variant"):
            for repeat_index in range(plan["repeats_per_condition"]):
                is_variant = condition == "variant"
                order = case["variant_order"] if is_variant else case["baseline_order"]
                winner = gold_variant if is_variant else gold_baseline
                # Synthetic bias canary: a stable judge changes its semantic
                # choice only when the confidence wording is made more assertive.
                if variant_bias_case == case["case_id"] and is_variant:
                    winner = "candidate_b"
                observations.append({
                    "condition": condition,
                    "repeat_index": repeat_index,
                    "input_sha256": (
                        case["variant_input_sha256"] if is_variant
                        else case["baseline_input_sha256"]
                    ),
                    "candidate_order": order,
                    "raw_decision": raw_decision_for(winner, order),
                    "judge_receipt_sha256": digest(
                        f"{case['case_id']}:{condition}:{repeat_index}:{winner}:{order}"
                    ),
                })
        records.append({
            "case_id": case["case_id"],
            "reference_baseline_decision": gold_baseline,
            "reference_variant_decision": gold_variant,
            "observations": observations,
        })
    return records


def require_invalid_plan(manifest: Path, path: Path, label: str) -> dict[str, Any]:
    _, result = run_cli(
        manifest, ["verify-judge-robustness-plan", str(path)], expected_exit=11
    )
    if result is None or result.get("valid") is not False:
        raise RuntimeError(f"{label} was not rejected as an invalid plan")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    revision = os.environ.get("SBR_GENERATOR_SOURCE_REVISION", "")
    checked_out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != checked_out or len(revision) != 40 or set(revision) == {"0"}:
        raise RuntimeError("judge source revision must match the exact checked-out commit")

    fields = {
        "schema": PLAN_SCHEMA,
        "purpose": "qualified_run",
        "audit_set_id": "synthetic-judge-audit-positive-control-v1",
        "judge_source_revision": revision,
        "judge_prompt_sha256": digest("synthetic-judge-prompt-v1"),
        "rubric_sha256": digest("synthetic-judge-rubric-v1"),
        "decision_policy_sha256": digest("synthetic-decision-policy-v1"),
        "repeats_per_condition": 3,
        "cases": case_definitions(),
    }
    fields_path = out / "plan-fields.json"
    write_json(fields_path, fields)
    _, commit = run_cli(
        args.manifest_path, ["commit-judge-robustness-plan", str(fields_path)]
    )
    if commit is None:
        raise RuntimeError("plan commit command returned no JSON")
    plan = dict(fields)
    plan["plan_commitment_sha256"] = commit["plan_commitment_sha256"]
    plan_path = out / "plan.json"
    plan_bytes = write_json(plan_path, plan)

    _, plan_verification = run_cli(
        args.manifest_path, ["verify-judge-robustness-plan", str(plan_path)]
    )
    if plan_verification is None or plan_verification.get("valid") is not True:
        raise RuntimeError(f"qualified judge plan did not verify: {plan_verification}")
    if plan_verification.get("planned_observation_count") != 36:
        raise RuntimeError("plan observation denominator should be 36")

    report_fields = {
        "schema": REPORT_SCHEMA,
        "purpose": "qualified_run",
        "plan_sha256": sha256_bytes(plan_bytes),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "judge_source_revision": revision,
        "judge_prompt_sha256": fields["judge_prompt_sha256"],
        "rubric_sha256": fields["rubric_sha256"],
        "decision_policy_sha256": fields["decision_policy_sha256"],
        "blind_to_system_identity": True,
        "cases": report_cases(plan, variant_bias_case="confidence-001"),
    }
    report_fields_path = out / "report-fields.json"
    write_json(report_fields_path, report_fields)
    _, report_commit = run_cli(
        args.manifest_path,
        ["commit-judge-robustness-report", str(plan_path), str(report_fields_path)],
    )
    if report_commit is None:
        raise RuntimeError("report commit command returned no JSON")
    report = dict(report_fields)
    report["report_commitment_sha256"] = report_commit["report_commitment_sha256"]
    report_path = out / "report.json"
    write_json(report_path, report)

    first, verified = run_cli(
        args.manifest_path,
        ["verify-judge-robustness-report", str(plan_path), str(report_path)],
    )
    second, _ = run_cli(
        args.manifest_path,
        ["verify-judge-robustness-report", str(plan_path), str(report_path)],
    )
    if verified is None or verified.get("valid") is not True:
        raise RuntimeError(f"synthetic qualified judge report did not verify: {verified}")
    if first.stdout != second.stdout:
        raise RuntimeError("repeat verification was not byte-deterministic")

    scorecard = verified.get("scorecard") or {}
    expected = {
        "planned_case_count": 6,
        "observed_case_count": 6,
        "missing_case_count": 0,
        "planned_observation_count": 36,
        "observed_observation_count": 36,
        "missing_observation_count": 0,
        "runner_execution_proven": False,
        "external_anchor_required": True,
    }
    for name, value in expected.items():
        if scorecard.get(name) != value:
            raise RuntimeError(f"scorecard {name}={scorecard.get(name)!r}, expected {value!r}")

    summaries = {item["family"]: item for item in scorecard.get("family_summaries", [])}
    if len(summaries) != 6:
        raise RuntimeError("expected separate metrics for all six judge audit families")
    if summaries["repeatability"]["repeat_disagreement_rate_millionths"] != 0:
        raise RuntimeError("clean repeatability control should have zero disagreement")
    if summaries["position_swap"]["relation_violation_count"] != 0:
        raise RuntimeError("candidate-identity-normalized position-swap control should pass")
    if summaries["position_swap"]["slot_following_pair_count"] != 0:
        raise RuntimeError("synthetic judge should follow semantic candidate, not raw slot")
    confidence = summaries["confidence_style_invariance"]
    if confidence["relation_violation_count"] != 3:
        raise RuntimeError("confidence-style bias canary should be detected in all three pairs")
    if confidence["gold_agreement_rate_millionths"] != 500_000:
        raise RuntimeError("confidence-style canary should have 50% gold agreement")
    if summaries["semantic_sensitivity_control"]["relation_violation_count"] != 0:
        raise RuntimeError("meaning-changing positive control should change the semantic choice")

    # A modified family/order is rejected even if the attacker leaves the old commitment.
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["cases"][1]["variant_order"] = "candidate_a_first"
    tampered_plan_path = out / "tampered-plan.json"
    write_json(tampered_plan_path, tampered_plan)
    require_invalid_plan(args.manifest_path, tampered_plan_path, "tampered plan")

    wrong_revision = json.loads(plan_path.read_text(encoding="utf-8"))
    wrong_revision["judge_source_revision"] = "a" * 40
    wrong_revision_path = out / "wrong-revision-plan.json"
    write_json(wrong_revision_path, wrong_revision)
    require_invalid_plan(args.manifest_path, wrong_revision_path, "wrong-revision plan")

    duplicate_fields = json.loads(report_fields_path.read_text(encoding="utf-8"))
    duplicate_fields["cases"][0]["observations"][1]["repeat_index"] = 0
    duplicate_path = out / "duplicate-observation-fields.json"
    write_json(duplicate_path, duplicate_fields)
    run_cli(
        args.manifest_path,
        ["commit-judge-robustness-report", str(plan_path), str(duplicate_path)],
        expected_exit=2,
    )

    missing_fields = json.loads(report_fields_path.read_text(encoding="utf-8"))
    missing_fields["cases"][0]["observations"].pop()
    missing_path = out / "missing-observation-fields.json"
    write_json(missing_path, missing_fields)
    run_cli(
        args.manifest_path,
        ["commit-judge-robustness-report", str(plan_path), str(missing_path)],
        expected_exit=2,
    )

    wrong_input_fields = json.loads(report_fields_path.read_text(encoding="utf-8"))
    wrong_input_fields["cases"][0]["observations"][0]["input_sha256"] = digest("wrong-input")
    wrong_input_path = out / "wrong-input-fields.json"
    write_json(wrong_input_path, wrong_input_fields)
    run_cli(
        args.manifest_path,
        ["commit-judge-robustness-report", str(plan_path), str(wrong_input_path)],
        expected_exit=2,
    )

    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["cases"][0]["observations"][0]["raw_decision"] = "right"
    tampered_report_path = out / "tampered-report.json"
    write_json(tampered_report_path, tampered_report)
    _, rejected = run_cli(
        args.manifest_path,
        ["verify-judge-robustness-report", str(plan_path), str(tampered_report_path)],
        expected_exit=11,
    )
    if rejected is None or rejected.get("valid") is not False or rejected.get("scorecard") is not None:
        raise RuntimeError("tampered judge report was not rejected without a scorecard")

    summary = {
        "schema": "symthaea.science_bench_refresh_judge_robustness_test_summary.v1",
        "source_revision": revision,
        "synthetic_positive_control": True,
        "planned_cases": 6,
        "planned_observations": 36,
        "repeat_verification_deterministic": True,
        "repeatability_control_passed": True,
        "position_swap_identity_control_passed": True,
        "confidence_style_bias_detected": True,
        "semantic_sensitivity_control_passed": True,
        "tampered_plan_rejected": True,
        "wrong_revision_rejected": True,
        "duplicate_and_missing_observations_rejected": True,
        "wrong_input_digest_rejected": True,
        "tampered_report_rejected": True,
        "runner_execution_proven": False,
        "external_anchor_required": True,
        "empirical_judge_result": False,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError) as error:
        print(f"judge robustness test vectors failed: {error}", file=sys.stderr)
        raise SystemExit(1)
