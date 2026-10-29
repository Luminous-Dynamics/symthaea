#!/usr/bin/env python3
"""Synthetic end-to-end controls for the Rust judge-reliability audit CLI.

This script is test-only. It never invokes a real model and its outputs are not
scientific evidence about an actual judge or the evaluated Symthaea system.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"
PLAN_SCHEMA = "symthaea.science_bench_refresh_judge_reliability_plan.v1"
REPORT_SCHEMA = "symthaea.science_bench_refresh_judge_reliability_report.v1"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def run_cli(
    manifest: Path,
    args: list[str],
    expected_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    command = [
        "cargo", "run", "--locked", "--quiet", "--manifest-path", str(manifest),
        "--", *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expected_exit:
        raise RuntimeError(
            f"{args[0]} exited {result.returncode}; expected {expected_exit}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed = None
    if result.stdout.strip():
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{args[0]} emitted invalid JSON: {error}") from error
        if isinstance(value, dict):
            parsed = value
    return result, parsed


def make_plan_fields(source_revision: str) -> dict[str, Any]:
    judge_prompt = digest("canonical-judge-prompt")
    variant_prompt = digest("semantically-equivalent-prompt-variant")
    definitions = [
        ("stable-correct", "update_compliance", "easy", "pass"),
        ("stable-wrong", "conflict_disclosure", "hard", "fail"),
        ("position-biased", "tool_workflow", "medium", "pass"),
        ("prompt-fragile", "evidence_force_monotonicity", "adversarial", "pass"),
    ]
    items = []
    trials = []
    for case_id, family, difficulty, reference_label in definitions:
        items.append({
            "case_id": case_id,
            "contract_family": family,
            "difficulty": difficulty,
            "reference_label": reference_label,
            "reference_evidence_sha256": digest(f"reference-evidence:{case_id}"),
        })
        variants = [
            ("baseline", "baseline", 0, "a_then_b", judge_prompt, False),
            ("prompt", "prompt_variant", 0, "a_then_b", variant_prompt, True),
            ("order", "order_swap", 0, "b_then_a", judge_prompt, True),
            ("repeat", "repeat", 1, "a_then_b", judge_prompt, True),
        ]
        for suffix, kind, replicate, order, prompt, invariant in variants:
            trials.append({
                "trial_id": f"{case_id}-{suffix}",
                "case_id": case_id,
                "perturbation_kind": kind,
                "variant_id": {
                    "baseline": "canonical",
                    "prompt_variant": "paraphrase-1",
                    "order_swap": "swap-ab",
                    "repeat": "repeat-1",
                }[kind],
                "replicate_index": replicate,
                "presentation_order": order,
                "prompt_sha256": prompt,
                "expected_invariant_to_baseline": invariant,
            })
    return {
        "schema": PLAN_SCHEMA,
        "purpose": "qualified_run",
        "audit_id": "synthetic-judge-reliability-positive-control-v1",
        "judge": {
            "provider": "synthetic",
            "model_id": "synthetic-judge-fixture",
            "model_revision": "known-behavior-v1",
            "api_version": "fixture",
            "prompt_sha256": judge_prompt,
            "rubric_sha256": digest("rubric-v1"),
            "settings_sha256": digest("settings-v1"),
            "evaluator_source_revision": source_revision,
        },
        "reference": {
            "label_set_sha256": digest("reference-label-set-v1"),
            "protocol_sha256": digest("blind-reference-protocol-v1"),
            "source": "deterministic_programmatic",
            "blinded_from_judge": True,
        },
        "perturbation_policy_sha256": digest("perturbation-policy-v1"),
        "scoring_rules_sha256": digest("judge-scoring-rules-v1"),
        "items": items,
        "trials": trials,
    }


def make_report_fields(plan: dict[str, Any], plan_bytes: bytes) -> dict[str, Any]:
    outputs = {
        "stable-correct": ["pass", "pass", "pass", "pass"],
        # Perfect repeatability must not conceal systematic wrongness.
        "stable-wrong": ["pass", "pass", "pass", "pass"],
        "position-biased": ["pass", "pass", "fail", "pass"],
        "prompt-fragile": ["pass", "fail", "pass", "pass"],
    }
    records = []
    for item in plan["items"]:
        for suffix, verdict in zip(
            ["baseline", "prompt", "order", "repeat"],
            outputs[item["case_id"]],
            strict=True,
        ):
            trial_id = f"{item['case_id']}-{suffix}"
            records.append({
                "trial_id": trial_id,
                "case_id": item["case_id"],
                "terminal_status": "completed",
                "verdict": verdict,
                "confidence_millionths": 950_000,
                "request_receipt_sha256": digest(f"request:{trial_id}"),
                "response_artifact_sha256": digest(f"response:{trial_id}"),
                "evaluator_record_sha256": digest(f"judge-record:{trial_id}"),
            })
    return {
        "schema": REPORT_SCHEMA,
        "purpose": "qualified_run",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "judge_source_revision": plan["judge"]["evaluator_source_revision"],
        "scoring_rules_sha256": plan["scoring_rules_sha256"],
        "records": records,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    revision = os.environ.get("SBR_GENERATOR_SOURCE_REVISION", "")
    checked_out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != checked_out:
        raise RuntimeError("test vector revision must equal exact checked-out source revision")
    if len(revision) != 40 or set(revision) == {"0"}:
        raise RuntimeError("test vector source revision is not a valid full SHA")

    fields = make_plan_fields(revision)
    fields_path = output / "plan-fields.json"
    write_json(fields_path, fields)
    _, commitment = run_cli(
        args.manifest_path, ["commit-judge-reliability-plan", str(fields_path)]
    )
    if commitment is None:
        raise RuntimeError("plan commitment command emitted no JSON")
    plan = dict(fields)
    plan["plan_commitment_sha256"] = commitment["plan_commitment_sha256"]
    plan_path = output / "plan.json"
    plan_bytes = write_json(plan_path, plan)

    _, plan_verification = run_cli(
        args.manifest_path, ["verify-judge-reliability-plan", str(plan_path)]
    )
    if plan_verification is None or plan_verification.get("valid") is not True:
        raise RuntimeError(f"valid synthetic plan rejected: {plan_verification}")

    report_fields = make_report_fields(plan, plan_bytes)
    report_fields_path = output / "report-fields.json"
    write_json(report_fields_path, report_fields)
    _, report_commitment = run_cli(
        args.manifest_path,
        ["commit-judge-reliability-report", str(plan_path), str(report_fields_path)],
    )
    if report_commitment is None:
        raise RuntimeError("report commitment command emitted no JSON")
    report = dict(report_fields)
    report["report_commitment_sha256"] = report_commitment["report_commitment_sha256"]
    report_path = output / "report.json"
    write_json(report_path, report)

    first, verification = run_cli(
        args.manifest_path,
        ["verify-judge-reliability-report", str(plan_path), str(report_path)],
    )
    second, _ = run_cli(
        args.manifest_path,
        ["verify-judge-reliability-report", str(plan_path), str(report_path)],
    )
    if verification is None or verification.get("valid") is not True:
        raise RuntimeError(f"synthetic report rejected: {verification}")
    if first.stdout != second.stdout:
        raise RuntimeError("verification is not byte-deterministic")

    score = verification.get("scorecard") or {}
    expected = {
        "planned_item_count": 4,
        "planned_trial_count": 16,
        "observed_record_count": 16,
        "baseline_accuracy_millionths": 750_000,
        "baseline_coverage_millionths": 1_000_000,
        "baseline_brier_score_millionths": 227_500,
        "baseline_expected_calibration_error_millionths": 200_000,
        "baseline_confidence_recorded_count": 4,
        "trusted_item_count": 1,
        "trusted_verdict_rate_millionths": 250_000,
        "completed_trial_count": 16,
        "abstained_trial_count": 0,
        "error_trial_count": 0,
        "missing_trial_count": 0,
    }
    for key, value in expected.items():
        if score.get(key) != value:
            raise RuntimeError(f"scorecard {key}={score.get(key)!r}; expected {value!r}")
    metrics = {entry["kind"]: entry for entry in score.get("perturbation_metrics", [])}
    for kind, expected_flips in [
        ("prompt_variant", 1),
        ("order_swap", 1),
        ("repeat", 0),
    ]:
        metric = metrics.get(kind)
        if metric is None or metric.get("flip_count") != expected_flips:
            raise RuntimeError(f"unexpected {kind} metric: {metric}")
    if score.get("judge_identity_authenticated") is not False:
        raise RuntimeError("scorecard incorrectly authenticates judge identity")
    if score.get("external_anchor_required") is not True:
        raise RuntimeError("scorecard must require an external anchor")

    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["audit_id"] += "-tampered"
    tampered_plan_path = output / "tampered-plan.json"
    write_json(tampered_plan_path, tampered)
    _, invalid_plan = run_cli(
        args.manifest_path,
        ["verify-judge-reliability-plan", str(tampered_plan_path)],
        expected_exit=11,
    )
    if invalid_plan is None or invalid_plan.get("valid") is not False:
        raise RuntimeError("tampered plan was not rejected")

    wrong_revision = json.loads(plan_path.read_text(encoding="utf-8"))
    wrong_revision["judge"]["evaluator_source_revision"] = "a" * 40
    wrong_revision_path = output / "wrong-revision-plan.json"
    write_json(wrong_revision_path, wrong_revision)
    run_cli(
        args.manifest_path,
        ["verify-judge-reliability-plan", str(wrong_revision_path)],
        expected_exit=11,
    )

    duplicate = json.loads(report_fields_path.read_text(encoding="utf-8"))
    duplicate["records"][1]["trial_id"] = duplicate["records"][0]["trial_id"]
    duplicate_path = output / "duplicate-report-fields.json"
    write_json(duplicate_path, duplicate)
    run_cli(
        args.manifest_path,
        ["commit-judge-reliability-report", str(plan_path), str(duplicate_path)],
        expected_exit=2,
    )

    missing = json.loads(report_fields_path.read_text(encoding="utf-8"))
    missing["records"].pop()
    missing_path = output / "missing-report-fields.json"
    write_json(missing_path, missing)
    run_cli(
        args.manifest_path,
        ["commit-judge-reliability-report", str(plan_path), str(missing_path)],
        expected_exit=2,
    )

    mutated_report = json.loads(report_path.read_text(encoding="utf-8"))
    mutated_report["records"][0]["verdict"] = "fail"
    mutated_report_path = output / "mutated-report.json"
    write_json(mutated_report_path, mutated_report)
    _, invalid_report = run_cli(
        args.manifest_path,
        ["verify-judge-reliability-report", str(plan_path), str(mutated_report_path)],
        expected_exit=11,
    )
    if invalid_report is None or invalid_report.get("valid") is not False:
        raise RuntimeError("mutated report was not rejected")

    summary = {
        "schema": "symthaea.science_bench_refresh_judge_reliability_test_summary.v1",
        "source_revision": revision,
        "synthetic_positive_control": True,
        "planned_items": 4,
        "planned_trials": 16,
        "baseline_accuracy_millionths": score["baseline_accuracy_millionths"],
        "baseline_brier_score_millionths": score["baseline_brier_score_millionths"],
        "baseline_expected_calibration_error_millionths": score[
            "baseline_expected_calibration_error_millionths"
        ],
        "trusted_verdict_rate_millionths": score["trusted_verdict_rate_millionths"],
        "prompt_flip_count": metrics["prompt_variant"]["flip_count"],
        "position_flip_count": metrics["order_swap"]["flip_count"],
        "repeat_flip_count": metrics["repeat"]["flip_count"],
        "deterministic_but_wrong_control_fails_trust_gate": True,
        "tampered_plan_rejected": True,
        "wrong_revision_rejected": True,
        "duplicate_and_missing_records_rejected": True,
        "mutated_report_rejected": True,
        "judge_identity_authenticated": False,
        "external_anchor_required": True,
        "empirical_judge_result": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"judge reliability test vectors failed: {error}", file=sys.stderr)
        raise SystemExit(1)
