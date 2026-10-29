#!/usr/bin/env python3
"""Independent synthetic vectors for Rust counterfactual outcome receipt checks.

No real question, answer, protected evidence, model output or oracle label is emitted.
This is test-only vector generation, not a production evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.evidence_outcomes.v1\0"
SCHEMA = "symthaea.science_bench_refresh_evidence_outcome_report.v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def framed_hash(domain: bytes, payload: bytes) -> str:
    return sha256(
        PREFIX
        + len(domain).to_bytes(8, "big")
        + domain
        + len(payload).to_bytes(8, "big")
        + payload
    )


def null_resources() -> dict[str, int | None]:
    return {
        "wall_clock_ms": None,
        "tool_calls": None,
        "experiment_count": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_micro_usd": None,
    }


def outcome_record(pair_id: str, trial_id: str, condition: str, index: int) -> dict[str, Any]:
    digest = lambda label: sha256(label.encode("utf-8"))
    base: dict[str, Any] = {
        "pair_id": pair_id,
        "trial_id": trial_id,
        "condition": condition,
        "terminal_status": "completed",
        "response_status": "answered",
        "runner_receipt_sha256": digest(f"runner-receipt:{trial_id}"),
        "response_artifact_sha256": digest(f"response-artifact:{trial_id}"),
        "evaluator_record_sha256": digest(f"evaluator-record:{trial_id}"),
        "task_correctness": "correct",
        "evidence_use": "used_relevant",
        "claim_support": "fully_supported",
        "abstention_quality": "not_applicable",
        "confidence_millionths": 900_000,
        "confidence_source": "elicited",
        "resources": {
            "wall_clock_ms": 1000 + index * 100,
            "tool_calls": index,
            "experiment_count": 1 + index,
            "input_tokens": 1000 + index * 10,
            "output_tokens": 100 + index * 5,
            "cost_micro_usd": 250 + index * 10,
        },
    }

    if condition == "network_isolated":
        base["task_correctness"] = "incorrect"
        base["evidence_use"] = "no_relevant_evidence"
        base["claim_support"] = "unsupported"
        base["confidence_millionths"] = 700_000
    elif condition == "gold_evidence_withheld":
        base.update({
            "response_status": "abstained",
            "task_correctness": "not_evaluated",
            "evidence_use": "no_relevant_evidence",
            "claim_support": "not_evaluated",
            "abstention_quality": "appropriate",
            "confidence_millionths": None,
            "confidence_source": "none",
        })
    elif condition == "source_contradiction":
        base.update({
            "task_correctness": "partial",
            "evidence_use": "used_relevant",
            "claim_support": "partially_supported",
            "confidence_millionths": 600_000,
        })
    elif condition == "decoy_noise":
        # Keep the assigned denominator and measured execution costs, but do
        # not convert timeout into a scored answer or a zero-cost trial.
        base.update({
            "terminal_status": "timeout",
            "response_status": "no_response",
            "response_artifact_sha256": None,
            "evaluator_record_sha256": None,
            "task_correctness": "not_evaluated",
            "evidence_use": "not_evaluated",
            "claim_support": "not_evaluated",
            "abstention_quality": "not_evaluated",
            "confidence_millionths": None,
            "confidence_source": "none",
        })
    return base


def make_fields(plan: dict[str, Any], plan_bytes: bytes) -> dict[str, Any]:
    records = []
    for pair in plan["pairs"]:
        for index, trial in enumerate(pair["trials"]):
            records.append(outcome_record(
                pair["pair_id"], trial["trial_id"], trial["condition"], index
            ))
    return {
        "schema": SCHEMA,
        "plan_sha256": sha256(plan_bytes),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "evaluator_revision_sha256": plan["evaluator_revision_sha256"],
        "decision_rules_sha256": plan["decision_rules_sha256"],
        "records": records,
    }


def report_commitment(fields: dict[str, Any]) -> str:
    # Field ordering mirrors Rust's typed OutcomeCommitmentPayloadV1.
    payload = {
        "schema": fields["schema"],
        "plan_sha256": fields["plan_sha256"],
        "plan_commitment_sha256": fields["plan_commitment_sha256"],
        "evaluator_revision_sha256": fields["evaluator_revision_sha256"],
        "decision_rules_sha256": fields["decision_rules_sha256"],
        "records": fields["records"],
    }
    return framed_hash(b"outcome_report.v1", encode(payload))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def make_full(fields: dict[str, Any]) -> dict[str, Any]:
    return {**fields, "report_commitment_sha256": report_commitment(fields)}


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: evidence_outcome_test_vectors.py PLAN.json OUTPUT_DIR",
            file=sys.stderr,
        )
        return 2
    plan_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    output.mkdir(parents=True, exist_ok=True)

    fields = make_fields(plan, plan_bytes)
    valid = make_full(fields)
    write_json(output / "outcome-fields.json", fields)
    write_json(output / "valid-outcome-report.json", valid)

    missing = copy.deepcopy(fields)
    missing["records"].pop()
    write_json(output / "missing-outcome-report.json", make_full(missing))

    duplicate = copy.deepcopy(fields)
    duplicate["records"][1] = copy.deepcopy(duplicate["records"][0])
    write_json(output / "duplicate-outcome-report.json", make_full(duplicate))

    wrong_condition = copy.deepcopy(fields)
    wrong_condition["records"][0]["condition"] = "source_contradiction"
    write_json(output / "wrong-condition-report.json", make_full(wrong_condition))

    changed_evaluator = copy.deepcopy(fields)
    changed_evaluator["evaluator_revision_sha256"] = sha256(b"post-hoc-evaluator-change")
    write_json(
        output / "changed-evaluator-report.json",
        make_full(changed_evaluator),
    )

    changed_rules = copy.deepcopy(fields)
    changed_rules["decision_rules_sha256"] = sha256(b"post-hoc-decision-rule-change")
    write_json(output / "changed-rules-report.json", make_full(changed_rules))

    contaminated_scored = copy.deepcopy(fields)
    contaminated_scored["records"][0]["terminal_status"] = "contaminated"
    write_json(output / "contaminated-scored-report.json", make_full(contaminated_scored))

    budget_violation_scored = copy.deepcopy(fields)
    budget_violation_scored["records"][0]["terminal_status"] = "budget_violation"
    write_json(
        output / "budget-violation-scored-report.json",
        make_full(budget_violation_scored),
    )

    tampered = copy.deepcopy(valid)
    tampered["report_commitment_sha256"] = "0" * 64
    write_json(output / "mutated-commitment-report.json", tampered)

    omitted_null = copy.deepcopy(valid)
    del omitted_null["records"][0]["resources"]["cost_micro_usd"]
    write_json(output / "omitted-null-field-report.json", omitted_null)

    print(json.dumps({
        "schema": "symthaea.evidence_outcome_test_vectors.v1",
        "expected_report_commitment_sha256": valid["report_commitment_sha256"],
        "planned_trial_count": len(plan["pairs"]) * len(plan["required_conditions"]),
        "timeout_trial_count": sum(
            record["terminal_status"] == "timeout" for record in valid["records"]
        ),
        "confidence_missing_answered_trial_count": sum(
            record["terminal_status"] == "completed"
            and record["response_status"] == "answered"
            and record["confidence_millionths"] is None
            for record in valid["records"]
        ),
        "output_dir": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
