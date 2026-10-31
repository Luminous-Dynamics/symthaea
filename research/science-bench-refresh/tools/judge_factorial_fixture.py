#!/usr/bin/env python3
"""Generate a deterministic synthetic full-factorial judge-bias fixture.

The fixture validates schemas, commitments, denominators and analysis wiring.
It is not a judge execution, randomized human study, or empirical model result.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

FACTOR_DOMAIN = b"symthaea.judge_factorial.condition.v1\0"
PLAN_SCHEMA = "symthaea.science_bench_refresh_judge_factorial_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_judge_factorial_receipt.v1"
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def condition_id(block_id: str, repeat_index: int, factors: dict[str, str]) -> str:
    identity = {
        "block_id": block_id,
        "repeat_index": repeat_index,
        "factors": factors,
    }
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()
    framed = FACTOR_DOMAIN + len(payload).to_bytes(8, "big") + payload
    return digest(framed)


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    block_id = "synthetic-block-001"
    criteria = ["accuracy", "evidence"]
    options = [
        {"option_id": "low", "score_millionths": 0},
        {"option_id": "middle", "score_millionths": 500_000},
        {"option_id": "high", "score_millionths": 1_000_000},
    ]
    factor_names = [
        "candidate_order",
        "score_option_order",
        "criterion_order",
        "provenance_cue",
        "presentation_format",
    ]
    levels = [
        ["candidate_a_first", "candidate_b_first"],
        ["canonical", "reversed"],
        ["canonical", "reversed"],
        ["anonymous", "displayed"],
        ["structured", "prose"],
    ]
    conditions = []
    for repeat_index in range(2):
        for code in range(32):
            factors = {
                factor_names[index]: levels[index][1 if code & (1 << index) else 0]
                for index in range(5)
            }
            condition = {
                "condition_id": condition_id(block_id, repeat_index, factors),
                "repeat_index": repeat_index,
                "factors": factors,
                "input_sha256": digest(("rendered-input:" + str(repeat_index) + ":" + str(code)).encode()),
            }
            conditions.append(condition)

    execution_order = [
        row["condition_id"]
        for row in sorted(
            conditions,
            key=lambda row: digest(("schedule-seed-001:" + row["condition_id"]).encode()),
        )
    ]
    block = {
        "block_id": block_id,
        "task_semantics_sha256": digest(b"fixed synthetic task meaning"),
        "candidate_a_id": "candidate-A",
        "candidate_b_id": "candidate-B",
        "criterion_ids": criteria,
        "score_options": options,
        "execution_order": execution_order,
        "conditions": conditions,
    }
    plan_fields = {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": git_revision(),
        "purpose": "example_only",
        "study_id": "synthetic-factorial-judge-bias-001",
        "judge_source_revision": "a" * 40,
        "judge_prompt_sha256": digest(b"synthetic frozen judge prompt"),
        "rubric_sha256": digest(b"synthetic frozen rubric"),
        "semantic_task_set_sha256": digest(b"synthetic semantic task set"),
        "repeats_per_condition": 2,
        "blocks": [block],
    }
    fields_path = output_dir / "judge-factorial-plan-fields.example.json"
    fields_path.write_text(
        json.dumps(plan_fields, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
        "--", "commit-judge-factorial-plan", str(fields_path),
    ]
    committed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    commitment = json.loads(committed.stdout)["plan_commitment_sha256"]
    plan = {**plan_fields, "plan_commitment_sha256": commitment}
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    plan_path = output_dir / "judge-factorial-plan.example.json"
    plan_path.write_bytes(plan_bytes)

    execution_ordinals = {
        condition_id: ordinal
        for ordinal, condition_id in enumerate(execution_order)
    }
    observations = []
    for row in conditions:
        factors = row["factors"]
        a_score = 700_000 if factors["candidate_order"] == "candidate_a_first" else 650_000
        if factors["candidate_order"] == "candidate_b_first" and factors["provenance_cue"] == "displayed":
            a_score += 40_000
        if factors["presentation_format"] == "prose":
            a_score += 15_000

        # The synthetic judge follows the first displayed score option for
        # accuracy, then exhibits an independent criterion-order perturbation
        # on evidence. These are intentionally planted signals for analyzer QA.
        accuracy_option = "low" if factors["score_option_order"] == "canonical" else "high"
        evidence_option = "low" if factors["criterion_order"] == "canonical" else "high"
        selected = {"accuracy": accuracy_option, "evidence": evidence_option}
        score_map = {option["option_id"]: option["score_millionths"] for option in options}
        criterion_scores = [
            {
                "criterion_id": criterion_id,
                "score_millionths": score_map[selected[criterion_id]],
                "selected_option_id": selected[criterion_id],
            }
            for criterion_id in criteria
        ]
        decision = (
            "candidate_a"
            if factors["candidate_order"] == "candidate_a_first"
            or factors["provenance_cue"] == "displayed"
            else "candidate_b"
        )
        observations.append({
            "condition_id": row["condition_id"],
            "execution_ordinal": execution_ordinals[row["condition_id"]],
            "input_sha256": row["input_sha256"],
            "status": "completed",
            "judge_receipt_sha256": digest(("synthetic-judge-receipt:" + row["condition_id"]).encode()),
            "decision": decision,
            "candidate_a_score_millionths": a_score,
            "candidate_b_score_millionths": 600_000,
            "criterion_scores": criterion_scores,
        })

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "analyzer_source_revision": plan_fields["analyzer_source_revision"],
        "purpose": "example_only",
        "run_id": "synthetic-run-not-authenticated-001",
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
        "judge_source_revision": plan_fields["judge_source_revision"],
        "judge_prompt_sha256": plan_fields["judge_prompt_sha256"],
        "rubric_sha256": plan_fields["rubric_sha256"],
        "observations": observations,
    }
    receipt_path = output_dir / "judge-factorial-receipt.example.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "symthaea.science_bench_refresh_judge_factorial_fixture_summary.v1",
        "synthetic_positive_control": True,
        "planned_blocks": 1,
        "factor_count": 5,
        "factorial_cells_per_repeat": 32,
        "repeats_per_condition": 2,
        "planned_conditions": len(conditions),
        "completed_synthetic_conditions": len(observations),
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
        "execution_authenticity_proven": False,
        "randomization_proven": False,
        "semantic_equivalence_proven": False,
        "causal_effect_identified": False,
        "scoring_eligible": False,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: judge_factorial_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    summary = generate(Path(sys.argv[1]))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
