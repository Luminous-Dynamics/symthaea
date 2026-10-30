#!/usr/bin/env python3
"""Generate a deterministic seven-relation, non-scoring metamorphic fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metamorphic_evaluator_audit import PLAN_SCHEMA, RECEIPT_SCHEMA, plan_commitment, sha256


def _scores(accuracy: int, evidence: int, clarity: int) -> list[dict[str, Any]]:
    return [
        {"criterion_id": "accuracy", "score_millionths": accuracy},
        {"criterion_id": "evidence", "score_millionths": evidence},
        {"criterion_id": "clarity", "score_millionths": clarity},
    ]


def _outcome(
    decision: str,
    scores: list[dict[str, Any]],
    *,
    candidate: str | None = None,
    position: int | None = None,
    groups: int = 2,
    pointwise: list[dict[str, Any]] | None = None,
    preferred: str | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "criterion_scores": scores,
        "hard_veto_active": False,
        "selected_candidate_id": candidate,
        "selected_position": position,
        "unique_source_group_count": groups,
        "pointwise_candidate_scores": pointwise or [],
        "pairwise_preferred_candidate_id": preferred,
    }


def _pair(pair_id: str, family: str, rule: str, *,
          affected: list[str] | None = None,
          unaffected: list[str] | None = None,
          expected_decision: str | None = None,
          expected_candidate: str | None = None) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "relation_family": family,
        "relation_rule": rule,
        "baseline_input_sha256": sha256((pair_id + ":baseline-input").encode()),
        "variant_input_sha256": sha256((pair_id + ":variant-input").encode()),
        "affected_criterion_ids": affected or [],
        "unaffected_criterion_ids": unaffected or [],
        "expected_variant_decision": expected_decision,
        "expected_selected_candidate_id": expected_candidate,
    }


def build_example() -> tuple[dict[str, Any], dict[str, Any]]:
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "example_only",
        "plan_id": "metamorphic-evaluator-example-001",
        "evaluator_source_revision": "a" * 40,
        "evaluator_prompt_sha256": sha256(b"example evaluator prompt"),
        "rubric_sha256": sha256(b"example fixed rubric"),
        "criterion_ids": ["accuracy", "evidence", "clarity"],
        "score_tolerance_millionths": 10_000,
        "pairs": [
            _pair("invariance-format-001", "invariance", "all_outputs_invariant"),
            _pair(
                "evidence-withdrawal-001", "directional_sensitivity",
                "support_non_increasing", affected=["evidence"],
                unaffected=["accuracy", "clarity"],
                expected_decision="insufficient_evidence",
            ),
            _pair(
                "evidence-addition-001", "directional_sensitivity",
                "support_non_decreasing", affected=["evidence"],
                unaffected=["accuracy", "clarity"],
                expected_decision="supported_within_scope",
            ),
            _pair(
                "criterion-locality-001", "locality",
                "only_unaffected_scores_invariant", affected=["evidence"],
                unaffected=["accuracy", "clarity"],
            ),
            _pair(
                "position-swap-001", "order_equivariance",
                "candidate_identity_invariant_position_changes",
                expected_candidate="candidate-A",
            ),
            _pair(
                "provenance-mirror-001", "provenance_non_duplication",
                "unique_origin_count_not_increased",
            ),
            _pair(
                "pointwise-pairwise-001", "cross_mode_consistency",
                "pointwise_pairwise_choice_agree",
            ),
            _pair(
                "unit-counterfactual-001", "counterfactual_consistency",
                "variant_decision_equals_declared",
                expected_decision="contradicted_within_scope",
            ),
        ],
    }
    plan["plan_commitment_sha256"] = plan_commitment(plan)
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")

    observations = [
        {
            "pair_id": "invariance-format-001",
            "baseline": _outcome("supported_within_scope", _scores(900_000, 900_000, 800_000)),
            "variant": _outcome("supported_within_scope", _scores(900_000, 900_000, 800_000)),
        },
        {
            "pair_id": "evidence-withdrawal-001",
            "baseline": _outcome("supported_within_scope", _scores(900_000, 900_000, 800_000)),
            "variant": _outcome("insufficient_evidence", _scores(550_000, 250_000, 800_000)),
        },
        {
            "pair_id": "evidence-addition-001",
            "baseline": _outcome("insufficient_evidence", _scores(550_000, 250_000, 800_000)),
            "variant": _outcome("supported_within_scope", _scores(550_000, 900_000, 800_000)),
        },
        {
            "pair_id": "criterion-locality-001",
            "baseline": _outcome("supported_within_scope", _scores(900_000, 900_000, 800_000)),
            "variant": _outcome("mixed_or_contested", _scores(900_000, 400_000, 800_000)),
        },
        {
            "pair_id": "position-swap-001",
            "baseline": _outcome(
                "supported_within_scope", _scores(850_000, 820_000, 800_000),
                candidate="candidate-A", position=0,
            ),
            "variant": _outcome(
                "supported_within_scope", _scores(850_000, 820_000, 800_000),
                candidate="candidate-A", position=1,
            ),
        },
        {
            "pair_id": "provenance-mirror-001",
            "baseline": _outcome("supported_within_scope", _scores(850_000, 850_000, 800_000), groups=2),
            "variant": _outcome("supported_within_scope", _scores(850_000, 850_000, 800_000), groups=2),
        },
        {
            "pair_id": "pointwise-pairwise-001",
            "baseline": _outcome(
                "supported_within_scope", _scores(800_000, 850_000, 800_000),
                pointwise=[
                    {"candidate_id": "candidate-A", "score_millionths": 820_000},
                    {"candidate_id": "candidate-B", "score_millionths": 700_000},
                ],
                preferred="candidate-A",
            ),
            "variant": _outcome(
                "supported_within_scope", _scores(810_000, 850_000, 800_000),
                pointwise=[
                    {"candidate_id": "candidate-A", "score_millionths": 830_000},
                    {"candidate_id": "candidate-B", "score_millionths": 700_000},
                ],
                preferred="candidate-A",
            ),
        },
        {
            "pair_id": "unit-counterfactual-001",
            "baseline": _outcome("supported_within_scope", _scores(900_000, 900_000, 700_000)),
            "variant": _outcome("contradicted_within_scope", _scores(200_000, 200_000, 700_000)),
        },
    ]
    pair_digests = {pair["pair_id"]: pair for pair in plan["pairs"]}
    for observation in observations:
        planned_pair = pair_digests[observation["pair_id"]]
        observation["baseline_input_sha256"] = planned_pair["baseline_input_sha256"]
        observation["variant_input_sha256"] = planned_pair["variant_input_sha256"]

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "purpose": "example_only",
        "run_id": "example-run-not-authenticated-001",
        "plan_sha256": sha256(plan_bytes),
        "evaluator_source_revision": plan["evaluator_source_revision"],
        "evaluator_prompt_sha256": plan["evaluator_prompt_sha256"],
        "rubric_sha256": plan["rubric_sha256"],
        "observations": observations,
    }
    return plan, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan, receipt = build_example()
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    (args.output_dir / "metamorphic-plan.example.json").write_bytes(plan_bytes)
    (args.output_dir / "metamorphic-receipt.example.json").write_bytes(receipt_bytes)
    print(json.dumps({
        "schema": "symthaea.science_bench_refresh_metamorphic_fixture_summary.v1",
        "synthetic_positive_control": True,
        "planned_pairs": len(plan["pairs"]),
        "families": sorted({pair["relation_family"] for pair in plan["pairs"]}),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "plan_sha256": receipt["plan_sha256"],
        "evaluator_execution_proven": False,
        "receipt_authenticity_proven": False,
        "causal_effect_identified": False,
        "scoring_eligible": False,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
