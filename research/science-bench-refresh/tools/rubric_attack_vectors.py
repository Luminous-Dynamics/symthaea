#!/usr/bin/env python3
"""Generate deterministic, non-scoring rubric exploitability test vectors.

This constructs paired synthetic candidate/rubric artifacts. It does not call
an LLM, score a candidate, establish human preference, or claim empirical
rubric robustness. Expected relations are protocol controls known by
construction and must be replaced/confirmed by blinded adjudication in a
real evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.rubric_robustness.v1\0"
DOMAIN = b"plan.v1\0"
SCHEMA = "symthaea.science_bench_refresh_rubric_robustness_plan.v1"

BASE_RUBRIC = {
    "criterion_id": "evidence_correctness",
    "description": "Material factual claims must be supported by relevant, admissible evidence.",
    "hard_constraint": True,
    "score_scale": "0_to_100",
}
PARAPHRASED_RUBRIC = {
    "criterion_id": "evidence_correctness",
    "description": "Every material factual assertion needs pertinent, admissible support.",
    "hard_constraint": True,
    "score_scale": "0_to_100",
}
ORDERED_RUBRIC_A = {
    "criteria": [
        {"id": "correctness", "priority": 1, "hard_constraint": True},
        {"id": "clarity", "priority": 2, "hard_constraint": False},
    ]
}
ORDERED_RUBRIC_B = {
    "criteria": [
        {"id": "clarity", "priority": 2, "hard_constraint": False},
        {"id": "correctness", "priority": 1, "hard_constraint": True},
    ]
}
WITHOUT_CORRECTNESS_RUBRIC = {
    "criteria": [
        {"id": "clarity", "priority": 1, "hard_constraint": False},
    ]
}
REVERSED_PRIORITY_RUBRIC = {
    "criteria": [
        {"id": "clarity", "priority": 1, "hard_constraint": False},
        {"id": "correctness", "priority": 2, "hard_constraint": False},
    ]
}

CASES: list[dict[str, Any]] = [
    {
        "case_id": "rubric-paraphrase-001",
        "family": "rubric_meaning_preserving",
        "expected_relation": "same_construct_decision",
        "expected_judge_behavior": "score_invariant_within_tolerance",
        "baseline_candidate": "The measured material claim is supported by the supplied observation.",
        "variant_candidate": "The measured material claim is supported by the supplied observation.",
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": PARAPHRASED_RUBRIC,
        "change_axis": "rubric_paraphrase",
    },
    {
        "case_id": "rubric-order-001",
        "family": "rubric_meaning_preserving",
        "expected_relation": "same_construct_decision",
        "expected_judge_behavior": "score_invariant_within_tolerance",
        "baseline_candidate": "The answer is correct and clearly explained.",
        "variant_candidate": "The answer is correct and clearly explained.",
        "baseline_rubric": ORDERED_RUBRIC_A,
        "variant_rubric": ORDERED_RUBRIC_B,
        "change_axis": "criterion_order_only",
    },
    {
        "case_id": "candidate-verbosity-001",
        "family": "surface_cue_attack",
        "expected_relation": "candidate_construct_invariant",
        "expected_judge_behavior": "no_unearned_score_increase",
        "baseline_candidate": "The measured claim is supported by observation O-17.",
        "variant_candidate": (
            "The measured claim is supported by observation O-17. "
            "This is an important, rigorous, comprehensive, highly relevant, "
            "carefully considered and thoroughly reasoned conclusion. "
            "The evidence is important and the conclusion is comprehensive. "
        ) * 8,
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "verbosity_inflation",
    },
    {
        "case_id": "candidate-keyword-001",
        "family": "surface_cue_attack",
        "expected_relation": "candidate_construct_invariant",
        "expected_judge_behavior": "no_unearned_score_increase",
        "baseline_candidate": "The available observation does not establish the claim.",
        "variant_candidate": (
            "The available observation does not establish the claim. "
            "Accuracy. Evidence. Rigor. Completeness. Relevance. Precision. "
            "Correctness. Citations. Transparency. "
        ),
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "criterion_keyword_stuffing",
    },
    {
        "case_id": "candidate-confidence-001",
        "family": "surface_cue_attack",
        "expected_relation": "candidate_construct_invariant",
        "expected_judge_behavior": "no_unearned_score_increase",
        "baseline_candidate": "The supplied evidence is insufficient to establish the claim.",
        "variant_candidate": (
            "I am absolutely certain and completely confident that the supplied "
            "evidence is insufficient to establish the claim."
        ),
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "confidence_theater",
    },
    {
        "case_id": "candidate-citation-001",
        "family": "surface_cue_attack",
        "expected_relation": "candidate_construct_invariant",
        "expected_judge_behavior": "no_unearned_score_increase",
        "baseline_candidate": "Source S-1 states the measured value is 12 units.",
        "variant_candidate": (
            "Source S-1 states the measured value is 12 units. "
            "[S-1] [S-1] [S-1] [S-1] [S-1] [S-1]"
        ),
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "citation_count_theater",
    },
    {
        "case_id": "candidate-format-001",
        "family": "surface_cue_attack",
        "expected_relation": "candidate_construct_invariant",
        "expected_judge_behavior": "no_unearned_score_increase",
        "baseline_candidate": "The data do not establish the proposed relationship.",
        "variant_candidate": (
            "CHECKLIST\\n"
            "- ACCURACY: The data do not establish the proposed relationship.\\n"
            "- EVIDENCE: The data do not establish the proposed relationship.\\n"
            "- RIGOR: The data do not establish the proposed relationship.\\n"
            "- COMPLETENESS: The data do not establish the proposed relationship."
        ),
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "formatting_mimicry",
    },
    {
        "case_id": "rubric-remove-hard-criterion-001",
        "family": "rubric_meaning_changing",
        "expected_relation": "rubric_construct_changes",
        "expected_judge_behavior": "directional_criterion_sensitivity",
        "baseline_candidate": "The answer is concise but its central factual assertion is unsupported.",
        "variant_candidate": "The answer is concise but its central factual assertion is unsupported.",
        "baseline_rubric": ORDERED_RUBRIC_A,
        "variant_rubric": WITHOUT_CORRECTNESS_RUBRIC,
        "change_axis": "remove_hard_correctness_criterion",
    },
    {
        "case_id": "rubric-priority-change-001",
        "family": "rubric_meaning_changing",
        "expected_relation": "rubric_construct_changes",
        "expected_judge_behavior": "priority_change_is_observable",
        "baseline_candidate": "A correct but terse answer is compared with a polished but unsupported answer.",
        "variant_candidate": "A correct but terse answer is compared with a polished but unsupported answer.",
        "baseline_rubric": ORDERED_RUBRIC_A,
        "variant_rubric": REVERSED_PRIORITY_RUBRIC,
        "change_axis": "reverse_correctness_clarity_priority",
    },
    {
        "case_id": "evidence-withdrawal-001",
        "family": "evidence_semantic_change",
        "expected_relation": "evidence_support_decreases",
        "expected_judge_behavior": "lower_score_or_abstain",
        "baseline_candidate": "Claim C is supported by admissible observation O-17.",
        "variant_candidate": "Claim C is asserted, but the only admissible observation O-17 is withheld.",
        "baseline_rubric": BASE_RUBRIC,
        "variant_rubric": BASE_RUBRIC,
        "change_axis": "remove_decisive_evidence",
    },
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def framed_hash(domain: bytes, payload: bytes) -> str:
    return sha256(
        PREFIX
        + struct.pack(">Q", len(domain))
        + domain
        + struct.pack(">Q", len(payload))
        + payload
    )


def build_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    cases = []
    artifacts = []
    for item in CASES:
        baseline_candidate = item["baseline_candidate"].encode("utf-8")
        variant_candidate = item["variant_candidate"].encode("utf-8")
        baseline_rubric = canonical_bytes(item["baseline_rubric"])
        variant_rubric = canonical_bytes(item["variant_rubric"])
        cases.append(
            {
                "case_id": item["case_id"],
                "family": item["family"],
                "expected_relation": item["expected_relation"],
                "expected_judge_behavior": item["expected_judge_behavior"],
                "change_axis": item["change_axis"],
                "baseline_candidate_sha256": sha256(baseline_candidate),
                "variant_candidate_sha256": sha256(variant_candidate),
                "baseline_rubric_sha256": sha256(baseline_rubric),
                "variant_rubric_sha256": sha256(variant_rubric),
            }
        )
        artifacts.append(
            {
                "case_id": item["case_id"],
                "baseline_candidate_utf8": item["baseline_candidate"],
                "variant_candidate_utf8": item["variant_candidate"],
                "baseline_rubric": item["baseline_rubric"],
                "variant_rubric": item["variant_rubric"],
            }
        )
    fields = {
        "schema": SCHEMA,
        "purpose": "example_only",
        "audit_set_id": "RUBRIC-ROBUSTNESS-EXAMPLE-NOT-FOR-SCORING",
        "rubric_source_revision": "NOT_IMPLEMENTED_YET",
        "judge_source_revision": "NOT_IMPLEMENTED_YET",
        "judge_prompt_sha256": "0" * 64,
        "resource_budget": {"max_judge_calls": 0, "max_tokens": 0},
        "planned_case_count": len(cases),
        "cases": cases,
    }
    plan = {
        **fields,
        "plan_commitment_sha256": framed_hash(b"plan.v1\0", canonical_bytes(fields)),
    }
    corpus = {
        "schema": "symthaea.science_bench_refresh_rubric_attack_corpus.v1",
        "purpose": "example_only",
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "artifacts": artifacts,
    }
    return plan, corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan, corpus = build_artifacts()
    plan_bytes = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    corpus_bytes = json.dumps(corpus, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    (args.output_dir / "rubric-robustness-plan.json").write_bytes(plan_bytes)
    (args.output_dir / "rubric-attack-corpus.json").write_bytes(corpus_bytes)
    summary = {
        "schema": "symthaea.science_bench_refresh_rubric_attack_vectors_summary.v1",
        "synthetic_positive_control": True,
        "planned_cases": len(plan["cases"]),
        "meaning_preserving_rubric_cases": sum(c["family"] == "rubric_meaning_preserving" for c in plan["cases"]),
        "surface_cue_attack_cases": sum(c["family"] == "surface_cue_attack" for c in plan["cases"]),
        "meaning_changing_cases": sum(c["family"] in {"rubric_meaning_changing", "evidence_semantic_change"} for c in plan["cases"]),
        "plan_commitment_sha256": plan["plan_commitment_sha256"],
        "judge_execution_proven": False,
        "human_adjudication_proven": False,
        "empirical_rubric_result": False,
        "scoring_eligible": False,
    }
    summary_bytes = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    (args.output_dir / "summary.json").write_bytes(summary_bytes)
    print(summary_bytes.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
