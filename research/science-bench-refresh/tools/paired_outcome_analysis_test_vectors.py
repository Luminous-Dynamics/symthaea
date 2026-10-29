#!/usr/bin/env python3
"""Generate synthetic paired-analysis inputs with a fully precommitted analysis plan.

Only synthetic IDs, hashes, expected action labels and metadata are written. The
analysis plan bytes are hashed into the intervention plan before outcome fixtures
are generated. This script is a CI vector generator, not a production evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.counterfactual_evidence.v1\0"
PLAN_SCHEMA = "symthaea.science_bench_refresh_counterfactual_evidence_plan.v1"
ANALYSIS_SCHEMA = "symthaea.science_bench_refresh_paired_analysis_plan.v1"


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


def write_json_bytes(path: Path, value: dict[str, Any]) -> bytes:
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return data


def make_analysis(
    plan_fields: dict[str, Any],
    revision: str,
    *,
    dependence_unknown: bool = False,
) -> dict[str, Any]:
    pair_ids = [pair["pair_id"] for pair in plan_fields["pairs"]]
    conditions = plan_fields["required_conditions"]
    contrasts = [
        {
            "contrast_id": "gold-injected-correctness-vs-natural",
            "treatment_condition": "gold_evidence_injected",
            "reference_condition": "natural_retrieval",
            "endpoint": "strict_correctness",
            "directional_hypothesis": "treatment_higher",
        },
        {
            "contrast_id": "withheld-expected-response-vs-natural",
            "treatment_condition": "gold_evidence_withheld",
            "reference_condition": "natural_retrieval",
            "endpoint": "expected_response",
            "directional_hypothesis": "no_directional_claim",
        },
        {
            "contrast_id": "source-contradiction-support-vs-natural",
            "treatment_condition": "source_contradiction",
            "reference_condition": "natural_retrieval",
            "endpoint": "fully_supported_claim",
            "directional_hypothesis": "no_directional_claim",
        },
        {
            "contrast_id": "source-contradiction-strict-correctness-vs-natural",
            "treatment_condition": "source_contradiction",
            "reference_condition": "natural_retrieval",
            "endpoint": "strict_correctness",
            "directional_hypothesis": "no_directional_claim",
        },
        {
            "contrast_id": "decoy-cost-vs-natural",
            "treatment_condition": "decoy_noise",
            "reference_condition": "natural_retrieval",
            "endpoint": "cost_micro_usd",
            "directional_hypothesis": "treatment_lower",
        },
    ]
    expected_for_condition = {
        "natural_retrieval": "answered",
        "network_isolated": "answered",
        "gold_evidence_injected": "answered",
        "gold_evidence_withheld": "abstained",
        "decoy_noise": "answered",
        "source_contradiction": "answered",
    }
    expected_responses = [
        {
            "pair_id": pair_id,
            "condition": condition,
            "expected_response": expected_for_condition[condition],
        }
        for pair_id in pair_ids
        for condition in conditions
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "analysis_implementation_revision": revision,
        "multiplicity_policy": (
            "descriptive_no_p_values"
            if dependence_unknown
            else "holm_across_binary_contrasts"
        ),
        "pair_dependence_policy": (
            "descriptive_only_dependence_unknown"
            if dependence_unknown
            else "exact_conditional_independent_pairs"
        ),
        "dependence_justification_sha256": sha256(
            b"synthetic fixture: pair dependence explicitly unknown"
            if dependence_unknown
            else b"synthetic fixture: one unique task/world pair per independent block"
        ),
        "contrasts": contrasts,
        "expected_responses": expected_responses,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: paired_outcome_analysis_test_vectors.py PLAN_FIELDS.json OUTPUT_DIR", file=sys.stderr)
        return 2
    source_fields = json.loads(Path(sys.argv[1]).read_bytes())
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    revision = os.environ.get("SBR_GENERATOR_SOURCE_REVISION", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        print("SBR_GENERATOR_SOURCE_REVISION must be a full lowercase Git SHA", file=sys.stderr)
        return 2

    analysis = make_analysis(source_fields, revision)
    analysis_bytes = write_json_bytes(output / "analysis-plan.json", analysis)

    fields = copy.deepcopy(source_fields)
    fields["analysis_plan_sha256"] = sha256(analysis_bytes)
    # Commitment payload follows the Rust typed field order and compact JSON
    # encoding: schema, generation/plan identity, frozen policy/configuration,
    # analysis/evaluator/rules/randomization pins, conditions, pairs.
    commitment = framed_hash(b"plan.v1", encode(fields))
    plan = {**fields, "plan_commitment_sha256": commitment}
    write_json_bytes(output / "plan.json", plan)
    write_json_bytes(output / "plan-fields.json", fields)

    tampered_analysis = copy.deepcopy(analysis)
    tampered_analysis["multiplicity_policy"] = "unadjusted_exploratory"
    write_json_bytes(output / "tampered-analysis-plan.json", tampered_analysis)
    bad_revision = copy.deepcopy(analysis)
    bad_revision["analysis_implementation_revision"] = "f" * 40
    write_json_bytes(output / "wrong-revision-analysis-plan.json", bad_revision)

    # A separate precommitted plan exercises the conservative path when the
    # evaluator cannot justify independent pair blocks.
    unknown_analysis = make_analysis(source_fields, revision, dependence_unknown=True)
    unknown_analysis_bytes = write_json_bytes(
        output / "unknown-dependence-analysis-plan.json", unknown_analysis
    )
    unknown_fields = copy.deepcopy(source_fields)
    unknown_fields["analysis_plan_sha256"] = sha256(unknown_analysis_bytes)
    unknown_commitment = framed_hash(b"plan.v1", encode(unknown_fields))
    unknown_plan = {
        **unknown_fields,
        "plan_commitment_sha256": unknown_commitment,
    }
    write_json_bytes(output / "unknown-dependence-plan-fields.json", unknown_fields)
    write_json_bytes(output / "unknown-dependence-plan.json", unknown_plan)

    summary = {
        "schema": "symthaea.paired_outcome_analysis_test_vectors.v1",
        "analysis_plan_sha256": sha256(analysis_bytes),
        "expected_plan_commitment_sha256": commitment,
        "unknown_dependence_plan_commitment_sha256": unknown_commitment,
        "analysis_implementation_revision": revision,
        "expected_pair_count": len(fields["pairs"]),
        "expected_contrast_count": len(analysis["contrasts"]),
        "expected_condition_count": len(conditions),
        "multiplicity_policy": analysis["multiplicity_policy"],
    }
    write_json_bytes(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
