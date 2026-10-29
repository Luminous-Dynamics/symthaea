#!/usr/bin/env python3
"""Create synthetic cross-language vectors for the counterfactual evidence-plan verifier.

Only fake IDs, hashes, and metadata are emitted. No benchmark targets or source text
are stored. This script is a test-vector generator, not a production evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.counterfactual_evidence.v1\0"
SCHEMA = "symthaea.science_bench_refresh_counterfactual_evidence_plan.v1"
CONDITIONS = [
    "natural_retrieval",
    "network_isolated",
    "gold_evidence_injected",
    "gold_evidence_withheld",
    "decoy_noise",
    "source_contradiction",
]
ORDER = [3, 0, 5, 2, 1, 4]


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


def make_fields() -> dict[str, Any]:
    pair_id = "synthetic-pair-001"
    trials = []
    for index, condition in enumerate(CONDITIONS):
        trials.append({
            "trial_id": f"synthetic-trial-{index:02d}",
            "condition": condition,
            "policy_input_sha256": sha256(f"policy-input:{pair_id}:{condition}".encode()),
            "evidence_manifest_sha256": sha256(f"evidence-manifest:{pair_id}:{condition}".encode()),
            "randomization_order": ORDER[index],
            "seed_commitment_sha256": sha256(f"seed-commitment:{pair_id}:{condition}".encode()),
        })
    return {
        "schema": SCHEMA,
        "generation_id": "synthetic-generation-001",
        "plan_id": "synthetic-counterfactual-plan-001",
        "policy_revision_sha256": sha256(b"frozen-policy-revision"),
        "configuration_sha256": sha256(b"frozen-agent-configuration"),
        "prompt_sha256": sha256(b"frozen-prompt"),
        "tool_manifest_sha256": sha256(b"frozen-tool-manifest"),
        "budget_sha256": sha256(b"frozen-resource-budget"),
        "analysis_plan_sha256": sha256(b"preregistered-analysis-plan"),
        "randomization_commitment_sha256": sha256(b"randomization-secret-commitment"),
        "required_conditions": CONDITIONS.copy(),
        "pairs": [{
            "pair_id": pair_id,
            "task_ref_sha256": sha256(b"opaque-synthetic-task-reference"),
            "task_family_id": "counterfactual-evidence-test",
            "trials": trials,
        }],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: counterfactual_evidence_test_vectors.py OUTPUT_DIR", file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)

    fields = make_fields()
    payload = encode(fields)
    commitment = framed_hash(b"plan.v1", payload)
    valid_plan = {**fields, "plan_commitment_sha256": commitment}
    write_json(output / "plan-fields.json", fields)
    write_json(output / "valid-plan.json", valid_plan)

    # Recompute a commitment over a deliberately incomplete plan. Structural
    # incompleteness—not a stale hash—must be sufficient to reject the plan.
    missing = copy.deepcopy(fields)
    missing["pairs"][0]["trials"].pop()
    missing_commitment = framed_hash(b"plan.v1", encode(missing))
    write_json(output / "missing-condition-plan.json", {
        **missing,
        "plan_commitment_sha256": missing_commitment,
    })

    # Structurally intact, but the commitment was edited after freezing.
    mutated = copy.deepcopy(valid_plan)
    mutated["plan_commitment_sha256"] = "0" * 64
    write_json(output / "mutated-commitment-plan.json", mutated)

    print(json.dumps({
        "schema": "symthaea.counterfactual_evidence_test_vectors.v1",
        "expected_plan_commitment_sha256": commitment,
        "expected_pair_count": 1,
        "expected_trial_count": len(CONDITIONS),
        "expected_conditions_per_pair": len(CONDITIONS),
        "output_dir": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
