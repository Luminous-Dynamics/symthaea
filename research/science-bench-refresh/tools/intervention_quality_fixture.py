#!/usr/bin/env python3
"""Generate a deterministic, explicitly non-scoring intervention-quality fixture."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"
PLAN_SCHEMA = "symthaea.science_bench_refresh_intervention_quality_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_intervention_quality_receipt.v1"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def vote_panel(candidate_id: str, verdicts: list[str]) -> list[dict[str, Any]]:
    panel = []
    for index, verdict in enumerate(verdicts):
        panel.append({
            "reviewer_id_sha256": digest(f"reviewer:{candidate_id}:{index}".encode()),
            "verdict": verdict,
            "blind_to_generator_identity": True,
            "blind_to_intended_relation": True,
            "blind_to_pipeline_disposition": True,
            "blind_to_evaluator_outcome": True,
            "adjudication_artifact_sha256": digest(f"adjudication:{candidate_id}:{index}".encode()),
        })
    return panel


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("case-001", "surface", "position", "cluster-1", "semantic_invariant", "accepted", ["meaning_preserving"] * 3),
        ("case-002", "surface", "format", "cluster-1", "semantic_invariant", "accepted", ["meaning_changing", "meaning_changing", "meaning_preserving"]),
        ("case-003", "evidence", "withdrawal", "cluster-2", "semantic_change", "accepted", ["meaning_changing"] * 3),
        ("case-004", "evidence", "conflict", "cluster-3", "semantic_change", "rejected", ["meaning_changing"] * 3),
        ("case-005", "authority", "provenance", "cluster-4", "semantic_invariant", "abstained", ["meaning_preserving", "meaning_changing", "ambiguous"]),
        ("case-006", "authority", "confidence", "cluster-5", "semantic_change", "rejected", ["meaning_changing", "ambiguous", "ambiguous"]),
        ("case-007", "authority", "reviewer-disagreement", "cluster-6", "semantic_change", "accepted", ["meaning_preserving", "meaning_changing", "ambiguous"]),
    ]
    plan_fields = {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": git_revision(),
        "purpose": "example_only",
        "study_id": "synthetic-intervention-quality-001",
        "generator_source_revision": "a" * 40,
        "generator_prompt_sha256": digest(b"synthetic generator prompt"),
        "semantic_verifier_source_revision": "b" * 40,
        "semantic_verifier_policy_sha256": digest(b"synthetic semantic verification policy"),
        "candidate_sampling_policy": "full_census",
        "adjudication_policy": {
            "reviewers_per_candidate": 3,
            "consensus_threshold_permille": 666,
            "require_blind_to_generator_identity": True,
            "require_blind_to_intended_relation": True,
            "require_blind_to_pipeline_disposition": True,
            "require_blind_to_evaluator_outcome": True,
        },
        "candidates": [],
    }
    adjudications = []
    for index, (candidate_id, family, failure_mode, cluster, relation, disposition, votes) in enumerate(candidates):
        plan_fields["candidates"].append({
            "candidate_id": candidate_id,
            "transformation_family": family,
            "target_failure_mode": failure_mode,
            "semantic_cluster_sha256": digest(cluster.encode()),
            "intended_relation": relation,
            "pipeline_disposition": disposition,
            "baseline_input_sha256": digest(f"{candidate_id}:baseline".encode()),
            "variant_input_sha256": digest(f"{candidate_id}:variant".encode()),
            "generation_tokens": 100,
            "generation_cost_micro_usd": 1000,
            "generation_wall_clock_ms": 50 + index,
        })
        adjudications.append({
            "candidate_id": candidate_id,
            "reviewers": vote_panel(candidate_id, votes),
        })

    fields_path = output_dir / "intervention-quality-plan-fields.example.json"
    fields_path.write_text(
        json.dumps(plan_fields, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
            "--", "commit-intervention-quality-plan", str(fields_path),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    commitment = json.loads(result.stdout)["plan_commitment_sha256"]
    plan = {**plan_fields, "plan_commitment_sha256": commitment}
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    plan_path = output_dir / "intervention-quality-plan.example.json"
    plan_path.write_bytes(plan_bytes)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "analyzer_source_revision": plan_fields["analyzer_source_revision"],
        "purpose": "example_only",
        "run_id": "synthetic-run-not-authenticated-001",
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
        "generator_source_revision": plan_fields["generator_source_revision"],
        "semantic_verifier_source_revision": plan_fields["semantic_verifier_source_revision"],
        "semantic_verifier_policy_sha256": plan_fields["semantic_verifier_policy_sha256"],
        "adjudications": adjudications,
    }
    receipt_path = output_dir / "intervention-quality-receipt.example.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "symthaea.science_bench_refresh_intervention_quality_fixture_summary.v1",
        "synthetic_positive_control": True,
        "full_census": True,
        "generated_candidates": len(candidates),
        "adjudicated_candidates": len(adjudications),
        "reviewers_per_candidate": 3,
        "ambiguous_candidates_preserved": 2,
        "empirical_human_adjudication": False,
        "reviewer_independence_proven": False,
        "semantic_truth_proven": False,
        "scoring_eligible": False,
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: intervention_quality_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    print(json.dumps(generate(Path(sys.argv[1])), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
