#!/usr/bin/env python3
"""Generate a deterministic target-delta fixture composed with quality v1."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from intervention_quality_fixture import generate as generate_quality

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"
PLAN_SCHEMA = "symthaea.science_bench_refresh_intervention_target_delta_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_intervention_target_delta_receipt.v1"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def run_cli(*arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
            "--", *arguments,
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def reviewer_panel(
    candidate_id: str,
    target_identity_sha256: str,
    baseline_artifact_sha256: str,
    variant_artifact_sha256: str,
    observed_change: str,
    changed_claim_ids: list[str],
    collateral: str = "none_observed",
) -> list[dict[str, Any]]:
    return [
        {
            "reviewer_id_sha256": digest(f"target-reviewer:{candidate_id}:{index}".encode()),
            "reviewed_target_identity_sha256": target_identity_sha256,
            "baseline_artifact_sha256": baseline_artifact_sha256,
            "variant_artifact_sha256": variant_artifact_sha256,
            "observed_target_change": observed_change,
            "observed_changed_claim_ids_sha256": changed_claim_ids,
            "collateral_change": collateral,
            "blind_to_generator_identity": True,
            "blind_to_intended_relation": True,
            "blind_to_pipeline_disposition": True,
            "blind_to_evaluator_outcome": True,
            "blind_to_expected_target_change": True,
            "blind_to_allowed_changed_claims": True,
            "adjudication_artifact_sha256": digest(f"target-adjudication:{candidate_id}:{index}".encode()),
        }
        for index in range(3)
    ]


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_quality(output_dir)
    quality_plan_path = output_dir / "intervention-quality-plan.example.json"
    quality_receipt_path = output_dir / "intervention-quality-receipt.example.json"
    quality_plan_bytes = quality_plan_path.read_bytes()
    quality_receipt_bytes = quality_receipt_path.read_bytes()
    quality_plan = json.loads(quality_plan_bytes)
    candidates = quality_plan["candidates"]

    target_candidates = []
    target_observations = []
    expected = {
        "case-001": ("unchanged", None),
        "case-002": ("unchanged", None),
        "case-003": ("support_decreased", "support_decreased"),
        "case-004": ("contradiction_introduced", "contradiction_introduced"),
        "case-005": ("unchanged", None),
        "case-006": ("unit_changed", "unit_changed"),
        "case-007": ("source_withdrawn", "source_withdrawn"),
    }

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relation, _ = expected[candidate_id]
        target_id = digest(("target-claim:" + candidate_id).encode())
        control_id = digest(("preserved-control:" + candidate_id).encode())
        baseline_state = digest(("target-state:" + candidate_id + ":baseline").encode())
        variant_state = baseline_state if relation == "unchanged" else digest(
            ("target-state:" + candidate_id + ":variant").encode()
        )
        if relation == "unchanged":
            changed_ids: list[str] = []
            preserved_ids = sorted([target_id, control_id])
            expected_target_id = target_id
        else:
            changed_ids = [target_id]
            preserved_ids = [control_id]
            expected_target_id = target_id

        baseline_artifact = candidate["baseline_input_sha256"]
        variant_artifact = candidate["variant_input_sha256"]
        target_candidates.append({
            "candidate_id": candidate_id,
            "target_identity_sha256": expected_target_id,
            "baseline_artifact_sha256": baseline_artifact,
            "variant_artifact_sha256": variant_artifact,
            "baseline_target_state_sha256": baseline_state,
            "variant_target_state_sha256": variant_state,
            "expected_target_change": relation,
            "expected_changed_claim_ids_sha256": changed_ids,
            "preserved_claim_ids_sha256": preserved_ids,
        })

        if candidate_id == "case-005":
            # A deliberately non-consensus panel. Global quality is also
            # ambiguous for this candidate; neither ambiguity is promoted.
            panels = [
                reviewer_panel(candidate_id, expected_target_id, baseline_artifact, variant_artifact, "unchanged", [], "none_observed")[0],
                reviewer_panel(candidate_id, expected_target_id, baseline_artifact, variant_artifact, "unchanged", [digest(b"unexpected claim one")], "none_observed")[1],
                reviewer_panel(candidate_id, expected_target_id, baseline_artifact, variant_artifact, "unchanged", [digest(b"unexpected claim two")], "none_observed")[2],
            ]
        else:
            observed_change = relation
            panels = reviewer_panel(candidate_id, expected_target_id, baseline_artifact, variant_artifact, observed_change, changed_ids)

        target_observations.append({
            "candidate_id": candidate_id,
            "reviewers": panels,
        })

    plan_fields = {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": git_revision(),
        "purpose": "example_only",
        "study_id": quality_plan["study_id"],
        "quality_plan_sha256": digest(quality_plan_bytes),
        "quality_plan_commitment_sha256": quality_plan["plan_commitment_sha256"],
        "reviewers_per_candidate": 3,
        "consensus_threshold_permille": 666,
        "candidates": target_candidates,
    }
    fields_path = output_dir / "intervention-target-delta-plan-fields.example.json"
    fields_path.write_text(
        json.dumps(plan_fields, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    commitment = run_cli(
        "commit-intervention-target-delta-plan",
        str(fields_path),
        str(quality_plan_path),
    )["plan_commitment_sha256"]
    plan = {**plan_fields, "plan_commitment_sha256": commitment}
    target_plan_bytes = (
        json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    target_plan_path = output_dir / "intervention-target-delta-plan.example.json"
    target_plan_path.write_bytes(target_plan_bytes)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "analyzer_source_revision": plan_fields["analyzer_source_revision"],
        "purpose": "example_only",
        "run_id": "target-delta-synthetic-not-authenticated-001",
        "target_plan_sha256": digest(target_plan_bytes),
        "target_plan_commitment_sha256": commitment,
        "quality_plan_sha256": digest(quality_plan_bytes),
        "quality_receipt_sha256": digest(quality_receipt_bytes),
        "adjudications": target_observations,
    }
    target_receipt_path = output_dir / "intervention-target-delta-receipt.example.json"
    target_receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = run_cli(
        "analyze-intervention-target-delta",
        str(quality_plan_path),
        str(quality_receipt_path),
        str(target_plan_path),
        str(target_receipt_path),
    )
    report_path = output_dir / "intervention-target-delta-report.example.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "symthaea.science_bench_refresh_intervention_target_delta_fixture_summary.v1",
        "synthetic_positive_control": True,
        "parent_quality_candidate_count": len(candidates),
        "target_delta_candidate_count": len(target_candidates),
        "target_delta_adjudicated_candidate_count": len(target_observations),
        "reviewers_per_candidate": 3,
        "global_relation_valid_count": report["global_relation_valid_count"],
        "global_relation_invalid_count": report["global_relation_invalid_count"],
        "global_relation_ambiguous_count": report["global_relation_ambiguous_count"],
        "target_delta_valid_count": report["target_delta_valid_count"],
        "target_delta_invalid_count": report["target_delta_invalid_count"],
        "target_delta_ambiguous_count": report["target_delta_ambiguous_count"],
        "final_valid_count": report["final_valid_count"],
        "final_invalid_count": report["final_invalid_count"],
        "final_ambiguous_count": report["final_ambiguous_count"],
        "reviewer_independence_proven": False,
        "semantic_truth_proven": False,
        "execution_authenticity_proven": False,
        "scoring_eligible": False,
        "target_plan_sha256": digest(target_plan_bytes),
        "target_plan_commitment_sha256": commitment,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: intervention_target_delta_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    print(json.dumps(generate(Path(sys.argv[1])), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
