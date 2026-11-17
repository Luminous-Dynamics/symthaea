#!/usr/bin/env python3
"""Generate a deterministic two-rubric structural commensurability fixture."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "symthaea.science_bench_refresh_rubric_commensurability_plan.v1"
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def criterion(
    criterion_id: str,
    construct_id: str,
    weight: int,
    measurement_spec: str,
    *,
    lower_is_better: bool = False,
    scale_factor: int = 1,
    panel: str,
) -> dict[str, Any]:
    if lower_is_better:
        anchors = [
            {"raw_value": 0, "normalized_score_millionths": 1_000_000, "semantic_anchor_sha256": digest("excellent")},
            {"raw_value": 5 * scale_factor, "normalized_score_millionths": 500_000, "semantic_anchor_sha256": digest("adequate")},
            {"raw_value": 10 * scale_factor, "normalized_score_millionths": 0, "semantic_anchor_sha256": digest("poor")},
        ]
        maximum = 10 * scale_factor
        direction = "lower_raw_is_better"
    else:
        anchors = [
            {"raw_value": 0, "normalized_score_millionths": 0, "semantic_anchor_sha256": digest("poor")},
            {"raw_value": 10 * scale_factor, "normalized_score_millionths": 500_000, "semantic_anchor_sha256": digest("adequate")},
            {"raw_value": 20 * scale_factor, "normalized_score_millionths": 1_000_000, "semantic_anchor_sha256": digest("excellent")},
        ]
        maximum = 20 * scale_factor
        direction = "higher_raw_is_better"
    return {
        "criterion_id": criterion_id,
        "definition_sha256": digest(f"criterion-definition:{construct_id}"),
        "measurement_spec_sha256": digest(measurement_spec),
        "score_min_raw": 0,
        "score_max_raw": maximum,
        "score_direction": direction,
        "scale_anchors": anchors,
        "weight_millionths": weight,
        "construct_mapping": {
            "construct_id": construct_id,
            "relation": "equivalent",
            "mapping_evidence_sha256": digest(f"{panel}:mapping-evidence:{construct_id}"),
            "adjudicator_id_sha256": digest(f"{panel}:independent-adjudicator"),
        },
    }


def build_constructs() -> list[dict[str, str]]:
    return [
        {
            "construct_id": "factual_accuracy",
            "definition_sha256": digest("construct:factual-accuracy:definition:v1"),
            "scope_sha256": digest("construct:factual-accuracy:scope:v1"),
        },
        {
            "construct_id": "evidence_support",
            "definition_sha256": digest("construct:evidence-support:definition:v1"),
            "scope_sha256": digest("construct:evidence-support:scope:v1"),
        },
    ]


def build_fields(constructs: list[dict[str, str]], ontology_sha256: str) -> dict[str, Any]:
    left = {
        "rubric_id": "rubric-independent-left",
        "rubric_revision": "v1",
        "rubric_sha256": digest("rubric-left-exact-bytes"),
        "criteria": [
            criterion("accuracy-left", "factual_accuracy", 600_000, "accuracy-measurement-spec-v1", panel="left"),
            criterion("evidence-left", "evidence_support", 400_000, "evidence-measurement-spec-v1", lower_is_better=True, panel="left"),
        ],
    }
    right = {
        "rubric_id": "rubric-independent-right",
        "rubric_revision": "v4",
        "rubric_sha256": digest("rubric-right-exact-bytes"),
        "criteria": [
            criterion("accuracy-right", "factual_accuracy", 600_000, "accuracy-measurement-spec-v1", scale_factor=5, panel="right"),
            criterion("evidence-right", "evidence_support", 400_000, "evidence-measurement-spec-v1", lower_is_better=True, scale_factor=10, panel="right"),
        ],
    }
    anchors = []
    for construct_id in ("factual_accuracy", "evidence_support"):
        for ordinal in range(2):
            anchor_id = f"{construct_id}-anchor-{ordinal + 1}"
            anchors.append({
                "anchor_id": anchor_id,
                "construct_id": construct_id,
                "semantics_sha256": digest(f"shared-anchor-semantics:{anchor_id}"),
            })
    return {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": git_revision(),
        "purpose": "qualified_run",
        "comparison_id": "rubric-commensurability-synthetic-001",
        "construct_ontology_sha256": ontology_sha256,
        "constructs": constructs,
        "left_instrument": left,
        "right_instrument": right,
        "shared_anchor_items": anchors,
        "minimum_anchor_items_per_construct": 2,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: rubric_commensurability_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    constructs = build_constructs()
    constructs_path = output_dir / "construct-ontology.example.json"
    constructs_path.write_text(
        json.dumps(constructs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ontology_command = [
        "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
        "--", "commit-rubric-construct-ontology", str(constructs_path),
    ]
    ontology_result = subprocess.run(
        ontology_command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    ontology_sha256 = json.loads(ontology_result.stdout)["construct_ontology_sha256"]
    fields = build_fields(constructs, ontology_sha256)
    fields_path = output_dir / "rubric-commensurability-plan-fields.example.json"
    fields_path.write_text(json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [
        "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
        "--", "commit-rubric-commensurability-plan", str(fields_path),
    ]
    committed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    commitment = json.loads(committed.stdout)["plan_commitment_sha256"]
    plan = {**fields, "plan_commitment_sha256": commitment}
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    plan_path = output_dir / "rubric-commensurability-plan.example.json"
    plan_path.write_bytes(plan_bytes)
    print(json.dumps({
        "schema": "symthaea.science_bench_refresh_rubric_commensurability_fixture_summary.v1",
        "synthetic_structural_positive_control": True,
        "rubric_instrument_count": 2,
        "construct_count": len(fields["constructs"]),
        "shared_anchor_item_count": len(fields["shared_anchor_items"]),
        "minimum_anchor_items_per_construct": fields["minimum_anchor_items_per_construct"],
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_commitment_sha256": commitment,
        "construct_validity_proven": False,
        "human_alignment_proven": False,
        "scoring_eligible": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
