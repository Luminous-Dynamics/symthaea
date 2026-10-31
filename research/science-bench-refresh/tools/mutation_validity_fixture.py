#!/usr/bin/env python3
"""Generate a deterministic synthetic semantic-mutation qualification fixture.

The fixture contains valid, invalid, disputed, abstained and unreviewed variants.
It never executes a judge; evaluator outcomes are planted receipts for analyzer QA.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "symthaea.science_bench_refresh_mutation_validity_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_mutation_validity_receipt.v1"
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


def artifact(label: str, modality: str = "text") -> dict[str, Any]:
    content = ("synthetic-artifact:" + label).encode()
    return {
        "artifact_id": label,
        "modality": modality,
        "content_sha256": digest(content),
        "byte_length": len(content),
    }


def make_variant(
    mutation_id: str,
    pair_id: str,
    family: str,
    relation: str,
    *,
    target_dimension: str | None = None,
    modality: str = "text",
) -> dict[str, Any]:
    return {
        "mutation_id": mutation_id,
        "metamorphic_pair_id": pair_id,
        "family": family,
        "intended_relation": relation,
        "target_dimension": target_dimension,
        "baseline_input_sha256": digest((mutation_id + ":baseline-input").encode()),
        "variant_input_sha256": digest((mutation_id + ":variant-input").encode()),
        "baseline_artifacts": [artifact(mutation_id + "-baseline", modality)],
        "variant_artifacts": [artifact(mutation_id + "-variant", modality)],
    }


def rating(mutation_id: str, reviewer_index: int, decision: str) -> dict[str, Any]:
    return {
        "adjudicator_id_sha256": digest(f"reviewer:{mutation_id}:{reviewer_index}".encode()),
        "independence_group_sha256": digest(f"independence:{mutation_id}:{reviewer_index}".encode()),
        "decision": decision,
        "confidence_millionths": 900_000 - reviewer_index * 50_000,
        "rationale_sha256": digest(f"rationale:{mutation_id}:{reviewer_index}:{decision}".encode()),
    }


def make_observation(
    variant: dict[str, Any],
    decisions: list[str],
    *,
    passed: bool | None,
    failure_fingerprint: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    mutation_id = variant["mutation_id"]
    completed = status == "completed"
    return {
        "mutation_id": mutation_id,
        "baseline_input_sha256": variant["baseline_input_sha256"],
        "variant_input_sha256": variant["variant_input_sha256"],
        "adjudications": [
            rating(mutation_id, index, decision)
            for index, decision in enumerate(decisions)
        ],
        "evaluation_status": status,
        "evaluator_relation_pass": passed if completed else None,
        "failure_fingerprint_sha256": (
            digest(("failure:" + failure_fingerprint).encode())
            if completed and failure_fingerprint is not None else None
        ),
        "generator_call_count": 1,
        "generator_token_count": 1_000,
        "semantic_assistance_call_count": 1,
        "semantic_assistance_token_count": 200,
        "evaluator_call_count": 1 if completed else 0,
        "evaluator_token_count": 300 if completed else 0,
        "human_review_seconds": 120 if decisions else 0,
        "wall_time_milliseconds": 1_500,
    }


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        make_variant("mutation-preserve-valid-001", "meta-pair-001", "paraphrase", "preserving"),
        make_variant("mutation-change-valid-001", "meta-pair-002", "unit_change", "changing", target_dimension="measurement_unit"),
        make_variant("mutation-preserve-invalid-001", "meta-pair-003", "entity_substitution", "preserving"),
        make_variant("mutation-preserve-disputed-001", "meta-pair-004", "formatting", "preserving"),
        make_variant("mutation-change-unreviewed-001", "meta-pair-005", "scope_change", "changing", target_dimension="jurisdiction"),
        make_variant("mutation-preserve-abstained-001", "meta-pair-006", "multimodal_composition", "preserving", modality="multimodal"),
        make_variant("mutation-preserve-valid-duplicate-failure-001", "meta-pair-007", "paraphrase", "preserving"),
    ]
    plan_fields = {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": git_revision(),
        "purpose": "example_only",
        "study_id": "synthetic-mutation-validity-001",
        "generator_source_revision": "b" * 40,
        "evaluator_source_revision": "c" * 40,
        "generator_policy_sha256": digest(b"synthetic mutation generation policy"),
        "semantic_task_set_sha256": digest(b"synthetic semantic task set"),
        "review_protocol_sha256": digest(b"synthetic independent review protocol"),
        "metamorphic_plan_sha256": digest(b"synthetic metamorphic plan"),
        "evaluator_prompt_sha256": digest(b"synthetic evaluator prompt"),
        "rubric_sha256": digest(b"synthetic evaluator rubric"),
        "reviewers_per_mutation": 3,
        "variants": variants,
    }
    fields_path = output_dir / "mutation-validity-plan-fields.example.json"
    fields_path.write_text(json.dumps(plan_fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [
        "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
        "--", "commit-mutation-validity-plan", str(fields_path),
    ]
    committed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    commitment = json.loads(committed.stdout)["plan_commitment_sha256"]
    plan = {**plan_fields, "plan_commitment_sha256": commitment}
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    plan_path = output_dir / "mutation-validity-plan.example.json"
    plan_path.write_bytes(plan_bytes)

    observations = [
        make_observation(variants[0], ["preserves_semantics"] * 3, passed=False, failure_fingerprint="duplicate-fault-A"),
        make_observation(variants[1], ["changes_semantics"] * 3, passed=True),
        make_observation(variants[2], ["changes_semantics"] * 3, passed=False, failure_fingerprint="invalid-mutation-fault"),
        make_observation(
            variants[3],
            ["preserves_semantics", "changes_semantics", "indeterminate"],
            passed=False,
            failure_fingerprint="disputed-mutation-fault",
        ),
        make_observation(variants[4], [], passed=None, status="not_started"),
        make_observation(variants[5], ["abstain"] * 3, passed=True),
        make_observation(variants[6], ["preserves_semantics"] * 3, passed=False, failure_fingerprint="duplicate-fault-A"),
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "analyzer_source_revision": plan_fields["analyzer_source_revision"],
        "purpose": "example_only",
        "run_id": "synthetic-run-not-authenticated-001",
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
        "metamorphic_receipt_sha256": digest(b"synthetic metamorphic outcome receipt"),
        "generator_source_revision": plan_fields["generator_source_revision"],
        "evaluator_source_revision": plan_fields["evaluator_source_revision"],
        "evaluator_prompt_sha256": plan_fields["evaluator_prompt_sha256"],
        "rubric_sha256": plan_fields["rubric_sha256"],
        "observations": observations,
    }
    receipt_path = output_dir / "mutation-validity-receipt.example.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "schema": "symthaea.science_bench_refresh_mutation_validity_fixture_summary.v1",
        "synthetic_positive_control": True,
        "planned_variants": len(variants),
        "reviewer_slots": len(variants) * plan_fields["reviewers_per_mutation"],
        "confirmed_valid_preserving": 2,
        "confirmed_valid_changing": 1,
        "confirmed_invalid": 1,
        "disputed": 1,
        "unreviewed": 1,
        "abstained": 1,
        "evaluator_failures": 4,
        "verified_failure_observations": 2,
        "unique_verified_failure_fingerprints": 1,
        "duplicate_verified_failure_observations": 1,
        "invalid_and_unconfirmed_failures_excluded": 2,
        "plan_sha256": digest(plan_bytes),
        "plan_commitment_sha256": commitment,
        "raw_artifact_custody_proven": False,
        "reviewer_identity_authenticated": False,
        "reviewer_blinding_proven": False,
        "panel_reference_validity_proven": False,
        "evaluator_execution_proven": False,
        "causal_effect_identified": False,
        "scoring_eligible": False,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mutation_validity_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    print(json.dumps(generate(Path(sys.argv[1])), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
