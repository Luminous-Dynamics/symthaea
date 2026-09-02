#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_qualified as qualified

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
ABLATION_POLICIES = {
    "no_embodied_experience",
    "no_persistent_memory",
    "no_depth",
    "no_counterfactual_evaluation",
    "no_independent_proposal_replay",
    "no_reality_ledger_context",
    "random_valid_judgment",
}


def _require_dict(value: Any, code: str, detail: str) -> dict[str, Any]:
    core.require(isinstance(value, dict), code, detail)
    return value


def _fixture_inventory(root: Path, freeze: dict[str, Any], campaign: str) -> set[str]:
    if campaign == "confirmatory_generalization":
        name = "generalization_fixture_inventory.json"
        expected = freeze.get("generalization_fixture_set_sha256")
    else:
        name = "fixture_inventory.json"
        expected = freeze.get("fixture_set_sha256")
    expected = core.require_sha256(expected, f"freeze.{name}")
    path = root / name
    actual = core.sha256_file(path)
    core.require(
        actual == expected,
        "FROZEN_FIXTURE_MISMATCH",
        f"{name}: {actual} != {expected}",
    )
    inventory = _require_dict(core.read_json(path), "FROZEN_FIXTURE_MISMATCH", name)
    fixtures = inventory.get("fixture_sha256")
    core.require(
        isinstance(fixtures, list)
        and fixtures
        and all(isinstance(v, str) for v in fixtures),
        "FROZEN_FIXTURE_MISMATCH",
        f"{name}: fixture_sha256",
    )
    return {core.require_sha256(v, f"{name}.fixture") for v in fixtures}


def _expected_policy_digest(freeze: dict[str, Any], policy: str) -> str:
    key = "ablation_policy_digests" if policy in ABLATION_POLICIES else "policy_digests"
    mapping = _require_dict(
        freeze.get(key), "FROZEN_POLICY_IMPLEMENTATION_MISMATCH", f"freeze.{key}"
    )
    return core.require_sha256(mapping.get(policy), f"freeze.{key}.{policy}")


def _verify_context_file(
    root: Path,
    trial_dir: Path,
    manifest: dict[str, Any],
    inventory_contexts: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    trial_id = manifest["trial_id"]
    manifest_context_sha = core.require_sha256(
        manifest.get("execution_context_sha256"), "execution_context_sha256"
    )
    expected_context_sha = core.require_sha256(
        inventory_contexts.get(trial_id), f"trial_contexts.{trial_id}"
    )
    core.require(
        manifest_context_sha == expected_context_sha,
        "EXECUTION_CONTEXT_INVENTORY_MISMATCH",
        f"{trial_id}: manifest context {manifest_context_sha} != frozen {expected_context_sha}",
    )

    idx = _require_dict(
        core.read_json(trial_dir / "evidence_index.json"),
        "EXECUTION_CONTEXT_DIGEST_MISMATCH",
        f"{trial_id}: evidence index",
    )
    context_path = core.file_from_index(trial_dir, idx, "execution_context")
    actual_context_sha = core.sha256_file(context_path)
    core.require(
        actual_context_sha == manifest_context_sha,
        "EXECUTION_CONTEXT_DIGEST_MISMATCH",
        f"{trial_id}: {actual_context_sha} != {manifest_context_sha}",
    )
    context = _require_dict(
        core.read_json(context_path),
        "EXECUTION_CONTEXT_DIGEST_MISMATCH",
        f"{trial_id}: context object",
    )
    core.require(
        context.get("schema")
        == "symthaea.vart-world-creative-001.execution-context.v1"
        and context.get("experiment_id") == EXPERIMENT_ID,
        "EXECUTION_CONTEXT_DIGEST_MISMATCH",
        f"{trial_id}: context schema/experiment",
    )

    identity_fields = (
        "campaign",
        "trial_id",
        "paired_block_id",
        "policy",
        "world_fixture_sha256",
        "seed",
        "revision_index",
        "metric_definition_set_sha256",
        "analysis_contract_sha256",
    )
    for field in identity_fields:
        core.require(
            context.get(field) == manifest.get(field),
            "EXECUTION_CONTEXT_DIGEST_MISMATCH",
            f"{trial_id}: context/manifest {field}",
        )

    policy_digest = _expected_policy_digest(freeze, manifest["policy"])
    core.require(
        manifest.get("policy_sha256") == policy_digest
        and context.get("policy_sha256") == policy_digest,
        "FROZEN_POLICY_IMPLEMENTATION_MISMATCH",
        trial_id,
    )

    fixtures = _fixture_inventory(root, freeze, manifest["campaign"])
    core.require(
        manifest["world_fixture_sha256"] in fixtures,
        "FROZEN_FIXTURE_MISMATCH",
        trial_id,
    )

    source = _require_dict(freeze.get("source"), "FROZEN_SOURCE_MISMATCH", "freeze.source")
    frozen_head = source.get("head")
    frozen_tree = source.get("tree")
    core.require(
        isinstance(frozen_head, str)
        and isinstance(frozen_tree, str)
        and context.get("source_head") == frozen_head
        and context.get("source_tree") == frozen_tree,
        "FROZEN_SOURCE_MISMATCH",
        trial_id,
    )

    frozen_environment = core.require_sha256(
        freeze.get("environment_digest"), "freeze.environment_digest"
    )
    core.require(
        context.get("environment_digest") == frozen_environment,
        "FROZEN_ENVIRONMENT_MISMATCH",
        trial_id,
    )
    frozen_generator = core.require_sha256(
        freeze.get("candidate_generator_sha256"), "freeze.candidate_generator_sha256"
    )
    core.require(
        context.get("candidate_generator_sha256") == frozen_generator,
        "FROZEN_CANDIDATE_GENERATOR_MISMATCH",
        trial_id,
    )
    frozen_admission = core.require_sha256(
        freeze.get("physical_admission_policy_sha256"),
        "freeze.physical_admission_policy_sha256",
    )
    core.require(
        context.get("physical_admission_policy_sha256") == frozen_admission,
        "FROZEN_ADMISSION_POLICY_MISMATCH",
        trial_id,
    )
    frozen_schema = core.require_sha256(
        freeze.get("trial_manifest_schema_sha256"),
        "freeze.trial_manifest_schema_sha256",
    )
    core.require(
        context.get("trial_manifest_schema_sha256") == frozen_schema,
        "FROZEN_SCHEMA_MISMATCH",
        trial_id,
    )

    core.require(
        context.get("metric_definition_set_sha256")
        == freeze.get("metric_definition_set_sha256")
        and context.get("analysis_contract_sha256")
        == freeze.get("analysis_contract_sha256"),
        "ANALYSIS_CONTRACT_MISMATCH",
        f"{trial_id}: context analysis/metric binding",
    )

    # Candidate-set bytes must remain identical across paired decision policies, so
    # they bind only the shared generation/admission context, never policy/trial identity.
    candidate_path = core.file_from_index(trial_dir, idx, "candidate_set")
    candidate_set = _require_dict(
        core.read_json(candidate_path),
        "FROZEN_CANDIDATE_GENERATOR_MISMATCH",
        f"{trial_id}: candidate set",
    )
    core.require(
        candidate_set.get("paired_block_id") == manifest["paired_block_id"]
        and candidate_set.get("world_fixture_sha256") == manifest["world_fixture_sha256"]
        and candidate_set.get("seed") == manifest["seed"]
        and candidate_set.get("revision_index") == manifest["revision_index"],
        "PAIRED_BLOCK_IDENTITY_MISMATCH",
        f"{trial_id}: candidate set generation identity",
    )
    core.require(
        candidate_set.get("candidate_generator_sha256") == frozen_generator,
        "FROZEN_CANDIDATE_GENERATOR_MISMATCH",
        f"{trial_id}: candidate set generator",
    )
    core.require(
        candidate_set.get("physical_admission_policy_sha256") == frozen_admission,
        "FROZEN_ADMISSION_POLICY_MISMATCH",
        f"{trial_id}: candidate set admission policy",
    )

    if manifest.get("trial_state") == "complete":
        receipt_path = core.file_from_index(trial_dir, idx, "applied_receipt")
        receipt = _require_dict(
            core.read_json(receipt_path),
            "EXECUTION_CONTEXT_DIGEST_MISMATCH",
            f"{trial_id}: applied receipt",
        )
        core.require(
            receipt.get("execution_context_sha256") == manifest_context_sha,
            "EXECUTION_CONTEXT_DIGEST_MISMATCH",
            f"{trial_id}: receipt context",
        )


def verify_context_qualified(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    result = qualified.verify_qualified(root, expected_freeze_sha256)
    freeze = qualified.preflight_freeze(root, expected_freeze_sha256)
    inventory = _require_dict(
        core.read_json(root / "trial_inventory.json"),
        "EXECUTION_CONTEXT_INVENTORY_MISMATCH",
        "trial inventory",
    )
    trial_ids = inventory.get("trial_ids")
    contexts = _require_dict(
        inventory.get("trial_contexts"),
        "EXECUTION_CONTEXT_INVENTORY_MISMATCH",
        "trial_contexts",
    )
    core.require(
        isinstance(trial_ids, list) and set(contexts) == set(trial_ids),
        "EXECUTION_CONTEXT_INVENTORY_MISMATCH",
        "trial_contexts must exactly cover trial_ids",
    )

    for trial_id in trial_ids:
        trial_dir = root / "trials" / trial_id
        manifest = _require_dict(
            core.read_json(trial_dir / "manifest.json"),
            "EXECUTION_CONTEXT_DIGEST_MISMATCH",
            trial_id,
        )
        _verify_context_file(root, trial_dir, manifest, contexts, freeze)

    out = dict(result)
    out.update(
        {
            "prospective_execution_context": "PASS",
            "frozen_policy_implementation_binding": "PASS",
            "frozen_fixture_binding": "PASS",
            "frozen_source_environment_binding": "PASS",
            "frozen_generator_admission_binding": "PASS",
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prospectively frozen execution-context verifier for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_context_qualified(args.root, args.expected_freeze_sha256)
    except core.Reject as exc:
        payload = {"verdict": "REJECT", "reason_class": exc.code, "detail": exc.detail}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            f"ACCEPT: {result['trial_count']} confirmatory trials; "
            "prospective execution context PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
