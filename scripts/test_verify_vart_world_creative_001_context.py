#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import test_verify_vart_world_creative_001_n1_n20 as n
import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_context as context_verify

SOURCE_HEAD = "1" * 40
SOURCE_TREE = "2" * 40
ENVIRONMENT = n.sha_text("environment:nix-vart-v1")
GENERATOR = n.sha_text("candidate-generator:v1")
ADMISSION = n.sha_text("physical-admission:v1")
SCHEMA_SHA = n.sha_text("trial-manifest-schema:v1")


def update_freeze(root: Path, mutator) -> str:
    path = root / "confirmatory_freeze.json"
    freeze = n.load(path)
    mutator(freeze)
    return n.save(path, freeze)


def update_inventory(root: Path, mutator) -> str:
    path = root / "trial_inventory.json"
    inventory = n.load(path)
    mutator(inventory)
    return n.save(path, inventory)


def recompute_evidence_index(root: Path, trial_id: str) -> None:
    path = n.index_path(root, trial_id)
    digest = n.save(path, n.load(path))
    n.update_manifest(root, trial_id, evidence_bundle_sha256=digest)


def augment_trial(root: Path, trial_id: str, analysis_sha: str, metric_sha: str) -> str:
    manifest = n.load(n.manifest_path(root, trial_id))
    cset_path = n.logical_path(root, trial_id, "candidate_set")
    cset = n.load(cset_path)
    cset.update(
        {
            "paired_block_id": manifest["paired_block_id"],
            "world_fixture_sha256": manifest["world_fixture_sha256"],
            "seed": manifest["seed"],
            "revision_index": manifest["revision_index"],
            "candidate_generator_sha256": GENERATOR,
            "physical_admission_policy_sha256": ADMISSION,
        }
    )
    candidate_sha = n.save(cset_path, cset)
    n.update_manifest(root, trial_id, candidate_set_sha256=candidate_sha)

    proposal_shas = n.candidate_shas(root, trial_id)
    if manifest["policy"] == "random_valid":
        selection_index, counter, digest_hex = core.sha256_counter_draw(
            manifest["seed"],
            manifest["paired_block_id"],
            candidate_sha,
            manifest["admissible_candidate_count"],
        )
        selected_file = f"candidate{selection_index}.json"
        selected_sha = proposal_shas[selection_index]
        draw = n.load(n.logical_path(root, trial_id, "random_draw_receipt"))
        draw.update(
            {
                "candidate_set_sha256": candidate_sha,
                "counter": counter,
                "accepted_digest_sha256": digest_hex,
                "selected_index": selection_index,
            }
        )
        draw_sha = n.save(n.logical_path(root, trial_id, "random_draw_receipt"), draw)
        idx = n.load(n.index_path(root, trial_id))
        idx["files"]["selected_proposal"] = selected_file
        n.save(n.index_path(root, trial_id), idx)
        n.update_manifest(
            root,
            trial_id,
            selection_index=selection_index,
            selected_proposal_sha256=selected_sha,
            random_draw_receipt_sha256=draw_sha,
        )
    else:
        manifest = n.load(n.manifest_path(root, trial_id))
        selected_sha = proposal_shas[manifest["selection_index"]]

    receipt_path = n.logical_path(root, trial_id, "applied_receipt")
    receipt = n.load(receipt_path)
    receipt.update(
        {
            "candidate_set_sha256": candidate_sha,
            "selected_proposal_sha256": selected_sha,
        }
    )
    receipt_sha = n.save(receipt_path, receipt)
    n.update_manifest(root, trial_id, applied_receipt_sha256=receipt_sha)

    manifest = n.load(n.manifest_path(root, trial_id))
    execution_context = {
        "schema": "symthaea.vart-world-creative-001.execution-context.v1",
        "experiment_id": n.EXPERIMENT_ID,
        "campaign": manifest["campaign"],
        "trial_id": trial_id,
        "paired_block_id": manifest["paired_block_id"],
        "policy": manifest["policy"],
        "policy_sha256": manifest["policy_sha256"],
        "world_fixture_sha256": manifest["world_fixture_sha256"],
        "seed": manifest["seed"],
        "revision_index": manifest["revision_index"],
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "environment_digest": ENVIRONMENT,
        "candidate_generator_sha256": GENERATOR,
        "physical_admission_policy_sha256": ADMISSION,
        "metric_definition_set_sha256": metric_sha,
        "analysis_contract_sha256": analysis_sha,
        "trial_manifest_schema_sha256": SCHEMA_SHA,
    }
    context_sha = n.dump(n.tdir(root, trial_id) / "execution_context.json", execution_context)

    receipt = n.load(receipt_path)
    receipt["execution_context_sha256"] = context_sha
    receipt_sha = n.save(receipt_path, receipt)

    idx = n.load(n.index_path(root, trial_id))
    idx["files"]["execution_context"] = "execution_context.json"
    evidence_sha = n.save(n.index_path(root, trial_id), idx)
    n.update_manifest(
        root,
        trial_id,
        execution_context_sha256=context_sha,
        applied_receipt_sha256=receipt_sha,
        evidence_bundle_sha256=evidence_sha,
    )
    return context_sha


def build_context_bundle(root: Path) -> str:
    n.build(root)
    freeze = n.load(root / "confirmatory_freeze.json")
    analysis_sha = freeze["analysis_contract_sha256"]
    metric_sha = freeze["metric_definition_set_sha256"]

    fixture_a = "b" * 64
    fixture_g = "c" * 64
    fixture_set_sha = n.dump(
        root / "fixture_inventory.json",
        {
            "schema": "symthaea.vart-world-creative-001.fixture-inventory.v1",
            "fixture_sha256": [fixture_a],
        },
    )
    generalization_set_sha = n.dump(
        root / "generalization_fixture_inventory.json",
        {
            "schema": "symthaea.vart-world-creative-001.fixture-inventory.v1",
            "fixture_sha256": [fixture_g],
        },
    )

    trial_contexts: dict[str, str] = {}
    for trial_id in n.EXPECTED_ORDER:
        trial_contexts[trial_id] = augment_trial(root, trial_id, analysis_sha, metric_sha)

    inventory_sha = update_inventory(
        root,
        lambda inv: inv.update(trial_contexts=trial_contexts),
    )

    policy_digests = {
        "full_symthaea": n.sha_text("policy:full_symthaea:v1"),
        "random_valid": n.sha_text("policy:random_valid:v1"),
        "heuristic": n.sha_text("policy:heuristic:v1"),
    }
    freeze = n.load(root / "confirmatory_freeze.json")
    freeze.update(
        {
            "source": {"head": SOURCE_HEAD, "tree": SOURCE_TREE, "dirty": False},
            "environment_digest": ENVIRONMENT,
            "fixture_set_sha256": fixture_set_sha,
            "generalization_fixture_set_sha256": generalization_set_sha,
            "candidate_generator_sha256": GENERATOR,
            "physical_admission_policy_sha256": ADMISSION,
            "policy_digests": policy_digests,
            "ablation_policy_digests": {},
            "trial_manifest_schema_sha256": SCHEMA_SHA,
            "trial_inventory_sha256": inventory_sha,
        }
    )
    return n.save(root / "confirmatory_freeze.json", freeze)


def expect_reject(root: Path, freeze_sha: str, expected: str) -> None:
    try:
        context_verify.verify_context_qualified(root, freeze_sha)
    except core.Reject as exc:
        assert exc.code == expected, f"expected {expected}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {expected}")


def coordinated_context_mutation(root: Path, trial_id: str, mutator) -> str:
    context_path = n.tdir(root, trial_id) / "execution_context.json"
    obj = n.load(context_path)
    mutator(obj)
    context_sha = n.save(context_path, obj)
    n.update_manifest(root, trial_id, execution_context_sha256=context_sha)

    inventory_sha = update_inventory(
        root,
        lambda inv: inv["trial_contexts"].update({trial_id: context_sha}),
    )
    return update_freeze(root, lambda freeze: freeze.update(trial_inventory_sha256=inventory_sha))


def run_suite(base: Path, freeze_sha: str) -> None:
    result = context_verify.verify_context_qualified(base, freeze_sha)
    assert result["verdict"] == "ACCEPT", result
    assert result["prospective_execution_context"] == "PASS"

    # C1 — alter execution-context bytes without changing the frozen inventory digest.
    b = n.clone(base)
    path = n.tdir(b, n.FULL) / "execution_context.json"
    obj = n.load(path)
    obj["source_tree"] = "f" * 40
    n.save(path, obj)
    expect_reject(b, freeze_sha, "EXECUTION_CONTEXT_DIGEST_MISMATCH")

    # C2 — point the manifest at a context digest not prospectively assigned to the trial.
    b = n.clone(base)
    n.update_manifest(b, n.FULL, execution_context_sha256="e" * 64)
    expect_reject(b, freeze_sha, "EXECUTION_CONTEXT_INVENTORY_MISMATCH")

    # C3 — retain the policy label but substitute its implementation digest.
    b = n.clone(base)
    n.update_manifest(b, n.FULL, policy_sha256="e" * 64)
    expect_reject(b, freeze_sha, "FROZEN_POLICY_IMPLEMENTATION_MISMATCH")

    # C4 — a prospectively anchored context still cannot name a different source tree.
    b = n.clone(base)
    new_freeze = coordinated_context_mutation(
        b, n.FULL, lambda obj: obj.update(source_tree="f" * 40)
    )
    expect_reject(b, new_freeze, "FROZEN_SOURCE_MISMATCH")

    # C5 — same for the frozen runtime environment.
    b = n.clone(base)
    new_freeze = coordinated_context_mutation(
        b, n.FULL, lambda obj: obj.update(environment_digest="e" * 64)
    )
    expect_reject(b, new_freeze, "FROZEN_ENVIRONMENT_MISMATCH")

    # C6 — candidate-generation implementation substitution.
    b = n.clone(base)
    new_freeze = coordinated_context_mutation(
        b, n.FULL, lambda obj: obj.update(candidate_generator_sha256="e" * 64)
    )
    expect_reject(b, new_freeze, "FROZEN_CANDIDATE_GENERATOR_MISMATCH")

    # C7 — physical-admission implementation substitution.
    b = n.clone(base)
    new_freeze = coordinated_context_mutation(
        b, n.FULL, lambda obj: obj.update(physical_admission_policy_sha256="e" * 64)
    )
    expect_reject(b, new_freeze, "FROZEN_ADMISSION_POLICY_MISMATCH")

    # C8 — manifest-schema substitution.
    b = n.clone(base)
    new_freeze = coordinated_context_mutation(
        b, n.FULL, lambda obj: obj.update(trial_manifest_schema_sha256="e" * 64)
    )
    expect_reject(b, new_freeze, "FROZEN_SCHEMA_MISMATCH")

    # C9 — candidate-set metadata cannot claim another generator while retaining bytes.
    b = n.clone(base)
    cset = n.load(n.logical_path(b, n.FULL, "candidate_set"))
    cset["candidate_generator_sha256"] = "e" * 64
    csha = n.rewrite_logical(b, n.FULL, "candidate_set", cset, "candidate_set_sha256")
    n.rewrite_receipt(b, n.FULL, lambda r: r.update(candidate_set_sha256=csha))
    expect_reject(b, freeze_sha, "PAIRED_CANDIDATE_SET_MISMATCH")

    # C10 — changing the frozen fixture inventory bytes is independently rejected.
    b = n.clone(base)
    inventory = n.load(b / "fixture_inventory.json")
    inventory["fixture_sha256"].append("d" * 64)
    n.save(b / "fixture_inventory.json", inventory)
    expect_reject(b, freeze_sha, "FROZEN_FIXTURE_MISMATCH")


with tempfile.TemporaryDirectory(prefix="vart-context-") as td:
    base = Path(td) / "base"
    base.mkdir()
    freeze_sha = build_context_bundle(base)
    run_suite(base, freeze_sha)

print("PASS: VART prospective execution-context acceptance + C1-C10 deterministic rejection")
