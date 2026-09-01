#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import test_verify_vart_world_creative_001_context as c
import test_verify_vart_world_creative_001_n1_n20 as n
import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_state as state_verify


def refresh_candidate_selection(root: Path, trial_id: str) -> None:
    manifest = n.load(n.manifest_path(root, trial_id))
    cset_path = n.logical_path(root, trial_id, "candidate_set")
    candidate_sha = core.sha256_file(cset_path)
    n.update_manifest(root, trial_id, candidate_set_sha256=candidate_sha)
    proposal_shas = n.candidate_shas(root, trial_id)

    if manifest["policy"] == "random_valid":
        selection_index, counter, digest_hex = core.sha256_counter_draw(
            manifest["seed"],
            manifest["paired_block_id"],
            candidate_sha,
            manifest["admissible_candidate_count"],
        )
        selected_sha = proposal_shas[selection_index]
        idx = n.load(n.index_path(root, trial_id))
        idx["files"]["selected_proposal"] = f"candidate{selection_index}.json"
        n.save(n.index_path(root, trial_id), idx)
        draw_path = n.logical_path(root, trial_id, "random_draw_receipt")
        draw = n.load(draw_path)
        draw.update(
            {
                "candidate_set_sha256": candidate_sha,
                "selected_index": selection_index,
                "counter": counter,
                "accepted_digest_sha256": digest_hex,
            }
        )
        draw_sha = n.save(draw_path, draw)
        n.update_manifest(
            root,
            trial_id,
            selection_index=selection_index,
            selected_proposal_sha256=selected_sha,
            random_draw_receipt_sha256=draw_sha,
        )
    else:
        selected_sha = proposal_shas[manifest["selection_index"]]
        n.update_manifest(root, trial_id, selected_proposal_sha256=selected_sha)

    receipt_path = n.logical_path(root, trial_id, "applied_receipt")
    receipt = n.load(receipt_path)
    receipt["candidate_set_sha256"] = candidate_sha
    receipt["selected_proposal_sha256"] = selected_sha
    receipt_sha = n.save(receipt_path, receipt)
    n.update_manifest(root, trial_id, applied_receipt_sha256=receipt_sha)


def augment_state(root: Path, trial_id: str, before_state_digest: str, after_state_digest: str) -> str:
    manifest = n.load(n.manifest_path(root, trial_id))
    trial_dir = n.tdir(root, trial_id)

    before_obj = {
        "schema": "symthaea.vart-world-creative-001.world-state-snapshot.v1",
        "experiment_id": n.EXPERIMENT_ID,
        "world_version": manifest["world_version_before"],
        "provenance_domain": "digital_committed",
        "state_digest": before_state_digest,
    }
    after_obj = {
        "schema": "symthaea.vart-world-creative-001.world-state-snapshot.v1",
        "experiment_id": n.EXPERIMENT_ID,
        "world_version": manifest["world_version_after"],
        "provenance_domain": "digital_committed",
        "state_digest": after_state_digest,
    }
    before_sha = n.dump(trial_dir / "world_state_before.json", before_obj)
    after_sha = n.dump(trial_dir / "world_state_after.json", after_obj)

    decision_path = n.logical_path(root, trial_id, "decision_input")
    decision = n.load(decision_path)
    decision["world_state_before_sha256"] = before_sha
    decision_sha = n.save(decision_path, decision)

    context_path = trial_dir / "execution_context.json"
    execution_context = n.load(context_path)
    execution_context["world_state_before_sha256"] = before_sha
    context_sha = n.save(context_path, execution_context)

    cset_path = n.logical_path(root, trial_id, "candidate_set")
    candidate_set = n.load(cset_path)
    candidate_set["world_state_before_sha256"] = before_sha
    n.save(cset_path, candidate_set)
    refresh_candidate_selection(root, trial_id)

    receipt_path = n.logical_path(root, trial_id, "applied_receipt")
    receipt = n.load(receipt_path)
    receipt.update(
        {
            "decision_input_sha256": decision_sha,
            "execution_context_sha256": context_sha,
            "world_state_before_sha256": before_sha,
            "world_state_after_sha256": after_sha,
        }
    )
    receipt_sha = n.save(receipt_path, receipt)

    revisit_path = n.logical_path(root, trial_id, "revisit_observation")
    revisit = n.load(revisit_path)
    revisit["world_state_sha256"] = after_sha
    revisit_sha = n.save(revisit_path, revisit)

    idx = n.load(n.index_path(root, trial_id))
    idx["files"].update(
        {
            "world_state_before": "world_state_before.json",
            "world_state_after": "world_state_after.json",
        }
    )
    evidence_sha = n.save(n.index_path(root, trial_id), idx)

    n.update_manifest(
        root,
        trial_id,
        execution_context_sha256=context_sha,
        decision_input_sha256=decision_sha,
        world_state_before_sha256=before_sha,
        world_state_after_sha256=after_sha,
        applied_receipt_sha256=receipt_sha,
        revisit_observation_sha256=revisit_sha,
        evidence_bundle_sha256=evidence_sha,
    )
    return context_sha


def build_state_bundle(root: Path) -> str:
    c.build_context_bundle(root)

    contexts: dict[str, str] = {}
    paired_before_digest = "a" * 64
    for trial_id in [n.FULL, n.RANDOM, n.HEURISTIC]:
        contexts[trial_id] = augment_state(
            root,
            trial_id,
            paired_before_digest,
            n.sha_text(f"post-state:{trial_id}"),
        )
    contexts[n.GENERALIZATION] = augment_state(
        root,
        n.GENERALIZATION,
        "d" * 64,
        n.sha_text("post-state:generalization"),
    )

    inventory = n.load(root / "trial_inventory.json")
    inventory["trial_contexts"] = contexts
    inventory_sha = n.save(root / "trial_inventory.json", inventory)

    freeze = n.load(root / "confirmatory_freeze.json")
    freeze["trial_inventory_sha256"] = inventory_sha
    return n.save(root / "confirmatory_freeze.json", freeze)


def expect_reject(root: Path, freeze_sha: str, expected: str) -> None:
    try:
        state_verify.verify_state_qualified(root, freeze_sha)
    except core.Reject as exc:
        assert exc.code == expected, f"expected {expected}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {expected}")


def run_suite(base: Path, freeze_sha: str) -> None:
    result = state_verify.verify_state_qualified(base, freeze_sha)
    assert result["verdict"] == "ACCEPT", result
    assert result["paired_world_state_equivalence"] == "PASS"

    # S1 — mutate committed pre-state bytes after manifest closure.
    b = n.clone(base)
    path = n.logical_path(b, n.FULL, "world_state_before")
    state = n.load(path)
    state["state_digest"] = "e" * 64
    n.save(path, state)
    expect_reject(b, freeze_sha, "WORLD_STATE_DIGEST_MISMATCH")

    # S2 — state artifact hash is updated, but its version no longer matches the trial.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "world_state_before")
    state = n.load(path)
    state["world_version"] = "wOTHER"
    state_sha = n.save(path, state)
    n.update_manifest(b, n.GENERALIZATION, world_state_before_sha256=state_sha)
    expect_reject(b, freeze_sha, "WORLD_STATE_VERSION_MISMATCH")

    # S3 — a counterfactual snapshot cannot occupy committed pre-state position.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "world_state_before")
    state = n.load(path)
    state["provenance_domain"] = "counterfactual"
    state_sha = n.save(path, state)
    n.update_manifest(b, n.GENERALIZATION, world_state_before_sha256=state_sha)
    expect_reject(b, freeze_sha, "WORLD_STATE_PROVENANCE_SUBSTITUTION")

    # S4 — policy decision input claims another starting-state digest.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "decision_input")
    decision = n.load(path)
    decision["world_state_before_sha256"] = "e" * 64
    decision_sha = n.save(path, decision)
    n.update_manifest(b, n.GENERALIZATION, decision_input_sha256=decision_sha)
    n.rewrite_receipt(b, n.GENERALIZATION, lambda r: r.update(decision_input_sha256=decision_sha))
    expect_reject(b, freeze_sha, "WORLD_STATE_DIGEST_MISMATCH")

    # S5 — shared candidate surface claims another starting-state digest.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "candidate_set")
    cset = n.load(path)
    cset["world_state_before_sha256"] = "e" * 64
    csha = n.save(path, cset)
    n.update_manifest(b, n.GENERALIZATION, candidate_set_sha256=csha)
    n.rewrite_receipt(b, n.GENERALIZATION, lambda r: r.update(candidate_set_sha256=csha))
    expect_reject(b, freeze_sha, "WORLD_STATE_DIGEST_MISMATCH")

    # S6 — typed application receipt splices another pre-state.
    b = n.clone(base)
    n.rewrite_receipt(
        b,
        n.GENERALIZATION,
        lambda r: r.update(world_state_before_sha256="e" * 64),
    )
    expect_reject(b, freeze_sha, "WORLD_STATE_DIGEST_MISMATCH")

    # S7 — revisit is attached to another post-state digest.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "revisit_observation")
    revisit = n.load(path)
    revisit["world_state_sha256"] = "e" * 64
    revisit_sha = n.save(path, revisit)
    n.update_manifest(b, n.GENERALIZATION, revisit_observation_sha256=revisit_sha)
    expect_reject(b, freeze_sha, "WORLD_STATE_DIGEST_MISMATCH")


with tempfile.TemporaryDirectory(prefix="vart-state-") as td:
    base = Path(td) / "base"
    base.mkdir()
    freeze_sha = build_state_bundle(base)
    run_suite(base, freeze_sha)

print("PASS: VART world-state equivalence acceptance + S1-S7 deterministic rejection")
