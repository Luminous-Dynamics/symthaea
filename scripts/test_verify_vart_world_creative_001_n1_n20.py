#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_vart_world_creative_001 as core  # noqa: E402
import verify_vart_world_creative_001_qualified as qualified  # noqa: E402

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
FULL = "blockA:full"
RANDOM = "blockA:random"
HEURISTIC = "blockA:heuristic"
GENERALIZATION = "blockG:full"
EXPECTED_ORDER = [FULL, RANDOM, HEURISTIC, GENERALIZATION]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(obj))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, obj: Any) -> str:
    return dump(path, obj)


def clone(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="vart-n1-n20-")) / "bundle"
    shutil.copytree(src, dst)
    return dst


def tdir(root: Path, trial_id: str) -> Path:
    return root / "trials" / trial_id


def manifest_path(root: Path, trial_id: str) -> Path:
    return tdir(root, trial_id) / "manifest.json"


def index_path(root: Path, trial_id: str) -> Path:
    return tdir(root, trial_id) / "evidence_index.json"


def update_manifest(root: Path, trial_id: str, **fields: Any) -> dict[str, Any]:
    path = manifest_path(root, trial_id)
    m = load(path)
    m.update(fields)
    save(path, m)
    return m


def update_index(root: Path, trial_id: str, mutator) -> dict[str, Any]:
    path = index_path(root, trial_id)
    idx = load(path)
    mutator(idx)
    index_sha = save(path, idx)
    update_manifest(root, trial_id, evidence_bundle_sha256=index_sha)
    return idx


def logical_path(root: Path, trial_id: str, logical_name: str) -> Path:
    idx = load(index_path(root, trial_id))
    return tdir(root, trial_id) / idx["files"][logical_name]


def rewrite_logical(
    root: Path,
    trial_id: str,
    logical_name: str,
    obj: Any,
    manifest_field: str,
) -> str:
    digest = save(logical_path(root, trial_id, logical_name), obj)
    update_manifest(root, trial_id, **{manifest_field: digest})
    return digest


def rewrite_receipt(root: Path, trial_id: str, mutator) -> str:
    path = logical_path(root, trial_id, "applied_receipt")
    receipt = load(path)
    mutator(receipt)
    digest = save(path, receipt)
    update_manifest(root, trial_id, applied_receipt_sha256=digest)
    return digest


def expect_reject(root: Path, freeze_sha: str, expected: str) -> None:
    try:
        qualified.verify_qualified(root, freeze_sha)
    except core.Reject as exc:
        assert exc.code == expected, f"expected {expected}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {expected}")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_trial(
    root: Path,
    *,
    trial_id: str,
    policy: str,
    campaign: str,
    pair: str,
    seed: int,
    fixture_sha: str,
    analysis_sha: str,
    metric_sha: str,
    world_before: str,
    world_after: str,
    revision_index: int = 0,
) -> None:
    d = tdir(root, trial_id)
    d.mkdir(parents=True, exist_ok=True)

    # Primary paired policies receive byte-identical decision inputs.
    decision_input = {
        "schema": "symthaea.vart-world-creative-001.decision-input.v1",
        "experiment_id": EXPERIMENT_ID,
        "paired_block_id": pair,
        "seed": seed,
        "revision_index": revision_index,
        "world_version_before": world_before,
        "observation_bundle_sha256": "1" * 64,
    }
    decision_sha = dump(d / "decision_input.json", decision_input)
    experience_sha = dump(
        d / "experience.json",
        {
            "schema": "experience-v1",
            "world_version": world_before,
            "provenance_domain": "digital_committed",
        },
    )
    hypothesis_sha = dump(
        d / "hypothesis.json",
        {
            "schema": "revision-hypothesis-v1",
            "prospective": True,
            "predicted": {"declared_goal_consequence": 0.1},
        },
    )

    proposal_objects = [
        {"candidate_id": "c0", "edit": {"x": 1}},
        {"candidate_id": "c1", "edit": {"x": 2}},
        {"candidate_id": "c2", "edit": {"x": 3}},
    ]
    proposal_files = ["candidate0.json", "candidate1.json", "candidate2.json"]
    proposal_shas = [dump(d / name, obj) for name, obj in zip(proposal_files, proposal_objects)]
    candidate_set = {
        "schema": "candidate-set-v1",
        "candidates": [
            {"proposal_sha256": proposal_shas[0], "physically_admitted": True},
            {"proposal_sha256": proposal_shas[1], "physically_admitted": True},
            {"proposal_sha256": proposal_shas[2], "physically_admitted": False},
        ],
    }
    candidate_set_sha = dump(d / "candidate_set.json", candidate_set)

    if policy == "random_valid":
        selected_index, counter, digest_hex = core.sha256_counter_draw(
            seed, pair, candidate_set_sha, 2
        )
    elif policy == "heuristic":
        selected_index = 1
        counter = None
        digest_hex = None
    else:
        selected_index = 0
        counter = None
        digest_hex = None

    selected_sha = proposal_shas[selected_index]
    selected_file = proposal_files[selected_index]

    receipt_sha = dump(
        d / "receipt.json",
        {
            "schema": "applied-receipt-v1",
            "decision_input_sha256": decision_sha,
            "revision_hypothesis_sha256": hypothesis_sha,
            "candidate_set_sha256": candidate_set_sha,
            "selected_proposal_sha256": selected_sha,
            "world_version_before": world_before,
            "world_version_after": world_after,
        },
    )
    revisit_sha = dump(
        d / "revisit.json",
        {
            "schema": "revisit-v1",
            "world_version": world_after,
            "provenance_domain": "digital_committed",
        },
    )
    outcome_sha = dump(
        d / "outcome.json",
        {
            "schema": "revision-outcome-v1",
            "actual": {"declared_goal_consequence": 0.2},
        },
    )

    files = {
        "decision_input": "decision_input.json",
        "experience_episode": "experience.json",
        "revision_hypothesis": "hypothesis.json",
        "candidate_set": "candidate_set.json",
        "selected_proposal": selected_file,
        "applied_receipt": "receipt.json",
        "revisit_observation": "revisit.json",
        "revision_outcome": "outcome.json",
    }
    draw_sha = None
    if policy == "random_valid":
        draw_sha = dump(
            d / "draw.json",
            {
                "algorithm": "sha256-counter-v1",
                "seed": seed,
                "paired_block_id": pair,
                "candidate_set_sha256": candidate_set_sha,
                "admissible_candidate_count": 2,
                "counter": counter,
                "accepted_digest_sha256": digest_hex,
                "selected_index": selected_index,
            },
        )
        files["random_draw_receipt"] = "draw.json"

    evidence_index: dict[str, Any] = {
        "trial_id": trial_id,
        "files": files,
        "timestamps_ns": {
            "hypothesis_closed": 10,
            "selection_closed": 20,
            "applied_receipt": 30,
            "revisit_closed": 40,
            "outcome_closed": 50,
        },
        "cross_policy_outcome_observed_before_selection": False,
        "prospective_exclusion_reason_classes": [],
    }
    if campaign == "confirmatory_generalization":
        evidence_index.update(
            {
                "fixture_disclosed_to_policy_at_ns": 100,
                "generalization_fixture_reveal_at_ns": 100,
            }
        )
    evidence_sha = dump(d / "evidence_index.json", evidence_index)

    dump(
        d / "manifest.json",
        {
            "schema": "symthaea.vart-world-creative-001.trial-manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "campaign": campaign,
            "trial_id": trial_id,
            "paired_block_id": pair,
            "policy": policy,
            "policy_sha256": sha_text(f"policy:{policy}:v1"),
            "world_fixture_sha256": fixture_sha,
            "seed": seed,
            "revision_index": revision_index,
            "world_version_before": world_before,
            "decision_input_sha256": decision_sha,
            "experience_episode_sha256": experience_sha,
            "revision_hypothesis_sha256": hypothesis_sha,
            "candidate_set_sha256": candidate_set_sha,
            "generated_candidate_count": 3,
            "admissible_candidate_count": 2,
            "selected_proposal_sha256": selected_sha,
            "selection_index": selected_index,
            "random_draw_receipt_sha256": draw_sha,
            "ablation_receipt_sha256": None,
            "applied_receipt_sha256": receipt_sha,
            "world_version_after": world_after,
            "revisit_observation_sha256": revisit_sha,
            "revision_outcome_sha256": outcome_sha,
            "included_in_confirmatory_analysis": True,
            "exclusion_reason": None,
            "trial_state": "complete",
            "abort_stage": None,
            "integrity_violations": [],
            "metric_definition_set_sha256": metric_sha,
            "analysis_contract_sha256": analysis_sha,
            "evidence_bundle_sha256": evidence_sha,
        },
    )


def build(root: Path) -> str:
    analysis_sha = dump(
        root / "analysis_contract.json",
        {
            "schema": "symthaea.vart-world-creative-001.analysis-contract.v1",
            "status": "frozen",
            "primary_comparison": "full-minus-random-valid",
            "metric_direction": "higher_is_better",
        },
    )
    metric_sha = dump(
        root / "metric_definitions.json",
        {
            "schema": "symthaea.vart-world-creative-001.metric-definitions.v1",
            "metrics": [
                {"id": "declared_goal_consequence", "direction": "higher_is_better"}
            ],
        },
    )

    fixture_a = "b" * 64
    fixture_g = "c" * 64
    for trial_id, policy in [
        (FULL, "full_symthaea"),
        (RANDOM, "random_valid"),
        (HEURISTIC, "heuristic"),
    ]:
        build_trial(
            root,
            trial_id=trial_id,
            policy=policy,
            campaign="confirmatory_longitudinal",
            pair="blockA",
            seed=42,
            fixture_sha=fixture_a,
            analysis_sha=analysis_sha,
            metric_sha=metric_sha,
            world_before="wA0",
            world_after="wA1",
        )

    build_trial(
        root,
        trial_id=GENERALIZATION,
        policy="full_symthaea",
        campaign="confirmatory_generalization",
        pair="blockG",
        seed=77,
        fixture_sha=fixture_g,
        analysis_sha=analysis_sha,
        metric_sha=metric_sha,
        world_before="wG0",
        world_after="wG1",
    )

    inventory_sha = dump(
        root / "trial_inventory.json",
        {
            "schema": "symthaea.vart-world-creative-001.trial-inventory.v1",
            "trial_ids": EXPECTED_ORDER,
            "expected_trial_count": len(EXPECTED_ORDER),
        },
    )
    dump(
        root / "primary_results.json",
        {"channels": {"declared_goal_consequence": {"full": 0.2}}},
    )
    freeze_sha = dump(
        root / "confirmatory_freeze.json",
        {
            "schema": "symthaea.vart-world-creative-001.confirmatory-freeze.v1",
            "experiment_id": EXPERIMENT_ID,
            "frozen": True,
            "analysis_contract_sha256": analysis_sha,
            "metric_definition_set_sha256": metric_sha,
            "trial_inventory_sha256": inventory_sha,
            "forbidden_primary_aggregates": ["world_quality", "creative_score"],
            "primary_threshold": 0.05,
            "stopping_rule": "fixed_trial_inventory",
        },
    )
    return freeze_sha


def candidate_shas(root: Path, trial_id: str) -> list[str]:
    return [
        hashlib.sha256((tdir(root, trial_id) / f"candidate{i}.json").read_bytes()).hexdigest()
        for i in range(3)
    ]


def run_suite(base: Path, freeze_sha: str) -> None:
    result = qualified.verify_qualified(base, freeze_sha)
    assert result["verdict"] == "ACCEPT", result

    # N1 — mutate the hypothesis coherently at the manifest/file level, but not the
    # prospective applied receipt binding.
    b = clone(base)
    h = load(logical_path(b, FULL, "revision_hypothesis"))
    h["predicted"]["declared_goal_consequence"] = 0.99
    rewrite_logical(b, FULL, "revision_hypothesis", h, "revision_hypothesis_sha256")
    expect_reject(b, freeze_sha, "POST_HOC_HYPOTHESIS_MUTATION")

    # N2 — substitute a physically plausible candidate set for one paired policy.
    b = clone(base)
    cset = load(logical_path(b, HEURISTIC, "candidate_set"))
    cset["candidates"][2]["note"] = "substituted-set"
    csha = rewrite_logical(b, HEURISTIC, "candidate_set", cset, "candidate_set_sha256")
    rewrite_receipt(b, HEURISTIC, lambda r: r.update(candidate_set_sha256=csha))
    expect_reject(b, freeze_sha, "PAIRED_CANDIDATE_SET_MISMATCH")

    # N3 — keep the draw admissible but alter the frozen deterministic assignment.
    b = clone(base)
    draw = load(logical_path(b, RANDOM, "random_draw_receipt"))
    draw["selected_index"] = (draw["selected_index"] + 1) % 2
    rewrite_logical(b, RANDOM, "random_draw_receipt", draw, "random_draw_receipt_sha256")
    expect_reject(b, freeze_sha, "RANDOM_VALID_DRAW_MISMATCH")

    # N4 — point the selected-proposal evidence at the rejected candidate.
    b = clone(base)
    p2 = candidate_shas(b, FULL)[2]
    update_index(
        b,
        FULL,
        lambda idx: idx["files"].update(selected_proposal="candidate2.json"),
    )
    update_manifest(b, FULL, selected_proposal_sha256=p2)
    rewrite_receipt(b, FULL, lambda r: r.update(selected_proposal_sha256=p2))
    expect_reject(b, freeze_sha, "SELECTION_NOT_PHYSICALLY_ADMITTED")

    # N5 — splice a revisit from another world lineage.
    b = clone(base)
    revisit = load(logical_path(b, FULL, "revisit_observation"))
    revisit["world_version"] = "wOTHER"
    rewrite_logical(b, FULL, "revisit_observation", revisit, "revisit_observation_sha256")
    expect_reject(b, freeze_sha, "WORLD_VERSION_CHAIN_MISMATCH")

    # N6 — promote counterfactual evidence into committed-history position.
    b = clone(base)
    revisit = load(logical_path(b, FULL, "revisit_observation"))
    revisit["provenance_domain"] = "counterfactual"
    rewrite_logical(b, FULL, "revisit_observation", revisit, "revisit_observation_sha256")
    expect_reject(b, freeze_sha, "PROVENANCE_DOMAIN_SUBSTITUTION")

    # N7 — retain an outcome while removing its required revisit observation.
    b = clone(base)
    logical_path(b, FULL, "revisit_observation").unlink()
    expect_reject(b, freeze_sha, "OUTCOME_WITHOUT_REVISIT")

    # N8 — bind the selected trial to a receipt for another admitted proposal.
    b = clone(base)
    p1 = candidate_shas(b, FULL)[1]
    rewrite_receipt(b, FULL, lambda r: r.update(selected_proposal_sha256=p1))
    expect_reject(b, freeze_sha, "SELECTED_PROPOSAL_RECEIPT_MISMATCH")

    # N9 — inject a pilot-labeled trial into the confirmatory inventory.
    b = clone(base)
    update_manifest(b, FULL, campaign="pilot", included_in_confirmatory_analysis=True)
    expect_reject(b, freeze_sha, "PILOT_CONFIRMATORY_CONTAMINATION")

    # N10 — selectively omit an interior failed/undesired trial.
    b = clone(base)
    shutil.rmtree(tdir(b, HEURISTIC))
    expect_reject(b, freeze_sha, "PREREGISTERED_TRIAL_MISSING")

    # N11 — duplicate a successful trial under another path while preserving identity.
    b = clone(base)
    shutil.copytree(tdir(b, FULL), b / "trials" / "duplicate-full")
    expect_reject(b, freeze_sha, "DUPLICATE_TRIAL_IDENTITY")

    # N12 — insert a forbidden scalar aggregate into the primary result surface.
    b = clone(base)
    save(b / "primary_results.json", {"world_quality": 0.99})
    expect_reject(b, freeze_sha, "FORBIDDEN_AGGREGATE")

    # N13 — invert a metric direction without changing the externally anchored freeze.
    b = clone(base)
    analysis = load(b / "analysis_contract.json")
    analysis["metric_direction"] = "lower_is_better"
    save(b / "analysis_contract.json", analysis)
    expect_reject(b, freeze_sha, "ANALYSIS_CONTRACT_MISMATCH")

    # N14 — change a frozen threshold/stopping contract after external anchoring.
    b = clone(base)
    freeze = load(b / "confirmatory_freeze.json")
    freeze["primary_threshold"] = 0.0001
    save(b / "confirmatory_freeze.json", freeze)
    expect_reject(b, freeze_sha, "POST_FREEZE_CONTRACT_MUTATION")

    # N15 — stop at a favorable prefix of the prospectively frozen inventory.
    b = clone(base)
    shutil.rmtree(tdir(b, GENERALIZATION))
    expect_reject(b, freeze_sha, "UNAUTHORIZED_EARLY_STOP")

    # N16 — truncate rejected candidate evidence while keeping the frozen generated count.
    b = clone(base)
    cset = load(logical_path(b, GENERALIZATION, "candidate_set"))
    cset["candidates"] = cset["candidates"][:2]
    csha = rewrite_logical(
        b, GENERALIZATION, "candidate_set", cset, "candidate_set_sha256"
    )
    rewrite_receipt(b, GENERALIZATION, lambda r: r.update(candidate_set_sha256=csha))
    expect_reject(b, freeze_sha, "INCOMPLETE_EVIDENCE_CLOSURE")

    # N17 — record a cross-policy outcome leak before selection.
    b = clone(base)
    update_index(
        b,
        FULL,
        lambda idx: idx.update(cross_policy_outcome_observed_before_selection=True),
    )
    expect_reject(b, freeze_sha, "POLICY_ORDER_INFORMATION_LEAK")

    # N18 — disclose the unseen generalization fixture before its frozen reveal point.
    b = clone(base)
    update_index(
        b,
        GENERALIZATION,
        lambda idx: idx.update(
            fixture_disclosed_to_policy_at_ns=50,
            generalization_fixture_reveal_at_ns=100,
        ),
    )
    expect_reject(b, freeze_sha, "GENERALIZATION_FIXTURE_LEAK")

    # N19 — mark an integrity-broken trial as an ordinary complete scientific result.
    b = clone(base)
    update_manifest(b, FULL, integrity_violations=["BROKEN_PROVENANCE"])
    expect_reject(b, freeze_sha, "INTEGRITY_FAILURE_RECLASSIFIED")

    # N20 — erase a complete scientific result by reclassifying it as integrity-invalid.
    b = clone(base)
    update_manifest(
        b,
        FULL,
        trial_state="invalid_integrity",
        integrity_violations=["POST_HOC_INVALIDATION"],
    )
    expect_reject(b, freeze_sha, "INVALID_EXCLUSION_RECLASSIFICATION")


with tempfile.TemporaryDirectory(prefix="vart-qualified-") as td:
    base = Path(td) / "base"
    base.mkdir()
    freeze_sha = build(base)
    run_suite(base, freeze_sha)

print("PASS: VART qualified verifier canonical acceptance + N1-N20 deterministic rejection")
