#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from verify_vart_world_creative_001 import (
    Reject,
    file_from_index,
    read_json,
    require,
    require_sha256,
    sha256_file,
    verify_bundle,
)

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
ABLATION_CHANNELS = {
    "no_embodied_experience": "embodied_experience",
    "no_persistent_memory": "persistent_memory",
    "no_depth": "depth_evidence",
    "no_counterfactual_evaluation": "counterfactual_evaluation",
    "no_independent_proposal_replay": "independent_proposal_replay",
    "no_reality_ledger_context": "reality_ledger_context",
    "random_valid_judgment": "experience_conditioned_judgment",
}


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def verify_ablation_receipt(trial_dir: Path, manifest: dict[str, Any]) -> None:
    policy = manifest.get("policy")
    expected_channel = ABLATION_CHANNELS.get(policy)
    if expected_channel is None:
        return

    receipt_sha = require_sha256(
        manifest.get("ablation_receipt_sha256"), "ablation_receipt_sha256"
    )
    idx = read_json(trial_dir / "evidence_index.json")
    receipt_path = file_from_index(trial_dir, idx, "ablation_receipt")
    require(
        sha256_file(receipt_path) == receipt_sha,
        "EVIDENCE_DIGEST_MISMATCH",
        f"{manifest['trial_id']}: ablation receipt",
    )
    receipt = read_json(receipt_path)
    require(
        isinstance(receipt, dict),
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: receipt object",
    )
    require(
        receipt.get("schema") == "symthaea.vart-world-creative-001.ablation-receipt.v1",
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: receipt schema",
    )
    require(
        receipt.get("experiment_id") == EXPERIMENT_ID,
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: receipt experiment",
    )
    require(
        receipt.get("trial_id") == manifest["trial_id"]
        and receipt.get("policy") == policy,
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: receipt identity",
    )
    removed = receipt.get("removed_channels")
    require(
        isinstance(removed, list) and expected_channel in removed,
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: expected removed channel {expected_channel}",
    )
    require(
        receipt.get("preregistered_ablation") is True,
        "ABLATION_SEMANTICS_MISMATCH",
        f"{manifest['trial_id']}: preregistered marker",
    )

    if policy == "no_embodied_experience":
        exp_path = file_from_index(trial_dir, idx, "experience_episode")
        require(
            sha256_file(exp_path) == manifest["experience_episode_sha256"],
            "EVIDENCE_DIGEST_MISMATCH",
            f"{manifest['trial_id']}: experience sentinel",
        )
        sentinel = read_json(exp_path)
        require(
            isinstance(sentinel, dict)
            and sentinel.get("schema")
            == "symthaea.vart-world-creative-001.ablation-sentinel.v1"
            and sentinel.get("experiment_id") == EXPERIMENT_ID
            and sentinel.get("trial_id") == manifest["trial_id"]
            and sentinel.get("policy") == policy
            and sentinel.get("channel") == "ExperienceEpisode"
            and sentinel.get("available") is False,
            "ABLATION_SEMANTICS_MISMATCH",
            f"{manifest['trial_id']}: ExperienceEpisode was not explicitly ablated",
        )
        assertions = receipt.get("assertions", {})
        require(
            isinstance(assertions, dict)
            and assertions.get("experience_episode_available") is False,
            "ABLATION_SEMANTICS_MISMATCH",
            f"{manifest['trial_id']}: experience assertion",
        )

    if policy == "no_counterfactual_evaluation":
        candidate_path = file_from_index(trial_dir, idx, "candidate_set")
        candidate_set = read_json(candidate_path)
        assertions = receipt.get("assertions", {})
        require(
            isinstance(assertions, dict)
            and assertions.get("counterfactual_evaluation_performed") is False,
            "ABLATION_SEMANTICS_MISMATCH",
            f"{manifest['trial_id']}: counterfactual assertion",
        )
        forbidden_keys = {
            "counterfactual_observation_sha256",
            "counterfactual_rgb_sha256",
            "counterfactual_render_sha256",
            "counterfactual_score",
        }
        present = forbidden_keys.intersection(set(walk_keys(candidate_set)))
        require(
            not present,
            "ABLATION_SEMANTICS_MISMATCH",
            f"{manifest['trial_id']}: counterfactual evidence leaked into candidate set: {sorted(present)}",
        )
        files = idx.get("files", {}) if isinstance(idx, dict) else {}
        if isinstance(files, dict):
            leaked = sorted(
                name for name in files if str(name).startswith("counterfactual_")
            )
            require(
                not leaked,
                "ABLATION_SEMANTICS_MISMATCH",
                f"{manifest['trial_id']}: counterfactual evidence files present: {leaked}",
            )


def verify_pilot(root: Path) -> dict[str, Any]:
    result = verify_bundle(root)
    require(
        result.get("verdict") == "ACCEPT",
        "PILOT_CORE_VERIFIER_REJECTED",
        str(result),
    )

    inventory = read_json(root / "trial_inventory.json")
    trial_ids = inventory.get("trial_ids")
    require(
        isinstance(trial_ids, list) and len(trial_ids) == 8,
        "PREREGISTERED_TRIAL_MISSING",
        "pilot inventory must contain exactly eight trials",
    )

    ablation_trials = 0
    for trial_id in trial_ids:
        trial_dir = root / "trials" / trial_id
        manifest = read_json(trial_dir / "manifest.json")
        require(
            manifest.get("campaign") == "pilot"
            and manifest.get("included_in_confirmatory_analysis") is False,
            "PILOT_CONFIRMATORY_CONTAMINATION",
            trial_id,
        )
        if manifest.get("policy") in ABLATION_CHANNELS:
            ablation_trials += 1
            verify_ablation_receipt(trial_dir, manifest)

    result = dict(result)
    result.update(
        {
            "pilot_ablation_semantics": "PASS",
            "ablation_trial_count": ablation_trials,
            "scientific_efficacy_claims_authorized": False,
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot-qualified wrapper for the independent VART evidence verifier"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_pilot(args.root)
    except Reject as exc:
        payload = {"verdict": "REJECT", "reason": exc.code, "detail": exc.detail}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}")
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            f"ACCEPT: {result['trial_count']} pilot trials; "
            f"ablation semantics {result['pilot_ablation_semantics']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
