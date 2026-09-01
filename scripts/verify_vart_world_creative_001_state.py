#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_context as context_verify

PAIR_POLICIES = {"full_symthaea", "random_valid", "heuristic"}
COMMITTED_PROVENANCE = {"digital_committed", "physical_grounded"}


def _dict(value: Any, code: str, detail: str) -> dict[str, Any]:
    core.require(isinstance(value, dict), code, detail)
    return value


def _state_snapshot(
    trial_dir: Path,
    idx: dict[str, Any],
    logical_name: str,
    expected_sha: str,
    expected_version: str,
    trial_id: str,
) -> dict[str, Any]:
    expected_sha = core.require_sha256(expected_sha, expected_sha)
    path = core.file_from_index(trial_dir, idx, logical_name)
    actual = core.sha256_file(path)
    core.require(
        actual == expected_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: {logical_name} {actual} != {expected_sha}",
    )
    state = _dict(
        core.read_json(path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: {logical_name}",
    )
    core.require(
        state.get("schema") == "symthaea.vart-world-creative-001.world-state-snapshot.v1"
        and state.get("experiment_id") == "VART-WORLD-CREATIVE-001",
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: {logical_name} schema",
    )
    core.require_sha256(state.get("state_digest"), f"{trial_id}.{logical_name}.state_digest")
    core.require(
        state.get("world_version") == expected_version,
        "WORLD_STATE_VERSION_MISMATCH",
        f"{trial_id}: {logical_name}",
    )
    core.require(
        state.get("provenance_domain") in COMMITTED_PROVENANCE,
        "WORLD_STATE_PROVENANCE_SUBSTITUTION",
        f"{trial_id}: {logical_name}",
    )
    return state


def _verify_trial_state(trial_dir: Path, manifest: dict[str, Any]) -> tuple[str, str | None]:
    trial_id = manifest["trial_id"]
    idx = _dict(
        core.read_json(trial_dir / "evidence_index.json"),
        "WORLD_STATE_DIGEST_MISMATCH",
        trial_id,
    )
    before_sha = core.require_sha256(
        manifest.get("world_state_before_sha256"), "world_state_before_sha256"
    )
    _state_snapshot(
        trial_dir,
        idx,
        "world_state_before",
        before_sha,
        manifest["world_version_before"],
        trial_id,
    )

    decision_path = core.file_from_index(trial_dir, idx, "decision_input")
    decision = _dict(
        core.read_json(decision_path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: decision input",
    )
    core.require(
        decision.get("world_state_before_sha256") == before_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: decision input pre-state",
    )

    context_path = core.file_from_index(trial_dir, idx, "execution_context")
    execution_context = _dict(
        core.read_json(context_path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: execution context",
    )
    core.require(
        execution_context.get("world_state_before_sha256") == before_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: execution context pre-state",
    )

    candidate_path = core.file_from_index(trial_dir, idx, "candidate_set")
    candidate_set = _dict(
        core.read_json(candidate_path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: candidate set",
    )
    core.require(
        candidate_set.get("world_state_before_sha256") == before_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: candidate set pre-state",
    )

    if manifest.get("trial_state") != "complete":
        return before_sha, None

    after_sha = core.require_sha256(
        manifest.get("world_state_after_sha256"), "world_state_after_sha256"
    )
    _state_snapshot(
        trial_dir,
        idx,
        "world_state_after",
        after_sha,
        manifest["world_version_after"],
        trial_id,
    )

    receipt_path = core.file_from_index(trial_dir, idx, "applied_receipt")
    receipt = _dict(
        core.read_json(receipt_path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: applied receipt",
    )
    core.require(
        receipt.get("world_state_before_sha256") == before_sha
        and receipt.get("world_state_after_sha256") == after_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: receipt state closure",
    )

    revisit_path = core.file_from_index(trial_dir, idx, "revisit_observation")
    revisit = _dict(
        core.read_json(revisit_path),
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: revisit",
    )
    core.require(
        revisit.get("world_state_sha256") == after_sha,
        "WORLD_STATE_DIGEST_MISMATCH",
        f"{trial_id}: revisit post-state",
    )
    return before_sha, after_sha


def verify_state_qualified(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    result = context_verify.verify_context_qualified(root, expected_freeze_sha256)
    inventory = _dict(
        core.read_json(root / "trial_inventory.json"),
        "WORLD_STATE_DIGEST_MISMATCH",
        "trial inventory",
    )
    trial_ids = inventory.get("trial_ids")
    core.require(isinstance(trial_ids, list), "PREREGISTRATION_INVALID", "trial_ids")

    by_block: dict[str, list[tuple[str, str, str]]] = {}
    chains: dict[tuple[str, str, int], list[tuple[int, str, str | None, str, str | None]]] = {}

    for trial_id in trial_ids:
        trial_dir = root / "trials" / trial_id
        manifest = _dict(
            core.read_json(trial_dir / "manifest.json"),
            "WORLD_STATE_DIGEST_MISMATCH",
            trial_id,
        )
        before_sha, after_sha = _verify_trial_state(trial_dir, manifest)
        if manifest["policy"] in PAIR_POLICIES:
            by_block.setdefault(manifest["paired_block_id"], []).append(
                (manifest["policy"], before_sha, manifest["world_version_before"])
            )
        key = (
            manifest["policy"],
            manifest["world_fixture_sha256"],
            manifest["seed"],
        )
        chains.setdefault(key, []).append(
            (
                manifest["revision_index"],
                before_sha,
                after_sha,
                manifest["world_version_before"],
                manifest.get("world_version_after"),
            )
        )

    for block_id, entries in by_block.items():
        if len(entries) > 1:
            core.require(
                len({before for _, before, _ in entries}) == 1
                and len({version for _, _, version in entries}) == 1,
                "PAIRED_WORLD_STATE_MISMATCH",
                block_id,
            )

    for key, chain in chains.items():
        ordered = sorted(chain, key=lambda x: x[0])
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] != previous[0] + 1:
                continue
            previous_after = previous[2]
            current_before = current[1]
            core.require(
                previous_after is not None and previous_after == current_before,
                "WORLD_STATE_CHAIN_MISMATCH",
                str(key),
            )
            core.require(
                previous[4] is not None and previous[4] == current[3],
                "WORLD_STATE_VERSION_MISMATCH",
                str(key),
            )

    out = dict(result)
    out.update(
        {
            "paired_world_state_equivalence": "PASS",
            "longitudinal_world_state_chain": "PASS",
            "committed_state_provenance": "PASS",
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="World-state equivalence verifier for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_state_qualified(args.root, args.expected_freeze_sha256)
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
            "paired world-state equivalence PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
