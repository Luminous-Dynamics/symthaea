#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_state as state_verify

PAIR_POLICIES = {"full_symthaea", "random_valid", "heuristic"}


def _dict(value: Any, code: str, detail: str) -> dict[str, Any]:
    core.require(isinstance(value, dict), code, detail)
    return value


def verify_identity_qualified(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    result = state_verify.verify_state_qualified(root, expected_freeze_sha256)
    inventory = _dict(
        core.read_json(root / "trial_inventory.json"),
        "WORLD_IDENTITY_INVENTORY_MISMATCH",
        "trial inventory",
    )
    trial_ids = inventory.get("trial_ids")
    clusters = _dict(
        inventory.get("world_clusters"),
        "WORLD_IDENTITY_INVENTORY_MISMATCH",
        "world_clusters",
    )
    lineages = _dict(
        inventory.get("world_lineages"),
        "WORLD_IDENTITY_INVENTORY_MISMATCH",
        "world_lineages",
    )
    core.require(
        isinstance(trial_ids, list)
        and set(clusters) == set(trial_ids)
        and set(lineages) == set(trial_ids),
        "WORLD_IDENTITY_INVENTORY_MISMATCH",
        "identity maps must exactly cover trial_ids",
    )

    by_block: dict[str, list[dict[str, Any]]] = {}
    by_lineage: dict[str, list[dict[str, Any]]] = {}
    cluster_ids: set[str] = set()
    lineage_ids: set[str] = set()

    for trial_id in trial_ids:
        manifest = _dict(
            core.read_json(root / "trials" / trial_id / "manifest.json"),
            "WORLD_IDENTITY_INVENTORY_MISMATCH",
            trial_id,
        )
        frozen_cluster = core.require_sha256(clusters.get(trial_id), f"world_clusters.{trial_id}")
        frozen_lineage = core.require_sha256(lineages.get(trial_id), f"world_lineages.{trial_id}")
        manifest_cluster = core.require_sha256(
            manifest.get("world_cluster_sha256"), "world_cluster_sha256"
        )
        manifest_lineage = core.require_sha256(
            manifest.get("world_lineage_sha256"), "world_lineage_sha256"
        )
        core.require(
            manifest_cluster == frozen_cluster and manifest_lineage == frozen_lineage,
            "WORLD_IDENTITY_INVENTORY_MISMATCH",
            trial_id,
        )
        cluster_ids.add(manifest_cluster)
        lineage_ids.add(manifest_lineage)
        manifest = dict(manifest)
        manifest["world_cluster_sha256"] = manifest_cluster
        manifest["world_lineage_sha256"] = manifest_lineage
        by_block.setdefault(manifest["paired_block_id"], []).append(manifest)
        by_lineage.setdefault(manifest_lineage, []).append(manifest)

    for block_id, trials in by_block.items():
        paired = [t for t in trials if t["policy"] in PAIR_POLICIES]
        if len(paired) > 1:
            core.require(
                len({t["world_cluster_sha256"] for t in paired}) == 1,
                "WORLD_CLUSTER_PAIRING_MISMATCH",
                block_id,
            )

    for lineage_id, trials in by_lineage.items():
        core.require(
            len({t["policy"] for t in trials}) == 1,
            "WORLD_LINEAGE_POLICY_MISMATCH",
            lineage_id,
        )
        core.require(
            len({t["world_cluster_sha256"] for t in trials}) == 1,
            "WORLD_LINEAGE_CLUSTER_MISMATCH",
            lineage_id,
        )
        core.require(
            len({t["world_fixture_sha256"] for t in trials}) == 1,
            "WORLD_LINEAGE_FIXTURE_MISMATCH",
            lineage_id,
        )
        core.require(
            len({t["seed"] for t in trials}) == 1,
            "WORLD_LINEAGE_SEED_MISMATCH",
            lineage_id,
        )
        revisions = [t["revision_index"] for t in trials]
        core.require(
            len(revisions) == len(set(revisions)),
            "WORLD_LINEAGE_DUPLICATE_REVISION",
            lineage_id,
        )
        complete = sorted(
            (t for t in trials if t["trial_state"] == "complete"),
            key=lambda t: t["revision_index"],
        )
        for previous, current in zip(complete, complete[1:]):
            if current["revision_index"] != previous["revision_index"] + 1:
                continue
            core.require(
                previous.get("world_state_after_sha256")
                == current.get("world_state_before_sha256"),
                "WORLD_LINEAGE_STATE_CHAIN_MISMATCH",
                lineage_id,
            )
            core.require(
                previous.get("world_version_after") == current.get("world_version_before"),
                "WORLD_LINEAGE_VERSION_CHAIN_MISMATCH",
                lineage_id,
            )

    out = dict(result)
    out.update(
        {
            "explicit_world_identity": "PASS",
            "world_cluster_count": len(cluster_ids),
            "world_lineage_count": len(lineage_ids),
            "paired_cluster_integrity": "PASS",
            "persistent_lineage_integrity": "PASS",
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit persistent-world identity verifier for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_identity_qualified(args.root, args.expected_freeze_sha256)
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
            f"ACCEPT: {result['world_cluster_count']} clusters / "
            f"{result['world_lineage_count']} lineages; explicit identity PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
