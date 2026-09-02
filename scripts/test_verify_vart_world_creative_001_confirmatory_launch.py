#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import verify_vart_world_creative_001_confirmatory_launch as gate


def h64(ch: str) -> str:
    return ch * 64


def dump(path: Path, obj: object) -> str:
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(longitudinal: bool, memory_clusters: int) -> dict:
    trials: list[dict] = []
    fixtures = ["ordinary", "PrettyTrap", "LocalOptimum", "HiddenDependency", "DelayedConsequence", "CounterfactualDecoy", "Path", "Plaza"]
    for i, fixture in enumerate(fixtures):
        cluster = f"{i+1:064x}"[-64:]
        for pidx, policy in enumerate(["full_symthaea", "random_valid", "heuristic"]):
            lineage = f"{1000 + i*10 + pidx:064x}"[-64:]
            trials.append({"trial_id": f"A-{i}-{policy}-r0", "subcampaign": "001A", "policy": policy,
                           "fixture": fixture, "seed": 100000 + i, "world_cluster_sha256": cluster,
                           "world_lineage_sha256": lineage, "revision_index": 0})
            if longitudinal and policy == "full_symthaea":
                for revision in (1, 2, 3):
                    trials.append({"trial_id": f"A-{i}-{policy}-r{revision}", "subcampaign": "001A", "policy": policy,
                                   "fixture": fixture, "seed": 100000 + i, "world_cluster_sha256": cluster,
                                   "world_lineage_sha256": lineage, "revision_index": revision})
    for i in range(memory_clusters):
        cluster = f"{9000+i:064x}"[-64:]
        for pidx, policy in enumerate(["full_symthaea", "no_reality_ledger_context"]):
            trials.append({"trial_id": f"B-{i}-{policy}-r0", "subcampaign": "001B", "policy": policy,
                           "fixture": "MemoryTrap", "seed": 200000 + i, "world_cluster_sha256": cluster,
                           "world_lineage_sha256": f"{9100+i*10+pidx:064x}"[-64:], "revision_index": 0})
    # The real v3 inventory stores a prospectively randomized permutation. Unit tests only need a frozen valid permutation.
    for order, row in enumerate(trials):
        row["run_order"] = order
    return {"schema": "symthaea.vart-world-creative-001.confirmatory-inventory.v3", "experiment_id": gate.EXPERIMENT_ID, "trials": trials}


def freeze(inv_sha: str) -> dict:
    return {
        "schema": "symthaea.vart-world-creative-001.confirmatory-freeze.v3",
        "experiment_id": gate.EXPERIMENT_ID,
        "frozen": True,
        "trial_inventory_sha256": inv_sha,
        "constraints": {"scalar_world_quality_forbidden": True, "zero_peeking_enforced": True},
        "claim": {"authorized": False},
    }


with tempfile.TemporaryDirectory(prefix="vart-launch-gate-") as td:
    root = Path(td)
    inv = root / "inventory.json"
    frz = root / "freeze.json"

    # L1: the reported 32-trial shape is structurally insufficient for H2.
    inv_sha = dump(inv, inventory(longitudinal=False, memory_clusters=4))
    freeze_sha = dump(frz, freeze(inv_sha))
    try:
        gate.verify(frz, freeze_sha, inv)
    except gate.Reject as exc:
        assert exc.code == "H2_LONGITUDINAL_DEPTH_INSUFFICIENT", (exc.code, exc.detail)
    else:
        raise AssertionError("32-trial shape unexpectedly passed H2 launch gate")

    # L2: adding H2 depth but keeping only four MemoryTrap clusters remains insufficient for H3.
    inv_sha = dump(inv, inventory(longitudinal=True, memory_clusters=4))
    freeze_sha = dump(frz, freeze(inv_sha))
    try:
        gate.verify(frz, freeze_sha, inv)
    except gate.Reject as exc:
        assert exc.code == "H3_MEMORYTRAP_CLUSTER_INSUFFICIENT", (exc.code, exc.detail)
    else:
        raise AssertionError("four-cluster H3 shape unexpectedly passed launch gate")

    # Canonical v3 shape: 24 H1 r0 + 24 FULL longitudinal continuation + 16 H3 = 64.
    inv_obj = inventory(longitudinal=True, memory_clusters=8)
    inv_sha = dump(inv, inv_obj)
    freeze_sha = dump(frz, freeze(inv_sha))
    result = gate.verify(frz, freeze_sha, inv)
    assert result["verdict"] == "CONFIRMATORY_LAUNCH_READY"
    assert result["trial_count"] == 64
    assert result["run_order_bound"] is True
    assert result["h1"]["qualified_cluster_count"] == 8
    assert result["h2"]["qualified_full_lineage_count"] == 8
    assert result["h3"]["qualified_cluster_count"] == 8

    # L3: raw freeze bytes must match the externally supplied anchor.
    try:
        gate.verify(frz, h64("f"), inv)
    except gate.Reject as exc:
        assert exc.code == "LAUNCH_FREEZE_ANCHOR_MISMATCH"
    else:
        raise AssertionError("wrong freeze anchor unexpectedly passed")

    # L4: schedule is prospective evidence; duplicate or missing run_order rejects.
    bad = inventory(longitudinal=True, memory_clusters=8)
    bad["trials"][1]["run_order"] = bad["trials"][0]["run_order"]
    bad_sha = dump(inv, bad)
    freeze_sha = dump(frz, freeze(bad_sha))
    try:
        gate.verify(frz, freeze_sha, inv)
    except gate.Reject as exc:
        assert exc.code == "LAUNCH_RUN_ORDER_INVALID", (exc.code, exc.detail)
    else:
        raise AssertionError("duplicate run_order unexpectedly passed")

print("PASS: H2/H3 sufficiency + 64-trial v3 + frozen seed/run-order gate")
