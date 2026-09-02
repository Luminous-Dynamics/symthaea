#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
H1_POLICIES = {"full_symthaea", "random_valid", "heuristic"}
H3_POLICIES = {"full_symthaea", "no_reality_ledger_context"}


class Reject(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise Reject(code, detail)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Reject("LAUNCH_EVIDENCE_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise Reject("LAUNCH_JSON_INVALID", f"{path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hex64(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            "LAUNCH_INVALID", label)
    return value


def normalize_policy(value: Any) -> str:
    require(isinstance(value, str) and value, "LAUNCH_INVENTORY_INVALID", "policy")
    aliases = {
        "FullSymthaea": "full_symthaea",
        "FULL": "full_symthaea",
        "RandomValid": "random_valid",
        "RANDOM_VALID": "random_valid",
        "Heuristic": "heuristic",
        "HEURISTIC": "heuristic",
        "NoLedger": "no_reality_ledger_context",
        "NO_LEDGER": "no_reality_ledger_context",
    }
    return aliases.get(value, value)


def rows_from_inventory(obj: Any) -> list[dict[str, Any]]:
    require(isinstance(obj, dict), "LAUNCH_INVENTORY_INVALID", "inventory root")
    require(obj.get("experiment_id") == EXPERIMENT_ID,
            "LAUNCH_INVENTORY_INVALID", "experiment_id")
    rows = obj.get("trials")
    require(isinstance(rows, list) and rows, "LAUNCH_INVENTORY_INVALID", "trials")
    seen: set[str] = set()
    seen_orders: set[int] = set()
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        require(isinstance(raw, dict), "LAUNCH_INVENTORY_INVALID", f"trials[{i}]")
        for key in ("trial_id", "subcampaign", "policy", "fixture", "seed", "world_cluster_sha256", "world_lineage_sha256", "revision_index", "run_order"):
            require(key in raw, "LAUNCH_INVENTORY_INVALID", f"trials[{i}].{key}")
        trial_id = raw["trial_id"]
        require(isinstance(trial_id, str) and trial_id and trial_id not in seen,
                "LAUNCH_TRIAL_ID_INVALID", str(trial_id))
        seen.add(trial_id)
        cluster = hex64(raw["world_cluster_sha256"], f"{trial_id}.world_cluster_sha256")
        lineage = hex64(raw["world_lineage_sha256"], f"{trial_id}.world_lineage_sha256")
        revision = raw["revision_index"]
        require(isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0,
                "LAUNCH_INVENTORY_INVALID", f"{trial_id}.revision_index")
        seed = raw["seed"]
        require(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= (1 << 64) - 1,
                "LAUNCH_INVENTORY_INVALID", f"{trial_id}.seed")
        run_order = raw["run_order"]
        require(isinstance(run_order, int) and not isinstance(run_order, bool) and run_order >= 0,
                "LAUNCH_RUN_ORDER_INVALID", f"{trial_id}.run_order")
        require(run_order not in seen_orders, "LAUNCH_RUN_ORDER_INVALID", f"duplicate run_order={run_order}")
        seen_orders.add(run_order)
        subcampaign = raw["subcampaign"]
        require(subcampaign in {"001A", "001B"}, "LAUNCH_INVENTORY_INVALID", f"{trial_id}.subcampaign")
        fixture = raw["fixture"]
        require(isinstance(fixture, str) and fixture, "LAUNCH_INVENTORY_INVALID", f"{trial_id}.fixture")
        out.append({**raw, "policy": normalize_policy(raw["policy"]), "world_cluster_sha256": cluster,
                    "world_lineage_sha256": lineage, "revision_index": revision, "seed": seed, "run_order": run_order})
    expected_orders = set(range(len(out)))
    require(seen_orders == expected_orders, "LAUNCH_RUN_ORDER_INVALID",
            f"run_order must be contiguous 0..{len(out)-1}; missing={sorted(expected_orders-seen_orders)} extra={sorted(seen_orders-expected_orders)}")
    return out


def find_bool(obj: Any, key: str) -> bool | None:
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], bool):
            return obj[key]
        for value in obj.values():
            found = find_bool(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_bool(value, key)
            if found is not None:
                return found
    return None


def frozen_inventory_sha(freeze: dict[str, Any]) -> str:
    candidates = [freeze.get("trial_inventory_sha256")]
    campaign = freeze.get("campaign")
    if isinstance(campaign, dict):
        candidates.extend([campaign.get("trial_inventory_sha256"), campaign.get("inventory_sha256")])
    for value in candidates:
        if isinstance(value, str) and HEX64.fullmatch(value):
            return value
    raise Reject("LAUNCH_FREEZE_INVENTORY_BINDING_MISSING", "trial_inventory_sha256")


def verify_h1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = [r for r in rows if r["subcampaign"] == "001A" and r["revision_index"] == 0]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in core:
        clusters[row["world_cluster_sha256"]].append(row)
    good: list[str] = []
    for cluster, members in clusters.items():
        policies = {m["policy"] for m in members}
        if policies == H1_POLICIES and len(members) == 3:
            good.append(cluster)
    require(len(good) >= 8, "H1_PAIRED_CLUSTER_INSUFFICIENT", f"qualified={len(good)} required=8")
    return {"qualified_cluster_count": len(good), "clusters": sorted(good)}


def verify_h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [r for r in rows if r["subcampaign"] == "001A" and r["policy"] == "full_symthaea"]
    lineages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full:
        lineages[row["world_lineage_sha256"]].append(row)
    qualified: list[dict[str, Any]] = []
    for lineage, members in lineages.items():
        revisions = sorted({m["revision_index"] for m in members})
        clusters = {m["world_cluster_sha256"] for m in members}
        if len(clusters) != 1:
            raise Reject("H2_LINEAGE_CLUSTER_SPLICE", lineage)
        if len(revisions) >= 4 and revisions[:4] == [0, 1, 2, 3]:
            qualified.append({"lineage": lineage, "cluster": next(iter(clusters)), "revisions": revisions})
    require(len(qualified) >= 8, "H2_LONGITUDINAL_DEPTH_INSUFFICIENT",
            f"qualified_full_lineages={len(qualified)} required=8; need revision indices 0,1,2,3")
    return {"qualified_full_lineage_count": len(qualified), "minimum_revision_points": 4,
            "lineages": qualified}


def verify_h3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mem = [r for r in rows if r["subcampaign"] == "001B" and r["revision_index"] == 0]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mem:
        clusters[row["world_cluster_sha256"]].append(row)
    good: list[str] = []
    for cluster, members in clusters.items():
        policies = {m["policy"] for m in members}
        fixtures = {m["fixture"].lower().replace("_", "") for m in members}
        is_memory = any("memorytrap" in f for f in fixtures)
        if policies == H3_POLICIES and len(members) == 2 and is_memory:
            good.append(cluster)
    require(len(good) >= 8, "H3_MEMORYTRAP_CLUSTER_INSUFFICIENT",
            f"qualified={len(good)} required=8; exact paired sign-flip inference at alpha=0.05 needs at least 5 pairs, and v3 freezes 8")
    return {"qualified_cluster_count": len(good), "clusters": sorted(good)}


def verify(freeze_path: Path, expected_freeze_sha256: str, inventory_path: Path) -> dict[str, Any]:
    expected = hex64(expected_freeze_sha256, "expected_freeze_sha256")
    actual = sha256_file(freeze_path)
    require(actual == expected, "LAUNCH_FREEZE_ANCHOR_MISMATCH", f"{actual} != {expected}")
    freeze = read_json(freeze_path)
    require(isinstance(freeze, dict) and freeze.get("experiment_id") == EXPERIMENT_ID,
            "LAUNCH_FREEZE_INVALID", "identity")
    require(freeze.get("frozen") is True or freeze.get("freeze_state") == "frozen",
            "LAUNCH_FREEZE_NOT_FROZEN", "freeze state")
    require(find_bool(freeze, "scalar_world_quality_forbidden") is True,
            "LAUNCH_SCALAR_BOUNDARY_MISSING", "scalar_world_quality_forbidden=true")
    require(find_bool(freeze, "zero_peeking_enforced") is True,
            "LAUNCH_ZERO_PEEKING_MISSING", "zero_peeking_enforced=true")

    inventory_sha = sha256_file(inventory_path)
    bound_sha = frozen_inventory_sha(freeze)
    require(inventory_sha == bound_sha, "LAUNCH_INVENTORY_DIGEST_MISMATCH", f"{inventory_sha} != {bound_sha}")
    rows = rows_from_inventory(read_json(inventory_path))

    h1 = verify_h1(rows)
    h2 = verify_h2(rows)
    h3 = verify_h3(rows)
    a_clusters = set(h1["clusters"])
    b_clusters = set(h3["clusters"])
    require(a_clusters.isdisjoint(b_clusters), "LAUNCH_CLUSTER_REUSE_ACROSS_SUBCAMPAIGNS",
            f"overlap={sorted(a_clusters & b_clusters)}")

    schedule = sorted((r["run_order"], r["trial_id"]) for r in rows)
    return {
        "verdict": "CONFIRMATORY_LAUNCH_READY",
        "experiment_id": EXPERIMENT_ID,
        "freeze_sha256": actual,
        "trial_inventory_sha256": inventory_sha,
        "trial_count": len(rows),
        "run_order_bound": True,
        "schedule_sha256": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "confirmatory_execution_authorized": True,
        "claim_authorized": False,
        "bounded_statement": "The frozen inventory is structurally sufficient to execute the preregistered hypotheses; no scientific result is implied."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VART-WORLD-CREATIVE-001 prospective confirmatory launch sufficiency gate")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--trial-inventory", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.freeze, args.expected_freeze_sha256, args.trial_inventory)
    except Reject as exc:
        payload = {"verdict": "CONFIRMATORY_LAUNCH_REJECT", "reason_class": exc.code, "detail": exc.detail,
                   "confirmatory_execution_authorized": False, "claim_authorized": False}
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")) if args.json else f"REJECT {exc.code}: {exc.detail}")
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")) if args.json else "CONFIRMATORY_LAUNCH_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
