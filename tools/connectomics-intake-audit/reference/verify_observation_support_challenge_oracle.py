#!/usr/bin/env python3
"""Independent support-expansion oracle using Python sets and combinations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def field(hasher, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def scenario_digest(scenarios: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(b"connectome-query-disambiguation-scenario-set-v1")
    rows = sorted(scenarios, key=lambda row: row["scenario_id"])
    h.update(len(rows).to_bytes(8, "big"))
    for row in rows:
        h.update(b"WORLD")
        field(h, row["scenario_id"])
        h.update(row["graph_sha256"].encode("ascii"))
        h.update(bytes([int(row["decision_met"])]))
    return h.hexdigest()


def action_digest(actions: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(b"connectome-robust-query-disambiguation-action-set-v1")
    rows = sorted(actions, key=lambda row: row["action_id"])
    h.update(len(rows).to_bytes(8, "big"))
    for action in rows:
        h.update(b"ACTION")
        field(h, action["action_id"])
        field(h, action["action_kind"])
        h.update(action["effort_units"].to_bytes(8, "big"))
        h.update(action["source_evidence_sha256"].encode("ascii"))
        h.update(action["observation_model_sha256"].encode("ascii"))
        groups = sorted(action["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            field(h, group)
        outcomes = sorted(action["outcomes"], key=lambda row: row["scenario_id"])
        h.update(len(outcomes).to_bytes(8, "big"))
        for outcome in outcomes:
            values = sorted(outcome["possible_outcomes"])
            h.update(b"WORLD-OUTCOMES")
            field(h, outcome["scenario_id"])
            h.update(len(values).to_bytes(8, "big"))
            for value in values:
                h.update(b"POSSIBLE")
                field(h, value)
    return h.hexdigest()


def challenge_digest(records: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(b"connectome-observation-support-challenge-set-v1")
    rows = sorted(records, key=lambda row: row["challenge_id"])
    h.update(len(rows).to_bytes(8, "big"))
    for row in rows:
        outcomes = sorted(row["additional_possible_outcomes"])
        groups = sorted(row["dependence_groups"])
        h.update(b"CHALLENGE")
        field(h, row["challenge_id"])
        field(h, row["action_id"])
        field(h, row["scenario_id"])
        h.update(row["source_evidence_sha256"].encode("ascii"))
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            field(h, group)
        h.update(len(outcomes).to_bytes(8, "big"))
        for value in outcomes:
            h.update(b"ADDITIONAL")
            field(h, value)
    return h.hexdigest()


def conflict_pairs(scenarios: list[dict]) -> list[dict]:
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    return [
        {"world_a": left["scenario_id"], "world_b": right["scenario_id"]}
        for i, left in enumerate(worlds)
        for right in worlds[i + 1 :]
        if left["decision_met"] != right["decision_met"]
    ]


def support_map(action: dict) -> dict[str, set[str]]:
    return {
        row["scenario_id"]: set(row["possible_outcomes"])
        for row in action["outcomes"]
    }


def separates(action: dict, pair: dict) -> bool:
    supports = support_map(action)
    return supports[pair["world_a"]].isdisjoint(supports[pair["world_b"]])


def coverage(actions: list[dict], pairs: list[dict]) -> list[dict]:
    return [
        {
            "action_id": action["action_id"],
            "effort_units": action["effort_units"],
            "robustly_separated_pairs": [
                pair for pair in pairs if separates(action, pair)
            ],
        }
        for action in sorted(actions, key=lambda row: row["action_id"])
    ]


def summarize(actions: list[dict], pairs: list[dict], budget: int) -> dict:
    undiscriminable = [
        pair
        for pair in pairs
        if not any(separates(action, pair) for action in actions)
    ]
    separable = len(pairs) - len(undiscriminable)
    if not pairs:
        return {
            "robustly_separable_pair_count": 0,
            "robustly_undiscriminable_pairs": [],
            "minimum_effort_units": 0,
            "minimum_plan_count": 1,
            "minimum_plans_returned": 1,
            "minimum_plans_truncated": False,
            "minimum_plans": [{"action_ids": [], "effort_units": 0}],
            "subset_count_evaluated": 1,
            "within_declared_budget": True,
        }
    if undiscriminable:
        return {
            "robustly_separable_pair_count": separable,
            "robustly_undiscriminable_pairs": undiscriminable,
            "minimum_effort_units": None,
            "minimum_plan_count": 0,
            "minimum_plans_returned": 0,
            "minimum_plans_truncated": False,
            "minimum_plans": [],
            "subset_count_evaluated": 0,
            "within_declared_budget": None,
        }

    best_cost = None
    best_plans = []
    minimum_count = 0
    subset_count = (1 << len(actions)) - 1
    for size in range(1, len(actions) + 1):
        for chosen in itertools.combinations(actions, size):
            covered = set()
            cost = 0
            for action in chosen:
                cost += action["effort_units"]
                covered.update(
                    (pair["world_a"], pair["world_b"])
                    for pair in pairs
                    if separates(action, pair)
                )
            required = {(p["world_a"], p["world_b"]) for p in pairs}
            if covered != required:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                minimum_count = 0
                best_plans = []
            if cost == best_cost:
                minimum_count += 1
                if len(best_plans) < 128:
                    best_plans.append(
                        {
                            "action_ids": sorted(a["action_id"] for a in chosen),
                            "effort_units": cost,
                        }
                    )
    if best_cost is None:
        raise AssertionError("oracle inconsistency: per-pair cover check passed but no plan")
    best_plans.sort(key=lambda p: p["action_ids"])
    return {
        "robustly_separable_pair_count": separable,
        "robustly_undiscriminable_pairs": [],
        "minimum_effort_units": best_cost,
        "minimum_plan_count": minimum_count,
        "minimum_plans_returned": len(best_plans),
        "minimum_plans_truncated": minimum_count > 128,
        "minimum_plans": best_plans,
        "subset_count_evaluated": subset_count,
        "within_declared_budget": best_cost <= budget,
    }


def run(base: dict, challenge: dict, base_bytes: bytes, challenge_bytes: bytes) -> dict:
    assert scenario_digest(base["scenarios"]) == base["scenario_set_sha256"]
    base_action_digest = action_digest(base["actions"])
    assert challenge["analysis_id"] == base["analysis_id"]
    assert challenge["dataset_revision_commit_sha"] == base["dataset_revision_commit_sha"]
    assert challenge["query_id"] == base["query_id"]
    assert challenge["query_spec_sha256"] == base["query_spec_sha256"]
    assert challenge["scenario_set_sha256"] == base["scenario_set_sha256"]
    assert challenge["baseline_action_set_sha256"] == base_action_digest

    world_ids = {world["scenario_id"] for world in base["scenarios"]}
    action_ids = {action["action_id"] for action in base["actions"]}
    record_ids = set()
    challenged = json.loads(json.dumps(base["actions"]))
    impacts = []
    new_count = 0
    for record in challenge["records"]:
        assert record["challenge_id"] not in record_ids
        record_ids.add(record["challenge_id"])
        assert record["action_id"] in action_ids and record["scenario_id"] in world_ids
        assert record["additional_possible_outcomes"]
        action = next(a for a in challenged if a["action_id"] == record["action_id"])
        world = next(o for o in action["outcomes"] if o["scenario_id"] == record["scenario_id"])
        added = []
        for value in record["additional_possible_outcomes"]:
            if value not in world["possible_outcomes"]:
                world["possible_outcomes"].append(value)
                added.append(value)
                new_count += 1
        impacts.append({
            "challenge_id": record["challenge_id"],
            "action_id": record["action_id"],
            "scenario_id": record["scenario_id"],
            "newly_admitted_outcome_count": len(added),
            "newly_admitted_outcomes": sorted(added),
        })
    impacts.sort(key=lambda x: x["challenge_id"])

    pairs = conflict_pairs(base["scenarios"])
    baseline = summarize(base["actions"], pairs, base["max_total_effort_units"])
    challenged_summary = summarize(challenged, pairs, base["max_total_effort_units"])
    before_coverage = coverage(base["actions"], pairs)
    after_coverage = coverage(challenged, pairs)
    lost_action = []
    for before in before_coverage:
        after = next(row for row in after_coverage if row["action_id"] == before["action_id"])
        for pair in before["robustly_separated_pairs"]:
            if pair not in after["robustly_separated_pairs"]:
                lost_action.append({"action_id": before["action_id"], "pair": pair})
    lost_action.sort(key=lambda row: (row["action_id"], row["pair"]["world_a"], row["pair"]["world_b"]))

    lost_global = list(baseline["robustly_undiscriminable_pairs"])
    for pair in pairs:
        was = any(separates(a, pair) for a in base["actions"])
        now = any(separates(a, pair) for a in challenged)
        if was and not now:
            lost_global.append(pair)
    lost_global = sorted({(p["world_a"], p["world_b"]) for p in lost_global})
    lost_global = [{"world_a": a, "world_b": b} for a, b in lost_global]

    violations = []
    before_by_id = {r["action_id"]: r for r in before_coverage}
    after_by_id = {r["action_id"]: r for r in after_coverage}
    for aid, before in before_by_id.items():
        if len(after_by_id[aid]["robustly_separated_pairs"]) > len(before["robustly_separated_pairs"]):
            violations.append(f"action {aid} gained robust coverage after support expansion")
    if challenged_summary["robustly_separable_pair_count"] > baseline["robustly_separable_pair_count"]:
        violations.append("global robust pair coverage increased after support expansion")
    bcost, ccost = baseline["minimum_effort_units"], challenged_summary["minimum_effort_units"]
    if bcost is not None and ccost is not None and ccost < bcost:
        violations.append(f"minimum effort decreased from {bcost} to {ccost} after support expansion")
    if any(p not in challenged_summary["robustly_undiscriminable_pairs"] for p in baseline["robustly_undiscriminable_pairs"]):
        violations.append("a baseline-undiscriminable pair became discriminable after support expansion")

    if not challenge["records"]:
        outcome = "NO_CHALLENGE_ADDITIONS"
    elif bcost is None:
        outcome = "BASELINE_ALREADY_HAS_NO_ROBUST_PLAN"
    elif ccost is None:
        outcome = "ROBUST_PLAN_DESTROYED_BY_SUPPORT_EXPANSION"
    elif ccost > bcost:
        outcome = "ROBUST_PLAN_COST_INCREASED"
    else:
        outcome = "ROBUST_PLAN_PRESERVED"

    return {
        "schema_version": 1,
        "tool": "connectome-observation-support-challenge-audit",
        "analysis_id": base["analysis_id"],
        "dataset_revision_commit_sha": base["dataset_revision_commit_sha"],
        "query_id": base["query_id"],
        "query_spec_sha256": base["query_spec_sha256"],
        "scenario_set_sha256": base["scenario_set_sha256"],
        "baseline_action_set_sha256": base_action_digest,
        "challenge_input_sha256": sha256(challenge_bytes),
        "challenge_set_sha256": challenge_digest(challenge["records"]),
        "challenger_protocol_sha256": challenge["challenger_protocol_sha256"],
        "effort_unit": base["effort_unit"],
        "max_total_effort_units": base["max_total_effort_units"],
        "challenge_record_count": len(challenge["records"]),
        "newly_admitted_outcome_count": new_count,
        "baseline": baseline,
        "challenged": challenged_summary,
        "baseline_action_coverage": before_coverage,
        "challenged_action_coverage": after_coverage,
        "lost_action_separations": lost_action,
        "lost_global_separations": lost_global,
        "record_impacts": impacts,
        "outcome": outcome,
        "support_expansion_monotone": not violations,
        "monotonicity_violations": violations,
        "search_complete": True,
        "claim_boundary": "Results are conditional on the baseline worlds, observation actions, and supplied challenge records. Challenge records only expand possible-outcome supports; they do not prove support completeness, authenticity, reviewer independence, or biological truth. No probabilities or likelihoods are inferred.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--challenge", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        base_bytes = args.input.read_bytes()
        challenge_bytes = args.challenge.read_bytes()
        base = json.loads(base_bytes)
        challenge = json.loads(challenge_bytes)
        expected = run(base, challenge, base_bytes, challenge_bytes)
        actual = json.loads(args.report.read_text(encoding="utf-8"))
        assert actual == expected, "report differs from independent combinations/set oracle"
        print(f"verified {expected['outcome']}: {expected['newly_admitted_outcome_count']} new outcomes; baseline min={expected['baseline']['minimum_effort_units']}; challenged min={expected['challenged']['minimum_effort_units']}; monotone={expected['support_expansion_monotone']}")
        return 0
    except Exception as exc:  # produce a crisp CI failure without hiding the assertion
        print(f"support challenge oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
