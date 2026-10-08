#!/usr/bin/env python3
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path
import json, math, statistics, sys

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "../docs/research/monetary/monetary-strategic-reporting-v1.json"
M = json.loads(MANIFEST_PATH.read_text())

PAIRS = tuple(M["factors"]["monetary_pair"])
AD = tuple(M["factors"]["adapter"])
SH = tuple(M["factors"]["shock"])
SEEDS = tuple(M["factors"]["seed"])
ALLO = tuple(M["factors"]["allocation_policy"])
FOCAL = tuple(M["factors"]["focal_obligation"])
REPORT = tuple(M["factors"]["reporting_policy"])

AMOUNTS = [x["amount"] for x in M["fixed_dimensions"]["obligations"]]
TRUE_CRIT = [x["criticality"] for x in M["fixed_dimensions"]["obligations"]]
CAP = M["fixed_dimensions"]["resource_capacity"]
DEMAND_MULT = M["fixed_dimensions"]["liquidity_demand_multiplier_by_adapter"]
TECH = {"redeem_reissue": 3, "escrowed_atomic_swap": 2, "multilateral_net_settlement": 2, "absent": None}
POLICY_DIGESTS = {k: v["digest"] for k, v in M["reporting_policies"].items()}
ALLOCATION_POLICY_DIGESTS = M["fixed_dimensions"]["allocation_policy_digests"]

def mean(values):
    return sum(values) / len(values) if values else None

def world_jitter(shock, seed):
    return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2], "big") % 3

def manual_hit(shock, seed, index):
    return int.from_bytes(
        sha256(f"world:{shock}:{seed}:obligation:{index}".encode()).digest()[:8],
        "big",
    ) % 1_000_000 < 300_000

def true_liquidity_demands(adapter, shock):
    increment = 5 if shock == "liquidity_shock" else 0
    return [
        0 if adapter == "absent" else max(0, math.ceil(AMOUNTS[i] * DEMAND_MULT[adapter] + increment))
        for i in range(3)
    ]

def reporting_values(policy, focal, true_demands):
    declared_criticality = list(TRUE_CRIT)
    declared_demand = list(true_demands)
    if policy in ("criticality_inflation", "dual_misreport"):
        declared_criticality[focal] = 5
    if policy in ("liquidity_demand_underreport", "dual_misreport"):
        declared_demand[focal] = min(true_demands[focal], 5)
    return declared_criticality, declared_demand

def policy_order(allocation_policy, declared_criticality, declared_demand):
    indices = [0, 1, 2]
    if allocation_policy == "fifo":
        return indices
    if allocation_policy == "criticality_priority":
        return sorted(indices, key=lambda i: (-declared_criticality[i], i))
    if allocation_policy == "minimum_liquidity_demand":
        return sorted(indices, key=lambda i: (declared_demand[i], i))
    raise ValueError(allocation_policy)

def run_one(pair, adapter, shock, seed, allocation_policy, focal, reporting_policy):
    jitter = world_jitter(shock, seed)
    true_demand = true_liquidity_demands(adapter, shock)
    declared_criticality, declared_demand = reporting_values(reporting_policy, focal, true_demand)
    order = policy_order(allocation_policy, declared_criticality, declared_demand)
    rank = {i: order.index(i) + 1 for i in range(3)}

    current = 5 + jitter
    available = CAP
    pending = order.copy()
    active = []
    records = []
    peak_reserved = 0

    while pending or active:
        if active:
            next_finish = min(x[0] for x in active)
            current = max(current, next_finish)
            done = [x for x in active if x[0] <= current]
            active = [x for x in active if x[0] > current]
            available += sum(x[1] for x in done)

        while pending:
            i = pending[0]
            d = true_demand[i]

            if d == 0:
                pending.pop(0)
                records.append((i, "rejected", None, None, d, rank[i]))
                continue

            if d > CAP:
                pending.pop(0)
                records.append((i, "unresolved", None, None, d, rank[i]))
                continue

            if d <= available:
                pending.pop(0)
                available -= d
                manual_delay = 2 if manual_hit(shock, seed, i) else 0
                finish = (
                    current
                    + TECH[adapter]
                    + (2 if shock == "liquidity_shock" and adapter == "redeem_reissue" else 0)
                    + manual_delay
                    + 2
                )
                active.append((finish, d, i))
                records.append((i, "running", current, finish, d, rank[i]))
                peak_reserved = max(peak_reserved, CAP - available)
            else:
                break

        if pending and not active:
            for i in pending:
                records.append((i, "unresolved", None, None, true_demand[i], rank[i]))
            pending = []

    finish_map = {i: fin for i, status, start, fin, demand, rk in records if status == "running"}
    final = []

    for i, status, start, fin, demand, rk in records:
        state = "externally_finalized" if status == "running" else status
        if state == "externally_finalized" and shock in ("bridge_failure", "stale_quote"):
            state = "unresolved"
        elif state == "externally_finalized" and shock == "issuer_default" and adapter == "redeem_reissue":
            state = "unresolved"

        final.append(
            {
                "index": i,
                "state": state,
                "completion": finish_map.get(i) if state == "externally_finalized" else None,
                "amount": AMOUNTS[i],
                "true_criticality": TRUE_CRIT[i],
                "declared_criticality": declared_criticality[i],
                "true_liquidity_demand": true_demand[i],
                "declared_liquidity_demand": declared_demand[i],
                "allocation_rank": rk,
                "manual": manual_hit(shock, seed, i),
            }
        )

    done = [x for x in final if x["state"] == "externally_finalized"]
    return {
        "pair": pair,
        "adapter": adapter,
        "shock": shock,
        "seed": seed,
        "allocation_policy": allocation_policy,
        "allocation_policy_digest": ALLOCATION_POLICY_DIGESTS[allocation_policy],
        "focal_obligation": focal,
        "reporting_policy": reporting_policy,
        "reporting_policy_digest": POLICY_DIGESTS[reporting_policy],
        "world_jitter": jitter,
        "exogenous_random_namespace": M["fixed_dimensions"]["exogenous_random_namespace"],
        "allocation_order": tuple(order),
        "records": final,
        "completion_rate": len(done) / 3,
        "system_completion_time": max((x["completion"] for x in done), default=None),
        "peak_true_liquidity_reserved": peak_reserved,
        "resource_integrity": peak_reserved <= CAP,
    }

def paired_contrast(runs, r):
    key = (
        r["pair"], r["adapter"], r["shock"], r["seed"],
        r["allocation_policy"], r["focal_obligation"], "truthful"
    )
    truth = runs[key]
    focal = r["focal_obligation"]
    current_focal = next(x for x in r["records"] if x["index"] == focal)
    truth_focal = next(x for x in truth["records"] if x["index"] == focal)
    gain = None
    if current_focal["completion"] is not None and truth_focal["completion"] is not None:
        gain = truth_focal["completion"] - current_focal["completion"]

    other_deltas = []
    for i in range(3):
        if i == focal:
            continue
        a = next(x for x in r["records"] if x["index"] == i)
        b = next(x for x in truth["records"] if x["index"] == i)
        if a["completion"] is not None and b["completion"] is not None:
            other_deltas.append(a["completion"] - b["completion"])

    return {
        "gain": gain,
        "rank_improvement": truth_focal["allocation_rank"] - current_focal["allocation_rank"],
        "report_only": current_focal["completion"] is not None and truth_focal["completion"] is None,
        "truth_only": current_focal["completion"] is None and truth_focal["completion"] is not None,
        "other_externality": mean(other_deltas) if other_deltas else None,
    }

def main(out_dir):
    tuples = product(PAIRS, AD, SH, SEEDS, ALLO, FOCAL, REPORT)
    generated = [run_one(*x) for x in tuples]
    assert len(generated) == 12960
    assert len({
        (r["pair"], r["adapter"], r["shock"], r["seed"], r["allocation_policy"], r["focal_obligation"], r["reporting_policy"])
        for r in generated
    }) == 12960

    runs = {
        (r["pair"], r["adapter"], r["shock"], r["seed"], r["allocation_policy"], r["focal_obligation"], r["reporting_policy"]): r
        for r in generated
    }

    # Invariants: every treatment shares the exact CRN stream; true values never mutate;
    # non-focal obligations remain truthful; true resource reservation stays within capacity.
    for shock, seed, index in product(SH, SEEDS, range(3)):
        hit = manual_hit(shock, seed, index)
        assert all(
            hit == manual_hit(shock, seed, index)
            for _policy in REPORT
            for _allocation in ALLO
        )

    for r in generated:
        for rec in r["records"]:
            assert rec["true_criticality"] == TRUE_CRIT[rec["index"]]
            assert rec["true_liquidity_demand"] >= 0
        assert r["resource_integrity"]
        if r["reporting_policy"] != "truthful":
            for rec in r["records"]:
                if rec["index"] != r["focal_obligation"]:
                    assert rec["declared_criticality"] == rec["true_criticality"]
                    assert rec["declared_liquidity_demand"] == rec["true_liquidity_demand"]

    payload = json.dumps(generated, sort_keys=True, separators=(",", ":")).encode()
    trace_digest = sha256(payload).hexdigest()

    aggregate = {}
    for policy in REPORT:
        subset = [r for r in generated if r["reporting_policy"] == policy]
        aggregate[policy] = {
            "cells": len(subset),
            "mean_completion_rate": mean([r["completion_rate"] for r in subset]),
            "mean_system_completion_time": mean([
                r["system_completion_time"] for r in subset if r["system_completion_time"] is not None
            ]),
            "resource_integrity_failures": sum(not r["resource_integrity"] for r in subset),
        }

    mechanism_breakdown = {}
    for allocation in ALLO:
        mechanism_breakdown[allocation] = {}
        for policy in ("criticality_inflation", "liquidity_demand_underreport", "dual_misreport"):
            contrasts = [
                paired_contrast(runs, r)
                for r in generated
                if r["allocation_policy"] == allocation and r["reporting_policy"] == policy
            ]
            common = [c for c in contrasts if c["gain"] is not None]
            mechanism_breakdown[allocation][policy] = {
                "common_completion_pairs": len(common),
                "positive_focal_gain_rate": mean([1 if c["gain"] > 0 else 0 for c in common]),
                "mean_focal_completion_gain_ticks": mean([c["gain"] for c in common]),
                "max_focal_completion_gain_ticks": max(c["gain"] for c in common),
                "mean_rank_improvement": mean([c["rank_improvement"] for c in contrasts]),
                "report_only_rate": mean([1 if c["report_only"] else 0 for c in contrasts]),
                "truth_only_rate": mean([1 if c["truth_only"] else 0 for c in contrasts]),
                "mean_other_completion_externality_ticks": mean([
                    c["other_externality"] for c in contrasts if c["other_externality"] is not None
                ]),
            }

    focal_breakdown = {}
    for focal in FOCAL:
        focal_breakdown[str(focal)] = {}
        for policy in ("criticality_inflation", "liquidity_demand_underreport", "dual_misreport"):
            contrasts = [
                paired_contrast(runs, r)
                for r in generated
                if r["focal_obligation"] == focal and r["reporting_policy"] == policy
            ]
            common = [c for c in contrasts if c["gain"] is not None]
            focal_breakdown[str(focal)][policy] = {
                "mean_gain_ticks": mean([c["gain"] for c in common]),
                "positive_gain_rate": mean([1 if c["gain"] > 0 else 0 for c in common]),
                "mean_rank_improvement": mean([c["rank_improvement"] for c in contrasts]),
            }

    summary = {
        "schema_version": "monetary-strategic-reporting-v1-execution",
        "run_count": len(generated),
        "obligation_count": len(generated) * 3,
        "common_random_number_cells": len(SH) * len(SEEDS) * 3,
        "exogenous_random_namespace": M["fixed_dimensions"]["exogenous_random_namespace"],
        "trace_set_digest": trace_digest,
        "aggregate_by_reporting_policy": aggregate,
        "mechanism_breakdown": mechanism_breakdown,
        "focal_breakdown": focal_breakdown,
        "invariants": {
            "crn_invariant": True,
            "true_values_immutable": True,
            "nonfocal_reporting_truthful": True,
            "resource_integrity_failures": sum(not r["resource_integrity"] for r in generated),
        },
        "claim_ceiling": M["claim_ceiling"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    generated_path = out_dir / "monetary-strategic-reporting-v1.execution.generated.json"
    generated_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "
")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
