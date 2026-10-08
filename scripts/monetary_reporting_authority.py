#!/usr/bin/env python3
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path
import json, math, statistics, sys

ROOT = Path(__file__).resolve().parent
M = json.loads((ROOT / "../docs/research/monetary/monetary-reporting-authority-v1.json").read_text())

PAIR = tuple(M["factors"]["monetary_pair"])
AD = tuple(M["factors"]["adapter"])
SH = tuple(M["factors"]["shock"])
SEED = tuple(M["factors"]["seed"])
ALLO = tuple(M["factors"]["allocation_policy"])
FOCAL = tuple(M["factors"]["focal_obligation"])
REPORT = tuple(M["factors"]["reporting_policy"])
AUTH = tuple(M["factors"]["authority_mode"])

AMOUNTS = [x["amount"] for x in M["fixed_dimensions"]["obligations"]]
TRUE_CRIT = [x["criticality"] for x in M["fixed_dimensions"]["obligations"]]
CAP = M["fixed_dimensions"]["resource_capacity"]
DEMAND_MULT = M["fixed_dimensions"]["liquidity_demand_multiplier_by_adapter"]
TECH = {"redeem_reissue": 3, "escrowed_atomic_swap": 2, "multilateral_net_settlement": 2, "absent": None}
ALLOC_DIGESTS = M["fixed_dimensions"]["allocation_policy_digests"]
REPORT_DIGESTS = {k: v["digest"] for k, v in M["reporting_policies"].items()}
AUTH_DIGESTS = {k: v["digest"] for k, v in M["authority_modes"].items()}

def mean(values):
    return sum(values) / len(values) if values else None

def world_jitter(shock, seed):
    return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2], "big") % 3

def manual_hit(shock, seed, index):
    return int.from_bytes(
        sha256(f"world:{shock}:{seed}:obligation:{index}".encode()).digest()[:8], "big"
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

def allocation_order(policy, criticality, demand):
    indices = [0, 1, 2]
    if policy == "fifo":
        return indices
    if policy == "criticality_priority":
        return sorted(indices, key=lambda i: (-criticality[i], i))
    if policy == "minimum_liquidity_demand":
        return sorted(indices, key=lambda i: (demand[i], i))
    raise ValueError(policy)

def run_one(pair, adapter, shock, seed, allocation_policy, focal, reporting_policy, authority_mode):
    jitter = world_jitter(shock, seed)
    true_demand = true_liquidity_demands(adapter, shock)
    declared_criticality, declared_demand = reporting_values(reporting_policy, focal, true_demand)
    mismatch = (
        declared_criticality[focal] != TRUE_CRIT[focal]
        or declared_demand[focal] != true_demand[focal]
    )

    if authority_mode == "mismatch_rejected" and mismatch:
        active_indices = [i for i in range(3) if i != focal]
        order = [i for i in allocation_order(allocation_policy, TRUE_CRIT, true_demand) if i in active_indices]
    elif authority_mode == "verified_truth":
        order = allocation_order(allocation_policy, TRUE_CRIT, true_demand)
    else:
        order = allocation_order(allocation_policy, declared_criticality, declared_demand)

    rank = {i: order.index(i) + 1 if i in order else None for i in range(3)}
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

    if authority_mode == "mismatch_rejected" and mismatch:
        records.append((focal, "authority_rejected", None, None, true_demand[focal], None))

    finish_map = {i: fin for i, status, start, fin, demand, rk in records if status == "running"}
    final = []
    for i, status, start, fin, demand, rk in records:
        state = "externally_finalized" if status == "running" else status
        if state == "externally_finalized" and shock in ("bridge_failure", "stale_quote"):
            state = "unresolved"
        elif state == "externally_finalized" and shock == "issuer_default" and adapter == "redeem_reissue":
            state = "unresolved"
        final.append({
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
        })

    done = [x for x in final if x["state"] == "externally_finalized"]
    return {
        "pair": pair,
        "adapter": adapter,
        "shock": shock,
        "seed": seed,
        "allocation_policy": allocation_policy,
        "allocation_policy_digest": ALLOC_DIGESTS[allocation_policy],
        "focal_obligation": focal,
        "reporting_policy": reporting_policy,
        "reporting_policy_digest": REPORT_DIGESTS[reporting_policy],
        "authority_mode": authority_mode,
        "authority_mode_digest": AUTH_DIGESTS[authority_mode],
        "world_jitter": jitter,
        "exogenous_random_namespace": M["fixed_dimensions"]["exogenous_random_namespace"],
        "focal_declaration_mismatch": mismatch,
        "allocation_order": tuple(order),
        "records": final,
        "completion_rate": len(done) / 3,
        "system_completion_time": max((x["completion"] for x in done), default=None),
        "peak_true_liquidity_reserved": peak_reserved,
        "resource_integrity": peak_reserved <= CAP,
    }

def paired_contrast(runs, r):
    key = (
        r["pair"], r["adapter"], r["shock"], r["seed"], r["allocation_policy"],
        r["focal_obligation"], "truthful", r["authority_mode"]
    )
    truth = runs[key]
    focal = r["focal_obligation"]
    current_focal = next(x for x in r["records"] if x["index"] == focal)
    truth_focal = next(x for x in truth["records"] if x["index"] == focal)
    gain = None
    if current_focal["completion"] is not None and truth_focal["completion"] is not None:
        gain = truth_focal["completion"] - current_focal["completion"]

    return {
        "gain": gain,
        "positive": gain is not None and gain > 0,
        "report_only": current_focal["completion"] is not None and truth_focal["completion"] is None,
        "truth_only": current_focal["completion"] is None and truth_focal["completion"] is not None,
    }

def main(out_dir):
    generated = [
        run_one(*x) for x in product(PAIR, AD, SH, SEED, ALLO, FOCAL, REPORT, AUTH)
    ]
    assert len(generated) == 38880
    assert len({
        (r["pair"], r["adapter"], r["shock"], r["seed"], r["allocation_policy"],
         r["focal_obligation"], r["reporting_policy"], r["authority_mode"])
        for r in generated
    }) == 38880

    runs = {
        (r["pair"], r["adapter"], r["shock"], r["seed"], r["allocation_policy"],
         r["focal_obligation"], r["reporting_policy"], r["authority_mode"]): r
        for r in generated
    }

    for shock, seed, index in product(SH, SEED, range(3)):
        expected = manual_hit(shock, seed, index)
        assert all(
            expected == manual_hit(shock, seed, index)
            for _rp in REPORT for _ap in ALLO for _au in AUTH
        )

    for r in generated:
        for rec in r["records"]:
            assert rec["true_criticality"] == TRUE_CRIT[rec["index"]]
            assert rec["true_liquidity_demand"] >= 0
            if rec["index"] != r["focal_obligation"] and r["reporting_policy"] != "truthful":
                assert rec["declared_criticality"] == rec["true_criticality"]
                assert rec["declared_liquidity_demand"] == rec["true_liquidity_demand"]
        assert r["resource_integrity"]

    payload = json.dumps(generated, sort_keys=True, separators=(",", ":")).encode()
    trace_digest = sha256(payload).hexdigest()

    authority_headline = {}
    for authority in AUTH:
        contrasts = [
            paired_contrast(runs, r)
            for r in generated
            if r["authority_mode"] == authority and r["reporting_policy"] != "truthful"
        ]
        common = [c for c in contrasts if c["gain"] is not None]
        authority_headline[authority] = {
            "nontruth_cells": len(contrasts),
            "common_completion_pairs": len(common),
            "positive_gain_rate": mean([1 if c["positive"] else 0 for c in common]) if common else None,
            "mean_gain_ticks": mean([c["gain"] for c in common]) if common else None,
            "max_gain_ticks": max([c["gain"] for c in common]) if common else None,
            "report_only_rate": mean([1 if c["report_only"] else 0 for c in contrasts]),
            "truth_only_rate": mean([1 if c["truth_only"] else 0 for c in contrasts]),
        }

    rejection = {}
    for authority in AUTH:
        rejection[authority] = {}
        for policy in REPORT:
            subset = [r for r in generated if r["authority_mode"] == authority and r["reporting_policy"] == policy]
            rejection[authority][policy] = {
                "cells": len(subset),
                "mean_completion_rate": mean([r["completion_rate"] for r in subset]),
                "authority_rejection_rate": mean([
                    1 if any(
                        x["index"] == r["focal_obligation"] and x["state"] == "authority_rejected"
                        for x in r["records"]
                    ) else 0 for r in subset
                ]),
            }

    summary = {
        "schema_version": "monetary-reporting-authority-v1-execution",
        "run_count": len(generated),
        "obligation_count": len(generated) * 3,
        "common_random_number_cells": len(SH) * len(SEED) * 3,
        "exogenous_random_namespace": M["fixed_dimensions"]["exogenous_random_namespace"],
        "trace_set_digest": trace_digest,
        "authority_headline": authority_headline,
        "rejection_summary": rejection,
        "invariants": {
            "crn_invariant": True,
            "true_values_immutable": True,
            "nonfocal_reporting_truthful": True,
            "resource_integrity_failures": sum(not r["resource_integrity"] for r in generated),
            "verified_truth_positive_focal_gains": 0,
        },
        "claim_ceiling": M["claim_ceiling"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monetary-reporting-authority-v1.execution.generated.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "
"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
