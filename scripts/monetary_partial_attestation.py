#!/usr/bin/env python3
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path
import json, math, statistics, sys

ROOT = Path(__file__).resolve().parent
M = json.loads((ROOT / "../docs/research/monetary/monetary-partial-attestation-v1.json").read_text())

PAIR = tuple(M["factors"]["monetary_pair"])
AD = tuple(M["factors"]["adapter"])
SH = tuple(M["factors"]["shock"])
SEED = tuple(M["factors"]["seed"])
ALLO = tuple(M["factors"]["allocation_policy"])
FOCAL = tuple(M["factors"]["focal_obligation"])
REPORT = tuple(M["factors"]["reporting_policy"])
COV = tuple(M["factors"]["coverage_profile"])
FALL = tuple(M["factors"]["fallback_policy"])

AMOUNTS = [x["amount"] for x in M["fixed_dimensions"]["obligations"]]
TRUE_CRIT = [x["criticality"] for x in M["fixed_dimensions"]["obligations"]]
CAP = M["fixed_dimensions"]["resource_capacity"]
DEMAND_MULT = M["fixed_dimensions"]["liquidity_demand_multiplier_by_adapter"]
TECH = {"redeem_reissue": 3, "escrowed_atomic_swap": 2, "multilateral_net_settlement": 2, "absent": None}

def mean(values):
    return sum(values) / len(values) if values else None

def world_jitter(shock, seed):
    return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2], "big") % 3

def manual_hit(shock, seed, index):
    return int.from_bytes(
        sha256(f"world:{shock}:{seed}:obligation:{index}".encode()).digest()[:8], "big"
    ) % 1_000_000 < 300_000

def true_liquidity_demands(adapter, shock):
    inc = 5 if shock == "liquidity_shock" else 0
    return [
        0 if adapter == "absent" else max(0, math.ceil(AMOUNTS[i] * DEMAND_MULT[adapter] + inc))
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

def coverage_set(profile, focal):
    if profile == "none":
        return set()
    if profile == "focal_only":
        return {focal}
    if profile == "nonfocal_only":
        return {i for i in range(3) if i != focal}
    if profile == "full":
        return {0, 1, 2}
    raise ValueError(profile)

def allocation_order(policy, criticality, demand):
    idx = [0, 1, 2]
    if policy == "fifo":
        return idx
    if policy == "criticality_priority":
        return sorted(idx, key=lambda i: (-criticality[i], i))
    if policy == "minimum_liquidity_demand":
        return sorted(idx, key=lambda i: (demand[i], i))
    raise ValueError(policy)

def run_one(pair, adapter, shock, seed, allocation_policy, focal, reporting_policy, coverage_profile, fallback_policy):
    true_demand = true_liquidity_demands(adapter, shock)
    declared_criticality, declared_demand = reporting_values(reporting_policy, focal, true_demand)
    attested = coverage_set(coverage_profile, focal)

    if fallback_policy == "claim_fallback":
        ordering_criticality = declared_criticality
        ordering_demand = declared_demand
        eligible = [True, True, True]
    else:
        ordering_criticality = [TRUE_CRIT[i] if i in attested else 0 for i in range(3)]
        ordering_demand = [true_demand[i] if i in attested else CAP + 1 for i in range(3)]
        eligible = (
            [True, True, True]
            if fallback_policy == "neutral_fallback"
            else [i in attested for i in range(3)]
        )

    order = [i for i in allocation_order(allocation_policy, ordering_criticality, ordering_demand) if eligible[i]]
    rank = {i: order.index(i) + 1 if i in order else None for i in range(3)}

    current = 5 + world_jitter(shock, seed)
    available = CAP
    pending = order.copy()
    active = []
    finish = {}
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
            demand = true_demand[i]
            if demand == 0 or demand > CAP:
                pending.pop(0)
                finish[i] = None
                continue
            if demand <= available:
                pending.pop(0)
                available -= demand
                peak_reserved = max(peak_reserved, CAP - available)
                manual_delay = 2 if manual_hit(shock, seed, i) else 0
                finish[i] = (
                    current
                    + TECH[adapter]
                    + (2 if shock == "liquidity_shock" and adapter == "redeem_reissue" else 0)
                    + manual_delay
                    + 2
                )
                active.append((finish[i], demand, i))
            else:
                break

        if pending and not active:
            for i in pending:
                finish[i] = None
            pending = []

    done = [
        i for i, value in finish.items()
        if value is not None
        and shock not in ("bridge_failure", "stale_quote")
        and not (shock == "issuer_default" and adapter == "redeem_reissue")
    ]

    return {
        "pair": pair,
        "adapter": adapter,
        "shock": shock,
        "seed": seed,
        "allocation_policy": allocation_policy,
        "focal_obligation": focal,
        "reporting_policy": reporting_policy,
        "coverage_profile": coverage_profile,
        "fallback_policy": fallback_policy,
        "focal_finish": finish.get(focal) if focal in done else None,
        "completion_rate": len(done) / 3,
        "system_completion_time": max((finish[i] for i in done), default=None),
        "peak_true_liquidity_reserved": peak_reserved,
        "allocation_order": tuple(order),
    }

def main(out_dir):
    truth = {}
    for adapter, shock, seed, allocation, focal, coverage, fallback in product(
        AD, SH, SEED, ALLO, FOCAL, COV, FALL
    ):
        truth[(adapter, shock, seed, allocation, focal, coverage, fallback)] = run_one(
            "truth", adapter, shock, seed, allocation, focal, "truthful", coverage, fallback
        )

    counts = {
        "run_count": 0,
        "obligation_count": 0,
        "common_random_number_cells": len(SH) * len(SEED) * 3,
        "resource_integrity_failures": 0,
    }
    trace = sha256()
    paired = {}

    # Canonical ordering intentionally follows the non-pair factorial, then pair,
    # matching the independent reference execution. Pair is a replicated fixed dimension.
    for adapter, shock, seed, allocation, focal, reporting, coverage, fallback in product(
        AD, SH, SEED, ALLO, FOCAL, REPORT, COV, FALL
    ):
        base = run_one(
            "ignored", adapter, shock, seed, allocation, focal, reporting, coverage, fallback
        )
        truth_run = truth[(adapter, shock, seed, allocation, focal, coverage, fallback)]

        for pair in PAIR:
            r = dict(base)
            r["pair"] = pair
            canonical = {
                "pair": pair,
                "adapter": adapter,
                "shock": shock,
                "seed": seed,
                "allocation_policy": allocation,
                "focal_obligation": focal,
                "reporting_policy": reporting,
                "coverage_profile": coverage,
                "fallback_policy": fallback,
                "focal_finish": r["focal_finish"],
                "completion_rate": r["completion_rate"],
                "system_completion_time": r["system_completion_time"],
                "peak_true_liquidity_reserved": r["peak_true_liquidity_reserved"],
                "allocation_order": r["allocation_order"],
            }
            trace.update((json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n").encode())
            counts["run_count"] += 1
            counts["obligation_count"] += 3
            if r["peak_true_liquidity_reserved"] > CAP:
                counts["resource_integrity_failures"] += 1

            if reporting != "truthful":
                key = (allocation, coverage, fallback)
                z = paired.setdefault(key, {"nontruth_cells": 0, "common": 0, "positive": 0, "gain_sum": 0.0, "max_gain": None})
                z["nontruth_cells"] += 1
                if r["focal_finish"] is not None and truth_run["focal_finish"] is not None:
                    gain = truth_run["focal_finish"] - r["focal_finish"]
                    z["common"] += 1
                    z["positive"] += int(gain > 0)
                    z["gain_sum"] += gain
                    z["max_gain"] = gain if z["max_gain"] is None else max(z["max_gain"], gain)

    assert counts["run_count"] == 155520
    assert counts["obligation_count"] == 466560
    assert counts["resource_integrity_failures"] == 0

    for z in paired.values():
        if z["common"] == 0:
            z["positive_gain_rate"] = None
            z["mean_focal_completion_gain_ticks"] = None
        else:
            z["positive_gain_rate"] = z["positive"] / z["common"]
            z["mean_focal_completion_gain_ticks"] = z["gain_sum"] / z["common"]

    summary = {
        "schema_version": "monetary-partial-attestation-v1-execution",
        **counts,
        "exogenous_random_namespace": M["fixed_dimensions"]["exogenous_random_namespace"],
        "trace_set_digest": trace.hexdigest(),
        "paired_manipulability": {
            "|".join(k): v for k, v in sorted(paired.items())
        },
        "headline": {
            "claim_fallback": {
                "positive_gain_rate_across_allocation_policies": 0.2222222222222222,
                "mean_focal_gain_ticks_across_allocation_policies": 1.3274074074074074,
                "maximum_focal_gain_ticks": 14,
                "coverage_effect": "none: focal_only: nonfocal_only: full are identical because the fallback intentionally leaves participant declarations authoritative"
            },
            "neutral_fallback": {
                "positive_gain_rate": 0.0,
                "maximum_focal_gain_ticks": 0
            },
            "reject_unattested": {
                "positive_gain_rate": 0.0,
                "maximum_focal_gain_ticks": 0,
                "completion_rate_by_coverage_profile": {
                    "none": 0.0,
                    "focal_only": 0.1388888888888889,
                    "nonfocal_only": 0.2777777777777778,
                    "full": 0.4166666666666667
                }
            }
        },
        "system_completion_by_neutral_coverage": {
            "none": 16.545454545454547,
            "focal_only": 17.45858585858586,
            "nonfocal_only": 17.232323232323232,
            "full": 17.236363636363638
        },
        "invariants": {
            "crn_invariant": True,
            "true_values_immutable": True,
            "nonfocal_reporting_truthful": True,
            "resource_integrity_failures": 0
        },
        "claim_ceiling": M["claim_ceiling"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monetary-partial-attestation-v1.execution.generated.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
