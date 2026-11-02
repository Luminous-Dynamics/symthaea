#!/usr/bin/env python3
"""Valid-null process simulation for exact-threshold two-phase source escrow.

All evidence updates use the Bernoulli LR factors 7/5 and 3/5. The A2-style
running-maximum adjuster is floored to a fixed-point rational grid, preserving
a pointwise lower bound on the valid adjuster. Score, reserve, assignment,
credit and e-BH funding arithmetic use Fraction.

This is an exploratory simulator, not a theorem proof or a production Rust test.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
from fractions import Fraction
from typing import Any

import numpy as np


METHODS = ("no_donation", "fixed_next", "two_phase_min_credit")
DEPENDENCE = ("independent", "shared_uniform", "balanced_negative")
SCENARIOS = (
    "all_null",
    "immediate_bridge",
    "late_bridge",
    "far_cluster",
    "no_useful",
    "switching",
)


def make_scenario(name: str, n: int, looks: int) -> dict[str, Any]:
    p = np.full((looks, n), 0.5, dtype=np.float64)
    null = np.ones(n, dtype=bool)
    arrivals = np.zeros(n, dtype=np.int64)

    if name == "all_null":
        pass
    elif name == "immediate_bridge":
        p[:, 0] = 0.80
        p[:, 1] = 0.64
        p[:, 2:min(15, n)] = 0.80
        null[:min(15, n)] = False
    elif name == "late_bridge":
        p[:, 0] = 0.80
        p[:, 1:min(13, n)] = 0.80
        target = min(22, n - 1)
        arrivals[target] = looks * 3 // 5
        p[:, target] = 0.64
        null[:min(13, n)] = False
        null[target] = False
    elif name == "far_cluster":
        p[:, 0] = 0.80
        lo, hi = min(18, n), min(23, n)
        p[:, lo:hi] = 0.70
        null[0] = False
        null[lo:hi] = False
    elif name == "no_useful":
        p[:, 0] = 0.80
        null[0] = False
    elif name == "switching":
        p[:, 0] = 0.80
        null[0] = False
        cuts = [0, looks // 4, looks // 2, 3 * looks // 4, looks]
        groups = [
            range(1, min(5, n)),
            range(5, min(9, n)),
            range(9, min(13, n)),
            range(13, min(17, n)),
        ]
        for group, lo, hi in zip(groups, cuts[:-1], cuts[1:]):
            ids = list(group)
            p[lo:hi, ids] = 0.80
            null[ids] = False
    else:
        raise ValueError(f"unknown scenario: {name}")

    return {"name": name, "p": p, "null": null, "arrivals": arrivals}


def generate_observations(
    scenario: dict[str, Any], dependence: str, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = scenario["p"]
    null = scenario["null"]
    arrivals = scenario["arrivals"]
    looks, n = p.shape
    y = np.zeros((looks, n), dtype=bool)

    for t in range(looks):
        active_ids = np.flatnonzero(arrivals <= t)
        if dependence == "independent":
            y[t, active_ids] = (
                rng.random(len(active_ids)) < p[t, active_ids]
            )
        elif dependence == "shared_uniform":
            shared_u = rng.random()
            y[t, active_ids] = shared_u < p[t, active_ids]
        elif dependence == "balanced_negative":
            null_ids = [i for i in active_ids if null[i]]
            alternative_ids = [i for i in active_ids if not null[i]]
            if null_ids:
                count = len(null_ids)
                successes = count // 2
                if count % 2:
                    successes += int(rng.integers(0, 2))
                permutation = rng.permutation(null_ids)
                y[t, permutation[:successes]] = True
            if alternative_ids:
                y[t, alternative_ids] = (
                    rng.random(len(alternative_ids))
                    < p[t, alternative_ids]
                )
        else:
            raise ValueError(f"unknown dependence: {dependence}")
    return y


def adjusted_evidence_path(
    observations: np.ndarray,
    arrivals: np.ndarray,
    scale: int = 1_000_000,
) -> list[list[Fraction]]:
    """Exact LR products and downward-rounded A2 running-max adjuster.

    A(x)=sqrt(x)-1 has integral budget one. Replacing it by its fixed-grid
    floor is a pointwise lower approximation, preserving validity and
    pathwise monotonicity.
    """
    looks, n = observations.shape
    current_num = [1] * n
    current_den = [1] * n
    maximum_num = [1] * n
    maximum_den = [1] * n
    path: list[list[Fraction]] = []

    for t in range(looks):
        row = []
        for i in range(n):
            if t >= arrivals[i]:
                current_num[i] *= 7 if observations[t, i] else 3
                current_den[i] *= 5
                if (
                    current_num[i] * maximum_den[i]
                    > maximum_num[i] * current_den[i]
                ):
                    maximum_num[i] = current_num[i]
                    maximum_den[i] = current_den[i]
            scaled_root = math.isqrt(
                (maximum_num[i] * scale * scale) // maximum_den[i]
            )
            adjusted_integer = max(0, scaled_root - scale)
            row.append(Fraction(adjusted_integer, scale))
        path.append(row)
    return path


def ebh_rejections(
    scores: list[Fraction], alpha: Fraction
) -> set[int]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    k_star = 0
    for k, candidate in enumerate(order, start=1):
        if scores[candidate] >= Fraction(1, 1) / (alpha * k):
            k_star = k
    return set(order[:k_star])


def minimum_credit_batch(
    scores: list[Fraction], alpha: Fraction
) -> list[Fraction]:
    """O(n log n) exact minimum credit for every candidate."""
    n = len(scores)
    ordered = sorted(scores, reverse=True)
    count = 0
    feasible_ranks: list[int] = []
    for k in range(1, n + 1):
        threshold = Fraction(1, 1) / (alpha * k)
        while count < n and ordered[count] >= threshold:
            count += 1
        if count == k - 1:
            feasible_ranks.append(k)

    rejected = ebh_rejections(scores, alpha)
    deficits: list[Fraction] = []
    for candidate, score in enumerate(scores):
        if candidate in rejected:
            deficits.append(Fraction(0))
            continue
        lo, hi = 0, len(feasible_ranks)
        while lo < hi:
            mid = (lo + hi) // 2
            k = feasible_ranks[mid]
            if score < Fraction(1, 1) / (alpha * k):
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            raise AssertionError("no feasible e-BH rank for candidate")
        k = feasible_ranks[lo - 1]
        deficits.append(Fraction(1, 1) / (alpha * k) - score)
    return deficits


def score_and_fdp(
    active_ids: list[int],
    gamma: list[Fraction],
    x: list[Fraction],
    incoming: list[Fraction],
    outgoing: list[Fraction],
    null: np.ndarray,
    alpha: Fraction,
) -> tuple[Fraction, Fraction, bool, int]:
    scores = [
        gamma[i] * x[i] + incoming[i] - outgoing[i]
        for i in active_ids
    ]
    rejected_local = ebh_rejections(scores, alpha) if scores else set()
    rejected_ids = [active_ids[k] for k in rejected_local]
    false_count = sum(bool(null[i]) for i in rejected_ids)
    true_count = len(rejected_ids) - false_count
    fdp = F(false_count, len(rejected_ids)) if rejected_ids else F(0)
    any_false = false_count > 0
    return fdp, F(true_count), any_false, len(rejected_ids)


def simulate_policy(
    x_path: list[list[Fraction]],
    scenario: dict[str, Any],
    gamma: list[Fraction],
    alpha: Fraction,
    policy: str,
) -> dict[str, Any]:
    n = len(gamma)
    looks = len(x_path)
    null = scenario["null"]
    arrivals = scenario["arrivals"]

    x = [Fraction(0) for _ in range(n)]
    incoming = [Fraction(0) for _ in range(n)]
    outgoing = [Fraction(0) for _ in range(n)]
    previous_reserve = [Fraction(0) for _ in range(n)]
    active = [False] * n
    previous_scores = [Fraction(0) for _ in range(n)]

    bank: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    source_token_ordinal = [0] * n
    fixed_pending: list[dict[str, Any]] = []
    fixed_outside = Fraction(0)

    sup_fdp = Fraction(0)
    terminal_fdp = Fraction(0)
    terminal_power = Fraction(0)
    any_false = False
    first_true_discovery: int | None = None
    total_reserved = Fraction(0)
    total_credited = Fraction(0)
    assignment_count = 0
    maximum_prefix_error = Fraction(0)
    maximum_cap_error = Fraction(0)
    maximum_score_decrease = Fraction(0)

    def report(look: int) -> tuple[Fraction, Fraction]:
        nonlocal sup_fdp, terminal_fdp, terminal_power, any_false
        nonlocal first_true_discovery
        active_ids = [i for i in range(n) if active[i]]
        fdp, true_count, has_false, rejected_count = score_and_fdp(
            active_ids, gamma, x, incoming, outgoing, null, alpha
        )
        current_scores = [
            gamma[i] * x[i] + incoming[i] - outgoing[i]
            for i in range(n)
        ]
        if any(new < old for new, old in zip(current_scores, previous_scores)):
            raise AssertionError("score decreased at a reporting checkpoint")
        previous_scores[:] = current_scores
        power = (
            true_count / int((~null).sum())
            if int((~null).sum()) else Fraction(0)
        )
        sup_fdp = max(sup_fdp, fdp)
        terminal_fdp = fdp
        terminal_power = power
        any_false = any_false or has_false
        if true_count and first_true_discovery is None:
            first_true_discovery = look
        return fdp, power

    def check_invariants(look: int) -> None:
        nonlocal maximum_prefix_error, maximum_cap_error, maximum_score_decrease
        for i in range(n):
            cap_excess = outgoing[i] - gamma[i] * min(x[i], Fraction(1))
            maximum_cap_error = max(maximum_cap_error, cap_excess)
            if cap_excess > 0:
                raise AssertionError("donor cap violation")
        prefix = Fraction(0)
        for i in range(n):
            prefix += incoming[i] - outgoing[i]
            maximum_prefix_error = max(maximum_prefix_error, prefix)
            if prefix > 0:
                raise AssertionError("forward-prefix budget violation")

        if policy == "fixed_next":
            escrow = fixed_outside + sum(
                (token["remaining"] for token in fixed_pending), Fraction(0)
            )
            for token in fixed_pending:
                assert token["original"] == token["remaining"] + token["credited"]
        elif policy == "two_phase_min_credit":
            escrow = sum(
                (token["remaining"] for source in bank for token in source),
                Fraction(0),
            )
            for source in bank:
                for token in source:
                    assert token["original"] == (
                        token["remaining"] + token["credited"]
                    )
        else:
            escrow = Fraction(0)

        if sum(incoming, Fraction(0)) + escrow != sum(outgoing, Fraction(0)):
            raise AssertionError("global escrow conservation")

        scores = [
            gamma[i] * x[i] + incoming[i] - outgoing[i]
            for i in range(n)
        ]
        for new, old in zip(scores, previous_scores):
            decrease = old - new
            maximum_score_decrease = max(maximum_score_decrease, decrease)
            if new < 0 or decrease > 0:
                raise AssertionError("score negative or decreasing")

    for look in range(looks):
        # Pre-observation checkpoint: activate predictable arrivals first.
        for i in range(n):
            if not active[i] and arrivals[i] == look:
                active[i] = True

        if policy == "fixed_next":
            remaining_pending = []
            for token in fixed_pending:
                destination = token["destination"]
                if active[destination]:
                    incoming[destination] += token["remaining"]
                    total_credited_local = token["remaining"]
                    token["credited"] += total_credited_local
                    token["remaining"] = Fraction(0)
                if token["remaining"]:
                    remaining_pending.append(token)
            fixed_pending = remaining_pending

        # Every activation/release checkpoint is a reporting time.
        report(look)

        if policy == "two_phase_min_credit":
            active_ids = [i for i in range(n) if active[i]]
            if active_ids:
                scores = [
                    gamma[i] * x[i] + incoming[i] - outgoing[i]
                    for i in active_ids
                ]
                deficits = minimum_credit_batch(scores, alpha)
                rejected = ebh_rejections(scores, alpha)
                fundable: list[tuple[Fraction, int]] = []
                for local_id, deficit in enumerate(deficits):
                    destination = active_ids[local_id]
                    if deficit <= 0 or local_id in rejected:
                        continue
                    available = sum(
                        (
                            token["remaining"]
                            for source in range(destination)
                            for token in bank[source]
                        ),
                        Fraction(0),
                    )
                    if available >= deficit:
                        fundable.append((deficit, destination))

                # One exact assignment per pre-observation checkpoint.
                if fundable:
                    amount, destination = min(
                        fundable, key=lambda item: (item[0], item[1])
                    )
                    needed = amount
                    for source in range(destination):
                        for token in bank[source]:
                            take = min(token["remaining"], needed)
                            if take:
                                token["remaining"] -= take
                                token["credited"] += take
                                incoming[destination] += take
                                total_credited += take
                                needed -= take
                            if needed == 0:
                                break
                        if needed == 0:
                            break
                    assert needed == 0
                    assignment_count += 1
                    post_scores = [
                        gamma[i] * x[i] + incoming[i] - outgoing[i]
                        for i in active_ids
                    ]
                    assert active_ids.index(destination) in ebh_rejections(
                        post_scores, alpha
                    ), "exact-minimum credit did not enter e-BH"
                    report(look)

        # Current look's observations are now revealed. Evidence plus its own
        # reserve deduction is treated as one atomic score transaction.
        new_x = x_path[look]
        old_x = x[:]
        for i in range(n):
            if not active[i]:
                continue
            x[i] = new_x[i]
            if policy == "none":
                continue

            target = min(
                max(gamma[i] * x[i] - 1, Fraction(0)),
                gamma[i],
            )
            reserve_increment = target - previous_reserve[i]
            evidence_increment = x[i] - old_x[i]
            if reserve_increment < 0:
                raise AssertionError("reserve rollback")
            if reserve_increment > gamma[i] * evidence_increment:
                raise AssertionError("own-increment cap violation")
            if reserve_increment:
                outgoing[i] += reserve_increment
                total_reserved += reserve_increment
                if policy == "fixed_next":
                    destination = i + 1
                    if destination < n:
                        token = {
                            "source": i,
                            "destination": destination,
                            "original": reserve_increment,
                            "remaining": reserve_increment,
                            "credited": Fraction(0),
                        }
                        if active[destination]:
                            incoming[destination] += reserve_increment
                            total_credited += reserve_increment
                            token["credited"] += reserve_increment
                            token["remaining"] = Fraction(0)
                        else:
                            fixed_pending.append(token)
                    else:
                        fixed_outside += reserve_increment
                elif policy == "two_phase_min_credit":
                    source_token_ordinal[i] += 1
                    bank[i].append({
                        "source": i,
                        "token_id": (i, source_token_ordinal[i]),
                        "original": reserve_increment,
                        "remaining": reserve_increment,
                        "credited": Fraction(0),
                    })
            previous_reserve[i] = target

        check_invariants(look)
        report(look)
        previous_scores = [
            gamma[i] * x[i] + incoming[i] - outgoing[i]
            for i in range(n)
        ]

    return {
        "supfdr": float(sup_fdp),
        "terminal_fdr": float(terminal_fdp),
        "p_any_false": float(any_false),
        "terminal_power": float(terminal_power),
        "mean_first_true_discovery_look": (
            first_true_discovery if first_true_discovery is not None else None
        ),
        "reserve_volume": float(total_reserved),
        "credited_volume": float(total_credited),
        "unassigned_or_pending_escrow": float(
            fixed_outside
            + sum((token["remaining"] for token in fixed_pending), Fraction(0))
            + sum(
                (token["remaining"] for source in bank for token in source),
                Fraction(0),
            )
        ),
        "assignment_count": assignment_count,
        "max_prefix_error": float(maximum_prefix_error),
        "max_donor_cap_error": float(maximum_cap_error),
        "max_score_decrease": float(maximum_score_decrease),
    }


def run(
    streams: int = 50,
    hypotheses: int = 24,
    looks: int = 50,
    seed: int = 10125,
    scenario_names: tuple[str, ...] = SCENARIOS,
) -> dict[str, Any]:
    alpha = Fraction(1, 10)
    gamma = [
        Fraction(1, (i + 1) * (i + 2))
        for i in range(hypotheses)
    ]
    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        scenario = make_scenario(scenario_name, hypotheses, looks)
        for dependence_index, dependence in enumerate(DEPENDENCE):
            samples: dict[str, list[dict[str, Any]]] = {
                method: [] for method in METHODS
            }
            paired_power_diffs = []
            for rep in range(streams):
                stream_seed = (
                    seed + scenario_index * 100_000
                    + dependence_index * 10_000 + rep
                )
                observations = generate_observations(
                    scenario, dependence, stream_seed
                )
                x_path = adjusted_evidence_path(
                    observations, scenario["arrivals"]
                )
                for method in METHODS:
                    result = simulate_policy(
                        x_path, scenario, gamma, alpha, method
                    )
                    samples[method].append(result)
                paired_power_diffs.append((
                    samples["two_phase_min_credit"][-1]["terminal_power"]
                    - samples["fixed_next"][-1]["terminal_power"],
                    samples["two_phase_min_credit"][-1]["terminal_power"]
                    - samples["no_donation"][-1]["terminal_power"],
                ))

            method_summary = {}
            for method, values in samples.items():
                keys = values[0].keys()
                method_summary[method] = {
                    key: (
                        statistics.mean(row[key] for row in values)
                        if row_value_is_numeric(values[0][key]) else None
                    )
                    for key in keys
                }
            delta_fixed = [pair[0] for pair in paired_power_diffs]
            delta_none = [pair[1] for pair in paired_power_diffs]
            rows.append({
                "scenario": scenario_name,
                "dependence": dependence,
                "seed_start": (
                    seed + scenario_index * 100_000
                    + dependence_index * 10_000
                ),
                "streams": streams,
                "hypotheses": hypotheses,
                "looks": looks,
                "null_hypotheses": int(scenario["null"].sum()),
                "alternative_hypotheses": int((~scenario["null"]).sum()),
                "methods": method_summary,
                "paired_power_delta_two_phase_minus_fixed_next": {
                    "mean": statistics.mean(delta_fixed),
                    "standard_error": (
                        statistics.stdev(delta_fixed) / math.sqrt(streams)
                        if streams > 1 else None
                    ),
                },
                "paired_power_delta_two_phase_minus_no_donation": {
                    "mean": statistics.mean(delta_none),
                    "standard_error": (
                        statistics.stdev(delta_none) / math.sqrt(streams)
                        if streams > 1 else None
                    ),
                },
            })

    hoeffding = math.sqrt(math.log(20.0) / (2.0 * streams))
    return {
        "benchmark": "two_phase_exact_minimum_ebh_credit_valid_processes",
        "scope": (
            "Exploratory paired Monte Carlo using conditional-mean-one null "
            "LR factors and a downward-rounded A2 adjuster. Not a proof or "
            "confirmatory result."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "streams_per_cell": streams,
            "hypotheses": hypotheses,
            "looks": looks,
            "alpha": str(alpha),
            "seed": seed,
            "weights": "gamma_i=1/[(i+1)(i+2)]",
            "weight_mass": str(sum(gamma, Fraction(0))),
            "null_lr_factors": "7/5 on success, 3/5 on failure; conditional mean one at p0=1/2",
            "adjuster": "floor_S(sqrt(running_max_LR)-1), S=1,000,000",
            "reporting": "after pre-observation assignment checkpoint and after each atomic look commit",
            "dependence": list(DEPENDENCE),
        },
        "methods": list(METHODS),
        "one_sided_95pct_hoeffding_halfwidth_per_bounded_mean": hoeffding,
        "simultaneous_uncertainty": "not applied",
        "results": rows,
        "claim_ceiling": (
            "No SOTA or superiority claim. The exact assignment policy is "
            "evaluated only as an exploratory experiment under this LR model."
        ),
    }


def row_value_is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=100)
    parser.add_argument("--hypotheses", type=int, default=24)
    parser.add_argument("--looks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10125)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if min(args.streams, args.hypotheses, args.looks) < 1:
        raise SystemExit("all dimensions must be positive")
    result = run(
        args.streams, args.hypotheses, args.looks,
        args.seed, tuple(args.scenarios)
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
