#!/usr/bin/env python3
"""Exploratory paired benchmark for slack-anchored compound e-BH.

Null observations use a Bernoulli LR martingale with p0=1/2 and factors 7/5
and 3/5. A2 adjustment is rounded down to an exact fixed-point grid.
The anchored closure algorithm is imported from anchored_compound_ebh.py.

This benchmark is not a theorem proof or confirmatory statistical result.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from fractions import Fraction as F
from typing import Any

from anchored_compound_ebh import (
    anchored_rejections,
    closure_deficit,
    finite_envelope_slack,
    finite_envelope_weight_prefix,
    projective_weight_prefix,
    weighted_compound_ebh,
)

DEPENDENCE = ("independent", "shared_uniform", "balanced_negative")
SCENARIOS = (
    "all_null",
    "sparse_strong",
    "dense_weak",
    "late_cluster",
    "switching",
)
DELTAS = (F(1, 100), F(1, 20), F(1, 10), F(1, 5))
SCALE = 1_000_000


def make_scenario(name: str, n: int, looks: int) -> dict[str, Any]:
    null = [True] * n
    p = [[0.5 for _ in range(n)] for _ in range(looks)]
    arrivals = [min((i // 4) * 3, looks - 1) for i in range(n)]

    def mark_alt(ids, probability):
        for i in ids:
            null[i] = False
            for t in range(looks):
                p[t][i] = probability

    if name == "all_null":
        pass
    elif name == "sparse_strong":
        mark_alt([0, min(8, n - 1), min(16, n - 1)], 0.78)
    elif name == "dense_weak":
        mark_alt(list(range(min(n // 2, 12))), 0.62)
    elif name == "late_cluster":
        mark_alt([0, 1], 0.80)
        late = list(range(max(0, n - 8), n))
        mark_alt(late, 0.70)
        for i in late:
            arrivals[i] = max(arrivals[i], looks // 2)
    elif name == "switching":
        groups = [
            list(range(0, min(3, n))),
            list(range(3, min(6, n))),
            list(range(6, min(9, n))),
            list(range(9, min(12, n))),
        ]
        for group in groups:
            for i in group:
                null[i] = False
        for t in range(looks):
            active_group = min(3, (4 * t) // looks)
            for g, group in enumerate(groups):
                for i in group:
                    p[t][i] = 0.76 if g == active_group else 0.51
    else:
        raise ValueError(f"unknown scenario: {name}")

    return {
        "name": name,
        "null": tuple(null),
        "p": p,
        "arrivals": tuple(arrivals),
    }


def draw_batch(
    scenario: dict[str, Any],
    round_no: int,
    dependence: str,
    rng: random.Random,
) -> dict[int, int]:
    active = [
        i for i, arrival in enumerate(scenario["arrivals"])
        if arrival <= round_no
    ]
    p = scenario["p"][round_no]
    null = scenario["null"]

    if dependence == "independent":
        return {i: int(rng.random() < p[i]) for i in active}
    if dependence == "shared_uniform":
        u = rng.random()
        return {i: int(u < p[i]) for i in active}
    if dependence == "balanced_negative":
        null_ids = [i for i in active if null[i]]
        alt_ids = [i for i in active if not null[i]]
        y: dict[int, int] = {}
        if null_ids:
            successes = len(null_ids) // 2
            if len(null_ids) % 2:
                successes += rng.randrange(2)
            shuffled = null_ids[:]
            rng.shuffle(shuffled)
            selected = set(shuffled[:successes])
            y.update({i: int(i in selected) for i in null_ids})
        y.update({i: int(rng.random() < p[i]) for i in alt_ids})
        return y
    raise ValueError(f"unknown dependence: {dependence}")


def adjusted_path(
    observations: list[dict[int, int]],
    arrivals: tuple[int, ...],
) -> list[dict[int, int]]:
    """Exact integer LR products with A2 floor on the running maximum."""
    n = len(arrivals)
    product_num = [1] * n
    product_den = [1] * n
    max_num = [1] * n
    max_den = [1] * n
    path: list[dict[int, int]] = []

    for t, batch in enumerate(observations):
        row: dict[int, int] = {}
        for i, y in batch.items():
            product_num[i] *= 7 if y else 3
            product_den[i] *= 5
            if (
                product_num[i] * max_den[i]
                > max_num[i] * product_den[i]
            ):
                max_num[i] = product_num[i]
                max_den[i] = product_den[i]
            scaled_square = (
                max_num[i] * SCALE * SCALE // max_den[i]
            )
            root = math.isqrt(scaled_square)
            row[i] = max(0, root - SCALE)
        path.append(row)
    return path


def fdp_and_power(
    rejected: tuple[int, ...],
    active_ids: tuple[int, ...],
    null_mask: tuple[bool, ...],
    total_alternatives: int,
) -> tuple[F, F, bool]:
    if not rejected:
        return F(0), F(0), False
    false = sum(null_mask[i] for i in rejected)
    true = len(rejected) - false
    fdp = F(false, len(rejected))
    power = F(true, total_alternatives) if total_alternatives else F(0)
    return fdp, power, false > 0


def simulate_stream(
    scenario: dict[str, Any],
    dependence: str,
    seed: int,
    deltas: tuple[F, ...],
) -> dict[str, Any]:
    looks = len(scenario["p"])
    n = len(scenario["null"])
    rng = random.Random(seed)
    observations = [
        draw_batch(scenario, t, dependence, rng)
        for t in range(looks)
    ]
    x_path = adjusted_path(observations, scenario["arrivals"])
    total_alternatives = sum(not value for value in scenario["null"])

    lambdas = {delta: F(1) - delta for delta in deltas}
    weight_vectors = {
        delta: projective_weight_prefix(n, lambdas[delta])
        for delta in deltas
    }
    # Full-mass reference is the globally summable telescoping prior.
    full_weights = finite_envelope_weight_prefix(n, F(1))

    finite_delta = finite_envelope_slack(n, F(1))
    method_names = ["finite_envelope_baseline_e_bh", "finite_envelope_anchored"]
    for delta in deltas:
        tag = f"scaled_e_bh_delta_{float(delta):.2f}"
        method_names.append(tag)
        method_names.append(f"anchored_delta_{float(delta):.2f}")

    fdp_paths = {name: [] for name in method_names}
    power_paths = {name: [] for name in method_names}
    any_false = {name: False for name in method_names}
    anchor_seconds = {delta: 0.0 for delta in deltas}
    anchor_calls = {delta: 0 for delta in deltas}
    added_terminal: dict[F, int] = {delta: 0 for delta in deltas}
    baseline_terminal: dict[F, int] = {delta: 0 for delta in deltas}
    anchored_terminal: dict[F, int] = {delta: 0 for delta in deltas}
    finite_anchor_seconds_total = 0.0
    finite_anchor_calls = 0
    finite_baseline_terminal = 0
    finite_anchor_terminal = 0

    for t in range(looks):
        row = x_path[t]
        active_ids = tuple(sorted(row))
        if not active_ids:
            continue
        null_active = tuple(scenario["null"][i] for i in active_ids)

        full_scores = [
            full_weights[i] * F(row.get(i, 0), SCALE)
            for i in range(n)
        ]
        full_reject = weighted_compound_ebh(full_scores, F(1, 10))
        fdp, power, has_false = fdp_and_power(
            full_reject, tuple(range(n)), scenario["null"], total_alternatives
        )
        fdp_paths["finite_envelope_baseline_e_bh"].append(fdp)
        power_paths["finite_envelope_baseline_e_bh"].append(power)
        any_false["finite_envelope_baseline_e_bh"] |= has_false

        finite_anchor_start = time.perf_counter()
        finite_anchor = anchored_rejections(
            full_scores, F(1, 10), finite_delta
        )
        finite_anchor_seconds = time.perf_counter() - finite_anchor_start
        if not set(full_reject).issubset(finite_anchor):
            raise AssertionError("finite-envelope anchor lost a baseline discovery")
        if any(i not in active_ids for i in finite_anchor):
            raise AssertionError("finite-envelope anchor selected a dormant candidate")
        if closure_deficit(full_scores, F(1, 10), finite_anchor) > F(1, 10) * finite_delta:
            raise AssertionError("finite-envelope anchor violates closure")
        fdp, power, has_false = fdp_and_power(
            finite_anchor, tuple(range(n)), scenario["null"], total_alternatives
        )
        fdp_paths["finite_envelope_anchored"].append(fdp)
        power_paths["finite_envelope_anchored"].append(power)
        any_false["finite_envelope_anchored"] |= has_false
        finite_anchor_seconds_total += finite_anchor_seconds
        finite_anchor_calls += 1
        if t == looks - 1:
            finite_baseline_terminal = len(full_reject)
            finite_anchor_terminal = len(finite_anchor)

        for delta in deltas:
            weights = weight_vectors[delta]
            scores = [
                weights[i] * F(row.get(i, 0), SCALE)
                for i in range(n)
            ]
            base_local = weighted_compound_ebh(scores, F(1, 10))
            anchored_start = time.perf_counter()
            anchored_local = anchored_rejections(scores, F(1, 10), delta)
            anchor_seconds[delta] += time.perf_counter() - anchored_start
            anchor_calls[delta] += 1
            if not set(base_local).issubset(anchored_local):
                raise AssertionError("anchored closure lost a baseline discovery")
            if closure_deficit(scores, F(1, 10), anchored_local) > F(1, 10) * delta:
                raise AssertionError("anchored output violates the exact closure deficit")

            base_reject = tuple(base_local)
            anchor_reject = tuple(anchored_local)
            if any(i not in active_ids for i in anchor_reject):
                raise AssertionError("global-slack anchor selected a dormant candidate")

            fdp, power, has_false = fdp_and_power(
                base_reject, active_ids, scenario["null"], total_alternatives
            )
            base_name = f"scaled_e_bh_delta_{float(delta):.2f}"
            fdp_paths[base_name].append(fdp)
            power_paths[base_name].append(power)
            any_false[base_name] |= has_false

            fdp, power, has_false = fdp_and_power(
                anchor_reject, active_ids, scenario["null"], total_alternatives
            )
            anchor_name = f"anchored_delta_{float(delta):.2f}"
            fdp_paths[anchor_name].append(fdp)
            power_paths[anchor_name].append(power)
            any_false[anchor_name] |= has_false

            if t == looks - 1:
                baseline_terminal[delta] = len(base_reject)
                anchored_terminal[delta] = len(anchor_reject)
                added_terminal[delta] = len(anchor_reject) - len(base_reject)

    result = {}
    for method in method_names:
        fdp_series = fdp_paths[method]
        power_series = power_paths[method]
        result[method] = {
            "sup_fdp": float(max(fdp_series, default=F(0))),
            "terminal_fdp": float(fdp_series[-1]) if fdp_series else 0.0,
            "p_any_false": float(any_false[method]),
            "terminal_power": float(power_series[-1]) if power_series else 0.0,
        }
    result["allocation_diagnostics"] = {
        "finite_envelope": {
            "delta": float(finite_delta),
            "baseline_terminal_discoveries": finite_baseline_terminal,
            "anchored_terminal_discoveries": finite_anchor_terminal,
            "added_terminal_discoveries": finite_anchor_terminal - finite_baseline_terminal,
            "mean_anchor_solver_seconds_per_call": (
                finite_anchor_seconds_total / finite_anchor_calls
                if finite_anchor_calls else 0.0
            ),
        },
        **{
        f"{float(delta):.2f}": {
            "baseline_terminal_discoveries": baseline_terminal[delta],
            "anchored_terminal_discoveries": anchored_terminal[delta],
            "added_terminal_discoveries": added_terminal[delta],
            "mean_anchor_solver_seconds_per_call": (
                anchor_seconds[delta] / anchor_calls[delta]
                if anchor_calls[delta] else 0.0
            ),
        }
        for delta in deltas
        }
    }
    return result


def run(
    streams: int = 30,
    hypotheses: int = 24,
    looks: int = 40,
    seed: int = 10150,
    delta_values: tuple[F, ...] = DELTAS,
) -> dict[str, Any]:
    rows = []
    for scenario_index, name in enumerate(SCENARIOS):
        scenario = make_scenario(name, hypotheses, looks)
        for dependence_index, dependence in enumerate(DEPENDENCE):
            sample = []
            for replicate in range(streams):
                cell_seed = (
                    seed + scenario_index * 1_000_000
                    + dependence_index * 10_000 + replicate
                )
                sample.append(simulate_stream(
                    scenario, dependence, cell_seed, delta_values
                ))
            method_names = list(sample[0].keys())
            summaries = {}
            for method in method_names:
                if method == "allocation_diagnostics":
                    summaries[method] = {
                        key: {
                            metric: statistics.mean(
                                row[method][key][metric] for row in sample
                            )
                            for metric in sample[0][method][key]
                        }
                        for key in sample[0][method]
                    }
                    continue
                summaries[method] = {
                    metric: statistics.mean(
                        row[method][metric] for row in sample
                    )
                    for metric in sample[0][method]
                }
            rows.append({
                "scenario": name,
                "dependence": dependence,
                "seed_start": (
                    seed + scenario_index * 1_000_000
                    + dependence_index * 10_000
                ),
                "streams": streams,
                "hypotheses": hypotheses,
                "looks": looks,
                "null_hypotheses": sum(scenario["null"]),
                "alternative_hypotheses": sum(not value for value in scenario["null"]),
                "results": summaries,
            })

    hoeffding = math.sqrt(math.log(20.0) / (2.0 * streams))
    return {
        "benchmark": "slack_anchored_compound_ebh_pilot",
        "scope": (
            "Exploratory paired simulation using exact Bernoulli LR factors "
            "and downward-rounded A2 running maxima. Not a proof or confirmatory result."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "parameters": {
            "streams_per_cell": streams,
            "hypotheses": hypotheses,
            "looks": looks,
            "alpha": "1/10",
            "seed": seed,
            "base_null_probability": "1/2",
            "betting_probability": "7/10",
            "null_lr_factors": "7/5 success, 3/5 failure; conditional mean one",
            "adjuster": "floor_1e6(sqrt(running_max_LR)-1)",
            "global_weight_sequence": "gamma_i=lambda/[i(i+1)], i>=1; fixed and not renormalized",
            "finite_envelope_baseline": "gamma_i=1/[i(i+1)] for i=1..n; sum n/(n+1); fixed anchor slack 1/(n+1)",
            "delta_values": [str(delta) for delta in delta_values],
            "finite_envelope": {
                "maximum_candidates_precommitted": hypotheses,
                "weights": "gamma_i=1/[i(i+1)] for i=1..n",
                "anchor_slack": f"1/(n+1)={F(1, hypotheses+1)}",
                "projectivity": "valid only within the precommitted finite envelope; extending beyond it requires a new policy contract"
            },
            "dependence": list(DEPENDENCE),
            "arrival_policy": "deterministic predictable staggered arrivals",
        },
        "methods": [
            "finite_envelope_baseline_e_bh",
            "finite_envelope_anchored",
            "scaled_e_bh_delta_<delta>",
            "anchored_delta_<delta>",
        ],
        "one_sided_95pct_hoeffding_halfwidth_per_bounded_mean": hoeffding,
        "simultaneous_uncertainty": "not applied",
        "results": rows,
        "claim_ceiling": (
            "No SOTA/power-superiority claim. All anchored decisions retain the "
            "same-weight baseline set; statistical validity remains conditional "
            "on the declared global-filtration e-process assumptions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=30)
    parser.add_argument("--hypotheses", type=int, default=24)
    parser.add_argument("--looks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=10150)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if min(args.streams, args.hypotheses, args.looks) < 1:
        raise SystemExit("all dimensions must be positive")
    report = run(args.streams, args.hypotheses, args.looks, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
