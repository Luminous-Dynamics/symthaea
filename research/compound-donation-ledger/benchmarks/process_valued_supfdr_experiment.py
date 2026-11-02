#!/usr/bin/env python3
"""Exploratory SupFDR simulation for process-valued compound donations.

Valid-null generators:
- independent Bernoulli likelihood-ratio factors;
- shared-uniform common shocks;
- balanced negative-dependence null coordinates.

Each null coordinate has conditional mean-one LR factors in the global
filtration. The raw LR martingale is transformed by A(x)=sqrt(x)-1 applied
to its running maximum. Donations use only each donor's own new adjusted
evidence reserve. This is a simulation, not a proof or production ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


METHODS = (
    "no_donation",
    "fixed_next",
    "telescope_fixed",
    "geometric_fixed",
    "prior_weighted_hedge",
    "fixed_share_hedge",
    "raw_supremum_invalid_control",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    p_by_time: np.ndarray  # [looks, hypotheses]
    is_null: np.ndarray  # [hypotheses]
    arrivals: np.ndarray  # zero-based first observable look


def make_scenarios(n: int, looks: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    arrivals_all = np.zeros(n, dtype=np.int64)
    null = np.ones(n, dtype=bool)
    scenarios.append(Scenario(
        "all_null", np.full((looks, n), 0.5), null.copy(),
        arrivals_all.copy()
    ))

    p = np.full((looks, n), 0.5)
    p[:, 0] = 0.80
    p[:, 1:9] = 0.62
    is_null = np.ones(n, dtype=bool)
    is_null[:9] = False
    scenarios.append(Scenario("strong_donor_many_weak", p, is_null,
                              arrivals_all.copy()))

    # One borderline recipient plus enough strong alternatives to lower the
    # self-consistent e-BH threshold: a targeted test of transfer utility.
    p = np.full((looks, n), 0.5)
    p[:, 0] = 0.80
    if n > 1:
        p[:, 1] = 0.62
    p[:, 2:min(20, n)] = 0.80
    is_null = np.ones(n, dtype=bool)
    is_null[:min(20, n)] = False
    scenarios.append(Scenario("donation_bridge_target", p, is_null,
                              arrivals_all.copy()))

    p = np.full((looks, n), 0.5)
    p[:, 2:18] = 0.65
    p[:, 0] = 0.80
    is_null = np.ones(n, dtype=bool)
    is_null[0] = False
    is_null[2:18] = False
    scenarios.append(Scenario("strong_donor_dense_weak", p, is_null,
                              arrivals_all.copy()))

    p = np.full((looks, n), 0.5)
    p[:, 0] = 0.80
    p[:, 20:min(30, n)] = 0.70
    is_null = np.ones(n, dtype=bool)
    is_null[0] = False
    is_null[20:min(30, n)] = False
    scenarios.append(Scenario("strong_donor_far_cluster", p, is_null,
                              arrivals_all.copy()))

    p = np.full((looks, n), 0.5)
    cuts = [0, looks // 4, looks // 2, 3 * looks // 4, looks]
    groups = [range(1, min(6, n)), range(6, min(11, n)),
              range(11, min(16, n)), range(16, min(21, n))]
    for g, (lo, hi) in zip(groups, zip(cuts[:-1], cuts[1:])):
        p[lo:hi, list(g)] = 0.70
    p[:, 0] = 0.80
    is_null = np.ones(n, dtype=bool)
    is_null[0] = False
    for g in groups:
        is_null[list(g)] = False
    scenarios.append(Scenario("switching_alternatives", p, is_null,
                              arrivals_all.copy()))

    # Predictable deterministic delayed activation. Index is fixed before data.
    delayed = np.minimum((np.arange(n) // 4) * max(1, looks // 20),
                         max(0, looks // 2)).astype(np.int64)
    p = np.full((looks, n), 0.5)
    p[:, 0] = 0.80
    p[:, 1:13] = 0.65
    is_null = np.ones(n, dtype=bool)
    is_null[:13] = False
    scenarios.append(Scenario("delayed_arrivals", p, is_null, delayed))
    return scenarios


def generate_observations(
    scenario: Scenario, streams: int, dependence: str, seed: int
) -> np.ndarray:
    """Return bool observations [stream, look, hypothesis]."""
    rng = np.random.default_rng(seed)
    looks, n = scenario.p_by_time.shape
    p = scenario.p_by_time
    active = np.arange(looks)[:, None] >= scenario.arrivals[None, :]
    if dependence == "independent":
        u = rng.random((streams, looks, n))
        y = u < p[None, :, :]
    elif dependence == "shared_uniform":
        u = rng.random((streams, looks, 1))
        y = u < p[None, :, :]
    elif dependence == "balanced_negative":
        y = rng.random((streams, looks, n)) < p[None, :, :]
        null_ids = np.flatnonzero(scenario.is_null)
        count = len(null_ids)
        if count:
            u0 = rng.random((streams, looks, count))
            ranks = np.argsort(np.argsort(u0, axis=2), axis=2)
            if count % 2 == 0:
                success_count = np.full((streams, looks), count // 2)
            else:
                success_count = count // 2 + rng.integers(
                    0, 2, size=(streams, looks)
                )
            y[:, :, null_ids] = ranks < success_count[:, :, None]
    else:
        raise ValueError(f"unknown dependence: {dependence}")
    y &= active[None, :, :]
    return y


def make_evidence_paths(
    observations: np.ndarray, arrivals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A2-adjusted running-max e-process and raw running-max negative control."""
    streams, looks, n = observations.shape
    m = np.ones((streams, n), dtype=np.float64)
    running_max = np.ones((streams, n), dtype=np.float64)
    adjusted = np.zeros((streams, looks, n), dtype=np.float64)
    raw_sup = np.ones((streams, looks, n), dtype=np.float64)
    for t in range(looks):
        active = t >= arrivals
        success = observations[:, t, :]
        factor = np.where(success, 1.4, 0.6)
        m *= np.where(active[None, :], factor, 1.0)
        running_max = np.maximum(running_max, m)
        raw_sup[:, t, :] = np.where(active[None, :], running_max, 1.0)
        adjusted[:, t, :] = np.where(
            active[None, :], np.sqrt(running_max) - 1.0, 0.0
        )
    return adjusted, raw_sup


def prior_for_donor(n: int, donor: int, kind: str) -> np.ndarray:
    """Future recipients followed by one compressed outside-horizon sink."""
    k = n - donor - 1
    if kind == "fixed_next":
        out = np.zeros(k + 1)
        out[0 if k else -1] = 1.0
        return out
    if kind == "telescope":
        out = np.array([1.0 / (d * (d + 1)) for d in range(1, k + 1)]
                       + [1.0 / (k + 1)])
        return out
    if kind == "geometric":
        out = np.array([2.0 ** (-d) for d in range(1, k + 1)]
                       + [2.0 ** (-k)])
        return out
    raise ValueError(kind)


def weighted_ebh(z: np.ndarray, alpha: float) -> np.ndarray:
    """Step-up e-BH on weighted compound scores gamma_i*C_i."""
    streams, n = z.shape
    order = np.argsort(-z, axis=1)
    ranked = np.take_along_axis(z, order, axis=1)
    rank = np.arange(1, n + 1, dtype=np.float64)
    passes = ranked >= (1.0 / (alpha * rank))[None, :]
    k = np.max(np.where(passes, rank[None, :], 0.0), axis=1).astype(int)
    positions = np.arange(n)[None, :]
    selected_ranked = positions < k[:, None]
    rejected = np.zeros((streams, n), dtype=bool)
    rows = np.arange(streams)[:, None]
    valid = np.broadcast_to(selected_ranked, order.shape)
    rejected[rows, order] = valid
    return rejected


def score_rejections(
    z: np.ndarray, is_null: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rejected = weighted_ebh(z, alpha)
    false = rejected & is_null[None, :]
    true = rejected & (~is_null[None, :])
    r = rejected.sum(axis=1)
    fdp = np.divide(false.sum(axis=1), r,
                    out=np.zeros(len(r), dtype=np.float64), where=r > 0)
    power = true.sum(axis=1) / max(1, int((~is_null).sum()))
    any_false = false.any(axis=1)
    return fdp, power, any_false, rejected


def evaluate_method(
    x_path: np.ndarray,
    raw_path: np.ndarray,
    scenario: Scenario,
    gamma: np.ndarray,
    method: str,
    alpha: float,
    hedge_eta: float,
    fixed_share: float,
) -> dict[str, Any]:
    streams, looks, n = x_path.shape
    incoming = np.zeros((streams, n), dtype=np.float64)
    outgoing = np.zeros((streams, n), dtype=np.float64)
    escrow = np.zeros((streams, n, n), dtype=np.float64)
    outside_escrow = np.zeros(streams, dtype=np.float64)
    previous_reserve = np.zeros((streams, n), dtype=np.float64)
    previous_x = np.zeros((streams, n), dtype=np.float64)
    previous_z = np.zeros((streams, n), dtype=np.float64)
    sup_fdp = np.zeros(streams, dtype=np.float64)
    terminal_fdp = np.zeros(streams, dtype=np.float64)
    terminal_power = np.zeros(streams, dtype=np.float64)
    any_false = np.zeros(streams, dtype=bool)
    max_invariant_error = 0.0
    invariant_checks = 0
    donor_cap_error = 0.0
    increment_cap_error = 0.0
    q_mass_error = 0.0
    concentration_numerator = np.zeros(n, dtype=np.float64)
    total_reserve = np.zeros(streams, dtype=np.float64)
    total_credited = np.zeros(streams, dtype=np.float64)

    adaptive = method in ("prior_weighted_hedge", "fixed_share_hedge")
    q_weights: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    if adaptive:
        for i in range(n):
            p = prior_for_donor(n, i, "telescope")
            priors.append(p)
            q_weights.append(np.broadcast_to(p, (streams, len(p))).copy())

    for t in range(looks):
        active = t >= scenario.arrivals
        # Pre-observation activation: previously escrowed tokens may now be credited.
        for j in np.flatnonzero(scenario.arrivals == t):
            received = escrow[:, :j, j].sum(axis=1)
            incoming[:, j] += received
            total_credited += received
            escrow[:, :j, j] = 0.0

        x = x_path[:, t, :]
        dx = np.maximum(x - previous_x, 0.0)
        old_reserve = previous_reserve.copy()
        if method == "raw_supremum_invalid_control":
            z = gamma[None, :] * raw_path[:, t, :]
        else:
            if method != "no_donation":
                reserve = np.minimum(
                    np.maximum(gamma[None, :] * x - 1.0, 0.0),
                    gamma[None, :]
                )
                dreserve = np.maximum(reserve - old_reserve, 0.0)
                for i in range(n):
                    k = n - i - 1
                    if adaptive:
                        q = q_weights[i]
                    elif method == "fixed_next":
                        q = prior_for_donor(n, i, "fixed_next")[None, :]
                    elif method == "telescope_fixed":
                        q = prior_for_donor(n, i, "telescope")[None, :]
                    elif method == "geometric_fixed":
                        q = prior_for_donor(n, i, "geometric")[None, :]
                    else:
                        raise ValueError(method)
                    qsum = q.sum(axis=1)
                    q_mass_error = max(q_mass_error,
                                       float(np.max(np.maximum(qsum - 1.0, 0.0))))
                    for offset in range(1, k + 1):
                        amount = dreserve[:, i] * q[:, offset - 1]
                        j = i + offset
                        if active[j]:
                            incoming[:, j] += amount
                            total_credited += amount
                        else:
                            escrow[:, i, j] += amount
                    tail_amount = dreserve[:, i] * q[:, -1]
                    outside_escrow += tail_amount
                    outgoing[:, i] += dreserve[:, i]
                    total_reserve += dreserve[:, i]
                    if adaptive:
                        # q_t was selected before the current observation.
                        # Reveal bounded recipient evidence increments only now.
                        rewards = np.zeros_like(q)
                        if k:
                            rewards[:, :k] = np.minimum(dx[:, i + 1:i + 1 + k], 1.0)
                        raw_weights = q * np.exp(hedge_eta * rewards)
                        posterior = raw_weights / np.maximum(
                            raw_weights.sum(axis=1, keepdims=True), 1e-300
                        )
                        share = fixed_share if method == "fixed_share_hedge" else 0.0
                        q_weights[i] = (
                            (1.0 - share) * posterior
                            + share * priors[i][None, :]
                        )
                previous_reserve = reserve

            z = gamma[None, :] * x + incoming - outgoing

        # Candidate scores must be nonnegative and pathwise nondecreasing.
        monotone_error = float(np.max(np.maximum(previous_z - z, 0.0)))
        negative_error = float(np.max(np.maximum(-z, 0.0)))
        max_invariant_error = max(max_invariant_error, monotone_error,
                                  negative_error)
        if monotone_error > 2e-8 or negative_error > 2e-8:
            raise AssertionError(
                f"score monotonicity/nonnegativity failed for {method}: "
                f"{monotone_error}, {negative_error}"
            )

        if method != "raw_supremum_invalid_control":
            cap = gamma[None, :] * np.minimum(x, 1.0)
            donor_cap_error = max(
                donor_cap_error,
                float(np.max(np.maximum(outgoing - cap, 0.0)))
            )
            increment_cap_error = max(
                increment_cap_error,
                float(np.max(np.maximum(
                    (outgoing - old_reserve)
                    - gamma[None, :] * dx, 0.0
                )))
            )
            prefix_balance = np.cumsum(incoming - outgoing, axis=1)
            prefix_error = float(np.max(np.maximum(prefix_balance, 0.0)))
            conservation_error = float(np.max(np.abs(
                incoming.sum(axis=1) + escrow.sum(axis=(1, 2))
                + outside_escrow - outgoing.sum(axis=1)
            )))
            max_invariant_error = max(max_invariant_error, prefix_error,
                                      conservation_error)
            if prefix_error > 2e-8 or conservation_error > 2e-8:
                raise AssertionError(
                    f"prefix/conservation invariant failed: {prefix_error}, "
                    f"{conservation_error}"
                )
            invariant_checks += 4 * n + 2
        else:
            invariant_checks += 1

        fdp, power, any_f, rejected = score_rejections(
            z, scenario.is_null, alpha
        )
        sup_fdp = np.maximum(sup_fdp, fdp)
        terminal_fdp = fdp
        terminal_power = power
        any_false |= any_f
        previous_x = x.copy()
        previous_z = z.copy()
        if method != "raw_supremum_invalid_control":
            concentration_numerator += incoming.sum(axis=0)

    return {
        "supfdr_estimate": float(np.mean(sup_fdp)),
        "terminal_fdr_estimate": float(np.mean(terminal_fdp)),
        "p_any_false_discovery": float(np.mean(any_false)),
        "terminal_power": float(np.mean(terminal_power)),
        "mean_terminal_discoveries": float(np.mean(
            weighted_ebh(
                gamma[None, :] * x_path[:, -1, :] + incoming - outgoing,
                alpha
            ).sum(axis=1)
        )) if method != "raw_supremum_invalid_control" else None,
        "mean_reserve": float(np.mean(total_reserve)),
        "mean_credited": float(np.mean(total_credited)),
        "mean_outside_escrow": float(np.mean(outside_escrow)),
        "recipient_incoming_total": float(concentration_numerator.sum()),
        "max_invariant_error": max_invariant_error,
        "max_donor_cap_error": donor_cap_error,
        "max_own_increment_error": increment_cap_error,
        "max_share_mass_excess": q_mass_error,
        "invariant_checks": invariant_checks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    n, looks, streams, alpha = (
        args.hypotheses, args.looks, args.streams, args.alpha
    )
    gamma = np.array([1.0 / ((i + 1) * (i + 2)) for i in range(n)])
    total_gamma = float(gamma.sum())
    scenario_specs = make_scenarios(n, looks)
    dependence_modes = (
        ["independent", "shared_uniform", "balanced_negative"]
        if args.dependence == "all" else [args.dependence]
    )
    all_results: list[dict[str, Any]] = []
    stream_id = 0
    for scenario in scenario_specs:
        for dependence in dependence_modes:
            seed = args.seed + stream_id * 100_003
            observations = generate_observations(
                scenario, streams, dependence, seed
            )
            x_path, raw_path = make_evidence_paths(
                observations, scenario.arrivals
            )
            cell_rows = {}
            for method in METHODS:
                cell_rows[method] = evaluate_method(
                    x_path, raw_path, scenario, gamma, method, alpha,
                    args.hedge_eta, args.fixed_share
                )
            # Analytic null factor mean is exactly one for every coordinate.
            null_factor_mean = 0.5 * 1.4 + 0.5 * 0.6
            assert abs(null_factor_mean - 1.0) < 1e-15
            null_ids = np.flatnonzero(scenario.is_null)
            null_adjusted_means = (
                x_path[:, :, null_ids].mean(axis=(0, 2)).tolist()
                if len(null_ids) else []
            )
            all_results.append({
                "scenario": scenario.name,
                "dependence": dependence,
                "seed": seed,
                "streams": streams,
                "hypotheses": n,
                "looks": looks,
                "null_hypotheses": int(scenario.is_null.sum()),
                "alternative_hypotheses": int((~scenario.is_null).sum()),
                "gamma_mass": total_gamma,
                "analytic_null_lr_factor_mean": null_factor_mean,
                "empirical_null_adjusted_evidence_mean_terminal": (
                    float(np.mean(x_path[:, -1, null_ids]))
                    if len(null_ids) else None
                ),
                "empirical_max_look_null_adjusted_evidence_mean": (
                    float(np.max(null_adjusted_means)) if null_adjusted_means else None
                ),
                "results": cell_rows,
            })
            stream_id += 1

    hoeffding_halfwidth = math.sqrt(
        math.log(1.0 / 0.05) / (2.0 * streams)
    )
    return {
        "benchmark": "process_valued_compound_donation_supfdr_exploratory",
        "version": 1,
        "scope": (
            "Exploratory Monte Carlo with conditional-mean-one null LR factors, "
            "A2-adjusted running maxima, source-only capped reserves and "
            "forward-only donation. Not a proof or confirmatory result."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "streams_per_cell": streams,
            "hypotheses": n,
            "looks": looks,
            "alpha": alpha,
            "seed": args.seed,
            "gamma_sequence": "gamma_i=1/[(i+1)(i+2)]",
            "gamma_mass": total_gamma,
            "null_lr_factor": "1.4 on success, 0.6 on failure; conditional mean 1",
            "adjuster": "A(x)=sqrt(x)-1 applied to running max of LR martingale",
            "alternative_lr_design": "fixed p1=0.7 against p0=0.5",
            "hedge_eta": args.hedge_eta,
            "fixed_share": args.fixed_share,
            "dependence": args.dependence,
        },
        "estimands": {
            "supfdr": "mean_stream max_t FDP_t",
            "terminal_fdr": "mean_stream FDP at final look",
            "p_any_false": "probability of at least one false rejection at any look",
            "terminal_power": "mean fraction of nonnull hypotheses rejected at final look",
        },
        "uncertainty": {
            "nominal_interval": "one-sided 95% Hoeffding bound for each [0,1] mean",
            "halfwidth": hoeffding_halfwidth,
            "simultaneous_adjustment": "not applied; exploratory grid only",
        },
        "methods": list(METHODS),
        "results": all_results,
        "claim_ceiling": (
            "No SOTA/power superiority claim. The raw running maximum is an "
            "invalid calibration negative control. Statistical validity follows "
            "only from the stated conditional-null, filtration, adjuster, "
            "weight, reserve, forward-flow and self-consistency assumptions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=1000)
    parser.add_argument("--hypotheses", type=int, default=32)
    parser.add_argument("--looks", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=10033)
    parser.add_argument("--dependence", choices=[
        "all", "independent", "shared_uniform", "balanced_negative"
    ], default="all")
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--fixed-share", type=float, default=0.05)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if min(args.streams, args.hypotheses, args.looks) < 1:
        raise SystemExit("streams, hypotheses and looks must be positive")
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
