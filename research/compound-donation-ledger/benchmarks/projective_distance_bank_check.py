#!/usr/bin/env python3
"""Exact-rational structural check for a projective distance-prior reserve bank.

This checks deterministic algebra and ledger inequalities, not e-process
validity, statistical FDR/SupFDR, Rust performance, or power superiority.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from fractions import Fraction


def rho_telescope(d: int) -> Fraction:
    return Fraction(1, d * (d + 1))


def rho_geometric(d: int) -> Fraction:
    return Fraction(1, 2**d)


def reserve(lam: Fraction, x: Fraction) -> Fraction:
    return min(max(lam * x - 1, Fraction(0)), lam)


def finite_bank(gamma: Fraction, x: Fraction, max_distance: int,
                prior: str = "telescope") -> tuple[Fraction, list[Fraction]]:
    weights = []
    for d in range(1, max_distance + 1):
        rho = rho_telescope(d) if prior == "telescope" else rho_geometric(d)
        weights.append(gamma * rho)
    return sum((reserve(lam, x) for lam in weights), Fraction(0)), weights


def property_screen(traces: int, universe: int, rounds: int,
                    max_distance: int) -> dict:
    checks = 0
    max_active = 0
    min_unallocated_mass = Fraction(1)
    stress_gammas = [Fraction(1, 10_000), Fraction(1, 100), Fraction(1, 10), Fraction(1)]
    stress_levels = [1, 2, 10, 100, 1_000, 10_000, 1_000_000]
    for gamma in stress_gammas:
        for level in stress_levels:
            x = Fraction(level)
            values = [reserve(gamma * rho_telescope(d), x)
                      for d in range(1, max_distance + 1)]
            total = sum(values, Fraction(0))
            assert total <= gamma * min(x, Fraction(1))
            max_active = max(max_active, sum(v > 0 for v in values))
            checks += 1
    for k in range(1, max_distance + 1):
        assert sum((rho_telescope(d) for d in range(1, k + 1)), Fraction(0)) == Fraction(k, k + 1)
        checks += 1
    for trace in range(traces):
        rng = random.Random(20261102 + trace * 101)
        gammas = [Fraction(rng.randint(1, 1000), 10_000) for _ in range(universe)]
        for donor, gamma in enumerate(gammas, start=1):
            max_d = min(max_distance, universe - donor)
            if max_d == 0:
                continue
            prior_mass = sum((rho_telescope(d) for d in range(1, max_d + 1)), Fraction(0))
            min_unallocated_mass = min(min_unallocated_mass, 1 - prior_mass)
            x = Fraction(1)
            previous = [Fraction(0)] * max_d
            for _look in range(rounds):
                dx = Fraction(rng.randint(0, 2_000_000), 1_000_000)
                x += dx
                current = []
                total_increment = Fraction(0)
                active = 0
                for d in range(1, max_d + 1):
                    lam = gamma * rho_telescope(d)
                    value = reserve(lam, x)
                    delta = value - previous[d-1]
                    assert delta >= 0
                    assert delta <= lam * dx
                    assert value <= lam * min(x, Fraction(1))
                    total_increment += delta
                    current.append(value)
                    if value > 0:
                        active += 1
                    checks += 3
                total = sum(current, Fraction(0))
                assert total <= gamma * min(x, Fraction(1))
                assert total_increment <= gamma * dx
                assert total >= sum(previous, Fraction(0))
                max_active = max(max_active, active)
                previous = current
                checks += 3
    return {
        "traces": traces,
        "global_candidate_universe": universe,
        "rounds_per_donor": rounds,
        "max_distance": max_distance,
        "exact_property_assertions": checks,
        "max_positive_tranches_on_any_look": max_active,
        "minimum_unallocated_prior_mass_in_finite_universe": str(min_unallocated_mass),
        "stress_gammas": ["1/10000", "1/100", "1/10", "1"],
        "stress_evidence_levels": [1, 2, 10, 100, 1000, 10000, 1000000],
    }


def full_reserve_route_checks(traces: int, universe: int, max_distance: int) -> dict:
    checks = 0
    max_exact_cut_error = Fraction(0)
    for trace in range(traces):
        rng = random.Random(90_071 + trace * 313)
        gammas = [Fraction(rng.randint(1, 10_000), 10_000) for _ in range(universe)]
        for donor in range(1, universe):
            gamma = gammas[donor - 1]
            dx = Fraction(rng.randint(1, 10_000_000), 100_000)
            x = Fraction(1) + dx
            total = reserve(gamma, x)
            previous = reserve(gamma, x - dx)
            delta_total = total - previous
            assert delta_total <= gamma * dx
            k = min(max_distance, universe - donor)
            visible_mass = sum((rho_telescope(d) for d in range(1, k + 1)), Fraction(0))
            tail_mass = 1 - visible_mass
            shares = [rho_telescope(d) * total for d in range(1, k + 1)]
            tail = tail_mass * total
            assert sum(shares, Fraction(0)) + tail == total
            for d, amount in enumerate(shares, start=1):
                assert amount <= rho_telescope(d) * gamma * min(x, Fraction(1))
                assert rho_telescope(d) * delta_total <= rho_telescope(d) * gamma * dx
                checks += 2
            for m in range(donor, universe):
                for n in range(m, universe):
                    a = m - donor + 1
                    b = n - donor
                    outside_short = total * Fraction(1, a)
                    newly_visible_external = sum(
                        (rho_telescope(d) * total for d in range(a, b + 1)),
                        Fraction(0)
                    ) if b >= a else Fraction(0)
                    outside_long_tail = total * Fraction(1, b + 1)
                    outside_long_cut = newly_visible_external + outside_long_tail
                    assert outside_short == outside_long_cut
                    max_exact_cut_error = max(max_exact_cut_error, abs(outside_short-outside_long_cut))
                    checks += 1
    return {
        "exact_split_and_cut_assertions": checks,
        "max_exact_cut_projection_error": str(max_exact_cut_error),
        "tail_representation": "for distances d>K, aggregate mass is total_reserve/(K+1) when K visible offsets are retained",
    }


def prefix_flow_checks(traces: int, universe: int) -> dict:
    checks = 0
    for trace in range(traces):
        rng = random.Random(700_001 + trace * 919)
        gammas = [Fraction(1, i * (i + 1)) for i in range(1, universe + 1)]
        evidence = [Fraction(rng.randint(1, 100_000), 1_000) for _ in range(universe)]
        active = [bool(rng.getrandbits(1)) for _ in range(universe)]
        incoming = [Fraction(0) for _ in range(universe)]
        outgoing = [Fraction(0) for _ in range(universe)]
        escrow = Fraction(0)
        for i in range(universe):
            gamma = gammas[i]
            reserve_total = reserve(gamma, evidence[i])
            outgoing[i] = reserve_total
            visible = Fraction(0)
            for d in range(1, universe - i):
                share = rho_telescope(d) * reserve_total
                visible += share
                j = i + d
                if active[j]:
                    incoming[j] += share
                else:
                    escrow += share
                checks += 1
            tail = reserve_total - visible
            assert tail >= 0
            escrow += tail
            checks += 1
        assert sum(incoming, Fraction(0)) + escrow == sum(outgoing, Fraction(0))
        for m in range(1, universe + 1):
            assert sum(incoming[:m], Fraction(0)) <= sum(outgoing[:m], Fraction(0))
            checks += 1
    return {
        "traces": traces,
        "prefix_and_conservation_assertions": checks,
        "donor_weights": "gamma_i=1/[i(i+1)]; finite prefix sum < 1",
        "activation": "random dormant/active destinations; dormant shares remain escrow"
    }


def comparison_table(gamma: Fraction, levels: list[int], max_distance: int) -> list[dict]:
    rows = []
    for level in levels:
        x = Fraction(level)
        single = reserve(gamma, x)
        telescope, tel_weights = finite_bank(gamma, x, max_distance, "telescope")
        geometric, geo_weights = finite_bank(gamma, x, max_distance, "geometric")
        rows.append({
            "evidence_x": level,
            "single_next_recipient_reserve": float(single),
            "telescope_bank_total_reserve": float(telescope),
            "geometric_bank_total_reserve": float(geometric),
            "telescope_bank_active_destinations": sum(1 for lam in tel_weights if reserve(lam, x) > 0),
            "geometric_bank_active_destinations": sum(1 for lam in geo_weights if reserve(lam, x) > 0),
            "telescope_first_recipient_reserve": float(reserve(tel_weights[0], x)),
            "geometric_first_recipient_reserve": float(reserve(geo_weights[0], x)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=100)
    parser.add_argument("--universe", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max-distance", type=int, default=15)
    args = parser.parse_args()
    if min(args.traces, args.universe, args.rounds, args.max_distance) < 1:
        raise SystemExit("all arguments must be positive")
    prop = property_screen(args.traces, args.universe, args.rounds, args.max_distance)
    full_route = full_reserve_route_checks(args.traces, args.universe, args.max_distance)
    prefix_flow = prefix_flow_checks(args.traces, args.universe)
    telescope_sq_constant = math.pi**2 / 3 - 3
    geometric_sq_constant = 1 / 3
    report = {
        "scope": "exact-rational pathwise reserve/portfolio screen; not e-process or FDR/SupFDR validation",
        "prior": "rho_d=1/(d(d+1)); unallocated tail mass is retained, never renormalized by active horizon",
        "reserve_formula": "D_{i,d}(x)=min((gamma_i*rho_d*x-1)_+, gamma_i*rho_d)",
        "pathwise_bounds": [
            "sum_d D_{i,d}(x) <= gamma_i*min(x,1)",
            "sum_d Delta D_{i,d} <= gamma_i*Delta x",
            "each destination is the fixed global position i+d"
        ],
        "null_debit_upper_bound": {
            "telescope_sum_lambda_squared_factor": telescope_sq_constant,
            "geometric_sum_lambda_squared_factor": geometric_sq_constant,
            "identity": "sum_{d>=1} 1/[d^2(d+1)^2] = pi^2/3 - 3",
            "bound": "E[total null withdrawal] <= gamma_i^2*(pi^2/3-3) for the infinite telescope threshold-sliced bank",
            "single_slice_coarse_bound": "E[D_gamma] <= gamma_i*log(1+gamma_i)",
            "qualification": "Ville tail-integral bound; not an empirical FDR or power claim"
        },
        "property_screen": prop,
        "full_reserve_route_screen": full_route,
        "prefix_flow_screen": prefix_flow,
        "reserve_comparison_gamma_0_2": comparison_table(Fraction(1, 5), [1, 5, 10, 20, 50, 100, 1000], args.max_distance),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
