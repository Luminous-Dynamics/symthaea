#!/usr/bin/env python3
"""Prequential recipient-allocation benchmark.

Synthetic utility only. No e-processes, null hypotheses, FDR/SupFDR estimates,
or power claims. Adaptive allocation q_t uses reward history through t-1.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from fractions import Fraction


def priors(k: int) -> dict[str, list[float]]:
    offsets = range(1, k + 1)
    telescope = [1.0 / (d * (d + 1)) for d in offsets]
    telescope.append(1.0 / (k + 1))  # exact mass beyond the finite cut
    geometric = [2.0 ** (-d) for d in offsets]
    geometric.append(2.0 ** (-k))
    uniform = [1.0 / k for _ in offsets] + [0.0]
    nearest = [1.0] + [0.0] * k
    for name, values in [("telescope", telescope), ("geometric", geometric)]:
        if abs(sum(values) - 1.0) > 1e-12:
            raise AssertionError(f"{name} prior does not conserve mass")
    return {
        "nearest_fixed": nearest,
        "telescoping_fixed": telescope,
        "geometric_fixed": geometric,
        "uniform_finite_horizon": uniform,
    }


def utility_scenarios(k: int, rounds: int, seed: int) -> dict[str, list[list[float]]]:
    rng = random.Random(seed)
    offsets = list(range(1, k + 1))
    scenarios: dict[str, list[list[float]]] = {}
    scenarios["nearest_recipient"] = [
        [math.exp(-(d - 1) / 3.0) for d in offsets] + [0.0]
        for _ in range(rounds)
    ]
    scenarios["mid_distance_peak"] = [
        [math.exp(-abs(d - 8) / 3.0) for d in offsets] + [0.0]
        for _ in range(rounds)
    ]
    scenarios["far_distance_peak"] = [
        [math.exp(-abs(d - 24) / 5.0) for d in offsets] + [0.0]
        for _ in range(rounds)
    ]
    targets = [2 if t < rounds // 4 else
               9 if t < rounds // 2 else
               19 if t < 3 * rounds // 4 else 29
               for t in range(rounds)]
    scenarios["switching_four_regimes"] = [
        [math.exp(-abs(d - target) / 2.5) for d in offsets] + [0.0]
        for target in targets
    ]
    spikes = {4, 12, 27}
    scenarios["sparse_opportunities"] = [
        [(0.9 if d in spikes else 0.05 + 0.1 * rng.random())
         for d in offsets] + [0.0]
        for _ in range(rounds)
    ]
    scenarios["diffuse_random"] = [
        [rng.random() for _ in offsets] + [0.0]
        for _ in range(rounds)
    ]
    scenarios["smooth_regime_shift"] = [
        [max(0.0, min(1.0, 0.15 + 0.8 * math.sin(0.12 * t + 0.17 * d)))
         for d in offsets] + [0.0]
        for t in range(rounds)
    ]
    return scenarios


def online_hedge(utilities: list[list[float]], prior: list[float],
                 eta: float, fixed_share: float = 0.0) -> list[list[float]]:
    """Returns q_t from past utility only; current utility is observed afterward."""
    weights = prior[:]
    allocations: list[list[float]] = []
    for utility in utilities:
        total = sum(weights)
        if total <= 0:
            raise AssertionError("Hedge weights lost all mass")
        q = [w / total for w in weights]
        allocations.append(q)
        # Reveal this round's full-information utility only after choosing q_t.
        raw = [q[i] * math.exp(eta * utility[i]) for i in range(len(q))]
        normalizer = sum(raw)
        updated = [v / normalizer for v in raw]
        weights = [
            (1.0 - fixed_share) * updated[i] + fixed_share * prior[i]
            for i in range(len(prior))
        ]
        if abs(sum(weights) - 1.0) > 1e-10:
            raise AssertionError("Hedge update failed to conserve allocation mass")
    return allocations


def score_policy(allocations: list[list[float]],
                 utilities: list[list[float]]) -> dict[str, float]:
    rounds = len(utilities)
    arms = len(allocations[0])
    total_reward = sum(
        sum(q[i] * u[i] for i in range(arms))
        for q, u in zip(allocations, utilities)
    )
    arm_rewards = [sum(u[i] for u in utilities) for i in range(arms)]
    best_fixed = max(arm_rewards)
    per_round_oracle = sum(max(u) for u in utilities)
    average_allocation = [
        sum(q[i] for q in allocations) / rounds for i in range(arms)
    ]
    entropy = -sum(p * math.log(p) for p in average_allocation if p > 0)
    concentration = sorted(average_allocation, reverse=True)
    return {
        "mean_utility_per_unit": total_reward / rounds,
        "best_fixed_recipient_utility": best_fixed / rounds,
        "static_regret": (best_fixed - total_reward) / rounds,
        "per_round_oracle_utility": per_round_oracle / rounds,
        "gap_to_per_round_oracle": (per_round_oracle - total_reward) / rounds,
        "effective_recipient_count": math.exp(entropy),
        "top1_allocation_share": concentration[0],
        "top5_allocation_share": sum(concentration[:5]),
        "off_horizon_tail_share": average_allocation[-1],
    }


def no_go_witness() -> dict:
    reserve = Fraction(1, 5)
    # Short horizon allocates the full reserve to recipient 2.
    short_q = {2: Fraction(1), 3: Fraction(0)}
    # Extended-horizon recipient history shifts it to candidate 3.
    long_q = {2: Fraction(0), 3: Fraction(1)}
    short_recipient_2 = reserve * short_q[2]
    long_recipient_2 = reserve * long_q[2]
    short_donor_outflow = reserve * sum(short_q.values())
    long_donor_outflow = reserve * sum(long_q.values())
    assert short_donor_outflow == long_donor_outflow == reserve
    assert short_recipient_2 != long_recipient_2
    return {
        "reserve": str(reserve),
        "short_horizon_route": {"2": str(short_q[2]), "3": str(short_q[3])},
        "extended_horizon_route": {"2": str(long_q[2]), "3": str(long_q[3])},
        "donor_outflow_equal": str(short_donor_outflow) == str(long_donor_outflow),
        "recipient_2_in_prefix_state_changes": True,
        "pathwise_total_budget_preserved": True,
        "strong_projectivity_fails": True,
    }


def exact_accounting_screen(traces: int = 1000, donors: int = 8,
                            rounds: int = 25) -> dict:
    """Check time-varying shares preserve donor and prefix accounting."""
    assertions = 0
    for trace in range(traces):
        rng = random.Random(31_000_019 + trace * 7919)
        gammas = [Fraction(1, (i + 1) * (i + 2)) for i in range(donors)]
        x = [Fraction(1) for _ in range(donors)]
        prior_reserve = [Fraction(0) for _ in range(donors)]
        outgoing = [Fraction(0) for _ in range(donors)]
        incoming = [Fraction(0) for _ in range(donors)]
        escrow = Fraction(0)
        for _t in range(rounds):
            # Stable donor order; every recipient index is strictly later.
            for i in range(donors):
                # Draw/commit the route before exposing this round's increment.
                weights = [Fraction(rng.randint(0, 100), 100)
                           for _j in range(i + 1, donors)]
                mass = sum(weights, Fraction(0))
                shares = ([w / mass for w in weights] if mass else weights)
                dx = Fraction(rng.randint(0, 1_000_000), 1_000_000)
                x[i] += dx
                gamma = gammas[i]
                target = min(max(gamma * x[i] - 1, Fraction(0)), gamma)
                delta_reserve = target - prior_reserve[i]
                assert delta_reserve >= 0
                assert delta_reserve <= gamma * dx
                if shares:
                    assert sum(shares, Fraction(0)) == 1
                allocated = Fraction(0)
                for j, q in zip(range(i + 1, donors), shares):
                    amount = q * delta_reserve
                    allocated += amount
                    # Immediate or escrowed credit is still a forward edge.
                    if rng.randrange(2):
                        incoming[j] += amount
                    else:
                        escrow += amount
                    assertions += 1
                assert allocated <= delta_reserve
                outgoing[i] += allocated
                prior_reserve[i] = target
                assert outgoing[i] <= gamma * min(x[i], Fraction(1))
                assert outgoing[i] <= target
                assertions += 4
            for m in range(1, donors + 1):
                assert sum(incoming[:m], Fraction(0)) <= sum(outgoing[:m], Fraction(0))
                assertions += 1
            assert sum(incoming, Fraction(0)) + escrow == sum(outgoing, Fraction(0))
            assertions += 1
    return {
        "traces": traces,
        "donors": donors,
        "rounds": rounds,
        "exact_fraction_assertions": assertions,
        "all_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipients", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--accounting-traces", type=int, default=1000)
    args = parser.parse_args()
    if min(args.recipients, args.rounds, args.accounting_traces) < 1:
        raise SystemExit("all dimensions must be positive")
    prior = priors(args.recipients)
    utilities = utility_scenarios(args.recipients, args.rounds, args.seed)
    eta = math.sqrt(8.0 * math.log(args.recipients + 1) / args.rounds)
    results: dict[str, dict] = {}
    for scenario, stream in utilities.items():
        methods = {}
        for name in ["nearest_fixed", "telescoping_fixed",
                     "geometric_fixed", "uniform_finite_horizon"]:
            methods[name] = {
                **score_policy([prior[name]] * args.rounds, stream),
                "projectivity": (
                    "non-projective" if name == "uniform_finite_horizon"
                    else "projective"
                ),
            }
        hedge = online_hedge(stream, prior["telescoping_fixed"], eta)
        methods["prior_weighted_hedge"] = {
            **score_policy(hedge, stream),
            "projectivity": "non-projective",
        }
        fixed_share = online_hedge(
            stream, prior["telescoping_fixed"], eta, fixed_share=0.05
        )
        methods["fixed_share_hedge"] = {
            **score_policy(fixed_share, stream),
            "projectivity": "non-projective",
        }
        results[scenario] = methods

    report = {
        "scope": (
            "Synthetic recipient-utility benchmark only; not e-process "
            "validity, FDR/SupFDR, or statistical power."
        ),
        "seed": args.seed,
        "recipient_offsets": args.recipients,
        "rounds": args.rounds,
        "utility_range": "[0,1]",
        "feedback_timing": (
            "q_t uses utility history through t-1; U_t is revealed after "
            "the allocation decision"
        ),
        "prior_definitions": {
            "telescoping": (
                "rho_d=1/[d(d+1)]; first k offsets plus exact outside tail "
                "mass 1/(k+1)"
            ),
            "geometric": (
                "rho_d=2^-d; first k offsets plus exact outside tail 2^-k"
            ),
            "uniform": (
                "1/k over currently visible offsets; deliberately "
                "horizon-dependent diagnostic"
            ),
        },
        "adaptive_parameters": {
            "hedge_eta": eta,
            "fixed_share_alpha": 0.05,
        },
        "metrics": {
            "static_regret": (
                "best fixed recipient cumulative utility minus policy "
                "utility, divided by rounds; can be negative"
            ),
            "dynamic_gap": (
                "per-round best recipient utility minus policy utility; "
                "unattainable oracle upper bound"
            ),
            "effective_recipient_count": (
                "exp(Shannon entropy of average allocation shares)"
            ),
        },
        "no_go_witness": no_go_witness(),
        "exact_accounting_screen": exact_accounting_screen(
            traces=args.accounting_traces
        ),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
