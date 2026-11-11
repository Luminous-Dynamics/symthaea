#!/usr/bin/env python3
"""Exact active probe-design benchmark for small deterministic causal mechanisms.

Standard-library only. This is an experiment-selection benchmark, not a Phi,
PID, PEID, or consciousness metric. It evaluates policies on a finite hypothesis
set where probe outcomes are known exactly and intervention costs/damage are
explicit. Run python validation/active_probe_design.py --self-test for checks,
or pass --instances 40 --budget 3 --source-sha <exact-git-sha> for a reproducible
benchmark report.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "active-probe-design-v1"
# A hypothesis is a truth table over the four input states, indexed by x0 + 2*x1.
@dataclass(frozen=True)
class Hypothesis:
    identifier: int
    prior_weight: int
    outputs: tuple[int, int, int, int]

@dataclass(frozen=True)
class Probe:
    identifier: int
    state_index: int
    cost: int
    damage: int

@dataclass(frozen=True)
class Case:
    seed: int
    hypotheses: tuple[Hypothesis, ...]
    probes: tuple[Probe, ...]
    budget: int

@dataclass(frozen=True)
class Result:
    strategy: str
    expected_posterior_entropy_bits: float
    probability_exact_identification: float
    expected_cost: float
    expected_damage: float


def entropy(weights: Iterable[int]) -> float:
    values = tuple(w for w in weights if w > 0)
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((w / total) * math.log2(w / total) for w in values)


def make_case(seed: int, budget: int) -> Case:
    """Generate a deterministic instance; seed controls hypotheses, priors, and probe costs."""
    rng = random.Random(seed)
    count = rng.randint(5, 10)
    function_ids = sorted(rng.sample(range(16), count))
    hypotheses = tuple(
        Hypothesis(i, rng.randint(1, 5), tuple((i >> state) & 1 for state in range(4)))
        for i in function_ids
    )
    probes = tuple(
        Probe(i, i, rng.randint(1, 3), rng.randint(0, 6)) for i in range(4)
    )
    return Case(seed, hypotheses, probes, budget)


def split_by_probe(case: Case, posterior: tuple[int, ...], probe: Probe) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {0: [], 1: []}
    for index in posterior:
        groups[case.hypotheses[index].outputs[probe.state_index]].append(index)
    return {outcome: tuple(indices) for outcome, indices in groups.items() if indices}


def expected_information_gain(case: Case, posterior: tuple[int, ...], probe: Probe) -> float:
    weights = [case.hypotheses[i].prior_weight for i in posterior]
    before = entropy(weights)
    total = sum(weights)
    after = 0.0
    for subset in split_by_probe(case, posterior, probe).values():
        subset_weights = [case.hypotheses[i].prior_weight for i in subset]
        probability = sum(subset_weights) / total
        after += probability * entropy(subset_weights)
    return max(0.0, before - after)


def expected_information_per_cost(case: Case, posterior: tuple[int, ...], probe: Probe,
                                  damage_multiplier: float = 0.25) -> float:
    gain = expected_information_gain(case, posterior, probe)
    denom = probe.cost + damage_multiplier * probe.damage
    return gain / denom if denom > 0 else (math.inf if gain > 0 else 0.0)


def evaluate_policy(case: Case, strategy: str, damage_multiplier: float = 0.25) -> Result:
    all_hypotheses = tuple(range(len(case.hypotheses)))
    prior_entropy = entropy(h.prior_weight for h in case.hypotheses)

    @lru_cache(maxsize=None)
    def recurse(posterior: tuple[int, ...], budget_left: int, used_mask: int) -> tuple[float, float, float, float]:
        """Return expected (posterior entropy, exact-ID probability, cost, damage)."""
        weights = [case.hypotheses[i].prior_weight for i in posterior]
        if len(posterior) <= 1 or budget_left <= 0:
            return entropy(weights), float(len(posterior) == 1), 0.0, 0.0
        available = [p for p in case.probes if not (used_mask & (1 << p.identifier)) and p.cost <= budget_left]
        if not available:
            return entropy(weights), float(len(posterior) == 1), 0.0, 0.0

        # The exact oracle searches every affordable next intervention and chooses
        # the lexicographically best expected residual entropy, damage, then cost.
        if strategy == "exact_oracle":
            stop = (entropy(weights), float(len(posterior) == 1), 0.0, 0.0)
            candidates = [(stop[0], stop[3], stop[2], -1, stop)]
            for probe in available:
                out = _branch_values(case, posterior, probe, budget_left, used_mask, recurse)
                candidates.append((out[0], out[3], out[2], probe.identifier, out))
            return min(candidates, key=lambda item: item[:4])[-1]
        if strategy in ("greedy_eig", "greedy_efficiency"):
            if strategy == "greedy_eig":
                key = lambda p: (-expected_information_gain(case, posterior, p), p.cost, p.damage, p.identifier)
            else:
                key = lambda p: (-expected_information_per_cost(case, posterior, p, damage_multiplier), p.damage, p.cost, p.identifier)
            chosen = min(available, key=key)
        elif strategy == "cost_first":
            chosen = min(available, key=lambda p: (p.cost, p.damage, p.identifier))
        elif strategy == "fixed_order":
            chosen = min(available, key=lambda p: p.identifier)
        else:
            raise ValueError(f"unknown strategy {strategy}")
        if expected_information_gain(case, posterior, chosen) <= 1e-12:
            return entropy(weights), float(len(posterior) == 1), 0.0, 0.0
        return _branch_values(case, posterior, chosen, budget_left, used_mask, recurse)

    result = recurse(all_hypotheses, case.budget, 0)
    # Expected exact identification must be in [0,1], entropy in [0, prior entropy].
    if not (-1e-12 <= result[1] <= 1.0 + 1e-12):
        raise AssertionError("invalid identification probability")
    if not (-1e-12 <= result[0] <= prior_entropy + 1e-12):
        raise AssertionError("invalid posterior entropy")
    return Result(strategy, *result)


def _branch_values(case: Case, posterior: tuple[int, ...], probe: Probe,
                   budget_left: int, used_mask: int, recurse: Callable) -> tuple[float, float, float, float]:
    weights = [case.hypotheses[i].prior_weight for i in posterior]
    total = sum(weights)
    entropy_sum = exact_sum = cost_sum = damage_sum = 0.0
    for subset in split_by_probe(case, posterior, probe).values():
        probability = sum(case.hypotheses[i].prior_weight for i in subset) / total
        child = recurse(subset, budget_left - probe.cost, used_mask | (1 << probe.identifier))
        entropy_sum += probability * child[0]
        exact_sum += probability * child[1]
        cost_sum += probability * child[2]
        damage_sum += probability * child[3]
    return entropy_sum, exact_sum, cost_sum + probe.cost, damage_sum + probe.damage


def run_benchmark(instances: int, budget: int, source_sha: str) -> dict:
    rows = []
    strategies = ("fixed_order", "cost_first", "greedy_eig", "greedy_efficiency", "exact_oracle")
    for seed in range(instances):
        case = make_case(seed, budget)
        results = {name: evaluate_policy(case, name) for name in strategies}
        oracle = results["exact_oracle"]
        for name, result in results.items():
            if result.expected_posterior_entropy_bits + 1e-10 < oracle.expected_posterior_entropy_bits:
                raise AssertionError(f"policy {name} beat alleged exact oracle on seed {seed}")
            rows.append({
                "seed": seed, "hypothesis_count": len(case.hypotheses), "budget": case.budget,
                "strategy": name,
                "expected_posterior_entropy_bits": result.expected_posterior_entropy_bits,
                "probability_exact_identification": result.probability_exact_identification,
                "expected_cost": result.expected_cost, "expected_damage": result.expected_damage,
                "probe_costs": [p.cost for p in case.probes],
                "probe_damage": [p.damage for p in case.probes],
            })
    summary = {}
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        summary[strategy] = {
            "mean_posterior_entropy_bits": sum(r["expected_posterior_entropy_bits"] for r in selected) / len(selected),
            "mean_exact_identification_probability": sum(r["probability_exact_identification"] for r in selected) / len(selected),
            "mean_intervention_cost": sum(r["expected_cost"] for r in selected) / len(selected),
            "mean_intervention_damage": sum(r["expected_damage"] for r in selected) / len(selected),
        }
    return {
        "schema": SCHEMA, "source_sha": source_sha,
        "generator": "seeded-subsets-of-16-two-input-Boolean-mechanisms; integer priors; input-setting probes",
        "instances": instances, "budget_requested": budget,
        "policy_count": len(strategies), "case_count": instances,
        "objective_note": "exact_oracle minimizes expected posterior entropy, then damage, then cost; others are heuristics",
        "summary": summary, "rows": rows,
    }


def self_test() -> None:
    # A 4-bit truth table uniquely names every Boolean mechanism on two inputs.
    for mechanism in range(16):
        values = tuple((mechanism >> state) & 1 for state in range(4))
        assert len(set(values)) <= 2 and all(value in (0, 1) for value in values)
    case = make_case(19, 3)
    initial = tuple(range(len(case.hypotheses)))
    for probe in case.probes:
        gain = expected_information_gain(case, initial, probe)
        assert -1e-12 <= gain <= entropy(h.prior_weight for h in case.hypotheses) + 1e-12
    results = {name: evaluate_policy(case, name) for name in
               ("fixed_order", "cost_first", "greedy_eig", "greedy_efficiency", "exact_oracle")}
    oracle = results["exact_oracle"]
    assert all(r.expected_posterior_entropy_bits + 1e-10 >= oracle.expected_posterior_entropy_bits for r in results.values())
    assert all(0.0 <= r.probability_exact_identification <= 1.0 for r in results.values())
    # Repeatability: same seed/configuration produces the same whole output.
    assert run_benchmark(4, 3, "self-test") == run_benchmark(4, 3, "self-test")
    # No observation should have information when all hypotheses predict the same output.
    all_zero = Case(0, tuple(Hypothesis(i, 1, (0, 0, 0, 0)) for i in range(4)),
                    tuple(Probe(i, i, 1, 0) for i in range(4)), 2)
    assert all(expected_information_gain(all_zero, (0, 1, 2, 3), probe) == 0.0 for probe in all_zero.probes)
    assert evaluate_policy(all_zero, "exact_oracle").expected_cost == 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--source-sha", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("ACTIVE_PROBE_DESIGN_SELF_TESTS_PASSED")
            return 0
        if args.instances < 1 or args.instances > 1000:
            raise ValueError("instances must be in 1..=1000")
        if args.budget < 1 or args.budget > 12:
            raise ValueError("budget must be in 1..=12")
        result = run_benchmark(args.instances, args.budget, args.source_sha)
        encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else:
            sys.stdout.buffer.write(encoded)
        return 0
    except (OSError, ValueError, AssertionError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
