#!/usr/bin/env python3
"""Exact active-intervention benchmark for small noisy Boolean SCMs.

Standard-library only. This benchmark evaluates which hard intervention to run
next when the candidate structural causal models and prior are explicit. It uses
exact enumeration of binary outcomes and Bayesian posterior updates; no Monte
Carlo estimate is used for reported expected metrics. It is a controlled pilot,
not an implementation of ACDB/CausalArena and not evidence of SOTA.

Intervention semantics in v1 are deliberately narrow and explicit: hard do(X=x).
The intervened node's structural mechanism is replaced by the assigned value;
its descendants still evaluate against that value. Candidate action cost and
disturbance are deterministic synthetic annotations, not measured risk.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "noisy-scm-active-probe-v1"
STRATEGIES = ("fixed_order", "random", "cost_first", "greedy_eig", "greedy_efficiency", "exact_oracle")


@dataclass(frozen=True)
class SCM:
    identifier: int
    # parents[node] is a sorted tuple of parent node indices.
    parents: tuple[tuple[int, ...], ...]
    # Bit i in truth_tables[node] is the output for parent-assignment i.
    truth_tables: tuple[int, ...]

    def graph_key(self) -> tuple[tuple[int, ...], ...]:
        return self.parents


@dataclass(frozen=True)
class Intervention:
    identifier: int
    target: int
    value: int
    cost: int
    disturbance: int
    kind: str = "hard_do"


@dataclass(frozen=True)
class Case:
    seed: int
    node_count: int
    hypotheses: tuple[SCM, ...]
    prior_weights: tuple[int, ...]
    interventions: tuple[Intervention, ...]
    budget: int
    noise: float


@dataclass(frozen=True)
class Value:
    expected_graph_entropy_bits: float
    expected_model_entropy_bits: float
    expected_map_graph_probability: float
    expected_cost: float
    expected_disturbance: float


def entropy(weights: Iterable[float]) -> float:
    vals = [float(w) for w in weights if w > 0.0]
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in vals)


def mechanism_output(scm: SCM, node: int, assignment: tuple[int, ...]) -> int:
    parent_index = 0
    for bit_index, parent in enumerate(scm.parents[node]):
        parent_index |= (assignment[parent] & 1) << bit_index
    return (scm.truth_tables[node] >> parent_index) & 1


def parent_is_essential(table: int, parent_count: int, parent_index: int) -> bool:
    """True iff flipping this parent can change the Boolean mechanism output."""
    for assignment in range(1 << parent_count):
        if (assignment >> parent_index) & 1:
            continue
        paired = assignment | (1 << parent_index)
        if ((table >> assignment) & 1) != ((table >> paired) & 1):
            return True
    return False


def random_dag_scm(rng: random.Random, identifier: int, node_count: int, dense: bool) -> SCM:
    """Draw a labeled DAG by randomizing a topological order first."""
    order = list(range(node_count))
    rng.shuffle(order)
    parents: list[list[int]] = [[] for _ in range(node_count)]
    probability = 0.62 if dense else 0.28
    for later_index in range(1, node_count):
        child = order[later_index]
        for earlier_index in range(later_index):
            parent = order[earlier_index]
            if rng.random() < probability:
                parents[child].append(parent)
    canonical_parents = tuple(tuple(sorted(ps)) for ps in parents)
    tables: list[int] = []
    for node in range(node_count):
        k = len(canonical_parents[node])
        if k == 0:
            table = rng.randrange(2)
        else:
            # Rejection-sample functions where every declared parent is essential.
            # This avoids materializing 65,536 truth tables at the 5-node limit.
            while True:
                table = rng.getrandbits(1 << k)
                if all(parent_is_essential(table, k, p) for p in range(k)):
                    break
        tables.append(table)
    return SCM(identifier, canonical_parents, tuple(tables))


def make_case(seed: int, budget: int = 2, node_count: int = 3, hypothesis_count: int = 8,
              noise: float = 0.1) -> Case:
    if not 3 <= node_count <= 5:
        raise ValueError("node_count must be in 3..=5 for the exact pilot")
    if not 4 <= hypothesis_count <= 16:
        raise ValueError("hypothesis_count must be in 4..=16 for the exact pilot")
    if not 1 <= budget <= 4:
        raise ValueError("budget must be in 1..=4")
    if not 0.0 < noise < 0.5:
        raise ValueError("noise must be strictly between 0 and 0.5")
    rng = random.Random(seed)
    hypotheses: list[SCM] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = set()
    attempts = 0
    while len(hypotheses) < hypothesis_count:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("could not generate enough unique SCM hypotheses")
        candidate = random_dag_scm(rng, len(hypotheses), node_count, dense=(len(hypotheses) % 2 == 1))
        signature = (candidate.parents, candidate.truth_tables)
        if signature in seen:
            continue
        seen.add(signature)
        hypotheses.append(candidate)
    priors = tuple(rng.randint(1, 5) for _ in hypotheses)
    # Synthetic action annotations are independent of the hidden SCM.
    interventions = tuple(
        Intervention(identifier=2 * node + value, target=node, value=value,
                     cost=1 + rng.randrange(2), disturbance=rng.randrange(3))
        for node in range(node_count) for value in (0, 1)
    )
    return Case(seed, node_count, tuple(hypotheses), priors, interventions, budget, noise)


def outcome_likelihood(scm: SCM, intervention: Intervention, outcome: tuple[int, ...], noise: float) -> float:
    """Likelihood under independent noisy Boolean structural equations and do(X=x)."""
    if outcome[intervention.target] != intervention.value:
        return 0.0
    likelihood = 1.0
    for node in range(len(outcome)):
        if node == intervention.target:
            continue  # hard intervention replaces this node's mechanism
        predicted = mechanism_output(scm, node, outcome)
        likelihood *= (1.0 - noise) if outcome[node] == predicted else noise
    return likelihood


def posterior_outcomes(case: Case, posterior: tuple[float, ...], action: Intervention):
    """Enumerate outcome probabilities and normalized posterior after an action."""
    other_nodes = [i for i in range(case.node_count) if i != action.target]
    for other_values in itertools.product((0, 1), repeat=len(other_nodes)):
        result = [0] * case.node_count
        result[action.target] = action.value
        for node, value in zip(other_nodes, other_values):
            result[node] = value
        outcome = tuple(result)
        unnormalized = tuple(
            posterior[h] * outcome_likelihood(model, action, outcome, case.noise)
            for h, model in enumerate(case.hypotheses)
        )
        probability = sum(unnormalized)
        if probability <= 0.0:
            continue
        yield probability, tuple(value / probability for value in unnormalized), outcome


def graph_weights(case: Case, posterior: tuple[float, ...]) -> dict[tuple[tuple[int, ...], ...], float]:
    result: dict[tuple[tuple[int, ...], ...], float] = {}
    for weight, model in zip(posterior, case.hypotheses):
        key = model.graph_key()
        result[key] = result.get(key, 0.0) + weight
    return result


def posterior_metrics(case: Case, posterior: tuple[float, ...]) -> tuple[float, float, float]:
    graph_post = graph_weights(case, posterior)
    return entropy(graph_post.values()), entropy(posterior), max(graph_post.values(), default=0.0)


def expected_action_value(case: Case, posterior: tuple[float, ...], action: Intervention,
                          remaining: tuple[int, ...], budget_left: int,
                          strategy: str, random_rank: tuple[int, ...], damage_weight: float) -> Value:
    branch_values = []
    for probability, updated, _outcome in posterior_outcomes(case, posterior, action):
        child = recurse(case, updated, tuple(i for i in remaining if i != action.identifier),
                        budget_left - action.cost, strategy, random_rank, damage_weight)
        branch_values.append((probability, child))
    if not branch_values:
        raise AssertionError("intervention had no possible outcomes")
    return Value(
        sum(p * v.expected_graph_entropy_bits for p, v in branch_values),
        sum(p * v.expected_model_entropy_bits for p, v in branch_values),
        sum(p * v.expected_map_graph_probability for p, v in branch_values),
        action.cost + sum(p * v.expected_cost for p, v in branch_values),
        action.disturbance + sum(p * v.expected_disturbance for p, v in branch_values),
    )


def recurse(case: Case, posterior: tuple[float, ...], remaining: tuple[int, ...],
            budget_left: int, strategy: str, random_rank: tuple[int, ...],
            damage_weight: float) -> Value:
    if budget_left <= 0 or not remaining:
        graph_h, model_h, map_p = posterior_metrics(case, posterior)
        return Value(graph_h, model_h, map_p, 0.0, 0.0)
    affordable = [case.interventions[i] for i in remaining if case.interventions[i].cost <= budget_left]
    if not affordable:
        graph_h, model_h, map_p = posterior_metrics(case, posterior)
        return Value(graph_h, model_h, map_p, 0.0, 0.0)

    current_graph_h = entropy(graph_weights(case, posterior).values())
    gains: dict[int, tuple[float, float]] = {}
    for action in affordable:
        branches = list(posterior_outcomes(case, posterior, action))
        after_graph_h = sum(p * posterior_metrics(case, updated)[0] for p, updated, _ in branches)
        after_model_h = sum(p * posterior_metrics(case, updated)[1] for p, updated, _ in branches)
        gains[action.identifier] = (max(0.0, current_graph_h - after_graph_h),
                                    max(0.0, entropy(posterior) - after_model_h))

    if strategy == "exact_oracle":
        stop_h, stop_m, stop_p = posterior_metrics(case, posterior)
        candidates: list[tuple[tuple[float, ...], Value]] = [
            ((stop_h, 0.0, 0.0, -stop_p), Value(stop_h, stop_m, stop_p, 0.0, 0.0))
        ]
        for action in affordable:
            value = expected_action_value(case, posterior, action, remaining, budget_left,
                                          strategy, random_rank, damage_weight)
            # Exact declared objective: graph entropy, then synthetic disturbance, then cost.
            candidates.append(((value.expected_graph_entropy_bits, value.expected_disturbance,
                                value.expected_cost, -value.expected_map_graph_probability), value))
        return min(candidates, key=lambda item: item[0])[-1]

    if strategy == "fixed_order":
        chosen = min(affordable, key=lambda a: a.identifier)
    elif strategy == "random":
        chosen = min(affordable, key=lambda a: (random_rank[a.identifier], a.identifier))
    elif strategy == "cost_first":
        chosen = min(affordable, key=lambda a: (a.cost, a.disturbance, a.identifier))
    elif strategy == "greedy_eig":
        chosen = min(affordable, key=lambda a: (-gains[a.identifier][0], -gains[a.identifier][1], a.cost, a.identifier))
    elif strategy == "greedy_efficiency":
        chosen = min(affordable, key=lambda a: (
            -(gains[a.identifier][0] / (a.cost + damage_weight * a.disturbance)),
            -gains[a.identifier][1], a.disturbance, a.cost, a.identifier))
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # Fixed/random/cost-first remain simple baselines and can waste a probe.
    if strategy in ("greedy_eig", "greedy_efficiency") and gains[chosen.identifier][0] <= 1e-12:
        graph_h, model_h, map_p = posterior_metrics(case, posterior)
        return Value(graph_h, model_h, map_p, 0.0, 0.0)
    return expected_action_value(case, posterior, chosen, remaining, budget_left,
                                 strategy, random_rank, damage_weight)


def evaluate_policy(case: Case, strategy: str, damage_weight: float = 0.25) -> Value:
    total = float(sum(case.prior_weights))
    posterior = tuple(weight / total for weight in case.prior_weights)
    random_rng = random.Random(case.seed ^ 0x5A17C9E3)
    order = list(range(len(case.interventions)))
    random_rng.shuffle(order)
    rank = [0] * len(order)
    for index, action_id in enumerate(order):
        rank[action_id] = index
    return recurse(case, posterior, tuple(range(len(case.interventions))), case.budget,
                   strategy, tuple(rank), damage_weight)


def run_benchmark(instances: int, budget: int, source_sha: str, node_count: int,
                  hypothesis_count: int, noise: float) -> dict:
    rows: list[dict] = []
    for seed in range(instances):
        case = make_case(seed, budget, node_count, hypothesis_count, noise)
        values = {name: evaluate_policy(case, name) for name in STRATEGIES}
        oracle = values["exact_oracle"]
        for name, value in values.items():
            if value.expected_graph_entropy_bits + 1e-9 < oracle.expected_graph_entropy_bits:
                raise AssertionError(f"policy {name} beat exact oracle's primary objective on seed={seed}")
            rows.append({
                "seed": seed,
                "strategy": name,
                "node_count": case.node_count,
                "hypothesis_count": len(case.hypotheses),
                "distinct_graph_count": len({h.graph_key() for h in case.hypotheses}),
                "budget": case.budget,
                "noise_probability": case.noise,
                "expected_graph_entropy_bits": value.expected_graph_entropy_bits,
                "expected_model_entropy_bits": value.expected_model_entropy_bits,
                "expected_map_graph_probability": value.expected_map_graph_probability,
                "expected_intervention_cost": value.expected_cost,
                "expected_synthetic_disturbance": value.expected_disturbance,
            })
    summary = {}
    for strategy in STRATEGIES:
        chosen = [r for r in rows if r["strategy"] == strategy]
        summary[strategy] = {
            key: sum(r[key] for r in chosen) / len(chosen) for key in (
                "expected_graph_entropy_bits", "expected_model_entropy_bits",
                "expected_map_graph_probability", "expected_intervention_cost",
                "expected_synthetic_disturbance")
        }
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "generator": "seeded-labeled-DAGs; Boolean mechanisms with every declared parent functionally essential; alternating sparse/dense candidate SCMs",
        "inference": "exact enumeration over all binary outcome assignments; exact Bayesian update under known independent mechanism-noise probability",
        "intervention_semantics": "hard do(X=x): replace the target equation; preserve descendant mechanism evaluation",
        "cost_disturbance_semantics": "deterministic synthetic action annotations; not empirical operational cost or safety risk",
        "primary_objective": "minimize expected posterior entropy over distinct graph structures; exact oracle tie-breaks by disturbance, cost, then MAP graph probability",
        "instances": instances,
        "budget_requested": budget,
        "node_count": node_count,
        "hypothesis_count": hypothesis_count,
        "noise_probability": noise,
        "policy_count": len(STRATEGIES),
        "case_count": instances,
        "summary": summary,
        "rows": rows,
    }


def self_test() -> None:
    case = make_case(seed=17, budget=2, node_count=3, hypothesis_count=6, noise=0.1)
    assert len(case.interventions) == 2 * case.node_count
    assert all(action.kind == "hard_do" for action in case.interventions)
    for model in case.hypotheses:
        for node, parents in enumerate(model.parents):
            for parent in parents:
                assert parent_is_essential(model.truth_tables[node], len(parents), parents.index(parent))
        remaining = set(range(case.node_count))
        while remaining:
            ready = {n for n in remaining if not (set(model.parents[n]) & remaining)}
            assert ready, "generator produced a directed cycle"
            remaining -= ready
    prior = tuple(w / sum(case.prior_weights) for w in case.prior_weights)
    for action in case.interventions:
        outcomes = list(posterior_outcomes(case, prior, action))
        assert abs(sum(row[0] for row in outcomes) - 1.0) <= 1e-10
        for probability, posterior_after, outcome in outcomes:
            assert probability > 0.0
            assert abs(sum(posterior_after) - 1.0) <= 1e-10
            assert outcome[action.target] == action.value
    values = {name: evaluate_policy(case, name) for name in STRATEGIES}
    oracle = values["exact_oracle"]
    assert all(v.expected_graph_entropy_bits + 1e-9 >= oracle.expected_graph_entropy_bits for v in values.values())
    assert all(0.0 <= v.expected_map_graph_probability <= 1.0 for v in values.values())
    assert run_benchmark(3, 2, "self-test", 3, 6, 0.1) == run_benchmark(3, 2, "self-test", 3, 6, 0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--hypotheses", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--source-sha", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("NOISY_SCM_PROBE_SELF_TESTS_PASSED")
            return 0
        if not 1 <= args.instances <= 200:
            raise ValueError("instances must be in 1..=200 for this exact pilot")
        result = run_benchmark(args.instances, args.budget, args.source_sha,
                               args.nodes, args.hypotheses, args.noise)
        encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else:
            sys.stdout.buffer.write(encoded)
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
