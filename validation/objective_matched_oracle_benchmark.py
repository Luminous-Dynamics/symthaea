#!/usr/bin/env python3
"""Objective-matched exact finite-horizon oracle for adaptive causal probing.

A reduced exact game (two graphs x 2^3 action-kernel assignments) allows Bellman
oracles for graph entropy, kernel entropy, and joint graph/kernel entropy. Every
policy is evaluated against the oracle for the same terminal objective.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from action_subset_kernel_benchmark import WORLD, GRAPH_NAMES, VARIABLES

SCHEMA = "symthaea-objective-matched-oracle-benchmark-v1"
BASE_SOURCE_BLOB = "ce19dccec3a81f4c6a18563de88152019b966f5d"
GRAPHS = (0, 1)
R_VALUES = (0.1, 0.6)
HYPOTHESES = tuple((g, *rs) for g in GRAPHS for rs in itertools.product(range(2), repeat=3))
H0 = {"graph": 1.0, "kernel": 3.0, "joint": 4.0}
ACTIONS = tuple(VARIABLES)
OBJECTIVES = ("graph", "kernel", "joint")
HORIZONS = (1, 2, 3)
POLICIES = ("round_robin", "random_uniform", "greedy_graph", "greedy_kernel", "greedy_joint",
            "coverage_graph", "coverage_kernel", "coverage_joint", "oracle_graph", "oracle_kernel", "oracle_joint")
OBSERVATIONS = {a: tuple((s, y) for s in (1 << a, (1 << a) | (1 << ((a + 1) % 3))) for y in range(8)) for a in ACTIONS}

def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0.0)

def likelihood(hyp, action, obs):
    graph_idx, *r_idx = hyp
    subset, outcome = obs
    target = 1 << action
    pair = target | (1 << ((action + 1) % 3))
    spill = R_VALUES[r_idx[action]]
    if subset == target:
        p_subset = 1.0 - spill
    elif subset == pair:
        p_subset = spill
    else:
        return 0.0
    return p_subset * WORLD[graph_idx][subset][outcome]

def predictive(posterior, action):
    return tuple((obs, sum(w * likelihood(h, action, obs) for h, w in zip(HYPOTHESES, posterior)))
                 for obs in OBSERVATIONS[action])

def update(posterior, action, obs):
    values = tuple(w * likelihood(h, action, obs) for h, w in zip(HYPOTHESES, posterior))
    total = sum(values)
    if total <= 0.0:
        raise ValueError("cannot update posterior with zero-probability observation")
    result = tuple(v / total for v in values)
    if not math.isclose(sum(result), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("posterior does not normalize")
    return result

def marginals(posterior):
    graph = [0.0, 0.0]
    kernel = [0.0] * 8
    for hyp, p in zip(HYPOTHESES, posterior):
        g, *rs = hyp
        graph[g] += p
        k = (rs[0] << 2) | (rs[1] << 1) | rs[2]
        kernel[k] += p
    return graph, kernel

def objective_entropy(posterior, objective):
    graph, kernel = marginals(posterior)
    if objective == "graph":
        return entropy(graph)
    if objective == "kernel":
        return entropy(kernel)
    if objective == "joint":
        return entropy(posterior)
    raise ValueError(f"unknown objective {objective}")

def information_gain(posterior, action, objective):
    before = objective_entropy(posterior, objective)
    after = sum(po * objective_entropy(update(posterior, action, obs), objective)
                for obs, po in predictive(posterior, action) if po > 0.0)
    gain = before - after
    if gain < -1e-9:
        raise AssertionError(f"negative {objective} information gain: {gain}")
    return max(0.0, gain)

def oracle_action(posterior, remaining, objective, cache):
    if remaining <= 0:
        raise ValueError("oracle action requires a positive remaining horizon")
    vals = []
    for action in ACTIONS:
        value = sum(po * oracle_value(update(posterior, action, obs), remaining - 1, objective, cache)
                    for obs, po in predictive(posterior, action) if po > 0.0)
        vals.append((value, action))
    return min(vals)[1]

def oracle_value(posterior, remaining, objective, cache):
    if remaining == 0:
        return objective_entropy(posterior, objective)
    key = (tuple(posterior), remaining, objective)
    if key in cache:
        return cache[key]
    cache[key] = min(sum(po * oracle_value(update(posterior, action, obs), remaining - 1, objective, cache)
                         for obs, po in predictive(posterior, action) if po > 0.0)
                     for action in ACTIONS)
    return cache[key]

def greedy_objective_for_policy(policy):
    if policy.startswith("greedy_"):
        return policy.split("_", 1)[1]
    if policy.startswith("coverage_"):
        return policy.split("_", 1)[1]
    if policy.startswith("oracle_"):
        return policy.split("_", 1)[1]
    return None

def choose_action(policy, posterior, action_history, remaining=None, oracle_cache=None):
    if policy == "round_robin":
        return ACTIONS[len(action_history) % len(ACTIONS)]
    objective = greedy_objective_for_policy(policy)
    if policy.startswith("oracle_"):
        if remaining is None or oracle_cache is None:
            raise ValueError("oracle policy needs remaining horizon and cache")
        return oracle_action(posterior, remaining, objective, oracle_cache)
    if objective is None:
        raise ValueError(f"policy does not choose one deterministic action: {policy}")
    candidates = ACTIONS
    if policy.startswith("coverage_"):
        tried = set(action_history)
        untried = tuple(a for a in ACTIONS if a not in tried)
        if untried:
            candidates = untried
    return min(candidates, key=lambda a: (-information_gain(posterior, a, objective), a))

def policy_value(posterior, remaining, policy, action_history, score_objective, cache):
    if remaining == 0:
        return objective_entropy(posterior, score_objective)
    if policy == "random_uniform":
        return sum(policy_value_after_action(posterior, remaining, a, policy, action_history,
                                             score_objective, cache) for a in ACTIONS) / len(ACTIONS)
    action = choose_action(policy, posterior, action_history, remaining=remaining, oracle_cache=cache["oracle"])
    return policy_value_after_action(posterior, remaining, action, policy, action_history,
                                     score_objective, cache)

def policy_value_after_action(posterior, remaining, action, policy, action_history, score_objective, cache):
    total = 0.0
    for obs, po in predictive(posterior, action):
        if po > 0.0:
            total += po * policy_value(update(posterior, action, obs), remaining - 1,
                                       policy, action_history + (action,), score_objective, cache)
    return total

def run_benchmark(max_horizon=3):
    if max_horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
    cache = {"oracle": {}}
    rows = []
    for horizon in range(1, max_horizon + 1):
        oracle_by_objective = {obj: oracle_value(prior, horizon, obj, cache["oracle"]) for obj in OBJECTIVES}
        for policy in POLICIES:
            for scored_objective in OBJECTIVES:
                terminal_h = policy_value(prior, horizon, policy, (), scored_objective, cache)
                regret = terminal_h - oracle_by_objective[scored_objective]
                if regret < -1e-8:
                    raise AssertionError(f"{policy} beats matched {scored_objective} oracle by {-regret} bits")
                rows.append({"horizon": horizon, "budget": horizon, "policy": policy,
                             "policy_optimization_objective": greedy_objective_for_policy(policy),
                             "scored_objective": scored_objective,
                             "expected_terminal_entropy_bits": terminal_h,
                             "expected_information_gain_bits": H0[scored_objective] - terminal_h,
                             "objective_matched_oracle_terminal_entropy_bits": oracle_by_objective[scored_objective],
                             "regret_bits_vs_objective_matched_oracle": max(0.0, regret),
                             "hypothesis_count": len(HYPOTHESES)})
    return {"schema": SCHEMA, "status": "EXACT_OBJECTIVE_MATCHED_FINITE_HORIZON_ORACLE_LOCAL_PILOT",
            "protocol": {"graph_hypotheses": [GRAPH_NAMES[i] for i in GRAPHS],
                         "off_target_probabilities": list(R_VALUES), "joint_hypotheses": len(HYPOTHESES),
                         "prior": "uniform over two graphs and all eight three-action kernel assignments",
                         "action_set": list(ACTIONS), "telemetry": "exact realized subset and three-bit outcome",
                         "objectives": {"graph": "minimize terminal graph entropy", "kernel": "minimize terminal joint action-kernel assignment entropy", "joint": "minimize terminal joint graph/kernel entropy"},
                         "oracle": "exact Bellman recursion and posterior tuples without rounding; one exact oracle per terminal objective",
                         "policies": list(POLICIES), "horizons": list(range(1, max_horizon + 1)),
                         "limitations": ["Reduced 16-hypothesis game", "Only chain and fork graph families", "Exact subset telemetry assumed", "No external baseline has been run"]},
            "oracle_cache_entries": len(cache["oracle"]), "rows": rows}

def self_test():
    prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
    assert len(HYPOTHESES) == 16
    for action in ACTIONS:
        assert math.isclose(sum(po for _, po in predictive(prior, action)), 1.0, abs_tol=1e-12)
        for obj in OBJECTIVES:
            assert information_gain(prior, action, obj) >= 0.0
    out = run_benchmark(2)
    assert len(out["rows"]) == 2 * len(POLICIES) * len(OBJECTIVES)
    for horizon in (1, 2):
        for obj in OBJECTIVES:
            rows = [r for r in out["rows"] if r["horizon"] == horizon and r["scored_objective"] == obj]
            oracle_row = next(r for r in rows if r["policy"] == f"oracle_{obj}")
            assert abs(oracle_row["regret_bits_vs_objective_matched_oracle"]) < 1e-10
            assert all(r["regret_bits_vs_objective_matched_oracle"] >= 0 for r in rows)
    assert run_benchmark(2) == out
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("OBJECTIVE_MATCHED_ORACLE_TESTS_PASSED"); return 0
        result = run_benchmark(args.horizon)
        result["source_git_blob"] = args.source_git_blob
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else: print(raw.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError, RecursionError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__ == "__main__": raise SystemExit(main())
