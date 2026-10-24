#!/usr/bin/env python3
"""Exact fixed-prior minimax adaptive policy via Pareto risk frontiers.

Each candidate true-world prior is fixed for the whole episode. The learner sees
observations but not the true-prior identity. A finite-horizon policy tree is scored
as a vector of expected terminal cross-entropies, one component per candidate prior;
Pareto-dominated risk vectors are pruned. The selected policy minimizes worst-case
risk across the original fixed scenarios, with mean risk and stable ordering as ties.

This avoids the branchwise scenario switching of rectangular robust Bellman planning.
The exact frontier can grow rapidly; horizon 1-2 is the primary supported scope.
Research pilot only: 16 hypotheses, chain/fork, exact realized-subset telemetry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from objective_matched_oracle_benchmark import ACTIONS, HYPOTHESES, OBJECTIVES, predictive, update
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from prior_misspecification_benchmark import terminal_cross_entropy, terminal_brier_risk
from robust_ambiguity_policy_benchmark import robust_action, update_scenarios

SCHEMA = "symthaea-fixed-prior-minimax-frontier-v1"
BASE_SOURCE_BLOBS = {
    "objective_matched_oracle_benchmark.py": "f101346c2f37cfad4e029ae280e9a002548f309c",
    "prior_misspecification_benchmark.py": "541c31521e1b9e49dfb84c0fa265e877ca9ea632",
}
POLICIES = ("fixed_prior_minimax_frontier", "robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform")
TOL = 1e-12


def dominates(a, b, tol=TOL):
    no_worse = all(x <= y + tol for x, y in zip(a, b))
    strictly = any(x < y - tol for x, y in zip(a, b))
    return no_worse and strictly


def prune_frontier(candidates):
    """Keep deterministic representatives of nondominated risk vectors."""
    ordered = sorted(candidates, key=lambda item: (max(item[0]), sum(item[0]), item[0]))
    kept = []
    for candidate in ordered:
        risk = candidate[0]
        if any(dominates(existing[0], risk) or all(abs(x-y) <= TOL for x, y in zip(existing[0], risk))
               for existing in kept):
            continue
        kept = [existing for existing in kept if not dominates(risk, existing[0])]
        kept.append(candidate)
    return kept


def make_frontier_solver(objective, candidate_prior_names):
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if not candidate_prior_names or any(n not in PRIOR_NAMES for n in candidate_prior_names):
        raise ValueError("candidate prior set must contain known prior names")
    initial_scenarios = tuple(tuple(build_prior(n)) for n in candidate_prior_names)
    frontier_peak = {"max": 0, "states": 0}

    @lru_cache(maxsize=None)
    def frontier(learner_posterior, scenarios, remaining):
        frontier_peak["states"] += 1
        if remaining == 0:
            risk = tuple(terminal_cross_entropy(q, learner_posterior, objective) for q in scenarios)
            result = [(risk, None)]
            frontier_peak["max"] = max(frontier_peak["max"], len(result))
            return tuple(result)

        all_candidates = []
        for action in ACTIONS:
            predictive_by_scenario = [dict(predictive(q, action)) for q in scenarios]
            observations = tuple(obs for obs, _ in predictive(scenarios[0], action))
            partial = [((0.0,) * len(scenarios), ())]
            for obs in observations:
                probabilities = tuple(pred[obs] for pred in predictive_by_scenario)
                if not any(prob > 0.0 for prob in probabilities):
                    continue
                next_scenarios = tuple(
                    update(q, action, obs) if prob > 0.0 else q
                    for q, prob in zip(scenarios, probabilities)
                )
                learner_next = update(learner_posterior, action, obs)
                child_frontier = frontier(learner_next, next_scenarios, remaining - 1)
                combined = []
                for partial_risk, branches in partial:
                    for child_risk, child_tree in child_frontier:
                        risk = tuple(x + prob * y for x, prob, y in zip(partial_risk, probabilities, child_risk))
                        combined.append((risk, branches + ((obs, child_tree),)))
                partial = prune_frontier(combined)
                frontier_peak["max"] = max(frontier_peak["max"], len(partial))
            for risk, branches in partial:
                all_candidates.append((risk, (action, branches)))
        result = tuple(prune_frontier(all_candidates))
        frontier_peak["max"] = max(frontier_peak["max"], len(result))
        return result

    def best_policy(learner_posterior, horizon):
        if horizon not in (1, 2, 3):
            raise ValueError("horizon must be 1, 2, or 3")
        options = frontier(tuple(learner_posterior), initial_scenarios, horizon)
        return min(options, key=lambda item: (max(item[0]), sum(item[0]) / len(item[0]), item[0]))

    return initial_scenarios, frontier, best_policy, frontier_peak


def evaluate_fixed_policy_tree(policy_tree, objective, true_prior_name, learner_prior_name, horizon):
    """Re-evaluate a selected tree under a specific fixed true-prior scenario."""
    q0, p0 = tuple(build_prior(true_prior_name)), tuple(build_prior(learner_prior_name))
    if horizon == 0:
        return terminal_cross_entropy(q0, p0, objective)
    action, branches = policy_tree
    branch_map = dict(branches)
    value = 0.0
    for obs, prob in predictive(q0, action):
        if prob <= 0.0:
            continue
        q_next, p_next = update(q0, action, obs), update(p0, action, obs)
        child = branch_map.get(obs)
        if horizon - 1 == 0 or child is None:
            value += prob * terminal_cross_entropy(q_next, p_next, objective)
        else:
            value += prob * evaluate_tree_from_posteriors(child, q_next, p_next, objective, horizon - 1)
    return value


def evaluate_tree_from_posteriors(policy_tree, q_true, learner_posterior, objective, remaining):
    if remaining == 0 or policy_tree is None:
        return terminal_cross_entropy(q_true, learner_posterior, objective)
    action, branches = policy_tree
    branch_map = dict(branches)
    value = 0.0
    for obs, prob in predictive(q_true, action):
        if prob <= 0.0:
            continue
        q_next, p_next = update(q_true, action, obs), update(learner_posterior, action, obs)
        child = branch_map.get(obs)
        value += prob * evaluate_tree_from_posteriors(child, q_next, p_next, objective, remaining - 1)
    return value


def evaluate_baseline(policy, objective, true_prior_name, learner_prior_name, horizon, candidate_prior_names=PRIOR_NAMES):
    q0, p0 = tuple(build_prior(true_prior_name)), tuple(build_prior(learner_prior_name))
    scenario0 = tuple((n, tuple(build_prior(n))) for n in candidate_prior_names)
    oracle_cache = {}

    @lru_cache(maxsize=None)
    def rec(q_true, p, scenarios, remaining, history):
        if remaining == 0:
            return terminal_cross_entropy(q_true, p, objective)
        if policy == "robust_minimax_one_step":
            actions = ((robust_action(scenarios, p, objective), 1.0),)
        elif policy == "greedy_joint":
            from objective_matched_oracle_benchmark import choose_action
            actions = ((choose_action("greedy_joint", p, history), 1.0),)
        elif policy == "round_robin":
            actions = ((ACTIONS[len(history) % len(ACTIONS)], 1.0),)
        elif policy == "random_uniform":
            actions = tuple((a, 1.0/len(ACTIONS)) for a in ACTIONS)
        else:
            raise ValueError(f"unknown baseline: {policy}")
        total = 0.0
        for action, pa in actions:
            for obs, po in predictive(q_true, action):
                if po > 0.0:
                    total += pa * po * rec(update(q_true, action, obs), update(p, action, obs),
                                           update_scenarios(scenarios, action, obs), remaining-1, history+(action,))
        return total
    return rec(q0, p0, scenario0, horizon, ())


def run_benchmark(max_horizon=2, candidate_prior_names=PRIOR_NAMES,
                  learner_prior_names=PRIOR_NAMES, objectives=OBJECTIVES):
    if max_horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    if not candidate_prior_names or any(n not in PRIOR_NAMES for n in candidate_prior_names):
        raise ValueError("unknown candidate prior list")
    if not learner_prior_names or any(n not in PRIOR_NAMES for n in learner_prior_names):
        raise ValueError("unknown learner prior list")
    if not objectives or any(o not in OBJECTIVES for o in objectives):
        raise ValueError("unknown objective list")
    rows, solvers = [], {}
    for objective in objectives:
        solver = make_frontier_solver(objective, candidate_prior_names)
        solvers[objective] = solver
        scenarios, frontier, best_policy, peak = solver
        for learner_name in learner_prior_names:
            p = tuple(build_prior(learner_name))
            for horizon in range(1, max_horizon + 1):
                vector, tree = best_policy(p, horizon)
                for i, true_name in enumerate(candidate_prior_names):
                    score = evaluate_tree_from_posteriors(tree, tuple(build_prior(true_name)), p, objective, horizon)
                    if abs(score - vector[i]) > 1e-8:
                        raise AssertionError(f"frontier vector mismatch {score} vs {vector[i]}")
                    rows.append({
                        "policy": "fixed_prior_minimax_frontier", "objective": objective,
                        "true_prior": true_name, "learner_prior": learner_name,
                        "candidate_prior_names": list(candidate_prior_names), "horizon": horizon,
                        "worst_case_cross_entropy_bits": max(vector),
                        "selected_policy_cross_entropy_bits": score,
                        "risk_vector": list(vector), "frontier_size": len(frontier(p, scenarios, horizon)),
                        "frontier_peak": peak["max"], "frontier_states": peak["states"],
                        "well_specified": true_name == learner_name,
                    })
                for baseline in ("robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform"):
                    for true_name in candidate_prior_names:
                        score = evaluate_baseline(baseline, objective, true_name, learner_name, horizon, candidate_prior_names)
                        rows.append({
                            "policy": baseline, "objective": objective, "true_prior": true_name,
                            "learner_prior": learner_name, "candidate_prior_names": list(candidate_prior_names),
                            "horizon": horizon, "selected_policy_cross_entropy_bits": score,
                            "well_specified": true_name == learner_name,
                        })
    return {
        "schema": SCHEMA,
        "status": "EXACT_FIXED_PRIOR_MINIMAX_FRONTIER_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "base_source_blobs": BASE_SOURCE_BLOBS,
        "protocol": {
            "hypotheses": len(HYPOTHESES), "candidate_prior_names": list(candidate_prior_names),
            "learner_prior_names": list(learner_prior_names), "objectives": list(objectives),
            "horizons": list(range(1, max_horizon+1)),
            "uncertainty_semantics": "candidate true-prior identity is fixed for a full episode; policy branches only on observable histories",
            "algorithm": "Pareto frontier of scenario-wise expected terminal cross-entropies; combine child-policy frontiers, prune dominated risk vectors, minimize maximum root risk",
            "score": "exact expected terminal categorical cross-entropy under each fixed candidate prior",
            "limitations": ["finite 16-hypothesis chain/fork game", "exact realized subset telemetry", "finite candidate prior set is not calibrated", "frontier growth can be exponential with horizon and scenario count", "no external benchmarks run"]
        },
        "frontier_stats": {obj: {"max_frontier": solvers[obj][3]["max"], "states": solvers[obj][3]["states"]} for obj in objectives},
        "rows": rows,
    }


def self_test():
    a, b = (0.1, 0.2, 0.4), (0.2, 0.3, 0.5)
    assert dominates(a, b)
    assert not dominates(b, a)
    result = run_benchmark(1, ("uniform", "high_spill_biased"), ("uniform", "chain_biased"), ("joint",))
    assert len(result["rows"]) == 2 * 2 * (1 + 4)  # one robust tree plus four baselines
    robust = [r for r in result["rows"] if r["policy"] == "fixed_prior_minimax_frontier"]
    for learner in ("uniform", "chain_biased"):
        for row in [x for x in robust if x["learner_prior"] == learner]:
            assert row["selected_policy_cross_entropy_bits"] <= row["worst_case_cross_entropy_bits"] + 1e-8
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--candidate-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--learner-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--source-git-blob", default="UNBOUND")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); print("FIXED_PRIOR_MINIMAX_FRONTIER_TESTS_PASSED"); return 0
        result = run_benchmark(args.horizon, tuple(args.candidate_priors.split(",")),
                               tuple(args.learner_priors.split(",")), tuple(args.objectives.split(",")))
        result["source_git_blob"] = args.source_git_blob
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else:
            print(raw.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError, RecursionError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2

if __name__ == "__main__":
    raise SystemExit(main())
