#!/usr/bin/env python3
"""Benchmark epsilon-bucket approximation for fixed-prior minimax policy frontiers.

The exact fixed-prior policy-tree solver is the oracle. Approximate search groups
risk vectors by ceil(loss/epsilon) coordinate buckets, selects a deterministic
representative, and then performs ordinary exact Pareto pruning. Every selected
approximate policy is re-evaluated exactly under each candidate fixed prior; the
reported optimality gap is its actual worst-case CE minus the exact frontier optimum.
No theoretical epsilon-regret guarantee is claimed: the exact oracle provides an
empirical/certified per-instance gap for this finite model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

from objective_matched_oracle_benchmark import ACTIONS, HYPOTHESES, OBJECTIVES, predictive, update
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from prior_misspecification_benchmark import terminal_cross_entropy
from fixed_prior_minimax_policy_benchmark import (
    TOL, dominates, prune_frontier, make_frontier_solver, evaluate_tree_from_posteriors,
)

SCHEMA = "symthaea-epsilon-frontier-approximation-v1"
BASE_SOURCE_BLOBS = {
    "fixed_prior_minimax_policy_benchmark.py": "438301f07649d7ffebca6690990a4909ea28e054",
}
DEFAULT_EPSILONS = (0.0, 0.005, 0.01, 0.025, 0.05)


def prune_frontier_epsilon(candidates, epsilon):
    """Keep one representative per upward-rounded risk box, then Pareto-prune.

    The risk-vector bucket is a compute-saving heuristic. It does not itself certify
    a horizon-independent regret bound; exact evaluation below measures the actual
    worst-case loss gap of the selected policy tree.
    """
    if epsilon <= 0:
        return prune_frontier(candidates)
    boxes = {}
    for candidate in candidates:
        risk = candidate[0]
        key = tuple(math.ceil((x - TOL) / epsilon) for x in risk)
        prior = boxes.get(key)
        score = (max(risk), sum(risk), risk)
        if prior is None or score < (max(prior[0]), sum(prior[0]), prior[0]):
            boxes[key] = candidate
    return prune_frontier(list(boxes.values()))


def make_epsilon_frontier_solver(objective, candidate_prior_names, epsilon):
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if not candidate_prior_names or any(n not in PRIOR_NAMES for n in candidate_prior_names):
        raise ValueError("candidate prior set must contain known prior names")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite nonnegative number")
    if epsilon == 0:
        return make_frontier_solver(objective, candidate_prior_names)
    initial_scenarios = tuple(tuple(build_prior(n)) for n in candidate_prior_names)
    stats = {"max": 0, "states": 0, "bucket_keys_seen": 0}

    @lru_cache(maxsize=None)
    def frontier(learner_posterior, scenarios, remaining):
        stats["states"] += 1
        if remaining == 0:
            risk = tuple(terminal_cross_entropy(q, learner_posterior, objective) for q in scenarios)
            return ((risk, None),)
        all_candidates = []
        for action in ACTIONS:
            pred_by_scenario = [dict(predictive(q, action)) for q in scenarios]
            observations = tuple(sorted(set().union(*(set(p) for p in pred_by_scenario)), key=repr))
            partial = [((0.0,) * len(scenarios), ())]
            for obs in observations:
                probs = tuple(pred.get(obs, 0.0) for pred in pred_by_scenario)
                if not any(v > 0.0 for v in probs):
                    continue
                scenarios_next = tuple(update(q, action, obs) if prob > 0.0 else q
                                        for q, prob in zip(scenarios, probs))
                learner_next = update(learner_posterior, action, obs)
                children = frontier(learner_next, scenarios_next, remaining - 1)
                combined = []
                for partial_risk, branches in partial:
                    for child_risk, child_tree in children:
                        risk = tuple(x + prob * y for x, prob, y in zip(partial_risk, probs, child_risk))
                        combined.append((risk, branches + ((obs, child_tree),)))
                stats["bucket_keys_seen"] += len(combined)
                partial = prune_frontier_epsilon(combined, epsilon)
                stats["max"] = max(stats["max"], len(partial))
            all_candidates.extend((risk, (action, branches)) for risk, branches in partial)
        stats["bucket_keys_seen"] += len(all_candidates)
        result = tuple(prune_frontier_epsilon(all_candidates, epsilon))
        stats["max"] = max(stats["max"], len(result))
        return result

    def best_policy(learner_posterior, horizon):
        if horizon not in (1, 2, 3):
            raise ValueError("horizon must be 1, 2, or 3")
        options = frontier(tuple(learner_posterior), initial_scenarios, horizon)
        return min(options, key=lambda item: (max(item[0]), sum(item[0]) / len(item[0]), item[0]))

    return initial_scenarios, frontier, best_policy, stats


def run_benchmark(candidate_prior_names=PRIOR_NAMES, learner_prior_names=PRIOR_NAMES,
                  objectives=OBJECTIVES, max_horizon=2, epsilons=DEFAULT_EPSILONS):
    if max_horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    for names in (candidate_prior_names, learner_prior_names, objectives, epsilons):
        if not names:
            raise ValueError("input lists must not be empty")
    if any(n not in PRIOR_NAMES for n in candidate_prior_names + learner_prior_names):
        raise ValueError("unknown candidate or learner prior")
    if any(o not in OBJECTIVES for o in objectives):
        raise ValueError("unknown objective")
    if any(not math.isfinite(e) or e < 0 for e in epsilons):
        raise ValueError("epsilons must be finite and nonnegative")
    rows, complexity = [], {}
    exact_solvers = {obj: make_frontier_solver(obj, candidate_prior_names) for obj in objectives}
    for objective in objectives:
        exact_scenarios, exact_frontier, exact_best, exact_stats = exact_solvers[objective]
        # Time the exact reference as its own cold search before approximate runs.
        exact_start = time.perf_counter()
        exact_cache = {}
        for learner_name in learner_prior_names:
            learner_p = tuple(build_prior(learner_name))
            for horizon in range(1, max_horizon + 1):
                exact_vector, exact_tree = exact_best(learner_p, horizon)
                exact_scores = tuple(evaluate_tree_from_posteriors(
                    exact_tree, tuple(build_prior(true_name)), learner_p, objective, horizon)
                    for true_name in candidate_prior_names)
                if any(abs(a-b) > 1e-8 for a, b in zip(exact_scores, exact_vector)):
                    raise AssertionError("exact frontier risk vector failed independent evaluation")
                exact_cache[(learner_name, horizon)] = (exact_vector, exact_tree, exact_scores)
        exact_elapsed = time.perf_counter() - exact_start

        for epsilon in epsilons:
            scenarios, approx_frontier, approx_best, stats = make_epsilon_frontier_solver(
                objective, candidate_prior_names, epsilon)
            per_epsilon_rows = []
            approximate_search_elapsed = 0.0
            evaluation_elapsed = 0.0
            for learner_name in learner_prior_names:
                learner_p = tuple(build_prior(learner_name))
                for horizon in range(1, max_horizon + 1):
                    exact_vector, exact_tree, exact_scores = exact_cache[(learner_name, horizon)]
                    policy_start = time.perf_counter()
                    approx_vector, approx_tree = approx_best(learner_p, horizon)
                    approximate_search_elapsed += time.perf_counter() - policy_start
                    evaluation_start = time.perf_counter()
                    actual_scores = tuple(evaluate_tree_from_posteriors(
                        approx_tree, tuple(build_prior(true_name)), learner_p, objective, horizon)
                        for true_name in candidate_prior_names)
                    evaluation_elapsed += time.perf_counter() - evaluation_start
                    exact_worst = max(exact_scores)
                    approx_worst = max(actual_scores)
                    if approx_worst < exact_worst - 1e-8:
                        raise AssertionError("approximate valid policy scored below exact minimax optimum")
                    row = {
                        "objective": objective, "epsilon_bits": epsilon, "learner_prior": learner_name,
                        "horizon": horizon, "candidate_prior_names": list(candidate_prior_names),
                        "exact_worst_case_ce_bits": exact_worst,
                        "approximate_policy_worst_case_ce_bits": approx_worst,
                        "worst_case_optimality_gap_bits": max(0.0, approx_worst-exact_worst),
                        "frontier_size_exact": len(exact_frontier(learner_p, exact_scenarios, horizon)),
                        "frontier_size_approx": len(approx_frontier(learner_p, scenarios, horizon)),
                        "frontier_peak_exact": exact_stats["max"],
                        "frontier_peak_approx": stats["max"],
                        "memoized_states_exact": exact_stats["states"],
                        "memoized_states_approx": stats["states"],
                        "bucket_candidates_seen": stats.get("bucket_keys_seen", 0),
                        "true_prior_scores_exact": list(exact_scores),
                        "true_prior_scores_approx": list(actual_scores),
                        "well_specified_candidates": [n == learner_name for n in candidate_prior_names],
                    }
                    per_epsilon_rows.append(row)
            complexity[f"{objective}/eps={epsilon:g}"] = {
                "exact_oracle_search_seconds": exact_elapsed,
                "approximate_policy_search_seconds": approximate_search_elapsed,
                "approximate_tree_rescore_seconds": evaluation_elapsed,
                "max_frontier_exact": exact_stats["max"],
                "max_frontier_approx": stats["max"],
                "memoized_states_exact": exact_stats["states"],
                "memoized_states_approx": stats["states"],
                "bucket_candidates_seen": stats.get("bucket_keys_seen", 0),
            }
            for row in per_epsilon_rows:
                row["exact_oracle_search_seconds_for_objective"] = exact_elapsed
                row["approximate_policy_search_seconds_for_objective_eps"] = approximate_search_elapsed
                row["approximate_tree_rescore_seconds_for_objective_eps"] = evaluation_elapsed
            rows.extend(per_epsilon_rows)
    return {
        "schema": SCHEMA,
        "status": "EXACT_ORACLE_EVALUATED_EPSILON_FRONTIER_PILOT_NOT_THEORETICAL_REGRET_GUARANTEE",
        "source_base_blobs": BASE_SOURCE_BLOBS,
        "protocol": {
            "candidate_priors": list(candidate_prior_names), "learner_priors": list(learner_prior_names),
            "objectives": list(objectives), "horizons": list(range(1, max_horizon+1)),
            "epsilons_bits": list(epsilons), "score": "exact expected terminal cross entropy under each fixed candidate prior",
            "approximation": "ceil-loss epsilon buckets; deterministic representative selection; exact Pareto prune; each resulting policy tree is independently re-evaluated under every candidate prior",
            "gap_semantics": "actual approximate policy worst-case loss minus exact Pareto-frontier minimax optimum for same learner/objective/horizon/candidate set",
            "limitations": ["epsilon bucketing is empirical, with no horizon-independent epsilon-regret theorem claimed", "finite 16-hypothesis chain/fork game", "six priors are uncalibrated candidates", "exact target-subset telemetry", "no external benchmark run"]
        },
        "complexity": complexity,
        "rows": rows,
    }


def self_test():
    candidates = [((0.101, 0.201), "a"), ((0.104, 0.204), "b"), ((0.201, 0.101), "c")]
    exact = prune_frontier(candidates)
    approx = prune_frontier_epsilon(candidates, 0.01)
    assert len(approx) <= len(exact)
    result = run_benchmark(("uniform", "high_spill_biased"), ("uniform", "chain_biased"),
                           ("joint",), 2, (0.0, 0.025))
    assert len(result["rows"]) == 2 * 2 * 2  # learner x horizon x epsilon; prior risks are vectors
    by = {(r["learner_prior"],r["horizon"],r["epsilon_bits"]):r for r in result["rows"]}
    for learner in ("uniform", "chain_biased"):
        for horizon in (1,2):
            exact_row = by[(learner,horizon,0.0)]
            approx_row = by[(learner,horizon,0.025)]
            assert approx_row["worst_case_optimality_gap_bits"] >= -1e-9
            if horizon == 1:
                assert abs(approx_row["worst_case_optimality_gap_bits"]) < 1e-8
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--horizon",type=int,default=2)
    ap.add_argument("--candidate-priors",default=",".join(PRIOR_NAMES))
    ap.add_argument("--learner-priors",default=",".join(PRIOR_NAMES))
    ap.add_argument("--objectives",default=",".join(OBJECTIVES))
    ap.add_argument("--epsilons",default=",".join(str(x) for x in DEFAULT_EPSILONS))
    ap.add_argument("--source-git-blob",default="UNBOUND")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    try:
        if args.self_test:
            self_test(); print("EPSILON_FRONTIER_TESTS_PASSED"); return 0
        result=run_benchmark(tuple(args.candidate_priors.split(",")),tuple(args.learner_priors.split(",")),
          tuple(args.objectives.split(",")),args.horizon,tuple(float(x) for x in args.epsilons.split(",")))
        result["source_git_blob"]=args.source_git_blob
        raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        digest=hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else: print(raw.decode(),end="")
        return 0
    except (ValueError,AssertionError,OSError,RuntimeError,RecursionError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2

if __name__=="__main__": raise SystemExit(main())
