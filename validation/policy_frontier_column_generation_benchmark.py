#!/usr/bin/env python3
"""Column-generation solver for fixed-prior minimax adaptive causal probes.

Rather than enumerate every nondominated policy-tree risk vector, maintain a
restricted master LP over discovered deterministic trees. Its dual gives scenario
weights; a weighted-belief dynamic-programming pricing oracle searches for a tree
with lower scalarized risk. Termination yields a primal feasible policy mixture
and a dual lower bound for the finite candidate-prior game.

Research-only Python/NumPy/SciPy. This is not a production Symthaea runtime path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
from objective_matched_oracle_benchmark import ACTIONS, OBJECTIVES, predictive, update
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from prior_misspecification_benchmark import terminal_cross_entropy
from fixed_prior_minimax_policy_benchmark import (
    make_frontier_solver, evaluate_tree_from_posteriors,
)
from randomized_minimax_policy_mix_benchmark import solve_minimax_mixture

SCHEMA = "symthaea-policy-frontier-column-generation-v1"
BASE_SOURCE_BLOBS = {
    "fixed_prior_minimax_policy_benchmark.py": "438301f07649d7ffebca6690990a4909ea28e054",
    "randomized_minimax_policy_mix_benchmark.py": "b967b77490842363705378a20097fb34e5aecef3",
    "objective_matched_oracle_benchmark.py": "f101346c2f37cfad4e029ae280e9a002548f309c",
    "prior_misspecification_benchmark.py": "541c31521e1b9e49dfb84c0fa265e877ca9ea632",
}
TOL = 1e-8


def _tree_key(tree):
    return json.dumps(tree, separators=(",", ":"), allow_nan=False)


def make_pricing_oracle(objective, candidate_prior_names):
    """Return exact scalarized-risk DP pricing oracle for a fixed dual vector.

    `masses` are unnormalized scenario weights including the probability of the
    action-observation history. Normalizing them at a node does not change the
    minimizing subtree. The returned risk vector remains conditional on each
    scenario at that node, permitting independent root-level re-scoring.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if not candidate_prior_names or any(n not in PRIOR_NAMES for n in candidate_prior_names):
        raise ValueError("invalid candidate prior set")

    def best_tree(learner_posterior, horizon, scenario_weights):
        if horizon not in (1, 2, 3):
            raise ValueError("horizon must be 1, 2, or 3")
        if len(scenario_weights) != len(candidate_prior_names):
            raise ValueError("scenario weight dimension mismatch")
        if any((not math.isfinite(float(w))) or w < -1e-10 for w in scenario_weights):
            raise ValueError("scenario weights must be finite and nonnegative")
        weights = tuple(max(0.0, float(w)) for w in scenario_weights)
        if sum(weights) <= 0.0:
            raise ValueError("scenario weights must have positive mass")
        initial_scenarios = tuple(tuple(build_prior(n)) for n in candidate_prior_names)
        stats = {"dp_states": 0, "pricing_action_evaluations": 0, "cache_hits": 0}

        # The DP state includes the normalized dual mass reaching this history.
        # Since policy choice depends only on relative scenario weights, normalize
        # before cache lookup; no approximation or weight quantization is used.
        def normalize_mass(masses):
            mass_sum = sum(masses)
            return tuple(m / mass_sum for m in masses) if mass_sum > 0.0 else tuple(0.0 for _ in masses)

        @lru_cache(maxsize=None)
        def rec(learner, scenarios, normalized, remaining):
            stats["dp_states"] += 1
            if remaining == 0:
                risks = tuple(terminal_cross_entropy(q, learner, objective) for q in scenarios)
                return risks, None, float(np.dot(normalized, risks))

            best = None
            for action in ACTIONS:
                stats["pricing_action_evaluations"] += 1
                predicted = [dict(predictive(q, action)) for q in scenarios]
                observations = tuple(sorted(set().union(*(set(p) for p in predicted)), key=repr))
                risk_acc = [0.0] * len(scenarios)
                branches = []
                for obs in observations:
                    probs = tuple(p.get(obs, 0.0) for p in predicted)
                    if not any(p > 0.0 for p in probs):
                        continue
                    next_scenarios = tuple(
                        update(q, action, obs) if prob > 0.0 else q
                        for q, prob in zip(scenarios, probs)
                    )
                    learner_next = update(learner, action, obs)
                    child_masses = normalize_mass(tuple(m * prob for m, prob in zip(normalized, probs)))
                    child_risk, child_tree, _ = rec(learner_next, next_scenarios, child_masses, remaining - 1)
                    for i, prob in enumerate(probs):
                        risk_acc[i] += prob * child_risk[i]
                    branches.append((obs, child_tree))
                risk_vector = tuple(risk_acc)
                scalar = float(np.dot(normalized, risk_vector))
                candidate = (scalar, risk_vector, (action, tuple(branches)))
                if best is None or (candidate[0], candidate[2][0], candidate[1]) < (best[0], best[2][0], best[1]):
                    best = candidate
            if best is None:
                raise RuntimeError("pricing oracle found no action")
            return best[1], best[2], best[0]

        risks, tree, scalar = rec(tuple(learner_posterior), initial_scenarios, normalize_mass(weights), horizon)
        return {
            "risk_vector": list(risks), "tree": tree,
            "scalarized_risk_bits": scalar,
            "dp_states": stats["dp_states"],
            "pricing_action_evaluations": stats["pricing_action_evaluations"],
            "pricing_cache_hits": rec.cache_info().hits,
        }
    return best_tree


def verify_pricing_result(result, objective, learner, candidate_prior_names, horizon, weights):
    """Independently re-score a tree before adding it to the master or certifying it."""
    independently_scored = tuple(
        evaluate_tree_from_posteriors(result["tree"], tuple(build_prior(name)), tuple(learner), objective, horizon)
        for name in candidate_prior_names
    )
    if any(abs(x-y) > 1e-8 for x, y in zip(result["risk_vector"], independently_scored)):
        raise AssertionError("pricing tree risk vector failed independent evaluation")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("pricing verification weights must have positive mass")
    scalar_rescored = float(np.dot(np.asarray(weights, dtype=float) / total, independently_scored))
    if abs(scalar_rescored - result["scalarized_risk_bits"]) > 1e-8:
        raise AssertionError("pricing scalar does not match independently scored risk vector")
    result["risk_vector"] = list(independently_scored)
    result["scalarized_risk_bits"] = scalar_rescored
    return result


def solve_column_generation(objective, learner_prior_name, candidate_prior_names, horizon,
                            max_iterations=100, tolerance=1e-8):
    if learner_prior_name not in PRIOR_NAMES:
        raise ValueError("unknown learner prior")
    if not candidate_prior_names or any(n not in PRIOR_NAMES for n in candidate_prior_names):
        raise ValueError("unknown candidate prior")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    learner = tuple(build_prior(learner_prior_name))
    pricing = make_pricing_oracle(objective, candidate_prior_names)
    columns = []
    by_key = {}
    total_pricing_states = 0
    total_pricing_actions = 0

    # One neutral seed keeps the restricted master small. Subsequent dual pricing
    # discovers only columns needed by the minimax certificate.
    seed_weights = [1.0 / len(candidate_prior_names)] * len(candidate_prior_names)
    seed = pricing(learner, horizon, seed_weights)
    seed = verify_pricing_result(seed, objective, learner, candidate_prior_names, horizon, seed_weights)
    total_pricing_states += seed["dp_states"]
    total_pricing_actions += seed["pricing_action_evaluations"]
    seed_key = _tree_key(seed["tree"])
    by_key[seed_key] = 0
    columns.append({"tree": seed["tree"], "risk": seed["risk_vector"], "discovered_by": "uniform-scenario-seed"})

    iteration_rows = []
    converged = False
    last_pricing_result = None
    last_pricing_dual = None
    for iteration in range(1, max_iterations + 1):
        risks = np.asarray([col["risk"] for col in columns], dtype=float)
        master = solve_minimax_mixture(risks)
        dual = master["scenario_dual_weights"]
        result = pricing(learner, horizon, dual)
        result = verify_pricing_result(result, objective, learner, candidate_prior_names, horizon, dual)
        last_pricing_result = result
        last_pricing_dual = list(dual)
        total_pricing_states += result["dp_states"]
        total_pricing_actions += result["pricing_action_evaluations"]
        price_value = result["scalarized_risk_bits"]
        restricted_lower = master["dual_lower_bound_bits"]
        certified_gap = max(0.0, master["worst_case_bits"] - price_value)
        key = _tree_key(result["tree"])
        is_new = key not in by_key
        iteration_rows.append({
            "iteration": iteration, "master_policy_count": len(columns),
            "master_primal_upper_bits": master["worst_case_bits"],
            "restricted_dual_value_bits": restricted_lower,
            "pricing_minimum_bits": price_value,
            "primal_to_pricing_gap_bits": certified_gap,
            "new_policy_tree": is_new,
            "pricing_dp_states": result["dp_states"],
            "pricing_action_evaluations": result["pricing_action_evaluations"],
        })
        if price_value >= restricted_lower - tolerance and certified_gap <= tolerance:
            converged = True
            break
        if not is_new:
            # Duplicate optimal columns plus a positive gap indicates either a
            # numeric/dual-oracle defect or an invalid stopping tolerance.
            if certified_gap > tolerance:
                raise RuntimeError("pricing returned an existing tree despite a nonzero certificate gap")
            converged = True
            break
        by_key[key] = len(columns)
        columns.append({"tree": result["tree"], "risk": result["risk_vector"], "discovered_by": f"dual-pricing-{iteration}"})

    final_risks = np.asarray([col["risk"] for col in columns], dtype=float)
    final_master = solve_minimax_mixture(final_risks)
    final_dual = final_master["scenario_dual_weights"]
    if converged and last_pricing_result is not None and last_pricing_dual is not None and \
            max(abs(a-b) for a, b in zip(final_dual, last_pricing_dual)) <= 1e-7:
        # The last pricing call already used this unchanged master dual and was
        # independently re-scored. Reuse it rather than repeat the same DP.
        final_pricing = last_pricing_result
    else:
        final_pricing = pricing(learner, horizon, final_dual)
        final_pricing = verify_pricing_result(final_pricing, objective, learner, candidate_prior_names, horizon, final_dual)
        total_pricing_states += final_pricing["dp_states"]
        total_pricing_actions += final_pricing["pricing_action_evaluations"]
    dual_lower = final_pricing["scalarized_risk_bits"]
    primal_upper = final_master["worst_case_bits"]
    gap = max(0.0, primal_upper - dual_lower)
    if converged and gap > tolerance * 2:
        raise AssertionError(f"convergence claim contradicts final certificate gap {gap}")

    # Every master column was independently re-scored before insertion.
    mixture_risks = final_risks.T @ np.asarray(final_master["weights"], dtype=float)
    if abs(float(max(mixture_risks)) - primal_upper) > 1e-7:
        raise AssertionError("final mixture objective differs from independently re-scored trees")

    return {
        "objective": objective, "learner_prior": learner_prior_name,
        "candidate_prior_names": list(candidate_prior_names), "horizon": horizon,
        "status": "CERTIFIED_COLUMN_GENERATION" if converged and gap <= tolerance * 2 else "NOT_CONVERGED",
        "converged": converged, "iterations": len(iteration_rows),
        "policy_tree_columns": len(columns), "full_frontier_not_enumerated_by_solver": True,
        "master_primal_upper_bits": primal_upper,
        "pricing_dual_lower_bits": dual_lower,
        "primal_dual_certificate_gap_bits": gap,
        "tolerance_bits": tolerance,
        "mixture_risk_vector_bits": mixture_risks.tolist(),
        "mixture_support_size": int(np.count_nonzero(np.asarray(final_master["weights"]) > 1e-8)),
        "mixture_weights": final_master["weights"],
        "mixture_column_indices": [i for i, w in enumerate(final_master["weights"]) if w > 1e-8],
        "scenario_dual_weights": final_dual,
        "iterations_detail": iteration_rows,
        "total_pricing_dp_states": total_pricing_states,
        "total_pricing_action_evaluations": total_pricing_actions,
        "policy_trees": [{"risk_vector_bits": c["risk"], "discovered_by": c["discovered_by"]} for c in columns],
    }

def run_benchmark(candidate_prior_names=PRIOR_NAMES, learner_prior_names=PRIOR_NAMES,
                  objectives=OBJECTIVES, horizons=(1, 2), max_iterations=100, tolerance=1e-8):
    rows = []
    for objective in objectives:
        # Cold exact-frontier oracle is used only to validate the column-generation solver.
        _, exact_frontier, _, exact_stats = make_frontier_solver(objective, candidate_prior_names)
        for learner_name in learner_prior_names:
            learner = tuple(build_prior(learner_name))
            for horizon in horizons:
                exact_start = time.perf_counter()
                exact_options = exact_frontier(learner, tuple(tuple(build_prior(n)) for n in candidate_prior_names), horizon)
                exact_frontier_seconds = time.perf_counter() - exact_start
                exact_mix = solve_minimax_mixture(np.asarray([risk for risk, _ in exact_options], dtype=float))
                exact_pipeline_seconds = time.perf_counter() - exact_start
                cg_start = time.perf_counter()
                cg = solve_column_generation(objective, learner_name, candidate_prior_names, horizon,
                                             max_iterations=max_iterations, tolerance=tolerance)
                cg_seconds = time.perf_counter() - cg_start
                exact_gap = cg["master_primal_upper_bits"] - exact_mix["worst_case_bits"]
                row = {
                    "objective": objective, "learner_prior": learner_name,
                    "candidate_prior_names": list(candidate_prior_names), "horizon": horizon,
                    "column_generation": cg,
                    "exact_frontier_oracle": {
                        "frontier_size": len(exact_options),
                        "frontier_peak": exact_stats["max"],
                        "frontier_states": exact_stats["states"],
                        "minimax_mixture_worst_case_bits": exact_mix["worst_case_bits"],
                        "minimax_mixture_primal_dual_gap_bits": exact_mix["primal_dual_gap_bits"],
                    },
                    "column_generation_minus_exact_bits": exact_gap,
                    "column_generation_runtime_seconds": cg_seconds,
                    "exact_frontier_runtime_seconds": exact_frontier_seconds,
                    "exact_frontier_plus_master_lp_runtime_seconds": exact_pipeline_seconds,
                }
                if cg["converged"] and abs(exact_gap) > max(2e-7, tolerance * 10):
                    raise AssertionError(f"column generation disagrees with exact frontier by {exact_gap} bits")
                rows.append(row)
    return {
        "schema": SCHEMA,
        "status": "FINITE_MODEL_COLUMN_GENERATION_WITH_EXACT_FRONTIER_ORACLE",
        "source_base_blobs": BASE_SOURCE_BLOBS,
        "protocol": {
            "candidate_true_prior_identity_fixed_per_episode": True,
            "true_prior_identity_hidden_from_policy": True,
            "score": "expected terminal categorical cross-entropy (bits)",
            "algorithm": "restricted master minimax LP + exact scalarized dynamic-programming pricing oracle + primal/dual stopping gap",
            "oracle_role": "exhaustive Pareto-frontier solver is used only for independent finite-model validation and frontier-size comparison",
            "limitations": ["16 joint hypotheses; chain/fork graphs only", "six finite, uncalibrated candidate priors", "exact target-subset telemetry", "no intervention costs, measurement noise, missing telemetry, or task disturbance", "no external benchmark executed", "runtime is environment-specific research measurement"],
        },
        "rows": rows,
    }


def self_test():
    # An explicit two-scenario game where the minimax optimum mixes policies.
    from randomized_minimax_policy_mix_benchmark import solve_minimax_mixture as lp
    r = lp([[0, 1], [1, 0]])
    assert abs(r["worst_case_bits"] - 0.5) < 1e-8
    result = run_benchmark(("uniform", "high_spill_biased"),
                           ("uniform", "chain_biased"), ("joint",), (1, 2))
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        cg = row["column_generation"]
        assert cg["converged"] and cg["primal_dual_certificate_gap_bits"] <= 2e-8
        assert abs(row["column_generation_minus_exact_bits"]) <= 2e-7
        assert cg["policy_tree_columns"] <= row["exact_frontier_oracle"]["frontier_size"]
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidate-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--learner-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--horizons", default="1,2")
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--tolerance", type=float, default=1e-8)
    ap.add_argument("--source-git-blob", default="UNBOUND")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); print("POLICY_FRONTIER_COLUMN_GENERATION_TESTS_PASSED"); return 0
        result = run_benchmark(tuple(args.candidate_priors.split(",")), tuple(args.learner_priors.split(",")),
                               tuple(args.objectives.split(",")), tuple(int(x) for x in args.horizons.split(",")),
                               args.max_iterations, args.tolerance)
        result["source_git_blob"] = args.source_git_blob
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else: print(raw.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError, RecursionError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2

if __name__ == "__main__": raise SystemExit(main())
