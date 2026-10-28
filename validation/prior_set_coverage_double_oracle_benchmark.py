#!/usr/bin/env python3
"""Exact finite-game benchmark: hand-picked prior coverage vs full-simplex adversary.

Research fixture only. It builds a transparent 16-hypothesis, two-action,
three-step-information (two observations) adaptive probing game. A policy tree
chooses the first action and the second action conditionally on the first binary
observation. For each terminal target (graph, kernel, joint), expected terminal
categorical cross-entropy is calculated using verify_prior_risk_linearity.py.

Because the risk of each fixed policy is linear in the true prior, worst-case
risk over the full simplex is attained at one of the 16 point-mass priors. The
script compares (a) all 8 deterministic policy trees robust to 6 hand-picked
priors, (b) the same trees robust to all 16 simplex vertices, and (c) a
prior-and-policy double-oracle procedure. It emits primal/dual values and an
independent full-simplex upper bound. This does NOT model the repository's exact
SCM likelihoods and is not an integration or SOTA result.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from verify_prior_risk_linearity import expected_terminal_cross_entropy, mixture

SCHEMA = "symthaea-prior-set-coverage-double-oracle-v1"
N_GRAPH = 4
N_KERNEL = 4
N_HYP = N_GRAPH * N_KERNEL
ACTIONS = (0, 1)
OBJECTIVES = ("graph", "kernel", "joint")
TOL = 1e-8
NUMERICAL_MARGIN_BITS = 1e-9  # explicit outward guard; not interval-arithmetic proof


def hypotheses():
    return tuple((g, k) for g in range(N_GRAPH) for k in range(N_KERNEL))


def class_map(objective: str) -> tuple[int, ...]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    if objective == "graph":
        return tuple(g for g, _ in hypotheses())
    if objective == "kernel":
        return tuple(k for _, k in hypotheses())
    return tuple(range(N_HYP))


def make_observation_model():
    """P(obs | theta, step, action); all outcomes have positive support.

    Action 0 is graph-sensitive with a small kernel-dependent nuisance effect;
    action 1 is kernel-sensitive with a small graph-dependent nuisance effect.
    Small time changes avoid making both observations exact duplicates.
    """
    graph_p = (0.08, 0.34, 0.66, 0.92)
    kernel_p = (0.10, 0.36, 0.64, 0.90)
    model = []
    for g, k in hypotheses():
        steps = []
        for step in range(2):
            rows = []
            for action in ACTIONS:
                if action == 0:
                    p1 = graph_p[g] + (k - 1.5) * 0.018 + (step * 2 - 1) * 0.012
                else:
                    p1 = kernel_p[k] + (g - 1.5) * 0.016 - (step * 2 - 1) * 0.010
                p1 = min(0.97, max(0.03, p1))
                rows.append([1.0 - p1, p1])
            steps.append(rows)
        model.append(steps)
    return model


def policy_trees():
    """All 2 * 2^2 complete deterministic trees for horizon 2."""
    trees = []
    for first, after_zero, after_one in itertools.product(ACTIONS, repeat=3):
        trees.append({"first": first, "after_obs0": after_zero, "after_obs1": after_one})
    return tuple(trees)


def policy_fn(tree):
    def action(history: tuple[int, ...]) -> int:
        if len(history) == 0:
            return int(tree["first"])
        if len(history) == 1:
            return int(tree["after_obs0"] if history[0] == 0 else tree["after_obs1"])
        raise ValueError("policy called beyond horizon")
    return action


def point_mass(index: int) -> tuple[float, ...]:
    if not 0 <= index < N_HYP:
        raise ValueError("point-mass index out of range")
    return tuple(1.0 if j == index else 0.0 for j in range(N_HYP))


def normalize(xs: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(xs, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("weights must be a nonempty finite nonnegative vector")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("weights must have positive sum")
    return tuple((arr / s).tolist())


def candidate_priors() -> dict[str, tuple[float, ...]]:
    hs = hypotheses()
    uniform = np.ones(N_HYP, dtype=float) / N_HYP
    # Six named, fixed, strictly positive priors. They are intentionally an
    # uncalibrated finite scenario set, not a claimed uncertainty set.
    graph_low = np.asarray([0.55, 0.25, 0.13, 0.07], dtype=float)
    graph_high = graph_low[::-1]
    kernel_low = np.asarray([0.52, 0.26, 0.14, 0.08], dtype=float)
    kernel_high = kernel_low[::-1]
    corr = np.asarray([[0.58 if g == k else 0.14 for k in range(4)] for g in range(4)], dtype=float)
    anti = np.asarray([[0.52 if k == (3 - g) else 0.16 for k in range(4)] for g in range(4)], dtype=float)
    raw = {
        "uniform": uniform,
        "graph_low": np.asarray([graph_low[g] / N_KERNEL for g, k in hs]),
        "graph_high": np.asarray([graph_high[g] / N_KERNEL for g, k in hs]),
        "kernel_low": np.asarray([kernel_low[k] / N_GRAPH for g, k in hs]),
        "graph_kernel_correlated": np.asarray([corr[g, k] for g, k in hs]),
        "anti_correlated": np.asarray([anti[g, k] for g, k in hs]),
    }
    return {name: normalize(v) for name, v in raw.items()}


def build_risk_tensor():
    model = make_observation_model()
    priors = candidate_priors()
    learner = tuple([1.0 / N_HYP] * N_HYP)
    trees = policy_trees()
    tensor = {}
    for objective in OBJECTIVES:
        A = np.zeros((len(trees), N_HYP), dtype=float)
        for pi, tree in enumerate(trees):
            pol = policy_fn(tree)
            for theta in range(N_HYP):
                A[pi, theta] = expected_terminal_cross_entropy(
                    point_mass(theta), learner, model, pol, class_map(objective), 2
                )
        # Independently score candidate mixed priors to enforce the linearity identity.
        checks = {}
        for name, q in priors.items():
            direct = np.asarray([
                expected_terminal_cross_entropy(q, learner, model, policy_fn(tree), class_map(objective), 2)
                for tree in trees
            ])
            linear = A @ np.asarray(q, dtype=float)
            checks[name] = float(np.max(np.abs(direct - linear)))
            if checks[name] > 2e-9:
                raise AssertionError(f"prior-linearity mismatch objective={objective} prior={name}: {checks[name]}")
        tensor[objective] = {"risk_by_policy_and_world": A, "linearity_checks": checks}
    return tensor, priors, trees


def solve_zero_sum(cost: np.ndarray) -> dict:
    """Solve min_policy_mix max_scenario expected cost and its dual game."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0 or np.any(~np.isfinite(cost)):
        raise ValueError("cost matrix must be finite, nonempty, two-dimensional")
    n_policy, n_scenario = cost.shape
    # Primal: min t; C^T mu - t <= 0; sum(mu)=1; mu>=0, t>=0.
    c = np.r_[np.zeros(n_policy), 1.0]
    A_ub = np.c_[cost.T, -np.ones(n_scenario)]
    b_ub = np.zeros(n_scenario)
    A_eq = np.r_[np.ones(n_policy), 0.0][None, :]
    b_eq = np.array([1.0])
    p = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * n_policy + [(0, None)], method="highs")
    if not p.success:
        raise RuntimeError(f"primal LP failed: {p.message}")
    mu = p.x[:n_policy]
    primal_value = float(p.x[-1])
    # Dual: max v; v <= min_policy C[policy] dot lambda; sum(lambda)=1.
    d_obj = np.r_[np.zeros(n_scenario), -1.0]
    d_ub = np.c_[-cost, np.ones(n_policy)]
    d_eq = np.r_[np.ones(n_scenario), 0.0][None, :]
    d = linprog(d_obj, A_ub=d_ub, b_ub=np.zeros(n_policy), A_eq=d_eq,
                b_eq=np.array([1.0]), bounds=[(0, None)] * n_scenario + [(None, None)], method="highs")
    if not d.success:
        raise RuntimeError(f"dual LP failed: {d.message}")
    lam = d.x[:n_scenario]
    dual_value = float(d.x[-1])
    gap = primal_value - dual_value
    if gap < -1e-7:
        raise AssertionError(f"invalid primal/dual ordering: gap={gap}")
    if gap > 2e-7:
        raise AssertionError(f"primal/dual gap too large: {gap}")
    return {
        "value_bits": primal_value,
        "dual_value_bits": dual_value,
        "primal_dual_gap_bits": max(0.0, gap),
        "policy_weights": mu.tolist(),
        "scenario_weights": lam.tolist(),
        "policy_support_size": int(np.count_nonzero(mu > 1e-8)),
        "scenario_support_size": int(np.count_nonzero(lam > 1e-8)),
        "max_scenario_cost_bits": float(np.max(cost.T @ mu)),
    }


def prior_risk_matrix(A: np.ndarray, priors: Sequence[Sequence[float]]) -> np.ndarray:
    return A @ np.asarray(priors, dtype=float).T


def solve_double_oracle(A: np.ndarray, named_priors: dict[str, tuple[float, ...]],
                        tolerance: float = TOL, max_iterations: int = 100) -> dict:
    """Policy/prior double oracle for Q=full simplex over the 16 hypotheses."""
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    n_policy, n_world = A.shape
    # Start with all six legacy prior scenarios and one policy column.
    scenarios = [(name, np.asarray(q, dtype=float)) for name, q in named_priors.items()]
    policies = [0]
    history = []
    converged = False
    last = None
    for iteration in range(1, max_iterations + 1):
        qmat = [q for _, q in scenarios]
        C = prior_risk_matrix(A[policies, :], qmat)
        game = solve_zero_sum(C)
        mu = np.asarray(game["policy_weights"], dtype=float)
        lam = np.asarray(game["scenario_weights"], dtype=float)
        q_bar = np.asarray(qmat, dtype=float).T @ lam
        q_bar = np.asarray(normalize(q_bar), dtype=float)
        # Best policy against dual-induced mixture prior.
        policy_risks = A @ q_bar
        best_policy = int(np.argmin(policy_risks))
        lower_raw = float(policy_risks[best_policy])
        # Current restricted master mix's exact upper bound over full simplex.
        world_risks = np.asarray(A[policies, :], dtype=float).T @ mu
        worst_world = int(np.argmax(world_risks))
        upper_raw = float(world_risks[worst_world])
        raw_gap = upper_raw - lower_raw
        if raw_gap < -2e-7:
            raise AssertionError(f"invalid robust interval: upper={upper_raw}, lower={lower_raw}")
        # Outward safety margin prevents a rounded LP endpoint from producing an
        # inverted interval. This is a numerical guard, not formal interval arithmetic.
        lower = lower_raw - NUMERICAL_MARGIN_BITS
        upper = upper_raw + NUMERICAL_MARGIN_BITS
        gap = upper - lower
        new_policy = best_policy not in policies
        already_world = any(np.max(np.abs(q - point_mass(worst_world))) < 1e-12 for _, q in scenarios)
        history.append({
            "iteration": iteration,
            "restricted_policy_count": len(policies),
            "restricted_scenario_count": len(scenarios),
            "restricted_game_value_bits": game["value_bits"],
            "robust_upper_bits": upper,
            "robust_lower_bits": lower,
            "raw_upper_bits": upper_raw,
            "raw_lower_bits": lower_raw,
            "raw_gap_bits": raw_gap,
            "numerical_margin_each_side_bits": NUMERICAL_MARGIN_BITS,
            "certificate_gap_bits": gap,
            "best_response_policy_index": best_policy,
            "worst_case_world_index": worst_world,
            "policy_response_added": bool(new_policy),
            "world_response_added": bool(not already_world),
            "master_lp_gap_bits": game["primal_dual_gap_bits"],
        })
        last = (game, mu, lam, q_bar, upper, lower, gap, worst_world)
        if gap <= tolerance:
            converged = True
            break
        if new_policy:
            policies.append(best_policy)
        if not already_world:
            scenarios.append((f"vertex_world_{worst_world:02d}", np.asarray(point_mass(worst_world), dtype=float)))
        if not new_policy and already_world:
            # Both exact best responses are already represented: nonzero gap means
            # numerical/master error or an implementation defect; fail closed.
            raise RuntimeError("double oracle stalled with positive robust-value gap")
    if last is None:
        raise RuntimeError("double oracle did not execute")
    game, mu, lam, q_bar, upper, lower, gap, worst_world = last
    # Rebuild mixture risks in global policy coordinates for auditability.
    mix_policy_weights = np.zeros(n_policy, dtype=float)
    for local_i, global_i in enumerate(policies):
        mix_policy_weights[global_i] = mu[local_i]
    qbar_scenarios = [q for _, q in scenarios]
    qbar = np.asarray(qbar_scenarios, dtype=float).T @ lam
    qbar = np.asarray(normalize(qbar), dtype=float)
    exact_policy_response = float(np.min(A @ qbar))
    exact_world_response = float(np.max(A.T @ mix_policy_weights))
    raw_exact_gap = exact_world_response - exact_policy_response
    if raw_exact_gap < -2e-7:
        raise AssertionError("independent recomputation produced an inverted robust interval")
    certified_lower = exact_policy_response - NUMERICAL_MARGIN_BITS
    certified_upper = exact_world_response + NUMERICAL_MARGIN_BITS
    exact_gap = certified_upper - certified_lower
    if converged and exact_gap > tolerance + 2e-7:
        raise AssertionError("reported convergence fails independent full-simplex recomputation")
    return {
        "status": "CERTIFIED_FINITE_GAME" if converged and exact_gap <= tolerance + 2e-7 else "NOT_CERTIFIED",
        "converged": converged,
        "iterations": len(history),
        "policy_indices": policies,
        "scenario_names": [name for name, _ in scenarios],
        "robust_upper_bits": certified_upper,
        "robust_lower_bits": certified_lower,
        "robust_upper_uninflated_bits": exact_world_response,
        "robust_lower_uninflated_bits": exact_policy_response,
        "raw_gap_bits": raw_exact_gap,
        "numerical_margin_each_side_bits": NUMERICAL_MARGIN_BITS,
        "certificate_gap_bits": exact_gap,
        "value_interval_bits": [certified_lower, certified_upper],
        "certificate_semantics": "floating-point LP bracket with explicit outward guard; not interval-arithmetic proof",
        "policy_weights_global": mix_policy_weights.tolist(),
        "policy_support_size": int(np.count_nonzero(mix_policy_weights > 1e-8)),
        "scenario_dual_weights": lam.tolist(),
        "dual_induced_prior": qbar.tolist(),
        "history": history,
        "max_master_lp_gap_bits": max(row["master_lp_gap_bits"] for row in history),
    }


def run_benchmark():
    benchmark_start = time.perf_counter()
    setup_start = time.perf_counter()
    tensor, priors, trees = build_risk_tensor()
    risk_setup_seconds = time.perf_counter() - setup_start
    rows = []
    for objective in OBJECTIVES:
        A = tensor[objective]["risk_by_policy_and_world"]
        legacy_names = list(priors)
        legacy_q = [priors[name] for name in legacy_names]
        six_matrix = prior_risk_matrix(A, legacy_q)
        full_matrix = A  # the 16 point-mass priors are exactly the columns of A
        six_start = time.perf_counter()
        six = solve_zero_sum(six_matrix)
        six_lp_seconds = time.perf_counter() - six_start
        full_start = time.perf_counter()
        full = solve_zero_sum(full_matrix)
        full_lp_seconds = time.perf_counter() - full_start
        do_start = time.perf_counter()
        do = solve_double_oracle(A, priors)
        double_oracle_seconds = time.perf_counter() - do_start
        # Exact independent re-score of the six-prior optimum against every world.
        legacy_mu = np.asarray(six["policy_weights"], dtype=float)
        legacy_world_risks = A.T @ legacy_mu
        legacy_full_upper = float(np.max(legacy_world_risks))
        legacy_worst_world = int(np.argmax(legacy_world_risks))
        rows.append({
            "objective": objective,
            "hypotheses": N_HYP,
            "deterministic_policy_trees": len(trees),
            "legacy_prior_scenarios": len(priors),
            "full_simplex_vertices": N_HYP,
            "risk_linearity_max_abs_error_bits": max(tensor[objective]["linearity_checks"].values()),
            "six_prior_game": {
                "restricted_value_bits": six["value_bits"],
                "primal_dual_gap_bits": six["primal_dual_gap_bits"],
                "policy_support_size": six["policy_support_size"],
                "out_of_set_full_simplex_upper_bits": legacy_full_upper,
                "worst_unrepresented_vertex_index": legacy_worst_world,
                "coverage_gap_vs_full_game_bits": max(0.0, legacy_full_upper - full["value_bits"]),
                "full_simplex_value_gap_vs_six_prior_game_bits": max(0.0, full["value_bits"] - six["value_bits"]),
            },
            "full_simplex_exact_game": {
                "value_bits": full["value_bits"],
                "primal_dual_gap_bits": full["primal_dual_gap_bits"],
                "policy_support_size": full["policy_support_size"],
                "scenario_support_size": full["scenario_support_size"],
            },
            "double_oracle": do,
            "timing_seconds_local": {
                "six_prior_master_lp": six_lp_seconds,
                "full_simplex_master_lp": full_lp_seconds,
                "double_oracle_total": double_oracle_seconds,
            },
            "comparisons": {
                "double_oracle_minus_full_game_bits": abs(do["robust_upper_uninflated_bits"] - full["value_bits"]),
                "six_prior_actual_upper_minus_full_value_bits": legacy_full_upper - full["value_bits"],
                "double_oracle_certified_gap_bits": do["certificate_gap_bits"],
            },
            "risk_coefficients_by_world_bits": A.tolist(),
            "policy_trees": list(trees),
        })
    return {
        "schema": SCHEMA,
        "status": "EXACT_SYNTHETIC_FINITE_GAME_PRIOR_COVERAGE_BENCHMARK_NOT_REPOSITORY_INTEGRATION",
        "model": {
            "latent_hypotheses": N_HYP,
            "factorization": "4 graph classes x 4 mechanism/kernel classes",
            "action_count": len(ACTIONS),
            "horizon": 2,
            "observations_per_step": 2,
            "deterministic_policy_trees": len(trees),
            "learner_prior": "uniform strictly-positive prior over all 16 hypotheses",
            "world_likelihood": "synthetic action-dependent Bernoulli observations with positive support",
            "candidate_priors": list(priors),
            "objectives": list(OBJECTIVES),
            "worst_case_set": "full simplex over 16 fixed latent hypotheses; exact extreme points are 16 point masses due verified linearity",
        },
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                      "risk_tensor_setup_seconds_local": risk_setup_seconds,
                      "total_benchmark_seconds_local": time.perf_counter() - benchmark_start},
        "limitations": [
            "standalone synthetic likelihood is not imported from the Symthaea repository",
            "only 8 deterministic horizon-two trees; a deliberately small exact game",
            "full-simplex guarantee applies to prior-weight uncertainty over this fixed 16-hypothesis support and shared likelihood model",
            "does not cover omitted hypotheses, likelihood misspecification, or noisy/missing semantics beyond this fixture",
            "runtime numbers are local and not an industry comparison",
            "no ACDB, CausalArena, or TCD-Arena run in this lane",
        ],
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-id", default="LOCAL_UNBOUND")
    args = ap.parse_args()
    try:
        result = run_benchmark()
        result["source_id"] = args.source_id
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
        print(json.dumps({"output": str(args.output), "bytes": len(raw), "sha256": digest,
                          "rows": len(result["rows"]),
                          "all_double_oracles_certified": all(r["double_oracle"]["status"] == "CERTIFIED_FINITE_GAME" for r in result["rows"]),
                          "max_double_oracle_gap_bits": max(r["double_oracle"]["certificate_gap_bits"] for r in result["rows"]),
                          "max_master_lp_gap_bits": max(r["double_oracle"]["max_master_lp_gap_bits"] for r in result["rows"])}, indent=2))
    except (ValueError, AssertionError, ArithmeticError, RuntimeError, RecursionError) as exc:
        raise SystemExit(f"NOT QUALIFIED: {exc}") from exc

if __name__ == "__main__":
    main()
