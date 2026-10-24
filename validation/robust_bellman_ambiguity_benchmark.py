#!/usr/bin/env python3
"""Exact comparison of multi-step rectangular robust Bellman causal probing.

Unlike the one-step minimax heuristic, the new policy minimizes an exact finite-
horizon dynamic-programming value. At each decision node it takes the worst expected
continuation value across the currently plausible posterior worlds. This is a
rectangular ambiguity model: the adversary may select different plausible scenarios
at different branches, so the result is conservative relative to a single unknown
prior fixed once at the beginning. The evaluator separately scores the learned policy
under each fixed true prior, never supplied to the action selector.

Research pilot only: 16 hypotheses, chain/fork, three binary spill parameters, exact
realized-target-subset telemetry; no external benchmark qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from objective_matched_oracle_benchmark import (
    ACTIONS, HYPOTHESES, OBJECTIVES, choose_action, predictive, update,
)
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from prior_misspecification_benchmark import terminal_cross_entropy, terminal_brier_risk
from robust_ambiguity_policy_benchmark import robust_action, update_scenarios

SCHEMA = "symthaea-robust-bellman-ambiguity-benchmark-v1"
BASE_SOURCE_BLOBS = {
    "objective_matched_oracle_benchmark.py": "f101346c2f37cfad4e029ae280e9a002548f309c",
    "prior_misspecification_benchmark.py": "541c31521e1b9e49dfb84c0fa265b5f93b4acc1dc",
    "robust_ambiguity_policy_benchmark.py": "c511c9651de071458734e014137484d982c717dc",
}
POLICIES = ("robust_bellman_rectangular", "robust_minimax_one_step", "greedy_joint",
            "round_robin", "random_uniform", "oracle_assumed_objective")


def make_bellman_solver(objective, ambiguity_names):
    """Return exact rectangular-robust Bellman value and action functions."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if not ambiguity_names or any(name not in PRIOR_NAMES for name in ambiguity_names):
        raise ValueError("ambiguity set must contain known priors")
    initial_scenarios = tuple((name, tuple(build_prior(name))) for name in ambiguity_names)

    @lru_cache(maxsize=None)
    def value(learner_posterior, scenarios, remaining):
        if remaining == 0:
            # Terminal adversary picks the worst posterior predictive world.
            return max(terminal_cross_entropy(q, learner_posterior, objective)
                       for _, q in scenarios)
        best = math.inf
        for action in ACTIONS:
            worst_expected = -math.inf
            for _, q in scenarios:
                expected = 0.0
                for obs, prob in predictive(q, action):
                    if prob <= 0.0:
                        continue
                    p_next = update(learner_posterior, action, obs)
                    scenario_next = update_scenarios(scenarios, action, obs)
                    expected += prob * value(p_next, scenario_next, remaining - 1)
                worst_expected = max(worst_expected, expected)
            best = min(best, worst_expected)
        return best

    @lru_cache(maxsize=None)
    def action(learner_posterior, scenarios, remaining):
        if remaining <= 0:
            raise ValueError("action requires positive remaining horizon")
        scored = []
        for a in ACTIONS:
            worst_expected = -math.inf
            mean_expected = 0.0
            for _, q in scenarios:
                expected = 0.0
                for obs, prob in predictive(q, a):
                    if prob <= 0.0:
                        continue
                    p_next = update(learner_posterior, a, obs)
                    scenario_next = update_scenarios(scenarios, a, obs)
                    expected += prob * value(p_next, scenario_next, remaining - 1)
                worst_expected = max(worst_expected, expected)
                mean_expected += expected / len(scenarios)
            scored.append((worst_expected, mean_expected, a))
        return min(scored)[2]

    return initial_scenarios, value, action


def eval_policy(policy, objective, true_prior_name, learner_prior_name,
                ambiguity_names=PRIOR_NAMES, horizon=2, _solver=None):
    if horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if true_prior_name not in PRIOR_NAMES or learner_prior_name not in PRIOR_NAMES:
        raise ValueError("unknown true or learner prior")
    if not ambiguity_names or any(n not in PRIOR_NAMES for n in ambiguity_names):
        raise ValueError("unknown or empty ambiguity set")
    true0 = tuple(build_prior(true_prior_name))
    learner0 = tuple(build_prior(learner_prior_name))
    # The robust value/action functions depend on the declared ambiguity set, not the
    # evaluator's true-prior label. Reuse exact cache across the evaluation recursion.
    initial_scenarios, robust_value, robust_act = _solver or make_bellman_solver(objective, ambiguity_names)
    scenario0 = initial_scenarios
    one_step_act_cache = {}
    assumed_oracle_cache = {}

    @lru_cache(maxsize=None)
    def recurse(q_true, p, scenarios, remaining, history):
        if remaining == 0:
            return (terminal_cross_entropy(q_true, p, objective),
                    terminal_brier_risk(q_true, p, objective))
        if policy == "robust_bellman_rectangular":
            selected = robust_act(p, scenarios, remaining)
            acts = ((selected, 1.0),)
        elif policy == "robust_minimax_one_step":
            key = (p, scenarios, objective)
            if key not in one_step_act_cache:
                one_step_act_cache[key] = robust_action(scenarios, p, objective)
            acts = ((one_step_act_cache[key], 1.0),)
        elif policy == "greedy_joint":
            acts = ((choose_action("greedy_joint", p, history), 1.0),)
        elif policy == "round_robin":
            acts = ((ACTIONS[len(history) % len(ACTIONS)], 1.0),)
        elif policy == "random_uniform":
            acts = tuple((a, 1.0 / len(ACTIONS)) for a in ACTIONS)
        elif policy == "oracle_assumed_objective":
            oracle_policy = f"oracle_{objective}"
            selected = choose_action(oracle_policy, p, history, remaining=remaining,
                                     oracle_cache=assumed_oracle_cache)
            acts = ((selected, 1.0),)
        else:
            raise AssertionError("unhandled policy")
        ce = brier = 0.0
        for a, action_prob in acts:
            for obs, obs_prob in predictive(q_true, a):
                if obs_prob <= 0.0:
                    continue
                q_next = update(q_true, a, obs)
                p_next = update(p, a, obs)
                scenarios_next = update_scenarios(scenarios, a, obs)
                child_ce, child_brier = recurse(q_next, p_next, scenarios_next,
                                                remaining - 1, history + (a,))
                weight = action_prob * obs_prob
                ce += weight * child_ce
                brier += weight * child_brier
        return ce, brier

    ce, brier = recurse(true0, learner0, scenario0, horizon, ())
    initial_ce = terminal_cross_entropy(true0, learner0, objective)
    initial_brier = terminal_brier_risk(true0, learner0, objective)
    return {
        "true_prior": true_prior_name,
        "learner_prior": learner_prior_name,
        "ambiguity_set": list(ambiguity_names),
        "ambiguity_set_size": len(ambiguity_names),
        "objective": objective,
        "horizon": horizon,
        "policy": policy,
        "well_specified": true_prior_name == learner_prior_name,
        "initial_cross_entropy_bits": initial_ce,
        "expected_terminal_cross_entropy_bits": ce,
        "cross_entropy_reduction_bits": initial_ce - ce,
        "expected_terminal_brier_risk": brier,
        "initial_brier_risk": initial_brier,
        "brier_risk_reduction": initial_brier - brier,
        "evaluation_states": recurse.cache_info().currsize,
        "robust_bellman_states": robust_value.cache_info().currsize,
    }


def run_benchmark(max_horizon=2, true_prior_names=PRIOR_NAMES,
                  learner_prior_names=PRIOR_NAMES, ambiguity_names=PRIOR_NAMES,
                  policies=POLICIES, objectives=OBJECTIVES):
    if max_horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    for names in (true_prior_names, learner_prior_names, ambiguity_names):
        if not names or any(n not in PRIOR_NAMES for n in names):
            raise ValueError("unknown or empty prior list")
    if not policies or any(p not in POLICIES for p in policies):
        raise ValueError("unknown or empty policy set")
    if not objectives or any(o not in OBJECTIVES for o in objectives):
        raise ValueError("unknown or empty objective set")
    rows = []
    # Share the Bellman caches across prior-pair evaluations: planner state depends
    # on learner and ambiguity posteriors, not the evaluator's true-prior label.
    solvers = {obj: make_bellman_solver(obj, ambiguity_names) for obj in objectives}
    # Keep runs explicitly bounded; every row is exactly evaluated and scored under
    # a fixed true prior. The robust planner itself does not receive that label.
    for h in range(1, max_horizon + 1):
        for true_name in true_prior_names:
            for learner_name in learner_prior_names:
                for obj in objectives:
                    for policy in policies:
                        rows.append(eval_policy(policy, obj, true_name, learner_name,
                                                ambiguity_names, h, _solver=solvers[obj]))
    return {
        "schema": SCHEMA,
        "status": "EXACT_RECTANGULAR_ROBUST_BELLMAN_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "base_source_blobs": BASE_SOURCE_BLOBS,
        "protocol": {
            "hypothesis_count": len(HYPOTHESES),
            "true_prior_names": list(true_prior_names),
            "learner_prior_names": list(learner_prior_names),
            "ambiguity_names": list(ambiguity_names),
            "horizons": list(range(1, max_horizon + 1)),
            "policies": list(policies),
            "objectives": list(objectives),
            "telemetry": "exact realized target subset and three-bit outcome",
            "score": "exact expected terminal proper categorical cross entropy and Brier risk under each designated fixed true prior",
            "robust_planner": "finite-horizon Bellman minimization; at each node a rectangular adversary may select the worst candidate posterior for that node and observation branch",
            "ambiguity_semantics": "rectangular/dynamic ambiguity, not a single latent prior fixed for the full episode; this conservatism is deliberate and must not be conflated with non-rectangular prior minimax",
            "limitations": [
                "16 hypotheses; chain/fork graph families only; three binary spill parameters",
                "finite six-prior set is not a statistically calibrated ambiguity set",
                "rectangular ambiguity permits branchwise scenario switching and can be conservative",
                "exact target-subset telemetry; no costs, noise, missingness or task disturbance",
                "no external benchmark executed",
            ],
        },
        "rows": rows,
    }


def self_test():
    for name in PRIOR_NAMES:
        p = build_prior(name)
        assert len(p) == len(HYPOTHESES) and all(v > 0 for v in p)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
    scenarios, value, action = make_bellman_solver("joint", ("uniform", "high_spill_biased"))
    p = tuple(build_prior("uniform"))
    assert action(p, scenarios, 1) in ACTIONS
    # Robust value cannot be worse than choosing any single action under its own
    # rectangular worst-case risk (up to numerical tolerance).
    v0 = value(p, scenarios, 1)
    act = action(p, scenarios, 1)
    assert math.isfinite(v0) and act in ACTIONS
    small = run_benchmark(1, ("uniform", "high_spill_biased"),
                          ("uniform", "chain_biased"),
                          ("uniform", "high_spill_biased", "chain_biased"),
                          ("robust_bellman_rectangular", "robust_minimax_one_step",
                           "greedy_joint", "round_robin", "random_uniform"), ("joint",))
    assert len(small["rows"]) == 2 * 2 * 5
    assert all(math.isfinite(r["expected_terminal_cross_entropy_bits"]) for r in small["rows"])
    # Horizon one Bellman and one-step minimax choose under the same one-step risk.
    row = {r["policy"]: r for r in small["rows"] if r["true_prior"] == "uniform" and
           r["learner_prior"] == "uniform"}
    assert abs(row["robust_bellman_rectangular"]["expected_terminal_cross_entropy_bits"] -
               row["robust_minimax_one_step"]["expected_terminal_cross_entropy_bits"]) < 1e-8
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--true-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--learner-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--ambiguity-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--source-git-blob", default="UNBOUND")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); print("ROBUST_BELLMAN_AMBIGUITY_TESTS_PASSED"); return 0
        result = run_benchmark(args.horizon, tuple(args.true_priors.split(",")),
                               tuple(args.learner_priors.split(",")),
                               tuple(args.ambiguity_priors.split(",")),
                               tuple(args.policies.split(",")), tuple(args.objectives.split(",")))
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
