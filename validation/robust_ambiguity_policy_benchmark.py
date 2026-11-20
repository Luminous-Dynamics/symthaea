#!/usr/bin/env python3
"""Exact evaluation of a deployable ambiguity-set policy for active causal probing.

Policy action choices use only the learner posterior plus a declared ambiguity set
of candidate true-world priors. For each action, it computes the worst expected
one-step terminal cross-entropy over all currently plausible world posteriors, and
chooses the action minimizing that worst-case score. It never receives the actual
true prior as an action-selection input. Exact expected scoring still integrates
observations under the designated true prior for evaluation.

Research pilot only: 16 hypotheses, chain/fork only, three action-specific spill
parameters, exact target-subset telemetry; no external benchmark qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from objective_matched_oracle_benchmark import (
    ACTIONS, HYPOTHESES, OBJECTIVES, POLICIES, choose_action, predictive, update,
)
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from prior_misspecification_benchmark import (
    terminal_cross_entropy, terminal_brier_risk,
)

SCHEMA = "symthaea-robust-ambiguity-policy-benchmark-v1"
SOURCE_BASE_BLOB = "541c31521e1b9e49dfb84c0fa265e877ca9ea632"
DEPLOYABLE_POLICIES = ("robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform", "oracle_joint_assumed")


def update_scenarios(scenarios, action, obs):
    """Bayes-update each plausible prior separately, dropping impossible worlds."""
    from objective_matched_oracle_benchmark import likelihood
    next_scenarios = []
    for name, posterior in scenarios:
        likelihoods = tuple(w * likelihood(h, action, obs) for h, w in zip(HYPOTHESES, posterior))
        total = sum(likelihoods)
        if total > 0.0:
            next_scenarios.append((name, tuple(v / total for v in likelihoods)))
    if not next_scenarios:
        raise AssertionError("all ambiguity scenarios eliminated after a possible true observation")
    return tuple(next_scenarios)


def robust_action(scenarios, learner_posterior, objective):
    """Minimize the worst candidate-world expected cross-entropy after one probe.

    This is a deployable one-step robust heuristic. The action rule sees the finite
    ambiguity set, not the true-prior label. Ties are broken by mean scenario risk
    and then action index for deterministic reproducibility.
    """
    candidates = []
    for action in ACTIONS:
        scenario_scores = []
        for _, q in scenarios:
            expected = 0.0
            for obs, prob in predictive(q, action):
                if prob <= 0.0:
                    continue
                q_next = update(q, action, obs)
                p_next = update(learner_posterior, action, obs)
                expected += prob * terminal_cross_entropy(q_next, p_next, objective)
            scenario_scores.append(expected)
        candidates.append((max(scenario_scores), sum(scenario_scores) / len(scenario_scores), action))
    return min(candidates)[2]


def eval_policy(policy, objective, true_prior_name, learner_prior_name,
                ambiguity_names=PRIOR_NAMES, horizon=2):
    """Exact expected terminal cross-entropy/Brier for one true/learner pair."""
    if horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    if policy not in DEPLOYABLE_POLICIES:
        raise ValueError(f"unknown deployable policy: {policy}")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if true_prior_name not in PRIOR_NAMES or learner_prior_name not in PRIOR_NAMES:
        raise ValueError("unknown true or learner prior")
    if not ambiguity_names or any(n not in PRIOR_NAMES for n in ambiguity_names):
        raise ValueError("ambiguity set must contain known priors")
    q0 = tuple(build_prior(true_prior_name))
    p0 = tuple(build_prior(learner_prior_name))
    scenarios0 = tuple((name, tuple(build_prior(name))) for name in ambiguity_names)
    oracle_cache = {}

    @lru_cache(maxsize=None)
    def recurse(q_true, p, scenarios, remaining, history):
        if remaining == 0:
            return (terminal_cross_entropy(q_true, p, objective),
                    terminal_brier_risk(q_true, p, objective))
        if policy == "robust_minimax_one_step":
            action = robust_action(scenarios, p, objective)
            actions = ((action, 1.0),)
        elif policy == "random_uniform":
            actions = tuple((a, 1.0 / len(ACTIONS)) for a in ACTIONS)
        elif policy == "round_robin":
            actions = ((ACTIONS[len(history) % len(ACTIONS)], 1.0),)
        elif policy == "greedy_joint":
            actions = ((choose_action("greedy_joint", p, history), 1.0),)
        elif policy == "oracle_joint_assumed":
            action = choose_action("oracle_joint", p, history, remaining=remaining, oracle_cache=oracle_cache)
            actions = ((action, 1.0),)
        else:
            raise AssertionError("unhandled policy")
        ce = brier = 0.0
        for action, action_prob in actions:
            for obs, obs_prob in predictive(q_true, action):
                if obs_prob <= 0.0:
                    continue
                q_next = update(q_true, action, obs)
                p_next = update(p, action, obs)
                scenarios_next = update_scenarios(scenarios, action, obs)
                child_ce, child_brier = recurse(q_next, p_next, scenarios_next,
                                                remaining - 1, history + (action,))
                weight = action_prob * obs_prob
                ce += weight * child_ce
                brier += weight * child_brier
        return ce, brier

    ce, brier = recurse(q0, p0, scenarios0, horizon, ())
    initial_ce = terminal_cross_entropy(q0, p0, objective)
    initial_brier = terminal_brier_risk(q0, p0, objective)
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
        "exact_recursive_states": recurse.cache_info().currsize,
        "oracle_cache_entries": len(oracle_cache),
    }


def run_benchmark(max_horizon=2, true_prior_names=PRIOR_NAMES, learner_prior_names=PRIOR_NAMES,
                  ambiguity_names=PRIOR_NAMES, policies=DEPLOYABLE_POLICIES, objectives=OBJECTIVES):
    if max_horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3")
    for names in (true_prior_names, learner_prior_names, ambiguity_names):
        if not names or any(n not in PRIOR_NAMES for n in names):
            raise ValueError("unknown or empty prior names")
    if not policies or any(p not in DEPLOYABLE_POLICIES for p in policies):
        raise ValueError("unknown or empty policy set")
    if not objectives or any(o not in OBJECTIVES for o in objectives):
        raise ValueError("unknown or empty objective set")
    rows = []
    for h in range(1, max_horizon + 1):
        for true_name in true_prior_names:
            for learner_name in learner_prior_names:
                for objective in objectives:
                    for policy in policies:
                        rows.append(eval_policy(policy, objective, true_name, learner_name,
                                                ambiguity_names, h))
    return {
        "schema": SCHEMA,
        "status": "EXACT_DEPLOYABLE_ROBUST_POLICY_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "source_base": {"path": "validation/prior_misspecification_benchmark.py", "git_blob": SOURCE_BASE_BLOB},
        "protocol": {
            "hypotheses": len(HYPOTHESES),
            "true_priors": list(true_prior_names),
            "learner_priors": list(learner_prior_names),
            "ambiguity_set": list(ambiguity_names),
            "ambiguity_set_size": len(ambiguity_names),
            "horizons": list(range(1, max_horizon + 1)),
            "objectives": list(objectives),
            "policies": list(policies),
            "robust_policy": "For each action, calculate expected next-step terminal cross entropy under every still-plausible world posterior; select action minimizing worst-case scenario score (mean risk then action index break ties). After every observation, Bayesian-update learner and all non-zero-likelihood scenario posteriors.",
            "evaluation_world": "expected scores are integrated under the designated true prior posterior; true_prior_name is only given to the evaluator and never supplied to robust_action",
            "telemetry": "exact realized intervention subset and three-bit outcome",
            "limitations": [
                "16 hypotheses, chain/fork only, three binary action-spill parameters",
                "ambiguity set is a finite set of six prior candidates, not a calibrated confidence set",
                "one-step minimax action rule, not a multi-step robust Bellman-optimal policy",
                "exact realized-subset telemetry; no action cost/noise/disturbance",
                "no external published baseline is run",
            ],
        },
        "rows": rows,
    }


def self_test():
    for name in PRIOR_NAMES:
        p = build_prior(name)
        assert len(p) == len(HYPOTHESES) and all(x > 0 for x in p)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
    scenarios = tuple((name, tuple(build_prior(name))) for name in PRIOR_NAMES)
    p = tuple(build_prior("uniform"))
    assert robust_action(scenarios, p, "joint") in ACTIONS
    rows = run_benchmark(1, ("uniform", "high_spill_biased"), ("uniform", "chain_biased"),
                         ambiguity_names=("uniform", "high_spill_biased", "chain_biased"),
                         policies=("robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform"),
                         objectives=("joint",))
    assert len(rows["rows"]) == 2 * 2 * 4
    assert all(math.isfinite(r["expected_terminal_cross_entropy_bits"]) for r in rows["rows"])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--true-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--learner-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--ambiguity-priors", default=",".join(PRIOR_NAMES))
    ap.add_argument("--policies", default=",".join(DEPLOYABLE_POLICIES))
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--source-git-blob", default="UNBOUND")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); print("ROBUST_AMBIGUITY_POLICY_TESTS_PASSED"); return 0
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
    except (ValueError, AssertionError, OSError, RuntimeError, RecursionError) as e:
        print(f"NOT QUALIFIED: {e}"); return 2

if __name__ == "__main__":
    raise SystemExit(main())
