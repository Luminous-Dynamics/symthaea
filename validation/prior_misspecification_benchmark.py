#!/usr/bin/env python3
"""Exact prior-misspecification stress test for adaptive causal probing.

Unlike a prior-sensitivity test where the world and learner share the same prior,
this benchmark separates the true distribution over hypotheses from the learner's
assumed prior. Histories are generated under the true posterior, while policies act
and update using the learner posterior. Scores are proper log-loss/cross-entropy
risks for graph, action-kernel, and joint hypothesis predictions.

Research pilot only: 16 hypotheses, two graph families, exact realized-subset
telemetry, finite horizons at most three, and no external benchmark qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache

from objective_matched_oracle_benchmark import (
    HYPOTHESES, OBJECTIVES, POLICIES, ACTIONS,
    predictive, update, marginals, entropy, choose_action,
)
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior

SCHEMA = "symthaea-prior-misspecification-cross-entropy-v1"
BASE_SOURCE_BLOB = "f101346c2f37cfad4e029ae280e9a002548f309c"
HORIZONS = (1, 2, 3)
POLICY_SET = tuple(p for p in POLICIES if not p.startswith("oracle_")) + tuple(
    p for p in POLICIES if p.startswith("oracle_")
) + ("oracle_robust_cross_entropy",)


def cross_entropy(q, p):
    """Cross entropy H(q,p); infinity if p assigns zero to q-supported mass."""
    out = 0.0
    for truth, forecast in zip(q, p):
        if truth > 0.0:
            if forecast <= 0.0:
                return math.inf
            out -= truth * math.log2(forecast)
    return out


def score_distributions(q, p, objective):
    q_graph, q_kernel = marginals(q)
    p_graph, p_kernel = marginals(p)
    if objective == "graph":
        return q_graph, p_graph
    if objective == "kernel":
        return q_kernel, p_kernel
    if objective == "joint":
        return q, p
    raise ValueError(f"unknown objective {objective}")


def terminal_cross_entropy(q, p, objective):
    q_score, p_score = score_distributions(q, p, objective)
    return cross_entropy(q_score, p_score)


def terminal_brier_risk(q, p, objective):
    """Expected categorical Brier loss under true posterior q, forecast p."""
    q_score, p_score = score_distributions(q, p, objective)
    return 1.0 - 2.0 * sum(x * y for x, y in zip(q_score, p_score)) + sum(y * y for y in p_score)


def _next_states(q, p, action):
    """Yield observation, true probability, updated true posterior, learner posterior."""
    for obs, prob in predictive(q, action):
        if prob <= 0.0:
            continue
        yield obs, prob, update(q, action, obs), update(p, action, obs)


def _choose(policy, q, p, history, remaining, assumed_oracle_cache):
    if policy == "oracle_robust_cross_entropy":
        raise RuntimeError("robust oracle action is selected by its pair-state Bellman recursion")
    if policy == "random_uniform":
        return None
    return choose_action(policy, p, history, remaining=remaining, oracle_cache=assumed_oracle_cache)


def make_robust_oracle(objective):
    """Construct exact oracle minimizing true expected log loss.

    Diagnostic oracle receives both true posterior q and learner posterior p when
    selecting an action. It is not deployable when the true prior is unknown.
    """
    @lru_cache(maxsize=None)
    def value(q, p, remaining):
        if remaining == 0:
            return terminal_cross_entropy(q, p, objective)
        best = math.inf
        for action in ACTIONS:
            candidate = sum(
                prob * value(q_next, p_next, remaining - 1)
                for _, prob, q_next, p_next in _next_states(q, p, action)
            )
            if candidate < best:
                best = candidate
        return best

    def action(q, p, remaining):
        vals = []
        for a in ACTIONS:
            candidate = sum(prob * value(q_next, p_next, remaining - 1)
                            for _, prob, q_next, p_next in _next_states(q, p, a))
            vals.append((candidate, a))
        return min(vals)[1]
    return value, action


def make_scorer(policy, objective, caches):
    """Create a memoized exact evaluator for one prior pair/policy/score."""
    @lru_cache(maxsize=None)
    def score(q, p, remaining, history):
        if remaining == 0:
            return (terminal_cross_entropy(q, p, objective),
                    terminal_brier_risk(q, p, objective))
        if policy == "oracle_robust_cross_entropy":
            action = caches["robust"][objective][1](q, p, remaining)
            actions = ((action, 1.0),)
        elif policy == "random_uniform":
            actions = tuple((a, 1.0 / len(ACTIONS)) for a in ACTIONS)
        else:
            action = _choose(policy, q, p, history, remaining, caches["assumed_oracle"])
            actions = ((action, 1.0),)
        ce = 0.0
        brier = 0.0
        for action, action_prob in actions:
            for _, obs_prob, q_next, p_next in _next_states(q, p, action):
                child_ce, child_brier = score(q_next, p_next, remaining - 1,
                                              history + (action,))
                weight = action_prob * obs_prob
                ce += weight * child_ce
                brier += weight * child_brier
        return ce, brier
    return score


def run_benchmark(max_horizon=3, true_prior_names=PRIOR_NAMES, learner_prior_names=None, policies=POLICY_SET):
    if max_horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if learner_prior_names is None:
        learner_prior_names = true_prior_names
    if not true_prior_names or not learner_prior_names:
        raise ValueError("true and learner prior lists must not be empty")
    if any(name not in PRIOR_NAMES for name in tuple(true_prior_names) + tuple(learner_prior_names)):
        raise ValueError("unknown prior name")
    if any(policy not in POLICY_SET for policy in policies):
        raise ValueError("unknown policy")
    rows = []
    aggregate = {}
    all_prior_names = tuple(dict.fromkeys(tuple(true_prior_names) + tuple(learner_prior_names)))
    all_priors = {name: tuple(build_prior(name)) for name in all_prior_names}

    for true_name in true_prior_names:
        q0 = all_priors[true_name]
        for assumed_name in learner_prior_names:
            p0 = all_priors[assumed_name]
            initial = {obj: terminal_cross_entropy(q0, p0, obj) for obj in OBJECTIVES}
            initial_brier = {obj: terminal_brier_risk(q0, p0, obj) for obj in OBJECTIVES}
            robust_caches = {obj: make_robust_oracle(obj) for obj in OBJECTIVES}
            caches = {"robust": robust_caches, "assumed_oracle": {}}
            scorers = {(policy, objective): make_scorer(policy, objective, caches)
                       for policy in policies for objective in OBJECTIVES}
            for horizon in range(1, max_horizon + 1):
                robust_values = {
                    obj: robust_caches[obj][0](q0, p0, horizon)
                    for obj in OBJECTIVES
                }
                for policy in policies:
                    for objective in OBJECTIVES:
                        ce, brier = scorers[(policy, objective)](q0, p0, horizon, ())
                        excess = ce - robust_values[objective]
                        if excess < -1e-8:
                            raise AssertionError(
                                f"policy {policy} beats robust oracle on {true_name}/{assumed_name}/h{horizon}/{objective}: {excess}"
                            )
                        rows.append({
                            "true_prior": true_name,
                            "learner_prior": assumed_name,
                            "horizon": horizon,
                            "budget": horizon,
                            "policy": policy,
                            "score_objective": objective,
                            "initial_cross_entropy_bits": initial[objective],
                            "expected_terminal_cross_entropy_bits": ce,
                            "cross_entropy_reduction_bits": initial[objective] - ce,
                            "expected_terminal_brier_risk": brier,
                            "initial_brier_risk": initial_brier[objective],
                            "brier_risk_reduction": initial_brier[objective] - brier,
                            "robust_oracle_cross_entropy_bits": robust_values[objective],
                            "excess_cross_entropy_bits_vs_robust_oracle": max(0.0, excess),
                            "well_specified": true_name == assumed_name,
                        })
            for horizon in range(1, max_horizon + 1):
                for objective in OBJECTIVES:
                    group = [r for r in rows if r["true_prior"] == true_name and
                             r["learner_prior"] == assumed_name and r["horizon"] == horizon and
                             r["score_objective"] == objective]
                    best = min(group, key=lambda r: r["expected_terminal_cross_entropy_bits"])
                    aggregate[f"{true_name}/{assumed_name}/h{horizon}/{objective}"] = {
                        "policy_count": len(group),
                        "best_policy_by_expected_cross_entropy": best["policy"],
                        "best_terminal_cross_entropy_bits": best["expected_terminal_cross_entropy_bits"],
                        "initial_cross_entropy_bits": initial[objective],
                        "best_cross_entropy_reduction_bits": initial[objective] - best["expected_terminal_cross_entropy_bits"],
                        "robust_oracle_cross_entropy_bits": robust_caches[objective][0](q0, p0, horizon),
                    }

    mismatch_groups = [r for r in rows if not r["well_specified"]]
    correct_groups = [r for r in rows if r["well_specified"]]
    return {
        "schema": SCHEMA,
        "status": "EXACT_PRIOR_MISSPECIFICATION_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "base_source": {"git_blob": BASE_SOURCE_BLOB, "path": "validation/objective_matched_oracle_benchmark.py"},
        "protocol": {
            "hypothesis_count": len(HYPOTHESES),
            "true_prior_set": list(true_prior_names),
            "learner_prior_set": list(learner_prior_names),
            "pairings": len(true_prior_names) * len(learner_prior_names),
            "well_specified_pairings": sum(1 for a in true_prior_names for b in learner_prior_names if a == b),
            "misspecified_pairings": sum(1 for a in true_prior_names for b in learner_prior_names if a != b),
            "horizons": list(range(1, max_horizon + 1)),
            "policies": list(policies),
            "objectives": list(OBJECTIVES),
            "true_world_semantics": "the realized data sequence is integrated using the true posterior q",
            "learner_semantics": "actions and Bayesian updates use assumed posterior p",
            "observation": "exact realized target-subset telemetry plus three-bit outcome",
            "score": "expected terminal proper categorical log loss (cross entropy) and Brier risk for graph, kernel assignment, or joint hypothesis prediction",
            "oracle": "exact finite-horizon dynamic program over (true posterior, learner posterior); picks interventions with access to q and p, so it is a diagnostic upper bound and not an implementable unknown-prior policy",
            "limitations": [
                "Reduced 16-hypothesis game; only chain/fork graph hypotheses",
                "Six declared priors over one fixed model family; no novel graph/mechanism family",
                "Exact realized-subset telemetry assumed",
                "No real-data or published benchmark baseline executed",
                "A good proper score does not itself establish causal graph identification or external robustness",
            ],
        },
        "well_specified_rows": len(correct_groups),
        "misspecified_rows": len(mismatch_groups),
        "aggregates": aggregate,
        "rows": rows,
    }


def self_test():
    for name in PRIOR_NAMES:
        prior = build_prior(name)
        assert len(prior) == len(HYPOTHESES)
        assert all(p > 0 for p in prior)
        assert math.isclose(sum(prior), 1.0, abs_tol=1e-12)
    q = tuple(build_prior("uniform"))
    p = tuple(build_prior("chain_biased"))
    assert terminal_cross_entropy(q, p, "joint") >= entropy(q) - 1e-12
    assert abs(terminal_brier_risk(q, q, "joint") - (1.0 - sum(x*x for x in q))) < 1e-10
    out = run_benchmark(2, true_prior_names=("uniform", "graph_kernel_correlated"),
                        learner_prior_names=("uniform", "graph_kernel_correlated"))
    assert len(out["rows"]) == 2 * 2 * 2 * len(POLICY_SET) * len(OBJECTIVES)
    assert all(r["excess_cross_entropy_bits_vs_robust_oracle"] >= 0 for r in out["rows"])
    for row in out["rows"]:
        if row["policy"] == "oracle_robust_cross_entropy":
            assert abs(row["excess_cross_entropy_bits_vs_robust_oracle"]) < 1e-8
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--true-priors", default=",".join(PRIOR_NAMES))
    parser.add_argument("--learner-priors", default=",".join(PRIOR_NAMES))
    parser.add_argument("--policies", default=",".join(POLICY_SET))
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("PRIOR_MISSPECIFICATION_TESTS_PASSED"); return 0
        true_names = tuple(args.true_priors.split(","))
        learner_names = tuple(args.learner_priors.split(","))
        selected_policies = tuple(args.policies.split(",")) if args.policies else POLICY_SET
        result = run_benchmark(args.horizon, true_names, learner_names, selected_policies)
        result["requested_true_priors"] = list(true_names)
        result["requested_learner_priors"] = list(learner_names)
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
        print(f"NOT QUALIFIED: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
