#!/usr/bin/env python3
"""Exact two-objective causal-probe benchmark under uncertain interventions.

Compares graph-identification EIG against a mixed objective that also values
separating the candidate SCM mixture from a held-out-family reference mixture.
Standard library only; a small falsification pilot, not an external leaderboard.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from intervention_reliability_audit import SCM, Action, entropy, outcome_distribution, predictive, update, graph_posterior, weighted_auc

SCHEMA = "symthaea-causal-probe-pareto-v1"
POLICIES = (("eig_only", 0.0), ("balanced_0_5", 0.5), ("falsification_2_0", 2.0))
BUDGET = 2


def fixtures():
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    # Reference OOD family has collider AND/OR mechanisms; held-out test is XOR.
    collider_and = SCM("reference_collider_and", ((), (), (0, 1)), (0, 0, 0b1000))
    collider_or = SCM("reference_collider_or", ((), (), (0, 1)), (0, 0, 0b1110))
    collider_xor = SCM("heldout_collider_xor", ((), (), (0, 1)), (0, 0, 0b0110))
    actions = tuple(Action(2 * node + value, node, value)
                    for node in range(3) for value in (0, 1))
    return (chain, fork), (collider_and, collider_or), collider_xor, actions


def distribution_mixture(models, weights, action, noise, q):
    if not models or len(models) != len(weights):
        raise ValueError("mixture model/weight lengths must match and be nonempty")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("mixture weights must be nonnegative")
    weight_total = sum(weights)
    if weight_total <= 0.0:
        raise ValueError("mixture weights must have positive total")
    out = {}
    for model, weight in zip(models, weights):
        normalized_weight = weight / weight_total
        for y, py in outcome_distribution(model, action, noise, q).items():
            out[y] = out.get(y, 0.0) + normalized_weight * py
    total = sum(out.values())
    if total <= 0.0:
        raise ValueError("mixture predictive has zero mass")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"mixture predictive failed to normalize: {total}")
    return out


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, range [0,1]."""
    keys = set(p) | set(q)
    value = 0.0
    for key in keys:
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0:
            value += 0.5 * a * math.log2(a / m)
        if b > 0.0:
            value += 0.5 * b * math.log2(b / m)
    return value


def action_components(models, posterior, ood_reference, action, noise, assumed_q):
    pred = distribution_mixture(models, posterior, action, noise, assumed_q)
    graph_h_before = entropy(graph_posterior(models, posterior).values())
    expected_h_after = 0.0
    for y, py in pred.items():
        post = update(models, posterior, action, y, noise, assumed_q)
        expected_h_after += py * entropy(graph_posterior(models, post).values())
    eig = max(0.0, graph_h_before - expected_h_after)
    ood_pred = distribution_mixture(ood_reference, [1.0] * len(ood_reference), action, noise, assumed_q)
    divergence = js_divergence(pred, ood_pred)
    return eig, divergence


def choose_action(policy_name, beta, models, posterior, ood_reference, remaining, noise, assumed_q):
    if not remaining:
        raise ValueError("no remaining actions")
    def score(action):
        eig, divergence = action_components(models, posterior, ood_reference, action, noise, assumed_q)
        return (-(eig + beta * divergence), action.identifier)
    return min(remaining, key=score)


def enumerate_policy_paths(models, world, ood_reference, actions, policy_name, beta,
                           noise, actual_q, assumed_q, budget=BUDGET):
    """Exact adaptive path enumeration; impossible learner paths terminate unresolved."""
    initial = tuple(1.0 / len(models) for _ in models)
    terminals = []
    def visit(posterior, remaining, depth, path_probability, used, cumulative_surprise, impossible):
        if depth <= 0 or not remaining or impossible:
            terminals.append({"p": path_probability, "posterior": posterior, "used": used,
                              "surprise": cumulative_surprise, "impossible": impossible})
            return
        action = choose_action(policy_name, beta, models, posterior, ood_reference,
                               remaining, noise, assumed_q)
        learner_pred = distribution_mixture(models, posterior, action, noise, assumed_q)
        actual_pred = outcome_distribution(world, action, noise, actual_q)
        next_remaining = tuple(a for a in remaining if a.identifier != action.identifier)
        for outcome, p_actual in actual_pred.items():
            p_learner = learner_pred.get(outcome, 0.0)
            surprise = math.inf if p_learner <= 0.0 else -math.log2(p_learner)
            total_surprise = cumulative_surprise + surprise
            new_path_p = path_probability * p_actual
            if p_learner <= 0.0:
                terminals.append({"p": new_path_p, "posterior": posterior, "used": used + 1,
                                  "surprise": math.inf, "impossible": True})
            else:
                next_posterior = update(models, posterior, action, outcome, noise, assumed_q)
                visit(next_posterior, next_remaining, depth - 1, new_path_p, used + 1,
                      total_surprise, False)
    visit(initial, actions, budget, 1.0, 0, 0.0, False)
    mass = sum(t["p"] for t in terminals)
    if not math.isclose(mass, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError(f"terminal path mass does not normalize: {mass}")
    return terminals


def score_policy(models, ood_reference, ood_test, actions, policy_name, beta,
                 noise, actual_q, assumed_q):
    # In-class graph utility; impossible paths receive zero utility credit and are
    # explicitly counted as unresolved rather than inheriting stale confidence.
    graph_information = 0.0
    zero_support_mass = 0.0
    expected_actions = 0.0
    valid_mass = 0.0
    for world in models:
        terms = enumerate_policy_paths(models, world, ood_reference, actions, policy_name,
                                       beta, noise, actual_q, assumed_q)
        for term in terms:
            weight = 0.5 * term["p"]
            expected_actions += weight * term["used"]
            if term["impossible"]:
                zero_support_mass += weight
            else:
                valid_mass += weight
                graph_information += weight * (1.0 - entropy(graph_posterior(models, term["posterior"]).values()))
    in_distribution = []
    for world in models:
        terms = enumerate_policy_paths(models, world, ood_reference, actions, policy_name,
                                       beta, noise, actual_q, assumed_q)
        in_distribution.extend((0.5 * t["p"], t["surprise"]) for t in terms)
    ood_terms = enumerate_policy_paths(models, ood_test, ood_reference, actions, policy_name,
                                       beta, noise, actual_q, assumed_q)
    ood_distribution = [(t["p"], t["surprise"]) for t in ood_terms]
    auc = weighted_auc(ood_distribution, in_distribution)
    return {
        "policy": policy_name,
        "ood_weight_beta": beta,
        "mechanism_noise": noise,
        "actual_compliance": actual_q,
        "assumed_compliance": assumed_q,
        "usable_in_class_graph_information_bits": graph_information,
        "in_class_valid_path_mass": valid_mass,
        "in_class_zero_support_mass": zero_support_mass,
        "ood_cumulative_surprise_auc": auc,
        "expected_interventions_in_class": expected_actions,
    }


def run(source_git_blob="UNBOUND"):
    models, ood_reference, ood_test, actions = fixtures()
    rows = []
    for noise in (0.10, 0.25):
        for actual_q in (0.50, 0.75, 1.00):
            for assumed_q in sorted({actual_q, 1.0}):
                for policy_name, beta in POLICIES:
                    rows.append(score_policy(models, ood_reference, ood_test, actions,
                                             policy_name, beta, noise, actual_q, assumed_q))
    return {
        "schema": SCHEMA,
        "status": "EXACT_LOCAL_PARETO_PILOT_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "protocol": {
            "nodes": 3, "adaptive_budget": BUDGET,
            "candidate_models": [m.name for m in models],
            "ood_reference_family": [m.name for m in ood_reference],
            "held_out_ood_test": ood_test.name,
            "actual_compliance_grid": [0.50, 0.75, 1.00],
            "mechanism_noise_grid": [0.10, 0.25],
            "policies": {name: {"formula": "EIG + beta * JSD(P_candidate(outcome), P_OOD-reference(outcome))", "beta": beta}
                         for name, beta in POLICIES},
            "reference_warning": "The policy has access to AND/OR collider reference models and is evaluated on a held-out XOR collider mechanism. This is a small within-family generalization test, not open-world generalization.",
            "budget_warning": "Two adaptive probes; no repeated exact action. Fixed intervention order is not the comparator in this experiment.",
        },
        "metrics": {
            "usable_in_class_graph_information_bits": "expected one-bit initial chain/fork graph entropy reduction; unresolved zero-support paths get zero credit",
            "in_class_valid_path_mass": "probability mass for which the candidate learner has nonzero support throughout the adaptive path",
            "in_class_zero_support_mass": "true in-class path mass on observations assigned exactly zero probability by the learner",
            "ood_cumulative_surprise_auc": "exact weighted ROC AUC ranking held-out XOR collider paths above an equal-prior chain/fork mixture; 0.5 is chance",
            "expected_interventions_in_class": "expected number of probes in the two-probe budget",
        },
        "rows": rows,
    }


def self_test():
    models, refs, test, actions = fixtures()
    for q in (0.0, 0.5, 0.75, 1.0):
        for noise in (0.0, 0.1, 0.25):
            for world in (*models, *refs, test):
                for action in actions:
                    p = outcome_distribution(world, action, noise, q)
                    assert math.isclose(sum(p.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    for action in actions:
        eig, jsd = action_components(models, (0.5, 0.5), refs, action, 0.1, 1.0)
        assert eig >= -1e-12 and 0.0 <= jsd <= 1.0 + 1e-12
    report = run("self-test")
    assert len(report["rows"]) == 30  # 2 noises * 5 (q,assumption) cells * 3 policies
    assert all(0.0 <= r["ood_cumulative_surprise_auc"] <= 1.0 for r in report["rows"])
    assert all(0.0 <= r["in_class_zero_support_mass"] <= 1.0 for r in report["rows"])
    for q in (0.5, 0.75):
        for noise in (0.1, 0.25):
            bad = score_policy(models, refs, test, actions, "eig_only", 0.0,
                               noise, q, 1.0)
            assert bad["in_class_zero_support_mass"] > 0.0
            aware = score_policy(models, refs, test, actions, "eig_only", 0.0,
                                 noise, q, q)
            assert aware["in_class_zero_support_mass"] < 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("CAUSAL_PROBE_PARETO_SELF_TESTS_PASSED")
            return 0
        encoded = (json.dumps(run(args.source_git_blob), sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else:
            print(encoded.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
