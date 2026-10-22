#!/usr/bin/env python3
"""Exact joint graph/actuator inference for active causal discovery.

This pilot treats uncertain action compliance as a latent nuisance parameter
rather than assuming hard interventions always execute exactly as requested.
It compares graph-only, joint graph+actuator, falsification-aware, known-q oracle,
and naive perfect-actuator policies on enumerable SCMs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from intervention_reliability_audit import SCM, Action, entropy, outcome_distribution, weighted_auc

SCHEMA = "symthaea-joint-graph-actuator-inference-v1"
Q_GRID = (0.50, 0.75, 1.00)
BUDGET = 2
POLICIES = (
    ("joint_graph_eig", "joint", "graph", 0.0),
    ("joint_graph_actuator_balanced", "joint", "balanced", 0.0),
    ("joint_graph_actuator_falsification", "joint", "balanced", 0.5),
    ("oracle_known_compliance", "oracle", "graph", 0.0),
    ("naive_perfect_compliance", "naive", "graph", 0.0),
)

def fixtures():
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    ref_and = SCM("reference_collider_and", ((), (), (0, 1)), (0, 0, 0b1000))
    ref_or = SCM("reference_collider_or", ((), (), (0, 1)), (0, 0, 0b1110))
    ood = {
        "collider_xor": SCM("heldout_collider_xor", ((), (), (0, 1)), (0, 0, 0b0110)),
        "reverse_chain": SCM("heldout_reverse_chain", ((1,), (2,), ()), (0b10, 0b10, 0)),
    }
    actions = tuple(Action(2 * node + value, node, value)
                    for node in range(3) for value in (0, 1))
    return (chain, fork), (ref_and, ref_or), ood, actions

def h_entropy(posterior):
    return entropy(posterior)

def graph_marginal(models, q_values, posterior):
    result = [0.0] * len(models)
    for w, (g, _q) in zip(posterior, hypothesis_index(models, q_values)):
        result[g] += w
    return result

def q_marginal(models, q_values, posterior):
    result = [0.0] * len(q_values)
    for w, (_g, qi) in zip(posterior, hypothesis_index(models, q_values)):
        result[qi] += w
    return result

def hypothesis_index(models, q_values):
    return tuple((g, qi) for g in range(len(models)) for qi in range(len(q_values)))

def mixture_predictive(models, q_values, posterior, action, noise):
    indices = hypothesis_index(models, q_values)
    if len(indices) != len(posterior):
        raise ValueError("posterior length does not match graph/compliance hypotheses")
    out = {}
    for weight, (gi, qi) in zip(posterior, indices):
        if weight == 0.0:
            continue
        distribution = outcome_distribution(models[gi], action, noise, q_values[qi])
        for y, p in distribution.items():
            out[y] = out.get(y, 0.0) + weight * p
    total = sum(out.values())
    if total <= 0.0:
        raise ValueError("joint predictive distribution has zero mass")
    out = {y: p / total for y, p in out.items()}
    if not math.isclose(sum(out.values()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("joint predictive does not normalize")
    return out

def update_joint(models, q_values, posterior, action, outcome, noise):
    new = []
    for weight, (gi, qi) in zip(posterior, hypothesis_index(models, q_values)):
        like = outcome_distribution(models[gi], action, noise, q_values[qi]).get(outcome, 0.0)
        new.append(weight * like)
    total = sum(new)
    if total <= 0.0:
        raise ValueError("evidence outside all graph/compliance hypotheses")
    normalized = tuple(x / total for x in new)
    if not math.isclose(sum(normalized), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("joint posterior failed to normalize")
    return normalized

def js_divergence(p, q):
    total = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0: total += 0.5 * a * math.log2(a / m)
        if b > 0.0: total += 0.5 * b * math.log2(b / m)
    return total

def action_components(models, q_values, posterior, reference_models, action, noise):
    candidate = mixture_predictive(models, q_values, posterior, action, noise)
    gp = graph_marginal(models, q_values, posterior)
    qp = q_marginal(models, q_values, posterior)
    hp = h_entropy(posterior)
    hg = entropy(gp)
    hq = entropy(qp)
    expected_joint = expected_graph = expected_q = 0.0
    for outcome, p in candidate.items():
        post = update_joint(models, q_values, posterior, action, outcome, noise)
        expected_joint += p * entropy(post)
        expected_graph += p * entropy(graph_marginal(models, q_values, post))
        expected_q += p * entropy(q_marginal(models, q_values, post))
    eig_joint = max(0.0, hp - expected_joint)
    eig_graph = max(0.0, hg - expected_graph)
    eig_q = max(0.0, hq - expected_q)
    if reference_models:
        ref_distributions = []
        for reference in reference_models:
            for q in q_values:
                ref_distributions.append(outcome_distribution(reference, action, noise, q))
        ref = {}
        weight = 1.0 / len(ref_distributions)
        for dist in ref_distributions:
            for y, p in dist.items():
                ref[y] = ref.get(y, 0.0) + weight * p
        divergence = js_divergence(candidate, ref)
    else:
        divergence = 0.0
    return {"eig_joint": eig_joint, "eig_graph": eig_graph, "eig_q": eig_q, "jsd_reference": divergence}

def choose_action(policy_kind, models, q_values, posterior, reference, actions, noise):
    def score(action):
        c = action_components(models, q_values, posterior, reference, action, noise)
        if policy_kind == "graph":
            objective = c["eig_graph"]
        elif policy_kind == "joint":
            objective = c["eig_joint"]
        elif policy_kind == "balanced":
            objective = c["eig_graph"] + 0.5 * c["eig_q"]
        elif policy_kind == "balanced_falsification":
            objective = c["eig_graph"] + 0.5 * c["eig_q"] + 0.5 * c["jsd_reference"]
        else:
            raise ValueError(f"unknown policy kind {policy_kind}")
        return (-objective, action.identifier)
    return min(actions, key=score)

def enumerate_paths(models, q_values, world, reference, actions, policy_kind,
                    noise, actual_q, budget=BUDGET):
    posterior = tuple([1.0 / (len(models) * len(q_values))] * (len(models) * len(q_values)))
    terminals = []
    def visit(post, remaining, depth, path_p, used, surprise, impossible):
        if depth <= 0 or not remaining or impossible:
            terminals.append({"probability": path_p, "posterior": post, "used": used,
                              "surprise": surprise, "impossible": impossible})
            return
        action = choose_action(policy_kind, models, q_values, post, reference, remaining, noise)
        predicted = mixture_predictive(models, q_values, post, action, noise)
        actual = outcome_distribution(world, action, noise, actual_q)
        after = tuple(a for a in remaining if a.identifier != action.identifier)
        for outcome, true_p in actual.items():
            pred_p = predicted.get(outcome, 0.0)
            score = math.inf if pred_p <= 0.0 else surprise - math.log2(pred_p)
            joint_path_p = path_p * true_p
            if pred_p <= 0.0:
                terminals.append({"probability": joint_path_p, "posterior": post,
                                  "used": used + 1, "surprise": math.inf, "impossible": True})
            else:
                post2 = update_joint(models, q_values, post, action, outcome, noise)
                visit(post2, after, depth - 1, joint_path_p, used + 1, score, False)
    visit(posterior, actions, budget, 1.0, 0, 0.0, False)
    total = sum(t["probability"] for t in terminals)
    if not math.isclose(total, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError(f"path probability does not normalize: {total}")
    return terminals

def eval_policy(models, q_inferred, reference, actions, ood_world, policy_name,
                policy_kind, beta, noise, actual_q, assumed_mode):
    if assumed_mode == "joint":
        q_values = tuple(q_inferred)
        chooser = "balanced_falsification" if beta > 0.0 else policy_kind
    elif assumed_mode == "oracle":
        q_values = (actual_q,)
        chooser = "graph"
    elif assumed_mode == "naive":
        q_values = (1.0,)
        chooser = "graph"
    else:
        raise ValueError(assumed_mode)
    graph_information = q_information = zero_support = valid_mass = expected_actions = 0.0
    in_scores = []
    hq_initial = entropy([1.0 / len(q_values)] * len(q_values))
    for world in models:
        terms = enumerate_paths(models, q_values, world, reference, actions, chooser, noise, actual_q)
        for t in terms:
            mass = 0.5 * t["probability"]
            in_scores.append((mass, t["surprise"]))
            expected_actions += mass * t["used"]
            if t["impossible"]:
                zero_support += mass
                continue
            valid_mass += mass
            gp = graph_marginal(models, q_values, t["posterior"])
            qp = q_marginal(models, q_values, t["posterior"])
            graph_information += mass * (1.0 - entropy(gp))
            q_information += mass * (hq_initial - entropy(qp))
    ood_terms = enumerate_paths(models, q_values, ood_world, reference, actions, chooser, noise, actual_q)
    auc = weighted_auc([(t["probability"], t["surprise"]) for t in ood_terms], in_scores)
    return {
        "policy": policy_name, "out_of_class_world": ood_world.name,
        "mechanism_noise": noise, "actual_compliance": actual_q,
        "inferred_compliance_values": list(q_values), "intervention_budget": BUDGET,
        "ood_surprise_auc": auc, "graph_information_bits": graph_information,
        "compliance_information_bits": q_information,
        "in_class_valid_path_mass": valid_mass,
        "in_class_zero_support_mass": zero_support,
        "expected_interventions": expected_actions,
    }

def run(source_git_blob="UNBOUND"):
    models, references, ood_worlds, actions = fixtures()
    rows = []
    for actual_q in Q_GRID:
        for noise in (0.10, 0.25):
            for world_key, world in ood_worlds.items():
                for name, mode, objective, beta in POLICIES:
                    rows.append(eval_policy(models, Q_GRID, references, actions, world,
                                            name, objective, beta, noise, actual_q, mode))
    return {
        "schema": SCHEMA, "status": "EXACT_LOCAL_JOINT_POSTERIOR_PILOT_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "protocol": {
            "candidate_graphs": [m.name for m in models],
            "compliance_grid": list(Q_GRID), "actual_compliance_grid": list(Q_GRID),
            "reference_models": [m.name for m in references],
            "out_of_class_worlds": {k: {"name": v.name, "parents": v.parents, "truth_tables": v.truth_tables} for k,v in ood_worlds.items()},
            "noise_grid": [0.10, 0.25], "budget": BUDGET,
            "policies": {name: {"mode": mode, "objective": objective, "beta": beta} for name,mode,objective,beta in POLICIES},
            "posterior": "uniform over graph x compliance hypotheses; Bayes update with exact outcome likelihood",
            "note": "The compliance values are known grid candidates but the joint policies do not know which candidate is the true one. The oracle receives actual q; the naive baseline fixes q=1. The reference bank is informed/human-authored and small."
        },
        "metrics": {
            "ood_surprise_auc": "exact weighted AUC for cumulative predictive surprise; chance=0.5",
            "graph_information_bits": "expected reduction in initial one-bit graph uncertainty; impossible paths get zero credit",
            "compliance_information_bits": "expected reduction in compliance entropy for policies that infer q; baseline values for fixed-q policies are zero",
            "in_class_zero_support_mass": "true in-class mass on trajectories assigned zero probability by the learner",
            "in_class_valid_path_mass": "mass of paths for which the learner retains predictive support"
        }, "rows": rows
    }

def self_test():
    models, refs, ood, actions = fixtures()
    for q in Q_GRID:
        for noise in (0.0, 0.1, 0.25):
            for world in (*models, *refs, *ood.values()):
                for action in actions:
                    dist = outcome_distribution(world, action, noise, q)
                    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-12, rel_tol=0.0)
    for policy in ("graph", "joint", "balanced", "balanced_falsification"):
        posterior = tuple([1.0 / (len(models) * len(Q_GRID))] * (len(models) * len(Q_GRID)))
        values = action_components(models, Q_GRID, posterior, refs, actions[0], 0.1)
        assert all(v >= -1e-12 for v in values.values())
        assert 0.0 <= values["jsd_reference"] <= 1.0 + 1e-12
    res = run("self-test")
    assert len(res["rows"]) == len(Q_GRID) * 2 * len(ood) * len(POLICIES)
    assert all(0.0 <= row["ood_surprise_auc"] <= 1.0 for row in res["rows"])
    for row in res["rows"]:
        if row["policy"].startswith("joint"):
            assert row["in_class_zero_support_mass"] < 1e-12
    assert any(row["in_class_zero_support_mass"] > 0.0 for row in res["rows"] if row["policy"] == "naive_perfect_compliance")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--source-git-blob",default="UNBOUND")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("JOINT_GRAPH_ACTUATOR_INFERENCE_SELF_TESTS_PASSED"); return 0
        raw=(json.dumps(run(args.source_git_blob),sort_keys=True,separators=(",",":"))+"\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
        else: print(raw.decode(),end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__=="__main__": raise SystemExit(main())
