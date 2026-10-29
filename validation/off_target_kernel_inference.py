#!/usr/bin/env python3
"""Exact graph/compliance/off-target inference for stochastic interventions.

An attempted do(X_i=v) can realize on the target alone, on target plus an
action-dependent collateral variable, or on no variable. The learner may infer
both target compliance q and spill probability r jointly with graph identity.
Standard-library-only finite-state research prototype, not a leaderboard method.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from intervention_reliability_audit import SCM, Action, entropy, weighted_auc

SCHEMA = "symthaea-off-target-kernel-inference-v1"
Q_GRID = (0.50, 0.75, 1.00)
R_GRID = (0.00, 0.25)
BUDGET = 2
REGIMES = (
    ("matched_clean", 0.10, 0.75, 0.00),
    ("matched_off_target", 0.10, 0.75, 0.25),
    ("weak_off_target", 0.10, 0.50, 0.25),
    ("noisy_off_target", 0.25, 0.75, 0.25),
)
POLICY_CONFIGS = (
    ("joint_graph_eig", "joint", "graph"),
    ("joint_graph_actuation_balanced", "joint", "balanced"),
    ("joint_state_eig", "joint", "joint"),
    ("clean_scalar_compliance", "clean", "graph"),
    ("naive_perfect_hard_do", "naive", "graph"),
    ("oracle_action_kernel", "oracle", "graph"),
)

def fixtures():
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    refs = (
        SCM("reference_collider_and", ((), (), (0, 1)), (0, 0, 0b1000)),
        SCM("reference_collider_or", ((), (), (0, 1)), (0, 0, 0b1110)),
    )
    ood = {
        "collider_xor": SCM("heldout_collider_xor", ((), (), (0, 1)), (0, 0, 0b0110)),
        "reverse_chain": SCM("heldout_reverse_chain", ((1,), (2,), ()), (0b10, 0b10, 0)),
    }
    actions = tuple(Action(2 * node + value, node, value)
                    for node in range(3) for value in (0, 1))
    return (chain, fork), refs, ood, actions

def worlds(n):
    return tuple(itertools.product((0, 1), repeat=n))

def mechanism_probability(model, assignment, node, noise):
    parent_assignment = 0
    for bit, parent in enumerate(model.parents[node]):
        parent_assignment |= (assignment[parent] & 1) << bit
    expected = (model.truth_tables[node] >> parent_assignment) & 1
    return (1.0 - noise) if assignment[node] == expected else noise

@lru_cache(maxsize=None)
def outcome_distribution_kernel(model, action, noise, q, r):
    """Exact P(y | attempted action, compliance q, spill conditional probability r).

    With probability 1-q the attempted action changes no variable. Otherwise,
    it sets the requested target to action.value. Conditional on success, with
    probability r it also sets the next cyclic variable (target+1 mod n) to that
    value. The action-dependent spill rule is explicit and intentionally simple.
    """
    n = len(model.parents)
    if not 0.0 <= noise < 0.5: raise ValueError("noise must be in [0, 0.5)")
    if not 0.0 <= q <= 1.0: raise ValueError("q must be in [0, 1]")
    if not 0.0 <= r <= 1.0: raise ValueError("r must be in [0, 1]")
    target = action.target
    collateral = (target + 1) % n
    subset_laws = (
        ((), 1.0 - q),
        ((target,), q * (1.0 - r)),
        ((target, collateral), q * r),
    )
    result = {}
    for y in worlds(n):
        probability = 0.0
        for intervened, subset_p in subset_laws:
            if subset_p <= 0.0: continue
            if any(y[node] != action.value for node in intervened):
                continue
            p = subset_p
            intervention_set = set(intervened)
            for node in range(n):
                if node not in intervention_set:
                    p *= mechanism_probability(model, y, node, noise)
            probability += p
        if probability > 0.0:
            result[y] = probability
    total = sum(result.values())
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError(f"stochastic-intervention distribution failed to normalize: {total}")
    return result

def hypothesis_index(models, q_values, r_values):
    return tuple((g, qi, ri) for g in range(len(models))
                 for qi in range(len(q_values)) for ri in range(len(r_values)))

def marginal(models, q_values, r_values, posterior, dimension):
    if dimension == "graph": size = len(models); idx = 0
    elif dimension == "q": size = len(q_values); idx = 1
    elif dimension == "r": size = len(r_values); idx = 2
    else: raise ValueError(dimension)
    result = [0.0] * size
    for weight, hypothesis in zip(posterior, hypothesis_index(models, q_values, r_values)):
        result[hypothesis[idx]] += weight
    return result

def mixture_predictive(models, q_values, r_values, posterior, action, noise):
    hypotheses = hypothesis_index(models, q_values, r_values)
    if len(posterior) != len(hypotheses):
        raise ValueError("posterior size does not match graph×q×r hypothesis space")
    result = {}
    for weight, (gi, qi, ri) in zip(posterior, hypotheses):
        if weight == 0.0: continue
        dist = outcome_distribution_kernel(models[gi], action, noise, q_values[qi], r_values[ri])
        for y, py in dist.items(): result[y] = result.get(y, 0.0) + weight * py
    total = sum(result.values())
    if total <= 0.0: raise ValueError("joint predictive has zero mass")
    result = {y: py / total for y, py in result.items()}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("joint predictive failed to normalize")
    return result

def update_posterior(models, q_values, r_values, posterior, action, outcome, noise):
    unnormalized = []
    for weight, (gi, qi, ri) in zip(posterior, hypothesis_index(models, q_values, r_values)):
        likelihood = outcome_distribution_kernel(models[gi], action, noise, q_values[qi], r_values[ri]).get(outcome, 0.0)
        unnormalized.append(weight * likelihood)
    total = sum(unnormalized)
    if total <= 0.0: raise ValueError("observation is outside all joint graph/actuator hypotheses")
    updated = tuple(p / total for p in unnormalized)
    if not math.isclose(sum(updated), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("joint graph/actuator posterior failed to normalize")
    return updated

def action_information(models, q_values, r_values, posterior, action, noise, refs):
    pred = mixture_predictive(models, q_values, r_values, posterior, action, noise)
    entropies = {name: entropy(marginal(models, q_values, r_values, posterior, name))
                 for name in ("graph", "q", "r")}
    joint_entropy = entropy(posterior)
    expected = {"graph": 0.0, "q": 0.0, "r": 0.0, "joint": 0.0}
    for outcome, py in pred.items():
        post = update_posterior(models, q_values, r_values, posterior, action, outcome, noise)
        for name in ("graph", "q", "r"):
            expected[name] += py * entropy(marginal(models, q_values, r_values, post, name))
        expected["joint"] += py * entropy(post)
    information = {
        "eig_graph": max(0.0, entropies["graph"] - expected["graph"]),
        "eig_q": max(0.0, entropies["q"] - expected["q"]),
        "eig_r": max(0.0, entropies["r"] - expected["r"]),
        "eig_joint": max(0.0, joint_entropy - expected["joint"]),
    }
    if refs:
        distributions = [outcome_distribution_kernel(ref, action, noise, q, r)
                         for ref in refs for q in q_values for r in r_values]
        ref_pred = {}
        weight = 1.0 / len(distributions)
        for dist in distributions:
            for y, py in dist.items(): ref_pred[y] = ref_pred.get(y, 0.0) + weight * py
        information["jsd_reference"] = js_divergence(pred, ref_pred)
    else:
        information["jsd_reference"] = 0.0
    return information

def js_divergence(p, q):
    value = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0: value += 0.5 * a * math.log2(a / m)
        if b > 0.0: value += 0.5 * b * math.log2(b / m)
    return value

def choose_action(kind, models, q_values, r_values, posterior, actions, noise, refs):
    def score(action):
        val = action_information(models, q_values, r_values, posterior, action, noise, refs)
        if kind == "graph": utility = val["eig_graph"]
        elif kind == "balanced": utility = val["eig_graph"] + 0.5 * val["eig_q"] + 0.5 * val["eig_r"]
        elif kind == "joint": utility = val["eig_joint"]
        else: raise ValueError(kind)
        return (-utility, action.identifier)
    return min(actions, key=score)

def enumerate_paths(models, q_values, r_values, world, refs, actions, kind,
                    noise, true_q, true_r, budget=BUDGET):
    count = len(models) * len(q_values) * len(r_values)
    initial = tuple([1.0 / count] * count)
    terminals = []
    def visit(posterior, remaining, depth, path_p, used, surprise, impossible):
        if depth <= 0 or not remaining or impossible:
            terminals.append({"p": path_p, "posterior": posterior, "used": used,
                              "surprise": surprise, "impossible": impossible})
            return
        action = choose_action(kind, models, q_values, r_values, posterior, remaining, noise, refs)
        predicted = mixture_predictive(models, q_values, r_values, posterior, action, noise)
        actual = outcome_distribution_kernel(world, action, noise, true_q, true_r)
        remaining2 = tuple(a for a in remaining if a.identifier != action.identifier)
        for y, p_actual in actual.items():
            p_predictive = predicted.get(y, 0.0)
            new_p = path_p * p_actual
            new_surprise = math.inf if p_predictive <= 0.0 else surprise - math.log2(p_predictive)
            if p_predictive <= 0.0:
                terminals.append({"p": new_p, "posterior": posterior, "used": used + 1,
                                  "surprise": math.inf, "impossible": True})
            else:
                posterior2 = update_posterior(models, q_values, r_values, posterior, action, y, noise)
                visit(posterior2, remaining2, depth - 1, new_p, used + 1, new_surprise, False)
    visit(initial, actions, budget, 1.0, 0, 0.0, False)
    total = sum(t["p"] for t in terminals)
    if not math.isclose(total, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError(f"path mass does not normalize: {total}")
    return terminals

def config_for(policy, actual_q, actual_r):
    if policy == "joint_graph_eig": return Q_GRID, R_GRID, "graph"
    if policy == "joint_graph_actuation_balanced": return Q_GRID, R_GRID, "balanced"
    if policy == "joint_state_eig": return Q_GRID, R_GRID, "joint"
    if policy == "clean_scalar_compliance": return Q_GRID, (0.0,), "graph"
    if policy == "naive_perfect_hard_do": return (1.0,), (0.0,), "graph"
    if policy == "oracle_action_kernel": return (actual_q,), (actual_r,), "graph"
    raise ValueError(policy)

def evaluate(models, refs, ood_world, actions, policy, noise, actual_q, actual_r):
    q_values, r_values, kind = config_for(policy, actual_q, actual_r)
    graph_info = q_info = r_info = zero_support = valid_mass = expected_actions = graph_error_mass = 0.0
    q_sq = r_sq = 0.0
    in_scores = []
    hq0 = entropy([1.0 / len(q_values)] * len(q_values))
    hr0 = entropy([1.0 / len(r_values)] * len(r_values))
    for true_graph in models:
        terms = enumerate_paths(models, q_values, r_values, true_graph, refs, actions,
                                kind, noise, actual_q, actual_r)
        for t in terms:
            mass = 0.5 * t["p"]
            in_scores.append((mass, t["surprise"]))
            expected_actions += mass * t["used"]
            if t["impossible"]:
                zero_support += mass
                continue
            valid_mass += mass
            gp = marginal(models, q_values, r_values, t["posterior"], "graph")
            qp = marginal(models, q_values, r_values, t["posterior"], "q")
            rp = marginal(models, q_values, r_values, t["posterior"], "r")
            graph_info += mass * (1.0 - entropy(gp))
            q_info += mass * (hq0 - entropy(qp))
            r_info += mass * (hr0 - entropy(rp))
            predicted_graph = max(range(len(models)), key=lambda i: (gp[i], -i))
            if models[predicted_graph].graph_key() != true_graph.graph_key():
                graph_error_mass += mass
            q_mean = sum(p * q for p, q in zip(qp, q_values))
            r_mean = sum(p * r for p, r in zip(rp, r_values))
            q_sq += mass * (q_mean - actual_q) ** 2
            r_sq += mass * (r_mean - actual_r) ** 2
    ood_terms = enumerate_paths(models, q_values, r_values, ood_world, refs, actions,
                                kind, noise, actual_q, actual_r)
    auc = weighted_auc([(t["p"], t["surprise"]) for t in ood_terms], in_scores)
    return {
        "policy": policy, "ood_world": ood_world.name, "mechanism_noise": noise,
        "actual_q": actual_q, "actual_off_target_probability": actual_r,
        "inferred_q_values": list(q_values), "inferred_off_target_values": list(r_values),
        "ood_surprise_auc": auc, "graph_information_bits": graph_info,
        "compliance_information_bits": q_info, "off_target_information_bits": r_info,
        "in_class_zero_support_mass": zero_support, "in_class_valid_path_mass": valid_mass,
        "graph_map_error_conditional_valid": graph_error_mass / valid_mass if valid_mass > 1e-15 else None,
        "q_posterior_mean_squared_error_mass": q_sq,
        "off_target_posterior_mean_squared_error_mass": r_sq,
        "expected_interventions": expected_actions,
    }

def run(source_git_blob="UNBOUND"):
    models, refs, ood, actions = fixtures()
    rows = []
    for _regime, noise, q, r in REGIMES:
        for test_name, test_world in ood.items():
            for policy, _mode, _objective in POLICY_CONFIGS:
                row = evaluate(models, refs, test_world, actions, policy, noise, q, r)
                row["regime"] = _regime
                row["heldout_case"] = test_name
                rows.append(row)
    return {
        "schema": SCHEMA, "status": "EXACT_LOCAL_OFF_TARGET_INFERENCE_PILOT_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "protocol": {
            "candidate_graphs": [m.name for m in models],
            "reference_models": [m.name for m in refs],
            "heldout_worlds": {k: {"name": v.name, "parents": v.parents, "truth_tables": v.truth_tables} for k, v in ood.items()},
            "q_grid": list(Q_GRID), "off_target_probability_grid": list(R_GRID),
            "regimes": [{"name": name, "noise": noise, "actual_q": q, "actual_r": r} for name, noise, q, r in REGIMES],
            "policies": [p[0] for p in POLICY_CONFIGS],
            "off_target_kernel": "attempt target i=value; with probability 1-q no variable is set, with probability q(1-r) only i is set, and with probability q*r both i and (i+1 mod n) are set to value",
            "enumeration": "all binary assignments and adaptive action paths enumerated exactly for two distinct intervention requests",
            "caveat": "One simple cyclic spill target and one scalar conditional spill probability are used. This is not a general stochastic target-set model and is not a benchmark leaderboard."
        },
        "metrics": {
            "ood_surprise_auc": "exact weighted AUC ranking held-out-world cumulative surprise above equal-prior chain/fork world paths; chance=0.5",
            "graph_information_bits": "expected reduction in one-bit candidate graph uncertainty; impossible paths receive zero credit",
            "compliance_information_bits": "expected reduction in q entropy",
            "off_target_information_bits": "expected reduction in r entropy",
            "in_class_zero_support_mass": "true in-class path mass assigned zero predictive probability by learner",
            "graph_map_error_conditional_valid": "graph MAP error probability conditional on retained predictive support",
            "q_posterior_mean_squared_error_mass": "probability-weighted squared error of posterior mean q",
            "off_target_posterior_mean_squared_error_mass": "probability-weighted squared error of posterior mean r"
        },
        "rows": rows
    }

def self_test():
    models, refs, ood, actions = fixtures()
    for q in (0.0, 0.5, 0.75, 1.0):
        for r in (0.0, 0.25, 1.0):
            for noise in (0.0, 0.1, 0.25):
                for world in (*models, *refs, *ood.values()):
                    for action in actions:
                        dist = outcome_distribution_kernel(world, action, noise, q, r)
                        assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-12, rel_tol=0.0)
    action = actions[0]
    clean = outcome_distribution_kernel(models[0], action, 0.1, 1.0, 0.0)
    spill = outcome_distribution_kernel(models[0], action, 0.1, 1.0, 0.25)
    assert all(y[action.target] == action.value for y in clean)
    assert clean != spill
    res = run("self-test")
    assert len(res["rows"]) == len(REGIMES) * len(ood) * len(POLICY_CONFIGS)
    assert all(0.0 <= x["ood_surprise_auc"] <= 1.0 for x in res["rows"])
    joint = [x for x in res["rows"] if x["policy"].startswith("joint")]
    assert all(x["in_class_zero_support_mass"] < 1e-12 for x in joint)
    naive = [x for x in res["rows"] if x["policy"] == "naive_perfect_hard_do" and x["actual_q"] < 1.0]
    assert any(x["in_class_zero_support_mass"] > 0.0 for x in naive)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("OFF_TARGET_KERNEL_INFERENCE_SELF_TESTS_PASSED"); return 0
        raw=(json.dumps(run(args.source_git_blob),sort_keys=True,separators=(",",":"))+"\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
        else: print(raw.decode(),end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__=="__main__": raise SystemExit(main())
