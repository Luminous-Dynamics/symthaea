#!/usr/bin/env python3
"""Stress-test causal probe Pareto objectives across held-out SCM families.

Standard-library-only exact enumeration. Reference-bank weights are uniform over
all supplied reference models (not hard-coded to a two-model bank). This pilot
exposes reference-coverage sensitivity; it is not an external leaderboard.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from intervention_reliability_audit import SCM, Action, entropy, outcome_distribution, update, graph_posterior, weighted_auc

SCHEMA = "symthaea-causal-open-world-reference-stress-v1"
POLICY_CONFIGS = (
    ("eig_only", 0.0, "narrow"),
    ("narrow_reference_jsd", 0.5, "narrow"),
    ("broad_reference_jsd", 0.5, "broad"),
)
REGIMES = (
    ("matched_perfect_low_noise", 0.10, 1.00, 1.00),
    ("matched_partial_low_noise", 0.10, 0.75, 0.75),
    ("misspecified_partial_low_noise", 0.10, 0.75, 1.00),
)
BUDGET = 2


def fixtures():
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    collider_and = SCM("reference_collider_and", ((), (), (0, 1)), (0, 0, 0b1000))
    collider_or = SCM("reference_collider_or", ((), (), (0, 1)), (0, 0, 0b1110))
    chain_first_not = SCM("reference_chain_not_then_id", ((), (0,), (1,)), (0, 0b01, 0b10))
    fork_first_not = SCM("reference_fork_not_and_id", ((), (0,), (0,)), (0, 0b01, 0b10))
    narrow = (collider_and, collider_or)
    broad = (*narrow, chain_first_not, fork_first_not)
    heldout = {
        "collider_xor_mechanism_ood": SCM("heldout_collider_xor", ((), (), (0, 1)), (0, 0, 0b0110)),
        "collider_xnor_mechanism_ood": SCM("heldout_collider_xnor", ((), (), (0, 1)), (0, 0, 0b1001)),
        "collider_nand_mechanism_ood": SCM("heldout_collider_nand", ((), (), (0, 1)), (0, 0, 0b0111)),
        "chain_second_edge_inverted": SCM("heldout_chain_id_then_not", ((), (0,), (1,)), (0, 0b10, 0b01)),
        "fork_both_children_inverted": SCM("heldout_fork_both_not", ((), (0,), (0,)), (0, 0b01, 0b01)),
        "reverse_chain_topology_ood": SCM("heldout_reverse_chain", ((1,), (2,), ()), (0b10, 0b10, 0)),
    }
    actions = tuple(Action(2 * node + value, node, value) for node in range(3) for value in (0, 1))
    return (chain, fork), narrow, broad, heldout, actions


def mixture_distribution(models, weights, action, noise, compliance):
    if len(models) != len(weights) or not models:
        raise ValueError("mixture model/weight lengths must match and be nonempty")
    if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
        raise ValueError("mixture weights must be nonnegative with positive total")
    normalizer = sum(weights)
    result = {}
    for model, weight in zip(models, weights):
        for outcome, p in outcome_distribution(model, action, noise, compliance).items():
            result[outcome] = result.get(outcome, 0.0) + (weight / normalizer) * p
    total = sum(result.values())
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError(f"mixture does not normalize: {total}")
    return result


def js_divergence(p, q):
    result = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0: result += 0.5 * a * math.log2(a / m)
        if b > 0.0: result += 0.5 * b * math.log2(b / m)
    return result


def action_components(models, posterior, reference, action, noise, assumed_q):
    candidate_pred = mixture_distribution(models, posterior, action, noise, assumed_q)
    graph_h_before = entropy(graph_posterior(models, posterior).values())
    expected_h_after = 0.0
    for outcome, p in candidate_pred.items():
        post = update(models, posterior, action, outcome, noise, assumed_q)
        expected_h_after += p * entropy(graph_posterior(models, post).values())
    eig = max(0.0, graph_h_before - expected_h_after)
    ref_pred = mixture_distribution(reference, [1.0] * len(reference), action, noise, assumed_q)
    divergence = js_divergence(candidate_pred, ref_pred)
    return eig, divergence


def choose_action(models, posterior, reference, remaining, beta, noise, assumed_q):
    if not remaining: raise ValueError("no remaining actions")
    def objective(action):
        eig, divergence = action_components(models, posterior, reference, action, noise, assumed_q)
        return (-(eig + beta * divergence), action.identifier)
    return min(remaining, key=objective)


def enumerate_paths(models, world, reference, actions, beta, noise, actual_q, assumed_q, budget=BUDGET):
    initial = tuple(1.0 / len(models) for _ in models)
    terminals = []
    def visit(posterior, remaining, depth, path_probability, used, cumulative_surprise, impossible):
        if depth <= 0 or not remaining or impossible:
            terminals.append({"p": path_probability, "posterior": posterior, "used": used,
                             "surprise": cumulative_surprise, "impossible": impossible})
            return
        action = choose_action(models, posterior, reference, remaining, beta, noise, assumed_q)
        learner_pred = mixture_distribution(models, posterior, action, noise, assumed_q)
        actual_pred = outcome_distribution(world, action, noise, actual_q)
        next_remaining = tuple(a for a in remaining if a.identifier != action.identifier)
        for outcome, p_actual in actual_pred.items():
            p_learner = learner_pred.get(outcome, 0.0)
            surprise = math.inf if p_learner <= 0.0 else -math.log2(p_learner)
            path_p = path_probability * p_actual
            if p_learner <= 0.0:
                terminals.append({"p": path_p, "posterior": posterior, "used": used + 1,
                                  "surprise": math.inf, "impossible": True})
            else:
                post = update(models, posterior, action, outcome, noise, assumed_q)
                visit(post, next_remaining, depth - 1, path_p, used + 1,
                      cumulative_surprise + surprise, False)
    visit(initial, actions, budget, 1.0, 0, 0.0, False)
    total = sum(t["p"] for t in terminals)
    if not math.isclose(total, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError(f"terminal path mass does not normalize: {total}")
    return terminals


def evaluate(models, reference, heldout, actions, beta, noise, actual_q, assumed_q):
    in_rows, graph_information = [], 0.0
    zero_support_mass, expected_actions, valid_mass = 0.0, 0.0, 0.0
    for world in models:
        for term in enumerate_paths(models, world, reference, actions, beta, noise, actual_q, assumed_q):
            mass = 0.5 * term["p"]
            in_rows.append((mass, term["surprise"]))
            expected_actions += mass * term["used"]
            if term["impossible"]:
                zero_support_mass += mass
            else:
                valid_mass += mass
                graph_information += mass * (1.0 - entropy(graph_posterior(models, term["posterior"]).values()))
    ood_terms = enumerate_paths(models, heldout, reference, actions, beta, noise, actual_q, assumed_q)
    auc = weighted_auc([(t["p"], t["surprise"]) for t in ood_terms], in_rows)
    return {
        "heldout_world": heldout.name,
        "beta": beta,
        "reference_bank": "narrow_collider_and_or" if len(reference) == 2 else "broad_collider_chain_fork",
        "mechanism_noise": noise,
        "actual_compliance": actual_q,
        "assumed_compliance": assumed_q,
        "ood_surprise_auc": auc,
        "usable_graph_information_bits": graph_information,
        "in_class_valid_path_mass": valid_mass,
        "in_class_zero_support_mass": zero_support_mass,
        "expected_in_class_interventions": expected_actions,
    }


def run(source_git_blob="UNBOUND"):
    models, narrow, broad, heldout, actions = fixtures()
    rows = []
    for regime, noise, actual_q, assumed_q in REGIMES:
        for case, test_world in heldout.items():
            for policy_name, beta, bank_name in POLICY_CONFIGS:
                reference = narrow if bank_name == "narrow" else broad
                row = evaluate(models, reference, test_world, actions, beta, noise, actual_q, assumed_q)
                row["policy"] = policy_name
                row["regime"] = regime
                row["heldout_case"] = case
                rows.append(row)
    return {
        "schema": SCHEMA,
        "status": "EXACT_LOCAL_OPEN_WORLD_STRESS_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "protocol": {
            "candidate_models": [m.name for m in models],
            "narrow_reference_bank": [m.name for m in narrow],
            "broad_reference_bank": [m.name for m in broad],
            "heldout_worlds": {k: {"name": v.name, "parents": v.parents, "truth_tables": v.truth_tables} for k, v in heldout.items()},
            "regimes": [{"name": n, "mechanism_noise": p, "actual_compliance": q, "assumed_compliance": qhat} for n,p,q,qhat in REGIMES],
            "policies": {name: {"beta": beta, "reference_bank": bank,
                         "objective": "expected graph information + beta * JSD(candidate predictive, equal-weight reference predictive)"}
                         for name, beta, bank in POLICY_CONFIGS},
            "enumeration": "exact binary outcomes across two adaptive distinct interventions; infinite surprise retained for learner-zero-probability outcomes",
            "leakage_warning": "Reference banks are small and hand-authored. They do not contain exact held-out worlds, but share causal-family knowledge with some tests. No arbitrary open-world generalization claim.",
            "baseline_warning": "EIG-only is the within-prototype control; no conventional PC/GES/FCI baseline or external benchmark adapter is included.",
        },
        "metrics": {
            "ood_surprise_auc": "exact weighted ROC AUC ranking heldout trajectories above equal-prior in-class chain/fork trajectories; 0.5 is chance",
            "usable_graph_information_bits": "expected usable reduction in initial one-bit chain/fork uncertainty; paths assigned zero probability receive zero credit",
            "in_class_zero_support_mass": "in-class probability mass where the learner assigns zero probability to an observed path",
            "expected_in_class_interventions": "expected number of distinct actions used within budget two",
        },
        "rows": rows,
    }


def self_test():
    models, narrow, broad, heldout, actions = fixtures()
    assert len(heldout) == 6 and len(narrow) == 2 and len(broad) == 4
    p = mixture_distribution(broad, [1.0] * len(broad), actions[0], 0.1, 1.0)
    p2 = mixture_distribution(narrow, [1.0] * len(narrow), actions[0], 0.1, 1.0)
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-12)
    assert any(abs(p.get(y, 0.0) - p2.get(y, 0.0)) > 1e-8 for y in set(p) | set(p2))
    report = run("self-test")
    assert len(report["rows"]) == len(REGIMES) * len(heldout) * len(POLICY_CONFIGS)
    assert all(0.0 <= r["ood_surprise_auc"] <= 1.0 for r in report["rows"])
    for actual_q, assumed_q in ((0.75, 0.75), (0.75, 1.0)):
        row = evaluate(models, narrow, heldout["collider_xor_mechanism_ood"], actions, 0.0,
                       0.1, actual_q, assumed_q)
        if actual_q == assumed_q:
            assert row["in_class_zero_support_mass"] < 1e-12
        else:
            assert row["in_class_zero_support_mass"] > 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("OPEN_WORLD_REFERENCE_STRESS_SELF_TESTS_PASSED"); return 0
        encoded = (json.dumps(run(args.source_git_blob), sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else:
            print(encoded.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2

if __name__ == "__main__": raise SystemExit(main())
