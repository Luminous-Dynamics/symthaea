#!/usr/bin/env python3
"""Exact audit of intervention noncompliance in tiny Boolean SCMs.

Standard-library-only, falsification-oriented reference. This is not an external
leaderboard adapter or a state-of-the-art causal discovery method.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "symthaea-intervention-reliability-audit-v1"

@dataclass(frozen=True)
class SCM:
    name: str
    parents: tuple[tuple[int, ...], ...]
    truth_tables: tuple[int, ...]
    def graph_key(self):
        return self.parents

@dataclass(frozen=True)
class Action:
    identifier: int
    target: int
    value: int

@dataclass(frozen=True)
class Terminal:
    probability: float
    posterior: tuple[float, ...]
    probes_used: int
    surprise_bits: float
    zero_likelihood: bool

def entropy(weights):
    xs = [float(x) for x in weights if x > 0.0]
    total = sum(xs)
    if total <= 0.0: return 0.0
    return -sum((x / total) * math.log2(x / total) for x in xs)

def mechanism_output(model, node, assignment):
    parent_bits = 0
    for bit, parent in enumerate(model.parents[node]):
        parent_bits |= (assignment[parent] & 1) << bit
    return (model.truth_tables[node] >> parent_bits) & 1

def natural_probability(model, assignment, noise):
    p = 1.0
    for node, value in enumerate(assignment):
        expected = mechanism_output(model, node, assignment)
        p *= (1.0 - noise) if value == expected else noise
    return p

def outcome_distribution(model, action, noise, compliance):
    """Exact P(Y=y | assigned action), with failed actuation reverting to SCM."""
    if not 0.0 <= noise < 0.5: raise ValueError("noise must be in [0, 0.5)")
    if not 0.0 <= compliance <= 1.0: raise ValueError("compliance must be in [0, 1]")
    result = {}
    for y in itertools.product((0, 1), repeat=len(model.parents)):
        p = 1.0
        for node, value in enumerate(y):
            natural = (1.0 - noise) if value == mechanism_output(model, node, y) else noise
            if node == action.target:
                p *= compliance * float(value == action.value) + (1.0 - compliance) * natural
            else:
                p *= natural
        if p > 0.0: result[y] = p
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError(f"outcome distribution does not normalize: {sum(result.values())}")
    return result

def predictive(models, posterior, action, noise, assumed_compliance):
    out = {}
    for model, w in zip(models, posterior):
        for y, p in outcome_distribution(model, action, noise, assumed_compliance).items():
            out[y] = out.get(y, 0.0) + w * p
    total = sum(out.values())
    if total <= 0.0: raise ValueError("learner predictive distribution has zero mass")
    return {y: p / total for y, p in out.items()}

def update(models, posterior, action, y, noise, assumed_compliance):
    likelihoods = [outcome_distribution(m, action, noise, assumed_compliance).get(y, 0.0) for m in models]
    unnorm = [w * p for w, p in zip(posterior, likelihoods)]
    total = sum(unnorm)
    if total <= 0.0: raise ValueError("observed result lies outside learner support")
    result = tuple(x / total for x in unnorm)
    if not math.isclose(sum(result), 1.0, abs_tol=1e-12, rel_tol=0.0): raise AssertionError("posterior failed to normalize")
    return result

def graph_posterior(models, posterior):
    out = {}
    for model, w in zip(models, posterior):
        out[model.graph_key()] = out.get(model.graph_key(), 0.0) + w
    return out

def select_action(policy, models, posterior, remaining, noise, assumed_compliance):
    if policy == "fixed_reference":
        # Predeclared reference sequence do(X0=0), do(X1=0), then X2=0 if needed.
        rank = {0: 0, 2: 1, 4: 2, 1: 3, 3: 4, 5: 5}
        return min(remaining, key=lambda a: rank[a.identifier])
    if policy != "greedy_eig": raise ValueError(f"unknown policy {policy}")
    def objective(action):
        pred = predictive(models, posterior, action, noise, assumed_compliance)
        expected = 0.0
        for y, p in pred.items():
            post = update(models, posterior, action, y, noise, assumed_compliance)
            expected += p * entropy(graph_posterior(models, post).values())
        return expected, action.identifier
    return min(remaining, key=objective)

def fixtures():
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    xor = SCM("true_xor_collider", ((), (), (0, 1)), (0, 0, 0b0110))
    actions = tuple(Action(2 * node + value, node, value) for node in range(3) for value in (0, 1))
    return (chain, fork), {"chain": chain, "fork": fork, "xor_collider": xor}, actions

def enumerate_paths(models, world, actions, policy, noise, actual_q, assumed_q, budget=2):
    initial = tuple(1.0 / len(models) for _ in models)
    terms = []
    def visit(post, remaining, depth, path_p, used, surprise, impossible):
        if depth <= 0 or not remaining or impossible:
            terms.append(Terminal(path_p, post, used, surprise, impossible)); return
        action = select_action(policy, models, post, remaining, noise, assumed_q)
        pred = predictive(models, post, action, noise, assumed_q)
        actual = outcome_distribution(world, action, noise, actual_q)
        rest = tuple(a for a in remaining if a.identifier != action.identifier)
        for y, py in actual.items():
            pp = pred.get(y, 0.0)
            s = math.inf if pp <= 0.0 else surprise - math.log2(pp)
            new_p = path_p * py
            if pp <= 0.0:
                # Fail closed: do not manufacture a posterior after zero likelihood.
                terms.append(Terminal(new_p, post, used + 1, math.inf, True))
            else:
                new_post = update(models, post, action, y, noise, assumed_q)
                visit(new_post, rest, depth - 1, new_p, used + 1, s, False)
    visit(initial, actions, budget, 1.0, 0, 0.0, False)
    if not math.isclose(sum(t.probability for t in terms), 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError("terminal path mass failed to normalize")
    return terms

def weighted_auc(positive, negative):
    """Exact weighted ROC AUC; larger score means more OOD-like."""
    auc = 0.0
    for p, sp in positive:
        for n, sn in negative:
            if sp > sn: auc += p * n
            elif sp == sn: auc += 0.5 * p * n
    return auc

def world_score_distribution(models, world, actions, policy, noise, actual_q, assumed_q, budget=2):
    terms = enumerate_paths(models, world, actions, policy, noise, actual_q, assumed_q, budget)
    return [(t.probability, t.surprise_bits) for t in terms]

def evaluate_cell(models, worlds, actions, policy, noise, actual_q, assumed_q):
    chain_score = world_score_distribution(models, worlds["chain"], actions, policy, noise, actual_q, assumed_q)
    fork_score = world_score_distribution(models, worlds["fork"], actions, policy, noise, actual_q, assumed_q)
    ood_score = world_score_distribution(models, worlds["xor_collider"], actions, policy, noise, actual_q, assumed_q)
    in_score = [(0.5*p, s) for p,s in chain_score] + [(0.5*p,s) for p,s in fork_score]
    auc = weighted_auc(ood_score, in_score)
    # Graph information is evaluated on equal-prior in-class chain/fork worlds.
    graph_entropy = 0.0
    zero_likelihood_mass = 0.0
    expected_actions = 0.0
    for world in (worlds["chain"], worlds["fork"]):
        terms = enumerate_paths(models, world, actions, policy, noise, actual_q, assumed_q)
        for t in terms:
            w = 0.5 * t.probability
            graph_entropy += w * entropy(graph_posterior(models, t.posterior).values())
            if t.zero_likelihood: zero_likelihood_mass += w
            expected_actions += w * t.probes_used
    return {
        "policy": policy,
        "assumed_compliance": assumed_q,
        "actual_compliance": actual_q,
        "mechanism_noise": noise,
        "expected_in_class_graph_information_bits": 1.0 - graph_entropy,
        "ood_surprise_auc": auc,
        "in_class_zero_likelihood_path_mass": zero_likelihood_mass,
        "expected_interventions_in_class": expected_actions,
    }

def run(source_git_blob="UNBOUND"):
    models, worlds, actions = fixtures()
    rows=[]
    for noise in (0.10, 0.25):
        for actual_q in (0.50, 0.75, 1.00):
            rows.append(evaluate_cell(models, worlds, actions, "fixed_reference", noise, actual_q, actual_q))
            rows.append(evaluate_cell(models, worlds, actions, "greedy_eig", noise, actual_q, 1.00))
            rows.append(evaluate_cell(models, worlds, actions, "greedy_eig", noise, actual_q, actual_q))
    return {
        "schema": SCHEMA,
        "status": "EXACT_LOCAL_FALSIFICATION_AUDIT_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "protocol": {
            "variables": 3, "candidate_graphs": [m.name for m in models],
            "out_of_class_world": worlds["xor_collider"].name, "budget": 2,
            "actual_compliance_grid": [0.50, 0.75, 1.00], "noise_grid": [0.10, 0.25],
            "actions": "assign target value; with compliance q clamp target, otherwise target follows its natural noisy mechanism; other nodes follow their mechanisms",
            "noise_assumption": "learner is given the true mechanism noise in this isolated actuator-reliability audit; noise misspecification is tested separately",
            "auc": "exact weighted ROC AUC of cumulative predictive surprisal for XOR-collider versus equal-prior chain/fork mixture; infinity is retained for zero-probability paths",
            "graph_information": "1 bit initial chain/fork graph entropy minus expected posterior graph entropy on an equal-prior in-class mixture",
            "fixed_policy": "predeclared do(X0=0), do(X1=0), third action fallback; reference only, not a best-fixed baseline",
        },
        "metrics": {
            "ood_surprise_auc": "0.5 is chance ranking; 1.0 perfect ranking",
            "in_class_zero_likelihood_path_mass": "true in-class probability mass on trajectories the learner assigns zero predictive probability; posterior is not updated on impossible evidence",
            "expected_in_class_graph_information_bits": "expected reduction in chain/fork posterior entropy under the learner's assumed actuator compliance",
        },
        "rows": rows,
    }

def self_test():
    models, worlds, actions = fixtures()
    for q in (0.0, 0.5, 0.75, 1.0):
        for noise in (0.0, 0.1, 0.25):
            for world in worlds.values():
                for action in actions:
                    d = outcome_distribution(world, action, noise, q)
                    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-12, rel_tol=0.0)
    # Full compliance clamps the target; zero compliance equals observational SCM likelihood.
    a=actions[0]
    hard=outcome_distribution(worlds["chain"], a, 0.1, 1.0)
    natural=outcome_distribution(worlds["chain"], a, 0.1, 0.0)
    assert all(y[a.target] == a.value for y in hard)
    assert any(y[a.target] != a.value for y in natural)
    post=update(models, (0.5,0.5), actions[0], (0,0,0), 0.1, 0.75)
    assert math.isclose(sum(post),1.0,abs_tol=1e-12)
    for p in ("fixed_reference", "greedy_eig"):
        for q in (0.5, 0.75, 1.0):
            ts=enumerate_paths(models, worlds["xor_collider"], actions, p, 0.1, q, q)
            assert math.isclose(sum(t.probability for t in ts),1.0,abs_tol=1e-10)
    report=run("self-test")
    assert len(report["rows"]) == 18
    assert all(0.0 <= row["ood_surprise_auc"] <= 1.0 for row in report["rows"])
    assert all(0.0 <= row["in_class_zero_likelihood_path_mass"] <= 1.0 for row in report["rows"])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--source-git-blob",default="UNBOUND")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("INTERVENTION_RELIABILITY_SELF_TESTS_PASSED"); return 0
        data=json.dumps(run(args.source_git_blob),sort_keys=True,separators=(",",":"))+"\n"
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True); raw=data.encode(); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
        else: print(data,end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__=="__main__": raise SystemExit(main())
