#!/usr/bin/env python3
"""Matched-budget test of repeatable versus distinct stochastic interventions.

Question: when graph, target compliance and collateral-effect probability are
all uncertain, are repeated probes at the same action more useful than forcing
every probe to be distinct? Exact small-SCM path enumeration; not a leaderboard.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from intervention_reliability_audit import entropy, weighted_auc
from off_target_kernel_inference import (
    Q_GRID, R_GRID, REGIMES, POLICY_CONFIGS, fixtures,
    outcome_distribution_kernel, hypothesis_index, marginal, mixture_predictive,
    update_posterior, action_information, choose_action,
)

SCHEMA = "symthaea-off-target-repeat-design-v1"
BUDGET = 2
POLICIES = (
    ("joint_graph_eig", "joint", "graph"),
    ("joint_graph_actuation_balanced", "joint", "balanced"),
    ("joint_state_eig", "joint", "joint"),
    ("clean_scalar_compliance", "clean", "graph"),
    ("naive_perfect_hard_do", "naive", "graph"),
    ("oracle_action_kernel", "oracle", "graph"),
)

def config_for(policy, actual_q, actual_r):
    if policy in ("joint_graph_eig", "joint_graph_actuation_balanced", "joint_state_eig"):
        mode = {"joint_graph_eig":"graph", "joint_graph_actuation_balanced":"balanced", "joint_state_eig":"joint"}[policy]
        return Q_GRID, R_GRID, mode
    if policy == "clean_scalar_compliance": return Q_GRID, (0.0,), "graph"
    if policy == "naive_perfect_hard_do": return (1.0,), (0.0,), "graph"
    if policy == "oracle_action_kernel": return (actual_q,), (actual_r,), "graph"
    raise ValueError(policy)

def enumerate_paths(models, q_values, r_values, world, refs, actions, kind,
                    noise, true_q, true_r, allow_repeats, budget=BUDGET):
    n_hyp = len(models) * len(q_values) * len(r_values)
    initial = tuple([1.0 / n_hyp] * n_hyp)
    terminals = []
    def visit(post, available, depth, path_p, used, surprise, impossible):
        if depth <= 0 or not available or impossible:
            terminals.append({"p":path_p, "posterior":post, "used":used,
                              "surprise":surprise, "impossible":impossible})
            return
        action = choose_action(kind, models, q_values, r_values, post, available, noise, refs)
        predicted = mixture_predictive(models, q_values, r_values, post, action, noise)
        actual = outcome_distribution_kernel(world, action, noise, true_q, true_r)
        if allow_repeats:
            next_available = available
        else:
            next_available = tuple(a for a in available if a.identifier != action.identifier)
        for y, p_actual in actual.items():
            p_predicted = predicted.get(y, 0.0)
            new_mass = path_p * p_actual
            new_surprise = math.inf if p_predicted <= 0.0 else surprise - math.log2(p_predicted)
            if p_predicted <= 0.0:
                terminals.append({"p":new_mass, "posterior":post, "used":used+1,
                                  "surprise":math.inf, "impossible":True})
            else:
                new_post = update_posterior(models, q_values, r_values, post, action, y, noise)
                visit(new_post, next_available, depth-1, new_mass, used+1, new_surprise, False)
    visit(initial, actions, budget, 1.0, 0, 0.0, False)
    total = sum(t["p"] for t in terminals)
    if not math.isclose(total, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise AssertionError(f"path mass does not normalize: {total}")
    return terminals

def evaluate(models, refs, ood_world, actions, policy, noise, true_q, true_r, allow_repeats):
    q_values, r_values, kind = config_for(policy, true_q, true_r)
    graph_info=q_info=r_info=zero_support=valid_mass=expected_actions=graph_error_mass=q_sq=r_sq=0.0
    in_scores=[]
    hq0=entropy([1.0/len(q_values)]*len(q_values))
    hr0=entropy([1.0/len(r_values)]*len(r_values))
    for truth in models:
        terms=enumerate_paths(models,q_values,r_values,truth,refs,actions,kind,noise,true_q,true_r,allow_repeats)
        for t in terms:
            mass=0.5*t["p"]
            in_scores.append((mass,t["surprise"]))
            expected_actions += mass*t["used"]
            if t["impossible"]:
                zero_support += mass
                continue
            valid_mass += mass
            gp=marginal(models,q_values,r_values,t["posterior"],"graph")
            qp=marginal(models,q_values,r_values,t["posterior"],"q")
            rp=marginal(models,q_values,r_values,t["posterior"],"r")
            graph_info += mass*(1.0-entropy(gp))
            q_info += mass*(hq0-entropy(qp))
            r_info += mass*(hr0-entropy(rp))
            predicted_graph=max(range(len(models)),key=lambda i:(gp[i],-i))
            if models[predicted_graph].graph_key()!=truth.graph_key(): graph_error_mass += mass
            q_mean=sum(p*q for p,q in zip(qp,q_values))
            r_mean=sum(p*r for p,r in zip(rp,r_values))
            q_sq += mass*(q_mean-true_q)**2
            r_sq += mass*(r_mean-true_r)**2
    ood_terms=enumerate_paths(models,q_values,r_values,ood_world,refs,actions,kind,noise,true_q,true_r,allow_repeats)
    auc=weighted_auc([(t["p"],t["surprise"]) for t in ood_terms],in_scores)
    return {"policy":policy,"design":"repeatable" if allow_repeats else "distinct_only",
        "budget":BUDGET,"ood_world":ood_world.name,"mechanism_noise":noise,
        "actual_q":true_q,"actual_r":true_r,"ood_surprise_auc":auc,
        "graph_information_bits":graph_info,"compliance_information_bits":q_info,
        "off_target_information_bits":r_info,"in_class_zero_support_mass":zero_support,
        "in_class_valid_path_mass":valid_mass,
        "graph_map_error_conditional_valid":graph_error_mass/valid_mass if valid_mass>1e-15 else None,
        "q_posterior_mean_squared_error_mass":q_sq,"r_posterior_mean_squared_error_mass":r_sq,
        "expected_interventions":expected_actions}

def run(source_git_blob="UNBOUND"):
    models,refs,ood,actions=fixtures(); rows=[]
    for regime,noise,q,r in REGIMES:
        for test_name,test_world in ood.items():
            for policy,_,_ in POLICIES:
                for repeat in (False,True):
                    row=evaluate(models,refs,test_world,actions,policy,noise,q,r,repeat)
                    row["regime"]=regime; row["heldout_case"]=test_name
                    rows.append(row)
    return {"schema":SCHEMA,"status":"EXACT_LOCAL_REPEAT_DESIGN_PILOT_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob":source_git_blob,"protocol":{"nodes":3,"action_budget":BUDGET,
        "action_set":[{"id":a.identifier,"target":a.target,"value":a.value} for a in actions],
        "design_comparison":"same two-intervention budget; same action may be selected repeatedly or chosen actions are removed after use",
        "q_grid":list(Q_GRID),"r_grid":list(R_GRID),"regimes":[{"name":n,"noise":p,"q":q,"r":r} for n,p,q,r in REGIMES],
        "heldout_worlds":{k:v.name for k,v in ood.items()},"policies":[p[0] for p in POLICIES],
        "limitations":"Three-node hand-authored models, scalar q and cyclic one-collateral r; repeated samples are independent conditional on graph/action-kernel parameters."},
        "metrics":{"graph_information_bits":"expected reduction in one-bit graph entropy","compliance_information_bits":"expected reduction in q entropy","off_target_information_bits":"expected reduction in r entropy","ood_surprise_auc":"exact weighted AUC; 0.5 is chance","zero_support":"in-class probability mass assigned zero predictive likelihood"},"rows":rows}

def self_test():
    models,refs,ood,actions=fixtures()
    for repeat in (False,True):
        terms=enumerate_paths(models,Q_GRID,R_GRID,models[0],refs,actions,"balanced",0.1,0.75,0.25,repeat,budget=BUDGET)
        assert math.isclose(sum(t["p"] for t in terms),1.0,abs_tol=1e-10)
        assert all(t["used"]<=BUDGET for t in terms)
    data=run("self-test")
    assert len(data["rows"])==4*2*len(POLICIES)*2
    assert all(0.0<=row["ood_surprise_auc"]<=1.0 for row in data["rows"])
    for regime in set(row["regime"] for row in data["rows"]):
        for policy in set(row["policy"] for row in data["rows"]):
            for world in set(row["ood_world"] for row in data["rows"]):
                group=[r for r in data["rows"] if r["regime"]==regime and r["policy"]==policy and r["ood_world"]==world]
                assert {r["design"] for r in group}=={"repeatable","distinct_only"}

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--source-git-blob",default="UNBOUND"); parser.add_argument("--output",type=Path); args=parser.parse_args()
    try:
        if args.self_test: self_test(); print("OFF_TARGET_REPEAT_DESIGN_SELF_TESTS_PASSED"); return 0
        raw=(json.dumps(run(args.source_git_blob),sort_keys=True,separators=(",",":"))+"\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(raw); print(f"wrote {args.output} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
        else: print(raw.decode(),end="")
        return 0
    except (ValueError,AssertionError,OSError,RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__=="__main__": raise SystemExit(main())
