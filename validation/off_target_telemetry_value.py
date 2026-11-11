#!/usr/bin/env python3
"""Matched-budget value of intervention-realization telemetry.

Compare outcome-only observations to a noisy spill flag or perfect observed
intervention subset while inferring graph, target compliance q and spill rate r.
All paths are exactly enumerated for small Boolean SCMs; this is not a leaderboard.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from functools import lru_cache
from pathlib import Path
from intervention_reliability_audit import entropy, weighted_auc
from off_target_kernel_inference import (
    Q_GRID, R_GRID, REGIMES, fixtures, mechanism_probability, hypothesis_index,
    marginal, js_divergence,
)

SCHEMA = "symthaea-off-target-telemetry-value-v1"
BUDGET = 2
TELEMETRY_MODES = ("outcome_only", "noisy_spill_sensor", "perfect_subset_telemetry")
POLICIES = ("graph_eig", "actuation_balanced")
SENSOR_SENSITIVITY = 0.90
SENSOR_SPECIFICITY = 0.90

def action_subset_law(action, node_count, q, r):
    target=action.target; collateral=(target+1)%node_count
    return (((),1.0-q),((target,),q*(1.0-r)),((target,collateral),q*r))

@lru_cache(maxsize=None)
def observation_distribution(model, action, noise, q, r, telemetry):
    """Joint outcome/telemetry likelihood under the declared stochastic action kernel."""
    n=len(model.parents); result={}
    if telemetry not in TELEMETRY_MODES: raise ValueError(telemetry)
    for subset, subset_p in action_subset_law(action,n,q,r):
        if subset_p<=0.0: continue
        forced=set(subset)
        for y in itertools.product((0,1), repeat=n):
            if any(y[node]!=action.value for node in forced): continue
            p=subset_p
            for node in range(n):
                if node not in forced: p*=mechanism_probability(model,y,node,noise)
            if p<=0.0: continue
            if telemetry=="outcome_only":
                result[y]=result.get(y,0.0)+p
            elif telemetry=="perfect_subset_telemetry":
                key=(y,tuple(sorted(subset)))
                result[key]=result.get(key,0.0)+p
            else:
                has_spill=len(subset)>1
                p_true=SENSOR_SENSITIVITY if has_spill else 1.0-SENSOR_SPECIFICITY
                p_false=1.0-p_true
                for flag,ps in ((True,p_true),(False,p_false)):
                    key=(y,flag)
                    result[key]=result.get(key,0.0)+p*ps
    total=sum(result.values())
    if not math.isclose(total,1.0,abs_tol=1e-12,rel_tol=0.0):
        raise AssertionError(f"observation distribution failed to normalize: {total}")
    return result

def marginal_posterior(models,q_values,r_values,posterior,dimension):
    if dimension=="graph": size=len(models); index=0
    elif dimension=="q": size=len(q_values); index=1
    elif dimension=="r": size=len(r_values); index=2
    else: raise ValueError(dimension)
    out=[0.0]*size
    for w,hyp in zip(posterior,hypothesis_index(models,q_values,r_values)):
        out[hyp[index]]+=w
    return out

@lru_cache(maxsize=200000)
def predictive(models,q_values,r_values,posterior,action,noise,telemetry):
    out={}
    hypotheses=hypothesis_index(models,q_values,r_values)
    if len(posterior)!=len(hypotheses): raise ValueError("posterior/hypothesis mismatch")
    for w,(gi,qi,ri) in zip(posterior,hypotheses):
        if w<=0.0: continue
        dist=observation_distribution(models[gi],action,noise,q_values[qi],r_values[ri],telemetry)
        for z,p in dist.items(): out[z]=out.get(z,0.0)+w*p
    total=sum(out.values())
    if total<=0.0: raise ValueError("predictive distribution has zero mass")
    out={z:p/total for z,p in out.items()}
    if not math.isclose(sum(out.values()),1.0,abs_tol=1e-12,rel_tol=0.0): raise AssertionError("predictive did not normalize")
    return out

@lru_cache(maxsize=200000)
def posterior_update(models,q_values,r_values,posterior,action,observation,noise,telemetry):
    unnorm=[]
    for w,(gi,qi,ri) in zip(posterior,hypothesis_index(models,q_values,r_values)):
        likelihood=observation_distribution(models[gi],action,noise,q_values[qi],r_values[ri],telemetry).get(observation,0.0)
        unnorm.append(w*likelihood)
    total=sum(unnorm)
    if total<=0.0: raise ValueError("observation outside every model/actuator hypothesis")
    out=tuple(x/total for x in unnorm)
    if not math.isclose(sum(out),1.0,abs_tol=1e-12,rel_tol=0.0): raise AssertionError("posterior did not normalize")
    return out

@lru_cache(maxsize=200000)
def action_information(models,q_values,r_values,posterior,action,noise,telemetry):
    pred=predictive(models,q_values,r_values,posterior,action,noise,telemetry)
    before={k:entropy(marginal_posterior(models,q_values,r_values,posterior,k)) for k in ("graph","q","r")}
    expected={k:0.0 for k in ("graph","q","r")}
    for z,p in pred.items():
        post=posterior_update(models,q_values,r_values,posterior,action,z,noise,telemetry)
        for k in expected: expected[k]+=p*entropy(marginal_posterior(models,q_values,r_values,post,k))
    return {"eig_graph":max(0.0,before["graph"]-expected["graph"]),
            "eig_q":max(0.0,before["q"]-expected["q"]),
            "eig_r":max(0.0,before["r"]-expected["r"])}

def choose_action(models,q_values,r_values,posterior,actions,noise,telemetry,policy):
    def score(action):
        info=action_information(models,q_values,r_values,posterior,action,noise,telemetry)
        if policy=="graph_eig": utility=info["eig_graph"]
        elif policy=="actuation_balanced": utility=info["eig_graph"]+0.5*info["eig_q"]+0.5*info["eig_r"]
        else: raise ValueError(policy)
        return (-utility,action.identifier)
    return min(actions,key=score)

def enumerate_paths(models,q_values,r_values,world,actions,noise,true_q,true_r,telemetry,policy,budget=BUDGET):
    n_hyp=len(models)*len(q_values)*len(r_values)
    initial=tuple([1.0/n_hyp]*n_hyp); terminals=[]
    def visit(post,remaining,depth,path_p,used,surprise,impossible):
        if depth<=0 or not remaining or impossible:
            terminals.append({"p":path_p,"posterior":post,"used":used,"surprise":surprise,"impossible":impossible}); return
        action=choose_action(models,q_values,r_values,post,remaining,noise,telemetry,policy)
        pred=predictive(models,q_values,r_values,post,action,noise,telemetry)
        actual=observation_distribution(world,action,noise,true_q,true_r,telemetry)
        after=tuple(a for a in remaining if a.identifier!=action.identifier)
        for z,p_actual in actual.items():
            p_pred=pred.get(z,0.0); mass=path_p*p_actual
            s=math.inf if p_pred<=0.0 else surprise-math.log2(p_pred)
            if p_pred<=0.0:
                terminals.append({"p":mass,"posterior":post,"used":used+1,"surprise":math.inf,"impossible":True})
            else:
                post2=posterior_update(models,q_values,r_values,post,action,z,noise,telemetry)
                visit(post2,after,depth-1,mass,used+1,s,False)
    visit(tuple([1.0/(len(models)*len(q_values)*len(r_values))]*(len(models)*len(q_values)*len(r_values))),actions,budget,1.0,0,0.0,False)
    total=sum(t["p"] for t in terminals)
    if not math.isclose(total,1.0,abs_tol=1e-10,rel_tol=0.0): raise AssertionError(f"path mass not normalized: {total}")
    return terminals

def evaluate(models,refs,ood_world,actions,noise,true_q,true_r,telemetry,policy):
    q_values=Q_GRID; r_values=R_GRID
    graph_info=q_info=r_info=zero_support=valid_mass=map_error_mass=0.0; in_scores=[]
    hq0=entropy([1.0/len(q_values)]*len(q_values)); hr0=entropy([1.0/len(r_values)]*len(r_values))
    for truth in models:
        terms=enumerate_paths(models,q_values,r_values,truth,actions,noise,true_q,true_r,telemetry,policy)
        for t in terms:
            mass=.5*t["p"]; in_scores.append((mass,t["surprise"]))
            if t["impossible"]: zero_support+=mass; continue
            valid_mass+=mass
            gp=marginal_posterior(models,q_values,r_values,t["posterior"],"graph")
            qp=marginal_posterior(models,q_values,r_values,t["posterior"],"q")
            rp=marginal_posterior(models,q_values,r_values,t["posterior"],"r")
            graph_info+=mass*(1.0-entropy(gp)); q_info+=mass*(hq0-entropy(qp)); r_info+=mass*(hr0-entropy(rp))
            map_g=max(range(len(models)),key=lambda i:(gp[i],-i))
            if models[map_g].graph_key()!=truth.graph_key(): map_error_mass+=mass
    ood_terms=enumerate_paths(models,q_values,r_values,ood_world,actions,noise,true_q,true_r,telemetry,policy)
    auc=weighted_auc([(t["p"],t["surprise"]) for t in ood_terms],in_scores)
    return {"telemetry":telemetry,"policy":policy,"ood_world":ood_world.name,"noise":noise,"q":true_q,"r":true_r,
        "ood_surprise_auc":auc,"graph_information_bits":graph_info,"compliance_information_bits":q_info,
        "off_target_information_bits":r_info,"in_class_zero_support_mass":zero_support,
        "in_class_valid_path_mass":valid_mass,"graph_map_error_conditional_valid":map_error_mass/valid_mass if valid_mass>1e-15 else None}

def run(source_git_blob="UNBOUND"):
    models,refs,ood,actions=fixtures(); rows=[]
    for regime,noise,q,r in (("matched_clean",.10,.75,0.0),("matched_off_target",.10,.75,.25),
                              ("weak_off_target",.10,.5,.25),("noisy_off_target",.25,.75,.25)):
        for _test_name,test_world in ood.items():
            for telemetry in TELEMETRY_MODES:
                for policy in POLICIES:
                    row=evaluate(models,refs,test_world,actions,noise,q,r,telemetry,policy)
                    row["regime"]=regime; rows.append(row)
    return {"schema":SCHEMA,"status":"EXACT_LOCAL_TELEMETRY_VALUE_PILOT_NOT_EXTERNAL_QUALIFICATION","source_git_blob":source_git_blob,
        "protocol":{"candidate_graphs":[m.name for m in models],"reference_models":[m.name for m in refs],
        "heldout_worlds":{k:v.name for k,v in ood.items()},"q_grid":list(Q_GRID),"r_grid":list(R_GRID),"budget":BUDGET,
        "telemetry_modes":{"outcome_only":"observe outcome vector only","noisy_spill_sensor":"observe a binary spill flag with 90% sensitivity and 90% specificity","perfect_subset_telemetry":"observe exact realized subset among no-op, target-only and target-plus-collateral"},
        "policies":list(POLICIES),"comparison":"same two-distinct-intervention budget, same SCM/action grid; only observation channel differs"},
        "metrics":{"ood_surprise_auc":"exact weighted AUC; 0.5 is chance","graph_information_bits":"expected reduction in one-bit graph entropy","compliance_information_bits":"expected reduction in q entropy","off_target_information_bits":"expected reduction in r entropy","zero_support":"in-class probability mass assigned zero predictive likelihood"},"rows":rows}

def self_test():
    models,refs,ood,actions=fixtures()
    for q in (0.0,.5,.75,1.0):
        for r in (0.0,.25,1.0):
            for noise in (0.0,.1,.25):
                for w in (*models,*refs,*ood.values()):
                    for a in actions:
                        for mode in TELEMETRY_MODES:
                            d=observation_distribution(w,a,noise,q,r,mode)
                            assert math.isclose(sum(d.values()),1.0,abs_tol=1e-12,rel_tol=0.0)
    a=actions[0]
    no=observation_distribution(models[0],a,.1,.75,.25,"outcome_only")
    yes=observation_distribution(models[0],a,.1,.75,.25,"perfect_subset_telemetry")
    assert len(yes)>=len(no)
    res=run("self-test"); assert len(res["rows"])==4*2*3*2
    assert all(0<=x["ood_surprise_auc"]<=1 for x in res["rows"])
    for row in res["rows"]:
        if row["telemetry"]!="outcome_only": assert row["in_class_zero_support_mass"]<1e-12

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");parser.add_argument("--source-git-blob",default="UNBOUND");parser.add_argument("--output",type=Path);args=parser.parse_args()
    try:
        if args.self_test:self_test();print("OFF_TARGET_TELEMETRY_VALUE_SELF_TESTS_PASSED");return 0
        raw=(json.dumps(run(args.source_git_blob),sort_keys=True,separators=(",",":"))+"\n").encode()
        if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(raw);print(f"wrote {args.output} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
        else:print(raw.decode(),end="")
        return 0
    except (ValueError,AssertionError,OSError,RuntimeError) as exc:
        print(f"NOT QUALIFIED: {exc}");return 2
if __name__=="__main__":raise SystemExit(main())
