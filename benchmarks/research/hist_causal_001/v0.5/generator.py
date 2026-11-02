#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.5 deterministic recipe generator.

Exact-rational, seeded task recipes. The solver receives agent_view only.
evaluator_key is private ground truth; never pass it to the solver.
Research-only; public seeds are development cases, not a sealed benchmark.
"""
from __future__ import annotations
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations, product
import json

MODELS=("genuine_mechanism","latent_common_cause","primary_sensor_artefact",
        "actuator_noncompliance","common_mode_reference_artefact","null_effect")
PROBES=("observational_primary","randomized_primary","secondary_reference",
        "actuator_readback","sensor_self_test","blank_common_mode_control")
COSTS={"observational_primary":F(1,4),"randomized_primary":F(1),
       "secondary_reference":F(2),"actuator_readback":F(3,2),
       "sensor_self_test":F(1),"blank_common_mode_control":F(2)}
MASK=(1<<64)-1

def _splitmix64(x:int)->tuple[int,int]:
    x=(x+0x9E3779B97F4A7C15)&MASK
    z=x
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK
    z=((z^(z>>27))*0x94D049BB133111EB)&MASK
    return x,(z^(z>>31))&MASK

def _choose(seed:int,slot:int,values):
    _,r=_splitmix64((seed+(slot+1)*0x9E3779B97F4A7C15)&MASK)
    return values[r%len(values)]

def _dist(model:str,probe:str,p:dict)->dict[tuple[int,...],F]:
    pe=p["p_effect"]; pb=1-pe; p0=p["p_no_dose"]; p1=p["p_dose"]
    d0=(pb-p0)/(p1-p0); d1=(pe-p0)/(p1-p0)
    def add(k,v):
        if v: out[k]=out.get(k,F(0))+v
    def bern(z,prob): return ((1,prob),(0,1-prob))
    def dose(z,delivered): return bern(z,p0*(1-delivered)+p1*delivered)
    out={}
    if probe=="observational_primary":
        if model=="null_effect":
            for z,y in product((0,1),repeat=2): add((z,y),F(1,4))
        else:
            add((0,0),pe/2); add((0,1),pb/2); add((1,0),pb/2); add((1,1),pe/2)
    elif probe=="randomized_primary":
        for z in (0,1):
            if model in ("genuine_mechanism","primary_sensor_artefact","actuator_noncompliance","common_mode_reference_artefact"):
                ys=dose(z,d1 if z else d0) if model=="actuator_noncompliance" else bern(z,pe if z else pb)
            else: ys=((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add((z,y),py/2)
    elif probe=="secondary_reference":
        for z in (0,1):
            if model in ("genuine_mechanism","actuator_noncompliance"):
                ys=dose(z,d1 if z else d0) if model=="actuator_noncompliance" else bern(z,pe if z else pb)
            elif model=="common_mode_reference_artefact": ys=bern(z,pe if z else pb)
            else: ys=((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add((z,y),py/2)
    elif probe=="actuator_readback":
        for z in (0,1):
            ys=bern(z,d1 if z else d0) if model=="actuator_noncompliance" else ((z,F(1)),)
            for actual,pa in ys: add((z,actual),pa/2)
    elif probe=="sensor_self_test":
        alarm=p["sensor_sensitivity"] if model=="primary_sensor_artefact" else p["sensor_fpr"]
        for z in (0,1):
            for a,pa in ((1,alarm),(0,1-alarm)): add((z,a),pa/2)
    elif probe=="blank_common_mode_control":
        for z in (0,1):
            ys=bern(z,pe if z else pb) if model=="common_mode_reference_artefact" else ((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add((z,y),py/2)
    else: raise ValueError(f"unknown probe {probe}")
    if sum(out.values(),F(0))!=1: raise ArithmeticError(f"bad mass {model}/{probe}")
    return out

def _equivalent(a,b,params,probes=PROBES):
    return all(_dist(a,p,params)==_dist(b,p,params) for p in probes)

def _validate_params(p):
    pe=p["p_effect"]; pb=1-pe; p0=p["p_no_dose"]; p1=p["p_dose"]
    if not (F(0)<p0<pb<pe<p1<F(1)): return False
    d0=(pb-p0)/(p1-p0); d1=(pe-p0)/(p1-p0)
    if not (0<=d0<=1 and 0<=d1<=1): return False
    if not (F(0)<p["sensor_fpr"]<p["sensor_sensitivity"]<F(1)): return False
    return all(sum(_dist(m,probe,p).values(),F(0))==1 for m in MODELS for probe in PROBES)

def generate_recipe(seed:int)->dict:
    """Generate an agent view plus an evaluator-only key; split them at the API."""
    if not isinstance(seed,int) or not 0<=seed<=MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    q=_choose(seed,0,(F(3,4),F(4,5),F(5,6),F(7,8)))
    pe=1-2*q+2*q*q
    params={
        "latent_q":q,"p_effect":pe,
        "p_no_dose":_choose(seed,1,(F(1,20),F(1,10),F(3,20))),
        "p_dose":_choose(seed,2,(F(17,20),F(9,10),F(19,20))),
        "sensor_sensitivity":_choose(seed,3,(F(9,10),F(19,20),F(24,25))),
        "sensor_fpr":_choose(seed,4,(F(1,100),F(1,50),F(1,20))),
    }
    if not _validate_params(params): raise AssertionError("generated invalid recipe")
    target=_choose(seed,5,MODELS)
    weights=_choose(seed,6,((1,1,1,1,1,1),(3,1,1,1,1,1),(1,2,3,4,5,6)))
    total=sum(weights); prior={m:F(w,total) for m,w in zip(MODELS,weights)}
    agent_view={
      "benchmark":"HIST-CAUSAL-001","recipe_version":"0.5",
      "task_id":sha256(f"HIST-CAUSAL-001|0.5|{seed}".encode()).hexdigest()[:20],
      "candidate_models":list(MODELS),
      "model_contracts":{
        "genuine_mechanism":"randomized assignment changes specimen; primary and secondary channels reflect specimen; realized action follows command",
        "latent_common_cause":"unobserved variable causes observational assignment/outcome association; randomized assignment does not change specimen",
        "primary_sensor_artefact":"primary measurement path carries assignment-correlated artefact; sensor self-test is informative",
        "actuator_noncompliance":"realized delivery differs probabilistically from command; specimen outcome tracks realized dose",
        "common_mode_reference_artefact":"primary and secondary measurement paths share calibration disturbance; blank/control through that path is informative",
        "null_effect":"no causal response to assignment or realized action"
      },
      "parameter_support":{
        "latent_q":["3/4","4/5","5/6","7/8"],
        "p_no_dose":["1/20","1/10","3/20"],"p_dose":["17/20","9/10","19/20"],
        "sensor_sensitivity":["9/10","19/20","24/25"],
        "sensor_fpr":["1/100","1/50","1/20"]
      },
      "prior":{m:str(prior[m]) for m in MODELS},
      "probes":[
        {"id":"observational_primary","cost":str(COSTS["observational_primary"]),"returns":"(assigned,primary_outcome)"},
        {"id":"randomized_primary","cost":str(COSTS["randomized_primary"]),"returns":"(randomized_assignment,primary_outcome)"},
        {"id":"secondary_reference","cost":str(COSTS["secondary_reference"]),"returns":"(randomized_assignment,secondary_outcome)"},
        {"id":"actuator_readback","cost":str(COSTS["actuator_readback"]),"returns":"(randomized_assignment,realized_action)"},
        {"id":"sensor_self_test","cost":str(COSTS["sensor_self_test"]),"returns":"(randomized_assignment,alarm)"},
        {"id":"blank_common_mode_control","cost":str(COSTS["blank_common_mode_control"]),"returns":"(randomized_assignment,blank_outcome)"}
      ],
      "allowed_outputs":["claim_model","equivalence_class","abstain"],
      "note":"ground-truth model and selected recipe parameters are evaluator-only; pass only agent_view to solver"
    }
    evaluator_key={"seed":seed,"target_model":target,
      "recipe_parameters":{k:str(v) for k,v in params.items()},
      "prior":{k:str(v) for k,v in prior.items()},
      "claim_ceiling":"one of six finite synthetic models; not a physical causal claim"}
    return {"agent_view":agent_view,"evaluator_key":evaluator_key,"_params":params}

def sample_probe(seed:int,probe:str,replicate:int,target_model:str,params:dict)->tuple[int,...]:
    """Deterministic sample for a probe replicate; evaluator only."""
    d=_dist(target_model,probe,params); ordered=sorted(d.items())
    salt=sum((i+1)*b for i,b in enumerate(probe.encode()))
    _,r=_splitmix64((seed^salt^((replicate+1)*0xD6E8FEB86659FD93))&MASK)
    u=F(r,1<<64); cumulative=F(0)
    for outcome,p in ordered:
        cumulative+=p
        if u<cumulative: return outcome
    return ordered[-1][0]

def validate_generated(task:dict)->dict:
    params=task["_params"]
    unresolved=[(a,b) for a,b in combinations(MODELS,2) if _equivalent(a,b,params)]
    return {"valid_mass_and_parameters":_validate_params(params),
      "all_models_separated_by_full_suite":len(unresolved)==0,
      "unresolved_pairs":unresolved,"model_pairs":len(list(combinations(MODELS,2))),
      "target_in_agent_view":"target_model" in task["agent_view"]}

def main():
    rows=[]
    for seed in range(100):
        task=generate_recipe(seed); result=validate_generated(task)
        if not result["valid_mass_and_parameters"] or not result["all_models_separated_by_full_suite"] or result["target_in_agent_view"]:
            raise SystemExit(f"invalid generated seed {seed}: {result}")
        rows.append({"seed":seed,"task_id":task["agent_view"]["task_id"],
          "target_model_evaluator_only":task["evaluator_key"]["target_model"],
          "recipe_parameters":task["evaluator_key"]["recipe_parameters"]})
    print(json.dumps({"recipe_version":"0.5","validated_seed_count":len(rows),
      "all_full_probe_sets_identifiable":True,"examples":rows,
      "claim_ceiling":"public development seeds only; confirmation seeds must remain sealed"},indent=2))

if __name__=="__main__": main()
