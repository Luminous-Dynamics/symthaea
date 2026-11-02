#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.4 exact causal-triangulation reference.
Finite toy models; exact rational arithmetic; stdlib only.
Research reference, not a physical model.
"""
from fractions import Fraction as F
from itertools import combinations, product

MODELS = ("genuine_mechanism","latent_common_cause","primary_sensor_artefact",
          "actuator_noncompliance","common_mode_reference_artefact","null_effect")
P_BASE, P_EFFECT = F(8,25), F(17,25)
P_DOSE_BASE, P_DOSE_EFFECT = F(1,10), F(9,10)
P_DELIVERED_Z0, P_DELIVERED_Z1 = F(11,40), F(29,40)
SENSOR_SENS, SENSOR_FPR = F(19,20), F(1,100)
PROBES = {
    "observational_primary": F(1,4), "randomized_primary": F(1),
    "secondary_reference": F(2), "actuator_readback": F(3,2),
    "sensor_self_test": F(1), "blank_common_mode_control": F(2),
}

def add(d, outcome, p):
    if p: d[outcome] = d.get(outcome,F(0)) + p

def binary(p1):
    return ((1,p1),(0,1-p1))

def obs(model):
    d={}
    if model=="null_effect":
        for z,y in product((0,1),repeat=2): add(d,(z,y),F(1,4))
    else:
        for outcome,p in (((0,0),F(17,50)),((0,1),F(8,50)),
                          ((1,0),F(8,50)),((1,1),F(17,50))): add(d,outcome,p)
    return d

def dose_response(z,pdel):
    p1=P_DOSE_BASE*(1-pdel)+P_DOSE_EFFECT*pdel
    return binary(p1)

def distribution(model,probe):
    if model not in MODELS or probe not in PROBES: raise ValueError("unknown model or probe")
    d={}
    if probe=="observational_primary": return obs(model)
    if probe=="randomized_primary":
        for z in (0,1):
            if model in ("genuine_mechanism","primary_sensor_artefact","actuator_noncompliance","common_mode_reference_artefact"):
                ys=dose_response(z,P_DELIVERED_Z1 if z else P_DELIVERED_Z0) if model=="actuator_noncompliance" else binary(P_EFFECT if z else P_BASE)
            else: ys=((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add(d,(z,y),py*F(1,2))
    elif probe=="secondary_reference":
        for z in (0,1):
            if model in ("genuine_mechanism","actuator_noncompliance"):
                ys=dose_response(z,P_DELIVERED_Z1 if z else P_DELIVERED_Z0) if model=="actuator_noncompliance" else binary(P_EFFECT if z else P_BASE)
            elif model=="common_mode_reference_artefact": ys=binary(P_EFFECT if z else P_BASE)
            else: ys=((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add(d,(z,y),py*F(1,2))
    elif probe=="actuator_readback":
        for z in (0,1):
            delivered=binary(P_DELIVERED_Z1 if z else P_DELIVERED_Z0) if model=="actuator_noncompliance" else ((z,F(1)),)
            for actual,pa in delivered: add(d,(z,actual),pa*F(1,2))
    elif probe=="sensor_self_test":
        alarm=SENSOR_SENS if model=="primary_sensor_artefact" else SENSOR_FPR
        for z in (0,1):
            for a,pa in ((1,alarm),(0,1-alarm)): add(d,(z,a),pa*F(1,2))
    elif probe=="blank_common_mode_control":
        for z in (0,1):
            ys=binary(P_EFFECT if z else P_BASE) if model=="common_mode_reference_artefact" else ((0,F(1,2)),(1,F(1,2)))
            for y,py in ys: add(d,(z,y),py*F(1,2))
    if sum(d.values(),F(0))!=1: raise ArithmeticError(f"non-normalized {model}/{probe}")
    return d

def equivalent(a,b,probes):
    return all(distribution(a,p)==distribution(b,p) for p in probes)

def score(probes):
    pairs=list(combinations(MODELS,2))
    unresolved=[(a,b) for a,b in pairs if equivalent(a,b,probes)]
    return {"probes":list(probes),"pairs_total":len(pairs),
            "pairs_separated":len(pairs)-len(unresolved),"unresolved_pairs":unresolved}

def minimum_cost_suite():
    pairs=list(combinations(MODELS,2)); best=None
    for n in range(len(PROBES)+1):
        for subset in combinations(PROBES,n):
            if all(not equivalent(a,b,subset) for a,b in pairs):
                cost=sum((PROBES[p] for p in subset),F(0)); key=(cost,subset)
                if best is None or key<best: best=key
    if best is None: return {"identified":False}
    return {"identified":True,"probes":list(best[1]),"cost_exact":str(best[0]),"cost":float(best[0])}

def run():
    obs_only=("observational_primary",)
    rand=obs_only+("randomized_primary",)
    ref=rand+("secondary_reference",)
    actuator=ref+("actuator_readback",)
    return {
        "benchmark":"HIST-CAUSAL-001","version":"0.4","arithmetic":"exact rational",
        "model_count":len(MODELS),"pair_count":15,
        "observational_only":score(obs_only),"randomized_primary":score(rand),
        "secondary_reference":score(ref),"actuator_witness":score(actuator),
        "full_witness_suite":score(PROBES),
        "minimum_cost_probe_suite":minimum_cost_suite(),
        "parameters":{"p_base":str(P_BASE),"p_effect":str(P_EFFECT),
          "p_delivery_z0":str(P_DELIVERED_Z0),"p_delivery_z1":str(P_DELIVERED_Z1),
          "sensor_sensitivity":str(SENSOR_SENS),"sensor_fpr":str(SENSOR_FPR)},
        "claim_ceiling":"identifiability within six finite toy models and the declared probe menu only"
    }

if __name__=="__main__":
    import json
    result=run()
    assert result["full_witness_suite"]["pairs_separated"]==15
    print(json.dumps(result,indent=2))
