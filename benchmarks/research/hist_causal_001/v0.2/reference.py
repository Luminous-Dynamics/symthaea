#!/usr/bin/env python3
"""Research-only exact finite identifiability oracle for HIST-CAUSAL-001 v0.2.

Standard-library only. Not a physical model, production subsystem, or evidence
for any cross-system historical effect.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import json

MODELS = ("common_cause", "open_transfer", "sealed_history",
          "sealed_leak", "null", "local_memory")
CONTEXTS = ("observational", "do_open_0", "do_open_1", "do_sealed_0",
            "do_sealed_1", "sealed_audit_0", "sealed_audit_1",
            "same_person_memory_0", "same_person_memory_1",
            "future_exposure_0", "future_exposure_1")
COST = {"observational": Q(1,4), "do_open_0": Q(1), "do_open_1": Q(1),
        "do_sealed_0": Q(1), "do_sealed_1": Q(1),
        "sealed_audit_0": Q(5,2), "sealed_audit_1": Q(5,2),
        "same_person_memory_0": Q(3,4), "same_person_memory_1": Q(3,4),
        "future_exposure_0": Q(1,2), "future_exposure_1": Q(1,2)}
SENS, FPR = Q(19,20), Q(1,100)

def dist(model, context):
    """Return exact P(exposure, outcome, audit_alarm | model, context)."""
    d = {}
    def add(x,y,z,p):
        if p: d[(x,y,z)] = d.get((x,y,z),Q(0)) + p
    transfer = model in ("open_transfer","sealed_history","sealed_leak")
    if context == "observational":
        if model == "common_cause" or transfer:
            add(0,0,0,Q(1,2)); add(1,1,0,Q(1,2))
        else:
            for x,y in product((0,1),repeat=2): add(x,y,0,Q(1,4))
    elif context.startswith("do_open_") or context.startswith("do_sealed_"):
        x = int(context[-1])
        active = transfer if context.startswith("do_open_") else model in ("sealed_history","sealed_leak")
        if active: add(x,x,0,Q(1))
        else: add(x,0,0,Q(1,2)); add(x,1,0,Q(1,2))
    elif context.startswith("sealed_audit_"):
        x = int(context[-1])
        active = model in ("sealed_history","sealed_leak")
        ys = ((x,Q(1)),) if active else ((0,Q(1,2)),(1,Q(1,2)))
        alarm = SENS if model == "sealed_leak" else FPR
        for y,py in ys:
            add(x,y,1,py*alarm); add(x,y,0,py*(1-alarm))
    elif context.startswith("same_person_memory_"):
        local = int(context[-1])
        ys = ((local,Q(1)),) if model == "local_memory" else ((0,Q(1,2)),(1,Q(1,2)))
        for y,py in ys: add(0,y,0,py)
    elif context.startswith("future_exposure_"):
        x = int(context[-1])
        add(x,0,0,Q(1,2)); add(x,1,0,Q(1,2))
    else:
        raise ValueError(context)
    assert sum(d.values(),Q(0)) == 1, (model,context,d)
    return d

def equivalent(a,b,contexts):
    return all(dist(a,c)==dist(b,c) for c in contexts)

def score(contexts):
    pairs=list(combinations(MODELS,2))
    unresolved=[(a,b) for a,b in pairs if equivalent(a,b,contexts)]
    return {"separated_pairs":len(pairs)-len(unresolved),"total_pairs":len(pairs),
            "unresolved_pairs":unresolved}

def min_cost_suite():
    pairs=list(combinations(MODELS,2))
    best=None
    for n in range(len(CONTEXTS)+1):
        for subset in combinations(CONTEXTS,n):
            cost=sum((COST[c] for c in subset),Q(0))
            if best is not None and cost>best[0]: continue
            if all(not equivalent(a,b,subset) for a,b in pairs):
                key=(cost,subset)
                if best is None or key<best: best=key
    if best is None: return None
    return {"contexts":list(best[1]),"cost_exact":str(best[0]),"cost":float(best[0])}

def run():
    observational=("observational",)
    no_audit=("observational","do_open_0","do_open_1","do_sealed_0","do_sealed_1")
    return {
        "benchmark":"HIST-CAUSAL-001","version":"0.2",
        "arithmetic":"exact rational",
        "audit":{"sensitivity":str(SENS),"false_positive":str(FPR)},
        "observational_only":score(observational),
        "interventions_without_audit":score(no_audit),
        "full_suite":score(CONTEXTS),
        "minimum_cost_identification":min_cost_suite(),
        "claim_ceiling":"identifiability only within six declared toy models and this finite probe menu"
    }

if __name__ == "__main__":
    result=run()
    assert result["observational_only"]["separated_pairs"] == 8
    assert result["interventions_without_audit"]["separated_pairs"] == 13
    assert result["full_suite"]["separated_pairs"] == 15
    print(json.dumps(result,indent=2))
