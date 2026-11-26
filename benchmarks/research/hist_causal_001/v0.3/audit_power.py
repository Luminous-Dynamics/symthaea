#!/usr/bin/env python3
"""Research-only exact binomial sensitivity for HIST-CAUSAL-001 audit quality."""
from fractions import Fraction as Q
from math import comb
import json

THRESHOLD=Q(95,100)
AUDITS={"high":(Q(95,100),Q(1,100)),
        "moderate":(Q(80,100),Q(5,100)),
        "weak":(Q(70,100),Q(10,100)),
        "poor":(Q(60,100),Q(20,100))}
PRIORS={"rare_leak":Q(1,10),"balanced":Q(1,2),"common_leak":Q(9,10)}

def binom(n,k,p):
    return Q(comb(n,k),1)*p**k*(1-p)**(n-k)

def posterior(n,k,sens,fpr,prior):
    lh=binom(n,k,fpr); ll=binom(n,k,sens)
    a=prior*ll; b=(1-prior)*lh
    return a/(a+b) if a+b else prior

def profile(n,sens,fpr,truth,prior):
    p=sens if truth=="leak" else fpr
    correct=wrong=abstain=Q(0)
    for k in range(n+1):
        mass=binom(n,k,p); post=posterior(n,k,sens,fpr,prior)
        if post>=THRESHOLD:
            if truth=="leak": correct+=mass
            else: wrong+=mass
        elif post<=1-THRESHOLD:
            if truth=="history": correct+=mass
            else: wrong+=mass
        else: abstain+=mass
    assert correct+wrong+abstain==1
    return {"correct_confident":correct,"wrong_confident":wrong,"abstain":abstain}

def min_n(sens,fpr,prior,max_n=256):
    for n in range(1,max_n+1):
        h=profile(n,sens,fpr,"history",prior)
        l=profile(n,sens,fpr,"leak",prior)
        if min(h["correct_confident"],l["correct_confident"])>=Q(9,10) and max(h["wrong_confident"],l["wrong_confident"])<=Q(1,100):
            return n
    return None

def run():
    out={}
    for name,(sens,fpr) in AUDITS.items():
        out[name]={}
        for prior_name,prior in PRIORS.items():
            out[name][prior_name]={
                "sensitivity":str(sens),"false_positive_rate":str(fpr),
                "prior_leak":str(prior),"minimum_independent_units":min_n(sens,fpr,prior)}
    return {"thresholds":{"posterior_confidence":"0.95","max_wrong_confident":"0.01","min_correct_confident":"0.90"},
            "results":out,
            "assumptions":"independent Bernoulli audits with known sensitivity/FPR and fixed prior; toy design sensitivity only"}
if __name__=="__main__":
    result=run()
    assert result["results"]["high"]["balanced"]["minimum_independent_units"]==2
    assert result["results"]["poor"]["balanced"]["minimum_independent_units"]==23
    print(json.dumps(result,indent=2))
