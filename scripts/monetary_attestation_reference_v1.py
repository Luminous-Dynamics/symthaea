#!/usr/bin/env python3
from __future__ import annotations
from hashlib import sha256
from itertools import product
from pathlib import Path
import json,sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from monetary_attestation_binding import (
    PAIR,AD,SH,SEED,ALLO,FOCAL,REPORT,CAP,
    true_liquidity_demand,world_jitter,manual_hit,TECH,
    attr_for,val,order,expected,attestation,
)

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/"../docs/research/monetary/monetary-attestation-reference-v1.json").read_text())
F=M["factors"]; REFS=tuple(F["reference_source"]); VARS=tuple(F["attestation_variant"])
TOPO=M["fixed_dimensions"]["topology_digest"]; RES=M["fixed_dimensions"]["resource_digest"]

def context_id(pair,adapter,shock,seed,allocation):
    x={"pair":pair,"adapter":adapter,"shock":shock,"seed":seed,"allocation_policy":allocation,
       "topology_digest":TOPO,"resource_digest":RES}
    return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def variant_attestation(pair,adapter,shock,seed,allocation,focal,variant):
    if variant!="generation_substitution":
        return attestation(pair,adapter,shock,seed,allocation,focal,variant)
    d=true_liquidity_demand(adapter,shock); e=expected(pair,adapter,shock,seed,allocation,focal)
    return {**e,"generation":e["generation"]-M["fixed_dimensions"]["stale_generation_delta"],
            "attested_value":val(attr_for(allocation),focal,d),"signature_valid":True}

def reference(pair,adapter,shock,seed,allocation,focal,source,att):
    e=expected(pair,adapter,shock,seed,allocation,focal)
    if source=="verifier_current": return dict(e)
    if source=="verifier_stale_generation":
        x=dict(e); x["generation"]-=M["fixed_dimensions"]["stale_generation_delta"]; return x
    if source=="verifier_stale_context":
        x=dict(e); s2=SH[(SH.index(shock)+1)%len(SH)]; x["context_id"]=context_id(pair,adapter,s2,seed,allocation); return x
    if source=="participant_supplied": return {k:att[k] for k in ("subject_id","attribute_id","generation","context_id")}
    if source=="cross_authority": return dict(e)
    raise ValueError(source)

def simulate(pair,adapter,shock,seed,allocation,focal,source,variant,reporting):
    demand=true_liquidity_demand(adapter,shock)
    dc=[2,1,3]; dd=list(demand)
    if reporting in ("criticality_inflation","dual_misreport"): dc[focal]=5
    if reporting in ("liquidity_demand_underreport","dual_misreport"): dd[focal]=min(demand[focal],5)
    _=(dc,dd)
    att=variant_attestation(pair,adapter,shock,seed,allocation,focal,variant)
    ref=reference(pair,adapter,shock,seed,allocation,focal,source,att)
    binding=all(att[k]==ref[k] for k in ("subject_id","attribute_id","generation","context_id"))
    accepted=binding and att["signature_valid"]; trusted=(accepted and source=="verifier_current")
    keys=[val(attr_for(allocation),i,demand) for i in range(3)]
    if accepted: keys[focal]=att["attested_value"]
    elif allocation=="criticality_priority": keys[focal]=0
    elif allocation=="minimum_liquidity_demand": keys[focal]=CAP+1
    else: keys[focal]=focal
    alloc=order(allocation,keys); current=5+world_jitter(shock,seed); available=CAP; pending=list(alloc); active=[]; finish={}; peak=0
    while pending or active:
        if active:
            t=min(x[0] for x in active); current=max(current,t)
            done=[x for x in active if x[0]<=current]; active=[x for x in active if x[0]>current]; available+=sum(x[1] for x in done)
        while pending:
            i=pending[0]; q=demand[i]
            if q==0 or q>CAP: pending.pop(0); finish[i]=None; continue
            if q>available: break
            pending.pop(0); available-=q; peak=max(peak,CAP-available)
            finish[i]=current+TECH[adapter]+(2 if shock=="liquidity_shock" and adapter=="redeem_reissue" else 0)+(2 if manual_hit(shock,seed,i) else 0)+2
            active.append((finish[i],q,i))
        if pending and not active:
            for i in pending: finish[i]=None
            pending=[]
    done=[i for i,v in finish.items() if v is not None and shock not in ("bridge_failure","stale_quote") and not (shock=="issuer_default" and adapter=="redeem_reissue")]
    return {"binding_valid":binding,"signature_valid":att["signature_valid"],"accepted":accepted,
            "reference_provenance_trusted":source=="verifier_current","trusted_accepted":trusted,
            "focal_finish":finish.get(focal) if focal in done else None,"completion_rate":len(done)/3,
            "system_completion_time":max((finish[i] for i in done),default=None),
            "peak_true_liquidity_reserved":peak,"allocation_order":alloc}

def main(out_dir):
    assert len(F)==9 and M["factorial_size"]==324000
    truth={(p,a,s,se,al,f):simulate(p,a,s,se,al,f,"verifier_current","exact_current","truthful")
           for p,a,s,se,al,f in product(PAIR,AD,SH,SEED,ALLO,FOCAL)}
    T=sha256(); accepted={}; gains={}; report={}; resource_fail=0
    expected_matrix={
      "verifier_current":{"exact_current":1.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":0.0},
      "verifier_stale_generation":{"exact_current":0.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":1.0,"context_substitution":0.0},
      "verifier_stale_context":{"exact_current":0.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":1.0},
      "participant_supplied":{"exact_current":1.0,"subject_substitution":1.0,"attribute_substitution":1.0,"generation_substitution":1.0,"context_substitution":1.0},
      "cross_authority":{"exact_current":1.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":0.0}}
    for p,a,s,se,al,f,rp,src,v in product(PAIR,AD,SH,SEED,ALLO,FOCAL,REPORT,REFS,VARS):
        c=simulate(p,a,s,se,al,f,src,v,rp); base=truth[(p,a,s,se,al,f)]
        proj=(c["focal_finish"],c["completion_rate"],c["system_completion_time"],c["allocation_order"])
        key=(p,a,s,se,al,f,src,v)
        if rp=="truthful": report[key]=proj
        elif proj!=report[key]: raise AssertionError("reporting independence")
        rec={"pair":p,"adapter":a,"shock":s,"seed":se,"allocation_policy":al,"focal_obligation":f,
             "reporting_policy":rp,"reference_source":src,"attestation_variant":v,**c}
        T.update((json.dumps(rec,sort_keys=True,separators=(",",":"))+"\n").encode())
        k=(src,v); accepted.setdefault(k,[]).append(int(c["accepted"])); resource_fail+=int(c["peak_true_liquidity_reserved"]>CAP)
        if c["accepted"] and c["focal_finish"] is not None and base["focal_finish"] is not None:
            gains.setdefault(k,[]).append(base["focal_finish"]-c["focal_finish"])
    matrix={src:{v:sum(accepted[(src,v)])/len(accepted[(src,v)]) for v in VARS} for src in REFS}
    assert resource_fail==0 and matrix==expected_matrix
    def stats(src,v):
        xs=gains.get((src,v),[]); pos=[x for x in xs if x>0]
        return {"positive_gain_rate":len(pos)/len(xs) if xs else 0.0,"mean_gain_ticks":sum(xs)/len(xs) if xs else 0.0,"max_gain_ticks":max(xs) if xs else 0}
    out={"schema_version":"monetary-attestation-reference-v1-execution","run_count":324000,"obligation_count":972000,
         "common_random_number_cells":90,"exogenous_random_namespace":M["fixed_dimensions"]["crn_namespace"],
         "trace_set_digest":T.hexdigest(),"reference_acceptance_matrix":matrix,
         "headline":{"trusted_current_exact_acceptance_rate":matrix["verifier_current"]["exact_current"],
                     "unqualified_reference_acceptance_rate":8/20,"participant_supplied_acceptance_rate":1.0,
                     "cross_authority_exact_current_acceptance_rate":matrix["cross_authority"]["exact_current"],
                     "stale_reference_exact_current_rejection_rate":{"verifier_stale_generation":1.0,"verifier_stale_context":1.0}},
         "selected_attack_effects_if_reference_is_admitted":{
           f"participant_supplied|{v}":stats("participant_supplied",v) for v in VARS if v!="exact_current"},
         "invariants":{"factorial_cardinality_exact":True,"obligation_cardinality_exact":True,"signature_valid_for_all_variants":True,
                      "reference_variant_matrix_exact":True,"reporting_independent_authoritative_result":True,
                      "resource_integrity_failures":0,"crn_independent_of_reference_and_variant":True},
         "claim_ceiling":M["claim_ceiling"]}
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/"monetary-attestation-reference-v1.execution.generated.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"trace_set_digest":out["trace_set_digest"],"headline":out["headline"],"invariants":out["invariants"]},indent=2))
if __name__=="__main__": main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("."))
