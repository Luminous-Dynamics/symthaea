#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent unified clock policy V4 oracle: quorum + continuity under one authority."""
import argparse, hashlib, json

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
QD=b"symthaea.trust.clock-quorum-policy.v1\0"
CD=b"symthaea.trust.clock-continuity-policy.v1\0"
ED=b"symthaea.trust.clock-evaluation-policy.v4\0"
CLAIM_D=b"symthaea.trust.clock-bootstrap-claim.v2\0"
EVID_D=b"symthaea.trust.clock-bootstrap-authority-evidence.v2\0"
AUTH_D=b"symthaea.trust.verified-clock-bootstrap-authority.v2\0"
ANCHOR_D=b"symthaea.trust.clock-bootstrap-anchor.v2\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v4\0"

EXPECTED={
"quorum_policy_id":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
"continuity_policy_id":"904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06",
"evaluation_policy_id":"4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234",
"claim_id":"c913a1829fe0bde63bbd2f0a4fda9344ed4de09fb9dece23ebf3189cb39e9fbd",
"authority_evidence_id":"42d1b984349521b7d9dfc0d7ef149094a15231bd557c698acc7e276efb8a901d",
"verified_authority_id":"aecff7b0ad2a40e2ab4a0dd7fc6dca4684046cceda5df0b36071e6bf623f8582",
"bootstrap_anchor_id":"c9fc46d4bbdccef92c2bc86c3682f2319d6837f753cf954fff30dcd451b5b588",
"permit_id":"1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f"}

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64

def quorum(max_u=5000,max_w=10000,min_sources=2,max_obs=8,div=True):
    if min_sources<1 or max_obs<min_sources or max_w<=0: raise Denied("invalid_quorum_policy")
    pre={"schema":"symthaea.trust.clock-quorum-policy.v1","minimum_distinct_sources":min_sources,
         "maximum_observations":max_obs,"maximum_uncertainty_ms":max_u,
         "maximum_consensus_width_ms":max_w,"require_algorithm_diversity":bool(div)}
    return {**pre,"id":dh(QD,pre)}

def continuity(max_epoch=1,max_gap=10000,max_jump=60000,min_shared=1,shared_alg=True):
    if max_epoch<=0 or max_jump<=0 or min_shared<=0: raise Denied("invalid_continuity_policy")
    pre={"schema":"symthaea.trust.clock-continuity-policy.v1","maximum_epoch_step":max_epoch,
         "maximum_forward_gap_ms":max_gap,"maximum_consensus_jump_ms":max_jump,
         "minimum_shared_sources":min_shared,"require_shared_algorithm":bool(shared_alg)}
    return {**pre,"id":dh(CD,pre)}

def eval_policy(q,c,step=2000,min_keys=2,key_div=True):
    if step<=0 or min_keys<2: raise Denied("invalid_evaluation_policy")
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v4","policy_record_digest":hx("b"),
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],
         "max_transition_ms":step,"minimum_eligible_clock_authority_keys":min_keys,
         "require_eligible_algorithm_diversity":bool(key_div)}
    return {**pre,"id":dh(ED,pre)}

def claim(ep):
    pre={"schema":"symthaea.trust.clock-bootstrap-claim.v2","purpose":"ClockBootstrap",
         "trust_snapshot_digest":TRUST,"clock_evaluation_policy_id":ep["id"],
         "trusted_lower_unix_ms":1499000,"trusted_upper_unix_ms":1499500}
    return {**pre,"id":dh(CLAIM_D,pre)}

def evidence(c):
    pre={"schema":"symthaea.trust.clock-bootstrap-authority-evidence.v2","claim_id":c["id"],
         "provider_id":"platform-root-01","authority_policy_digest":hx("b"),
         "external_evidence_digest":hx("c")}
    return {**pre,"id":dh(EVID_D,pre)}

def authority(c,e):
    pre={"schema":"symthaea.trust.verified-clock-bootstrap-authority.v2","claim_id":c["id"],
         "authority_evidence_id":e["id"],"provider_id":"platform-root-01",
         "authority_policy_digest":hx("b")}
    return {**pre,"id":dh(AUTH_D,pre)}

def permit(a,c,ep,q,cp):
    if a["claim_id"]!=c["id"]: raise Denied("authority_claim_mismatch")
    if c["clock_evaluation_policy_id"]!=ep["id"]: raise Denied("evaluation_policy_mismatch")
    if ep["clock_quorum_policy_id"]!=q["id"]: raise Denied("quorum_policy_mismatch")
    if ep["clock_continuity_policy_id"]!=cp["id"]: raise Denied("continuity_policy_mismatch")
    lo=c["trusted_lower_unix_ms"]; hi=c["trusted_upper_unix_ms"]+ep["max_transition_ms"]
    anchor_pre={"schema":"symthaea.trust.clock-bootstrap-anchor.v2","kind":"ExternalBootstrap",
                "authority_record_digest":a["id"],"trust_snapshot_digest":TRUST,
                "trusted_lower_unix_ms":c["trusted_lower_unix_ms"],
                "trusted_upper_unix_ms":c["trusted_upper_unix_ms"]}
    anchor=dh(ANCHOR_D,anchor_pre)
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v4","basis_id":anchor,
         "basis_kind":"BootstrapAnchor","evaluation_policy_id":ep["id"],
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "trust_snapshot_digest":TRUST,"evaluation_lower_unix_ms":lo,
         "evaluation_upper_unix_ms":hi,
         "minimum_eligible_clock_authority_keys":ep["minimum_eligible_clock_authority_keys"],
         "require_eligible_algorithm_diversity":ep["require_eligible_algorithm_diversity"],
         "eligible_clock_keys":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
    return {"anchor_id":anchor,**pre,"id":dh(PERMIT_D,pre),
            "clock_quorum_policy":dict(q),"clock_continuity_policy":dict(cp)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    q=quorum(); cp=continuity(); ep=eval_policy(q,cp); cl=claim(ep); ev=evidence(cl); au=authority(cl,ev)
    p=permit(au,cl,ep,q,cp)
    v={"quorum_policy_id":q["id"],"continuity_policy_id":cp["id"],
       "evaluation_policy_id":ep["id"],"claim_id":cl["id"],
       "authority_evidence_id":ev["id"],"verified_authority_id":au["id"],
       "bootstrap_anchor_id":p["anchor_id"],"permit_id":p["id"]}
    assert v==EXPECTED,(v,EXPECTED)

    weak_cp=continuity(max_gap=20000)
    deny(lambda:permit(au,cl,ep,q,weak_cp),"continuity_policy_mismatch")
    ep_weak_cp=eval_policy(q,weak_cp)
    deny(lambda:permit(au,cl,ep_weak_cp,q,weak_cp),"evaluation_policy_mismatch")

    weak_q=quorum(max_u=10000)
    deny(lambda:permit(au,cl,ep,weak_q,cp),"quorum_policy_mismatch")

    cl2=claim(ep_weak_cp); ev2=evidence(cl2); au2=authority(cl2,ev2)
    p2=permit(au2,cl2,ep_weak_cp,q,weak_cp)
    assert p2["id"]!=p["id"]
    assert p["clock_continuity_policy"]["id"]==cp["id"]
    assert p["clock_quorum_policy"]["id"]==q["id"]
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else:
        print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
