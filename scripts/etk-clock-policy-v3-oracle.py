#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent authority-bound clock quorum/evaluation policy V3 oracle."""
import argparse, hashlib, json

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
QD=b"symthaea.trust.clock-quorum-policy.v1\0"
ED=b"symthaea.trust.clock-evaluation-policy.v3\0"
CLAIM_D=b"symthaea.trust.clock-bootstrap-claim.v2\0"
EVID_D=b"symthaea.trust.clock-bootstrap-authority-evidence.v2\0"
AUTH_D=b"symthaea.trust.verified-clock-bootstrap-authority.v2\0"
ANCHOR_D=b"symthaea.trust.clock-bootstrap-anchor.v2\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v3\0"

EXPECTED={
"quorum_policy_id":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
"evaluation_policy_id":"7a782b8adbf4e66a352c4a26ae614a458971d78c33c136cb8eeee6c6c28433b4",
"claim_id":"840d076db077b2df95277923b4be3b38f1e2a7027222298876041c036456d212",
"authority_evidence_id":"0f3169321314c398a6ca68be67447b14ffd3d4409d228662d1effaa3d60f55a6",
"verified_authority_id":"a0286a04ec3f4c0c6f8f2cc4bf97c7638bed43f913c09456e0931ea9c49e5f7c",
"bootstrap_anchor_id":"61fe43c253ca8dde9c152b641208f4eddc2ddbbfc263124d9b2a712e1e23d9e5",
"permit_id":"716c65376a721574bb53d2415e6b58f83d9c7483666b3799ca8fc4fe214f2cce"}

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64

def quorum(min_sources=2,max_obs=8,max_u=5000,max_w=10000,div=True):
    if min_sources<1 or max_obs<min_sources or max_u<0 or max_w<=0: raise Denied("invalid_quorum_policy")
    pre={"schema":"symthaea.trust.clock-quorum-policy.v1",
         "minimum_distinct_sources":min_sources,"maximum_observations":max_obs,
         "maximum_uncertainty_ms":max_u,"maximum_consensus_width_ms":max_w,
         "require_algorithm_diversity":bool(div)}
    return {**pre,"id":dh(QD,pre)}

def eval_policy(q,step=2000,min_keys=2,key_div=True):
    if step<=0 or min_keys<2: raise Denied("invalid_evaluation_policy")
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v3",
         "policy_record_digest":hx("b"),"clock_quorum_policy_id":q["id"],
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
         "provider_id":"platform-root-01","authority_policy_digest":hx("b"),"external_evidence_digest":hx("c")}
    return {**pre,"id":dh(EVID_D,pre)}

def authority(c,e):
    pre={"schema":"symthaea.trust.verified-clock-bootstrap-authority.v2","claim_id":c["id"],
         "authority_evidence_id":e["id"],"provider_id":"platform-root-01","authority_policy_digest":hx("b")}
    return {**pre,"id":dh(AUTH_D,pre)}

def permit(a,c,ep,q):
    if a["claim_id"]!=c["id"]: raise Denied("authority_claim_mismatch")
    if c["clock_evaluation_policy_id"]!=ep["id"]: raise Denied("evaluation_policy_mismatch")
    if ep["clock_quorum_policy_id"]!=q["id"]: raise Denied("quorum_policy_mismatch")
    lo=c["trusted_lower_unix_ms"]; hi=c["trusted_upper_unix_ms"]+ep["max_transition_ms"]
    anchor_pre={"schema":"symthaea.trust.clock-bootstrap-anchor.v2","kind":"ExternalBootstrap",
                "authority_record_digest":a["id"],"trust_snapshot_digest":TRUST,
                "trusted_lower_unix_ms":c["trusted_lower_unix_ms"],"trusted_upper_unix_ms":c["trusted_upper_unix_ms"]}
    anchor=dh(ANCHOR_D,anchor_pre)
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v3","basis_id":anchor,"basis_kind":"BootstrapAnchor",
         "evaluation_policy_id":ep["id"],"clock_quorum_policy_id":q["id"],"trust_snapshot_digest":TRUST,
         "evaluation_lower_unix_ms":lo,"evaluation_upper_unix_ms":hi,
         "minimum_eligible_clock_authority_keys":ep["minimum_eligible_clock_authority_keys"],
         "require_eligible_algorithm_diversity":ep["require_eligible_algorithm_diversity"],
         "eligible_clock_keys":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
    return {"anchor_id":anchor,**pre,"id":dh(PERMIT_D,pre)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    q=quorum(); ep=eval_policy(q); c=claim(ep); e=evidence(c); a=authority(c,e); p=permit(a,c,ep,q)
    v={"quorum_policy_id":q["id"],"evaluation_policy_id":ep["id"],"claim_id":c["id"],
       "authority_evidence_id":e["id"],"verified_authority_id":a["id"],
       "bootstrap_anchor_id":p["anchor_id"],"permit_id":p["id"]}
    assert v==EXPECTED,(v,EXPECTED)

    q_weak=quorum(max_u=10000)
    deny(lambda:permit(a,c,ep,q_weak),"quorum_policy_mismatch")

    ep_weak=eval_policy(q_weak)
    deny(lambda:permit(a,c,ep_weak,q_weak),"evaluation_policy_mismatch")

    c_weak=claim(ep_weak); e_weak=evidence(c_weak); a_weak=authority(c_weak,e_weak)
    p_weak=permit(a_weak,c_weak,ep_weak,q_weak)
    assert p_weak["id"]!=p["id"]

    q_wide=quorum(max_w=20000)
    assert q_wide["id"]!=q["id"]
    assert eval_policy(q_wide)["id"]!=ep["id"]
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else:
        print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
