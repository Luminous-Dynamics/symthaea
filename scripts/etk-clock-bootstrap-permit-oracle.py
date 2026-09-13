#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent composition: verified bootstrap authority -> evaluation permit V2."""
import argparse, hashlib, json

CLAIM_D=b"symthaea.trust.clock-bootstrap-claim.v2\0"
EVID_D=b"symthaea.trust.clock-bootstrap-authority-evidence.v2\0"
AUTH_D=b"symthaea.trust.verified-clock-bootstrap-authority.v2\0"
POLICY_D=b"symthaea.trust.clock-evaluation-policy.v2\0"
ANCHOR_D=b"symthaea.trust.clock-bootstrap-anchor.v2\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v2\0"
TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64

def policy(step=2000,n=2,div=True):
    if isinstance(step,bool) or not isinstance(step,int) or step<=0 or isinstance(n,bool) or not isinstance(n,int) or n<2:
        raise Denied("invalid_policy")
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v2","policy_record_digest":hx("b"),
         "max_transition_ms":step,"minimum_clock_authority_keys":n,"require_algorithm_diversity":bool(div)}
    return {**pre,"id":dh(POLICY_D,pre)}

def claim(policy_id, trust=TRUST):
    pre={"schema":"symthaea.trust.clock-bootstrap-claim.v2","purpose":"ClockBootstrap",
         "trust_snapshot_digest":trust,"clock_evaluation_policy_id":policy_id,
         "trusted_lower_unix_ms":1_499_000,"trusted_upper_unix_ms":1_499_500}
    return {**pre,"id":dh(CLAIM_D,pre)}

def evidence(c):
    pre={"schema":"symthaea.trust.clock-bootstrap-authority-evidence.v2","claim_id":c["id"],
         "provider_id":"platform-root-01","authority_policy_digest":hx("b"),"external_evidence_digest":hx("c")}
    return {**pre,"id":dh(EVID_D,pre)}

def authority(c,e):
    pre={"schema":"symthaea.trust.verified-clock-bootstrap-authority.v2","claim_id":c["id"],
         "authority_evidence_id":e["id"],"provider_id":"platform-root-01","authority_policy_digest":hx("b")}
    return {**pre,"id":dh(AUTH_D,pre)}

def snapshot():
    return {"digest":TRUST,"issued_at_unix_s":1000,"expires_at_unix_s":2000,"keys":[
      {"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,"not_after_unix_s":3000,
       "status":"Active","usages":["ClockAuthority","ClockContinuity"]},
      {"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,"not_after_unix_s":3000,
       "status":"Active","usages":["ClockAuthority","ClockContinuity"]}]}

def eligible(s,lo,hi,p):
    if s["digest"]!=TRUST: raise Denied("snapshot_digest_mismatch")
    if lo < s["issued_at_unix_s"]*1000 or hi >= s["expires_at_unix_s"]*1000:
        raise Denied("snapshot_not_valid_for_envelope")
    out=[]
    for k in s["keys"]:
        if k["status"]!="Active" or "ClockAuthority" not in k["usages"]: continue
        if lo < k["not_before_unix_s"]*1000: continue
        if k["not_after_unix_s"] is not None and hi >= k["not_after_unix_s"]*1000: continue
        out.append([k["algorithm"],k["key_id"]])
    out.sort()
    if len(out)<p["minimum_clock_authority_keys"]: raise Denied("insufficient_keys_for_envelope")
    if p["require_algorithm_diversity"] and len({x[0] for x in out})<2:
        raise Denied("eligible_algorithm_diversity_missing")
    return out

def permit(auth,c,p,s):
    if auth["claim_id"]!=c["id"]: raise Denied("authority_claim_mismatch")
    if c["clock_evaluation_policy_id"]!=p["id"]: raise Denied("bootstrap_policy_mismatch")
    if c["trust_snapshot_digest"]!=s["digest"]: raise Denied("bootstrap_snapshot_mismatch")
    lo=c["trusted_lower_unix_ms"]; hi=c["trusted_upper_unix_ms"]+p["max_transition_ms"]
    keys=eligible(s,lo,hi,p)
    anchor_pre={"schema":"symthaea.trust.clock-bootstrap-anchor.v2","kind":"ExternalBootstrap",
                "authority_record_digest":auth["id"],"trust_snapshot_digest":s["digest"],
                "trusted_lower_unix_ms":c["trusted_lower_unix_ms"],"trusted_upper_unix_ms":c["trusted_upper_unix_ms"]}
    anchor_id=dh(ANCHOR_D,anchor_pre)
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v2","basis_id":anchor_id,"basis_kind":"BootstrapAnchor",
         "policy_id":p["id"],"trust_snapshot_digest":s["digest"],"evaluation_lower_unix_ms":lo,
         "evaluation_upper_unix_ms":hi,"minimum_clock_authority_keys":p["minimum_clock_authority_keys"],
         "require_algorithm_diversity":p["require_algorithm_diversity"],"eligible_clock_keys":keys}
    return {"anchor_id":anchor_id,**pre,"id":dh(PERMIT_D,pre)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    p=policy(); c=claim(p["id"]); e=evidence(c); a=authority(c,e); s=snapshot(); out=permit(a,c,p,s)
    v={"policy_id":p["id"],"claim_id":c["id"],"verified_authority_id":a["id"],
       "bootstrap_anchor_id":out["anchor_id"],"permit_id":out["id"]}
    expected={
      "policy_id":"7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318",
      "claim_id":"15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877",
      "verified_authority_id":"b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3",
      "bootstrap_anchor_id":"fdfb5cfe737fc326792ed57dfe6aa93b1a2005fef021de6ddcb60e6a56144282",
      "permit_id":"12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9"}
    assert v==expected,(v,expected)
    p2=policy(step=1999)
    deny(lambda:permit(a,c,p2,s),"bootstrap_policy_mismatch")
    s2=snapshot(); s2["digest"]=hx("d")
    deny(lambda:permit(a,c,p,s2),"bootstrap_snapshot_mismatch")
    s3=snapshot(); s3["expires_at_unix_s"]=1501
    deny(lambda:permit(a,c,p,s3),"snapshot_not_valid_for_envelope")
    s4=snapshot(); s4["keys"][1]["status"]="Retired"
    deny(lambda:permit(a,c,p,s4),"insufficient_keys_for_envelope")
    pdiv=policy(div=True); s5=snapshot(); s5["keys"][1]["algorithm"]="Ed25519"
    deny(lambda:permit(authority(claim(pdiv["id"]),evidence(claim(pdiv["id"]))),claim(pdiv["id"]),pdiv,s5),"eligible_algorithm_diversity_missing")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); a=ap.parse_args(); v=run()
    if a.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if a.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
