#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent V2 clock-bootstrap external-authority binding theorem.

The initial clock bootstrap claim is descriptive until an external authority
adapter authenticates evidence bound to that exact claim. V2 additionally
binds the exact initial ClockEvaluationPolicyV2 identity, preventing a valid
bootstrap authority from being paired with a caller-chosen transition policy.
No candidate clock window/time participates in bootstrap authentication.
"""
import argparse, hashlib, json

CLAIM_D=b"symthaea.trust.clock-bootstrap-claim.v2\0"
EVID_D=b"symthaea.trust.clock-bootstrap-authority-evidence.v2\0"
CAP_D=b"symthaea.trust.verified-clock-bootstrap-authority.v2\0"
TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
POLICY="7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318"

def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64
class Denied(ValueError): pass

def dg(v,n):
    if not isinstance(v,str) or len(v)!=64 or v.lower()!=v or any(c not in "0123456789abcdef" for c in v):
        raise Denied("invalid_digest:"+n)
    return v

def claim(lower=1_499_000,upper=1_499_500,trust=TRUST,policy=POLICY):
    if isinstance(lower,bool) or isinstance(upper,bool) or not isinstance(lower,int) or not isinstance(upper,int) or lower>upper:
        raise Denied("invalid_claim_window")
    pre={"schema":"symthaea.trust.clock-bootstrap-claim.v2","purpose":"ClockBootstrap",
         "trust_snapshot_digest":dg(trust,"trust_snapshot"),
         "clock_evaluation_policy_id":dg(policy,"clock_evaluation_policy"),
         "trusted_lower_unix_ms":lower,"trusted_upper_unix_ms":upper}
    return {**pre,"id":dh(CLAIM_D,pre)}

def evidence(c,provider="platform-root-01",authority_policy=hx("b"),external=hx("c")):
    if not provider or provider!=provider.strip(): raise Denied("invalid_provider")
    pre={"schema":"symthaea.trust.clock-bootstrap-authority-evidence.v2",
         "claim_id":dg(c["id"],"claim_id"),"provider_id":provider,
         "authority_policy_digest":dg(authority_policy,"authority_policy"),
         "external_evidence_digest":dg(external,"external_evidence")}
    return {**pre,"id":dh(EVID_D,pre)}

def valid_claim(c):
    keys=("schema","purpose","trust_snapshot_digest","clock_evaluation_policy_id","trusted_lower_unix_ms","trusted_upper_unix_ms")
    pre={k:c[k] for k in keys}
    if c.get("id")!=dh(CLAIM_D,pre): raise Denied("claim_id_mismatch")
    if c["schema"]!="symthaea.trust.clock-bootstrap-claim.v2" or c["purpose"]!="ClockBootstrap" or c["trusted_lower_unix_ms"]>c["trusted_upper_unix_ms"]:
        raise Denied("invalid_claim")

def valid_evidence(e):
    keys=("schema","claim_id","provider_id","authority_policy_digest","external_evidence_digest")
    pre={k:e[k] for k in keys}
    if e.get("id")!=dh(EVID_D,pre): raise Denied("evidence_id_mismatch")
    if e["schema"]!="symthaea.trust.clock-bootstrap-authority-evidence.v2" or not e["provider_id"] or e["provider_id"]!=e["provider_id"].strip():
        raise Denied("invalid_evidence")

def verify_external(c,e,configured_provider,configured_authority_policy,decision=True):
    valid_claim(c); valid_evidence(e)
    if e["claim_id"]!=c["id"]: raise Denied("evidence_claim_mismatch")
    if e["provider_id"]!=configured_provider: raise Denied("provider_mismatch")
    if e["authority_policy_digest"]!=configured_authority_policy: raise Denied("authority_policy_mismatch")
    if decision is not True: raise Denied("external_authority_rejected")
    pre={"schema":"symthaea.trust.verified-clock-bootstrap-authority.v2",
         "claim_id":c["id"],"authority_evidence_id":e["id"],"provider_id":e["provider_id"],
         "authority_policy_digest":e["authority_policy_digest"]}
    return {**pre,"id":dh(CAP_D,pre)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    c=claim(); e=evidence(c); cap=verify_external(c,e,"platform-root-01",hx("b"))
    v={"claim_id":c["id"],"authority_evidence_id":e["id"],"verified_authority_id":cap["id"]}
    expected={
      "claim_id":"15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877",
      "authority_evidence_id":"62991edeba3a575b28801bcceaca99dceaec476a0bab699792df6fbe2ec0cee7",
      "verified_authority_id":"b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3"}
    assert v==expected,(v,expected)
    tampered=dict(c); tampered["trusted_upper_unix_ms"]+=1
    deny(lambda:verify_external(tampered,e,"platform-root-01",hx("b")),"claim_id_mismatch")
    changed_policy=claim(policy=hx("e"))
    deny(lambda:verify_external(changed_policy,e,"platform-root-01",hx("b")),"evidence_claim_mismatch")
    bad=dict(e); bad["external_evidence_digest"]=hx("d")
    deny(lambda:verify_external(c,bad,"platform-root-01",hx("b")),"evidence_id_mismatch")
    deny(lambda:verify_external(c,e,"wrong-provider",hx("b")),"provider_mismatch")
    deny(lambda:verify_external(c,e,"platform-root-01",hx("e")),"authority_policy_mismatch")
    deny(lambda:verify_external(c,e,"platform-root-01",hx("b"),False),"external_authority_rejected")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); a=ap.parse_args(); v=run()
    if a.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if a.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
