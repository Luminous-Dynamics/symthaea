#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent clock-bootstrap external-authority binding theorem.

The initial clock bootstrap claim is descriptive until an external authority
adapter authenticates evidence bound to that exact claim. This theorem does not
implement TPM/HSM/operator cryptography; it freezes the binding contract that a
production adapter must satisfy without consulting the candidate clock itself.
"""
import argparse, hashlib, json

CLAIM_D=b"symthaea.trust.clock-bootstrap-claim.v1\0"
EVID_D=b"symthaea.trust.clock-bootstrap-authority-evidence.v1\0"
CAP_D=b"symthaea.trust.verified-clock-bootstrap-authority.v1\0"
TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"

def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64
class Denied(ValueError): pass

def dg(v,n):
    if not isinstance(v,str) or len(v)!=64 or v.lower()!=v or any(c not in "0123456789abcdef" for c in v):
        raise Denied("invalid_digest:"+n)
    return v

def claim(lower=1_499_000,upper=1_499_500,trust=TRUST):
    if isinstance(lower,bool) or isinstance(upper,bool) or not isinstance(lower,int) or not isinstance(upper,int) or lower>upper:
        raise Denied("invalid_claim_window")
    pre={"schema":"symthaea.trust.clock-bootstrap-claim.v1","purpose":"ClockBootstrap",
         "trust_snapshot_digest":dg(trust,"trust_snapshot"),"trusted_lower_unix_ms":lower,"trusted_upper_unix_ms":upper}
    return {**pre,"id":dh(CLAIM_D,pre)}

def evidence(c,provider="platform-root-01",policy=hx("b"),external=hx("c")):
    if not provider or provider!=provider.strip(): raise Denied("invalid_provider")
    pre={"schema":"symthaea.trust.clock-bootstrap-authority-evidence.v1","claim_id":dg(c["id"],"claim_id"),
         "provider_id":provider,"authority_policy_digest":dg(policy,"authority_policy"),
         "external_evidence_digest":dg(external,"external_evidence")}
    return {**pre,"id":dh(EVID_D,pre)}

def valid_claim(c):
    pre={k:c[k] for k in ("schema","purpose","trust_snapshot_digest","trusted_lower_unix_ms","trusted_upper_unix_ms")}
    if c.get("id")!=dh(CLAIM_D,pre): raise Denied("claim_id_mismatch")
    if c["schema"]!="symthaea.trust.clock-bootstrap-claim.v1" or c["purpose"]!="ClockBootstrap" or c["trusted_lower_unix_ms"]>c["trusted_upper_unix_ms"]:
        raise Denied("invalid_claim")

def valid_evidence(e):
    pre={k:e[k] for k in ("schema","claim_id","provider_id","authority_policy_digest","external_evidence_digest")}
    if e.get("id")!=dh(EVID_D,pre): raise Denied("evidence_id_mismatch")
    if e["schema"]!="symthaea.trust.clock-bootstrap-authority-evidence.v1" or not e["provider_id"] or e["provider_id"]!=e["provider_id"].strip():
        raise Denied("invalid_evidence")

def verify_external(c,e,configured_provider,configured_policy,decision=True):
    # Production calls an external cryptographic/hardware/operator verifier here.
    # This reference models only the trust-kernel side of that boundary.
    valid_claim(c); valid_evidence(e)
    if e["claim_id"]!=c["id"]: raise Denied("evidence_claim_mismatch")
    if e["provider_id"]!=configured_provider: raise Denied("provider_mismatch")
    if e["authority_policy_digest"]!=configured_policy: raise Denied("authority_policy_mismatch")
    if decision is not True: raise Denied("external_authority_rejected")
    pre={"schema":"symthaea.trust.verified-clock-bootstrap-authority.v1","claim_id":c["id"],
         "authority_evidence_id":e["id"],"provider_id":e["provider_id"],
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
    vectors={"claim_id":c["id"],"authority_evidence_id":e["id"],"verified_authority_id":cap["id"]}
    expected={
      "claim_id":"5bebb59ba5b95bd487977ffd296bf398e815a80eebca36813f80344e3b68439d",
      "authority_evidence_id":"1bf502c3fe07d5dc7151b35eb93e6c12e49bb1f465ae55b74a60ca3aabc09d7d",
      "verified_authority_id":"0298a2d298c7e1f1cfebee21fb91aac61c08f749f11de368910f4d3118e48760"}
    assert vectors==expected,(vectors,expected)
    tampered=dict(c); tampered["trusted_upper_unix_ms"]+=1
    deny(lambda:verify_external(tampered,e,"platform-root-01",hx("b")),"claim_id_mismatch")
    other=claim(1_499_100,1_499_500)
    deny(lambda:verify_external(other,e,"platform-root-01",hx("b")),"evidence_claim_mismatch")
    bad=dict(e); bad["external_evidence_digest"]=hx("d")
    deny(lambda:verify_external(c,bad,"platform-root-01",hx("b")),"evidence_id_mismatch")
    deny(lambda:verify_external(c,e,"wrong-provider",hx("b")),"provider_mismatch")
    deny(lambda:verify_external(c,e,"platform-root-01",hx("e")),"authority_policy_mismatch")
    deny(lambda:verify_external(c,e,"platform-root-01",hx("b"),False),"external_authority_rejected")
    # Critically: verification has no candidate clock-window/time argument.
    return vectors

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); a=ap.parse_args(); v=run()
    if a.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if a.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
