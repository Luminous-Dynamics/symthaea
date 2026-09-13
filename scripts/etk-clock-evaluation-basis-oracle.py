#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent non-circular clock-evaluation-basis reference theorem.

Core law:
  signed clock sample
  != authority to choose the time used to validate its own signer.

A candidate clock window may be considered only after an evaluation permit has
been derived from a previously trusted basis + explicit transition policy.
The permit proves the exact trust snapshot and every required ClockAuthority key
remain temporally eligible across the whole conservative evaluation envelope.

This freezes deterministic binding semantics only. It does not authenticate the
bootstrap authority record, signatures, TPM state, operator action, trust
snapshot, or lower clock-quorum / continuity proof.
"""
from __future__ import annotations
import argparse, hashlib, json

ANCHOR_D=b"symthaea.trust.clock-bootstrap-anchor.v1\0"
POLICY_D=b"symthaea.trust.clock-evaluation-policy.v1\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v1\0"
BASIS_D=b"symthaea.trust.accepted-clock-basis.v1\0"

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
W42="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"
W43="d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b"
CONTINUITY_42_43="ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15"

EXPECTED={
 "anchor_id":"b05334b495bf37646e5e1f64617d90cba9eabb1adf1526dabf740c2dbbca4b47",
 "policy_id":"a3e42a624211880b576978213c03515cffb38fd4827fa72352c3fe0addf00124",
 "bootstrap_permit_id":"3f2419469946f1c3d902825d351ec64e6df5804c9f2cd46eeed67f86d6f78646",
 "accepted_basis_42_id":"a7ede1785e3e718f6fc8fad03650568b4c3ae93b88ee87088b6b643140211f95",
 "continuity_permit_id":"fc71389839735f0e5688fddd4a102e46c263262309a1996ecf62dfc6ae536619",
 "accepted_basis_43_id":"eb29b247cf88f26885a71b0e7424f04b7ff6e8a31479495fc30dfb0c21d8afea",
}

class Denied(ValueError): pass

def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def dh(domain,v): return hashlib.sha256(domain+cj(v).encode()).hexdigest()
def d(ch): return ch*64
def dg(v,name):
    if not isinstance(v,str) or len(v)!=64 or v.lower()!=v or any(c not in "0123456789abcdef" for c in v):
        raise Denied("invalid_digest:"+name)
    return v

def trust_snapshot():
    return {
      "digest":TRUST,
      "issued_at_unix_s":1000,
      "expires_at_unix_s":2000,
      "keys":[
        {"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,"not_after_unix_s":3000,
         "status":"Active","usages":["ClockAuthority","ClockContinuity"]},
        {"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,"not_after_unix_s":3000,
         "status":"Active","usages":["ClockAuthority","ClockContinuity"]},
      ],
    }

def bootstrap_anchor(snapshot_digest=TRUST):
    dg(snapshot_digest,"trust_snapshot")
    pre={"authority_record_digest":dg(d("a"),"authority_record"),"kind":"ExternalBootstrap",
         "schema":"symthaea.trust.clock-bootstrap-anchor.v1",
         "trusted_lower_unix_ms":1_499_000,"trusted_upper_unix_ms":1_499_500,
         "trust_snapshot_digest":snapshot_digest}
    if pre["trusted_lower_unix_ms"]>pre["trusted_upper_unix_ms"]: raise Denied("invalid_anchor_window")
    return {**pre,"id":dh(ANCHOR_D,pre)}

def policy(max_transition_ms=2000):
    if isinstance(max_transition_ms,bool) or not isinstance(max_transition_ms,int) or max_transition_ms<=0:
        raise Denied("invalid_transition_policy")
    pre={"max_transition_ms":max_transition_ms,"policy_record_digest":dg(d("b"),"policy_record"),
         "schema":"symthaea.trust.clock-evaluation-policy.v1"}
    return {**pre,"id":dh(POLICY_D,pre)}

def require_snapshot_and_keys_cover(snapshot, lower_ms, upper_ms, snapshot_digest):
    dg(snapshot_digest,"trust_snapshot")
    if snapshot.get("digest")!=snapshot_digest: raise Denied("snapshot_digest_mismatch")
    if lower_ms>upper_ms: raise Denied("invalid_evaluation_envelope")
    issued_ms=snapshot["issued_at_unix_s"]*1000
    expires_ms=snapshot["expires_at_unix_s"]*1000
    if lower_ms<issued_ms or upper_ms>=expires_ms: raise Denied("snapshot_not_valid_for_envelope")
    eligible=[]
    for key in snapshot["keys"]:
        if key["status"]!="Active": continue
        if "ClockAuthority" not in key["usages"]: continue
        if lower_ms < key["not_before_unix_s"]*1000: continue
        not_after=key["not_after_unix_s"]
        if not_after is not None and upper_ms>=not_after*1000: continue
        eligible.append([key["algorithm"],key["key_id"]])
    eligible.sort()
    if len(eligible)<2: raise Denied("insufficient_keys_for_envelope")
    return eligible

def permit_from_anchor(anchor, pol, snapshot):
    # Critically, there is no candidate clock window argument here.
    lower=anchor["trusted_lower_unix_ms"]
    upper=anchor["trusted_upper_unix_ms"]+pol["max_transition_ms"]
    eligible=require_snapshot_and_keys_cover(snapshot,lower,upper,anchor["trust_snapshot_digest"])
    pre={"basis_id":anchor["id"],"basis_kind":"BootstrapAnchor","eligible_clock_keys":eligible,
         "evaluation_lower_unix_ms":lower,"evaluation_upper_unix_ms":upper,
         "policy_id":pol["id"],"schema":"symthaea.trust.clock-evaluation-permit.v1",
         "trust_snapshot_digest":anchor["trust_snapshot_digest"]}
    return {**pre,"id":dh(PERMIT_D,pre)}

def _window_shape(window):
    dg(window["evidence_digest"],"clock_window")
    if window["lower_unix_ms"]>window["upper_unix_ms"]: raise Denied("invalid_candidate_window")

def accept_bootstrap_window(window, permit):
    _window_shape(window)
    if permit["basis_kind"]!="BootstrapAnchor": raise Denied("wrong_permit_kind")
    if window["lower_unix_ms"]<permit["evaluation_lower_unix_ms"] or window["upper_unix_ms"]>permit["evaluation_upper_unix_ms"]:
        raise Denied("candidate_outside_permit")
    pre={"acceptance_kind":"Bootstrap","clock_window_evidence_digest":window["evidence_digest"],
         "lower_unix_ms":window["lower_unix_ms"],"permit_id":permit["id"],
         "schema":"symthaea.trust.accepted-clock-basis.v1",
         "trust_snapshot_digest":permit["trust_snapshot_digest"],"upper_unix_ms":window["upper_unix_ms"]}
    return {**pre,"id":dh(BASIS_D,pre)}

def permit_from_prior_basis(prior, pol, snapshot):
    # No successor candidate is an input. The envelope comes only from the
    # already accepted prior basis + frozen transition policy.
    if prior.get("trust_snapshot_digest")!=snapshot.get("digest"):
        raise Denied("snapshot_rotation_requires_separate_authority")
    lower=prior["lower_unix_ms"]
    upper=prior["upper_unix_ms"]+pol["max_transition_ms"]
    eligible=require_snapshot_and_keys_cover(snapshot,lower,upper,prior["trust_snapshot_digest"])
    pre={"basis_id":prior["id"],"basis_kind":"PriorAcceptedClock","eligible_clock_keys":eligible,
         "evaluation_lower_unix_ms":lower,"evaluation_upper_unix_ms":upper,
         "policy_id":pol["id"],"schema":"symthaea.trust.clock-evaluation-permit.v1",
         "trust_snapshot_digest":prior["trust_snapshot_digest"]}
    return {**pre,"id":dh(PERMIT_D,pre)}

def accept_successor_window(prior, window, permit, continuity_digest):
    _window_shape(window); dg(continuity_digest,"clock_continuity")
    if permit["basis_kind"]!="PriorAcceptedClock" or permit["basis_id"]!=prior["id"]:
        raise Denied("permit_prior_mismatch")
    if window["lower_unix_ms"]<permit["evaluation_lower_unix_ms"] or window["upper_unix_ms"]>permit["evaluation_upper_unix_ms"]:
        raise Denied("candidate_outside_permit")
    pre={"acceptance_kind":"Continuous","clock_continuity_digest":continuity_digest,
         "clock_window_evidence_digest":window["evidence_digest"],"lower_unix_ms":window["lower_unix_ms"],
         "permit_id":permit["id"],"prior_basis_id":prior["id"],
         "schema":"symthaea.trust.accepted-clock-basis.v1",
         "trust_snapshot_digest":permit["trust_snapshot_digest"],"upper_unix_ms":window["upper_unix_ms"]}
    return {**pre,"id":dh(BASIS_D,pre)}

def window42():
    return {"evidence_digest":W42,"lower_unix_ms":1_499_920,"upper_unix_ms":1_500_100}
def window43():
    return {"evidence_digest":W43,"lower_unix_ms":1_500_420,"upper_unix_ms":1_500_600}

def expect_denied(fn,reason):
    try: fn()
    except Denied as e:
        assert reason in str(e),(reason,str(e)); return
    raise AssertionError(reason)

def self_test():
    snap=trust_snapshot(); a=bootstrap_anchor(); p=policy()
    permit0=permit_from_anchor(a,p,snap)
    b42=accept_bootstrap_window(window42(),permit0)
    permit1=permit_from_prior_basis(b42,p,snap)
    b43=accept_successor_window(b42,window43(),permit1,CONTINUITY_42_43)
    vectors={"anchor_id":a["id"],"policy_id":p["id"],"bootstrap_permit_id":permit0["id"],
             "accepted_basis_42_id":b42["id"],"continuity_permit_id":permit1["id"],
             "accepted_basis_43_id":b43["id"]}
    assert vectors==EXPECTED,(vectors,EXPECTED)

    fake={"evidence_digest":d("f"),"lower_unix_ms":2_500_000,"upper_unix_ms":2_500_100}
    expect_denied(lambda: accept_bootstrap_window(fake,permit0),"candidate_outside_permit")

    stale=trust_snapshot(); stale["expires_at_unix_s"]=1501
    expect_denied(lambda: permit_from_anchor(a,p,stale),"snapshot_not_valid_for_envelope")
    expiring=trust_snapshot(); expiring["keys"][0]["not_after_unix_s"]=1501; expiring["keys"][1]["not_after_unix_s"]=1501
    expect_denied(lambda: permit_from_anchor(a,p,expiring),"insufficient_keys_for_envelope")

    expect_denied(lambda: permit_from_anchor(bootstrap_anchor(d("c")),p,snap),"snapshot_digest_mismatch")
    expect_denied(lambda: permit_from_anchor(a,policy(600_000),snap),"snapshot_not_valid_for_envelope")

    rotated=trust_snapshot(); rotated["digest"]=d("e")
    expect_denied(lambda: permit_from_prior_basis(b42,p,rotated),"snapshot_rotation_requires_separate_authority")

    wrong_prior=dict(b42); wrong_prior["id"]=d("d")
    expect_denied(lambda: accept_successor_window(wrong_prior,window43(),permit1,CONTINUITY_42_43),"permit_prior_mismatch")

    assert accept_bootstrap_window(window42(),permit0)["id"]==b42["id"]
    return vectors

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=self_test()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    elif args.vectors: print(cj(v))
    else: print(cj({"decision":"SelfTest","vectors":v}))
    return 0
if __name__=="__main__": raise SystemExit(main())