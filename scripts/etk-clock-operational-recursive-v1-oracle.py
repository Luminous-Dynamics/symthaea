#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent recursive operational-clock authority reference.

The theorem intentionally proves two successor transitions with the same
OperationalClockBasisV1 output type so normal operation does not require an
ever-growing sequence of basis versions.
"""
import argparse
import hashlib
import json
import struct
from copy import deepcopy

TRUST_D=b"symthaea.fabrication.trust-snapshot-digest.v1\0"
OBS_D=b"symthaea.fabrication.clock-observation-digest.v1\0"
WIN_D=b"symthaea.fabrication.verified-clock-window.v1\0"
WIT_D=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
CONT_D=b"symthaea.fabrication.clock-continuity-digest.v1\0"
Q_D=b"symthaea.trust.clock-quorum-policy.v1\0"
C_D=b"symthaea.trust.clock-continuity-policy.v1\0"
E_D=b"symthaea.trust.clock-evaluation-policy.v4\0"
B5_D=b"symthaea.trust.accepted-clock-basis.v5\0"
ROOT_D=b"symthaea.trust.clock-operational-basis.v1\0"
SP2_D=b"symthaea.trust.clock-successor-evaluation-permit.v2\0"

EXPECTED={
 "trust":"609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633",
 "quorum":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
 "continuity_policy":"904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06",
 "evaluation":"4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234",
 "bootstrap_v5":"241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8",
 "root42":"f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b",
 "permit43":"750c044e48a5395ee04a457ba7b89cb2397c2bf2cea76e96e71e50dcd338b73b",
 "window43":"d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b",
 "witness43":"6d330cbf15d9dde7bf11aada0fb5b53feed204573308c6bcafd5569f07b4e3a3",
 "continuity43":"ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15",
 "root43":"0e36e238e300ae1d626846220bbba84f5e87a50be7ffa2ac3b8e7822f4d47810",
 "permit44":"68b6dd92d42f3b42093f33d310d69eb8ac1bb259b02bd698256f67ea872b8a3d",
 "window44":"2304236bcb292476137284a25e943fe54bd0addac1e48ab917ba353bd25d1ffa",
 "witness44":"0bd33fe50684886e8cd5fea899ac73c57add275855778bde69f5e396e0f3db89",
 "continuity44":"194ce5eb8c491f02fe2d596801100abf3bd587513d1b6b569c692ae6e2cbfbff",
 "root44":"9373fac49b1b488859de56cd6cd588b8e5868cf9fb455c8b3546b46974d8ffce",
}

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(domain,v): return hashlib.sha256(domain+cj(v)).hexdigest()
def hx(c): return c*64

def snapshot(expires=2000):
    s={"schema_version":"symthaea.fabrication.trust-snapshot.v1","sequence":7,
       "issued_at_unix_s":1000,"expires_at_unix_s":expires,
       "keys":[
        {"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,
         "not_after_unix_s":3000,"status":"Active",
         "usages":["ClockAuthority","ClockContinuity"]},
        {"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,
         "not_after_unix_s":3000,"status":"Active",
         "usages":["ClockAuthority","ClockContinuity"]}]}
    return s
def snapshot_digest(s): return hashlib.sha256(TRUST_D+lj(s)).hexdigest()

def quorum():
    p={"schema":"symthaea.trust.clock-quorum-policy.v1",
       "minimum_distinct_sources":2,"maximum_observations":8,
       "maximum_uncertainty_ms":5000,"maximum_consensus_width_ms":10000,
       "require_algorithm_diversity":True}
    return {**p,"id":dh(Q_D,p)}
def continuity_policy():
    p={"schema":"symthaea.trust.clock-continuity-policy.v1",
       "maximum_epoch_step":1,"maximum_forward_gap_ms":10000,
       "maximum_consensus_jump_ms":60000,"minimum_shared_sources":1,
       "require_shared_algorithm":True}
    return {**p,"id":dh(C_D,p)}
def evaluation(q,cp):
    p={"schema":"symthaea.trust.clock-evaluation-policy.v4",
       "policy_record_digest":hx("b"),"clock_quorum_policy_id":q["id"],
       "clock_continuity_policy_id":cp["id"],"max_transition_ms":2000,
       "minimum_eligible_clock_authority_keys":2,
       "require_eligible_algorithm_diversity":True}
    return {**p,"id":dh(E_D,p)}

def obs(src,t,u,e,a,k):
    p=["symthaea.fabrication.clock-observation.v1",src,t,u,e,a,k]
    return {"source_id":src,"observed_unix_ms":t,"uncertainty_ms":u,
            "epoch":e,"algorithm":a,"key_id":k,
            "observation_digest":hashlib.sha256(OBS_D+lj(p)).hexdigest()}

def window(oo,q,sd):
    if not oo or len(oo)>q["maximum_observations"]: raise Denied("observation_count")
    if any(o["uncertainty_ms"]>q["maximum_uncertainty_ms"] for o in oo): raise Denied("uncertainty")
    e=oo[0]["epoch"]
    if any(o["epoch"]!=e for o in oo): raise Denied("epoch_mismatch")
    sources=[o["source_id"] for o in oo]
    signers=[(o["algorithm"],o["key_id"]) for o in oo]
    if len(sources)!=len(set(sources)) or len(signers)!=len(set(signers)): raise Denied("duplicate")
    if len(set(sources))<q["minimum_distinct_sources"]: raise Denied("insufficient_sources")
    if q["require_algorithm_diversity"] and len({o["algorithm"] for o in oo})<2: raise Denied("algorithm_diversity")
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in oo)
    hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in oo)
    if lo>hi: raise Denied("no_common_interval")
    if hi-lo>q["maximum_consensus_width_ms"]: raise Denied("consensus_width")
    h=hashlib.sha256(); h.update(WIN_D); h.update(struct.pack("<Q",e))
    h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(bytes.fromhex(sd))
    for d in sorted(bytes.fromhex(o["observation_digest"]) for o in oo): h.update(d)
    return {"epoch":e,"lower_unix_ms":lo,"upper_unix_ms":hi,
            "consensus_unix_ms":lo+(hi-lo)//2,"source_ids":sorted(sources),
            "algorithms":sorted({o["algorithm"] for o in oo}),
            "trust_snapshot_digest":sd,"evidence_digest":h.hexdigest(),
            "observations":deepcopy(oo),"signers":sorted([list(x) for x in signers])}

def witness_digest(w):
    accepted=sorted(o["observation_digest"] for o in w["observations"])
    observations=sorted(w["observations"],key=lambda o:o["observation_digest"])
    h=hashlib.sha256(); h.update(WIT_D)
    def count(n): h.update(struct.pack("<Q",n))
    def string(s): b=s.encode(); count(len(b)); h.update(b)
    def alg(a): h.update(bytes([{"Ed25519":0,"MlDsa65":1,"MlDsa87":2}[a]]))
    string("symthaea.trust.clock-window-evaluation-witness.v1")
    h.update(bytes.fromhex(w["evidence_digest"])); h.update(bytes.fromhex(w["trust_snapshot_digest"]))
    h.update(struct.pack("<Q",w["epoch"])); count(len(accepted))
    for d in accepted: h.update(bytes.fromhex(d))
    count(len(w["signers"]))
    for a,k in w["signers"]: alg(a); string(k)
    count(len(observations))
    for o in observations:
        string(o["source_id"]); h.update(struct.pack("<Q",o["observed_unix_ms"]))
        h.update(struct.pack("<Q",o["uncertainty_ms"])); h.update(struct.pack("<Q",o["epoch"]))
        alg(o["algorithm"]); string(o["key_id"]); h.update(bytes.fromhex(o["observation_digest"]))
    return h.hexdigest()

def continuity(prev,nxt,cp):
    if nxt["epoch"]<=prev["epoch"]: raise Denied("epoch_rollback")
    if nxt["epoch"]-prev["epoch"]>cp["maximum_epoch_step"]: raise Denied("epoch_step")
    if nxt["consensus_unix_ms"]<prev["consensus_unix_ms"]: raise Denied("time_regression")
    gap=max(0,nxt["lower_unix_ms"]-prev["upper_unix_ms"])
    jump=nxt["consensus_unix_ms"]-prev["consensus_unix_ms"]
    if gap>cp["maximum_forward_gap_ms"]: raise Denied("forward_gap")
    if jump>cp["maximum_consensus_jump_ms"]: raise Denied("consensus_jump")
    ss=sorted(set(prev["source_ids"])&set(nxt["source_ids"]))
    aa=sorted(set(prev["algorithms"])&set(nxt["algorithms"]))
    if len(ss)<cp["minimum_shared_sources"]: raise Denied("shared_sources")
    if cp["require_shared_algorithm"] and not aa: raise Denied("shared_algorithm")
    p={"schema_version":"symthaea.fabrication.clock-continuity.v1",
       "previous_evidence_digest":list(bytes.fromhex(prev["evidence_digest"])),
       "successor_evidence_digest":list(bytes.fromhex(nxt["evidence_digest"])),
       "previous_epoch":prev["epoch"],"successor_epoch":nxt["epoch"],
       "forward_gap_ms":gap,"consensus_jump_ms":jump,
       "shared_sources":ss,"shared_algorithms":aa,"continuity_digest":[0]*32}
    return hashlib.sha256(CONT_D+lj(p)).hexdigest()

def bootstrap_v5_id(permit_id,q,cp,e,w):
    wd=witness_digest(w)
    p={"schema":"symthaea.trust.accepted-clock-basis.v5","acceptance_kind":"Bootstrap",
       "permit_id":permit_id,"clock_evaluation_policy_id":e["id"],
       "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
       "trust_snapshot_digest":w["trust_snapshot_digest"],
       "clock_window_evidence_digest":w["evidence_digest"],
       "clock_window_witness_digest":wd,"epoch":w["epoch"],
       "lower_unix_ms":w["lower_unix_ms"],"upper_unix_ms":w["upper_unix_ms"],
       "consensus_unix_ms":w["consensus_unix_ms"]}
    return dh(B5_D,p)

def operational_bootstrap(source_v5,e,q,cp,w):
    p={"basis_kind":"Bootstrap","clock_continuity_policy_id":cp["id"],
       "clock_evaluation_policy_id":e["id"],"clock_quorum_policy_id":q["id"],
       "clock_window_evidence_digest":w["evidence_digest"],
       "clock_window_witness_digest":witness_digest(w),
       "consensus_unix_ms":w["consensus_unix_ms"],"epoch":w["epoch"],
       "lower_unix_ms":w["lower_unix_ms"],
       "schema":"symthaea.trust.clock-operational-basis.v1",
       "source_basis_id":source_v5,"trust_snapshot_digest":w["trust_snapshot_digest"],
       "upper_unix_ms":w["upper_unix_ms"]}
    return {**p,"id":dh(ROOT_D,p),"window":deepcopy(w)}

def eligible(s,lo,hi,e):
    if lo>hi or lo<s["issued_at_unix_s"]*1000 or hi>=s["expires_at_unix_s"]*1000:
        raise Denied("snapshot_not_valid_for_envelope")
    out=[]
    for k in s["keys"]:
        if k["status"]!="Active" or "ClockAuthority" not in k["usages"]: continue
        if lo<k["not_before_unix_s"]*1000: continue
        if k["not_after_unix_s"] is not None and hi>=k["not_after_unix_s"]*1000: continue
        out.append([k["algorithm"],k["key_id"]])
    out.sort()
    if len(out)<e["minimum_eligible_clock_authority_keys"]: raise Denied("insufficient_keys")
    if e["require_eligible_algorithm_diversity"] and len({a for a,_ in out})<2:
        raise Denied("eligible_algorithm_diversity")
    return out

def successor_permit(root,e,q,cp,s):
    lo=root["lower_unix_ms"]; hi=root["upper_unix_ms"]+e["max_transition_ms"]
    keys=eligible(s,lo,hi,e)
    p={"clock_continuity_policy_id":cp["id"],"clock_quorum_policy_id":q["id"],
       "eligible_clock_keys":keys,"evaluation_lower_unix_ms":lo,
       "evaluation_policy_id":e["id"],"evaluation_upper_unix_ms":hi,
       "minimum_eligible_clock_authority_keys":e["minimum_eligible_clock_authority_keys"],
       "prior_operational_basis_id":root["id"],
       "prior_clock_window_evidence_digest":root["clock_window_evidence_digest"],
       "require_eligible_algorithm_diversity":e["require_eligible_algorithm_diversity"],
       "schema":"symthaea.trust.clock-successor-evaluation-permit.v2",
       "trust_snapshot_digest":snapshot_digest(s)}
    return {**p,"id":dh(SP2_D,p)}

def accept_successor(root,permit,oo,e,q,cp,s):
    if permit["prior_operational_basis_id"]!=root["id"]: raise Denied("wrong_prior_root")
    if permit["evaluation_policy_id"]!=e["id"] or permit["clock_quorum_policy_id"]!=q["id"] or permit["clock_continuity_policy_id"]!=cp["id"]:
        raise Denied("policy_mismatch")
    if permit["trust_snapshot_digest"]!=snapshot_digest(s): raise Denied("snapshot_mismatch")
    w=window(oo,q,permit["trust_snapshot_digest"]); wd=witness_digest(w)
    if w["lower_unix_ms"]<permit["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>permit["evaluation_upper_unix_ms"]:
        raise Denied("window_outside_permit")
    allowed={tuple(x) for x in permit["eligible_clock_keys"]}
    if not {tuple(x) for x in w["signers"]}<=allowed: raise Denied("unpermitted_signer")
    for o in w["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<permit["evaluation_lower_unix_ms"] or o["observed_unix_ms"]+o["uncertainty_ms"]>permit["evaluation_upper_unix_ms"]:
            raise Denied("observation_interval_outside_permit")
    cd=continuity(root["window"],w,cp)
    p={"basis_kind":"Continuous","clock_continuity_digest":cd,
       "clock_continuity_policy_id":cp["id"],"clock_evaluation_policy_id":e["id"],
       "clock_quorum_policy_id":q["id"],"clock_window_evidence_digest":w["evidence_digest"],
       "clock_window_witness_digest":wd,"consensus_unix_ms":w["consensus_unix_ms"],
       "epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],
       "prior_clock_window_evidence_digest":root["clock_window_evidence_digest"],
       "prior_operational_basis_id":root["id"],
       "schema":"symthaea.trust.clock-operational-basis.v1",
       "successor_permit_id":permit["id"],"trust_snapshot_digest":w["trust_snapshot_digest"],
       "upper_unix_ms":w["upper_unix_ms"]}
    return {**p,"id":dh(ROOT_D,p),"window":deepcopy(w)}

def deny(fn,why):
    try: fn()
    except Denied as exc:
        if why not in str(exc): raise AssertionError((why,str(exc)))
    else: raise AssertionError(f"expected denial: {why}")

def self_test():
    s=snapshot(); q=quorum(); cp=continuity_policy(); e=evaluation(q,cp)
    sd=snapshot_digest(s)
    assert sd==EXPECTED["trust"]; assert q["id"]==EXPECTED["quorum"]
    assert cp["id"]==EXPECTED["continuity_policy"]; assert e["id"]==EXPECTED["evaluation"]
    o42=[obs("source-b",1500040,120,42,"MlDsa65","clock-b"),
         obs("source-a",1500000,100,42,"Ed25519","clock-a")]
    w42=window(o42,q,sd)
    assert w42["evidence_digest"]=="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"
    assert witness_digest(w42)=="e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2"
    v5=bootstrap_v5_id("1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f",q,cp,e,w42)
    assert v5==EXPECTED["bootstrap_v5"]
    r42=operational_bootstrap(v5,e,q,cp,w42); assert r42["id"]==EXPECTED["root42"]
    p43=successor_permit(r42,e,q,cp,s); assert p43["id"]==EXPECTED["permit43"]
    o43=[obs("source-b",1500540,120,43,"MlDsa65","clock-b"),
         obs("source-a",1500500,100,43,"Ed25519","clock-a")]
    r43=accept_successor(r42,p43,o43,e,q,cp,s)
    assert r43["clock_window_evidence_digest"]==EXPECTED["window43"]
    assert r43["clock_window_witness_digest"]==EXPECTED["witness43"]
    assert r43["clock_continuity_digest"]==EXPECTED["continuity43"]
    assert r43["id"]==EXPECTED["root43"]
    p44=successor_permit(r43,e,q,cp,s); assert p44["id"]==EXPECTED["permit44"]
    o44=[obs("source-b",1501040,120,44,"MlDsa65","clock-b"),
         obs("source-a",1501000,100,44,"Ed25519","clock-a")]
    r44=accept_successor(r43,p44,o44,e,q,cp,s)
    assert r44["clock_window_evidence_digest"]==EXPECTED["window44"]
    assert r44["clock_window_witness_digest"]==EXPECTED["witness44"]
    assert r44["clock_continuity_digest"]==EXPECTED["continuity44"]
    assert r44["id"]==EXPECTED["root44"]

    weakened=deepcopy(cp); weakened["maximum_forward_gap_ms"]=20000
    weakened["id"]=dh(C_D,{k:weakened[k] for k in (
        "schema","maximum_epoch_step","maximum_forward_gap_ms",
        "maximum_consensus_jump_ms","minimum_shared_sources","require_shared_algorithm")})
    deny(lambda: accept_successor(r43,p44,o44,e,q,weakened,s),"policy_mismatch")
    changed=deepcopy(s); changed["sequence"]=8
    deny(lambda: accept_successor(r43,p44,o44,e,q,cp,changed),"snapshot_mismatch")
    deny(lambda: accept_successor(r43,p43,o44,e,q,cp,s),"wrong_prior_root")
    one=[obs("source-a",1501000,100,44,"Ed25519","clock-a")]
    deny(lambda: accept_successor(r43,p44,one,e,q,cp,s),"insufficient_sources")
    far=[obs("source-a",1511000,100,44,"Ed25519","clock-a"),
         obs("source-b",1511040,120,44,"MlDsa65","clock-b")]
    deny(lambda: accept_successor(r43,p44,far,e,q,cp,s),"window_outside_permit")
    expired=snapshot(expires=1502)
    deny(lambda: successor_permit(r43,e,q,cp,expired),"snapshot_not_valid_for_envelope")
    assert r43["id"]!=r44["id"]
    return {"root42":r42["id"],"permit43":p43["id"],"root43":r43["id"],
            "permit44":p44["id"],"root44":r44["id"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true")
    ns=ap.parse_args()
    if ns.self_test:
        out=self_test()
        print(json.dumps(out,sort_keys=True))
    else:
        print(json.dumps(EXPECTED,sort_keys=True,indent=2))

if __name__=="__main__": main()
