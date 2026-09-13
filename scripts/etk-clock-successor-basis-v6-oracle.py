#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent successor clock authority reference over AcceptedClockBasisV5."""
import argparse, hashlib, json, struct
from copy import deepcopy

TRUST_D=b"symthaea.fabrication.trust-snapshot-digest.v1\0"
OBS_D=b"symthaea.fabrication.clock-observation-digest.v1\0"
WIN_D=b"symthaea.fabrication.verified-clock-window.v1\0"
WIT_D=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
CONT_D=b"symthaea.fabrication.clock-continuity-digest.v1\0"
Q_D=b"symthaea.trust.clock-quorum-policy.v1\0"
C_D=b"symthaea.trust.clock-continuity-policy.v1\0"
E_D=b"symthaea.trust.clock-evaluation-policy.v4\0"
P_D=b"symthaea.trust.clock-evaluation-permit.v4\0"
B5_D=b"symthaea.trust.accepted-clock-basis.v5\0"
SP_D=b"symthaea.trust.clock-successor-evaluation-permit.v1\0"
B6_D=b"symthaea.trust.accepted-clock-basis.v6\0"

EXPECTED={
"trust_snapshot_id":"609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633",
"quorum_policy_id":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
"continuity_policy_id":"904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06",
"evaluation_policy_id":"4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234",
"bootstrap_permit_id":"1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f",
"prior_basis_v5_id":"241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8",
"successor_permit_id":"eec874620319419ac4ef0663997ee2a6700db268fad6367ee28b684816c5d5d8",
"successor_window_id":"d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b",
"successor_witness_id":"6d330cbf15d9dde7bf11aada0fb5b53feed204573308c6bcafd5569f07b4e3a3",
"continuity_id":"ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15",
"successor_basis_v6_id":"6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e",
}
class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64

def snapshot(expires=2000):
    s={"schema_version":"symthaea.fabrication.trust-snapshot.v1","sequence":7,
       "issued_at_unix_s":1000,"expires_at_unix_s":expires,
       "keys":[
       {"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,"not_after_unix_s":3000,
        "status":"Active","usages":["ClockAuthority","ClockContinuity"]},
       {"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,"not_after_unix_s":3000,
        "status":"Active","usages":["ClockAuthority","ClockContinuity"]}]}
    s["digest"]=hashlib.sha256(TRUST_D+lj(s)).hexdigest(); return s
def valid_snapshot(s):
    pre={k:deepcopy(s[k]) for k in ("schema_version","sequence","issued_at_unix_s","expires_at_unix_s","keys")}
    d=hashlib.sha256(TRUST_D+lj(pre)).hexdigest()
    if d!=s.get("digest"): raise Denied("snapshot_id_mismatch")
    return d

def quorum():
    pre={"schema":"symthaea.trust.clock-quorum-policy.v1","minimum_distinct_sources":2,
         "maximum_observations":8,"maximum_uncertainty_ms":5000,
         "maximum_consensus_width_ms":10000,"require_algorithm_diversity":True}
    return {**pre,"id":dh(Q_D,pre)}
def continuity_policy():
    pre={"schema":"symthaea.trust.clock-continuity-policy.v1","maximum_epoch_step":1,
         "maximum_forward_gap_ms":10000,"maximum_consensus_jump_ms":60000,
         "minimum_shared_sources":1,"require_shared_algorithm":True}
    return {**pre,"id":dh(C_D,pre)}
def evaluation(q,cp):
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v4","policy_record_digest":hx("b"),
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "max_transition_ms":2000,"minimum_eligible_clock_authority_keys":2,
         "require_eligible_algorithm_diversity":True}
    return {**pre,"id":dh(E_D,pre)}
def valid_policy_record(v, domain, keys, expected_schema):
    pre={k:v[k] for k in keys}
    if v.get("schema")!=expected_schema or v.get("id")!=dh(domain,pre): raise Denied("policy_identity_mismatch")

def eligible(s,lo,hi,ep):
    sd=valid_snapshot(s)
    if lo>hi or lo<s["issued_at_unix_s"]*1000 or hi>=s["expires_at_unix_s"]*1000:
        raise Denied("snapshot_not_valid_for_envelope")
    out=[]
    for k in s["keys"]:
        if k["status"]!="Active" or "ClockAuthority" not in k["usages"] or lo<k["not_before_unix_s"]*1000:
            continue
        if k["not_after_unix_s"] is not None and hi>=k["not_after_unix_s"]*1000:
            continue
        out.append([k["algorithm"],k["key_id"]])
    out.sort()
    if len(out)<ep["minimum_eligible_clock_authority_keys"]: raise Denied("insufficient_keys")
    if ep["require_eligible_algorithm_diversity"] and len({a for a,_ in out})<2:
        raise Denied("eligible_algorithm_diversity")
    return sd,out

def bootstrap_permit(q,cp,ep,s):
    sd,keys=eligible(s,1499000,1501500,ep)
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v4",
         "basis_id":"c9fc46d4bbdccef92c2bc86c3682f2319d6837f753cf954fff30dcd451b5b588",
         "basis_kind":"BootstrapAnchor","evaluation_policy_id":ep["id"],
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "trust_snapshot_digest":sd,"evaluation_lower_unix_ms":1499000,
         "evaluation_upper_unix_ms":1501500,"minimum_eligible_clock_authority_keys":2,
         "require_eligible_algorithm_diversity":True,"eligible_clock_keys":keys}
    return {**pre,"id":dh(P_D,pre),"clock_evaluation_policy":deepcopy(ep),
            "clock_quorum_policy":deepcopy(q),"clock_continuity_policy":deepcopy(cp),
            "trust_snapshot":deepcopy(s)}

def obs(src,t,u,e,a,k):
    pre=["symthaea.fabrication.clock-observation.v1",src,t,u,e,a,k]
    return {"source_id":src,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":e,
            "algorithm":a,"key_id":k,"observation_digest":hashlib.sha256(OBS_D+lj(pre)).hexdigest()}
def window(oo,q,sd):
    if not oo or len(oo)>q["maximum_observations"]: raise Denied("observation_count")
    if any(o["uncertainty_ms"]>q["maximum_uncertainty_ms"] for o in oo): raise Denied("uncertainty")
    e=oo[0]["epoch"]
    if any(o["epoch"]!=e for o in oo): raise Denied("epoch_mismatch")
    src=[o["source_id"] for o in oo]; sig=[(o["algorithm"],o["key_id"]) for o in oo]
    if len(src)!=len(set(src)) or len(sig)!=len(set(sig)): raise Denied("duplicate")
    if len(set(src))<q["minimum_distinct_sources"]: raise Denied("insufficient_sources")
    if q["require_algorithm_diversity"] and len({o["algorithm"] for o in oo})<2: raise Denied("algorithm_diversity")
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in oo)
    hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in oo)
    if lo>hi: raise Denied("no_common_interval")
    if hi-lo>q["maximum_consensus_width_ms"]: raise Denied("consensus_width")
    h=hashlib.sha256(); h.update(WIN_D); h.update(struct.pack("<Q",e)); h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(bytes.fromhex(sd))
    for d in sorted(bytes.fromhex(o["observation_digest"]) for o in oo): h.update(d)
    return {"epoch":e,"lower_unix_ms":lo,"upper_unix_ms":hi,"consensus_unix_ms":lo+(hi-lo)//2,
            "source_ids":sorted(src),"algorithms":sorted(set(o["algorithm"] for o in oo)),
            "trust_snapshot_digest":sd,"evidence_digest":h.hexdigest(),"observations":deepcopy(oo),
            "signers":sorted([list(x) for x in sig])}
def witness_digest(w):
    accepted=sorted(o["observation_digest"] for o in w["observations"])
    observations=sorted(w["observations"],key=lambda o:o["observation_digest"])
    h=hashlib.sha256(); h.update(WIT_D)
    def count(n): h.update(struct.pack("<Q",n))
    def string(s): b=s.encode(); count(len(b)); h.update(b)
    def alg(a): h.update(bytes([{"Ed25519":0,"MlDsa65":1,"MlDsa87":2}[a]]))
    string("symthaea.trust.clock-window-evaluation-witness.v1")
    h.update(bytes.fromhex(w["evidence_digest"])); h.update(bytes.fromhex(w["trust_snapshot_digest"])); h.update(struct.pack("<Q",w["epoch"]))
    count(len(accepted))
    for d in accepted: h.update(bytes.fromhex(d))
    count(len(w["signers"]))
    for a,k in w["signers"]: alg(a); string(k)
    count(len(observations))
    for o in observations:
        string(o["source_id"]); h.update(struct.pack("<Q",o["observed_unix_ms"])); h.update(struct.pack("<Q",o["uncertainty_ms"])); h.update(struct.pack("<Q",o["epoch"]))
        alg(o["algorithm"]); string(o["key_id"]); h.update(bytes.fromhex(o["observation_digest"]))
    return h.hexdigest()

def bootstrap_basis(p,w):
    wd=witness_digest(w)
    pre={"schema":"symthaea.trust.accepted-clock-basis.v5","acceptance_kind":"Bootstrap",
         "permit_id":p["id"],"clock_evaluation_policy_id":p["evaluation_policy_id"],
         "clock_quorum_policy_id":p["clock_quorum_policy_id"],"clock_continuity_policy_id":p["clock_continuity_policy_id"],
         "trust_snapshot_digest":w["trust_snapshot_digest"],"clock_window_evidence_digest":w["evidence_digest"],
         "clock_window_witness_digest":wd,"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],
         "upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}
    return {**pre,"id":dh(B5_D,pre),"originating_permit":deepcopy(p),"window":deepcopy(w)}

def validate_retained(prior):
    p=prior["originating_permit"]; q=p["clock_quorum_policy"]; cp=p["clock_continuity_policy"]; ep=p["clock_evaluation_policy"]; s=p["trust_snapshot"]
    valid_policy_record(q,Q_D,("schema","minimum_distinct_sources","maximum_observations","maximum_uncertainty_ms","maximum_consensus_width_ms","require_algorithm_diversity"),"symthaea.trust.clock-quorum-policy.v1")
    valid_policy_record(cp,C_D,("schema","maximum_epoch_step","maximum_forward_gap_ms","maximum_consensus_jump_ms","minimum_shared_sources","require_shared_algorithm"),"symthaea.trust.clock-continuity-policy.v1")
    valid_policy_record(ep,E_D,("schema","policy_record_digest","clock_quorum_policy_id","clock_continuity_policy_id","max_transition_ms","minimum_eligible_clock_authority_keys","require_eligible_algorithm_diversity"),"symthaea.trust.clock-evaluation-policy.v4")
    if ep["clock_quorum_policy_id"]!=q["id"] or ep["clock_continuity_policy_id"]!=cp["id"]: raise Denied("retained_policy_link")
    if prior["clock_evaluation_policy_id"]!=ep["id"] or prior["clock_quorum_policy_id"]!=q["id"] or prior["clock_continuity_policy_id"]!=cp["id"]: raise Denied("basis_policy_link")
    if valid_snapshot(s)!=prior["trust_snapshot_digest"]: raise Denied("basis_snapshot_link")
    return p,q,cp,ep,s

def successor_permit(prior):
    p,q,cp,ep,s=validate_retained(prior)
    lo=prior["lower_unix_ms"]; hi=prior["upper_unix_ms"]+ep["max_transition_ms"]
    sd,keys=eligible(s,lo,hi,ep)
    pre={"schema":"symthaea.trust.clock-successor-evaluation-permit.v1",
         "prior_basis_id":prior["id"],"prior_clock_window_evidence_digest":prior["clock_window_evidence_digest"],
         "evaluation_policy_id":ep["id"],"clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "trust_snapshot_digest":sd,"evaluation_lower_unix_ms":lo,"evaluation_upper_unix_ms":hi,
         "minimum_eligible_clock_authority_keys":ep["minimum_eligible_clock_authority_keys"],
         "require_eligible_algorithm_diversity":ep["require_eligible_algorithm_diversity"],
         "eligible_clock_keys":keys}
    return {**pre,"id":dh(SP_D,pre),"clock_evaluation_policy":deepcopy(ep),
            "clock_quorum_policy":deepcopy(q),"clock_continuity_policy":deepcopy(cp),
            "trust_snapshot":deepcopy(s)}

def continuity(previous,successor,cp):
    if successor["epoch"]<=previous["epoch"]: raise Denied("epoch_rollback")
    if successor["epoch"]-previous["epoch"]>cp["maximum_epoch_step"]: raise Denied("epoch_step")
    if successor["consensus_unix_ms"]<previous["consensus_unix_ms"]: raise Denied("time_regression")
    gap=max(0,successor["lower_unix_ms"]-previous["upper_unix_ms"])
    jump=successor["consensus_unix_ms"]-previous["consensus_unix_ms"]
    if gap>cp["maximum_forward_gap_ms"]: raise Denied("forward_gap")
    if jump>cp["maximum_consensus_jump_ms"]: raise Denied("consensus_jump")
    shared_sources=sorted(set(previous["source_ids"])&set(successor["source_ids"]))
    shared_algorithms=sorted(set(previous["algorithms"])&set(successor["algorithms"]))
    if len(shared_sources)<cp["minimum_shared_sources"]: raise Denied("shared_sources")
    if cp["require_shared_algorithm"] and not shared_algorithms: raise Denied("shared_algorithm")
    pre={"schema_version":"symthaea.fabrication.clock-continuity.v1",
         "previous_evidence_digest":list(bytes.fromhex(previous["evidence_digest"])),
         "successor_evidence_digest":list(bytes.fromhex(successor["evidence_digest"])),
         "previous_epoch":previous["epoch"],"successor_epoch":successor["epoch"],
         "forward_gap_ms":gap,"consensus_jump_ms":jump,
         "shared_sources":shared_sources,"shared_algorithms":shared_algorithms,
         "continuity_digest":[0]*32}
    return hashlib.sha256(CONT_D+lj(pre)).hexdigest()

def accept_successor(prior,sp,oo):
    _,q,cp,ep,s=validate_retained(prior)
    if sp["prior_basis_id"]!=prior["id"] or sp["prior_clock_window_evidence_digest"]!=prior["clock_window_evidence_digest"]: raise Denied("successor_permit_prior")
    if sp["evaluation_policy_id"]!=ep["id"] or sp["clock_quorum_policy_id"]!=q["id"] or sp["clock_continuity_policy_id"]!=cp["id"]: raise Denied("successor_permit_policy")
    if sp["trust_snapshot_digest"]!=prior["trust_snapshot_digest"]: raise Denied("successor_permit_snapshot")
    w=window(oo,q,sp["trust_snapshot_digest"]); wd=witness_digest(w)
    if w["lower_unix_ms"]<sp["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>sp["evaluation_upper_unix_ms"]: raise Denied("window_outside_permit")
    eligible_keys={tuple(x) for x in sp["eligible_clock_keys"]}
    if not {tuple(x) for x in w["signers"]}<=eligible_keys: raise Denied("unpermitted_signer")
    for o in w["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<sp["evaluation_lower_unix_ms"] or o["observed_unix_ms"]+o["uncertainty_ms"]>sp["evaluation_upper_unix_ms"]:
            raise Denied("observation_interval_outside_permit")
    cd=continuity(prior["window"],w,cp)
    pre={"schema":"symthaea.trust.accepted-clock-basis.v6","acceptance_kind":"Continuous",
         "successor_permit_id":sp["id"],"prior_basis_id":prior["id"],
         "clock_evaluation_policy_id":ep["id"],"clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "trust_snapshot_digest":w["trust_snapshot_digest"],"prior_clock_window_evidence_digest":prior["clock_window_evidence_digest"],
         "clock_window_evidence_digest":w["evidence_digest"],"clock_window_witness_digest":wd,
         "clock_continuity_digest":cd,"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],
         "upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}
    return {**pre,"id":dh(B6_D,pre)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    s=snapshot(); q=quorum(); cp=continuity_policy(); ep=evaluation(q,cp); p=bootstrap_permit(q,cp,ep,s)
    w42=window([obs("source-a",1500000,100,42,"Ed25519","clock-a"),obs("source-b",1500040,120,42,"MlDsa65","clock-b")],q,s["digest"])
    prior=bootstrap_basis(p,w42); sp=successor_permit(prior)
    oo43=[obs("source-b",1500540,120,43,"MlDsa65","clock-b"),obs("source-a",1500500,100,43,"Ed25519","clock-a")]
    out=accept_successor(prior,sp,oo43)
    w43=window(oo43,q,s["digest"])
    v={"trust_snapshot_id":s["digest"],"quorum_policy_id":q["id"],"continuity_policy_id":cp["id"],
       "evaluation_policy_id":ep["id"],"bootstrap_permit_id":p["id"],"prior_basis_v5_id":prior["id"],
       "successor_permit_id":sp["id"],"successor_window_id":w43["evidence_digest"],
       "successor_witness_id":witness_digest(w43),"continuity_id":continuity(w42,w43,cp),
       "successor_basis_v6_id":out["id"]}
    assert v==EXPECTED,(v,EXPECTED)

    # Normal successor API has no caller policy/snapshot choice: retained witness governs.
    tampered=deepcopy(prior); tampered["originating_permit"]["clock_continuity_policy"]["maximum_forward_gap_ms"]=20000
    deny(lambda: successor_permit(tampered),"policy_identity_mismatch")
    rotated=deepcopy(prior); rotated["originating_permit"]["trust_snapshot"]["sequence"]=8
    deny(lambda: successor_permit(rotated),"snapshot_id_mismatch")

    # Envelope and continuity are independently fail-closed.
    one=[obs("source-a",1500500,100,43,"Ed25519","clock-a")]
    deny(lambda: accept_successor(prior,sp,one),"insufficient_sources")
    far=[obs("source-a",1511000,100,43,"Ed25519","clock-a"),obs("source-b",1511040,120,43,"MlDsa65","clock-b")]
    deny(lambda: accept_successor(prior,sp,far),"window_outside_permit")
    skip=[obs("source-a",1500500,100,44,"Ed25519","clock-a"),obs("source-b",1500540,120,44,"MlDsa65","clock-b")]
    deny(lambda: accept_successor(prior,sp,skip),"epoch_step")

    # A snapshot can be sufficient for bootstrap yet expire before the next full envelope.
    short=snapshot(expires=1502)
    p_short=bootstrap_permit(q,cp,ep,short)
    w42_short=window([obs("source-a",1500000,100,42,"Ed25519","clock-a"),obs("source-b",1500040,120,42,"MlDsa65","clock-b")],q,short["digest"])
    prior_short=bootstrap_basis(p_short,w42_short)
    deny(lambda: successor_permit(prior_short),"snapshot_not_valid_for_envelope")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else:
        print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
