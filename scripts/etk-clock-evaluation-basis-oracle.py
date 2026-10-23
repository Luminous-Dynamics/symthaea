#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ETK clock-evaluation-basis V2 independent reference.

A candidate clock cannot choose the time used to validate its own signers.
Permits are derived first from a trusted prior interval + policy + exact trust
snapshot. Candidate window witnesses are then reconstructed and checked against
that permit. Successors also require continuity whose exact endpoints/epochs
match the prior and successor windows. Reference semantics only; bootstrap,
policy-record and signature authenticity remain external/lower-layer inputs.
"""
import argparse, hashlib, json, struct
from copy import deepcopy

TD=b"symthaea.fabrication.trust-snapshot-digest.v1\0"; OD=b"symthaea.fabrication.clock-observation-digest.v1\0"
WD=b"symthaea.fabrication.verified-clock-window.v1\0"; CD=b"symthaea.fabrication.clock-continuity-digest.v1\0"
AD=b"symthaea.trust.clock-bootstrap-anchor.v2\0"; PD=b"symthaea.trust.clock-evaluation-policy.v2\0"
ED=b"symthaea.trust.clock-evaluation-permit.v2\0"; BD=b"symthaea.trust.accepted-clock-basis.v2\0"
TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
W42="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"; W43="d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b"
C43="ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15"
EXPECTED={"anchor_id":"34a24d22b986744a2f823eead0b4a0566806763df7a19fdc9af97467dc5f5e68","policy_id":"7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318","bootstrap_permit_id":"746a0d77056d1e35a4ec2f32c2cc56a97e5778544f9653cec170e65b062371a8","accepted_basis_42_id":"c84a78db6f6396593b63f2207f604d94be6022788c9647ec981934d2095a7b9f","continuity_permit_id":"d76e0dc9214eec044e3fef0c939e7002870bef47e7732f3e2662f8f0e2127e62","accepted_basis_43_id":"719d387aa5d1e7f74cd21e12c8acbd671491bd6c57b0b2945ea717126c5a0888"}
class Denied(ValueError): pass

def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64
def req(v,n):
    if not isinstance(v,str) or len(v)!=64 or v.lower()!=v or any(c not in "0123456789abcdef" for c in v): raise Denied("invalid_digest:"+n)
    return v

def snap_pre(s): return {k:deepcopy(s[k]) for k in ("schema_version","sequence","issued_at_unix_s","expires_at_unix_s","keys")}
def snapshot():
    s={"schema_version":"symthaea.fabrication.trust-snapshot.v1","sequence":7,"issued_at_unix_s":1000,"expires_at_unix_s":2000,"keys":[{"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,"not_after_unix_s":3000,"status":"Active","usages":["ClockAuthority","ClockContinuity"]},{"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,"not_after_unix_s":3000,"status":"Active","usages":["ClockAuthority","ClockContinuity"]}]}
    s["digest"]=hashlib.sha256(TD+lj(s)).hexdigest(); assert s["digest"]==TRUST; return s
def valid_snapshot(s):
    x=hashlib.sha256(TD+lj(snap_pre(s))).hexdigest()
    if s.get("digest")!=x: raise Denied("snapshot_content_digest_mismatch")
    return x

def obs(src,t,u,e,a,k):
    pre=["symthaea.fabrication.clock-observation.v1",src,t,u,e,a,k]
    return {"source_id":src,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":e,"algorithm":a,"key_id":k,"observation_digest":hashlib.sha256(OD+lj(pre)).hexdigest()}
def window(e,oo,sd=TRUST):
    req(sd,"trust_snapshot")
    if not oo or any(o["epoch"]!=e for o in oo): raise Denied("invalid_observation_set")
    ss=[o["source_id"] for o in oo]; kk=[(o["algorithm"],o["key_id"]) for o in oo]
    if len(ss)!=len(set(ss)): raise Denied("duplicate_source")
    if len(kk)!=len(set(kk)): raise Denied("duplicate_signer")
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in oo); hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in oo)
    if lo>hi: raise Denied("no_common_interval")
    h=hashlib.sha256(); h.update(WD); h.update(struct.pack("<Q",e)); h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(bytes.fromhex(sd))
    for d in sorted(bytes.fromhex(o["observation_digest"]) for o in oo): h.update(d)
    return {"epoch":e,"lower_unix_ms":lo,"upper_unix_ms":hi,"consensus_unix_ms":lo+(hi-lo)//2,"source_ids":sorted(ss),"algorithms":sorted(set(o["algorithm"] for o in oo)),"signer_keys":sorted([list(x) for x in kk]),"trust_snapshot_digest":sd,"accepted_observation_digests":sorted(o["observation_digest"] for o in oo),"observations":deepcopy(oo),"evidence_digest":h.hexdigest()}
def w42():
    w=window(42,[obs("source-a",1500000,100,42,"Ed25519","clock-a"),obs("source-b",1500040,120,42,"MlDsa65","clock-b")]); assert w["evidence_digest"]==W42; return w
def w43():
    w=window(43,[obs("source-a",1500500,100,43,"Ed25519","clock-a"),obs("source-b",1500540,120,43,"MlDsa65","clock-b")]); assert w["evidence_digest"]==W43; return w
def valid_window(w):
    r=window(w["epoch"],w["observations"],w["trust_snapshot_digest"])
    for k in ("epoch","lower_unix_ms","upper_unix_ms","consensus_unix_ms","source_ids","algorithms","signer_keys","trust_snapshot_digest","accepted_observation_digests","evidence_digest"):
        if w.get(k)!=r.get(k): raise Denied("window_witness_mismatch:"+k)

def continuity(a,b):
    valid_window(a); valid_window(b)
    if b["epoch"]<=a["epoch"]: raise Denied("epoch_rollback")
    if b["consensus_unix_ms"]<a["consensus_unix_ms"]: raise Denied("time_regression")
    ss=sorted(set(a["source_ids"])&set(b["source_ids"])); aa=sorted(set(a["algorithms"])&set(b["algorithms"]));
    if not ss or not aa: raise Denied("continuity_independence_missing")
    pre={"schema_version":"symthaea.fabrication.clock-continuity.v1","previous_evidence_digest":list(bytes.fromhex(a["evidence_digest"])),"successor_evidence_digest":list(bytes.fromhex(b["evidence_digest"])),"previous_epoch":a["epoch"],"successor_epoch":b["epoch"],"forward_gap_ms":max(0,b["lower_unix_ms"]-a["upper_unix_ms"]),"consensus_jump_ms":b["consensus_unix_ms"]-a["consensus_unix_ms"],"shared_sources":ss,"shared_algorithms":aa,"continuity_digest":[0]*32}
    d=hashlib.sha256(CD+lj(pre)).hexdigest(); return {**pre,"previous_evidence_digest":a["evidence_digest"],"successor_evidence_digest":b["evidence_digest"],"continuity_digest":d}

def anchor(sd=TRUST):
    pre={"schema":"symthaea.trust.clock-bootstrap-anchor.v2","kind":"ExternalBootstrap","authority_record_digest":req(hx("a"),"authority_record"),"trust_snapshot_digest":req(sd,"trust_snapshot"),"trusted_lower_unix_ms":1499000,"trusted_upper_unix_ms":1499500}
    return {**pre,"id":dh(AD,pre)}
def valid_anchor(a):
    pre={k:a[k] for k in ("schema","kind","authority_record_digest","trust_snapshot_digest","trusted_lower_unix_ms","trusted_upper_unix_ms")}
    if a.get("id")!=dh(AD,pre): raise Denied("anchor_id_mismatch")
    if a["schema"]!="symthaea.trust.clock-bootstrap-anchor.v2" or a["kind"]!="ExternalBootstrap" or a["trusted_lower_unix_ms"]>a["trusted_upper_unix_ms"]: raise Denied("invalid_anchor")
def policy(step=2000,n=2,div=True):
    if isinstance(step,bool) or not isinstance(step,int) or step<=0 or isinstance(n,bool) or not isinstance(n,int) or n<2: raise Denied("invalid_policy")
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v2","policy_record_digest":req(hx("b"),"policy_record"),"max_transition_ms":step,"minimum_clock_authority_keys":n,"require_algorithm_diversity":bool(div)}
    return {**pre,"id":dh(PD,pre)}
def valid_policy(p):
    pre={k:p[k] for k in ("schema","policy_record_digest","max_transition_ms","minimum_clock_authority_keys","require_algorithm_diversity")}
    if p.get("id")!=dh(PD,pre): raise Denied("policy_id_mismatch")
    if p["schema"]!="symthaea.trust.clock-evaluation-policy.v2": raise Denied("invalid_policy")
def eligible(s,lo,hi,sd,p):
    valid_policy(p); x=valid_snapshot(s)
    if x!=sd: raise Denied("snapshot_digest_mismatch")
    if lo>hi or lo<s["issued_at_unix_s"]*1000 or hi>=s["expires_at_unix_s"]*1000: raise Denied("snapshot_not_valid_for_envelope")
    out=[]
    for k in s["keys"]:
        if k["status"]!="Active" or "ClockAuthority" not in k["usages"] or lo<k["not_before_unix_s"]*1000: continue
        if k["not_after_unix_s"] is not None and hi>=k["not_after_unix_s"]*1000: continue
        out.append([k["algorithm"],k["key_id"]])
    out.sort()
    if len(out)<p["minimum_clock_authority_keys"]: raise Denied("insufficient_keys_for_envelope")
    if p["require_algorithm_diversity"] and len({x[0] for x in out})<2: raise Denied("eligible_algorithm_diversity_missing")
    return out
def mkpermit(bid,bkind,lo,hi,sd,p,s):
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v2","basis_id":bid,"basis_kind":bkind,"policy_id":p["id"],"trust_snapshot_digest":sd,"evaluation_lower_unix_ms":lo,"evaluation_upper_unix_ms":hi,"minimum_clock_authority_keys":p["minimum_clock_authority_keys"],"require_algorithm_diversity":p["require_algorithm_diversity"],"eligible_clock_keys":eligible(s,lo,hi,sd,p)}
    return {**pre,"id":dh(ED,pre)}
def valid_permit(p):
    pre={k:p[k] for k in ("schema","basis_id","basis_kind","policy_id","trust_snapshot_digest","evaluation_lower_unix_ms","evaluation_upper_unix_ms","minimum_clock_authority_keys","require_algorithm_diversity","eligible_clock_keys")}
    if p.get("id")!=dh(ED,pre): raise Denied("permit_id_mismatch")
    if p["schema"]!="symthaea.trust.clock-evaluation-permit.v2" or p["eligible_clock_keys"]!=sorted(p["eligible_clock_keys"]): raise Denied("invalid_permit")
def permit_anchor(a,p,s):
    valid_anchor(a); valid_policy(p); return mkpermit(a["id"],"BootstrapAnchor",a["trusted_lower_unix_ms"],a["trusted_upper_unix_ms"]+p["max_transition_ms"],a["trust_snapshot_digest"],p,s)
def basis_pre(b):
    keys=["schema","acceptance_kind","permit_id"]
    if b["acceptance_kind"]=="Continuous": keys.append("prior_basis_id")
    keys += ["trust_snapshot_digest","clock_window_evidence_digest"]
    if b["acceptance_kind"]=="Continuous": keys.append("clock_continuity_digest")
    keys += ["epoch","lower_unix_ms","upper_unix_ms","consensus_unix_ms"]
    return {k:b[k] for k in keys}
def valid_basis(b):
    if b.get("id")!=dh(BD,basis_pre(b)): raise Denied("basis_id_mismatch")
    if b["schema"]!="symthaea.trust.accepted-clock-basis.v2" or b["acceptance_kind"] not in ("Bootstrap","Continuous"): raise Denied("invalid_basis")
def permit_prior(b,p,s):
    valid_basis(b); valid_policy(p); valid_snapshot(s)
    if b["trust_snapshot_digest"]!=s["digest"]: raise Denied("snapshot_rotation_requires_separate_authority")
    return mkpermit(b["id"],"PriorAcceptedClock",b["lower_unix_ms"],b["upper_unix_ms"]+p["max_transition_ms"],b["trust_snapshot_digest"],p,s)
def allowed(w,p):
    valid_permit(p); valid_window(w)
    if w["trust_snapshot_digest"]!=p["trust_snapshot_digest"]: raise Denied("window_trust_snapshot_mismatch")
    if w["lower_unix_ms"]<p["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>p["evaluation_upper_unix_ms"]: raise Denied("candidate_outside_permit")
    es={tuple(x) for x in p["eligible_clock_keys"]}; ss={tuple(x) for x in w["signer_keys"]}
    if not ss<=es: raise Denied("candidate_uses_unpermitted_signer")
    if len(ss)<p["minimum_clock_authority_keys"]: raise Denied("candidate_has_insufficient_signers")
    if p["require_algorithm_diversity"] and len({a for a,_ in ss})<2: raise Denied("candidate_algorithm_diversity_missing")
    for o in w["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<p["evaluation_lower_unix_ms"] or o["observed_unix_ms"]+o["uncertainty_ms"]>p["evaluation_upper_unix_ms"]: raise Denied("observation_interval_outside_permit")
def accept0(w,p):
    if p["basis_kind"]!="BootstrapAnchor": raise Denied("wrong_permit_kind")
    allowed(w,p); pre={"schema":"symthaea.trust.accepted-clock-basis.v2","acceptance_kind":"Bootstrap","permit_id":p["id"],"trust_snapshot_digest":p["trust_snapshot_digest"],"clock_window_evidence_digest":w["evidence_digest"],"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],"upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}; return {**pre,"id":dh(BD,pre)}
def accept1(b,pw,w,p,c):
    valid_basis(b); valid_window(pw)
    if b["clock_window_evidence_digest"]!=pw["evidence_digest"] or b["epoch"]!=pw["epoch"] or b["consensus_unix_ms"]!=pw["consensus_unix_ms"]: raise Denied("prior_basis_window_mismatch")
    if p["basis_kind"]!="PriorAcceptedClock" or p["basis_id"]!=b["id"]: raise Denied("permit_prior_mismatch")
    allowed(w,p)
    if c["previous_evidence_digest"]!=pw["evidence_digest"]: raise Denied("continuity_previous_window_mismatch")
    if c["successor_evidence_digest"]!=w["evidence_digest"]: raise Denied("continuity_successor_window_mismatch")
    if c["previous_epoch"]!=pw["epoch"] or c["successor_epoch"]!=w["epoch"]: raise Denied("continuity_epoch_mismatch")
    if continuity(pw,w)["continuity_digest"]!=c["continuity_digest"]: raise Denied("continuity_digest_mismatch")
    pre={"schema":"symthaea.trust.accepted-clock-basis.v2","acceptance_kind":"Continuous","permit_id":p["id"],"prior_basis_id":b["id"],"trust_snapshot_digest":p["trust_snapshot_digest"],"clock_window_evidence_digest":w["evidence_digest"],"clock_continuity_digest":c["continuity_digest"],"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],"upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}; return {**pre,"id":dh(BD,pre)}
def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    s=snapshot(); a=anchor(); p=policy(); p0=permit_anchor(a,p,s); x=w42(); b=accept0(x,p0); p1=permit_prior(b,p,s); y=w43(); c=continuity(x,y); assert c["continuity_digest"]==C43; z=accept1(b,x,y,p1,c)
    v={"anchor_id":a["id"],"policy_id":p["id"],"bootstrap_permit_id":p0["id"],"accepted_basis_42_id":b["id"],"continuity_permit_id":p1["id"],"accepted_basis_43_id":z["id"]}; assert v==EXPECTED
    f=window(42,[obs("source-a",1500000,100,42,"Ed25519","clock-x"),obs("source-b",1500040,120,42,"MlDsa65","clock-b")]); deny(lambda:accept0(f,p0),"unpermitted_signer")
    one=window(42,[obs("source-a",1500000,100,42,"Ed25519","clock-a")]); deny(lambda:accept0(one,p0),"insufficient_signers")
    out=window(42,[obs("source-a",1500000,1600,42,"Ed25519","clock-a"),obs("source-b",1500040,120,42,"MlDsa65","clock-b")]); deny(lambda:accept0(out,p0),"observation_interval_outside_permit")
    bad=deepcopy(x); bad["signer_keys"][0][1]="clock-x"; deny(lambda:accept0(bad,p0),"window_witness_mismatch")
    for k,ch,why in (("previous_evidence_digest","c","previous_window_mismatch"),("successor_evidence_digest","d","successor_window_mismatch"),("continuity_digest","e","continuity_digest_mismatch")):
        q=deepcopy(c); q[k]=hx(ch); deny(lambda q=q:accept1(b,x,y,p1,q),why)
    stale=snapshot(); stale["expires_at_unix_s"]=1501; stale["digest"]=hashlib.sha256(TD+lj(snap_pre(stale))).hexdigest(); deny(lambda:permit_anchor(anchor(stale["digest"]),p,stale),"snapshot_not_valid_for_envelope")
    rot=snapshot(); rot["sequence"]=8; rot["digest"]=hashlib.sha256(TD+lj(snap_pre(rot))).hexdigest(); deny(lambda:permit_prior(b,p,rot),"snapshot_rotation_requires_separate_authority")
    tp=deepcopy(p); tp["max_transition_ms"]+=1; deny(lambda:permit_anchor(a,tp,s),"policy_id_mismatch")
    ta=deepcopy(a); ta["trusted_upper_unix_ms"]+=1; deny(lambda:permit_anchor(ta,p,s),"anchor_id_mismatch")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); x=ap.parse_args(); v=run()
    if x.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if x.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
