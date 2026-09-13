#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent unified-policy permit -> accepted bootstrap clock basis V5."""
import argparse, hashlib, json, struct
from copy import deepcopy

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
QD=b"symthaea.trust.clock-quorum-policy.v1\0"
CD=b"symthaea.trust.clock-continuity-policy.v1\0"
ED=b"symthaea.trust.clock-evaluation-policy.v4\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v4\0"
OBS_D=b"symthaea.fabrication.clock-observation-digest.v1\0"
WIN_D=b"symthaea.fabrication.verified-clock-window.v1\0"
WIT_D=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
BASIS_D=b"symthaea.trust.accepted-clock-basis.v5\0"

EXPECTED={
"quorum_policy_id":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
"continuity_policy_id":"904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06",
"evaluation_policy_id":"4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234",
"permit_id":"1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f",
"window_id":"5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229",
"witness_id":"e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2",
"accepted_basis_v5_id":"241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8"}

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()
def hx(c): return c*64

def quorum(min_sources=2,max_obs=8,max_u=5000,max_w=10000,div=True):
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

def evaluation(q,c):
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v4","policy_record_digest":hx("b"),
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],
         "max_transition_ms":2000,"minimum_eligible_clock_authority_keys":2,
         "require_eligible_algorithm_diversity":True}
    return {**pre,"id":dh(ED,pre)}

def permit(q,c,ep):
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v4",
         "basis_id":"c9fc46d4bbdccef92c2bc86c3682f2319d6837f753cf954fff30dcd451b5b588",
         "basis_kind":"BootstrapAnchor","evaluation_policy_id":ep["id"],
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],
         "trust_snapshot_digest":TRUST,"evaluation_lower_unix_ms":1499000,
         "evaluation_upper_unix_ms":1501500,"minimum_eligible_clock_authority_keys":2,
         "require_eligible_algorithm_diversity":True,
         "eligible_clock_keys":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
    return {**pre,"id":dh(PERMIT_D,pre),"clock_evaluation_policy":deepcopy(ep),
            "clock_quorum_policy":deepcopy(q),"clock_continuity_policy":deepcopy(c)}

def valid_permit(p):
    q=p["clock_quorum_policy"]; cp=p["clock_continuity_policy"]; ep=p["clock_evaluation_policy"]
    qpre={k:q[k] for k in ("schema","minimum_distinct_sources","maximum_observations","maximum_uncertainty_ms","maximum_consensus_width_ms","require_algorithm_diversity")}
    cpre={k:cp[k] for k in ("schema","maximum_epoch_step","maximum_forward_gap_ms","maximum_consensus_jump_ms","minimum_shared_sources","require_shared_algorithm")}
    epre={k:ep[k] for k in ("schema","policy_record_digest","clock_quorum_policy_id","clock_continuity_policy_id","max_transition_ms","minimum_eligible_clock_authority_keys","require_eligible_algorithm_diversity")}
    if q.get("id")!=dh(QD,qpre): raise Denied("quorum_policy_id_mismatch")
    if cp.get("id")!=dh(CD,cpre): raise Denied("continuity_policy_id_mismatch")
    if ep.get("id")!=dh(ED,epre): raise Denied("evaluation_policy_id_mismatch")
    if ep["clock_quorum_policy_id"]!=q["id"] or p["clock_quorum_policy_id"]!=q["id"]: raise Denied("permit_quorum_policy_mismatch")
    if ep["clock_continuity_policy_id"]!=cp["id"] or p["clock_continuity_policy_id"]!=cp["id"]: raise Denied("permit_continuity_policy_mismatch")
    if p["evaluation_policy_id"]!=ep["id"]: raise Denied("permit_evaluation_policy_mismatch")
    pre={k:p[k] for k in ("schema","basis_id","basis_kind","evaluation_policy_id","clock_quorum_policy_id","clock_continuity_policy_id","trust_snapshot_digest","evaluation_lower_unix_ms","evaluation_upper_unix_ms","minimum_eligible_clock_authority_keys","require_eligible_algorithm_diversity","eligible_clock_keys")}
    if p.get("id")!=dh(PERMIT_D,pre): raise Denied("permit_id_mismatch")
    return q,cp,ep

def observation(source,t,u,epoch,algorithm,key_id):
    pre=["symthaea.fabrication.clock-observation.v1",source,t,u,epoch,algorithm,key_id]
    return {"source_id":source,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":epoch,
            "algorithm":algorithm,"key_id":key_id,
            "observation_digest":hashlib.sha256(OBS_D+lj(pre)).hexdigest()}

def window(observations,q):
    if not observations or len(observations)>q["maximum_observations"]: raise Denied("observation_count")
    if any(o["uncertainty_ms"]>q["maximum_uncertainty_ms"] for o in observations): raise Denied("uncertainty_too_large")
    epoch=observations[0]["epoch"]
    if any(o["epoch"]!=epoch for o in observations): raise Denied("epoch_mismatch")
    sources=[o["source_id"] for o in observations]; signers=[(o["algorithm"],o["key_id"]) for o in observations]
    if len(sources)!=len(set(sources)): raise Denied("duplicate_source")
    if len(signers)!=len(set(signers)): raise Denied("duplicate_signer")
    if len(set(sources))<q["minimum_distinct_sources"]: raise Denied("insufficient_sources")
    if q["require_algorithm_diversity"] and len({o["algorithm"] for o in observations})<2: raise Denied("algorithm_diversity_missing")
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in observations)
    hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in observations)
    if lo>hi: raise Denied("no_common_interval")
    if hi-lo>q["maximum_consensus_width_ms"]: raise Denied("consensus_too_wide")
    accepted=sorted(bytes.fromhex(o["observation_digest"]) for o in observations)
    h=hashlib.sha256(); h.update(WIN_D); h.update(struct.pack("<Q",epoch)); h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(bytes.fromhex(TRUST))
    for d in accepted: h.update(d)
    return {"epoch":epoch,"lower_unix_ms":lo,"upper_unix_ms":hi,"consensus_unix_ms":lo+(hi-lo)//2,
            "trust_snapshot_digest":TRUST,"evidence_digest":h.hexdigest(),
            "observations":sorted(observations,key=lambda o:o["observation_digest"]),
            "signers":sorted([list(x) for x in signers])}

def witness(w):
    return {"schema":"symthaea.trust.clock-window-evaluation-witness.v1",
            "window_evidence_digest":w["evidence_digest"],"trust_snapshot_digest":w["trust_snapshot_digest"],
            "epoch":w["epoch"],"accepted_observation_digests":sorted(o["observation_digest"] for o in w["observations"]),
            "signers":w["signers"],"observations":w["observations"]}

def witness_digest(x):
    h=hashlib.sha256(); h.update(WIT_D)
    def count(n): h.update(struct.pack("<Q",n))
    def string(s): b=s.encode(); count(len(b)); h.update(b)
    def alg(a):
        tags={"Ed25519":0,"MlDsa65":1,"MlDsa87":2}
        if a not in tags: raise Denied("unsupported_algorithm")
        h.update(bytes([tags[a]]))
    string(x["schema"]); h.update(bytes.fromhex(x["window_evidence_digest"])); h.update(bytes.fromhex(x["trust_snapshot_digest"])); h.update(struct.pack("<Q",x["epoch"]))
    count(len(x["accepted_observation_digests"])); [h.update(bytes.fromhex(d)) for d in x["accepted_observation_digests"]]
    count(len(x["signers"])); [(alg(a),string(k)) for a,k in x["signers"]]
    count(len(x["observations"]))
    for o in x["observations"]:
        string(o["source_id"]); h.update(struct.pack("<Q",o["observed_unix_ms"])); h.update(struct.pack("<Q",o["uncertainty_ms"])); h.update(struct.pack("<Q",o["epoch"])); alg(o["algorithm"]); string(o["key_id"]); h.update(bytes.fromhex(o["observation_digest"]))
    return h.hexdigest()

def accept(p,observations):
    q,cp,ep=valid_permit(p); w=window(observations,q); x=witness(w); wd=witness_digest(x)
    if w["trust_snapshot_digest"]!=p["trust_snapshot_digest"]: raise Denied("window_snapshot_mismatch")
    if w["lower_unix_ms"]<p["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>p["evaluation_upper_unix_ms"]: raise Denied("window_outside_permit")
    eligible={tuple(v) for v in p["eligible_clock_keys"]}; signers={tuple(v) for v in x["signers"]}
    if not signers<=eligible: raise Denied("unpermitted_signer")
    for o in x["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<p["evaluation_lower_unix_ms"] or o["observed_unix_ms"]+o["uncertainty_ms"]>p["evaluation_upper_unix_ms"]: raise Denied("observation_interval_outside_permit")
    pre={"schema":"symthaea.trust.accepted-clock-basis.v5","acceptance_kind":"Bootstrap",
         "permit_id":p["id"],"clock_evaluation_policy_id":ep["id"],
         "clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":cp["id"],
         "trust_snapshot_digest":w["trust_snapshot_digest"],"clock_window_evidence_digest":w["evidence_digest"],
         "clock_window_witness_digest":wd,"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],
         "upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}
    return {**pre,"id":dh(BASIS_D,pre),"window":w,"witness":x,"permit":deepcopy(p)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    q=quorum(); cp=continuity(); ep=evaluation(q,cp); p=permit(q,cp,ep)
    a=observation("source-a",1500000,100,42,"Ed25519","clock-a"); b=observation("source-b",1500040,120,42,"MlDsa65","clock-b")
    out=accept(p,[b,a]); w=out["window"]; x=out["witness"]
    v={"quorum_policy_id":q["id"],"continuity_policy_id":cp["id"],"evaluation_policy_id":ep["id"],
       "permit_id":p["id"],"window_id":w["evidence_digest"],"witness_id":witness_digest(x),
       "accepted_basis_v5_id":out["id"]}
    assert v==EXPECTED,(v,EXPECTED)

    deny(lambda:accept(p,[a]),"insufficient_sources")
    weak=deepcopy(p); weak["clock_continuity_policy"]["maximum_forward_gap_ms"]=20000
    deny(lambda:accept(weak,[a,b]),"continuity_policy_id_mismatch")
    wide=observation("source-a",1500000,6000,42,"Ed25519","clock-a")
    deny(lambda:accept(p,[wide,b]),"uncertainty_too_large")
    bad=observation("source-a",1500000,100,42,"Ed25519","clock-x")
    deny(lambda:accept(p,[bad,b]),"unpermitted_signer")
    outlier=observation("source-a",1500000,1600,42,"Ed25519","clock-a")
    deny(lambda:accept(p,[outlier,b]),"observation_interval_outside_permit")
    assert out["permit"]["clock_continuity_policy_id"]==cp["id"]
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
