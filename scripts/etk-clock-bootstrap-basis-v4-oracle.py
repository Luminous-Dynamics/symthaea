#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent quorum-authorized permit -> accepted bootstrap clock basis V4."""
import argparse, hashlib, json, struct
from copy import deepcopy

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
QD=b"symthaea.trust.clock-quorum-policy.v1\0"
ED=b"symthaea.trust.clock-evaluation-policy.v3\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v3\0"
OBS_D=b"symthaea.fabrication.clock-observation-digest.v1\0"
WIN_D=b"symthaea.fabrication.verified-clock-window.v1\0"
WIT_D=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
BASIS_D=b"symthaea.trust.accepted-clock-basis.v4\0"
EXPECTED={
 "quorum_policy_id":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304",
 "evaluation_policy_id":"7a782b8adbf4e66a352c4a26ae614a458971d78c33c136cb8eeee6c6c28433b4",
 "permit_id":"716c65376a721574bb53d2415e6b58f83d9c7483666b3799ca8fc4fe214f2cce",
 "window_id":"5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229",
 "witness_id":"e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2",
 "accepted_basis_v4_id":"77728f18e60bb095bcc744657bd1c2faa15f86e85853d363ebb4145d8a4ee3de"}

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

def eval_policy(q):
    pre={"schema":"symthaea.trust.clock-evaluation-policy.v3","policy_record_digest":hx("b"),
         "clock_quorum_policy_id":q["id"],"max_transition_ms":2000,
         "minimum_eligible_clock_authority_keys":2,"require_eligible_algorithm_diversity":True}
    return {**pre,"id":dh(ED,pre)}

def permit(q,ep):
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v3",
         "basis_id":"61fe43c253ca8dde9c152b641208f4eddc2ddbbfc263124d9b2a712e1e23d9e5",
         "basis_kind":"BootstrapAnchor","evaluation_policy_id":ep["id"],
         "clock_quorum_policy_id":q["id"],"trust_snapshot_digest":TRUST,
         "evaluation_lower_unix_ms":1499000,"evaluation_upper_unix_ms":1501500,
         "minimum_eligible_clock_authority_keys":2,"require_eligible_algorithm_diversity":True,
         "eligible_clock_keys":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
    return {**pre,"id":dh(PERMIT_D,pre),"clock_quorum_policy":deepcopy(q)}

def valid_permit(p):
    q=p["clock_quorum_policy"]
    qpre={k:q[k] for k in ("schema","minimum_distinct_sources","maximum_observations",
          "maximum_uncertainty_ms","maximum_consensus_width_ms","require_algorithm_diversity")}
    if q.get("id")!=dh(QD,qpre): raise Denied("quorum_policy_id_mismatch")
    if q["id"]!=p["clock_quorum_policy_id"]: raise Denied("permit_quorum_policy_mismatch")
    pre={k:p[k] for k in ("schema","basis_id","basis_kind","evaluation_policy_id","clock_quorum_policy_id",
         "trust_snapshot_digest","evaluation_lower_unix_ms","evaluation_upper_unix_ms",
         "minimum_eligible_clock_authority_keys","require_eligible_algorithm_diversity","eligible_clock_keys")}
    if p.get("id")!=dh(PERMIT_D,pre): raise Denied("permit_id_mismatch")
    if p["schema"]!="symthaea.trust.clock-evaluation-permit.v3" or p["basis_kind"]!="BootstrapAnchor":
        raise Denied("wrong_permit_kind")
    return q

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
    q=valid_permit(p); w=window(observations,q); x=witness(w); wd=witness_digest(x)
    if w["trust_snapshot_digest"]!=p["trust_snapshot_digest"]: raise Denied("window_snapshot_mismatch")
    if w["lower_unix_ms"]<p["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>p["evaluation_upper_unix_ms"]: raise Denied("window_outside_permit")
    eligible={tuple(v) for v in p["eligible_clock_keys"]}; signers={tuple(v) for v in x["signers"]}
    if not signers<=eligible: raise Denied("unpermitted_signer")
    for o in x["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<p["evaluation_lower_unix_ms"] or o["observed_unix_ms"]+o["uncertainty_ms"]>p["evaluation_upper_unix_ms"]:
            raise Denied("observation_interval_outside_permit")
    pre={"schema":"symthaea.trust.accepted-clock-basis.v4","acceptance_kind":"Bootstrap",
         "permit_id":p["id"],"clock_quorum_policy_id":p["clock_quorum_policy_id"],
         "trust_snapshot_digest":w["trust_snapshot_digest"],"clock_window_evidence_digest":w["evidence_digest"],
         "clock_window_witness_digest":wd,"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],
         "upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]}
    return {**pre,"id":dh(BASIS_D,pre),"window":w,"witness":x}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    q=quorum(); ep=eval_policy(q); p=permit(q,ep)
    a=observation("source-a",1500000,100,42,"Ed25519","clock-a"); b=observation("source-b",1500040,120,42,"MlDsa65","clock-b")
    out=accept(p,[b,a]); w=out["window"]; x=out["witness"]
    v={"quorum_policy_id":q["id"],"evaluation_policy_id":ep["id"],"permit_id":p["id"],"window_id":w["evidence_digest"],"witness_id":witness_digest(x),"accepted_basis_v4_id":out["id"]}
    assert v==EXPECTED,(v,EXPECTED)

    one=[a]; deny(lambda:accept(p,one),"insufficient_sources")
    weak=deepcopy(p); weak["clock_quorum_policy"]["maximum_uncertainty_ms"]=10000
    deny(lambda:accept(weak,[a,b]),"quorum_policy_id_mismatch")
    wide=observation("source-a",1500000,6000,42,"Ed25519","clock-a")
    deny(lambda:accept(p,[wide,b]),"uncertainty_too_large")
    bad=observation("source-a",1500000,100,42,"Ed25519","clock-x")
    deny(lambda:accept(p,[bad,b]),"unpermitted_signer")
    outlier=observation("source-a",1500000,1600,42,"Ed25519","clock-a")
    # 1600 is below verifier max=5000; final intersection remains inside, but this observation exceeds permit.
    deny(lambda:accept(p,[outlier,b]),"observation_interval_outside_permit")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
