#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent permit+witness -> accepted bootstrap clock basis V3 oracle."""
import argparse, hashlib, json, struct

TRUST="609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633"
PERMIT="12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9"
WINDOW="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"
WITNESS="e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2"
OBS_D=b"symthaea.fabrication.clock-observation-digest.v1\0"
WIN_D=b"symthaea.fabrication.verified-clock-window.v1\0"
WIT_D=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
PERMIT_D=b"symthaea.trust.clock-evaluation-permit.v2\0"
BASIS_D=b"symthaea.trust.accepted-clock-basis.v3\0"
EXPECTED_BASIS="1442964fea0e569d5ef59e2df593b5399eb2a7cc568417e1a4c34c796a20d72e"

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()

def observation(source,t,u,epoch,algorithm,key_id):
    pre=["symthaea.fabrication.clock-observation.v1",source,t,u,epoch,algorithm,key_id]
    return {"source_id":source,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":epoch,
            "algorithm":algorithm,"key_id":key_id,
            "observation_digest":hashlib.sha256(OBS_D+lj(pre)).hexdigest()}

def window(observations):
    if not observations: raise Denied("empty_window")
    epoch=observations[0]["epoch"]
    if any(o["epoch"]!=epoch for o in observations): raise Denied("epoch_mismatch")
    sources=[o["source_id"] for o in observations]
    signers=[(o["algorithm"],o["key_id"]) for o in observations]
    if len(sources)!=len(set(sources)): raise Denied("duplicate_source")
    if len(signers)!=len(set(signers)): raise Denied("duplicate_signer")
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in observations)
    hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in observations)
    if lo>hi: raise Denied("no_common_interval")
    accepted=sorted(bytes.fromhex(o["observation_digest"]) for o in observations)
    h=hashlib.sha256(); h.update(WIN_D); h.update(struct.pack("<Q",epoch))
    h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(bytes.fromhex(TRUST))
    for d in accepted: h.update(d)
    return {"epoch":epoch,"lower_unix_ms":lo,"upper_unix_ms":hi,
            "consensus_unix_ms":lo+(hi-lo)//2,"trust_snapshot_digest":TRUST,
            "evidence_digest":h.hexdigest(),"observations":sorted(observations,key=lambda o:o["observation_digest"]),
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
    string(x["schema"]); h.update(bytes.fromhex(x["window_evidence_digest"]))
    h.update(bytes.fromhex(x["trust_snapshot_digest"])); h.update(struct.pack("<Q",x["epoch"]))
    count(len(x["accepted_observation_digests"]))
    for d in x["accepted_observation_digests"]: h.update(bytes.fromhex(d))
    count(len(x["signers"]))
    for a,k in x["signers"]: alg(a); string(k)
    count(len(x["observations"]))
    for o in x["observations"]:
        string(o["source_id"]); h.update(struct.pack("<Q",o["observed_unix_ms"]))
        h.update(struct.pack("<Q",o["uncertainty_ms"])); h.update(struct.pack("<Q",o["epoch"]))
        alg(o["algorithm"]); string(o["key_id"]); h.update(bytes.fromhex(o["observation_digest"]))
    return h.hexdigest()

def permit():
    pre={"schema":"symthaea.trust.clock-evaluation-permit.v2",
         "basis_id":"fdfb5cfe737fc326792ed57dfe6aa93b1a2005fef021de6ddcb60e6a56144282",
         "basis_kind":"BootstrapAnchor",
         "policy_id":"7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318",
         "trust_snapshot_digest":TRUST,"evaluation_lower_unix_ms":1499000,
         "evaluation_upper_unix_ms":1501500,"minimum_clock_authority_keys":2,
         "require_algorithm_diversity":True,
         "eligible_clock_keys":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
    return {**pre,"id":dh(PERMIT_D,pre)}

def valid_permit(p):
    pre={k:p[k] for k in ("schema","basis_id","basis_kind","policy_id","trust_snapshot_digest",
         "evaluation_lower_unix_ms","evaluation_upper_unix_ms","minimum_clock_authority_keys",
         "require_algorithm_diversity","eligible_clock_keys")}
    if p.get("id")!=dh(PERMIT_D,pre): raise Denied("permit_id_mismatch")
    if p["schema"]!="symthaea.trust.clock-evaluation-permit.v2" or p["basis_kind"]!="BootstrapAnchor":
        raise Denied("wrong_permit_kind")

def valid_window(w):
    r=window(w["observations"])
    for k in ("epoch","lower_unix_ms","upper_unix_ms","consensus_unix_ms","trust_snapshot_digest","evidence_digest","signers"):
        if w.get(k)!=r.get(k): raise Denied("window_witness_mismatch:"+k)

def valid_witness(w,x):
    valid_window(w)
    if x["window_evidence_digest"]!=w["evidence_digest"] or x["trust_snapshot_digest"]!=w["trust_snapshot_digest"] \
       or x["epoch"]!=w["epoch"] or x["observations"]!=w["observations"] or x["signers"]!=w["signers"]:
        raise Denied("witness_window_mismatch")
    expected=sorted(o["observation_digest"] for o in w["observations"])
    if x["accepted_observation_digests"]!=expected: raise Denied("witness_observation_set_mismatch")
    return witness_digest(x)

def accept_bootstrap(p,w,x):
    valid_permit(p); wd=valid_witness(w,x)
    if w["trust_snapshot_digest"]!=p["trust_snapshot_digest"]: raise Denied("window_snapshot_mismatch")
    if w["lower_unix_ms"]<p["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>p["evaluation_upper_unix_ms"]:
        raise Denied("window_outside_permit")
    eligible={tuple(v) for v in p["eligible_clock_keys"]}; signers={tuple(v) for v in x["signers"]}
    if not signers<=eligible: raise Denied("unpermitted_signer")
    if len(signers)<p["minimum_clock_authority_keys"]: raise Denied("insufficient_signers")
    if p["require_algorithm_diversity"] and len({a for a,_ in signers})<2:
        raise Denied("algorithm_diversity_missing")
    for o in x["observations"]:
        if o["observed_unix_ms"]-o["uncertainty_ms"]<p["evaluation_lower_unix_ms"] \
           or o["observed_unix_ms"]+o["uncertainty_ms"]>p["evaluation_upper_unix_ms"]:
            raise Denied("observation_interval_outside_permit")
    pre={"schema":"symthaea.trust.accepted-clock-basis.v3","acceptance_kind":"Bootstrap",
         "permit_id":p["id"],"trust_snapshot_digest":w["trust_snapshot_digest"],
         "clock_window_evidence_digest":w["evidence_digest"],"clock_window_witness_digest":wd,
         "epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],"upper_unix_ms":w["upper_unix_ms"],
         "consensus_unix_ms":w["consensus_unix_ms"]}
    return {**pre,"id":dh(BASIS_D,pre)}

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    p=permit(); assert p["id"]==PERMIT
    a=observation("source-a",1500000,100,42,"Ed25519","clock-a")
    b=observation("source-b",1500040,120,42,"MlDsa65","clock-b")
    w=window([b,a]); x=witness(w)
    assert w["evidence_digest"]==WINDOW
    assert witness_digest(x)==WITNESS
    basis=accept_bootstrap(p,w,x); assert basis["id"]==EXPECTED_BASIS,(basis["id"],EXPECTED_BASIS)

    tx=json.loads(json.dumps(x)); tx["signers"][0][1]="clock-x"
    deny(lambda:accept_bootstrap(p,w,tx),"witness_window_mismatch")

    ox=observation("source-a",1500000,100,42,"Ed25519","clock-x")
    wx=window([ox,b]); xx=witness(wx)
    deny(lambda:accept_bootstrap(p,wx,xx),"unpermitted_signer")

    one=window([a]); one_x=witness(one)
    deny(lambda:accept_bootstrap(p,one,one_x),"insufficient_signers")

    wide_a=observation("source-a",1500000,1600,42,"Ed25519","clock-a")
    wide=window([wide_a,b]); wide_x=witness(wide)
    deny(lambda:accept_bootstrap(p,wide,wide_x),"observation_interval_outside_permit")

    wrong=dict(p); wrong["id"]="0"*64
    deny(lambda:accept_bootstrap(wrong,w,x),"permit_id_mismatch")
    return {"permit_id":p["id"],"window_id":w["evidence_digest"],
            "witness_id":witness_digest(x),"accepted_basis_v3_id":basis["id"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else:
        print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
