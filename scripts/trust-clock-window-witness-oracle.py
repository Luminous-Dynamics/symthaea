#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ClockWindowEvaluationWitnessV1 identity oracle."""
import argparse, hashlib, json, struct

OD=b"symthaea.fabrication.clock-observation-digest.v1\0"
WD=b"symthaea.fabrication.verified-clock-window.v1\0"
XD=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0"
SCHEMA="symthaea.trust.clock-window-evaluation-witness.v1"
TRUST=bytes.fromhex("609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633")
EXPECTED_WINDOW="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"
EXPECTED_WITNESS="e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2"
ALG={"Ed25519":0,"MlDsa65":1,"MlDsa87":2}

def compact(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def observation(source,t,u,algorithm,key):
    pre=["symthaea.fabrication.clock-observation.v1",source,t,u,42,algorithm,key]
    return {"source_id":source,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":42,
            "algorithm":algorithm,"key_id":key,
            "observation_digest":hashlib.sha256(OD+compact(pre)).digest()}
def window_digest(observations):
    lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in observations)
    hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in observations)
    h=hashlib.sha256(); h.update(WD); h.update(struct.pack("<Q",42))
    h.update(struct.pack("<Q",lo)); h.update(struct.pack("<Q",hi)); h.update(TRUST)
    for d in sorted(o["observation_digest"] for o in observations): h.update(d)
    return h.digest()
def put_count(h,n): h.update(struct.pack("<Q",n))
def put_string(h,s):
    b=s.encode(); put_count(h,len(b)); h.update(b)
def put_algorithm(h,a):
    if a in ALG: h.update(bytes([ALG[a]]))
    else: h.update(b"\x03"); put_string(h,a)
def witness_digest(observations):
    observations=sorted(observations,key=lambda o:o["observation_digest"])
    digests=[o["observation_digest"] for o in observations]
    signers=sorted((ALG.get(o["algorithm"],3),o["algorithm"],o["key_id"]) for o in observations)
    w=window_digest(observations)
    h=hashlib.sha256(); h.update(XD); put_string(h,SCHEMA); h.update(w); h.update(TRUST)
    h.update(struct.pack("<Q",42)); put_count(h,len(digests))
    for d in digests: h.update(d)
    put_count(h,len(signers))
    for _,algorithm,key in signers: put_algorithm(h,algorithm); put_string(h,key)
    put_count(h,len(observations))
    for o in observations:
        put_string(h,o["source_id"]); h.update(struct.pack("<Q",o["observed_unix_ms"]))
        h.update(struct.pack("<Q",o["uncertainty_ms"])); h.update(struct.pack("<Q",o["epoch"]))
        put_algorithm(h,o["algorithm"]); put_string(h,o["key_id"]); h.update(o["observation_digest"])
    return w,h.digest()

def self_test():
    a=observation("source-a",1_500_000,100,"Ed25519","clock-a")
    b=observation("source-b",1_500_040,120,"MlDsa65","clock-b")
    window,witness=witness_digest([b,a])
    assert window.hex()==EXPECTED_WINDOW
    assert witness.hex()==EXPECTED_WITNESS
    assert witness_digest([a,b])==(window,witness)
    changed=observation("source-a",1_500_000,100,"Ed25519","clock-x")
    assert witness_digest([changed,b])[1]!=witness
    return {"window_e42":window.hex(),"witness_e42":witness.hex()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    v=self_test()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else: print(json.dumps(v,sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
