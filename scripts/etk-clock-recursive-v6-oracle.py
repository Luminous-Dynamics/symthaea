#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recursive AcceptedClockBasisV6 reference: V6 -> V6 -> V6."""
import argparse,hashlib,json,struct,copy
TD=b"symthaea.fabrication.trust-snapshot-digest.v1\0";OD=b"symthaea.fabrication.clock-observation-digest.v1\0";WD=b"symthaea.fabrication.verified-clock-window.v1\0";WID=b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0";CD=b"symthaea.fabrication.clock-continuity-digest.v1\0";QD=b"symthaea.trust.clock-quorum-policy.v1\0";CPD=b"symthaea.trust.clock-continuity-policy.v1\0";EPD=b"symthaea.trust.clock-evaluation-policy.v4\0";SPD=b"symthaea.trust.clock-successor-evaluation-permit.v1\0";B6D=b"symthaea.trust.accepted-clock-basis.v6\0"
X={"trust":"609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633","q":"4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304","c":"904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06","e":"4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234","b43":"6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e","p44":"d154b7bc136371d3d0bd24389dcc139769a20fa36dcf34dd4d0b2162658345e7","w44":"2304236bcb292476137284a25e943fe54bd0addac1e48ab917ba353bd25d1ffa","x44":"0bd33fe50684886e8cd5fea899ac73c57add275855778bde69f5e396e0f3db89","c44":"194ce5eb8c491f02fe2d596801100abf3bd587513d1b6b569c692ae6e2cbfbff","b44":"e1d9035a7b938521e4339836fd48f74d54ce685795b8acdefe42e18b02c86825","p45":"6b8bf16525b315c9d20c0b5cad0a5c3f61c241cddc60ecc1fde7b5708d090b91","w45":"2322df198ef985211b8c2af47e317b729926786be770b85c4777213e5dc4441e","x45":"8d4b6e26fd9f7a5f659938a6ab4fc617a32b9e45913d170c1e91356fd5837da0","c45":"e2fa905fb5b1791bdd3dbbd240e468ab9352c6bdebc9f5977158110aebd0e4ec","b45":"825721c953d67cf60e7882a6c0547e00166906f330ed012f8ef4a5a8372a1686"}
class D(ValueError):pass
def cj(v):return json.dumps(v,sort_keys=True,separators=(",",":")).encode()
def lj(v):return json.dumps(v,separators=(",",":")).encode()
def dh(d,v):return hashlib.sha256(d+cj(v)).hexdigest()
def snap():
 s={"schema_version":"symthaea.fabrication.trust-snapshot.v1","sequence":7,"issued_at_unix_s":1000,"expires_at_unix_s":2000,"keys":[{"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,"not_after_unix_s":3000,"status":"Active","usages":["ClockAuthority","ClockContinuity"]},{"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,"not_after_unix_s":3000,"status":"Active","usages":["ClockAuthority","ClockContinuity"]}]};s["digest"]=hashlib.sha256(TD+lj(s)).hexdigest();return s
def policies():
 q={"schema":"symthaea.trust.clock-quorum-policy.v1","minimum_distinct_sources":2,"maximum_observations":8,"maximum_uncertainty_ms":5000,"maximum_consensus_width_ms":10000,"require_algorithm_diversity":True};q["id"]=dh(QD,q)
 c={"schema":"symthaea.trust.clock-continuity-policy.v1","maximum_epoch_step":1,"maximum_forward_gap_ms":10000,"maximum_consensus_jump_ms":60000,"minimum_shared_sources":1,"require_shared_algorithm":True};c["id"]=dh(CPD,c)
 e={"schema":"symthaea.trust.clock-evaluation-policy.v4","policy_record_digest":"b"*64,"clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],"max_transition_ms":2000,"minimum_eligible_clock_authority_keys":2,"require_eligible_algorithm_diversity":True};e["id"]=dh(EPD,e);return q,c,e
def ob(src,t,u,n,a,k):
 p=["symthaea.fabrication.clock-observation.v1",src,t,u,n,a,k];return {"source_id":src,"observed_unix_ms":t,"uncertainty_ms":u,"epoch":n,"algorithm":a,"key_id":k,"observation_digest":hashlib.sha256(OD+lj(p)).hexdigest()}
def win(n,q,s):
 oo=[ob("source-a",1500500+500*(n-43),100,n,"Ed25519","clock-a"),ob("source-b",1500540+500*(n-43),120,n,"MlDsa65","clock-b")]
 lo=max(o["observed_unix_ms"]-o["uncertainty_ms"] for o in oo);hi=min(o["observed_unix_ms"]+o["uncertainty_ms"] for o in oo);h=hashlib.sha256();h.update(WD);h.update(struct.pack("<Q",n));h.update(struct.pack("<Q",lo));h.update(struct.pack("<Q",hi));h.update(bytes.fromhex(s["digest"]))
 for d in sorted(bytes.fromhex(o["observation_digest"]) for o in oo):h.update(d)
 return {"epoch":n,"lower_unix_ms":lo,"upper_unix_ms":hi,"consensus_unix_ms":lo+(hi-lo)//2,"source_ids":["source-a","source-b"],"algorithms":["Ed25519","MlDsa65"],"trust_snapshot_digest":s["digest"],"evidence_digest":h.hexdigest(),"observations":oo,"signers":[["Ed25519","clock-a"],["MlDsa65","clock-b"]]}
def wit(w):
 h=hashlib.sha256();h.update(WID)
 def u(n):h.update(struct.pack("<Q",n))
 def st(x):b=x.encode();u(len(b));h.update(b)
 def al(a):h.update(bytes([0 if a=="Ed25519" else 1]))
 ds=sorted(o["observation_digest"] for o in w["observations"]);oo=sorted(w["observations"],key=lambda o:o["observation_digest"]);st("symthaea.trust.clock-window-evaluation-witness.v1");h.update(bytes.fromhex(w["evidence_digest"]));h.update(bytes.fromhex(w["trust_snapshot_digest"]));u(w["epoch"]);u(len(ds))
 for d in ds:h.update(bytes.fromhex(d))
 u(2)
 for a,k in w["signers"]:al(a);st(k)
 u(2)
 for o in oo:st(o["source_id"]);u(o["observed_unix_ms"]);u(o["uncertainty_ms"]);u(o["epoch"]);al(o["algorithm"]);st(o["key_id"]);h.update(bytes.fromhex(o["observation_digest"]))
 return h.hexdigest()
def permit(b,e,q,c,s):
 if (b["e"],b["q"],b["c"],b["s"])!=(e["id"],q["id"],c["id"],s["digest"]):raise D("substitution")
 lo=b["window"]["lower_unix_ms"];hi=b["window"]["upper_unix_ms"]+e["max_transition_ms"]
 if hi>=s["expires_at_unix_s"]*1000:raise D("snapshot_expired")
 keys=[["Ed25519","clock-a"],["MlDsa65","clock-b"]]
 p={"schema":"symthaea.trust.clock-successor-evaluation-permit.v1","prior_basis_id":b["id"],"prior_clock_window_evidence_digest":b["window"]["evidence_digest"],"evaluation_policy_id":e["id"],"clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],"trust_snapshot_digest":s["digest"],"evaluation_lower_unix_ms":lo,"evaluation_upper_unix_ms":hi,"minimum_eligible_clock_authority_keys":2,"require_eligible_algorithm_diversity":True,"eligible_clock_keys":keys};p["id"]=dh(SPD,p);return p
def cont(a,b,c):
 if b["epoch"]-a["epoch"]!=1:raise D("epoch")
 gap=max(0,b["lower_unix_ms"]-a["upper_unix_ms"]);jump=b["consensus_unix_ms"]-a["consensus_unix_ms"]
 p={"schema_version":"symthaea.fabrication.clock-continuity.v1","previous_evidence_digest":list(bytes.fromhex(a["evidence_digest"])),"successor_evidence_digest":list(bytes.fromhex(b["evidence_digest"])),"previous_epoch":a["epoch"],"successor_epoch":b["epoch"],"forward_gap_ms":gap,"consensus_jump_ms":jump,"shared_sources":["source-a","source-b"],"shared_algorithms":["Ed25519","MlDsa65"],"continuity_digest":[0]*32};return hashlib.sha256(CD+lj(p)).hexdigest()
def accept(b,p,w,e,q,c,s):
 if w["lower_unix_ms"]<p["evaluation_lower_unix_ms"] or w["upper_unix_ms"]>p["evaluation_upper_unix_ms"]:raise D("outside")
 cd=cont(b["window"],w,c);wd=wit(w);x={"schema":"symthaea.trust.accepted-clock-basis.v6","acceptance_kind":"Continuous","successor_permit_id":p["id"],"prior_basis_id":b["id"],"clock_evaluation_policy_id":e["id"],"clock_quorum_policy_id":q["id"],"clock_continuity_policy_id":c["id"],"trust_snapshot_digest":s["digest"],"prior_clock_window_evidence_digest":b["window"]["evidence_digest"],"clock_window_evidence_digest":w["evidence_digest"],"clock_window_witness_digest":wd,"clock_continuity_digest":cd,"epoch":w["epoch"],"lower_unix_ms":w["lower_unix_ms"],"upper_unix_ms":w["upper_unix_ms"],"consensus_unix_ms":w["consensus_unix_ms"]};x["id"]=dh(B6D,x);x.update(window=w,e=e["id"],q=q["id"],c=c["id"],s=s["digest"]);return x
def test():
 s=snap();q,c,e=policies();w43=win(43,q,s);b={"id":X["b43"],"window":w43,"e":e["id"],"q":q["id"],"c":c["id"],"s":s["digest"]}
 p44=permit(b,e,q,c,s);w44=win(44,q,s);b44=accept(b,p44,w44,e,q,c,s);p45=permit(b44,e,q,c,s);w45=win(45,q,s);b45=accept(b44,p45,w45,e,q,c,s)
 g={"trust":s["digest"],"q":q["id"],"c":c["id"],"e":e["id"],"b43":b["id"],"p44":p44["id"],"w44":w44["evidence_digest"],"x44":wit(w44),"c44":cont(w43,w44,c),"b44":b44["id"],"p45":p45["id"],"w45":w45["evidence_digest"],"x45":wit(w45),"c45":cont(w44,w45,c),"b45":b45["id"]}
 assert g==X,(g,X)
 bad=copy.deepcopy(b44);bad["e"]="0"*64
 try:permit(bad,e,q,c,s);raise AssertionError("policy substitution accepted")
 except D:pass
 print(json.dumps(g,sort_keys=True))
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--self-test",action="store_true");n=a.parse_args()
 if n.self_test:test()
