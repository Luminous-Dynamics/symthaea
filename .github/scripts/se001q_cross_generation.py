#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

DOMAIN="symthaea.se001q.cross-generation-corroboration.v1"
ACCEPTED={"symthaea.se001q.lock-delta-verification.v1","symthaea.se001q.lock-delta-verification.v2"}
STRONG="symthaea.se001q.lock-delta-verification.v2"

def canonical(o): return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha_bytes(b): return "sha256:"+hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def die(m): raise SystemExit(m)

def reject_authority(v,path="$"):
    if isinstance(v,dict):
        for k,c in v.items():
            q=f"{path}.{k}"
            if k=="repair_authority": die(f"forbidden repair_authority field at {q}")
            if k=="repair_authority_claim" and c!="NONE": die(f"repair authority violation at {q}")
            if k=="qualification_claim" and c!="NONE": die(f"qualification authority violation at {q}")
            reject_authority(c,q)
    elif isinstance(v,list):
        for i,c in enumerate(v): reject_authority(c,f"{path}[{i}]")

def verify_manifest(root,m):
    if m.get("schema")!="symthaea.se001q.lock-delta-manifest.v1": die(f"{root}: bad manifest schema")
    if m.get("manifest_id")!=sha_bytes(canonical(m["identity"])): die(f"{root}: manifest id mismatch")
    exp={e["path"]:e for e in m["identity"]["files"]}
    act={p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
    if set(exp)!=set(act): die(f"{root}: manifest file set mismatch")
    for rel,p in act.items():
        e=exp[rel]
        if e["sha256"]!=sha_file(p) or e["bytes"]!=p.stat().st_size: die(f"{root}: manifest mismatch: {rel}")

def source_lock(root):
    ps=[p for p in [root/"Cargo.lock.before",root/"Cargo.lock.online-before",root/"Cargo.lock.offline-before"] if p.exists()]
    if not ps: die(f"{root}: no retained source lock")
    b=ps[0].read_bytes()
    if any(p.read_bytes()!=b for p in ps[1:]): die(f"{root}: source lock copies disagree")
    return ps[0]

def generated_lock(root):
    ps=[p for p in [root/"Cargo.lock.offline-after",root/"Cargo.lock.online-after"] if p.exists()]
    if not ps: die(f"{root}: no retained generated lock")
    b=ps[0].read_bytes()
    if any(p.read_bytes()!=b for p in ps[1:]): die(f"{root}: generated lock copies disagree")
    return ps[0]

def load(root_arg,verification_arg,prov):
    root=Path(root_arg).resolve()
    s=json.loads((root/"summary.json").read_text())
    m=json.loads((root/"manifest.json").read_text())
    w=json.loads((root/"lock-delta-witness.json").read_text())
    v=json.loads(Path(verification_arg).read_text())
    for o in (s,m,w,v): reject_authority(o)
    if s.get("schema")!="symthaea.se001q.lock-delta-observation.v1": die(f"{root}: bad summary schema")
    if s.get("observation_id")!=sha_bytes(canonical(s["identity"])): die(f"{root}: observation id mismatch")
    verify_manifest(root,m)
    if m["identity"].get("observation_id")!=s["observation_id"]: die(f"{root}: manifest observation mismatch")
    if w.get("schema")!="symthaea.lock-delta-witness.v1" or w.get("witness_id")!=sha_bytes(canonical(w["identity"])): die(f"{root}: lock witness invalid")
    if v.get("schema") not in ACCEPTED or v.get("result")!="PASS" or v.get("status")!="REPRODUCIBLE_LOCK_DELTA": die(f"{root}: verification not accepted")
    if v.get("observation_id")!=s["observation_id"] or v.get("manifest_id")!=m["manifest_id"] or v.get("lock_delta_witness_id")!=w["witness_id"]: die(f"{root}: verification bindings mismatch")
    si=s["identity"]; wi=w["identity"]
    if si.get("status")!="REPRODUCIBLE_LOCK_DELTA": die(f"{root}: summary status mismatch")
    sl=source_lock(root); gl=generated_lock(root)
    if sha_file(sl)!=wi.get("source_lock_sha256") or sha_file(gl)!=wi.get("generated_lock_sha256"): die(f"{root}: retained lock bytes mismatch witness")
    for k in ("run_id","artifact_id","artifact_zip_sha256","verifier_sha"):
        if not prov.get(k): die(f"{root}: missing provenance {k}")
    if not prov["artifact_zip_sha256"].startswith("sha256:"): die(f"{root}: bad artifact digest")
    return dict(root=root,summary=s,manifest=m,witness=w,verification=v,source=sl,generated=gl,prov=prov)

def rec(label,c):
    si=c["summary"]["identity"]; p=c["prov"]
    return {"label":label,"run_id":str(p["run_id"]),"artifact_id":str(p["artifact_id"]),"artifact_zip_sha256":p["artifact_zip_sha256"],"verifier_sha":p["verifier_sha"],"verification_schema":c["verification"]["schema"],"observation_id":c["summary"]["observation_id"],"manifest_id":c["manifest"]["manifest_id"],"lock_delta_witness_id":c["witness"]["witness_id"],"experiment_id":si["experiment_id"],"toolchain":si["toolchain"]}

def main():
    ap=argparse.ArgumentParser()
    for x in ("a","b"):
        ap.add_argument(f"--evidence-{x}",required=True); ap.add_argument(f"--verification-{x}",required=True)
        ap.add_argument(f"--run-id-{x}",required=True); ap.add_argument(f"--artifact-id-{x}",required=True)
        ap.add_argument(f"--artifact-zip-sha256-{x}",required=True); ap.add_argument(f"--verifier-sha-{x}",required=True)
    ap.add_argument("--output",required=True); ns=ap.parse_args()
    def prov(x): return {k:getattr(ns,f"{k}_{x}") for k in ("run_id","artifact_id","artifact_zip_sha256","verifier_sha")}
    a=load(ns.evidence_a,ns.verification_a,prov("a")); b=load(ns.evidence_b,ns.verification_b,prov("b"))
    ai=a["summary"]["identity"]; bi=b["summary"]["identity"]
    if ai.get("subject",{}).get("sha")!=bi.get("subject",{}).get("sha"): die("subject mismatch")
    if ai.get("experiment_id")!=bi.get("experiment_id"): die("experiment mismatch")
    if ai.get("toolchain")!=bi.get("toolchain"): die("toolchain mismatch")
    if a["source"].read_bytes()!=b["source"].read_bytes(): die("source lock mismatch")
    if a["generated"].read_bytes()!=b["generated"].read_bytes(): die("generated lock mismatch")
    ra,rb=rec("A",a),rec("B",b)
    if ra["run_id"]==rb["run_id"] or ra["artifact_id"]==rb["artifact_id"] or ra["verifier_sha"]==rb["verifier_sha"]: die("independence predicate failed")
    if sum(x["verification_schema"]==STRONG for x in (ra,rb))<1: die("at least one v2 verifier required")
    ident={"domain":DOMAIN,"subject_sha":ai["subject"]["sha"],"experiment_id":ai["experiment_id"],"source_lock_sha256":sha_file(a["source"]),"generated_lock_sha256":sha_file(a["generated"]),"toolchain":ai["toolchain"],"runs":[ra,rb],"independence":{"distinct_run_ids":True,"distinct_artifact_ids":True,"distinct_verifier_generations":True,"strong_independent_verifier_present":True,"source_lock_bytes_identical":True,"generated_lock_bytes_identical":True},"result":"CORROBORATED_GENERATED_LOCK","authority":{"meaning":"cross-generation corroboration only","sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"}}
    out={"schema":DOMAIN,"corroboration_id":sha_bytes(canonical(ident)),"identity":ident}; reject_authority(out)
    p=Path(ns.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"schema":DOMAIN,"result":ident["result"],"corroboration_id":out["corroboration_id"],"generated_lock_sha256":ident["generated_lock_sha256"],"sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"},sort_keys=True))
if __name__=="__main__": main()
