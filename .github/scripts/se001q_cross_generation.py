#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

DOMAIN="symthaea.se001q.cross-generation-corroboration.v1.2"
V1="symthaea.se001q.lock-delta-verification.v1"
V2="symthaea.se001q.lock-delta-verification.v2"
V21="symthaea.se001q.lock-delta-verification.v2.1"
ACCEPTED={V1,V2,V21}
STRONG={V2,V21}
SHA256_RE=re.compile(r"^sha256:[0-9a-f]{64}$")
GITSHA_RE=re.compile(r"^[0-9a-f]{40}$")
EXPECTED_REQUIREMENTS=[
    "both capsules report REPRODUCIBLE_LOCK_DELTA and contain a content-addressed LockDeltaWitness",
    "both manifests exactly match retained file sets, sizes, and SHA-256 digests",
    "same frozen subject, experiment_id, exact experiment_sha256, and capture-recorded toolchain",
    "byte-identical source Cargo.lock and generated Cargo.lock",
    "provenance run/artifact IDs are positive decimal integers, artifact digests are exact sha256:<64-lowercase-hex>, verifier SHAs are exact 40-lowercase-hex, and each supplied verifier SHA equals that capsule observation verifier_sha",
    "distinct run IDs, artifact IDs, and verifier SHAs after capsule binding",
    "at least one capsule passed verification v2 or v2.1",
    "for verification v2.1, the wrapper verification_id is content-addressed, its nested base verification is v2 PASS, all toolchain predicates are true, capture/live toolchain strings equal the observation toolchain, experiment SHA-256 matches the observation, and parsed Rust/Cargo/Clippy releases are 1.96.0/1.96.0/0.1.96",
]

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

def verify_contract(path):
    p=Path(path).resolve()
    c=json.loads(p.read_text(encoding="utf-8"))
    reject_authority(c,"$.contract")
    if c.get("schema")!="symthaea.se001q.cross-generation-corroboration-contract.v1.2":
        die("bad corroboration contract schema")
    if c.get("accepted_verification_schemas")!=[V1,V2,V21]:
        die("corroboration contract verification schema set mismatch")
    if c.get("minimum_strong_verifier_count")!=1:
        die("corroboration contract strong-verifier count mismatch")
    if c.get("requirements")!=EXPECTED_REQUIREMENTS:
        die("corroboration contract requirements mismatch")
    if c.get("result")!="CORROBORATED_GENERATED_LOCK":
        die("corroboration contract result mismatch")
    expected_authority={
        "meaning":"corroboration only",
        "sufficient_for_repair_grant":False,
        "qualification_claim":"NONE",
        "repair_authority_claim":"NONE",
    }
    if c.get("authority")!=expected_authority:
        die("corroboration contract authority mismatch")
    return c,sha_file(p)

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

def positive_decimal(value,what):
    value=str(value)
    if not value.isdigit() or int(value)<=0: die(f"invalid {what}")
    return value

def exact_sha256(value,what):
    if not isinstance(value,str) or not SHA256_RE.fullmatch(value): die(f"invalid {what}")
    return value

def exact_gitsha(value,what):
    if not isinstance(value,str) or not GITSHA_RE.fullmatch(value): die(f"invalid {what}")
    return value

def verification_binding(root,v,s):
    schema=v.get("schema")
    if schema not in ACCEPTED: die(f"{root}: verification schema not accepted")
    if schema!=V21:
        return v
    if v.get("verification_id")!=sha_bytes(canonical(v["identity"])):
        die(f"{root}: v2.1 verification id mismatch")
    ident=v["identity"]
    if ident.get("domain")!=V21:
        die(f"{root}: v2.1 verification domain mismatch")
    base=ident.get("base_verification")
    if not isinstance(base,dict) or base.get("schema")!=V2:
        die(f"{root}: v2.1 missing base v2 verification")
    reject_authority(base,f"{root}.base_verification")
    predicates=ident.get("toolchain_predicates")
    required={
        "capture_matches_live",
        "rustc_release_matches_channel",
        "cargo_release_matches_channel",
        "clippy_release_matches_channel",
    }
    if not isinstance(predicates,dict) or set(predicates)!=required or not all(predicates.values()):
        die(f"{root}: v2.1 toolchain predicates not all true")
    if ident.get("capture_toolchain")!=ident.get("live_toolchain"):
        die(f"{root}: v2.1 capture/live toolchain mismatch")
    si=s["identity"]
    if ident.get("capture_toolchain")!=si.get("toolchain"):
        die(f"{root}: v2.1 toolchain does not bind observation")
    if ident.get("experiment_sha256")!=si.get("experiment_sha256"):
        die(f"{root}: v2.1 experiment digest mismatch")
    if ident.get("experiment_toolchain_channel")!="1.96.0":
        die(f"{root}: v2.1 unexpected toolchain channel")
    releases=ident.get("parsed_releases")
    if releases!={"rustc":"1.96.0","cargo":"1.96.0","clippy":"0.1.96"}:
        die(f"{root}: v2.1 parsed releases mismatch")
    if ident.get("expected_clippy_release")!="0.1.96":
        die(f"{root}: v2.1 expected clippy release mismatch")
    return base

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
    vb=verification_binding(root,v,s)
    if vb.get("result")!="PASS" or vb.get("status")!="REPRODUCIBLE_LOCK_DELTA": die(f"{root}: verification not accepted")
    if vb.get("observation_id")!=s["observation_id"] or vb.get("manifest_id")!=m["manifest_id"] or vb.get("lock_delta_witness_id")!=w["witness_id"]: die(f"{root}: verification bindings mismatch")
    si=s["identity"]; wi=w["identity"]
    if si.get("status")!="REPRODUCIBLE_LOCK_DELTA": die(f"{root}: summary status mismatch")
    experiment_sha=si.get("experiment_sha256")
    if not isinstance(experiment_sha,str) or not SHA256_RE.fullmatch(experiment_sha): die(f"{root}: missing or invalid experiment digest")
    capture_protocol=si.get("capture_protocol")
    if not isinstance(capture_protocol,str) or not capture_protocol: die(f"{root}: missing capture protocol")
    capsule_verifier=exact_gitsha(si.get("verifier_sha"),f"{root} capsule verifier SHA")
    sl=source_lock(root); gl=generated_lock(root)
    if sha_file(sl)!=wi.get("source_lock_sha256") or sha_file(gl)!=wi.get("generated_lock_sha256"): die(f"{root}: retained lock bytes mismatch witness")
    run_id=positive_decimal(prov.get("run_id"),f"{root} provenance run id")
    artifact_id=positive_decimal(prov.get("artifact_id"),f"{root} provenance artifact id")
    artifact_digest=exact_sha256(prov.get("artifact_zip_sha256"),f"{root} provenance artifact digest")
    verifier_sha=exact_gitsha(prov.get("verifier_sha"),f"{root} provenance verifier SHA")
    if verifier_sha!=capsule_verifier:
        die(f"{root}: provenance verifier SHA does not equal capsule verifier SHA")
    bound_prov={"run_id":run_id,"artifact_id":artifact_id,"artifact_zip_sha256":artifact_digest,"verifier_sha":verifier_sha}
    return dict(root=root,summary=s,manifest=m,witness=w,verification=v,verification_binding=vb,source=sl,generated=gl,prov=bound_prov)

def rec(label,c):
    si=c["summary"]["identity"]; p=c["prov"]; v=c["verification"]; vb=c["verification_binding"]
    return {"label":label,"run_id":p["run_id"],"artifact_id":p["artifact_id"],"artifact_zip_sha256":p["artifact_zip_sha256"],"verifier_sha":p["verifier_sha"],"verification_schema":v["schema"],"base_verification_schema":vb["schema"],"observation_id":c["summary"]["observation_id"],"manifest_id":c["manifest"]["manifest_id"],"lock_delta_witness_id":c["witness"]["witness_id"],"experiment_id":si["experiment_id"],"experiment_sha256":si["experiment_sha256"],"capture_protocol":si["capture_protocol"],"toolchain":si["toolchain"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--contract",required=True)
    for x in ("a","b"):
        ap.add_argument(f"--evidence-{x}",required=True); ap.add_argument(f"--verification-{x}",required=True)
        ap.add_argument(f"--run-id-{x}",required=True); ap.add_argument(f"--artifact-id-{x}",required=True)
        ap.add_argument(f"--artifact-zip-sha256-{x}",required=True); ap.add_argument(f"--verifier-sha-{x}",required=True)
    ap.add_argument("--output",required=True); ns=ap.parse_args()
    def prov(x): return {k:getattr(ns,f"{k}_{x}") for k in ("run_id","artifact_id","artifact_zip_sha256","verifier_sha")}
    contract,contract_sha=verify_contract(ns.contract)
    a=load(ns.evidence_a,ns.verification_a,prov("a")); b=load(ns.evidence_b,ns.verification_b,prov("b"))
    ai=a["summary"]["identity"]; bi=b["summary"]["identity"]
    if ai.get("subject",{}).get("sha")!=bi.get("subject",{}).get("sha"): die("subject mismatch")
    if ai.get("experiment_id")!=bi.get("experiment_id"): die("experiment id mismatch")
    if ai.get("experiment_sha256")!=bi.get("experiment_sha256"): die("experiment digest mismatch")
    if ai.get("toolchain")!=bi.get("toolchain"): die("toolchain mismatch")
    if a["source"].read_bytes()!=b["source"].read_bytes(): die("source lock mismatch")
    if a["generated"].read_bytes()!=b["generated"].read_bytes(): die("generated lock mismatch")
    ra,rb=rec("A",a),rec("B",b)
    if ra["run_id"]==rb["run_id"] or ra["artifact_id"]==rb["artifact_id"] or ra["verifier_sha"]==rb["verifier_sha"]: die("independence predicate failed")
    if sum(x["verification_schema"] in STRONG for x in (ra,rb))<1: die("at least one strong independent verifier required")
    ident={"domain":DOMAIN,"contract_sha256":contract_sha,"subject_sha":ai["subject"]["sha"],"experiment_id":ai["experiment_id"],"experiment_sha256":ai["experiment_sha256"],"source_lock_sha256":sha_file(a["source"]),"generated_lock_sha256":sha_file(a["generated"]),"toolchain":ai["toolchain"],"runs":[ra,rb],"independence":{"distinct_run_ids":True,"distinct_artifact_ids":True,"distinct_verifier_generations":True,"verifier_metadata_bound_to_capsules":True,"strong_independent_verifier_present":True,"experiment_bytes_identical":True,"source_lock_bytes_identical":True,"generated_lock_bytes_identical":True},"result":"CORROBORATED_GENERATED_LOCK","authority":{"meaning":"cross-generation corroboration only","sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"}}
    out={"schema":DOMAIN,"corroboration_id":sha_bytes(canonical(ident)),"identity":ident}; reject_authority(out)
    p=Path(ns.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"schema":DOMAIN,"result":ident["result"],"corroboration_id":out["corroboration_id"],"contract_sha256":ident["contract_sha256"],"experiment_sha256":ident["experiment_sha256"],"generated_lock_sha256":ident["generated_lock_sha256"],"sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"},sort_keys=True))
if __name__=="__main__": main()
