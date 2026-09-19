#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

LOCK_DIAGNOSTICS=(b"cannot update the lock file", b"lock file", b"needs to be updated")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data): return "sha256:" + hashlib.sha256(data).hexdigest()
def sha_file(path): return sha_bytes(Path(path).read_bytes())

def run(argv, cwd, outdir, env=None):
    outdir.mkdir(parents=True, exist_ok=True)
    proc=subprocess.run(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (outdir/"stdout.log").write_bytes(proc.stdout)
    (outdir/"stderr.log").write_bytes(proc.stderr)
    (outdir/"exit-code.txt").write_text(str(proc.returncode)+"\n")
    command={"argv":argv,"cwd":".","exit_code":proc.returncode,
             "stdout_sha256":sha_bytes(proc.stdout),"stderr_sha256":sha_bytes(proc.stderr)}
    (outdir/"command-result.json").write_text(json.dumps(command,indent=2,sort_keys=True)+"\n")
    return command, proc.stdout, proc.stderr

def git(cwd,*args,check=True):
    p=subprocess.run(["git","-C",str(cwd),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check and p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace')}")
    return p

def input_hashes(root, keys): return {k:sha_file(Path(root)/k) for k in keys}

def changed_tracked(root):
    p=git(root,"status","--porcelain=v1","--untracked-files=no")
    paths=[]
    for line in p.stdout.decode().splitlines():
        if not line: continue
        path=line[3:]
        if " -> " in path: path=path.split(" -> ",1)[1]
        paths.append(path)
    return sorted(paths)

def command_contract(exp, section, ident):
    for item in exp[section]:
        if item["id"]==ident: return item["argv"]
    raise SystemExit(f"missing experiment command {section}/{ident}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject",required=True)
    ap.add_argument("--experiment",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--work-root",required=True)
    ns=ap.parse_args()
    subject=Path(ns.subject).resolve(); output=Path(ns.output).resolve(); workroot=Path(ns.work_root).resolve()
    exp=json.loads(Path(ns.experiment).read_text())
    if exp.get("schema")!="symthaea.se001q.lock-delta-experiment.v1": raise SystemExit("bad experiment schema")
    if exp.get("qualification_claim")!="NONE" or exp.get("repair_authority_claim")!="NONE": raise SystemExit("authority boundary violated")
    subject_sha=exp["subject"]["sha"]; subject_tree=exp["subject"]["tree"]
    if git(subject,"rev-parse","HEAD").stdout.decode().strip()!=subject_sha: raise SystemExit("subject head mismatch")
    if git(subject,"rev-parse","HEAD^{tree}").stdout.decode().strip()!=subject_tree: raise SystemExit("subject tree mismatch")
    if git(subject,"status","--porcelain=v1","--untracked-files=all").stdout: raise SystemExit("subject not clean")
    before_inputs=input_hashes(subject,exp["expected_inputs"])
    if before_inputs!=exp["expected_inputs"]: raise SystemExit("subject input hash mismatch")

    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    if workroot.exists(): shutil.rmtree(workroot)
    workroot.mkdir(parents=True)
    a=workroot/"online"; b=workroot/"offline"
    git(subject,"worktree","add","--detach",str(a),subject_sha)
    git(subject,"worktree","add","--detach",str(b),subject_sha)
    try:
        shutil.copy2(a/"Cargo.lock",output/"Cargo.lock.before")
        before_lock=sha_file(output/"Cargo.lock.before")
        env=os.environ.copy(); env["CARGO_TERM_COLOR"]="never"

        online_argv=command_contract(exp,"resolution_probes","online-all-targets")
        online_res,_,online_err=run(online_argv,a,output/"online-all-targets",env)
        shutil.copy2(a/"Cargo.lock",output/"Cargo.lock.online-after")
        online_lock=sha_file(output/"Cargo.lock.online-after")
        patch=git(a,"diff","--binary","--","Cargo.lock").stdout
        (output/"Cargo.lock.patch").write_bytes(patch)
        online_changed=changed_tracked(a)

        offline_argv=command_contract(exp,"resolution_probes","offline-reproduction-all-targets")
        offline_res,_,offline_err=run(offline_argv,b,output/"offline-reproduction-all-targets",env)
        shutil.copy2(b/"Cargo.lock",output/"Cargo.lock.offline-after")
        offline_lock=sha_file(output/"Cargo.lock.offline-after")
        offline_changed=changed_tracked(b)

        counter=[]
        generated_lock=offline_lock
        for item in exp["counterfactual_locked_gates"]:
            lock_before=sha_file(b/"Cargo.lock")
            res,stdout,stderr=run(item["argv"],b,output/("counterfactual-"+item["id"]),env)
            lock_after=sha_file(b/"Cargo.lock")
            counter.append({"gate_id":item["id"],"result":res,
                            "lock_sha256_before":lock_before,"lock_sha256_after":lock_after,
                            "lock_unchanged":lock_before==lock_after,
                            "lock_update_diagnostic_present": any(x in stderr for x in LOCK_DIAGNOSTICS)})

        after_inputs_a=input_hashes(a,[k for k in exp["expected_inputs"] if k!="Cargo.lock"])
        after_inputs_b=input_hashes(b,[k for k in exp["expected_inputs"] if k!="Cargo.lock"])
        expected_nonlock={k:v for k,v in exp["expected_inputs"].items() if k!="Cargo.lock"}
        subject_clean=(git(subject,"status","--porcelain=v1","--untracked-files=all").stdout==b"")
        subject_head=(git(subject,"rev-parse","HEAD").stdout.decode().strip()==subject_sha)
        source_preserved=(after_inputs_a==expected_nonlock and after_inputs_b==expected_nonlock)
        delta_exists=(online_lock!=before_lock and offline_lock!=before_lock)
        reproduced=(online_lock==offline_lock)
        tracked_safe=(online_changed==["Cargo.lock"] and offline_changed==["Cargo.lock"])
        counter_lock_stable=all(x["lock_unchanged"] for x in counter)
        lock_boundary_cleared=all(not x["lock_update_diagnostic_present"] for x in counter)

        if delta_exists and reproduced and tracked_safe and counter_lock_stable and lock_boundary_cleared and source_preserved and subject_clean and subject_head:
            status="REPRODUCIBLE_LOCK_DELTA"
        elif not delta_exists:
            status="NO_REPRODUCIBLE_LOCK_DELTA"
        else:
            status="INCOMPLETE_OR_NONREPRODUCIBLE"

        toolchain={}
        for name,argv in {"rustc":["rustc","--version","--verbose"],"cargo":["cargo","--version"],"clippy":["cargo","clippy","--version"]}.items():
            p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
            toolchain[name]=p.stdout.decode(errors="replace").strip()

        summary_identity={
            "domain":"symthaea.se001q.lock-delta-observation.v1","experiment_id":exp["experiment_id"],
            "subject":exp["subject"],"parent_evidence":exp["parent_evidence"],"toolchain":toolchain,
            "before_lock_sha256":before_lock,"online_after_lock_sha256":online_lock,"offline_after_lock_sha256":offline_lock,
            "online_changed_tracked_paths":online_changed,"offline_changed_tracked_paths":offline_changed,
            "online_resolution_result":online_res,"offline_resolution_result":offline_res,
            "counterfactual_locked_gates":counter,"source_preserved":source_preserved,
            "frozen_subject_unchanged":subject_clean and subject_head,"lock_delta_reproduced_offline":reproduced and delta_exists,
            "counterfactual_lock_stable":counter_lock_stable,"lock_boundary_cleared":lock_boundary_cleared,
            "status":status,"qualification_claim":"NONE","repair_authority_claim":"NONE"
        }
        summary={"schema":"symthaea.se001q.lock-delta-observation.v1","observation_id":sha_bytes(canonical(summary_identity)),"identity":summary_identity}
        (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

        if status=="REPRODUCIBLE_LOCK_DELTA":
            witness_identity={
                "domain":"symthaea.lock-delta-witness.v1","subject_sha":subject_sha,
                "source_lock_sha256":before_lock,"generated_lock_sha256":offline_lock,
                "lock_patch_sha256":sha_file(output/"Cargo.lock.patch"),
                "online_resolution_result_sha256":sha_file(output/"online-all-targets/command-result.json"),
                "offline_resolution_result_sha256":sha_file(output/"offline-reproduction-all-targets/command-result.json"),
                "changed_tracked_paths":["Cargo.lock"],"offline_reproduction":"PASS",
                "counterfactual_lock_stability":"PASS","lock_boundary_cleared":"PASS",
                "source_preservation":"PASS","qualification_claim":"NONE","repair_authority_claim":"NONE"
            }
            witness={"schema":"symthaea.lock-delta-witness.v1","witness_id":sha_bytes(canonical(witness_identity)),"identity":witness_identity}
            (output/"lock-delta-witness.json").write_text(json.dumps(witness,indent=2,sort_keys=True)+"\n")

        files=[]
        for p in sorted(x for x in output.rglob('*') if x.is_file() and x.name!="manifest.json"):
            files.append({"path":p.relative_to(output).as_posix(),"sha256":sha_file(p),"bytes":p.stat().st_size})
        manifest_identity={"domain":"symthaea.se001q.lock-delta-manifest.v1","subject_sha":subject_sha,"observation_id":summary["observation_id"],"files":files,"qualification_claim":"NONE","repair_authority_claim":"NONE"}
        manifest={"schema":"symthaea.se001q.lock-delta-manifest.v1","manifest_id":sha_bytes(canonical(manifest_identity)),"identity":manifest_identity}
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"result":"CAPTURE_COMPLETE","status":status,"observation_id":summary["observation_id"],"manifest_id":manifest["manifest_id"],"lock_delta_witness": (json.loads((output/"lock-delta-witness.json").read_text())["witness_id"] if (output/"lock-delta-witness.json").exists() else None),"qualification_claim":"NONE","repair_authority_claim":"NONE"},sort_keys=True))
    finally:
        git(subject,"worktree","remove","--force",str(a),check=False)
        git(subject,"worktree","remove","--force",str(b),check=False)

if __name__=="__main__": main()
