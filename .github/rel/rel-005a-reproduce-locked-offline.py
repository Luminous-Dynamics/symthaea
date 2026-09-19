#!/usr/bin/env python3
"""Run REL-005A independent reproduction with locked dependency preparation.

Authority is split deliberately:
- IndependentReproductionDependencyPreparationOnly prepares the exact locked Cargo graph.
- IndependentReproductionEnvironmentOnly constrains ambient Cargo/Rust environment state.
- IndependentReproductionOnly remains in the frozen v3 harness and alone adjudicates the
  fresh reproduction through the frozen generic ComparisonOnly evaluator.

This wrapper does not parse scientific observations, apply scientific predicates, or
change the original REL-005A qualification result. CARGO_NET_OFFLINE is a Cargo-level
network control only; this is not an OS/network sandbox.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, runpy, subprocess, sys, tempfile
from typing import Any

DEPENDENCY_AUTHORITY="IndependentReproductionDependencyPreparationOnly"
DEPENDENCY_SCHEMA="symthaea.rel.reproduction-dependency-preparation-receipt.v1"
ENV_AUTHORITY="IndependentReproductionEnvironmentOnly"
ENV_SCHEMA="symthaea.rel.reproduction-environment-envelope-receipt.v2"
MANIFEST_SCHEMA="symthaea.rel.reproduction-environment-envelope-manifest.v2"

def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def no_duplicates(pairs:list[tuple[str,Any]])->dict[str,Any]:
    out={}
    for k,v in pairs: require(k not in out,f"duplicate JSON key: {k}"); out[k]=v
    return out
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text(),object_pairs_hook=no_duplicates); require(isinstance(v,dict),f"{p}: expected JSON object"); return v
def write(p:pathlib.Path,v:Any)->None:p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha(p:pathlib.Path)->str:return sha_bytes(p.read_bytes())
def git_at(cwd:pathlib.Path,*args:str)->str:return subprocess.check_output(["git",*args],cwd=cwd,text=True).strip()
def run_capture(cmd:list[str],cwd:pathlib.Path,env:dict[str,str])->subprocess.CompletedProcess[str]:return subprocess.run(cmd,cwd=cwd,env=env,text=True,capture_output=True)
def save_process(out:pathlib.Path,stem:str,r:subprocess.CompletedProcess[str])->None:
    (out/f"{stem}.stdout.log").write_text(r.stdout); (out/f"{stem}.stderr.log").write_text(r.stderr)
def inside(p:pathlib.Path,root:pathlib.Path)->bool:
    try:p.relative_to(root);return True
    except ValueError:return False

def validate_dependency_contract(root:pathlib.Path,recipe:dict[str,Any])->dict[str,Any]:
    d=recipe.get("dependency_preparation"); require(isinstance(d,dict),"dependency preparation contract missing")
    require(d.get("authority")==DEPENDENCY_AUTHORITY,"dependency preparation authority mismatch")
    require(d.get("execution_subject_head")==recipe.get("execution_subject_head"),"dependency subject mismatch")
    require(d.get("cargo_lock_path")=="Cargo.lock","Cargo.lock path mismatch")
    require(d.get("cargo_lock_blob")==recipe.get("expected_git_blobs",{}).get("Cargo.lock"),"Cargo.lock blob binding mismatch")
    require(d.get("online_prefetch_allowed") is True,"locked online prefetch not allowed")
    require(d.get("offline_probe_required") is True,"offline dependency probe not required")
    require(d.get("actual_measurement_cargo_offline_required") is True,"offline measurement not required")
    require(d.get("cargo_level_offline_only") is True,"Cargo-level offline limitation missing")
    require(d.get("os_network_sandbox") is False,"dependency contract overclaims OS network sandbox")
    require(d.get("prefetch_command")=="rustup run 1.96.0 cargo fetch --locked","prefetch command drift")
    require(d.get("offline_probe_command")=="rustup run 1.96.0 cargo fetch --locked --offline","offline probe command drift")
    wp=root/d["wrapper_path"]; require(wp.is_file(),"v5 dependency wrapper missing")
    require(git_at(root,"rev-parse",f"HEAD:{d['wrapper_path']}")==d["wrapper_blob"],"v5 dependency wrapper blob mismatch")
    require(not any(d.get("claims",{}).values()),"dependency preparation contract makes runtime/scientific claims")
    return d

def verify_prefetch_subject(root:pathlib.Path,recipe:dict[str,Any],d:dict[str,Any])->None:
    require(git_at(root,"rev-parse","HEAD")==recipe["execution_subject_head"],"prefetch worktree subject mismatch")
    require(git_at(root,"rev-parse","HEAD:Cargo.lock")==d["cargo_lock_blob"],"prefetch Cargo.lock blob mismatch")
    require(git_at(root,"status","--porcelain=v1","--untracked-files=all")=="","prefetch worktree not clean")
    require(not (root/".cargo/config").exists(),"unexpected exact-subject .cargo/config")
    require(not (root/".cargo/config.toml").exists(),"unexpected exact-subject .cargo/config.toml")
def environment_receipt(root:pathlib.Path,recipe_path:pathlib.Path,c:dict[str,Any],dep_receipt:pathlib.Path,result:subprocess.CompletedProcess[str],dropped_count:int,dropped_build:list[str],inherited_hashes:dict[str,str],inner_path:pathlib.Path)->dict[str,Any]:
    present=inner_path.is_file(); inner_sha=sha(inner_path) if present else None; completed=False
    if present:
        i=load(inner_path); require(i.get("authority")=="IndependentReproductionOnly","inner reproduction authority mismatch")
        require(i.get("original_rel_005a_qualification_changed") is False,"inner reproduction changed original qualification")
        claims=i.get("claims"); require(isinstance(claims,dict),"inner reproduction claims missing")
        require(claims.get("original_rel_005a_qualification_changed") is False,"inner claims changed original qualification")
        require(claims.get("scientific_pass") is False and claims.get("scientific_fail") is False,"inner reproduction exceeded authority")
        completed=claims.get("independent_reproduction_completed") is True
    return {"schema":ENV_SCHEMA,"authority":ENV_AUTHORITY,"relation":"REL-005A","harness_subject_head":git_at(root,"rev-parse","HEAD"),"harness_blob":c["harness_blob"],"environment_wrapper_blob":c["wrapper_blob"],"dependency_preparation_receipt_sha256":sha(dep_receipt),"recipe_sha256":sha(recipe_path),"harness_exit_code":result.returncode,"fresh_cargo_home_used":True,"fresh_cargo_home_removed":True,"cargo_home_outside_harness_worktree":True,"dropped_non_allowlisted_environment_key_count":dropped_count,"dropped_build_control_keys":dropped_build,"dropped_build_control_key_count":len(dropped_build),"inherited_environment_keys":sorted(inherited_hashes),"inherited_nonsecret_environment_value_sha256":inherited_hashes,"sensitive_inherited_values_recorded":False,"path_sha256":inherited_hashes.get("PATH"),"rustup_home_inherited":"RUSTUP_HOME" in inherited_hashes,"cargo_net_offline_injected_after_dependency_preparation":True,"cargo_net_offline_value":"true","cargo_level_offline_only":True,"os_network_sandbox":False,"inner_reproduction_receipt_present":present,"inner_reproduction_receipt_sha256":inner_sha,"inner_reproduction_completed":completed,"scientific_payload_parsed":False,"scientific_result_interpreted":False,"original_rel_005a_qualification_changed":False,"claims":{"environment_envelope_applied":True,"dependency_preparation_completed":True,"independent_reproduction_completed":False,"original_rel_005a_qualification_changed":False,"scientific_pass":False,"scientific_fail":False}}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--recipe",type=pathlib.Path,required=True); ap.add_argument("--output-dir",type=pathlib.Path,required=True); a=ap.parse_args()
    root=pathlib.Path(git_at(pathlib.Path.cwd(),"rev-parse","--show-toplevel")).resolve(); recipe_path=a.recipe.expanduser().resolve(); recipe=load(recipe_path); dep=validate_dependency_contract(root,recipe)
    v4_path=root/recipe["environment_envelope"]["wrapper_path"]; require(v4_path.is_file(),"v4 environment wrapper missing")
    v4=runpy.run_path(str(v4_path),run_name="rel005a_environment_policy"); env_contract=v4["validate_contract"](root,recipe_path,recipe); build_sanitized_env=v4["build_sanitized_env"]
    output=a.output_dir.expanduser().resolve(); require(not inside(output,root),"output directory must be outside harness worktree")
    if output.exists():require(not any(output.iterdir()),"output directory must start empty")
    harness_path=root/env_contract["harness_path"]; worktrees_before=git_at(root,"worktree","list","--porcelain")
    with tempfile.TemporaryDirectory(prefix="rel005a-cargo-home-v5-") as cargo_home_tmp,tempfile.TemporaryDirectory(prefix="rel005a-prefetch-parent-") as prefetch_parent,tempfile.TemporaryDirectory(prefix="rel005a-dependency-evidence-") as prep_tmp:
        cargo_home=pathlib.Path(cargo_home_tmp).resolve(); prep=pathlib.Path(prep_tmp).resolve(); require(not inside(cargo_home,root),"fresh CARGO_HOME must be outside harness worktree")
        sanitized,dropped_count,dropped_build,inherited_hashes=build_sanitized_env(cargo_home)
        prefetch_root=pathlib.Path(prefetch_parent).resolve()/"subject"; add=run_capture(["git","worktree","add","--detach",str(prefetch_root),recipe["execution_subject_head"]],root,sanitized); save_process(prep,"dependency-worktree-add",add); require(add.returncode==0,"failed to create exact-subject dependency worktree")
        prefetch=None; offline_probe=None
        try:
            verify_prefetch_subject(prefetch_root,recipe,dep); prefetch=run_capture(dep["prefetch_command"].split(),prefetch_root,sanitized); save_process(prep,"dependency-prefetch-online",prefetch); require(prefetch.returncode==0,"locked dependency prefetch failed")
            probe_env=dict(sanitized); probe_env["CARGO_NET_OFFLINE"]="true"; offline_probe=run_capture(dep["offline_probe_command"].split(),prefetch_root,probe_env); save_process(prep,"dependency-prefetch-offline-probe",offline_probe); require(offline_probe.returncode==0,"locked dependency graph not available offline after prefetch")
            require(git_at(prefetch_root,"status","--porcelain=v1","--untracked-files=all")=="","dependency preparation changed exact-subject worktree")
        finally:
            remove=run_capture(["git","worktree","remove","--force",str(prefetch_root)],root,sanitized); save_process(prep,"dependency-worktree-remove",remove); require(remove.returncode==0,"failed to remove dependency worktree"); require(not prefetch_root.exists(),"dependency worktree leaked")
        require(prefetch is not None and offline_probe is not None,"dependency preparation did not complete")
        dep_receipt={"schema":DEPENDENCY_SCHEMA,"authority":DEPENDENCY_AUTHORITY,"relation":"REL-005A","execution_subject_head":recipe["execution_subject_head"],"cargo_lock_path":dep["cargo_lock_path"],"cargo_lock_blob":dep["cargo_lock_blob"],"prefetch_command":dep["prefetch_command"],"prefetch_exit_code":prefetch.returncode,"offline_probe_command":dep["offline_probe_command"],"offline_probe_exit_code":offline_probe.returncode,"fresh_cargo_home_used":True,"repository_local_cargo_config_present":False,"dependency_graph_available_to_cargo_offline":True,"actual_measurement_cargo_offline_required":True,"cargo_level_offline_only":True,"os_network_sandbox":False,"scientific_payload_parsed":False,"scientific_result_interpreted":False,"original_rel_005a_qualification_changed":False,"claims":{"dependency_preparation_completed":True,"independent_reproduction_completed":False,"original_rel_005a_qualification_changed":False,"scientific_pass":False,"scientific_fail":False}}
        measurement_env=dict(sanitized); measurement_env["CARGO_NET_OFFLINE"]="true"; require(measurement_env.get("CARGO_NET_OFFLINE")=="true","Cargo offline mode not injected")
        result=subprocess.run([sys.executable,str(harness_path),"--recipe",str(recipe_path),"--output-dir",str(output)],cwd=root,env=measurement_env,text=True,capture_output=True); cargo_home_path=str(cargo_home)
        output.mkdir(parents=True,exist_ok=True); save_process(output,"hermetic-harness",result)
        for p in prep.iterdir():
            if p.is_file():(output/p.name).write_bytes(p.read_bytes())
        dep_receipt_path=output/"reproduction-dependency-preparation-receipt.json"; write(dep_receipt_path,dep_receipt)
    require(not pathlib.Path(cargo_home_path).exists(),"temporary CARGO_HOME leaked after reproduction"); require(git_at(root,"worktree","list","--porcelain")==worktrees_before,"dependency worktree census changed")
    envelope=environment_receipt(root,recipe_path,env_contract,dep_receipt_path,result,dropped_count,dropped_build,inherited_hashes,output/"independent-reproduction-receipt.json"); write(output/"reproduction-environment-envelope-receipt.json",envelope)
    targets=["dependency-worktree-add.stdout.log","dependency-worktree-add.stderr.log","dependency-prefetch-online.stdout.log","dependency-prefetch-online.stderr.log","dependency-prefetch-offline-probe.stdout.log","dependency-prefetch-offline-probe.stderr.log","dependency-worktree-remove.stdout.log","dependency-worktree-remove.stderr.log","hermetic-harness.stdout.log","hermetic-harness.stderr.log","reproduction-dependency-preparation-receipt.json","reproduction-environment.json","reproduction-execution-receipt.json","reproduction-observation-seal.json","reproduction-comparison-only.json","independent-reproduction-receipt.json","reproduction-environment-envelope-receipt.json"]
    entries=[]
    for n in targets:
        p=output/n
        if p.is_file():entries.append({"basename":n,"byte_length":p.stat().st_size,"sha256":sha(p)})
    manifest={"schema":MANIFEST_SCHEMA,"authority":ENV_AUTHORITY,"relation":"REL-005A","files":sorted(entries,key=lambda x:x["basename"].encode()),"dependency_preparation_receipt_sha256":sha(dep_receipt_path),"harness_exit_code":result.returncode,"cargo_level_offline_measurement":True,"os_network_sandbox":False,"original_rel_005a_qualification_changed":False,"claims":{"scientific_pass":False,"scientific_fail":False}}
    write(output/"reproduction-environment-envelope-manifest.json",manifest); return result.returncode

if __name__=="__main__":raise SystemExit(main())
