#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, pathlib, tempfile
from typing import Any

POLICY="ll009v.product90-rms-semantics-policy.v1"; LOCK="ll009n.nasa-source-lock.v1"; LCFG="ll009l.cog-materialization-config.v1"; MREC="ll009m.radial-uncertainty-receipt.v1"; OUT="ll009v.product90-rms-semantics-receipt.v1"
class VError(RuntimeError): pass

def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(",",": "))+"\n").encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def read(p:pathlib.Path,n:str)->dict:
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as e:raise VError(f"cannot read {n}: {e}") from e
 if not isinstance(v,dict):raise VError(f"{n} must contain object")
 return v
def hex64(v,n):
 if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):raise VError(f"{n} must be lowercase SHA-256")
 return v
def selfhash(v,n):
 x=hex64(v.get("receipt_sha256"),f"{n} receipt_sha256");b=json.loads(json.dumps(v));b.pop("receipt_sha256",None)
 if x!=hb(cb(b)):raise VError(f"{n}: receipt self-hash mismatch")
def check_policy(p):
 if p.get("schema_version")!=POLICY:raise VError("policy schema mismatch")
 for k in ("study_id","layer_id","uncertainty_source_id","required_source_role","quantity","direction_semantics"):
  if not isinstance(p.get(k),str) or not p[k]:raise VError(f"policy missing {k}")
 if p.get("semantics_class")!="rms_error":raise VError("V may only qualify rms_error")
 blocked=set(p.get("blocked_claims",[]))
 if not {"hard_upper_bound","deterministic_maximum_error"}<=blocked:raise VError("policy must explicitly block deterministic bound semantics")
 refs=p.get("methodology_lineage")
 if not isinstance(refs,list) or len(refs)<2 or not all(isinstance(x,dict) and isinstance(x.get("reference"),str) and isinstance(x.get("statement"),str) for x in refs):raise VError("methodology_lineage incomplete")
 return p
def lock_source(lock,p):
 if lock.get("schema_version")!=LOCK or lock.get("study_id")!=p["study_id"]:raise VError("source lock schema/study mismatch")
 fs=lock.get("files")
 if not isinstance(fs,list):raise VError("source lock files missing")
 a=[x for x in fs if isinstance(x,dict) and x.get("source_id")==p["uncertainty_source_id"]]
 if len(a)!=1:raise VError("expected exactly one Product 90 uncertainty source")
 x=a[0]
 if x.get("role")!=p["required_source_role"]:raise VError("Product 90 source role mismatch")
 hex64(x.get("sha256"),"locked Product 90 uncertainty hash")
 if not isinstance(x.get("artifact_path"),str) or not x["artifact_path"]:raise VError("locked Product 90 artifact_path missing")
 return x
def l_bind(l,p,src):
 if l.get("schema_version")!=LCFG or l.get("study_id")!=p["study_id"]:raise VError("L config schema/study mismatch")
 a=[x for x in l.get("layers",[]) if isinstance(x,dict) and x.get("layer_id")==p["layer_id"]]
 if len(a)!=1:raise VError("expected exactly one L companion layer")
 u=a[0].get("uncertainty_source")
 if not isinstance(u,dict) or u.get("path")!=src["artifact_path"] or u.get("sha256")!=src["sha256"]:raise VError("L uncertainty source does not bind exact Product 90 bytes")
 return a[0]
def m_bind(m,p,src,l_path):
 if m.get("schema_version")!=MREC or m.get("status")!="pass" or m.get("study_id")!=p["study_id"]:raise VError("M receipt schema/status/study mismatch")
 selfhash(m,"M receipt")
 if m.get("l_config_sha256")!=hf(l_path):raise VError("M does not bind exact L config")
 a=[x for x in m.get("layers",[]) if isinstance(x,dict) and x.get("layer_id")==p["layer_id"]]
 if len(a)!=1 or a[0].get("uncertainty_source_sha256")!=src["sha256"]:raise VError("M does not bind exact Product 90 uncertainty bytes")
 return a[0]
def qualify(policy_path,lock_path,l_path,m_path=None):
 p=check_policy(read(policy_path,"policy"));lock=read(lock_path,"source lock");l=read(l_path,"L config");src=lock_source(lock,p);layer=l_bind(l,p,src);m_hash=None
 if m_path is not None:
  m=read(m_path,"M receipt");m_bind(m,p,src,l_path);m_hash=hf(m_path)
 out={"schema_version":OUT,"status":"pass","study_id":p["study_id"],"layer_id":p["layer_id"],"uncertainty_source_id":p["uncertainty_source_id"],"uncertainty_source_sha256":src["sha256"],"uncertainty_artifact_path":src["artifact_path"],"semantics_class":"rms_error","quantity":p["quantity"],"direction_semantics":p["direction_semantics"],"deterministic_upper_bound_eligible":False,"statistical_closure_required":True,"policy_sha256":hf(policy_path),"source_lock_sha256":hf(lock_path),"l_config_sha256":hf(l_path),"m_receipt_sha256":m_hash,"l_layer_range":{"min_range_m":layer.get("min_range_m"),"max_range_m":layer.get("max_range_m"),"nominal_resolution_m":layer.get("expected_pixel_size_m")},"methodology_lineage":p["methodology_lineage"],"allowed_claims":p.get("allowed_claims",[]),"blocked_claims":p["blocked_claims"],"claim_rule":"Product 90 ADJ_ERR is admitted as RMS/second-moment terrain-height uncertainty only; no multiplier, tail probability, independence assumption or deterministic maximum is implied.","non_claims":p.get("non_claims",[])};out["receipt_sha256"]=hb(cb(out));return out
def write(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise VError(f"refusing to overwrite differing output {p}")
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def selftest():
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td);digest=hb(b"rms");pol={"schema_version":POLICY,"study_id":"s","layer_id":"far","uncertainty_source_id":"u","required_source_role":"surface_height_error_m","semantics_class":"rms_error","quantity":"gridded_surface_height_error_m_relative_true_average_pixel_height","direction_semantics":"scalar_lunar_surface_height_not_deterministic_vector_bound","methodology_lineage":[{"reference":"a","statement":"rms method"},{"reference":"b","statement":"product mapping"}],"allowed_claims":["rms_error"],"blocked_claims":["hard_upper_bound","deterministic_maximum_error","implicit_gaussian"]};pp=r/"p.json";pp.write_bytes(cb(pol));lock={"schema_version":LOCK,"study_id":"s","files":[{"source_id":"u","role":"surface_height_error_m","artifact_path":"u.tif","sha256":digest}]};lp=r/"lock.json";lp.write_bytes(cb(lock));lc={"schema_version":LCFG,"study_id":"s","layers":[{"layer_id":"far","min_range_m":1.,"max_range_m":10.,"expected_pixel_size_m":80.,"uncertainty_source":{"path":"u.tif","sha256":digest}}]};lcp=r/"l.json";lcp.write_bytes(cb(lc));m={"schema_version":MREC,"status":"pass","study_id":"s","l_config_sha256":hf(lcp),"layers":[{"layer_id":"far","uncertainty_source_sha256":digest}]};m["receipt_sha256"]=hb(cb(m));mp=r/"m.json";mp.write_bytes(cb(m));a=qualify(pp,lp,lcp,mp);b=qualify(pp,lp,lcp,mp);assert cb(a)==cb(b) and a["semantics_class"]=="rms_error" and not a["deterministic_upper_bound_eligible"]
  bad=json.loads(json.dumps(lock));bad["files"][0]["sha256"]="1"*64;bp=r/"bad.json";bp.write_bytes(cb(bad))
  try:qualify(pp,bp,lcp,mp);raise AssertionError("hash drift accepted")
  except VError:pass
  hard=json.loads(json.dumps(pol));hard["semantics_class"]="hard_upper_bound";hp=r/"hard.json";hp.write_bytes(cb(hard))
  try:qualify(hp,lp,lcp,mp);raise AssertionError("hard semantics accepted")
  except VError as e:assert "rms_error" in str(e)
  print("LL-009V Product 90 RMS semantics self-test: PASS")
def main():
 a=argparse.ArgumentParser();a.add_argument("--policy");a.add_argument("--source-lock");a.add_argument("--l-config");a.add_argument("--m-receipt");a.add_argument("--output");a.add_argument("--self-test",action="store_true");z=a.parse_args()
 try:
  if z.self_test:selftest();return 0
  if not all((z.policy,z.source_lock,z.l_config,z.output)):raise VError("--policy --source-lock --l-config --output required")
  o=qualify(pathlib.Path(z.policy),pathlib.Path(z.source_lock),pathlib.Path(z.l_config),pathlib.Path(z.m_receipt) if z.m_receipt else None);write(pathlib.Path(z.output),o);print(json.dumps(o,sort_keys=True,indent=2));return 0
 except (OSError,json.JSONDecodeError,VError) as e:raise SystemExit(f"LL-009V failure: {e}") from e
if __name__=="__main__":raise SystemExit(main())
