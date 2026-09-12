#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, tempfile
from typing import Any

CONFIG_SCHEMA="ll009s.semantic-visibility-config.v1"; OUT="ll009s.semantic-visibility-receipt.v1"
K_SCHEMA="ll009k.horizon-pack.v1"; O_SCHEMA="ll009o.uncertainty-semantics-receipt.v1"; R_SCHEMA="ll009r.spatial-support-classification-receipt.v1"
CLAIMS={"deterministic_visibility_bound","risk_qualified_visibility","empirical_sampled_visibility","descriptive_geometry_only"}; KINDS={"sun","earth","relay"}
class SError(RuntimeError): pass

def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(",",": "))+"\n").encode()
def hbytes(v:bytes)->str:return hashlib.sha256(v).hexdigest()
def hfile(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def v3(v,n):
 if not isinstance(v,(list,tuple)) or len(v)!=3 or not all(finite(x) for x in v):raise SError(f"{n} must be finite vec3")
 return tuple(float(x) for x in v)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def scale(a,s):return tuple(x*s for x in a)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def unit(a,n):
 m=norm(a)
 if m<=1e-12 or not math.isfinite(m):raise SError(f"{n} degenerate")
 return scale(a,1/m)
def azel(d,b):
 n,e,u,_=b; d=unit(d,"target direction"); return math.degrees(math.atan2(dot(d,e),dot(d,n)))%360,math.degrees(math.atan2(dot(d,u),math.hypot(dot(d,n),dot(d,e))))
def rel(v):
 p=pathlib.PurePosixPath(v)
 if not isinstance(v,str) or not v or p.is_absolute() or ".." in p.parts:raise SError(f"unsafe path {v!r}")
 return p
def read(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as e:raise SError(f"cannot read {n}: {e}") from e
 if not isinstance(v,dict):raise SError(f"{n} must be object")
 return v

def sources(cfg,root):
 ids=set()
 for s in cfg.get("sources",[]):
  if not isinstance(s,dict):raise SError("source must be object")
  i,p,d=s.get("source_id"),s.get("path"),s.get("sha256")
  if not all(isinstance(x,str) and x for x in (i,p,d)) or len(d)!=64:raise SError("malformed source")
  if i in ids:raise SError(f"duplicate source {i}")
  f=root/pathlib.Path(*rel(p).parts)
  if not f.is_file() or hfile(f)!=d:raise SError(f"source missing/hash mismatch {i}")
  ids.add(i)
 if not ids:raise SError("sources required")
 return ids

def validate_cfg(c):
 if c.get("schema_version")!=CONFIG_SCHEMA or c.get("requested_claim_class") not in CLAIMS:raise SError("invalid S config/schema/claim")
 p=c.get("policies",{}); mt=p.get("max_time_gap_s"); lm=p.get("los_margin_deg")
 if not finite(mt) or mt<=0 or not finite(lm) or lm<0:raise SError("invalid temporal/LOS policy")
 if not isinstance(c.get("targets"),list) or not c["targets"] or not isinstance(c.get("sources"),list):raise SError("targets/sources required")
 return c

def validate_k(k):
 if k.get("schema_version")!=K_SCHEMA or k.get("status")!="pass":raise SError("invalid K pack")
 for x in ("study_id","frame_contract_id","epoch_contract_id","site_ref","frame"):
  if not isinstance(k.get(x),str) or not k[x]:raise SError(f"K missing {x}")
 w,n=k.get("azimuth_bin_width_deg"),k.get("bin_count")
 if not finite(w) or w<=0 or not isinstance(n,int) or n<4 or abs(w*n-360)>1e-9:raise SError("invalid K bins")
 bs=k.get("bins")
 if not isinstance(bs,list) or len(bs)!=n:raise SError("K bin count mismatch")
 idx={}
 for x in bs:
  i=x.get("bin_index"); e=x.get("conservative_elevation_deg")
  if not isinstance(i,int) or i in idx or not 0<=i<n or not finite(e):raise SError("invalid/duplicate K bin")
  if abs(float(x.get("azimuth_start_deg",math.nan))-i*w)>1e-9 or abs(float(x.get("azimuth_end_deg",math.nan))-(i+1)*w)>1e-9:raise SError("K bin boundary drift")
  idx[i]=x
 if set(idx)!=set(range(n)):raise SError("K coverage incomplete")
 b=k.get("basis",{}); basis=(v3(b.get("north"),"north"),v3(b.get("east"),"east"),v3(b.get("up"),"up"),b.get("fallback_used"))
 if not isinstance(basis[3],bool) or any(abs(norm(a)-1)>1e-9 for a in basis[:3]) or any(abs(dot(a,b))>1e-9 for a,b in ((basis[0],basis[1]),(basis[0],basis[2]),(basis[1],basis[2]))):raise SError("K basis invalid")
 return idx,basis

def capability(o,r,k):
 if o.get("schema_version")!=O_SCHEMA or r.get("schema_version")!=R_SCHEMA or o.get("study_id")!=k["study_id"] or r.get("study_id")!=k["study_id"]:raise SError("O/R/K lineage mismatch")
 vd=o.get("deterministic_upper_bound_eligible"); vr=o.get("risk_qualified_horizon_eligible"); sp=r.get("strongest_common_spatial_support")
 if not isinstance(vd,bool) or not isinstance(vr,bool) or sp not in {"continuous_hard_bound","empirical_multiscale_bound","resolution_qualified","sample_points_only","unknown"}:raise SError("invalid O/R capability")
 stat=isinstance(k.get("statistical_horizon_binding"),dict) and k["statistical_horizon_binding"].get("status")=="bound"
 det=vd and sp=="continuous_hard_bound"; risk=vr and sp in {"continuous_hard_bound","empirical_multiscale_bound","resolution_qualified"} and stat; emp=vr and sp!="unknown" and stat
 if det:strong,ec="deterministic_visibility_bound","derived_deterministic_bound"
 elif risk:strong,ec="risk_qualified_visibility","derived_risk_qualified"
 elif emp:strong,ec="empirical_sampled_visibility","derived_empirical_sampled"
 else:strong,ec="descriptive_geometry_only","descriptive_geometry_only"
 return {"vertical_deterministic_upper_bound":vd,"vertical_risk_eligible_upstream":vr,"spatial_support_class":sp,"k_statistical_horizon_binding_present":stat,"deterministic_visibility_bound_eligible":det,"risk_qualified_visibility_eligible":risk,"empirical_sampled_visibility_eligible":emp,"strongest_available_claim_class":strong,"metric_evidence_class":ec}
def allow(req,c):
 key={"deterministic_visibility_bound":"deterministic_visibility_bound_eligible","risk_qualified_visibility":"risk_qualified_visibility_eligible","empirical_sampled_visibility":"empirical_sampled_visibility_eligible"}.get(req)
 if key is None:return True,"descriptive geometry allowed with explicit capability disclosure"
 ok=c[key]
 if ok:return True,f"{req} capability established"
 if req=="deterministic_visibility_bound":return False,"deterministic visibility requires O deterministic eligibility and R continuous-hard spatial support"
 return False,f"{req} requires upstream O/R eligibility and an exact statistical_horizon_binding in K"
def bin_at(a,w,n):return int(math.floor(((a%360)+1e-9)/w))%n

def tint(rows):
 t=[x["t_s"] for x in rows]; v=[x["visible"] for x in rows]; total=t[-1]-t[0]
 if total<=0:raise SError("time span invalid")
 lo=ce=hi=0.0
 for i in range(len(rows)-1):
  dt=t[i+1]-t[i]
  if v[i] and v[i+1]:lo+=dt;ce+=dt;hi+=dt
  elif v[i]!=v[i+1]:ce+=dt/2;hi+=dt
 return {"low":lo/total,"central":ce/total,"high":hi/total}
def blocked(rows):
 t=[x["t_s"] for x in rows];v=[x["visible"] for x in rows];best={"lower_s":0.0,"central_s":0.0,"upper_s":0.0,"start_index":None,"end_index":None};i=0
 while i<len(v):
  if v[i]:i+=1;continue
  j=i
  while j+1<len(v) and not v[j+1]:j+=1
  lo=max(0,t[j]-t[i]);L=t[i]-t[i-1] if i else 0;R=t[j+1]-t[j] if j+1<len(t) else 0;up=lo+L+R;ce=lo+(L+R)/2
  if up>best["upper_s"]+1e-12:best={"lower_s":lo,"central_s":ce,"upper_s":up,"start_index":i,"end_index":j}
  i=j+1
 return best
def metric(cat,iv,unit,refs,ec):return {"category":cat,"status":"available","low":iv["low"],"central":iv["central"],"high":iv["high"],"unit":unit,"evidence_class":ec,"source_refs":refs}

def target(t,frame,maxt,basis,bins,w,n,margin):
 if not isinstance(t,dict) or not isinstance(t.get("target_id"),str) or t.get("kind") not in KINDS or t.get("frame")!=frame:raise SError("invalid target")
 tid,kind=t["target_id"],t["kind"]; rad=t.get("apparent_angular_radius_deg",0.0)
 if not finite(rad) or rad<0 or (kind=="sun" and "apparent_angular_radius_deg" not in t):raise SError("invalid target angular radius")
 ss=t.get("samples")
 if not isinstance(ss,list) or len(ss)<2:raise SError("target samples required")
 out=[];prev=None
 for i,s in enumerate(ss):
  if not isinstance(s,dict) or not finite(s.get("t_s")):raise SError("invalid target epoch")
  tm=float(s["t_s"])
  if prev is not None and (tm<=prev or tm-prev>maxt+1e-12):raise SError("invalid target time gap")
  prev=tm;az,el=azel(v3(s.get("direction"),f"{tid}[{i}] direction"),basis);bi=bin_at(az,w,n);kh=float(bins[bi]["conservative_elevation_deg"]);hz=kh+margin
  out.append({"t_s":tm,"azimuth_deg":az,"elevation_deg":el,"k_bin_index":bi,"k_bin_horizon_deg":kh,"los_margin_deg":margin,"effective_horizon_deg":hz,"center_visible":el>hz,"full_disc_visible":el-float(rad)>hz})
 return tid,kind,float(rad),out

def generate(cp,kp,op,rp,root):
 cfg=validate_cfg(read(cp,"S config"));k=read(kp,"K pack");o=read(op,"O receipt");r=read(rp,"R receipt");bins,basis=validate_k(k);cap=capability(o,r,k);ok,reason=allow(cfg["requested_claim_class"],cap);ids=sources(cfg,root)
 maxt=float(cfg["policies"]["max_time_gap_s"]);margin=float(cfg["policies"]["los_margin_deg"]);w=float(k["azimuth_bin_width_deg"]);n=int(k["bin_count"]);ec=cap["metric_evidence_class"] if ok else "descriptive_geometry_only";metrics={};tout=[]
 for t in cfg["targets"]:
  refs=t.get("source_refs",[]) if isinstance(t,dict) else []
  if not isinstance(refs,list) or not refs or any(x not in ids for x in refs):raise SError("target source_refs invalid")
  tid,kind,rad,rows=target(t,k["frame"],maxt,basis,bins,w,n,margin);cr=[{"t_s":x["t_s"],"visible":x["center_visible"]} for x in rows];fr=[{"t_s":x["t_s"],"visible":x["full_disc_visible"]} for x in rows];ci=tint(cr);cbk=blocked(cr)
  if kind=="sun":
   fi=tint(fr);fb=blocked(fr);metrics["solar_center_visibility_fraction"]=metric("illumination",ci,"fraction",refs,ec);metrics["solar_full_disc_visibility_fraction"]=metric("illumination",fi,"fraction",refs,ec);metrics["longest_full_solar_occlusion_h"]=metric("illumination",{"low":fb["lower_s"]/3600,"central":fb["central_s"]/3600,"high":fb["upper_s"]/3600},"h",refs,ec)
  elif kind=="earth":metrics["dte_los_availability_fraction"]=metric("communications",ci,"fraction",refs,ec);metrics["max_contiguous_dte_outage_h"]=metric("communications",{"low":cbk["lower_s"]/3600,"central":cbk["central_s"]/3600,"high":cbk["upper_s"]/3600},"h",refs,ec)
  else:
   key=tid.lower().replace("-","_");metrics[f"{key}_los_availability_fraction"]=metric("communications",ci,"fraction",refs,ec);metrics[f"max_contiguous_{key}_outage_h"]=metric("communications",{"low":cbk["lower_s"]/3600,"central":cbk["central_s"]/3600,"high":cbk["upper_s"]/3600},"h",refs,ec)
  tout.append({"target_id":tid,"kind":kind,"apparent_angular_radius_deg":rad,"samples":rows,"center_visibility_interval":ci,"longest_center_blocked_s":cbk,"full_disc_visibility_interval":tint(fr) if kind=="sun" else None,"longest_full_disc_blocked_s":blocked(fr) if kind=="sun" else None})
 out={"schema_version":OUT,"status":"pass" if ok else "claim_blocked_geometry_available","study_id":k["study_id"],"frame_contract_id":k["frame_contract_id"],"epoch_contract_id":k["epoch_contract_id"],"site_ref":k["site_ref"],"frame":k["frame"],"requested_claim_class":cfg["requested_claim_class"],"requested_claim_allowed":ok,"claim_reason":reason,"strongest_available_claim_class":cap["strongest_available_claim_class"],"capability":cap,"metric_evidence_class":ec,"config_sha256":hfile(cp),"k_horizon_pack_sha256":hfile(kp),"o_uncertainty_receipt_sha256":hfile(op),"r_spatial_support_receipt_sha256":hfile(rp),"source_artifact_hashes":{x["source_id"]:x["sha256"] for x in cfg["sources"]},"horizon_numeric_authority":K_SCHEMA,"horizon_lookup_policy":"piecewise_constant_maximum_by_k_bin","azimuth_bin_width_deg":w,"los_margin_deg":margin,"metrics":metrics,"targets":tout,"non_claims":["A LOS calculation is not promoted beyond exact O/R capability.","K v1 lacks a bound statistical horizon, so RMS/ensemble evidence does not automatically become risk-qualified visibility.","LOS is not an RF link budget, delivered power model, site qualification, or operations authority."]};out["receipt_sha256"]=hbytes(cb(out));return out

def direction(b,az,el):
 n,e,u,_=b;A=math.radians(az);E=math.radians(el);h=add(scale(n,math.cos(A)),scale(e,math.sin(A)));return list(add(scale(h,math.cos(E)),scale(u,math.sin(E))))
def write(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise SError(f"refusing to overwrite {p}")
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def self_test():
 with tempfile.TemporaryDirectory() as d:
  r=pathlib.Path(d);src=r/"ephem";src.write_text("{}");sh=hfile(src);basis=((0.,0.,1.),(0.,1.,0.),(1.,0.,0.),False);bins=[{"bin_index":i,"azimuth_start_deg":i*90.,"azimuth_end_deg":((i+1)*90.),"conservative_elevation_deg":e} for i,e in enumerate((0.,15.,.75,0.))];k={"schema_version":K_SCHEMA,"status":"pass","study_id":"s","frame_contract_id":"f","epoch_contract_id":"e","site_ref":"site","frame":"FRAME","basis":{"north":basis[0],"east":basis[1],"up":basis[2],"fallback_used":False},"azimuth_bin_width_deg":90.,"bin_count":4,"bins":bins};o={"schema_version":O_SCHEMA,"study_id":"s","deterministic_upper_bound_eligible":False,"risk_qualified_horizon_eligible":True};rr={"schema_version":R_SCHEMA,"study_id":"s","strongest_common_spatial_support":"sample_points_only"};cfg={"schema_version":CONFIG_SCHEMA,"requested_claim_class":"deterministic_visibility_bound","policies":{"max_time_gap_s":3600.,"los_margin_deg":0.},"sources":[{"source_id":"ephem","path":"ephem","sha256":sh}],"targets":[{"target_id":"SUN","kind":"sun","frame":"FRAME","apparent_angular_radius_deg":.5,"source_refs":["ephem"],"samples":[{"t_s":0.,"direction":direction(basis,0,10)},{"t_s":3600.,"direction":direction(basis,90,10)},{"t_s":7200.,"direction":direction(basis,180,1)}]}]};paths=[r/x for x in ("c","k","o","r")]
  for p,v in zip(paths,(cfg,k,o,rr)):p.write_bytes(cb(v))
  a=generate(paths[0],paths[1],paths[2],paths[3],r);assert a["status"]=="claim_blocked_geometry_available" and a["metric_evidence_class"]=="descriptive_geometry_only";assert [x["k_bin_index"] for x in a["targets"][0]["samples"]]==[0,1,2];assert bin_at(359.999,90,4)==3 and bin_at(360,90,4)==0
  o["deterministic_upper_bound_eligible"]=True;rr["strongest_common_spatial_support"]="continuous_hard_bound";paths[2].write_bytes(cb(o));paths[3].write_bytes(cb(rr));b=generate(paths[0],paths[1],paths[2],paths[3],r);assert b["metric_evidence_class"]=="derived_deterministic_bound" and cb(a["targets"])==cb(b["targets"])
  cfg["requested_claim_class"]="risk_qualified_visibility";paths[0].write_bytes(cb(cfg));c=generate(paths[0],paths[1],paths[2],paths[3],r);assert not c["requested_claim_allowed"] and "statistical_horizon_binding" in c["claim_reason"];print("LL-009S semantic visibility self-test: PASS")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--k-pack");p.add_argument("--o-receipt");p.add_argument("--r-receipt");p.add_argument("--artifact-root",default=".");p.add_argument("--output");p.add_argument("--self-test",action="store_true");a=p.parse_args()
 try:
  if a.self_test:self_test();return 0
  if not all((a.config,a.k_pack,a.o_receipt,a.r_receipt,a.output)):raise SError("required paths missing")
  out=generate(pathlib.Path(a.config),pathlib.Path(a.k_pack),pathlib.Path(a.o_receipt),pathlib.Path(a.r_receipt),pathlib.Path(a.artifact_root));write(pathlib.Path(a.output),out);print(json.dumps(out,sort_keys=True,indent=2));return 0
 except (OSError,json.JSONDecodeError,SError) as e:raise SystemExit(f"LL-009S failure: {e}") from e
if __name__=="__main__":raise SystemExit(main())
