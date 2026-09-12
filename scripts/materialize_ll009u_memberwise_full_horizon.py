#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, tempfile
from typing import Any

POLICY="ll009u.memberwise-full-horizon-policy.v1"; QPOL="ll009q.ensemble-policy.v1"; QREC="ll009q.clone-horizon-ensemble-receipt.v1"
OREC="ll009o.uncertainty-semantics-receipt.v1"; RREC="ll009r.spatial-support-classification-receipt.v1"; LOCK="ll009n.nasa-source-lock.v1"
LCFG="ll009l.cog-materialization-config.v1"; KOUT="ll009k.horizon-pack.v1"
SUPPORT={"continuous_hard_bound","empirical_multiscale_bound","resolution_qualified","sample_points_only","unknown"}
class UError(RuntimeError): pass

def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(",",": "))+"\n").encode()
def hb(v:bytes)->str:return hashlib.sha256(v).hexdigest()
def hf(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(4<<20),b""):h.update(c)
 return h.hexdigest()
def fin(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def obj(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as e:raise UError(f"cannot read {n}: {e}") from e
 if not isinstance(v,dict):raise UError(f"{n} must be object")
 return v
def selfhash(v,n):
 x=v.get("receipt_sha256")
 if not isinstance(x,str) or len(x)!=64:raise UError(f"{n}: receipt_sha256 required")
 b=json.loads(json.dumps(v));b.pop("receipt_sha256",None)
 if x!=hb(cb(b)):raise UError(f"{n}: receipt self-hash mismatch")
def rel(v):
 p=pathlib.PurePosixPath(v) if isinstance(v,str) else pathlib.PurePosixPath(".")
 if not isinstance(v,str) or not v or p.is_absolute() or ".." in p.parts:raise UError(f"unsafe artifact path {v!r}")
 return p
def rio():
 try:
  import numpy as np, rasterio
  from rasterio.warp import transform
 except Exception as e:raise UError("Rasterio/Numpy required only for offline LL-009U materialization") from e
 return np,rasterio,transform

def policy(v):
 if v.get("schema_version")!=POLICY:raise UError("U policy schema mismatch")
 req={"observer_policy":"same_realization_site_pixel","site_statistical_closure":"q_member_site_elevation_is_authoritative_for_each_realization","companion_vertical_semantics":"hard_upper_bound","companion_uncertainty_direction":"terrain_radial","full_horizon_composition":"max_components_per_member_then_summarize_members","spatial_support_margin_policy":"apply_observed_positive_excursion_if_present"}
 for k,x in req.items():
  if v.get(k)!=x:raise UError(f"U policy requires {k}={x}")
 for k in ("study_id","ensemble_layer_id","companion_layer_id","statistic_mode"):
  if not isinstance(v.get(k),str) or not v[k]:raise UError(f"U policy missing {k}")
 if v["ensemble_layer_id"]==v["companion_layer_id"]:raise UError("ensemble/companion layer collision")
 if v["statistic_mode"] not in {"empirical_ensemble_quantile","finite_ensemble_observed_max"}:raise UError("unsupported statistic_mode")
 if v["statistic_mode"]=="empirical_ensemble_quantile" and (not fin(v.get("quantile")) or not 0<float(v["quantile"])<=1):raise UError("quantile invalid")
 if v.get("require_all_members_all_bins") is not True:raise UError("all members/all bins required")
 return v
def layers(v,n,support=False):
 a=v.get("layers")
 if not isinstance(a,list) or not a:raise UError(f"{n}: layers missing")
 d={}
 for x in a:
  if not isinstance(x,dict) or not isinstance(x.get("layer_id"),str) or x["layer_id"] in d:raise UError(f"{n}: layer invalid/duplicate")
  if support and x.get("support_class") not in SUPPORT:raise UError(f"{n}: support class invalid")
  d[x["layer_id"]]=x
 return d
def qcheck(p,qp_path,qp,q,lock_path,l_path):
 if qp.get("schema_version")!=QPOL or qp.get("study_id")!=p["study_id"] or qp.get("ensemble_layer_id")!=p["ensemble_layer_id"]:raise UError("Q/U policy mismatch")
 if qp.get("observer_policy")!=p["observer_policy"] or qp.get("quantile_estimator")!="empirical_cdf_nearest_rank":raise UError("Q observer/estimator mismatch")
 if q.get("schema_version")!=QREC or q.get("status")!="pass" or q.get("semantics_class")!="empirical_ensemble":raise UError("Q receipt invalid")
 selfhash(q,"Q receipt")
 if q.get("study_id")!=p["study_id"] or q.get("policy_sha256")!=hf(qp_path) or q.get("nominal_lock_sha256")!=hf(lock_path) or q.get("l_config_sha256")!=hf(l_path):raise UError("Q exact lineage mismatch")
 if q.get("observer_policy")!=p["observer_policy"]:raise UError("Q receipt observer mismatch")
 w,n,m=q.get("azimuth_bin_width_deg"),q.get("azimuth_bin_count"),q.get("member_count")
 if not fin(w) or not isinstance(n,int) or n<4 or abs(float(w)*n-360)>1e-9 or not isinstance(m,int) or m<2:raise UError("Q bin/member geometry invalid")
 ms=q.get("members")
 if not isinstance(ms,list) or len(ms)!=m:raise UError("Q member count mismatch")
 for i,x in enumerate(ms):
  if not isinstance(x,dict) or x.get("ordinal")!=i or not fin(x.get("site_elevation_m")):raise UError(f"Q member {i} invalid")
  h=x.get("horizon_deg")
  if not isinstance(h,list) or len(h)!=n or not all(fin(z) for z in h):raise UError(f"Q member {i} horizon incomplete")
 qv=float(p["quantile"]) if p["statistic_mode"]=="empirical_ensemble_quantile" else None
 if qv is not None:
  declared=qp.get("quantiles")
  if not isinstance(declared,list) or not any(abs(float(x)-qv)<=1e-12 for x in declared):raise UError("requested U quantile not declared by Q")
 return ms,float(w),n,qv
def orcheck(p,o,r,lock_path):
 if o.get("schema_version")!=OREC or r.get("schema_version")!=RREC:raise UError("O/R schema mismatch")
 selfhash(o,"O receipt");selfhash(r,"R receipt")
 if o.get("study_id")!=p["study_id"] or r.get("study_id")!=p["study_id"] or o.get("source_lock_sha256")!=hf(lock_path):raise UError("O/R exact lineage mismatch")
 ol,rl=layers(o,"O"),layers(r,"R",True); expected={p["ensemble_layer_id"],p["companion_layer_id"]}
 if set(ol)!=expected or set(rl)!=expected:raise UError("O/R/U layer set mismatch")
 if ol[p["ensemble_layer_id"]].get("semantics_class") not in {"rms_error","empirical_ensemble"}:raise UError("near layer not eligible for Q closure")
 if ol[p["companion_layer_id"]].get("semantics_class")!="hard_upper_bound":raise UError("U V1 companion far-field uncertainty must be hard_upper_bound")
 site=o.get("site_vertical_uncertainty")
 if not isinstance(site,dict) or site.get("semantics_class") not in {"rms_error","empirical_ensemble"}:raise UError("U requires Q statistical site closure")
 if o.get("risk_qualified_horizon_eligible") is not True:raise UError("O does not permit risk-qualified vertical semantics")
 return ol,rl
def support_margin(x):
 v=x.get("observed_positive_excursion_margin_deg",0.0)
 if not fin(v) or float(v)<0:raise UError("invalid R spatial-support margin")
 return float(v)
def lock_index(lock):
 if lock.get("schema_version")!=LOCK or not isinstance(lock.get("files"),list):raise UError("source lock invalid")
 return lock["files"]
def bind_source(lock,entry,label,roles):
 if not isinstance(entry,dict) or not isinstance(entry.get("path"),str) or not isinstance(entry.get("sha256"),str):raise UError(f"{label}: L source binding missing")
 a=[x for x in lock_index(lock) if isinstance(x,dict) and x.get("artifact_path")==entry["path"] and x.get("sha256")==entry["sha256"]]
 if len(a)!=1 or a[0].get("role") not in roles:raise UError(f"{label}: exact source-lock binding/role mismatch")
 return a[0]
def resolve(root,x,label):
 f=root/pathlib.Path(*rel(x["artifact_path"]).parts)
 if not f.is_file() or hf(f)!=x["sha256"]:raise UError(f"{label}: source missing/hash mismatch")
 return f
def affine(a,b,tol=1e-12):return all(abs(float(x)-float(y))<=tol for x,y in zip(a,b))
def units(np,lon,lat):
 L=np.radians(np.asarray(lon,dtype="float64")); B=np.radians(np.asarray(lat,dtype="float64")); c=np.cos(B)
 return np.stack((c*np.cos(L),c*np.sin(L),np.sin(B)),axis=1)
def unit(np,x,n):
 a=np.asarray(x,dtype="float64");m=float(np.linalg.norm(a))
 if not math.isfinite(m) or m<=1e-15:raise UError(f"{n} degenerate")
 return a/m
def basis(np,up,pole):
 u=unit(np,up,"site up");p=unit(np,pole,"pole");nr=p-u*float(np.dot(p,u));fb=False
 if float(np.linalg.norm(nr))<=1e-10:
  fb=True; axes=np.eye(3); ref=axes[int(np.argmin(np.abs(axes@u)))];nr=ref-u*float(np.dot(ref,u))
 n=unit(np,nr,"north");e=unit(np,np.cross(n,u),"east");n=unit(np,np.cross(u,e),"north")
 return n,e,u,fb
def nrank(v,q):
 s=sorted(float(x) for x in v);return s[max(1,int(math.ceil(float(q)*len(s))))-1]

def run(pp,qpp,qrp,op,rp,lp,lcp,root):
 np,rasterio,warp=rio(); p=policy(obj(pp,"U policy")); qp=obj(qpp,"Q policy");q=obj(qrp,"Q receipt");o=obj(op,"O receipt");r=obj(rp,"R receipt");lock=obj(lp,"source lock");lc=obj(lcp,"L config")
 ms,w,n,qv=qcheck(p,qpp,qp,q,lp,lcp);ol,rl=orcheck(p,o,r,lp)
 if lock.get("study_id")!=p["study_id"] or lc.get("schema_version")!=LCFG or lc.get("study_id")!=p["study_id"]:raise UError("lock/L/U study or schema mismatch")
 if lc.get("site_ref")!=q.get("site_ref") or lc.get("native_frame")!=q.get("native_frame") or abs(float(lc.get("azimuth_bin_width_deg",math.nan))-w)>1e-12:raise UError("Q/L site/frame/bin mismatch")
 ls=[x for x in lc.get("layers",[]) if isinstance(x,dict) and x.get("layer_id")==p["companion_layer_id"]]
 if len(ls)!=1:raise UError("exactly one companion L layer required")
 L=ls[0]; ep=bind_source(lock,L.get("elevation_source"),"far elevation",{"surface_elevation"});up=bind_source(lock,L.get("uncertainty_source"),"far uncertainty",{"surface_height_error_m","vertical_uncertainty_m","elevation_rms_uncertainty_m"});er=bind_source(lock,L.get("effective_resolution_source"),"far effective resolution",{"effective_resolution_m"}) if isinstance(L.get("effective_resolution_source"),dict) else None
 if ol[p["companion_layer_id"]].get("uncertainty_source_sha256")!=up["sha256"]:raise UError("O far uncertainty does not bind same source bytes as L/N")
 ef,uf=resolve(root,ep,"far elevation"),resolve(root,up,"far uncertainty");rf=resolve(root,er,"far effective resolution") if er else None
 proj=lc.get("projection_contract",{});rad=proj.get("reference_radius_m");lon0=proj.get("central_meridian_deg",0.0)
 if proj.get("projection")!="south_polar_stereographic" or not fin(rad) or float(rad)<=0:raise UError("L lunar projection invalid")
 site=lc.get("site",{});sx,sy=site.get("x_m"),site.get("y_m")
 if not fin(sx) or not fin(sy):raise UError("L site x/y invalid")
 minr,maxr,pix=L.get("min_range_m"),L.get("max_range_m"),L.get("expected_pixel_size_m")
 if not all(fin(x) for x in (minr,maxr,pix)) or float(minr)<0 or float(maxr)<=float(minr) or float(pix)<=0:raise UError("companion layer range/resolution invalid")
 near=np.asarray([x["horizon_deg"] for x in ms],dtype="float64"); site_z=np.asarray([x["site_elevation_m"] for x in ms],dtype="float64"); far=np.full((len(ms),n),-np.inf)
 scanned=admitted=nodata=0
 with rasterio.open(ef) as E,rasterio.open(uf) as U:
  R=rasterio.open(rf) if rf else None
  try:
   if E.count!=1 or U.count!=1 or E.crs is None or E.width!=U.width or E.height!=U.height or E.crs!=U.crs or not affine(E.transform,U.transform):raise UError("far elevation/uncertainty alignment invalid")
   if R and (E.width!=R.width or E.height!=R.height or E.crs!=R.crs or not affine(E.transform,R.transform)):raise UError("far effective-resolution alignment invalid")
   a,b,_,d,e,_=list(E.transform)[:6]
   if abs(math.hypot(a,d)-float(pix))>1e-8 or abs(math.hypot(b,e)-float(pix))>1e-8:raise UError("far pixel scale mismatch")
   expected=rasterio.crs.CRS.from_proj4(f"+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0={float(lon0):.12f} +R={float(rad):.12f} +units=m +no_defs")
   if E.crs!=expected:raise UError("far CRS not declared lunar stereographic CRS")
   wh=L.get("expected_crs_wkt_sha256")
   if isinstance(wh,str) and wh!=hb(E.crs.to_wkt().encode()):raise UError("far CRS WKT hash mismatch")
   geo=rasterio.crs.CRS.from_proj4(f"+proj=longlat +R={float(rad):.12f} +no_defs +type=crs");lo,la=warp(E.crs,geo,[float(sx)],[float(sy)]);sup=units(np,lo,la)[0]
   pole=lc.get("pole_vector");
   if not isinstance(pole,list) or len(pole)!=3 or not all(fin(x) for x in pole):raise UError("L pole vector invalid")
   north,east,upv,fb=basis(np,sup,pole); sites=sup[None,:]*(float(rad)+site_z)[:,None];box=(float(sx)-float(maxr),float(sy)-float(maxr),float(sx)+float(maxr),float(sy)+float(maxr));maxeff=L.get("max_effective_resolution_m")
   for _,win in E.block_windows(1):
    l,bt,rr,t=rasterio.windows.bounds(win,E.transform)
    if rr<box[0] or l>box[2] or t<box[1] or bt>box[3]:continue
    z=E.read(1,window=win,masked=True);u=U.read(1,window=win,masked=True);ev=R.read(1,window=win,masked=True) if R else None;zm=np.ma.getmaskarray(z);um=np.ma.getmaskarray(u);em=np.ma.getmaskarray(ev) if ev is not None else np.zeros(z.shape,dtype=bool)
    rows,cols=np.indices(z.shape);rg=rows+int(win.row_off);cg=cols+int(win.col_off);aa,bb,cc,dd,ee,ff=list(E.transform)[:6];xs=aa*(cg+.5)+bb*(rg+.5)+cc;ys=dd*(cg+.5)+ee*(rg+.5)+ff;rgxy=np.hypot(xs-float(sx),ys-float(sy));ann=(rgxy>=float(minr)-1e-9)&(rgxy<=float(maxr)+1e-9);scanned+=int(np.count_nonzero(ann));mask=zm|um|em;nodata+=int(np.count_nonzero(ann&mask))
    if L.get("nodata_policy","fail_required_coverage")=="fail_required_coverage" and np.any(ann&mask):raise UError("nodata in required far annulus")
    valid=ann&~mask
    if not np.any(valid):continue
    zv=np.asarray(z.data[valid],dtype="float64");uv=np.asarray(u.data[valid],dtype="float64");xv=xs[valid];yv=ys[valid]
    if not np.all(np.isfinite(zv)) or not np.all(np.isfinite(uv)) or np.any(uv<0):raise UError("invalid far elevation/hard-bound uncertainty")
    if maxeff is not None:
     if ev is None:raise UError("effective-resolution evidence required")
     rv=np.asarray(ev.data[valid],dtype="float64");keep=np.isfinite(rv)&(rv>0)&(rv<=float(maxeff));zv,uv,xv,yv=zv[keep],uv[keep],xv[keep],yv[keep]
    if not len(zv):continue
    admitted+=len(zv);lo,la=warp(E.crs,geo,xv.tolist(),yv.tolist());tup=units(np,lo,la);high=tup*(float(rad)+zv+uv)[:,None]
    for j,sp in enumerate(sites):
     los=high-sp;norm=np.linalg.norm(los,axis=1)
     if np.any(~np.isfinite(norm)) or np.any(norm<=1e-12):raise UError("degenerate far LOS")
     dvec=los/norm[:,None];az=np.degrees(np.arctan2(dvec@east,dvec@north))%360;el=np.degrees(np.arctan2(dvec@upv,np.hypot(dvec@north,dvec@east)));bi=np.floor((az+1e-9)/w).astype("int64")%n;np.maximum.at(far[j],bi,el)
  finally:
   if R:R.close()
 if admitted==0 or not np.all(np.isfinite(far)):raise UError("far scan lacks complete member/bin coverage")
 nm,fm=support_margin(rl[p["ensemble_layer_id"]]),support_margin(rl[p["companion_layer_id"]]);near=near+nm;far=far+fm;full=np.maximum(near,far)
 vals=[]
 for i in range(n):
  col=full[:,i];v=float(np.max(col)) if qv is None else nrank(col,qv);vals.append({"bin_index":i,"azimuth_start_deg":i*w,"azimuth_end_deg":(i+1)*w,"conservative_elevation_deg":v,"winner":{"source_kind":"ll009u_memberwise_full_horizon_statistic","near_dominant_member_count":int(np.count_nonzero(near[:,i]>=far[:,i])),"far_dominant_member_count":int(len(ms)-np.count_nonzero(near[:,i]>=far[:,i])),"statistic_mode":p["statistic_mode"],"quantile":qv},"runner_up":None,"candidate_count":len(ms),"numeric_semantics":"max_components_per_member_then_summarize_members"})
 members=[{"ordinal":i,"source_id":m.get("source_id"),"site_elevation_m":float(m["site_elevation_m"]),"near_horizon_deg":[float(x) for x in near[i]],"far_horizon_deg":[float(x) for x in far[i]],"full_horizon_deg":[float(x) for x in full[i]]} for i,m in enumerate(ms)]
 bind={"status":"bound","schema_version":"ll009u.memberwise-statistical-horizon-binding.v1","mode":p["statistic_mode"],"estimator":"empirical_cdf_nearest_rank" if qv is not None else "finite_observed_max","quantile":qv,"covered_layer_ids":[p["ensemble_layer_id"],p["companion_layer_id"]],"q_policy_sha256":hf(qpp),"q_receipt_sha256":hf(qrp),"o_uncertainty_receipt_sha256":hf(op),"r_spatial_support_receipt_sha256":hf(rp),"source_lock_sha256":hf(lp),"l_config_sha256":hf(lcp),"numeric_composition_policy_sha256":hf(pp),"observer_policy":p["observer_policy"],"site_statistical_closure":p["site_statistical_closure"],"companion_vertical_semantics":p["companion_vertical_semantics"],"companion_uncertainty_direction":p["companion_uncertainty_direction"],"full_horizon_composition":p["full_horizon_composition"],"spatial_support_margin_policy":p["spatial_support_margin_policy"],"far_source_hashes":{"elevation":ep["sha256"],"uncertainty":up["sha256"],"effective_resolution":er["sha256"] if er else None}}
 nominal=sup*(float(rad)+float(site.get("elevation_m",0.0)))
 out={"schema_version":KOUT,"status":"pass","study_id":p["study_id"],"frame_contract_id":lc["frame_contract_id"],"epoch_contract_id":lc["epoch_contract_id"],"site_ref":lc["site_ref"],"frame":lc["native_frame"],"input_sha256":hf(pp),"source_hashes":{"ll009q-receipt":hf(qrp),"ll009o-receipt":hf(op),"ll009r-receipt":hf(rp),"source-lock":hf(lp),"far-elevation":ep["sha256"],"far-uncertainty":up["sha256"],"far-effective-resolution":er["sha256"] if er else None},"site_position_m":nominal.tolist(),"site_vertical_uncertainty_m":float(site.get("vertical_uncertainty_m",0.0)),"basis":{"north":north.tolist(),"east":east.tolist(),"up":upv.tolist(),"fallback_used":fb},"azimuth_bin_width_deg":w,"bin_count":n,"layers":[{"layer_id":p["ensemble_layer_id"],"composition_role":"q_empirical_member_horizon","spatial_support_class":rl[p["ensemble_layer_id"]]["support_class"],"spatial_support_margin_deg":nm},{"layer_id":p["companion_layer_id"],"composition_role":"hard_bound_far_geometry_recomputed_per_q_observer","spatial_support_class":rl[p["companion_layer_id"]]["support_class"],"spatial_support_margin_deg":fm,"scanned_annulus_pixels":scanned,"admitted_pixels":admitted,"nodata_pixels_in_annulus":nodata}],"bins":vals,"statistical_horizon_binding":bind,"member_count":len(ms),"members":members,"semantics":"Each exact Q member supplies both its near terrain horizon and observer/site elevation; the hard-bound far skyline is independently rescanned against that same observer state, then near/far are maximized memberwise before the finite ensemble statistic is taken.","non_claims":["Finite ensemble statistics are not deterministic terrain bounds or population confidence guarantees.","Product 90 ADJ_ERR remains blocked until its semantics are source-qualified for the required hard-bound companion mode.","LL-009R spatial-support limitations remain in force.","The nominal site_vertical_uncertainty_m field is retained only for K schema compatibility; Q member site elevations are the statistical observer authority.","This does not qualify visibility, power, communications, a site, architecture, or operations authority."]};out["receipt_sha256"]=hb(cb(out));return out

def write(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise UError(f"refusing to overwrite differing output {p}")
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def selftest():
 np,rasterio,_=rio();a=np.array([10.,0,0,0]);b=np.array([0,10.,0,0]);assert nrank(a,.75)==0 and nrank(b,.75)==0 and nrank(np.maximum(a,b),.75)==10
 from rasterio.crs import CRS
 from rasterio.transform import from_origin
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td);d=r/"d";d.mkdir();crs=CRS.from_proj4("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs");tr=from_origin(-4500,4500,1000,1000);z=np.zeros((9,9),dtype="float32");z[4,7]=50;u=np.full((9,9),.5,dtype="float32");er=np.full((9,9),1000,dtype="float32")
  def wr(n,a):
   p=d/n
   with rasterio.open(p,"w",driver="GTiff",width=9,height=9,count=1,dtype="float32",crs=crs,transform=tr,nodata=-9999) as x:x.write(a,1)
   return p
  E,U,R=wr("e.tif",z),wr("u.tif",u),wr("r.tif",er);eh,uh,rh=hf(E),hf(U),hf(R);lock={"schema_version":LOCK,"study_id":"s","files":[{"source_id":"e","role":"surface_elevation","artifact_path":"e.tif","sha256":eh},{"source_id":"u","role":"surface_height_error_m","artifact_path":"u.tif","sha256":uh},{"source_id":"r","role":"effective_resolution_m","artifact_path":"r.tif","sha256":rh}]};lp=r/"lock.json";lp.write_bytes(cb(lock))
  with rasterio.open(E) as x:wh=hb(x.crs.to_wkt().encode())
  lc={"schema_version":LCFG,"study_id":"s","frame_contract_id":"f","epoch_contract_id":"e","site_ref":"site","native_frame":"MOON_ME_DE421","projection_contract":{"projection":"south_polar_stereographic","reference_radius_m":1737400.,"central_meridian_deg":0.,"true_scale_at_pole":True,"pixel_registration":"center"},"site":{"x_m":0.,"y_m":0.,"elevation_m":0.,"vertical_uncertainty_m":.5},"pole_vector":[0.,0.,1.],"azimuth_bin_width_deg":90.,"layers":[{"layer_id":"far","min_range_m":500.,"max_range_m":4500.,"expected_pixel_size_m":1000.,"expected_crs_wkt_sha256":wh,"max_effective_resolution_m":1001.,"nodata_policy":"fail_required_coverage","elevation_source":{"path":"e.tif","sha256":eh},"uncertainty_source":{"path":"u.tif","sha256":uh},"effective_resolution_source":{"path":"r.tif","sha256":rh}}]};lcp=r/"l.json";lcp.write_bytes(cb(lc));qp={"schema_version":QPOL,"study_id":"s","ensemble_layer_id":"near","observer_policy":"same_realization_site_pixel","quantile_estimator":"empirical_cdf_nearest_rank","quantiles":[.75]};qpp=r/"qp.json";qpp.write_bytes(cb(qp));ms=[{"ordinal":i,"source_id":f"m{i}","site_elevation_m":se,"horizon_deg":[10. if i==0 else 0.,0.,0.,0.]} for i,se in enumerate((-10.,0.,10.,20.))];q={"schema_version":QREC,"status":"pass","semantics_class":"empirical_ensemble","study_id":"s","site_ref":"site","native_frame":"MOON_ME_DE421","policy_sha256":hf(qpp),"nominal_lock_sha256":hf(lp),"l_config_sha256":hf(lcp),"observer_policy":"same_realization_site_pixel","azimuth_bin_width_deg":90.,"azimuth_bin_count":4,"member_count":4,"members":ms};q["receipt_sha256"]=hb(cb(q));qrp=r/"q.json";qrp.write_bytes(cb(q));o={"schema_version":OREC,"status":"pass","study_id":"s","source_lock_sha256":hf(lp),"risk_qualified_horizon_eligible":True,"site_vertical_uncertainty":{"semantics_class":"rms_error"},"layers":[{"layer_id":"near","semantics_class":"rms_error","uncertainty_source_sha256":"0"*64},{"layer_id":"far","semantics_class":"hard_upper_bound","uncertainty_source_sha256":uh}]};o["receipt_sha256"]=hb(cb(o));op=r/"o.json";op.write_bytes(cb(o));rr={"schema_version":RREC,"status":"pass","study_id":"s","layers":[{"layer_id":"near","support_class":"sample_points_only"},{"layer_id":"far","support_class":"resolution_qualified"}]};rr["receipt_sha256"]=hb(cb(rr));rp=r/"r.json";rp.write_bytes(cb(rr));p={"schema_version":POLICY,"study_id":"s","ensemble_layer_id":"near","companion_layer_id":"far","statistic_mode":"empirical_ensemble_quantile","quantile":.75,"observer_policy":"same_realization_site_pixel","site_statistical_closure":"q_member_site_elevation_is_authoritative_for_each_realization","companion_vertical_semantics":"hard_upper_bound","companion_uncertainty_direction":"terrain_radial","full_horizon_composition":"max_components_per_member_then_summarize_members","spatial_support_margin_policy":"apply_observed_positive_excursion_if_present","require_all_members_all_bins":True};pp=r/"p.json";pp.write_bytes(cb(p));x=run(pp,qpp,qrp,op,rp,lp,lcp,d);y=run(pp,qpp,qrp,op,rp,lp,lcp,d);assert cb(x)==cb(y) and x["statistical_horizon_binding"]["status"]=="bound";bad=json.loads(json.dumps(o));bad["layers"][1]["semantics_class"]="unknown";bad.pop("receipt_sha256");bad["receipt_sha256"]=hb(cb(bad));bp=r/"bad.json";bp.write_bytes(cb(bad))
  try:run(pp,qpp,qrp,bp,rp,lp,lcp,d);raise AssertionError("unknown far semantics accepted")
  except UError as e:assert "hard_upper_bound" in str(e)
  print("LL-009U memberwise full-horizon self-test: PASS",rasterio.__version__)
def main():
 a=argparse.ArgumentParser();
 for x in ("policy","q-policy","q-receipt","o-receipt","r-receipt","source-lock","l-config","output"):a.add_argument("--"+x)
 a.add_argument("--artifact-root",default=".");a.add_argument("--self-test",action="store_true");z=a.parse_args()
 try:
  if z.self_test:selftest();return 0
  req=(z.policy,z.q_policy,z.q_receipt,z.o_receipt,z.r_receipt,z.source_lock,z.l_config,z.output)
  if not all(req):raise UError("required inputs missing")
  out=run(pathlib.Path(z.policy),pathlib.Path(z.q_policy),pathlib.Path(z.q_receipt),pathlib.Path(z.o_receipt),pathlib.Path(z.r_receipt),pathlib.Path(z.source_lock),pathlib.Path(z.l_config),pathlib.Path(z.artifact_root));write(pathlib.Path(z.output),out);print(json.dumps(out,sort_keys=True,indent=2));return 0
 except (OSError,json.JSONDecodeError,UError) as e:raise SystemExit(f"LL-009U failure: {e}") from e
if __name__=="__main__":raise SystemExit(main())
