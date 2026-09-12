#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,tempfile
from typing import Any

AA="ll009aa.sdem-archive-member-manifest.v1";RM="ll009ab.sdem-role-map.v1";RR="ll009ab.sdem-role-binding-receipt.v1"
AP="ll009ab.cross-method-audit-policy.v1";AR="ll009ab.cross-method-terrain-audit-receipt.v1";LS="ll009l.cog-materialization-config.v1"
ROLES={"sdem_elevation","ldem_reference","sdem_minus_ldem","image_coverage","solar_bin_count","best_input_resolution"}

class E(RuntimeError):pass
def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(",",": "))+"\n").encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(4<<20),b""):h.update(c)
 return h.hexdigest()
def rd(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as x:raise E(f"cannot read {n}: {x}") from x
 if not isinstance(v,dict):raise E(f"{n} must be object")
 return v
def sh(v,n):
 x=v.get("receipt_sha256");b=json.loads(json.dumps(v));b.pop("receipt_sha256",None)
 if not isinstance(x,str) or len(x)!=64 or x!=hb(cb(b)):raise E(f"{n} self-hash mismatch")
def fin(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def rel(x):
 if not isinstance(x,str) or not x or "\\" in x:raise E(f"unsafe path {x!r}")
 p=pathlib.PurePosixPath(x)
 if p.is_absolute() or ".." in p.parts or any(q in ("",".","..") for q in p.parts):raise E(f"unsafe path {x!r}")
 return p
def req():
 try:
  import numpy as np, rasterio
  from rasterio.warp import transform as wt
  return np,rasterio,wt
 except Exception as x:raise E("Rasterio/Numpy required only for offline LL-009AB evidence") from x
def resolve(root,entry,n):
 p=root/pathlib.Path(*rel(entry["path"]).parts)
 if not p.is_file() or hf(p)!=entry["sha256"]:raise E(f"{n} missing/hash mismatch")
 return p
def meta(ds,pix,wsha,n):
 if ds.count!=1 or ds.crs is None:raise E(f"{n}: single-band CRS raster required")
 a,b,c,d,e,f=list(ds.transform)[:6];sx,sy=math.hypot(a,d),math.hypot(b,e);w=hb(ds.crs.to_wkt().encode())
 if abs(sx-pix)>1e-8 or abs(sy-pix)>1e-8:raise E(f"{n}: pixel size mismatch")
 if w!=wsha:raise E(f"{n}: CRS WKT hash mismatch")
 return {"width":ds.width,"height":ds.height,"dtype":ds.dtypes[0],"nodata":ds.nodata,"transform":[a,b,c,d,e,f],"pixel_scale_x_m":sx,"pixel_scale_y_m":sy,"crs_wkt_sha256":w}

def manifest(m):
 if m.get("schema_version")!=AA or m.get("status")!="pass":raise E("AA manifest must pass")
 sh(m,"AA manifest")
 if m.get("role_assignment")!="intentionally_unresolved_until_ll009ab":raise E("AA role boundary drift")
 out={}
 for x in m.get("members",[]):
  if not isinstance(x,dict) or not isinstance(x.get("path"),str) or not isinstance(x.get("sha256"),str) or len(x["sha256"])!=64:raise E("bad AA member")
  rel(x["path"])
  if x["path"] in out:raise E("duplicate AA member")
  out[x["path"]]=x
 if not out:raise E("empty AA manifest")
 return out
def rolemap(v,mp):
 if v.get("schema_version")!=RM or not isinstance(v.get("study_id"),str) or v.get("aa_manifest_sha256")!=hf(mp):raise E("bad role map lineage")
 out=[];sr=set();sp=set()
 for x in v.get("roles",[]):
  if not isinstance(x,dict) or x.get("role") not in ROLES or x["role"] in sr:raise E("bad/duplicate role")
  if not isinstance(x.get("path"),str) or not isinstance(x.get("sha256"),str) or len(x["sha256"])!=64:raise E("bad role path/hash")
  rel(x["path"])
  if x["path"] in sp:raise E("one member cannot fill multiple roles")
  if not isinstance(x.get("documentary_basis"),str) or len(x["documentary_basis"].strip())<12:raise E("role documentary basis required")
  if not fin(x.get("expected_pixel_size_m")) or x["expected_pixel_size_m"]<=0 or not isinstance(x.get("expected_crs_wkt_sha256"),str) or len(x["expected_crs_wkt_sha256"])!=64:raise E("role raster contract invalid")
  if not isinstance(x.get("pixelwise_comparable_to_sdem"),bool):raise E("pixelwise comparability flag required")
  sr.add(x["role"]);sp.add(x["path"]);out.append(dict(x))
 if "sdem_elevation" not in sr:raise E("sdem_elevation role required")
 return out
def bind(mp,rmp,root):
 _,rio,_=req();m=rd(mp,"manifest");mi=manifest(m);rm=rd(rmp,"role map");roles=rolemap(rm,mp);ds={};o={}
 try:
  for x in roles:
   r=x["role"];mm=mi.get(x["path"])
   if mm is None or mm.get("sha256")!=x["sha256"]:raise E(f"{r}: absent from AA manifest")
   p=resolve(root,x,r);ds[r]=rio.open(p);md=meta(ds[r],float(x["expected_pixel_size_m"]),x["expected_crs_wkt_sha256"],r)
   o[r]={"path":x["path"],"sha256":x["sha256"],"documentary_basis":x["documentary_basis"],"pixelwise_comparable_to_sdem":x["pixelwise_comparable_to_sdem"],"raster_metadata":md}
  s=ds["sdem_elevation"]
  for x in roles:
   if x["role"]=="sdem_elevation" or not x["pixelwise_comparable_to_sdem"]:continue
   q=ds[x["role"]]
   if q.crs!=s.crs or q.width!=s.width or q.height!=s.height or any(abs(a-b)>1e-10 for a,b in zip(list(q.transform)[:6],list(s.transform)[:6])):raise E(f"{x['role']}: grid differs from SDEM")
 finally:
  for q in ds.values():q.close()
 z={"schema_version":RR,"status":"pass","study_id":rm["study_id"],"aa_manifest_sha256":hf(mp),"aa_manifest_receipt_sha256":m["receipt_sha256"],"role_map_sha256":hf(rmp),"roles":o,"semantics_class":"exact_scientific_role_binding","non_claims":["Role binding identifies exact members and registration; it does not establish terrain truth.","SfS remains a reconstructed surface with illumination, photometric and regularization limitations."]};z["receipt_sha256"]=hb(cb(z));return z

class Stats:
 def __init__(s):s.n=0;s.a=0.;s.q=0.;s.lo=math.inf;s.hi=-math.inf;s.g=[0,0,0]
 def add(s,x):
  import numpy as np
  x=np.asarray(x,dtype="float64").ravel();x=x[np.isfinite(x)]
  if not len(x):return
  s.n+=len(x);s.a+=float(np.sum(x));s.q+=float(np.sum(x*x));s.lo=min(s.lo,float(np.min(x)));s.hi=max(s.hi,float(np.max(x)));u=np.abs(x)
  for i,t in enumerate((1.,2.,3.)):s.g[i]+=int(np.count_nonzero(u>t))
 def out(s):
  if not s.n:return {"count":0}
  m=s.a/s.n;v=max(0.,s.q/s.n-m*m)
  return {"count":int(s.n),"mean":m,"stddev_population":math.sqrt(v),"rms":math.sqrt(s.q/s.n),"min":s.lo,"max":s.hi,"fraction_abs_gt_1":s.g[0]/s.n,"fraction_abs_gt_2":s.g[1]/s.n,"fraction_abs_gt_3":s.g[2]/s.n}

def policy(v,lp,rrp):
 if v.get("schema_version")!=AP or v.get("l_config_sha256")!=hf(lp) or v.get("role_receipt_sha256")!=hf(rrp):raise E("audit policy lineage mismatch")
 if v.get("grid_alignment_policy")!="exact_same_pixel_lattice_no_resampling":raise E("hidden resampling forbidden")
 if not isinstance(v.get("study_id"),str) or not isinstance(v.get("ldem_layer_id"),str):raise E("study/layer required")
 s=v.get("site");pr=v.get("projection_contract");po=v.get("pole_vector")
 if not isinstance(s,dict) or not all(fin(s.get(k)) for k in ("x_m","y_m","elevation_m")):raise E("site invalid")
 if not isinstance(pr,dict) or pr.get("projection")!="south_polar_stereographic" or not fin(pr.get("reference_radius_m")):raise E("projection invalid")
 if not isinstance(po,list) or len(po)!=3 or not all(fin(x) for x in po):raise E("pole invalid")
 w=v.get("azimuth_bin_width_deg");lo=v.get("min_range_m");hi=v.get("max_range_m")
 if not fin(w) or w<=0 or abs(round(360/w)*w-360)>1e-9 or not fin(lo) or not fin(hi) or lo<0 or hi<=lo:raise E("bin/range invalid")
 if not fin(v.get("published_difference_tolerance_m",1e-4)) or v.get("published_difference_tolerance_m",1e-4)<0:raise E("difference tolerance invalid")
 seen=set()
 for q in v.get("strata",[]):
  if not isinstance(q,dict) or not isinstance(q.get("stratum_id"),str) or q["stratum_id"] in seen or q.get("role") not in {"image_coverage","solar_bin_count","best_input_resolution"}:raise E("bad stratum")
  if not fin(q.get("min_inclusive")) or not fin(q.get("max_exclusive")) or q["max_exclusive"]<=q["min_inclusive"]:raise E("bad stratum bounds")
  seen.add(q["stratum_id"])
 return v
def offset(base,other):
 if base.crs!=other.crs:raise E("SDEM/LDEM CRS mismatch")
 a=list(base.transform)[:6];b=list(other.transform)[:6]
 if any(abs(a[i]-b[i])>1e-10 for i in (0,1,3,4)):raise E("pixel lattice matrix mismatch")
 c,r=(~base.transform)*(other.transform.c,other.transform.f);ci,ri=round(c),round(r)
 if abs(c-ci)>1e-8 or abs(r-ri)>1e-8:raise E("origins not integer-pixel aligned")
 return int(ri),int(ci)
def unit(np,v):
 a=np.asarray(v,dtype="float64");n=float(np.linalg.norm(a))
 if n<=1e-15:raise E("degenerate vector")
 return a/n
def basis(np,u,p):
 u=unit(np,u);p=unit(np,p);n=p-u*float(np.dot(p,u))
 if float(np.linalg.norm(n))<=1e-10:
  ax=np.eye(3);x=ax[int(np.argmin(np.abs(ax@u)))];n=x-u*float(np.dot(x,u))
 n=unit(np,n);e=unit(np,np.cross(n,u));n=unit(np,np.cross(u,e));return n,e,u
def units(np,lo,la):
 lo=np.radians(np.asarray(lo));la=np.radians(np.asarray(la));c=np.cos(la);return np.stack((c*np.cos(lo),c*np.sin(lo),np.sin(la)),axis=1)

def audit(pp,rrp,mp,rmp,sroot,lp,lroot):
 np,rio,wt=req();rr=rd(rrp,"role receipt");sh(rr,"role receipt")
 if rr.get("schema_version")!=RR or rr.get("status")!="pass" or rr.get("aa_manifest_sha256")!=hf(mp) or rr.get("role_map_sha256")!=hf(rmp):raise E("role receipt lineage mismatch")
 p=policy(rd(pp,"audit policy"),lp,rrp)
 if rr.get("study_id")!=p["study_id"]:raise E("study mismatch")
 l=rd(lp,"L config")
 if l.get("schema_version")!=LS or l.get("study_id")!=p["study_id"]:raise E("L config mismatch")
 ls=[x for x in l.get("layers",[]) if isinstance(x,dict) and x.get("layer_id")==p["ldem_layer_id"]]
 if len(ls)!=1:raise E("expected one LDEM layer")
 lay=ls[0];es=lay.get("elevation_source");us=lay.get("uncertainty_source");roles=rr["roles"]
 if not isinstance(es,dict) or "sdem_elevation" not in roles:raise E("terrain sources missing")
 sdmp=resolve(sroot,roles["sdem_elevation"],"SDEM");ldmp=resolve(lroot,es,"LDEM");rmsp=resolve(lroot,us,"LDEM RMS") if isinstance(us,dict) else None
 aux={r:resolve(sroot,roles[r],r) for r in ("sdem_minus_ldem","image_coverage","solar_bin_count","best_input_resolution") if r in roles}
 rs=Stats();ab=Stats();nr=Stats();de=Stats();st={x["stratum_id"]:Stats() for x in p.get("strata",[])}
 w=float(p["azimuth_bin_width_deg"]);bn=round(360/w);hs=np.full(bn,-np.inf);hl=np.full(bn,-np.inf);site=p["site"];rad=float(p["projection_contract"]["reference_radius_m"])
 with rio.open(ldmp) as ld,rio.open(sdmp) as sd:
  ro,co=offset(ld,sd);sr0=max(0,-ro);sc0=max(0,-co);sr1=min(sd.height,ld.height-ro);sc1=min(sd.width,ld.width-co)
  if sr1<=sr0 or sc1<=sc0:raise E("no aligned overlap")
  overlap=(sr1-sr0)*(sc1-sc0);validn=0;ads={k:rio.open(v) for k,v in aux.items()};rds=rio.open(rmsp) if rmsp else None
  try:
   for k,d in ads.items():
    if d.crs!=sd.crs or d.width!=sd.width or d.height!=sd.height or any(abs(a-b)>1e-10 for a,b in zip(list(d.transform)[:6],list(sd.transform)[:6])):raise E(f"{k}: grid drift")
   if rds and offset(ld,rds)!=(0,0):raise E("RMS grid drift")
   geo=rio.crs.CRS.from_proj4(f"+proj=longlat +R={rad:.12f} +no_defs +type=crs");lo,la=wt(sd.crs,geo,[float(site["x_m"])],[float(site["y_m"])]);up=units(np,lo,la)[0];n,e,u=basis(np,up,p["pole_vector"]);sp=(rad+float(site["elevation_m"]))*up
   for r0 in range(sr0,sr1,512):
    r1=min(sr1,r0+512)
    for c0 in range(sc0,sc1,512):
     c1=min(sc1,c0+512);sw=rio.windows.Window(c0,r0,c1-c0,r1-r0);lw=rio.windows.Window(c0+co,r0+ro,c1-c0,r1-r0)
     sa=sd.read(1,window=sw,masked=True);laa=ld.read(1,window=lw,masked=True);sv=np.asarray(sa.data,dtype="float64");lv=np.asarray(laa.data,dtype="float64")
     ok=~(np.ma.getmaskarray(sa)|np.ma.getmaskarray(laa)|~np.isfinite(sv)|~np.isfinite(lv))
     if not np.any(ok):continue
     validn+=int(np.count_nonzero(ok));res=sv[ok]-lv[ok];rs.add(res);ab.add(np.abs(res))
     if "sdem_minus_ldem" in ads:
      da=ads["sdem_minus_ldem"].read(1,window=sw,masked=True);dv=np.asarray(da.data,dtype="float64");q=ok&~np.ma.getmaskarray(da)&np.isfinite(dv)
      if np.any(q):
       err=dv[q]-(sv[q]-lv[q]);de.add(err)
       if float(np.max(np.abs(err)))>float(p.get("published_difference_tolerance_m",1e-4))+1e-12:raise E("published SDEM-LDEM difference mismatch")
     if rds:
      ra=rds.read(1,window=lw,masked=True);rv=np.asarray(ra.data,dtype="float64");q=ok&~np.ma.getmaskarray(ra)&np.isfinite(rv)&(rv>0)
      if np.any(q):nr.add((sv[q]-lv[q])/rv[q])
     for x in p.get("strata",[]):
      d=ads.get(x["role"])
      if d is None:raise E("stratum role not bound")
      aa=d.read(1,window=sw,masked=True);av=np.asarray(aa.data,dtype="float64");q=ok&~np.ma.getmaskarray(aa)&np.isfinite(av)&(av>=x["min_inclusive"])&(av<x["max_exclusive"])
      if np.any(q):st[x["stratum_id"]].add(sv[q]-lv[q])
     rr0,cc0=np.indices(sa.shape);rows=rr0+r0;cols=cc0+c0;xs=sd.transform.c+(cols+.5)*sd.transform.a+(rows+.5)*sd.transform.b;ys=sd.transform.f+(cols+.5)*sd.transform.d+(rows+.5)*sd.transform.e
     rng=np.hypot(xs-site["x_m"],ys-site["y_m"]);q=ok&(rng>=p["min_range_m"]-1e-9)&(rng<=p["max_range_m"]+1e-9)&(rng>1e-9)
     if np.any(q):
      lo,lat=wt(sd.crs,geo,xs[q].tolist(),ys[q].tolist());uu=units(np,lo,lat)
      def hv(z):
       los=uu*(rad+z)[:,None]-sp;d=los/np.linalg.norm(los,axis=1)[:,None];nc=d@n;ec=d@e;uc=d@u;az=np.degrees(np.arctan2(ec,nc))%360;el=np.degrees(np.arctan2(uc,np.hypot(nc,ec)));return np.floor((az+1e-9)/w).astype("int64")%bn,el
      bi,se=hv(sv[q]);_,le=hv(lv[q]);np.maximum.at(hs,bi,se);np.maximum.at(hl,bi,le)
  finally:
   for d in ads.values():d.close()
   if rds:rds.close()
 common=np.isfinite(hs)&np.isfinite(hl)
 if p.get("require_all_bins",True) and not bool(np.all(common)):raise E(f"missing horizon bins {np.nonzero(~common)[0].tolist()}")
 bins=[];pos=0.;neg=0.
 for i in range(bn):
  if not common[i]:bins.append({"bin_index":i,"supported":False});continue
  d=float(hs[i]-hl[i]);pos=max(pos,d);neg=min(neg,d);bins.append({"bin_index":i,"supported":True,"ldem_horizon_deg":float(hl[i]),"sdem_horizon_deg":float(hs[i]),"sdem_minus_ldem_horizon_deg":d,"positive_excursion_deg":max(0.,d),"negative_excursion_deg":min(0.,d)})
 z={"schema_version":AR,"status":"pass","study_id":p["study_id"],"semantics_class":"empirical_cross_method_terrain_discrepancy","policy_sha256":hf(pp),"role_receipt_sha256":hf(rrp),"aa_manifest_sha256":hf(mp),"l_config_sha256":hf(lp),"ldem_layer_id":p["ldem_layer_id"],"grid_alignment_policy":p["grid_alignment_policy"],"overlap_pixel_count":overlap,"valid_pair_count":validn,"elevation_residual_m":rs.out(),"absolute_elevation_residual_m":ab.out(),"published_difference_reconstruction_error_m":de.out() if "sdem_minus_ldem" in aux else None,"normalized_residual_by_ldem_rms":nr.out() if rmsp else None,"normalized_residual_interpretation":"descriptive_model_check_only_not_a_calibrated_probability_theorem" if rmsp else None,"strata":{k:v.out() for k,v in st.items()},"azimuth_bin_width_deg":w,"per_bin_horizon_discrepancy":bins,"max_positive_sdem_minus_ldem_horizon_excursion_deg":pos,"most_negative_sdem_minus_ldem_horizon_excursion_deg":neg,"spatial_support_semantics":"empirical_cross_method_terrain_discrepancy_not_continuous_hard_bound","calibration_semantics":"SDEM-LDEM normalized residuals are model-checking evidence only; SDEM regularization toward LDEM forbids treating them as independent truth or an automatic correction.","non_claims":["No hidden reprojection or resampling is performed.","The SDEM is not independent ground truth.","Residual/RMS ratios do not automatically rescale Product 90 ADJ_ERR or prove a true second-moment upper bound.","Positive SDEM skyline excursions are empirical missed-obstruction evidence, not a continuous-terrain hard bound.","No visibility, site-safety, RF, power, mission or architecture authority is emitted."]};z["receipt_sha256"]=hb(cb(z));return z

def wr(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise E(f"refusing differing output {p}")
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def selftest():
 np,rio,_=req()
 from rasterio.crs import CRS
 from rasterio.transform import from_origin
 with tempfile.TemporaryDirectory() as td:
  q=pathlib.Path(td);s=q/"s";l=q/"l";s.mkdir();l.mkdir();crs=CRS.from_proj4("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs");lt=from_origin(-60,60,10,10);stf=from_origin(-40,40,10,10)
  la=np.zeros((12,12),dtype="float32");sa=np.zeros((8,8),dtype="float32");sa[1,4]=12.;rms=np.ones((12,12),dtype="float32");cov=np.full((8,8),4,dtype="float32")
  def t(p,a,tr):
   with rio.open(p,"w",driver="GTiff",width=a.shape[1],height=a.shape[0],count=1,dtype="float32",crs=crs,transform=tr,nodata=-9999.) as d:d.write(a,1)
  t(l/"ldem.tif",la,lt);t(l/"rms.tif",rms,lt);t(s/"sdem.tif",sa,stf);t(s/"diff.tif",sa,stf);t(s/"cov.tif",cov,stf)
  ms=[{"path":n,"sha256":hf(s/n)} for n in ("sdem.tif","diff.tif","cov.tif")];m={"schema_version":AA,"status":"pass","study_id":"ab","role_assignment":"intentionally_unresolved_until_ll009ab","members":ms};m["receipt_sha256"]=hb(cb(m));mp=q/"m.json";wr(mp,m)
  with rio.open(s/"sdem.tif") as d:wsha=hb(d.crs.to_wkt().encode())
  rm={"schema_version":RM,"study_id":"ab","aa_manifest_sha256":hf(mp),"roles":[{"role":"sdem_elevation","path":"sdem.tif","sha256":hf(s/"sdem.tif"),"documentary_basis":"reviewed synthetic SDEM role","expected_pixel_size_m":10.,"expected_crs_wkt_sha256":wsha,"pixelwise_comparable_to_sdem":True},{"role":"sdem_minus_ldem","path":"diff.tif","sha256":hf(s/"diff.tif"),"documentary_basis":"reviewed synthetic difference role","expected_pixel_size_m":10.,"expected_crs_wkt_sha256":wsha,"pixelwise_comparable_to_sdem":True},{"role":"image_coverage","path":"cov.tif","sha256":hf(s/"cov.tif"),"documentary_basis":"reviewed synthetic coverage role","expected_pixel_size_m":10.,"expected_crs_wkt_sha256":wsha,"pixelwise_comparable_to_sdem":True}]};rmp=q/"rm.json";wr(rmp,rm);r=bind(mp,rmp,s);rrp=q/"rr.json";wr(rrp,r)
  lc={"schema_version":LS,"study_id":"ab","layers":[{"layer_id":"near","elevation_source":{"path":"ldem.tif","sha256":hf(l/"ldem.tif")},"uncertainty_source":{"path":"rms.tif","sha256":hf(l/"rms.tif")}}]};lp=q/"l.json";wr(lp,lc)
  p={"schema_version":AP,"study_id":"ab","role_receipt_sha256":hf(rrp),"l_config_sha256":hf(lp),"ldem_layer_id":"near","grid_alignment_policy":"exact_same_pixel_lattice_no_resampling","site":{"x_m":0.,"y_m":0.,"elevation_m":0.},"projection_contract":{"projection":"south_polar_stereographic","reference_radius_m":1737400.},"pole_vector":[0,0,-1],"azimuth_bin_width_deg":90.,"min_range_m":5.,"max_range_m":80.,"require_all_bins":True,"published_difference_tolerance_m":1e-5,"strata":[{"stratum_id":"cov3","role":"image_coverage","min_inclusive":3.,"max_exclusive":99.}]};pp=q/"p.json";wr(pp,p);a=audit(pp,rrp,mp,rmp,s,lp,l)
  assert a["max_positive_sdem_minus_ldem_horizon_excursion_deg"]>0 and a["elevation_residual_m"]["max"]==12 and a["normalized_residual_by_ldem_rms"]["max"]==12 and cb(a)==cb(audit(pp,rrp,mp,rmp,s,lp,l))
  bad=json.loads(json.dumps(rm));bad["roles"][0]["sha256"]="0"*64;bp=q/"bad.json";wr(bp,bad)
  try:bind(mp,bp,s);raise AssertionError("bad role accepted")
  except E:pass
  shdir=q/"shift";shdir.mkdir();shift=from_origin(-39,40,10,10)
  for n,a0 in (("sdem.tif",sa),("diff.tif",sa),("cov.tif",cov)):t(shdir/n,a0,shift)
  ms2=[{"path":n,"sha256":hf(shdir/n)} for n in ("sdem.tif","diff.tif","cov.tif")];m2={"schema_version":AA,"status":"pass","study_id":"ab","role_assignment":"intentionally_unresolved_until_ll009ab","members":ms2};m2["receipt_sha256"]=hb(cb(m2));mp2=q/"m2";wr(mp2,m2);rm2=json.loads(json.dumps(rm));rm2["aa_manifest_sha256"]=hf(mp2)
  for x in rm2["roles"]:x["sha256"]=hf(shdir/x["path"])
  rmp2=q/"rm2";wr(rmp2,rm2);rr2=bind(mp2,rmp2,shdir);rrp2=q/"rr2";wr(rrp2,rr2);p2=json.loads(json.dumps(p));p2["role_receipt_sha256"]=hf(rrp2);pp2=q/"p2";wr(pp2,p2)
  try:audit(pp2,rrp2,mp2,rmp2,shdir,lp,l);raise AssertionError("subpixel shift accepted")
  except E:pass
  print("LL-009AB role binding + cross-method audit self-test: PASS")
def main():
 x=argparse.ArgumentParser();s=x.add_subparsers(dest="cmd",required=True);b=s.add_parser("bind");b.add_argument("--manifest",required=True);b.add_argument("--role-map",required=True);b.add_argument("--extracted-root",required=True);b.add_argument("--output",required=True);a=s.add_parser("audit")
 for k in ("policy","role-receipt","manifest","role-map","extracted-root","l-config","ldem-root","output"):a.add_argument("--"+k,required=True)
 s.add_parser("self-test");z=x.parse_args()
 try:
  if z.cmd=="self-test":selftest();return 0
  if z.cmd=="bind":o=bind(pathlib.Path(z.manifest),pathlib.Path(z.role_map),pathlib.Path(z.extracted_root))
  else:o=audit(pathlib.Path(z.policy),pathlib.Path(z.role_receipt),pathlib.Path(z.manifest),pathlib.Path(z.role_map),pathlib.Path(z.extracted_root),pathlib.Path(z.l_config),pathlib.Path(z.ldem_root))
  wr(pathlib.Path(z.output),o);print(json.dumps(o,sort_keys=True,indent=2));return 0
 except (E,OSError,ValueError) as q:raise SystemExit(f"LL-009AB failure: {q}") from q
if __name__=="__main__":raise SystemExit(main())
